import optuna
import pandas as pd
import numpy as np # For subsampling and float32
from sentence_transformers import SentenceTransformer
from umap import UMAP
from sklearn.cluster import HDBSCAN # Using sklearn's version
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance

from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
import nltk
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
from nltk.corpus import stopwords
import logging
from functools import partial
import random # For subsampling

# --- 0. Global Settings ---
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' # Consider 'all-MiniLM-L6-v2' for speed/less memory if quality is acceptable
TOP_N_WORDS_FOR_COHERENCE = 10
FINAL_MODEL_TOP_N_WORDS = 10
STOP_WORDS_GNSM = stopwords.words('english')

# !!! CRUCIAL FOR LARGE DATASETS !!!
# Define a maximum number of documents for Optuna trials
# Adjust this based on your available RAM. Start low (e.g., 50k-100k) and increase if stable.
# For 1M+ sentences, even 100k might be a lot for many trials.
# If len(DOCS) from your CSV is > MAX_DOCS_FOR_OPTUNA, subsampling will occur.
MAX_DOCS_FOR_OPTUNA_TRIALS = 100_000 # Example: 100,000 sentences

# --- Helper Functions ---
def preprocess_for_gensim(documents):
    texts = []
    for doc in documents:
        try:
            tokens = [word for word in doc.lower().split() if word.isalpha() and word not in STOP_WORDS_GNSM]
            texts.append(tokens)
        except AttributeError: # Handles if a 'doc' is not a string (e.g., float if data had NaNs not dropped)
            logger.warning(f"Skipping non-string document in preprocess_for_gensim: {doc}")
            texts.append([]) # Append empty list to maintain structure
    return texts

def calculate_topic_diversity(topic_word_lists):
    if not topic_word_lists or len(topic_word_lists) == 0: return 0.0
    unique_words = set(word for topic in topic_word_lists for word in topic)
    total_words = sum(len(topic) for topic in topic_word_lists)
    if total_words == 0: return 0.0
    return len(unique_words) / total_words

# --- 1. Define the Objective Function ---
def objective_function_with_data(trial, docs_list, embeddings_array, tokenized_docs_for_gensim_list, gensim_dictionary_obj):
    logger.info(f"\n--- Trial {trial.number} --- (Docs: {len(docs_list)})")

    # --- Hyperparameter Suggestions (More Conservative Ranges for Stability) ---
    # UMAP
    # With low_memory=True, n_neighbors has less impact on peak memory but still on quality/speed
    umap_n_neighbors = trial.suggest_int('umap_n_neighbors', 5, 30) # Reduced upper bound
    umap_n_components = trial.suggest_int('umap_n_components', 2, 10) # Reduced upper bound, higher can be slow for HDBSCAN
    umap_min_dist = trial.suggest_float('umap_min_dist', 0.0, 0.3)    # Reduced upper bound
    umap_metric = trial.suggest_categorical('umap_metric', ['cosine', 'euclidean']) # Manhattan can be slow

    # HDBSCAN
    hdbscan_min_cluster_size = trial.suggest_int('hdbscan_min_cluster_size', 5, 20) # Increased min, reduced max
    # min_samples is crucial. If too low relative to min_cluster_size, can lead to many tiny clusters or issues.
    # Making min_samples a bit higher, or at least not 1 unless min_cluster_size is also very small.
    if hdbscan_min_cluster_size <= 5: # If min_cluster_size is very small, allow min_samples to be small too
        hdbscan_min_samples = trial.suggest_int('hdbscan_min_samples', 1, hdbscan_min_cluster_size)
    else: # Otherwise, encourage min_samples to be a bit larger to avoid instability
        hdbscan_min_samples = trial.suggest_int('hdbscan_min_samples', max(2, hdbscan_min_cluster_size // 2), hdbscan_min_cluster_size)

    hdbscan_metric = trial.suggest_categorical('hdbscan_metric', ['euclidean', 'manhattan', 'cosine']) # These are on UMAP output, so less memory concern here
    hdbscan_selection_method = trial.suggest_categorical('hdbscan_cluster_selection_method', ['eom', 'leaf'])
    # HDBSCAN `allow_single_cluster` can sometimes be problematic if it collapses everything.
    # Default is False, which is usually fine.

    # CountVectorizer
    ngram_max = trial.suggest_int('ngram_max', 1, 2) # Trigrams can explode vocab size; stick to 1 or 2 for stability
    vectorizer_min_df = trial.suggest_int('vectorizer_min_df', 2, 5)   # min_df=1 can include too many rare/noisy words
    vectorizer_max_df = trial.suggest_float('vectorizer_max_df', 0.7, 0.95)

    # ClassTfidfTransformer
    ctfidf_reduce_frequent_words = trial.suggest_categorical('ctfidf_reduce_frequent_words', [True, False]) # True is often good
    ctfidf_bm25_weighting = trial.suggest_categorical('ctfidf_bm25_weighting', [True, False])

    # BERTopic nr_topics
    nr_topics_strategy = trial.suggest_categorical('nr_topics_strategy', ['auto', 'target_range'])
    if nr_topics_strategy == 'auto':
        nr_topics_suggestion = None
    else:
        # Adjust target range based on expected number of topics from your subset
        max_possible_topics = len(docs_list) // hdbscan_min_cluster_size if hdbscan_min_cluster_size > 0 else len(docs_list)
        nr_topics_upper_bound = min(30, max(5, max_possible_topics // 2) ) # Don't aim for too many topics from a subset
        nr_topics_suggestion = trial.suggest_int('nr_topics_target', 5, nr_topics_upper_bound if nr_topics_upper_bound > 5 else 6)


    # Representation Model
    representation_choice = trial.suggest_categorical('representation_model_type', ['default_ctfidf', 'keybert', 'mmr'])
    current_representation_model = None
    if representation_choice == 'keybert':
        current_representation_model = KeyBERTInspired()
    elif representation_choice == 'mmr':
        mmr_diversity = trial.suggest_float('mmr_diversity', 0.1, 0.7) # Slightly more conservative diversity
        current_representation_model = MaximalMarginalRelevance(diversity=mmr_diversity)

    log_params = (
        f"UMAP(nn={umap_n_neighbors}, nc={umap_n_components}, md={umap_min_dist:.2f}, m='{umap_metric}'), "
        f"HDBSCAN(mcs={hdbscan_min_cluster_size}, ms={hdbscan_min_samples}, m='{hdbscan_metric}', csm='{hdbscan_selection_method}'), "
        f"CV(ngram=(1,{ngram_max}), min_df={vectorizer_min_df}, max_df={vectorizer_max_df:.2f}), "
        f"CTFIDF(rfw={ctfidf_reduce_frequent_words}, bm25={ctfidf_bm25_weighting}), "
        f"BERTopic(nr_topics='{nr_topics_suggestion}', rep_model='{representation_choice}')"
    )
    logger.info(f"Params: {log_params}")

    # --- Model Instantiation ---
    # ALWAYS use low_memory=True for UMAP with large N
    umap_model = UMAP(n_neighbors=umap_n_neighbors, n_components=umap_n_components, min_dist=umap_min_dist,
                      metric=umap_metric, random_state=42, low_memory=True, verbose=False)

    # Using sklearn.cluster.HDBSCAN
    hdbscan_model = HDBSCAN(min_cluster_size=hdbscan_min_cluster_size,
                            min_samples=hdbscan_min_samples,
                            metric=hdbscan_metric, # This is on UMAP output, less critical for memory
                            cluster_selection_method=hdbscan_selection_method,
                            # core_dist_n_jobs=-1 # Can uncomment if HDBSCAN itself is slow and you have cores
                            )

    vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, ngram_max),
                                       min_df=vectorizer_min_df, max_df=vectorizer_max_df)

    ctfidf_model_instance = ClassTfidfTransformer(reduce_frequent_words=ctfidf_reduce_frequent_words,
                                            bm25_weighting=ctfidf_bm25_weighting)

    topic_model = BERTopic(
        embedding_model=None, # Embeddings are pre-computed and passed
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model_instance,
        representation_model=current_representation_model if representation_choice != 'default_ctfidf' else None,
        language="english",
        top_n_words=TOP_N_WORDS_FOR_COHERENCE,
        calculate_probabilities=False, # Keep False for speed during Optuna
        verbose=False,
        nr_topics=nr_topics_suggestion
    )

    try:
        topics, _ = topic_model.fit_transform(docs_list, embeddings=embeddings_array)
    except MemoryError as e:
        logger.warning(f"MemoryError during BERTopic fit_transform: {e}. Pruning trial.")
        raise optuna.exceptions.TrialPruned()
    except ValueError as e: # Catches UMAP/HDBSCAN value errors (e.g. not enough samples)
        logger.warning(f"ValueError during BERTopic fit_transform: {e}. Pruning trial.")
        raise optuna.exceptions.TrialPruned()
    except Exception as e: # Catch any other unexpected errors
        logger.error(f"Unexpected error during BERTopic fit_transform: {e}. Pruning trial.", exc_info=False) # Set exc_info=True for full traceback if needed
        raise optuna.exceptions.TrialPruned()

    topic_info_df = topic_model.get_topic_info()
    if topic_info_df is None or topic_info_df.empty:
        logger.warning("  No topic_info_df generated. Pruning trial.")
        raise optuna.exceptions.TrialPruned()

    num_topics = len(topic_info_df[topic_info_df.Topic != -1])
    num_outliers = 0
    if -1 in topic_info_df.Topic.values:
        count_series = topic_info_df[topic_info_df.Topic == -1]['Count']
        if not count_series.empty:
            num_outliers = count_series.iloc[0]

    percent_outliers = (num_outliers / len(docs_list)) * 100 if len(docs_list) > 0 else 0
    logger.info(f"  Generated {num_topics} topics, {percent_outliers:.2f}% outliers.")

    # More robust pruning conditions
    if num_topics < 2 : # Need at least 2 topics (excluding outliers)
        logger.warning(f"  Less than 2 actual topics ({num_topics}). Pruning trial.")
        raise optuna.exceptions.TrialPruned()
    if num_topics > len(docs_list) / 2: # Unlikely to have more topics than half the docs
        logger.warning(f"  Too many topics ({num_topics}) relative to docs. Pruning trial.")
        raise optuna.exceptions.TrialPruned()
    if percent_outliers > 90: # If almost everything is an outlier
         logger.warning(f"  Excessive outliers ({percent_outliers:.2f}%). Pruning trial.")
         raise optuna.exceptions.TrialPruned()

    bertopic_topics_words = []
    try:
        raw_bertopic_topics = topic_model.get_topics()
        if not raw_bertopic_topics: # Check if get_topics() returned empty
            logger.warning("  model.get_topics() returned empty. Pruning trial.")
            raise optuna.exceptions.TrialPruned()

        for topic_id in sorted(raw_bertopic_topics.keys()):
            if topic_id == -1: continue
            topic_words = [word for word, _ in raw_bertopic_topics[topic_id][:TOP_N_WORDS_FOR_COHERENCE]]
            if topic_words: bertopic_topics_words.append(topic_words)
    except Exception as e:
        logger.warning(f"  Error processing topic words: {e}. Pruning trial.")
        raise optuna.exceptions.TrialPruned()


    if not bertopic_topics_words or len(bertopic_topics_words) < num_topics: # Ensure we have words for most topics
        logger.warning(f"  Not enough valid topic words extracted (Got {len(bertopic_topics_words)} for {num_topics} topics). Pruning trial.")
        raise optuna.exceptions.TrialPruned()

    coherence_cv = -1.0 # Default to worst coherence
    diversity = 0.0    # Default to worst diversity
    silhouette = -1.0  # Default to worst silhouette

    try:
        if gensim_dictionary_obj and len(gensim_dictionary_obj) > 0 and tokenized_docs_for_gensim_list:
            coherence_model_cv = CoherenceModel(
                topics=bertopic_topics_words, texts=tokenized_docs_for_gensim_list,
                dictionary=gensim_dictionary_obj, coherence='c_v', processes=1)
            coherence_cv = coherence_model_cv.get_coherence()
        else:
            logger.warning("  Skipping Coherence: Gensim dictionary or tokenized docs empty/missing.")
    except Exception as e:
        logger.warning(f"  Error calculating c_v coherence: {e}")
        # coherence_cv remains -1.0

    diversity = calculate_topic_diversity(bertopic_topics_words)

    # Silhouette score can be memory intensive if embeddings_array is huge.
    # It's calculated on the original embeddings, so consider if this is feasible for large N.
    # For Optuna, if embeddings_array is the subsampled one, this is fine.
    valid_indices = [i for i, label in enumerate(topics) if label != -1 and i < len(embeddings_array)]
    if len(valid_indices) > 1: # Need at least 2 points for silhouette
        # Ensure embeddings_array is correctly indexed
        try:
            valid_embeddings_for_silhouette = embeddings_array[valid_indices]
            valid_labels_for_silhouette = np.array(topics)[valid_indices] # Ensure this is an array for proper indexing
            if len(set(valid_labels_for_silhouette)) > 1: # Need at least 2 clusters
                silhouette = silhouette_score(valid_embeddings_for_silhouette, valid_labels_for_silhouette)
        except IndexError as e:
            logger.warning(f"  IndexError calculating silhouette score (likely topics/embeddings mismatch): {e}")
        except ValueError as e: # Catches "Number of labels is 1. Valid values are 2 to n_samples - 1 (inclusive)"
             logger.warning(f"  ValueError calculating silhouette score (likely only 1 cluster): {e}")
        except Exception as e:
            logger.warning(f"  Unexpected error calculating silhouette score: {e}")

    logger.info(f"  Metrics: Coherence_CV={coherence_cv:.4f}, Diversity={diversity:.4f}, Silhouette={silhouette:.4f}")

    # --- Define Combined Score (Weights can also be tuned with Optuna if desired) ---
    w_coherence = trial.suggest_float("w_coherence", 0.3, 0.6) # Optuna can tune weights
    w_silhouette = trial.suggest_float("w_silhouette", 0.1, 0.4)
    w_diversity = trial.suggest_float("w_diversity", 0.05, 0.2)
    w_outliers_penalty = 1.0 - (w_coherence + w_silhouette + w_diversity) # Ensure weights sum to 1
    # Ensure w_outliers_penalty is not negative if sum exceeds 1 due to float precision
    w_outliers_penalty = max(0.05, w_outliers_penalty) # Give outlier penalty at least some weight


    norm_coherence = max(0, min(1, (coherence_cv + 1) / 2 if coherence_cv else 0)) # Map C_v from approx [-1, 1] to [0, 1]
    norm_silhouette = (silhouette + 1) / 2 if silhouette is not None else 0
    norm_diversity = diversity if diversity is not None else 0
    outlier_score_component = (1 - (percent_outliers / 100))

    combined_score = (w_coherence * norm_coherence) + \
                     (w_silhouette * norm_silhouette) + \
                     (w_diversity * norm_diversity) + \
                     (w_outliers_penalty * outlier_score_component)

    # Penalty/Bonus for number of topics
    target_min_topics = 5
    # Adjust max topics based on subset size, e.g. not more than 1 topic per 1000 docs in the subset
    target_max_topics = min(50, max(10, len(docs_list) // 1000 if len(docs_list) > 5000 else 10))

    if num_topics < target_min_topics:
        combined_score -= 0.1 * (target_min_topics - num_topics)
    elif num_topics > target_max_topics:
        combined_score -= 0.05 * (num_topics - target_max_topics)

    # Handle NaN or Inf scores before returning
    if np.isnan(combined_score) or np.isinf(combined_score):
        logger.warning(f"  Combined score is NaN or Inf ({combined_score}). Returning very low score.")
        return -float('inf') # Optuna maximizes, so this is bad

    logger.info(f"  Combined Score: {combined_score:.4f}")
    return combined_score

# --- 2. Main Execution Block ---
if __name__ == "__main__":
    logger.info("--- Starting Data Setup ---")
    try:
        # Corrected path for example, adjust to your actual path
        df = pd.read_csv(r'../data/test/phil_nlp.csv', low_memory=False)
    except FileNotFoundError:
        logger.error(r"Error: phil_nlp.csv not found. Please check the path.")
        exit()
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}"); exit()

    # Ensure 'sentence_str' column exists
    if 'sentence_str' not in df.columns:
        logger.error("Column 'sentence_str' not found in the CSV.")
        exit()

    DOCS_FULL = df['sentence_str'].dropna().astype(str).tolist() # Ensure string type
    if not DOCS_FULL: logger.error("No documents loaded from 'sentence_str'."); exit()
    logger.info(f"Loaded {len(DOCS_FULL)} documents total.")

    # --- Subsampling for Optuna Trials ---
    if len(DOCS_FULL) > MAX_DOCS_FOR_OPTUNA_TRIALS:
        logger.info(f"Subsampling {MAX_DOCS_FOR_OPTUNA_TRIALS} documents for Optuna from {len(DOCS_FULL)} total.")
        # Create a reproducible sample if needed, or random for exploration
        # random.seed(42) # for reproducible subsample
        sample_indices = random.sample(range(len(DOCS_FULL)), MAX_DOCS_FOR_OPTUNA_TRIALS)
        DOCS_OPTUNA = [DOCS_FULL[i] for i in sample_indices]
    else:
        DOCS_OPTUNA = DOCS_FULL
    logger.info(f"Using {len(DOCS_OPTUNA)} documents for Optuna trials.")


    logger.info(f"Loading sentence transformer: {EMBEDDING_MODEL_NAME}")
    sentence_model_local = SentenceTransformer(EMBEDDING_MODEL_NAME)

    logger.info("Encoding documents for Optuna...")
    # Ensure embeddings are float32 to save memory
    EMBEDDINGS_OPTUNA = sentence_model_local.encode(DOCS_OPTUNA, show_progress_bar=True).astype(np.float32)

    logger.info("Preprocessing for Gensim (Optuna subset)...")
    TOKENIZED_DOCS_FOR_GENSIM_OPTUNA = preprocess_for_gensim(DOCS_OPTUNA)
    if not any(TOKENIZED_DOCS_FOR_GENSIM_OPTUNA): logger.warning("Gensim tokenized docs for Optuna are all empty!"); # Don't exit, allow trial to proceed and likely score low on coherence
    GENSIM_DICTIONARY_OPTUNA = Dictionary(TOKENIZED_DOCS_FOR_GENSIM_OPTUNA)
    if len(GENSIM_DICTIONARY_OPTUNA) == 0: logger.warning("Gensim dictionary for Optuna is empty!");

    logger.info("--- Data Setup Finished ---")

    N_TRIALS = 50
    objective_with_fixed_data = partial(objective_function_with_data,
                                        docs_list=DOCS_OPTUNA,
                                        embeddings_array=EMBEDDINGS_OPTUNA,
                                        tokenized_docs_for_gensim_list=TOKENIZED_DOCS_FOR_GENSIM_OPTUNA,
                                        gensim_dictionary_obj=GENSIM_DICTIONARY_OPTUNA)

    storage_name = "sqlite:///bertopic_optuna_sentence_tuned.db" # New DB name
    study_name = "bertopic_sentence_tuned_study"
    study = optuna.create_study(study_name=study_name, storage=storage_name,
                                direction="maximize", load_if_exists=True)
    try:
        logger.info(f"Starting/Resuming Optuna optimization for {N_TRIALS} trials (Study: {study_name})...")
        study.optimize(objective_with_fixed_data, n_trials=N_TRIALS, gc_after_trial=True) # Add gc_after_trial
        logger.info("Optuna optimization completed for this session.")
    except KeyboardInterrupt: logger.info("Optimization interrupted by user.")
    except Exception as e: logger.error(f"An error occurred during optimization: {e}", exc_info=True)

    logger.info("\n--- Optimization Finished (Overall Study) ---")
    logger.info(f"Number of finished trials in study: {len(study.trials)}")

    if study.best_trial:
        logger.info("Best trial overall in study (on subset):")
        logger.info(f"  Value (Combined Score): {study.best_trial.value:.4f}")
        best_params = study.best_trial.params
        logger.info("  Best Params (from Optuna on subset): ")
        for key, value in best_params.items(): logger.info(f"    {key}: {value}")

        logger.info("\n--- Training Final BERTopic Model with Best Parameters ON FULL DATASET ---")
        # Re-encode full dataset for final model
        logger.info("Encoding FULL dataset for final model...")
        EMBEDDINGS_FULL = sentence_model_local.encode(DOCS_FULL, show_progress_bar=True).astype(np.float32)

        final_umap_metric = best_params.get('umap_metric', 'cosine')
        final_umap_model = UMAP(n_neighbors=best_params['umap_n_neighbors'],
                                n_components=best_params['umap_n_components'],
                                min_dist=best_params['umap_min_dist'],
                                metric=final_umap_metric, random_state=42,
                                low_memory=True, verbose=True) # low_memory=True and verbose for final model on full data

        final_hdbscan_min_cluster_size = best_params['hdbscan_min_cluster_size']
        # Recalculate min_samples based on the logic in objective function
        if final_hdbscan_min_cluster_size <= 5:
            final_hdbscan_min_samples = best_params.get('hdbscan_min_samples', 1) # Fallback if not tuned that way
        else:
            final_hdbscan_min_samples = best_params.get('hdbscan_min_samples', max(2, final_hdbscan_min_cluster_size // 2))


        final_hdbscan_metric = best_params.get('hdbscan_metric', 'euclidean')
        final_hdbscan_model = HDBSCAN(min_cluster_size=final_hdbscan_min_cluster_size,
                                      min_samples=final_hdbscan_min_samples, # Use recalculated
                                      metric=final_hdbscan_metric,
                                      cluster_selection_method=best_params['hdbscan_cluster_selection_method'],
                                      # prediction_data=True, # Not a param for sklearn.cluster.HDBSCAN
                                      # gen_min_span_tree=True # Not a param for sklearn.cluster.HDBSCAN
                                      core_dist_n_jobs=-1) # Use multiple cores for final model if possible


        final_vectorizer_model = CountVectorizer(stop_words="english",
                                                 ngram_range=(1, best_params['ngram_max']),
                                                 min_df=best_params['vectorizer_min_df'],
                                                 max_df=best_params.get('vectorizer_max_df', 0.95))

        final_ctfidf_model_instance = ClassTfidfTransformer(
            reduce_frequent_words=best_params.get('ctfidf_reduce_frequent_words', True),
            bm25_weighting=best_params.get('ctfidf_bm25_weighting', False))

        final_nr_topics_strategy = best_params.get('nr_topics_strategy', 'auto')
        final_nr_topics_suggestion = None
        if final_nr_topics_strategy == 'target_range':
            final_nr_topics_suggestion = best_params.get('nr_topics_target', None)
            # Potentially re-scale nr_topics_target if full dataset is much larger than subset
            # This is heuristic:
            if len(DOCS_FULL) > 2 * MAX_DOCS_FOR_OPTUNA_TRIALS and final_nr_topics_suggestion is not None:
                scale_factor = len(DOCS_FULL) / MAX_DOCS_FOR_OPTUNA_TRIALS
                final_nr_topics_suggestion = int(final_nr_topics_suggestion * (scale_factor**0.5)) # Non-linear scaling
                final_nr_topics_suggestion = max(5, min(100, final_nr_topics_suggestion)) # Bounds
                logger.info(f"Adjusted nr_topics_target for full dataset to: {final_nr_topics_suggestion}")


        final_representation_choice = best_params.get('representation_model_type', 'default_ctfidf')
        final_current_representation_model = None
        if final_representation_choice == 'keybert':
            final_current_representation_model = KeyBERTInspired()
        elif final_representation_choice == 'mmr':
            final_mmr_diversity = best_params.get('mmr_diversity', 0.5)
            final_current_representation_model = MaximalMarginalRelevance(diversity=final_mmr_diversity)

        final_topic_model = BERTopic(
            embedding_model=None, umap_model=final_umap_model, hdbscan_model=final_hdbscan_model,
            vectorizer_model=final_vectorizer_model, ctfidf_model=final_ctfidf_model_instance,
            representation_model=final_current_representation_model,
            language="english", top_n_words=FINAL_MODEL_TOP_N_WORDS,
            calculate_probabilities=True, verbose=True, nr_topics=final_nr_topics_suggestion)

        logger.info("Fitting the final BERTopic model ON FULL DATASET...")
        try:
            final_topics, final_probabilities = final_topic_model.fit_transform(DOCS_FULL, embeddings=EMBEDDINGS_FULL)

            logger.info("\n--- Final Model Results (on Full Dataset) ---")
            final_topic_info = final_topic_model.get_topic_info()
            print("Topic Info (Sample):")
            print(final_topic_info.head(min(15, len(final_topic_info))))

            logger.info(f"\nKeywords for each topic (Top {min(10, len(final_topic_info)-1 if -1 in final_topic_info.Topic.values else len(final_topic_info) )} topics, Top 7 words):")
            topics_to_show_ids = [tid for tid in final_topic_info.sort_values(by="Count", ascending=False).Topic if tid != -1][:10]

            for topic_id in topics_to_show_ids:
                topic_name = final_topic_info[final_topic_info.Topic == topic_id]['Name'].values[0]
                topic_count = final_topic_info[final_topic_info.Topic == topic_id]['Count'].values[0]
                print(f"\nTopic {topic_id} ({topic_name}): Count={topic_count}")
                topic_words = final_topic_model.get_topic(topic_id)
                if topic_words: print(topic_words[:7])
                else: print("  (No words found for this topic)")

            final_model_path = "final_bertopic_model_optimized_full_data"
            # When saving, if you used a specific sentence_model instance for embeddings,
            # you can pass its name or the instance itself.
            # Here, we pass the name used for loading, assuming it can be loaded by SentenceTransformer
            final_topic_model.save(final_model_path, serialization="pickle", save_ctfidf=True, save_embedding_model=EMBEDDING_MODEL_NAME)
            logger.info(f"\nFinal model saved to '{final_model_path}'")
            # For loading: BERTopic.load(final_model_path, embedding_model=EMBEDDING_MODEL_NAME) # or the actual model instance if not a standard name

        except MemoryError as e:
            logger.error(f"MemoryError during FINAL model fitting on FULL dataset: {e}. Try reducing sentence count further or using a machine with more RAM.", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error during FINAL model fitting: {e}", exc_info=True)
    else:
        logger.info("No successful trials completed in the study, cannot train final model.")