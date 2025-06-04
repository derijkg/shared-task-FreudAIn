import json
import torch
from torch.utils.data import IterableDataset
from transformers.tokenization_utils_base import BatchEncoding # For your decode_from_json
import logging
import random
import numpy as np
import torch.nn.functional as F
from transformers import BertModel, BertConfig, get_linear_schedule_with_warmup
from typing import Optional, Tuple, Dict, Union
from torch import nn
# Removed optuna for this specific overfitting script
# import optuna
from torch.utils.data import DataLoader
import gc
# from transformers.tokenization_utils_base import BatchEncoding # Already imported
import torch.optim as optim
import os
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants for JSON (ensure these match what you used when saving) ---
_TENSOR_MARKER = "__tensor__"
_TENSOR_DTYPE_MARKER = "__tensor_dtype__"
_BATCH_ENCODING_MARKER = "__batch_encoding__"
_BATCH_ENCODING_DATA_MARKER = "data"

def _convert_str_to_dtype(dtype_str: str) -> torch.dtype:
    if not dtype_str.startswith("torch."):
        try:
            return torch.__getattribute__(dtype_str)
        except AttributeError:
            return torch.dtype(dtype_str)
    dtype_name = dtype_str.split('.')[1]
    return torch.__getattribute__(dtype_name)

def _json_object_hook_for_dataset(dct: dict) -> any:
    if _TENSOR_MARKER in dct:
        dtype_str = dct.get(_TENSOR_DTYPE_MARKER, 'float32')
        dtype = _convert_str_to_dtype(dtype_str)
        return torch.tensor(dct[_BATCH_ENCODING_DATA_MARKER], dtype=dtype)
    elif _BATCH_ENCODING_MARKER in dct:
        reconstructed_data_for_be = {}
        batch_encoding_payload = dct.get(_BATCH_ENCODING_DATA_MARKER, {})
        for k, v_data in batch_encoding_payload.items():
            if isinstance(v_data, list) and k in ["input_ids", "token_type_ids", "attention_mask"]:
                try:
                    tensor_dtype = torch.long if k in ["input_ids", "token_type_ids"] else torch.long
                    reconstructed_data_for_be[k] = torch.tensor(v_data, dtype=tensor_dtype)
                except Exception as e:
                    logger.error(f"Error converting field '{k}' in BatchEncoding to tensor: {e}. Keeping as list.")
                    reconstructed_data_for_be[k] = v_data
            else:
                reconstructed_data_for_be[k] = v_data
        return BatchEncoding(reconstructed_data_for_be)
    return dct

class JsonlIterableDataset(IterableDataset):
    def __init__(self, file_path, trait_names, n_comments_to_process,
                 other_numerical_feature_names, num_q_features_per_comment,
                 is_test_set=False, transform_fn=None, num_samples = None):
        super().__init__()
        self.file_path = file_path
        self.trait_names_ordered = trait_names
        self.n_comments_to_process = n_comments_to_process
        self.other_numerical_feature_names = other_numerical_feature_names
        self.num_q_features_per_comment = num_q_features_per_comment
        self.is_test_set = is_test_set
        self.transform_fn = self._default_transform if transform_fn is None else transform_fn
        if num_samples is None:
            logger.info(f'Counting samples in {file_path} for __len__ was not provided...')
            self.num_samples = self._count_samples_in_file()
            logger.info(f"Counted {self.num_samples} samples in {self.file_path}.")
        else:
            self.num_samples = num_samples
        if self.num_samples == 0:
            logger.warning(f"Initialized JsonlIterableDataset for {self.file_path} with 0 samples. DataLoader will be empty.")

    def _count_samples_in_file(self):
            count = 0
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    for _ in f:
                        count += 1
            except FileNotFoundError:
                logger.error(f"File not found during initial sample count: {self.file_path}. Returning 0 samples.")
                return 0
            except Exception as e:
                logger.error(f"Error during initial sample count for {self.file_path}: {e}. Returning 0 samples.")
                return 0
            return count

    def _process_line(self, line):
        try:
            sample = json.loads(line, object_hook=_json_object_hook_for_dataset)
            return self.transform_fn(sample, idx=None)
        except json.JSONDecodeError:
            logger.warning(f"Skipping line due to JSONDecodeError.")
            return None
        except Exception as e:
            logger.warning(f"Skipping line due to generic error in _process_line: {e}")
            return None

    def __len__(self):
        return self.num_samples

    def _default_transform(self, sample, idx):
        tokenized_info = sample.get('features', {}).get('comments_tokenized', {})
        all_input_ids = tokenized_info['input_ids']
        all_attention_mask = tokenized_info['attention_mask']

        num_actual_comments = all_input_ids.shape[0] if hasattr(all_input_ids, 'shape') else 0
        # Ensure all_input_ids has a sequence length dimension, even if empty or 1D
        seq_len = all_input_ids.shape[1] if num_actual_comments > 0 and all_input_ids.ndim == 2 else GLOBAL_CONFIG.get('TOKENIZER_MAX_LENGTH', 256)


        final_input_ids = torch.zeros((self.n_comments_to_process, seq_len), dtype=torch.long)
        final_attention_mask = torch.zeros((self.n_comments_to_process, seq_len), dtype=torch.long)
        comment_active_flags = torch.zeros(self.n_comments_to_process, dtype=torch.bool)

        indices_to_select = list(range(num_actual_comments))
        if num_actual_comments > self.n_comments_to_process:
            indices_to_select = random.sample(indices_to_select, self.n_comments_to_process)
            comments_to_fill = self.n_comments_to_process
        else:
            comments_to_fill = num_actual_comments

        for i in range(comments_to_fill):
            original_idx = indices_to_select[i]
            if all_input_ids.ndim == 2 and original_idx < all_input_ids.shape[0]: # Check original_idx
                 final_input_ids[i] = all_input_ids[original_idx]
                 final_attention_mask[i] = all_attention_mask[original_idx]
                 comment_active_flags[i] = True
            elif all_input_ids.ndim == 1 and i == 0 : # Special case for single comment stored as 1D
                 final_input_ids[i] = all_input_ids
                 final_attention_mask[i] = all_attention_mask
                 comment_active_flags[i] = True


        raw_q_scores = sample['features'].get('q_scores', [])
        final_q_scores = torch.zeros((self.n_comments_to_process, self.num_q_features_per_comment), dtype=torch.float)

        selected_raw_q_scores = []
        for i in range(comments_to_fill):
            original_comment_idx = indices_to_select[i] # Use the same selected indices
            if original_comment_idx < len(raw_q_scores):
                qs_for_comment = raw_q_scores[original_comment_idx][:self.num_q_features_per_comment]
                # Ensure qs_for_comment is a list of floats before padding
                if not isinstance(qs_for_comment, list): qs_for_comment = [] # Default to empty list if not list
                
                padded_qs = qs_for_comment + [0.0] * (self.num_q_features_per_comment - len(qs_for_comment))
                selected_raw_q_scores.append(padded_qs[:self.num_q_features_per_comment])
            else:
                selected_raw_q_scores.append([0.0] * self.num_q_features_per_comment)

        if comments_to_fill > 0 and selected_raw_q_scores:
            try:
                # Ensure all elements in selected_raw_q_scores are lists of numbers
                valid_scores = True
                for score_list in selected_raw_q_scores:
                    if not (isinstance(score_list, list) and all(isinstance(s, (int, float)) for s in score_list)):
                        valid_scores = False
                        break
                if valid_scores:
                    final_q_scores[:comments_to_fill] = torch.tensor(selected_raw_q_scores, dtype=torch.float)
                else:
                    logger.error(f"Invalid data in selected_raw_q_scores. Cannot convert to tensor. Data: {selected_raw_q_scores}")

            except Exception as e:
                logger.error(f"Error converting selected_raw_q_scores to tensor: {e}. Data: {selected_raw_q_scores}")


        other_numerical_features_list = []
        for fname in self.other_numerical_feature_names:
            val = sample['features'].get(fname, 0.0)
            try:
                other_numerical_features_list.append(float(val))
            except (ValueError, TypeError):
                other_numerical_features_list.append(0.0)
        other_numerical_features_tensor = torch.tensor(other_numerical_features_list, dtype=torch.float)

        if not self.is_test_set:
            labels_dict = sample['labels']
            regression_labels = []
            for trait_key in self.trait_names_ordered:
                label_val = labels_dict.get(trait_key.title(), labels_dict.get(trait_key, 0.0)) # Case-insensitive for trait key
                try:
                    label_float = float(label_val)
                    if not (0.0 <= label_float <= 1.0): label_float = np.clip(label_float, 0.0, 1.0)
                    regression_labels.append(label_float)
                except (ValueError, TypeError): regression_labels.append(0.0) # Default to 0.0 if conversion fails
            labels_tensor = torch.tensor(regression_labels, dtype=torch.float)
            return (final_input_ids, final_attention_mask, final_q_scores, comment_active_flags, other_numerical_features_tensor, labels_tensor)
        else:
            return (final_input_ids, final_attention_mask, final_q_scores, comment_active_flags, other_numerical_features_tensor)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        try:
            file_iter = open(self.file_path, 'r', encoding='utf-8')
        except FileNotFoundError:
            logger.error(f"File not found in __iter__: {self.file_path}. Yielding nothing.")
            return

        if worker_info is None: # Single-process data loading
            for line_idx, line in enumerate(file_iter):
                # logger.debug(f"Processing line {line_idx+1} in single worker mode.")
                processed_item = self._process_line(line)
                if processed_item:
                    yield processed_item
        else: # Multi-process data loading
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            # logger.debug(f"Worker {worker_id}/{num_workers} starting.")
            for i, line in enumerate(file_iter):
                if i % num_workers == worker_id:
                    # logger.debug(f"Worker {worker_id} processing line index {i} (original file index).")
                    processed_item = self._process_line(line)
                    if processed_item:
                        yield processed_item
            # logger.debug(f"Worker {worker_id}/{num_workers} finished processing its lines.")
        file_iter.close()

class PersonalityModelV3(nn.Module):
    def __init__(self,
                 bert_model_name: str,
                 num_traits: int,
                 n_comments_to_process: int = 3,
                 dropout_rate: float = 0.2, # Will set to 0 for overfitting
                 attention_hidden_dim: int = 128,
                 num_bert_layers_to_pool: int = 4,
                 num_q_features_per_comment: int = 3,
                 num_other_numerical_features: int = 0,
                 numerical_embedding_dim: int = 64,
                 num_additional_dense_layers: int = 0,
                 additional_dense_hidden_dim: int = 256,
                 additional_layers_dropout_rate: float = 0.3 # Will set to 0 for overfitting
                ):
        super().__init__()
        self.bert_config = BertConfig.from_pretrained(bert_model_name, output_hidden_states=True)
        self.bert = BertModel.from_pretrained(bert_model_name, config=self.bert_config)
        self.n_comments_to_process = n_comments_to_process
        self.num_bert_layers_to_pool = num_bert_layers_to_pool
        bert_hidden_size = self.bert.config.hidden_size
        self.num_q_features_per_comment = num_q_features_per_comment

        comment_feature_dim = bert_hidden_size + self.num_q_features_per_comment
        self.attention_w = nn.Linear(comment_feature_dim, attention_hidden_dim)
        self.attention_v = nn.Linear(attention_hidden_dim, 1, bias=False)
        
        # For overfitting, dropout_rate will be set to 0.0 when instantiating
        self.final_dropout_layer = nn.Dropout(dropout_rate)

        self.num_other_numerical_features = num_other_numerical_features
        self.uses_other_numerical_features = self.num_other_numerical_features > 0
        self.other_numerical_processor_output_dim = 0
        
        aggregated_comment_feature_dim = comment_feature_dim
        combined_input_dim_for_block = aggregated_comment_feature_dim

        if self.uses_other_numerical_features:
            self.other_numerical_processor_output_dim = numerical_embedding_dim
            self.other_numerical_processor = nn.Sequential(
                nn.Linear(self.num_other_numerical_features, self.other_numerical_processor_output_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate) # This dropout will also be 0 for overfitting
            )
            combined_input_dim_for_block += self.other_numerical_processor_output_dim
            logger.info(f"Model will use {self.num_other_numerical_features} other numerical features, processed to dim {self.other_numerical_processor_output_dim}.")
        else:
            logger.info("Model will NOT use other numerical features.")

        self.num_additional_dense_layers = num_additional_dense_layers
        self.additional_dense_block = nn.Sequential()
        current_dim_for_dense_block = combined_input_dim_for_block

        if self.num_additional_dense_layers > 0:
            logger.info(f"Model using {self.num_additional_dense_layers} additional dense layers with hidden_dim {additional_dense_hidden_dim} and dropout {additional_layers_dropout_rate}")
            for i in range(self.num_additional_dense_layers):
                self.additional_dense_block.add_module(f"add_dense_{i}_linear", nn.Linear(current_dim_for_dense_block, additional_dense_hidden_dim))
                self.additional_dense_block.add_module(f"add_dense_{i}_relu", nn.ReLU())
                # For overfitting, additional_layers_dropout_rate will be 0
                self.additional_dense_block.add_module(f"add_dense_{i}_dropout", nn.Dropout(additional_layers_dropout_rate))
                current_dim_for_dense_block = additional_dense_hidden_dim
            input_dim_for_regressors = current_dim_for_dense_block
        else:
            logger.info("Model not using additional dense layers. Will use final_dropout_layer if dropout_rate > 0.")
            input_dim_for_regressors = combined_input_dim_for_block

        self.trait_regressors = nn.ModuleList()
        for _ in range(num_traits):
            self.trait_regressors.append(
                nn.Sequential(
                    nn.Linear(input_dim_for_regressors, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1)
                )
            )
    def _pool_bert_layers(self, all_hidden_states: Tuple[torch.Tensor, ...], attention_mask: torch.Tensor) -> torch.Tensor:
        layers_to_pool = all_hidden_states[-self.num_bert_layers_to_pool:]
        pooled_outputs = []
        # Ensure attention_mask is expanded to match the dimensions of layer_hidden_states
        # Expected shape for layer_hidden_states: (batch_size * n_comments, seq_len, hidden_size)
        # Expected shape for attention_mask: (batch_size * n_comments, seq_len)
        expanded_attention_mask = attention_mask.unsqueeze(-1).expand_as(layers_to_pool[0])

        for layer_hidden_states in layers_to_pool:
            # Apply mask before sum
            masked_embeddings = layer_hidden_states * expanded_attention_mask
            sum_embeddings = torch.sum(masked_embeddings, dim=1)
            sum_mask = expanded_attention_mask.sum(dim=1)
            sum_mask = torch.clamp(sum_mask, min=1e-9) # Avoid division by zero
            pooled_outputs.append(sum_embeddings / sum_mask)
            
        stacked_pooled_outputs = torch.stack(pooled_outputs, dim=0)
        mean_pooled_layers_embedding = torch.mean(stacked_pooled_outputs, dim=0)
        return mean_pooled_layers_embedding

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                q_scores: torch.Tensor,
                comment_active_mask: torch.Tensor,
                other_numerical_features: Optional[torch.Tensor] = None
               ):
        batch_size = input_ids.shape[0]
        
        input_ids_flat = input_ids.view(-1, input_ids.shape[-1])
        attention_mask_flat = attention_mask.view(-1, attention_mask.shape[-1])
        
        bert_outputs = self.bert(input_ids=input_ids_flat, attention_mask=attention_mask_flat)
        comment_bert_embeddings_flat = self._pool_bert_layers(bert_outputs.hidden_states, attention_mask_flat)
        comment_bert_embeddings = comment_bert_embeddings_flat.view(batch_size, self.n_comments_to_process, -1)
        
        # Ensure q_scores are compatible for concatenation
        # q_scores expected shape: (batch_size, n_comments_to_process, num_q_features_per_comment)
        # comment_bert_embeddings shape: (batch_size, n_comments_to_process, bert_hidden_size)
        if q_scores.shape[0] != batch_size or q_scores.shape[1] != self.n_comments_to_process:
             raise ValueError(f"q_scores shape mismatch. Expected ({batch_size}, {self.n_comments_to_process}, ...), got {q_scores.shape}")

        comment_features_with_q = torch.cat((comment_bert_embeddings, q_scores), dim=2)
        
        u = torch.tanh(self.attention_w(comment_features_with_q))
        scores = self.attention_v(u).squeeze(-1) # Shape: (batch_size, n_comments_to_process)
        
        if comment_active_mask is not None:
             # Ensure comment_active_mask is boolean. If not, convert.
            if not comment_active_mask.dtype == torch.bool:
                comment_active_mask = comment_active_mask.bool()
            # Mask scores where comments are not active
            scores = scores.masked_fill(~comment_active_mask, -1e9) # Use ~ for boolean negation
            
        attention_weights = F.softmax(scores, dim=1) # Shape: (batch_size, n_comments_to_process)
        attention_weights_expanded = attention_weights.unsqueeze(-1) # Shape: (batch_size, n_comments_to_process, 1)
        
        # Weighted sum of comment features
        # comment_features_with_q shape: (batch_size, n_comments_to_process, feature_dim)
        aggregated_comment_features = torch.sum(attention_weights_expanded * comment_features_with_q, dim=1)

        final_features_for_processing = aggregated_comment_features
        if self.uses_other_numerical_features:
            if other_numerical_features is None or other_numerical_features.shape[1] != self.num_other_numerical_features:
                raise ValueError(
                    f"Other numerical features expected but not provided correctly. "
                    f"Expected {self.num_other_numerical_features}, got shape {other_numerical_features.shape if other_numerical_features is not None else 'None'}"
                )
            processed_other_numerical_features = self.other_numerical_processor(other_numerical_features)
            final_features_for_processing = torch.cat((aggregated_comment_features, processed_other_numerical_features), dim=1)
        
        if self.num_additional_dense_layers > 0:
            features_for_trait_heads = self.additional_dense_block(final_features_for_processing)
        else: # Apply final_dropout_layer only if no additional dense block
            features_for_trait_heads = self.final_dropout_layer(final_features_for_processing)
        
        trait_regression_outputs = []
        for regressor_head in self.trait_regressors:
            trait_regression_outputs.append(regressor_head(features_for_trait_heads))
        
        all_trait_outputs_raw = torch.cat(trait_regression_outputs, dim=1)
        all_trait_outputs_sigmoid = torch.sigmoid(all_trait_outputs_raw)
        
        return all_trait_outputs_sigmoid

    def predict_scores(self, outputs: torch.Tensor) -> torch.Tensor:
        return outputs

# --- CONFIGURATIONS (Similar to your Cell 2, but adapted) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# Assume your original train_data.jsonl exists
ORIGINAL_TRAIN_DATA_FILE = "train_data.jsonl"
OVERFIT_DATA_FILE = "overfit_data.jsonl" # New small file
NUM_SAMPLES_FOR_OVERFIT = 4 # e.g., one batch worth of data, or a few samples
OVERFIT_BATCH_SIZE = NUM_SAMPLES_FOR_OVERFIT # Train on the whole tiny dataset as one batch
# Or, for very small NUM_SAMPLES_FOR_OVERFIT (e.g. < 4), use a smaller batch size like 1 or 2
# OVERFIT_BATCH_SIZE = min(NUM_SAMPLES_FOR_OVERFIT, 4)


_trait_names_ordered_config = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Emotional stability', 'Humility']
_other_numerical_features_config = [
    'mean_words_per_comment', 'mean_sents_per_comment',
    'median_words_per_comment', 'mean_words_per_sentence', 'median_words_per_sentence',
    'sents_per_comment_skew', 'words_per_sentence_skew', 'total_double_whitespace',
    'punc_em_total', 'punc_qm_total', 'punc_period_total', 'punc_comma_total',
    'punc_colon_total', 'punc_semicolon_total', 'flesch_reading_ease_agg',
    'gunning_fog_agg', 'mean_word_len_overall', 'ttr_overall',
    'mean_sentiment_neg', 'mean_sentiment_neu', 'mean_sentiment_pos',
    'mean_sentiment_compound', 'std_sentiment_compound'
]

GLOBAL_CONFIG = {
    'BERT_MODEL_NAME': "bert-base-uncased",
    'TRAIT_NAMES_ORDERED': _trait_names_ordered_config,
    'TRAIT_NAMES': _trait_names_ordered_config, # Usually same as ordered for consistency
    'MAX_COMMENTS_TO_PROCESS_PHYSICAL': 6, # Or a value you expect to use
    'NUM_Q_FEATURES_PER_COMMENT': 3,
    'OTHER_NUMERICAL_FEATURE_NAMES': _other_numerical_features_config,
    'TOKENIZER_MAX_LENGTH': 256 # Ensure this matches your pre-tokenized data
}

def create_small_subset_file(original_file, subset_file, num_lines):
    """Creates a small subset file from the original."""
    lines_written = 0
    try:
        with open(original_file, 'r', encoding='utf-8') as infile, \
             open(subset_file, 'w', encoding='utf-8') as outfile:
            for i, line in enumerate(infile):
                if i < num_lines:
                    outfile.write(line)
                    lines_written +=1
                else:
                    break
        logger.info(f"Created subset file '{subset_file}' with {lines_written} lines from '{original_file}'.")
        if lines_written < num_lines and lines_written > 0:
            logger.warning(f"Original file '{original_file}' had fewer than {num_lines} lines. Subset contains all {lines_written} lines.")
        elif lines_written == 0:
            logger.error(f"Original file '{original_file}' might be empty or not found. Subset file is empty.")
            return 0
        return lines_written
    except FileNotFoundError:
        logger.error(f"Original file '{original_file}' not found. Cannot create subset.")
        return 0
    except Exception as e:
        logger.error(f"Error creating subset file: {e}")
        return 0

def run_overfitting_test(
    overfit_config: Dict,
    global_model_config: Dict,
    num_epochs: int,
    device: torch.device
):
    """Runs a training loop designed to overfit the model."""

    logger.info("--- Starting Overfitting Test ---")
    logger.info(f"Overfitting Config: {overfit_config}")

    # 1. Create/Verify the small data file
    actual_samples_in_subset = create_small_subset_file(
        ORIGINAL_TRAIN_DATA_FILE,
        OVERFIT_DATA_FILE,
        overfit_config['num_samples_for_overfit']
    )
    if actual_samples_in_subset == 0:
        logger.error("Overfitting data subset could not be created or is empty. Aborting test.")
        return

    # 2. Create Dataset and DataLoader
    try:
        overfit_dataset = JsonlIterableDataset(
            file_path=OVERFIT_DATA_FILE,
            trait_names=global_model_config['TRAIT_NAMES_ORDERED'],
            n_comments_to_process=overfit_config['n_comments_to_process'],
            other_numerical_feature_names=global_model_config.get('OTHER_NUMERICAL_FEATURE_NAMES', []),
            num_q_features_per_comment=global_model_config.get('NUM_Q_FEATURES_PER_COMMENT', 3),
            is_test_set=False,
            num_samples=actual_samples_in_subset # Use actual count
        )
        # If batch_size is same as num_samples, persistent_workers should be False or num_workers=0
        overfit_loader = DataLoader(
            overfit_dataset,
            batch_size=overfit_config['batch_size'],
            num_workers=0, # Simpler for small iterable datasets
            pin_memory=True if device.type == 'cuda' else False,
            persistent_workers=False
        )
    except Exception as e:
        logger.error(f"Error creating dataset/dataloader for overfitting: {e}", exc_info=True)
        return

    if len(overfit_loader) == 0:
        logger.error("DataLoader is empty. Check data file and dataset initialization. Aborting.")
        return

    # 3. Initialize Model
    model = PersonalityModelV3(
        bert_model_name=global_model_config['BERT_MODEL_NAME'],
        num_traits=len(global_model_config['TRAIT_NAMES']),
        n_comments_to_process=overfit_config['n_comments_to_process'],
        dropout_rate=overfit_config['dropout_rate'], # Key for overfitting
        attention_hidden_dim=overfit_config['attention_hidden_dim'],
        num_bert_layers_to_pool=overfit_config['num_bert_layers_to_pool'],
        num_q_features_per_comment=global_model_config.get('NUM_Q_FEATURES_PER_COMMENT', 3),
        num_other_numerical_features=len(global_model_config.get('OTHER_NUMERICAL_FEATURE_NAMES', [])),
        numerical_embedding_dim=overfit_config.get('other_numerical_embedding_dim', 0),
        num_additional_dense_layers=overfit_config.get('num_additional_dense_layers', 0),
        additional_dense_hidden_dim=overfit_config.get('additional_dense_hidden_dim', 0),
        additional_layers_dropout_rate=overfit_config.get('additional_layers_dropout_rate', 0.0) # Key
    ).to(device)

    # Optionally, unfreeze more BERT layers to increase capacity
    if overfit_config.get('unfreeze_all_bert', False):
        logger.info("Unfreezing all BERT parameters for overfitting test.")
        for param in model.bert.parameters():
            param.requires_grad = True
    elif overfit_config.get('num_unfrozen_bert_layers', 0) > 0:
        num_unfrozen = overfit_config['num_unfrozen_bert_layers']
        logger.info(f"Unfreezing last {num_unfrozen} BERT layers.")
        for name, param in model.bert.named_parameters(): param.requires_grad = False # Freeze all first
        if hasattr(model.bert, 'embeddings'):
            for param in model.bert.embeddings.parameters(): param.requires_grad = True
        actual_layers_to_unfreeze = min(num_unfrozen, model.bert.config.num_hidden_layers)
        for i in range(model.bert.config.num_hidden_layers - actual_layers_to_unfreeze, model.bert.config.num_hidden_layers):
            if i >= 0 and i < model.bert.config.num_hidden_layers:
                for param in model.bert.encoder.layer[i].parameters(): param.requires_grad = True
        if hasattr(model.bert, 'pooler') and model.bert.pooler is not None:
            for param in model.bert.pooler.parameters(): param.requires_grad = True


    # 4. Optimizer
    optimizer_grouped_parameters = []
    bert_params_to_tune = [p for p in model.bert.parameters() if p.requires_grad]
    if bert_params_to_tune and overfit_config['lr_bert'] > 0:
         optimizer_grouped_parameters.append({"params": bert_params_to_tune, "lr": overfit_config['lr_bert'], "weight_decay": 0.0}) # No weight decay

    head_params = []
    # Make sure to collect all non-BERT params that should be trained
    for name, module in model.named_children():
        if name != 'bert': # Assuming 'bert' is the name of your BertModel instance
            head_params.extend(list(module.parameters()))
    
    if head_params:
        optimizer_grouped_parameters.append({"params": head_params, "lr": overfit_config['lr_head'], "weight_decay": 0.0}) # No weight decay

    if not any(pg['params'] for pg in optimizer_grouped_parameters if pg.get('params')):
        logger.warning("No parameters to optimize. Skipping training.")
        return

    optimizer = optim.AdamW(optimizer_grouped_parameters)
    loss_fn = nn.L1Loss().to(device) # Or nn.MSELoss() if you prefer for monitoring

    # 5. Training Loop
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        batches_processed = 0
        for batch_idx, batch_tuple in enumerate(overfit_loader):
            try:
                input_ids, attention_m, q_s, comment_active_m, other_num_feats, labels_reg = [b.to(device) for b in batch_tuple]
            except ValueError as e:
                logger.error(f"Error unpacking batch tuple at epoch {epoch+1}, batch {batch_idx}: {e}. Skipping batch.")
                logger.error(f"Batch tuple items: {[type(b) for b in batch_tuple]}")
                continue # Skip this batch


            optimizer.zero_grad()
            predicted_scores = model(input_ids, attention_m, q_s, comment_active_m, other_num_feats)
            current_batch_loss = loss_fn(predicted_scores, labels_reg)

            if torch.isnan(current_batch_loss) or torch.isinf(current_batch_loss):
                logger.warning(f"Epoch {epoch+1}, Batch {batch_idx}: NaN or Inf loss detected. Skipping update.")
                torch.cuda.empty_cache()
                continue

            current_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Keep grad clipping
            optimizer.step()
            total_train_loss += current_batch_loss.item()
            batches_processed += 1
        
        if batches_processed > 0:
            avg_train_loss = total_train_loss / batches_processed
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Avg Train Loss: {avg_train_loss:.6f}")
            if avg_train_loss < 1e-5: # If loss is very small, it's likely overfit
                logger.info("Training loss is very low. Model has likely overfit the small dataset.")
                break
        else:
            logger.warning(f"Epoch {epoch+1}/{num_epochs} - No batches processed. DataLoader might be empty or data issues.")
            # This might happen if create_small_subset_file returned 0 or dataset issues.
            # Or if the single batch had an error during unpacking.
            if epoch > 0 and len(overfit_loader) > 0: # If not first epoch and loader *should* have data
                logger.error("No batches processed in an epoch where data was expected. Check dataset integrity and batch processing loop.")


    logger.info("--- Overfitting Test Finished ---")
    # Optionally save the overfit model
    # torch.save(model.state_dict(), "overfit_model_final.pth")
    # logger.info("Saved overfit model state to overfit_model_final.pth")

    del model, overfit_loader, overfit_dataset, optimizer
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == '__main__':
    # --- Define Overfitting Parameters ---
    # These parameters are chosen to encourage overfitting
    overfitting_specific_config = {
        'num_samples_for_overfit': NUM_SAMPLES_FOR_OVERFIT, # How many samples to take from train_data.jsonl
        'batch_size': OVERFIT_BATCH_SIZE,       # Use a batch size equal to the number of samples to process all in one go
        'n_comments_to_process': GLOBAL_CONFIG['MAX_COMMENTS_TO_PROCESS_PHYSICAL'], # Use a reasonable number, or same as your main config
        'dropout_rate': 0.0,                 # No dropout
        'additional_layers_dropout_rate': 0.0, # No dropout in additional layers
        'attention_hidden_dim': 128,         # Standard value
        'num_bert_layers_to_pool': 2,        # Standard value
        'lr_bert': 1e-5,                     # Learning rate for BERT parts (if unfrozen)
        'lr_head': 1e-3,                     # Learning rate for the rest of the model
        'num_unfrozen_bert_layers': 1,       # Unfreeze some BERT layers to give more capacity
        #'unfreeze_all_bert': True,         # Alternative: unfreeze everything in BERT for max capacity
        'other_numerical_embedding_dim': 64 if GLOBAL_CONFIG.get('OTHER_NUMERICAL_FEATURE_NAMES') else 0,
        'num_additional_dense_layers': 1,    # Can add a layer
        'additional_dense_hidden_dim': 128   # If using additional layers
    }

    NUM_EPOCHS_FOR_OVERFIT = 500 # Run for many epochs on the small dataset

    # Call the overfitting test function
    # Ensure ORIGINAL_TRAIN_DATA_FILE exists and has at least NUM_SAMPLES_FOR_OVERFIT lines
    if not os.path.exists(ORIGINAL_TRAIN_DATA_FILE):
        logger.error(f"The {ORIGINAL_TRAIN_DATA_FILE} is required for the overfitting test but was not found.")
        # Create a dummy train_data.jsonl for testing if you don't have the real one
        # This is just for the script to run. Replace with your actual data creation.
        if not os.path.exists(ORIGINAL_TRAIN_DATA_FILE):
            logger.info(f"Creating a dummy {ORIGINAL_TRAIN_DATA_FILE} for demonstration purposes.")
            dummy_sample = {
                "features": {
                    "comments_tokenized": {
                        # Corrected: BatchEncoding expects data as dict of lists/tensors
                        _BATCH_ENCODING_MARKER: True,
                        _BATCH_ENCODING_DATA_MARKER: {
                            "input_ids": [[101, 100, 102] + [0]* (GLOBAL_CONFIG['TOKENIZER_MAX_LENGTH']-3)] * GLOBAL_CONFIG['MAX_COMMENTS_TO_PROCESS_PHYSICAL'],
                            "attention_mask": [[1,1,1] + [0]* (GLOBAL_CONFIG['TOKENIZER_MAX_LENGTH']-3)] * GLOBAL_CONFIG['MAX_COMMENTS_TO_PROCESS_PHYSICAL']
                        }
                    },
                    "q_scores": [[0.1, 0.2, 0.3]] * GLOBAL_CONFIG['MAX_COMMENTS_TO_PROCESS_PHYSICAL'],
                },
                "labels": {trait: random.random() for trait in GLOBAL_CONFIG['TRAIT_NAMES_ORDERED']}
            }
            # Add other numerical features if your config uses them
            if GLOBAL_CONFIG.get('OTHER_NUMERICAL_FEATURE_NAMES'):
                for fname in GLOBAL_CONFIG['OTHER_NUMERICAL_FEATURE_NAMES']:
                    dummy_sample["features"][fname] = random.random()

            with open(ORIGINAL_TRAIN_DATA_FILE, 'w') as f:
                for _ in range(NUM_SAMPLES_FOR_OVERFIT * 2): # Create a bit more than needed
                    f.write(json.dumps(dummy_sample) + '\n')
    
    run_overfitting_test(
        overfit_config=overfitting_specific_config,
        global_model_config=GLOBAL_CONFIG,
        num_epochs=NUM_EPOCHS_FOR_OVERFIT,
        device=DEVICE
    )