import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig, BertTokenizerFast, get_linear_schedule_with_warmup
from typing import Optional, Tuple, List, Dict
import random
from torch.utils.data import Dataset, DataLoader
import json
import pandas as pd
import optuna
import torch.optim as optim
import os
import logging # Import logging
import gc

# --- Setup Logger ---
# Configure logging once at the start
# You can customize the format, level, and handlers (e.g., FileHandler)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- CORAL-style Loss Function (Helper - UNCHANGED, but could add logging if needed) ---
def coral_style_loss_calculation(logits_per_trait: torch.Tensor,
                                 labels_ordinal_per_trait: torch.Tensor,
                                 num_classes: int,
                                 device: torch.device) -> torch.Tensor:
    loss = 0.0
    if num_classes <= 1:
        return torch.tensor(0.0, device=device, requires_grad=True)
    bce_loss_fn = nn.BCEWithLogitsLoss().to(device)
    for k in range(num_classes - 1):
        binary_target_k = (labels_ordinal_per_trait > k).float().to(device)
        loss += bce_loss_fn(logits_per_trait[:, k], binary_target_k)
    return loss / (num_classes - 1)


# --- PersonalityModelV2 (UNCHANGED, but could add logging if needed for debugging its internals) ---
class PersonalityModelV2(nn.Module):
    def __init__(self,
                 bert_model_name: str,
                 num_traits: int,
                 ordinal_values_per_trait: int,
                 n_comments_to_process: int = 3,
                 dropout_rate: float = 0.2,
                 attention_hidden_dim: int = 128,
                 num_bert_layers_to_pool: int = 4):
        super().__init__()
        self.bert_config = BertConfig.from_pretrained(bert_model_name, output_hidden_states=True)
        self.bert = BertModel.from_pretrained(bert_model_name, config=self.bert_config)
        self.n_comments_to_process = n_comments_to_process
        self.ordinal_values_per_trait = ordinal_values_per_trait
        self.num_binary_classifiers_per_trait = ordinal_values_per_trait - 1
        if self.ordinal_values_per_trait <=1:
             raise ValueError("ordinal_values_per_trait must be at least 2 for CORAL-style classification.")
        self.num_bert_layers_to_pool = num_bert_layers_to_pool
        bert_hidden_size = self.bert.config.hidden_size
        self.attention_w = nn.Linear(bert_hidden_size, attention_hidden_dim)
        self.attention_v = nn.Linear(attention_hidden_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.trait_classifiers = nn.ModuleList()
        for _ in range(num_traits):
            self.trait_classifiers.append(
                nn.Linear(bert_hidden_size, self.num_binary_classifiers_per_trait)
            )

    def _pool_bert_layers(self, all_hidden_states: Tuple[torch.Tensor, ...], attention_mask: torch.Tensor) -> torch.Tensor:
        layers_to_pool = all_hidden_states[-self.num_bert_layers_to_pool:]
        pooled_outputs = []
        expanded_attention_mask = attention_mask.unsqueeze(-1).expand_as(layers_to_pool[0])
        for layer_hidden_states in layers_to_pool:
            sum_embeddings = torch.sum(layer_hidden_states * expanded_attention_mask, dim=1)
            sum_mask = expanded_attention_mask.sum(dim=1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            pooled_outputs.append(sum_embeddings / sum_mask)
        stacked_pooled_outputs = torch.stack(pooled_outputs, dim=0)
        mean_pooled_layers_embedding = torch.mean(stacked_pooled_outputs, dim=0)
        return mean_pooled_layers_embedding

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                comment_active_mask: torch.Tensor):
        batch_size = input_ids.shape[0]
        input_ids_flat = input_ids.view(-1, input_ids.shape[-1])
        attention_mask_flat = attention_mask.view(-1, attention_mask.shape[-1])
        outputs = self.bert(input_ids=input_ids_flat, attention_mask=attention_mask_flat)
        comment_embeddings_flat = self._pool_bert_layers(outputs.hidden_states, attention_mask_flat)
        comment_embeddings = comment_embeddings_flat.view(batch_size, self.n_comments_to_process, -1)
        u = torch.tanh(self.attention_w(comment_embeddings))
        scores = self.attention_v(u).squeeze(-1)
        if comment_active_mask is not None:
            scores = scores.masked_fill(~comment_active_mask, -1e9)
        attention_weights = F.softmax(scores, dim=1)
        attention_weights_expanded = attention_weights.unsqueeze(-1)
        aggregated_comment_embedding = torch.sum(attention_weights_expanded * comment_embeddings, dim=1)
        combined_features_dropped = self.dropout(aggregated_comment_embedding)
        trait_specific_logits = []
        for classifier_head in self.trait_classifiers:
            trait_specific_logits.append(classifier_head(combined_features_dropped))
        all_logits = torch.cat(trait_specific_logits, dim=1)
        return all_logits

    def predict_classes(self, logits: torch.Tensor) -> torch.Tensor:
        batch_size = logits.shape[0]
        if self.num_binary_classifiers_per_trait == 0:
            if self.ordinal_values_per_trait == 1:
                 num_traits_from_model = len(self.trait_classifiers)
                 return torch.zeros(batch_size, num_traits_from_model, dtype=torch.long, device=logits.device)
            # This case should ideally be caught by the __init__ check for ordinal_values_per_trait > 1
            # If somehow it's not, this indicates a setup error.
            logger.error("predict_classes called with num_binary_classifiers_per_trait=0 but ordinal_values_per_trait > 1. This is an inconsistent state.")
            # Fallback or raise error. For now, assuming num_traits can be derived.
            # This indicates an issue in model setup logic if hit.
            num_traits_from_model = len(self.trait_classifiers)
            return torch.zeros(batch_size, num_traits_from_model, dtype=torch.long, device=logits.device)


        num_total_binary_outputs = logits.shape[1]
        num_traits = num_total_binary_outputs // self.num_binary_classifiers_per_trait
        logits_reshaped = logits.view(batch_size, num_traits, self.num_binary_classifiers_per_trait)
        probs_greater_than_k = torch.sigmoid(logits_reshaped)
        predicted_classes = (probs_greater_than_k > 0.5).sum(dim=2)
        return predicted_classes


# --- PersonalityDataset (UNCHANGED, but could add logging if needed) ---
class PersonalityDataset(Dataset):
    def __init__(self,
                 data: List[Dict],
                 tokenizer: BertTokenizerFast,
                 max_seq_length: int,
                 trait_names: List[str],
                 ordinal_values_per_trait: int,
                 num_comments_to_process: int): # Make sure this is used
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.trait_names = trait_names
        self.ordinal_values_per_trait = ordinal_values_per_trait
        self.num_comments_to_process = num_comments_to_process # Crucial
        self.num_traits = len(trait_names)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        user_comments_all = sample['comments']
        if len(user_comments_all) > self.num_comments_to_process:
            comments_to_process_or_pad = random.sample(user_comments_all, self.num_comments_to_process)
        else:
            comments_to_process_or_pad = user_comments_all
        processed_comments_input_ids = []
        processed_comments_attention_mask = []
        active_comment_flags = []
        num_actual_comments = len(comments_to_process_or_pad)
        for i in range(self.num_comments_to_process):
            if i < num_actual_comments:
                comment_text = comments_to_process_or_pad[i]
                active_comment_flags.append(True)
            else:
                comment_text = self.tokenizer.pad_token
                active_comment_flags.append(False)
            encoding = self.tokenizer.encode_plus(
                comment_text, add_special_tokens=True, max_length=self.max_seq_length,
                padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt'
            )
            processed_comments_input_ids.append(encoding['input_ids'].squeeze(0))
            processed_comments_attention_mask.append(encoding['attention_mask'].squeeze(0))
        input_ids_tensor = torch.stack(processed_comments_input_ids)
        attention_mask_tensor = torch.stack(processed_comments_attention_mask)
        comment_active_mask_tensor = torch.tensor(active_comment_flags, dtype=torch.bool)
        integer_labels = []
        for trait_name in self.trait_names:
            label = sample['labels'][trait_name]
            if not isinstance(label, int):
                try: label = int(label)
                except ValueError:
                    raise ValueError(f"Label for trait {trait_name} in sample {sample.get('id', idx)} is not int and cannot be cast: {label} ({type(label)})")
            if not (0 <= label < self.ordinal_values_per_trait):
                raise ValueError(f"Label {label} for trait {trait_name} in sample {sample.get('id', idx)} out of range [0, {self.ordinal_values_per_trait-1}]")
            integer_labels.append(label)
        return (
            input_ids_tensor,
            attention_mask_tensor,
            comment_active_mask_tensor,
            torch.tensor(integer_labels, dtype=torch.long)
        )


# --- Optuna Objective Function (Modified with logging) ---
def objective(trial: optuna.trial.Trial,
              # Pass global data/tokenizer, not DataLoaders directly if params change
              full_train_data: List[Dict],
              full_val_data: List[Dict],
              tokenizer: BertTokenizerFast,
              global_config: Dict, # To pass SEQ_LEN, TRAIT_NAMES etc.
              device: torch.device,
              num_epochs: int = 10): # Default num_epochs for a trial

    logger.info(f"Starting Optuna Trial {trial.number}")

    num_traits = len(global_config['TRAIT_NAMES'])
    ordinal_values_per_trait = global_config['ORDINAL_VALUES_PER_TRAIT']
    bert_model_name = global_config['BERT_MODEL_NAME']

    # --- Suggest Hyperparameters ---
    # Existing
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    attention_hidden_dim = trial.suggest_categorical("attention_hidden_dim", [64, 128, 256])
    lr_bert = trial.suggest_float("lr_bert", 1e-6, 5e-4, log=True)
    lr_head = trial.suggest_float("lr_head", 5e-5, 5e-3, log=True) # Consider a slightly wider range like 5e-5 to 5e-3
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True) # Adjusted lower bound
    num_bert_layers_to_pool = trial.suggest_int("num_bert_layers_to_pool", 1, 4)

    # New
    n_comments_trial = trial.suggest_int("n_comments_to_process", 3, 15) # Example range
    # For bert-base (12 layers), 0 means freeze all BERT, 1-12 means unfreeze top N
    num_unfrozen_bert_layers = trial.suggest_int("num_unfrozen_bert_layers", 0, 12)
    patience_early_stopping = trial.suggest_int("patience_early_stopping", 3, 6)
    # Optional: Learning rate scheduler type
    scheduler_type = trial.suggest_categorical("scheduler_type", ["none", "linear_warmup"])
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.15) if scheduler_type != "none" else 0.0
    batch_size_trial = trial.suggest_categorical("batch_size", [8, 16])


    logger.info(f"Trial {trial.number} - Suggested Parameters: {trial.params}")

    # --- Create Datasets and DataLoaders for this trial ---
    # (Because n_comments_to_process can change per trial)
    train_dataset_trial = PersonalityDataset(
        data=full_train_data, tokenizer=tokenizer,
        max_seq_length=global_config['SEQ_LEN'],
        trait_names=global_config['TRAIT_NAMES'],
        ordinal_values_per_trait=global_config['ORDINAL_VALUES_PER_TRAIT'],
        num_comments_to_process=n_comments_trial
    )
    val_dataset_trial = PersonalityDataset(
        data=full_val_data, tokenizer=tokenizer,
        max_seq_length=global_config['SEQ_LEN'],
        trait_names=global_config['TRAIT_NAMES'],
        ordinal_values_per_trait=global_config['ORDINAL_VALUES_PER_TRAIT'],
        num_comments_to_process=n_comments_trial # Use trial specific value
    )
    train_loader_trial = DataLoader(train_dataset_trial, batch_size=batch_size_trial, shuffle=True)
    val_loader_trial = DataLoader(val_dataset_trial, batch_size=batch_size_trial, shuffle=False)



    # --- Instantiate Model ---
    model = PersonalityModelV2(
        bert_model_name=global_config['BERT_MODEL_NAME'],
        num_traits=len(global_config['TRAIT_NAMES']),
        ordinal_values_per_trait=global_config['ORDINAL_VALUES_PER_TRAIT'],
        n_comments_to_process=n_comments_trial, # Use trial specific value
        dropout_rate=dropout_rate,
        attention_hidden_dim=attention_hidden_dim,
        num_bert_layers_to_pool=num_bert_layers_to_pool
    ).to(device)

    # --- BERT Layer Freezing ---
    # Freeze all BERT layers initially
    for param in model.bert.parameters():
        param.requires_grad = False

    # Unfreeze specified top layers and embeddings
    if num_unfrozen_bert_layers > 0:
        # Unfreeze embedding layer always if any transformer layer is unfrozen
        for param in model.bert.embeddings.parameters():
            param.requires_grad = True
        
        # Unfreeze transformer layers from the top
        for i in range(model.bert.config.num_hidden_layers - num_unfrozen_bert_layers, model.bert.config.num_hidden_layers):
            for param in model.bert.encoder.layer[i].parameters():
                param.requires_grad = True
        
        # Unfreeze pooler if it exists (though we are not using model.bert.pooler directly, it's good practice if it were used)
        if hasattr(model.bert, 'pooler') and model.bert.pooler is not None:
            for param in model.bert.pooler.parameters():
                param.requires_grad = True
    
    logger.debug(f"Trial {trial.number} - BERT parameters requiring grad: "
                 f"{sum(p.numel() for p in model.bert.parameters() if p.requires_grad)}")


    # --- Optimizer ---
    optimizer_grouped_parameters = []
    # Add BERT parameters only if they require grad and lr_bert is positive
    bert_params_to_tune = [p for p in model.bert.parameters() if p.requires_grad]
    if bert_params_to_tune and lr_bert > 0:
         optimizer_grouped_parameters.append({"params": bert_params_to_tune, "lr": lr_bert})
    
    # Add other parameters (attention, classifiers)
    optimizer_grouped_parameters.extend([
        {"params": model.attention_w.parameters(), "lr": lr_head},
        {"params": model.attention_v.parameters(), "lr": lr_head},
    ])
    for classifier_head in model.trait_classifiers:
        optimizer_grouped_parameters.append({"params": classifier_head.parameters(), "lr": lr_head})
    
    if not optimizer_grouped_parameters: # Should not happen if heads are always trained
        logger.warning(f"Trial {trial.number} - No parameters to optimize. Skipping training.")
        return 0.0 # Or some other indicator of failure

    optimizer = optim.AdamW(optimizer_grouped_parameters, weight_decay=weight_decay)

    # --- Optional: Learning Rate Scheduler ---

    if scheduler_type == "linear_warmup":
        num_training_steps = len(train_loader_trial) * num_epochs
        num_warmup_steps = int(num_training_steps * warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    elif scheduler_type == "none":
        scheduler = None



    # --- Training & Validation Loop with Early Stopping ---
    best_trial_val_accuracy = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for batch_idx, batch_tuple in enumerate(train_loader_trial):
            input_ids, attention_m, comment_active_m, labels_ord = [b.to(device) for b in batch_tuple]
            optimizer.zero_grad()
            all_logits = model(input_ids, attention_m, comment_active_m)
            logits_reshaped = all_logits.view(
                input_ids.shape[0],      # batch_size (e.g., 8)
                num_traits,              # number of traits (e.g., 6)
                model.num_binary_classifiers_per_trait # (e.g., 2)
            )

            current_batch_loss = 0
            num_actual_traits = logits_reshaped.size(1) # Get actual number of traits from reshaped logits
            for i in range(num_actual_traits):
                trait_logits = logits_reshaped[:, i, :]
                trait_labels_ordinal = labels_ord[:, i]
                current_batch_loss += coral_style_loss_calculation(
                    trait_logits, trait_labels_ordinal, model.ordinal_values_per_trait, device
                )
            
            if num_actual_traits > 0:
                final_loss = current_batch_loss / num_actual_traits
                if final_loss.requires_grad:
                    final_loss.backward()
                    # Optional: Gradient Clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    if scheduler and scheduler_type == "linear_warmup":
                        scheduler.step()
                total_train_loss += final_loss.item()
            else:
                total_train_loss += 0.0
            


        avg_train_loss = total_train_loss / len(train_loader_trial) if len(train_loader_trial) > 0 else 0.0
        logger.info(f"Trial {trial.number}, Epoch {epoch+1}/{num_epochs} completed. Avg Train Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        current_epoch_val_loss = 0
        current_epoch_all_val_preds = []
        current_epoch_all_val_labels = []
        with torch.no_grad():
            for batch_tuple in val_loader_trial:
                input_ids, attention_m, comment_active_m, labels_ord = [b.to(device) for b in batch_tuple]
                if input_ids.numel() == 0: continue
                all_logits = model(input_ids, attention_m, comment_active_m)
                if all_logits.numel() == 0: continue
                
                num_actual_traits_val = len(model.trait_classifiers) # Assuming this is consistent
                logits_reshaped = all_logits.view(
                    input_ids.shape[0],      # batch_size (e.g., 8)
                    num_traits,              # number of traits (e.g., 6)
                    model.num_binary_classifiers_per_trait # (e.g., 2)
                )
                batch_val_loss = 0
                for i in range(num_actual_traits_val):
                    trait_logits = logits_reshaped[:, i, :]
                    trait_labels_ordinal = labels_ord[:, i]
                    batch_val_loss += coral_style_loss_calculation(
                        trait_logits, trait_labels_ordinal, model.ordinal_values_per_trait, device
                    )
                if num_actual_traits_val > 0:
                    final_val_loss_batch = batch_val_loss / num_actual_traits_val
                    current_epoch_val_loss += final_val_loss_batch.item()
                
                predicted_classes_batch = model.predict_classes(all_logits)
                current_epoch_all_val_preds.append(predicted_classes_batch.cpu())
                current_epoch_all_val_labels.append(labels_ord.cpu())
        
        avg_val_loss_epoch = current_epoch_val_loss / len(val_loader_trial) if len(val_loader_trial) > 0 else 0.0
        current_epoch_val_accuracy = 0.0
        if current_epoch_all_val_labels:
            all_val_labels_cat_epoch = torch.cat(current_epoch_all_val_labels, dim=0)
            if all_val_labels_cat_epoch.numel() > 0:
                all_val_preds_cat_epoch = torch.cat(current_epoch_all_val_preds, dim=0)
                correct_predictions_epoch = (all_val_preds_cat_epoch == all_val_labels_cat_epoch).float().sum()
                total_predictions_epoch = all_val_labels_cat_epoch.numel()
                current_epoch_val_accuracy = correct_predictions_epoch / total_predictions_epoch
        
        logger.info(f"Trial {trial.number}, Epoch {epoch+1} Val Loss: {avg_val_loss_epoch:.4f}, Val Accuracy: {current_epoch_val_accuracy:.4f}")

        if current_epoch_val_accuracy > best_trial_val_accuracy:
            best_trial_val_accuracy = current_epoch_val_accuracy
            patience_counter = 0
            logger.debug(f"Trial {trial.number}, Epoch {epoch+1}: New best val_accuracy for this trial: {best_trial_val_accuracy:.4f}")
        else:
            patience_counter += 1
        
        trial.report(current_epoch_val_accuracy, epoch)
        if trial.should_prune():
            logger.info(f"Trial {trial.number} pruned by Optuna at epoch {epoch+1}.")
            return best_trial_val_accuracy 
        
        if patience_counter >= patience_early_stopping:
            logger.info(f"Trial {trial.number} - Early stopping at epoch {epoch+1} (Patience: {patience_early_stopping}).")
            break
            
    logger.info(f"Trial {trial.number} finished. Best Val Accuracy for this trial: {best_trial_val_accuracy:.4f}")
    return best_trial_val_accuracy


# --- Helper Functions (Modified with logging) ---
def test_data_transform(path):
    logger.info(f"Transforming test data from: {path}")
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        logger.error(f"Validation data file not found: {path}")
        raise
    cols = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Humility']
    conversion = {'low': 0, 'medium': 1, 'high': 2}
    df[cols] = df[cols].apply(lambda col: col.map(conversion))
    conversion_es_neuro = {'low': 2, 'medium': 1, 'high': 0}
    df['Emotional stability'] = df['Emotional stability'].map(conversion_es_neuro)
    data = []
    for idx, row in df.iterrows():
        comments = [str(row[col]) if pd.notna(row[col]) else "" for col in ['Q1','Q2','Q3']]
        labels = {
            'openness': int(row['Openness']), 'conscientiousness': int(row['Conscientiousness']),
            'extraversion': int(row['Extraversion']), 'agreeableness': int(row['Agreeableness']),
            'neuroticism': int(row['Emotional stability']), 'humility': int(row['Humility'])
        }
        new_dict ={'id': row['id'], 'comments': comments, 'labels': labels}
        data.append(new_dict)
    logger.info(f"Finished transforming test data. {len(data)} samples processed.")
    return data

def transform_train_label(value, trait_name="<unknown>", sample_id="<unknown>"): # Added context for logging
    if not isinstance(value, (int, float)):
        logger.warning(f"Non-numeric label value '{value}' for trait '{trait_name}' in sample '{sample_id}'. Defaulting to 0.")
        return 0
    if value <= 0.33: return 0
    elif value <= 0.66: return 1
    else: return 2

def transform_train_data(path):
    logger.info(f"Transforming training data from: {path}")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            raw_train_data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Training data file not found: {path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from training data file: {path}")
        raise

    transformed_data = []
    for author_original in raw_train_data:
        author = author_original.copy()
        author_id = author.get('id', '<unknown_id>') # Get author_id for logging
        labels = author.get('labels', {})
        new_labels = {}
        trait_keys = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism', 'humility']
        for key in trait_keys:
            new_labels[key] = transform_train_label(labels.get(key, 0.0), trait_name=key, sample_id=author_id) # Pass context
        author['labels'] = new_labels
        if 'comments' in author and isinstance(author['comments'], list):
            author['comments'] = [str(c) if c is not None else "" for c in author['comments']]
        else:
            logger.warning(f"Comments missing or malformed for sample '{author_id}'. Using default empty comments.")
            author['comments'] = ["", "", ""]
        transformed_data.append(author)
    logger.info(f"Finished transforming training data. {len(transformed_data)} samples processed.")
    return transformed_data


# --- Main Script ---
if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {DEVICE}")

    # --- Global Configuration (to be passed to objective) ---
    GLOBAL_CONFIG = {
        'BERT_MODEL_NAME': "bert-base-uncased",
        'TRAIT_NAMES': ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism', 'humility'],
        'SEQ_LEN': 128,
        'ORDINAL_VALUES_PER_TRAIT': 3,
        'BATCH_SIZE': 16, # Optuna can also tune batch size if memory allows
    }

    NUM_EPOCHS_PER_TRIAL = 25 # Max epochs per trial, early stopping will handle actual
    N_OPTUNA_TRIALS = 30     # Number of Optuna trials

    # --- DATA Loading and Preprocessing ---
    base_data_path = os.path.join("..", "shared task","data")
    train_data_path = os.path.join(base_data_path, "humility_added.json")
    val_data_path = os.path.join(base_data_path, "val_data.csv")

    logger.info(f"Attempting to load training data from: {os.path.abspath(train_data_path)}")
    logger.info(f"Attempting to load validation data from: {os.path.abspath(val_data_path)}")

    # Load data once
    full_train_data = transform_train_data(train_data_path)
    full_val_data = test_data_transform(val_data_path)

    tokenizer = BertTokenizerFast.from_pretrained(GLOBAL_CONFIG['BERT_MODEL_NAME'])
    logger.info("Tokenizer loaded.")

    logger.info(f"Full train dataset size: {len(full_train_data)}, Full val dataset size: {len(full_val_data)}")
    if len(full_train_data) == 0 or len(full_val_data) == 0:
        logger.error("One of the datasets is empty. Please check data loading and paths.")
        exit()

    # --- Optuna Study ---
    logger.info(f"Starting Optuna study: {N_OPTUNA_TRIALS} trials, up to {NUM_EPOCHS_PER_TRIAL} epochs/trial (with early stopping).")
    # optuna.logging.set_verbosity(optuna.logging.WARNING) # Control Optuna's own logger

    study = optuna.create_study(study_name='personality_bertv3',
                                direction="maximize",
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
                                storage="sqlite:///personality_bertv3.db",
                                load_if_exists=True)
    if study.trials:
        logger.info(f"Resuming existing study {study.study_name} with {len(study.trials)} trials.")
    
    try:
        study.optimize(
            lambda trial: objective(
                trial,
                full_train_data, # Pass full data lists
                full_val_data,
                tokenizer,
                GLOBAL_CONFIG,
                DEVICE,
                num_epochs=NUM_EPOCHS_PER_TRIAL
            ),
            n_trials=N_OPTUNA_TRIALS,
            gc_after_trial=True # Helps manage memory with large models
        )
    except Exception as e:
        logger.exception("An error occurred during the Optuna study.")
        raise

    logger.info("\n--- Optuna Study Finished ---")
    logger.info(f"Number of finished trials: {len(study.trials)}")
    
    best_trial = study.best_trial
    logger.info(f"Best trial number: {best_trial.number}")
    logger.info(f"  Value (Validation Accuracy): {best_trial.value:.4f}")
    logger.info("  Params: ")
    for key, value in best_trial.params.items():
        logger.info(f"    {key}: {value}")

    # Save study results
    try:
        study_df = study.trials_dataframe()
        study_df.to_csv("optuna_study_results.csv", index=False)
        logger.info("Optuna study results saved to optuna_study_results.csv")
    except Exception as e:
        logger.error(f"Could not save Optuna study results: {e}")