import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig, BertTokenizerFast, get_linear_schedule_with_warmup
from typing import Optional, Tuple, List, Dict
# import random # No longer needed for PersonalityDataset
from torch.utils.data import Dataset, DataLoader
import json
import pandas as pd
import optuna
import torch.optim as optim
import os
import logging # Import logging
import gc

# --- Setup Logger ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- CORAL-style Loss Function (Helper - UNCHANGED) ---
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


# --- PersonalityModelV2 (Modified to remove attention) ---
class PersonalityModelV2(nn.Module):
    def __init__(self,
                 bert_model_name: str,
                 num_traits: int,
                 ordinal_values_per_trait: int,
                 n_comments_to_process: int = 3,
                 dropout_rate: float = 0.2,
                 # attention_hidden_dim: int = 128, # REMOVED
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
        # REMOVED Attention Layers
        # self.attention_w = nn.Linear(bert_hidden_size, attention_hidden_dim)
        # self.attention_v = nn.Linear(attention_hidden_dim, 1, bias=False)
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
                comment_active_mask: torch.Tensor): # comment_active_mask (batch_size, n_comments_to_process)
        batch_size = input_ids.shape[0]
        input_ids_flat = input_ids.view(-1, input_ids.shape[-1])
        attention_mask_flat = attention_mask.view(-1, attention_mask.shape[-1])
        outputs = self.bert(input_ids=input_ids_flat, attention_mask=attention_mask_flat)
        comment_embeddings_flat = self._pool_bert_layers(outputs.hidden_states, attention_mask_flat)
        # comment_embeddings shape: (batch_size, n_comments_to_process, bert_hidden_size)
        comment_embeddings = comment_embeddings_flat.view(batch_size, self.n_comments_to_process, -1)

        # --- MODIFIED: Replace Attention with Mean Pooling over active comments ---
        if comment_active_mask is not None:
            # Expand comment_active_mask to be [batch_size, n_comments_to_process, 1] for broadcasting
            comment_active_mask_expanded = comment_active_mask.unsqueeze(-1).float()
            # Zero out embeddings of inactive/padded comments
            masked_comment_embeddings = comment_embeddings * comment_active_mask_expanded
            # Sum embeddings for active comments
            summed_comment_embeddings = torch.sum(masked_comment_embeddings, dim=1) # Shape: (batch_size, bert_hidden_size)
            # Count active comments per sample, ensure it's at least 1 to avoid div by zero
            num_active_comments = comment_active_mask.sum(dim=1, keepdim=True).float().clamp(min=1.0) # Shape: (batch_size, 1)
            aggregated_comment_embedding = summed_comment_embeddings / num_active_comments
        else:
            # Fallback if no mask provided (though it should be): average all comment embeddings
            logger.warning("comment_active_mask is None in PersonalityModelV2 forward pass. Averaging all comment embeddings.")
            aggregated_comment_embedding = torch.mean(comment_embeddings, dim=1)
        # --- END MODIFICATION ---

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
            logger.error("predict_classes called with num_binary_classifiers_per_trait=0 but ordinal_values_per_trait > 1. This is an inconsistent state.")
            num_traits_from_model = len(self.trait_classifiers)
            return torch.zeros(batch_size, num_traits_from_model, dtype=torch.long, device=logits.device)

        num_total_binary_outputs = logits.shape[1]
        num_traits = num_total_binary_outputs // self.num_binary_classifiers_per_trait
        logits_reshaped = logits.view(batch_size, num_traits, self.num_binary_classifiers_per_trait)
        probs_greater_than_k = torch.sigmoid(logits_reshaped)
        predicted_classes = (probs_greater_than_k > 0.5).sum(dim=2)
        return predicted_classes


# --- PersonalityDataset (Modified to remove random sampling) ---
class PersonalityDataset(Dataset):
    def __init__(self,
                 data: List[Dict],
                 tokenizer: BertTokenizerFast,
                 max_seq_length: int,
                 trait_names: List[str],
                 ordinal_values_per_trait: int,
                 num_comments_to_process: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.trait_names = trait_names
        self.ordinal_values_per_trait = ordinal_values_per_trait
        self.num_comments_to_process = num_comments_to_process
        self.num_traits = len(trait_names)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        user_comments_all = sample['comments']

        # --- MODIFIED: Remove random sampling, take first N comments ---
        # Take the first self.num_comments_to_process comments
        # If fewer than self.num_comments_to_process are available, all of them will be taken.
        # The loop below will handle padding up to self.num_comments_to_process.
        comments_to_use = user_comments_all[:self.num_comments_to_process]
        # --- END MODIFICATION ---

        processed_comments_input_ids = []
        processed_comments_attention_mask = []
        active_comment_flags = []

        # num_actual_comments is the number of comments we actually took from the user's list
        num_actual_comments = len(comments_to_use)

        for i in range(self.num_comments_to_process): # This loop iterates N times (N = num_comments_to_process)
            if i < num_actual_comments: # If we have a real comment for this slot
                comment_text = comments_to_use[i]
                active_comment_flags.append(True)
            else: # Otherwise, this slot is for padding
                comment_text = self.tokenizer.pad_token # Use pad token for empty/padding slots
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


# --- Optuna Objective Function (Modified for model changes) ---
def objective(trial: optuna.trial.Trial,
              full_train_data: List[Dict],
              full_val_data: List[Dict],
              tokenizer: BertTokenizerFast,
              global_config: Dict,
              device: torch.device,
              num_epochs: int = 10):

    logger.info(f"Starting Optuna Trial {trial.number}")

    num_traits = len(global_config['TRAIT_NAMES'])
    # ordinal_values_per_trait = global_config['ORDINAL_VALUES_PER_TRAIT'] # Already available in model from global_config
    # bert_model_name = global_config['BERT_MODEL_NAME'] # Already available in model from global_config

    # --- Suggest Hyperparameters ---
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    # REMOVED: attention_hidden_dim = trial.suggest_categorical("attention_hidden_dim", [64, 128, 256])
    lr_bert = trial.suggest_float("lr_bert", 1e-6, 5e-4, log=True)
    lr_head = trial.suggest_float("lr_head", 5e-5, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    num_bert_layers_to_pool = trial.suggest_int("num_bert_layers_to_pool", 1, 4)
    #n_comments_trial = trial.suggest_int("n_comments_to_process", 3, 15)
    num_unfrozen_bert_layers = trial.suggest_int("num_unfrozen_bert_layers", 0, 12)
    patience_early_stopping = trial.suggest_int("patience_early_stopping", 3, 6)
    scheduler_type = trial.suggest_categorical("scheduler_type", ["none", "linear_warmup"])
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.15) if scheduler_type != "none" else 0.0
    batch_size_trial = trial.suggest_categorical("batch_size", [8, 16])

    logger.info(f"Trial {trial.number} - Suggested Parameters: {trial.params}")

    train_dataset_trial = PersonalityDataset(
        data=full_train_data, tokenizer=tokenizer,
        max_seq_length=global_config['SEQ_LEN'],
        trait_names=global_config['TRAIT_NAMES'],
        ordinal_values_per_trait=global_config['ORDINAL_VALUES_PER_TRAIT'],
        num_comments_to_process=3#n_comments_trial
    )
    val_dataset_trial = PersonalityDataset(
        data=full_val_data, tokenizer=tokenizer,
        max_seq_length=global_config['SEQ_LEN'],
        trait_names=global_config['TRAIT_NAMES'],
        ordinal_values_per_trait=global_config['ORDINAL_VALUES_PER_TRAIT'],
        num_comments_to_process=3#n_comments_trial
    )
    train_loader_trial = DataLoader(train_dataset_trial, batch_size=batch_size_trial, shuffle=True)
    val_loader_trial = DataLoader(val_dataset_trial, batch_size=batch_size_trial, shuffle=False)

    model = PersonalityModelV2(
        bert_model_name=global_config['BERT_MODEL_NAME'],
        num_traits=len(global_config['TRAIT_NAMES']),
        ordinal_values_per_trait=global_config['ORDINAL_VALUES_PER_TRAIT'],
        n_comments_to_process=3#n_comments_trial,
        dropout_rate=dropout_rate,
        # attention_hidden_dim=attention_hidden_dim, # REMOVED
        num_bert_layers_to_pool=num_bert_layers_to_pool
    ).to(device)

    for param in model.bert.parameters():
        param.requires_grad = False
    if num_unfrozen_bert_layers > 0:
        for param in model.bert.embeddings.parameters():
            param.requires_grad = True
        for i in range(model.bert.config.num_hidden_layers - num_unfrozen_bert_layers, model.bert.config.num_hidden_layers):
            for param in model.bert.encoder.layer[i].parameters():
                param.requires_grad = True
        if hasattr(model.bert, 'pooler') and model.bert.pooler is not None:
            for param in model.bert.pooler.parameters():
                param.requires_grad = True
    logger.debug(f"Trial {trial.number} - BERT parameters requiring grad: "
                 f"{sum(p.numel() for p in model.bert.parameters() if p.requires_grad)}")

    optimizer_grouped_parameters = []
    bert_params_to_tune = [p for p in model.bert.parameters() if p.requires_grad]
    if bert_params_to_tune and lr_bert > 0:
         optimizer_grouped_parameters.append({"params": bert_params_to_tune, "lr": lr_bert})
    
    # REMOVED Attention parameters from optimizer
    # optimizer_grouped_parameters.extend([
    #     {"params": model.attention_w.parameters(), "lr": lr_head},
    #     {"params": model.attention_v.parameters(), "lr": lr_head},
    # ])
    for classifier_head in model.trait_classifiers:
        optimizer_grouped_parameters.append({"params": classifier_head.parameters(), "lr": lr_head})
    
    if not optimizer_grouped_parameters:
        logger.warning(f"Trial {trial.number} - No parameters to optimize. Skipping training.")
        # If lr_head is 0 and BERT is frozen, this could happen.
        # Ensure at least classifier heads are trained or handle this case.
        # For now, assuming lr_head > 0 or BERT unfrozen layers exist.
        if not any(pg['params'] for pg in optimizer_grouped_parameters if pg.get('params')): # Check if any actual params
            logger.error(f"Trial {trial.number} - Optimizer has no parameters. This trial will likely fail or do nothing.")
            return 0.0 # Return a poor score

    optimizer = optim.AdamW(optimizer_grouped_parameters, weight_decay=weight_decay)

    if scheduler_type == "linear_warmup":
        num_training_steps = len(train_loader_trial) * num_epochs
        num_warmup_steps = int(num_training_steps * warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    elif scheduler_type == "none":
        scheduler = None

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
                input_ids.shape[0],
                num_traits,
                model.num_binary_classifiers_per_trait
            )
            current_batch_loss = 0
            num_actual_traits_from_logits = logits_reshaped.size(1)
            for i in range(num_actual_traits_from_logits): # Use num_actual_traits from logits
                trait_logits = logits_reshaped[:, i, :]
                trait_labels_ordinal = labels_ord[:, i]
                current_batch_loss += coral_style_loss_calculation(
                    trait_logits, trait_labels_ordinal, model.ordinal_values_per_trait, device
                )
            if num_actual_traits_from_logits > 0:
                final_loss = current_batch_loss / num_actual_traits_from_logits
                if final_loss.requires_grad: # Ensure loss requires grad before backward
                    final_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    if scheduler and scheduler_type == "linear_warmup": # check scheduler exists
                        scheduler.step()
                total_train_loss += final_loss.item()
            # If num_actual_traits_from_logits is 0, no loss is added (total_train_loss remains unchanged for this batch)
        
        avg_train_loss = total_train_loss / len(train_loader_trial) if len(train_loader_trial) > 0 else 0.0
        logger.info(f"Trial {trial.number}, Epoch {epoch+1}/{num_epochs} completed. Avg Train Loss: {avg_train_loss:.4f}")

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
                
                logits_reshaped = all_logits.view(
                    input_ids.shape[0],
                    num_traits, # Use num_traits from optuna config, should match model's output structure
                    model.num_binary_classifiers_per_trait
                )
                batch_val_loss = 0
                # num_actual_traits_val should be consistent with num_traits
                for i in range(num_traits): 
                    trait_logits = logits_reshaped[:, i, :]
                    trait_labels_ordinal = labels_ord[:, i]
                    batch_val_loss += coral_style_loss_calculation(
                        trait_logits, trait_labels_ordinal, model.ordinal_values_per_trait, device
                    )
                if num_traits > 0:
                    final_val_loss_batch = batch_val_loss / num_traits
                    current_epoch_val_loss += final_val_loss_batch.item()
                
                predicted_classes_batch = model.predict_classes(all_logits)
                current_epoch_all_val_preds.append(predicted_classes_batch.cpu())
                current_epoch_all_val_labels.append(labels_ord.cpu())
        
        avg_val_loss_epoch = current_epoch_val_loss / len(val_loader_trial) if len(val_loader_trial) > 0 else 0.0
        current_epoch_val_accuracy = 0.0
        if current_epoch_all_val_labels: # Check if list is not empty
            all_val_labels_cat_epoch = torch.cat(current_epoch_all_val_labels, dim=0)
            if all_val_labels_cat_epoch.numel() > 0: # Check if tensor is not empty
                all_val_preds_cat_epoch = torch.cat(current_epoch_all_val_preds, dim=0)
                correct_predictions_epoch = (all_val_preds_cat_epoch == all_val_labels_cat_epoch).float().sum()
                total_predictions_epoch = all_val_labels_cat_epoch.numel()
                if total_predictions_epoch > 0: # Avoid division by zero
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
            # Ensure resources are released before returning from a pruned trial
            del model, optimizer, train_loader_trial, val_loader_trial, train_dataset_trial, val_dataset_trial
            if 'scheduler' in locals(): del scheduler
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            return best_trial_val_accuracy # Optuna expects a float value
        
        if patience_counter >= patience_early_stopping:
            logger.info(f"Trial {trial.number} - Early stopping at epoch {epoch+1} (Patience: {patience_early_stopping}).")
            break
            
    logger.info(f"Trial {trial.number} finished. Best Val Accuracy for this trial: {best_trial_val_accuracy:.4f}")
    # Ensure resources are released at the end of a successful trial too
    del model, optimizer, train_loader_trial, val_loader_trial, train_dataset_trial, val_dataset_trial
    if 'scheduler' in locals(): del scheduler # Check if scheduler was defined
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return best_trial_val_accuracy


# --- Helper Functions (UNCHANGED) ---
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

def transform_train_label(value, trait_name="<unknown>", sample_id="<unknown>"):
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
        author_id = author.get('id', '<unknown_id>')
        labels = author.get('labels', {})
        new_labels = {}
        trait_keys = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism', 'humility']
        for key in trait_keys:
            new_labels[key] = transform_train_label(labels.get(key, 0.0), trait_name=key, sample_id=author_id)
        author['labels'] = new_labels
        if 'comments' in author and isinstance(author['comments'], list):
            author['comments'] = [str(c) if c is not None else "" for c in author['comments']]
        else:
            logger.warning(f"Comments missing or malformed for sample '{author_id}'. Using default empty comments.")
            author['comments'] = ["", "", ""] # Default to 3 empty strings if comments are missing/malformed
        transformed_data.append(author)
    logger.info(f"Finished transforming training data. {len(transformed_data)} samples processed.")
    return transformed_data


# --- Main Script (UNCHANGED except for possibly removing unused imports if any) ---
if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {DEVICE}")

    GLOBAL_CONFIG = {
        'BERT_MODEL_NAME': "bert-base-uncased",
        'TRAIT_NAMES': ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism', 'humility'],
        'SEQ_LEN': 128,
        'ORDINAL_VALUES_PER_TRAIT': 3,
        # 'BATCH_SIZE': 16, # Optuna will tune this now
    }

    NUM_EPOCHS_PER_TRIAL = 25
    N_OPTUNA_TRIALS = 30

    base_data_path = os.path.join("..", "shared task","data") # Consider making this configurable or ensuring path exists
    train_data_path = os.path.join(base_data_path, "best_comments_only.json")
    val_data_path = os.path.join(base_data_path, "val_data.csv")

    # Check if paths are absolute or relative and exist
    if not os.path.isabs(train_data_path): train_data_path = os.path.abspath(train_data_path)
    if not os.path.isabs(val_data_path): val_data_path = os.path.abspath(val_data_path)

    logger.info(f"Attempting to load training data from: {train_data_path}")
    logger.info(f"Attempting to load validation data from: {val_data_path}")

    full_train_data = transform_train_data(train_data_path)
    full_val_data = test_data_transform(val_data_path)

    tokenizer = BertTokenizerFast.from_pretrained(GLOBAL_CONFIG['BERT_MODEL_NAME'])
    logger.info("Tokenizer loaded.")

    logger.info(f"Full train dataset size: {len(full_train_data)}, Full val dataset size: {len(full_val_data)}")
    if len(full_train_data) == 0 or len(full_val_data) == 0:
        logger.error("One of the datasets is empty. Please check data loading and paths.")
        exit() # Exit if data loading fails

    logger.info(f"Starting Optuna study: {N_OPTUNA_TRIALS} trials, up to {NUM_EPOCHS_PER_TRIAL} epochs/trial (with early stopping).")
    
    study_name = 'personality_bert_no_attention_v1' # Changed study name to reflect modification
    db_name = f"{study_name}.db"
    study = optuna.create_study(study_name=study_name,
                                direction="maximize",
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=3, n_min_trials=2), # Added n_min_trials
                                storage=f"sqlite:///{db_name}",
                                load_if_exists=True)
    if study.trials:
        logger.info(f"Resuming existing study {study.study_name} with {len(study.trials)} trials.")
    
    try:
        study.optimize(
            lambda trial: objective(
                trial,
                full_train_data,
                full_val_data,
                tokenizer,
                GLOBAL_CONFIG,
                DEVICE,
                num_epochs=NUM_EPOCHS_PER_TRIAL
            ),
            n_trials=N_OPTUNA_TRIALS,
            gc_after_trial=True
        )
    except Exception as e:
        logger.exception("An error occurred during the Optuna study.")
        # Potentially save study before raising, or ensure Optuna's SQLite storage handles this
        raise

    logger.info("\n--- Optuna Study Finished ---")
    logger.info(f"Study Name: {study.study_name}")
    logger.info(f"Number of finished trials: {len(study.trials)}")
    
    if study.best_trial: # Check if a best trial exists (e.g., if all trials pruned or errored)
        best_trial = study.best_trial
        logger.info(f"Best trial number: {best_trial.number}")
        logger.info(f"  Value (Validation Accuracy): {best_trial.value:.4f}")
        logger.info("  Params: ")
        for key, value in best_trial.params.items():
            logger.info(f"    {key}: {value}")
    else:
        logger.warning("No best trial found. The study may have been interrupted or all trials failed/pruned early.")

    try:
        study_df = study.trials_dataframe()
        results_csv_path = f"optuna_study_results_{study_name}.csv"
        study_df.to_csv(results_csv_path, index=False)
        logger.info(f"Optuna study results saved to {results_csv_path}")
    except Exception as e:
        logger.error(f"Could not save Optuna study results: {e}")