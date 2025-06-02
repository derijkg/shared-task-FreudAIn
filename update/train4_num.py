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
import logging
import gc

# --- Setup Logger (Unchanged) ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- CORAL-style Loss Function (Unchanged) ---
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


# --- PersonalityModelV2 (MODIFIED) ---
class PersonalityModelV2(nn.Module):
    def __init__(self,
                 bert_model_name: str,
                 num_traits: int,
                 ordinal_values_per_trait: int,
                 n_comments_to_process: int = 3,
                 dropout_rate: float = 0.2,
                 attention_hidden_dim: int = 128,
                 num_bert_layers_to_pool: int = 4,
                 num_numerical_features: int = 0, # New
                 numerical_embedding_dim: int = 64 # New: Dim for processed numerical features
                ):
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

        # Text processing part
        self.attention_w = nn.Linear(bert_hidden_size, attention_hidden_dim)
        self.attention_v = nn.Linear(attention_hidden_dim, 1, bias=False)
        
        self.dropout = nn.Dropout(dropout_rate)

        # Numerical features processing part
        self.num_numerical_features = num_numerical_features
        self.uses_numerical_features = self.num_numerical_features > 0
        self.numerical_processor_output_dim = 0

        combined_input_dim = bert_hidden_size # Start with BERT output size

        if self.uses_numerical_features:
            self.numerical_processor_output_dim = numerical_embedding_dim
            self.numerical_processor = nn.Sequential(
                nn.Linear(self.num_numerical_features, self.numerical_processor_output_dim),
                nn.ReLU(), # Common activation
                nn.Dropout(dropout_rate) # Can use the same or a different dropout
            )
            combined_input_dim += self.numerical_processor_output_dim
            logger.info(f"Model will use {self.num_numerical_features} numerical features, processed to dim {self.numerical_processor_output_dim}.")
        else:
            logger.info("Model will NOT use numerical features.")

        # Trait classifiers based on combined input dimension
        self.trait_classifiers = nn.ModuleList()
        for _ in range(num_traits):
            self.trait_classifiers.append(
                nn.Linear(combined_input_dim, self.num_binary_classifiers_per_trait)
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
                comment_active_mask: torch.Tensor,
                numerical_features: Optional[torch.Tensor] = None): # New
        batch_size = input_ids.shape[0]
        input_ids_flat = input_ids.view(-1, input_ids.shape[-1])
        attention_mask_flat = attention_mask.view(-1, attention_mask.shape[-1])
        
        outputs = self.bert(input_ids=input_ids_flat, attention_mask=attention_mask_flat)
        comment_embeddings_flat = self._pool_bert_layers(outputs.hidden_states, attention_mask_flat)
        comment_embeddings = comment_embeddings_flat.view(batch_size, self.n_comments_to_process, -1)
        
        # Attention over comment embeddings
        u = torch.tanh(self.attention_w(comment_embeddings))
        scores = self.attention_v(u).squeeze(-1)
        if comment_active_mask is not None:
            scores = scores.masked_fill(~comment_active_mask, -1e9)
        attention_weights = F.softmax(scores, dim=1)
        attention_weights_expanded = attention_weights.unsqueeze(-1)
        aggregated_comment_embedding = torch.sum(attention_weights_expanded * comment_embeddings, dim=1)

        # Combine with numerical features
        final_features_to_classify = aggregated_comment_embedding
        if self.uses_numerical_features:
            if numerical_features is None or numerical_features.shape[1] != self.num_numerical_features:
                raise ValueError(
                    f"Numerical features are expected but not provided correctly. "
                    f"Expected {self.num_numerical_features} features, got shape {numerical_features.shape if numerical_features is not None else 'None'}"
                )
            processed_numerical_features = self.numerical_processor(numerical_features)
            final_features_to_classify = torch.cat((aggregated_comment_embedding, processed_numerical_features), dim=1)
        
        combined_features_dropped = self.dropout(final_features_to_classify)
        
        trait_specific_logits = []
        for classifier_head in self.trait_classifiers:
            trait_specific_logits.append(classifier_head(combined_features_dropped))
        all_logits = torch.cat(trait_specific_logits, dim=1)
        return all_logits

    def predict_classes(self, logits: torch.Tensor) -> torch.Tensor: # Unchanged
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



# --- PersonalityDataset (MODIFIED for Pre-tokenized Input) ---
class PersonalityDataset(Dataset):
    def __init__(self,
                 data: List[Dict],
                 # tokenizer: BertTokenizerFast, # REMOVED
                 max_seq_length: int,           # Still needed for creating padding tensors
                 pad_token_id: int,             # New: To create padding tensors
                 trait_names: List[str],
                 ordinal_values_per_trait: int,
                 num_comments_to_process: int,
                 numerical_feature_names: Optional[List[str]] = None):
        self.data = data
        # self.tokenizer = tokenizer # REMOVED
        self.max_seq_length = max_seq_length
        self.pad_token_id = pad_token_id
        self.trait_names = trait_names
        self.ordinal_values_per_trait = ordinal_values_per_trait
        self.num_comments_to_process = num_comments_to_process
        self.num_traits = len(trait_names)
        self.numerical_feature_names = numerical_feature_names if numerical_feature_names else []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Retrieve pre-tokenized comments (e.g., from sample['comments_tokenized'])
        # This list contains dicts: {'input_ids': [...], 'attention_mask': [...]}
        user_tokenized_comments_all = sample.get('comments_tokenized', []) # Ensure this key matches your preprocessed data
        
        if len(user_tokenized_comments_all) > self.num_comments_to_process:
            comments_to_use_or_pad_info = random.sample(user_tokenized_comments_all, self.num_comments_to_process)
        else:
            comments_to_use_or_pad_info = user_tokenized_comments_all
        
        processed_comments_input_ids = []
        processed_comments_attention_mask = []
        active_comment_flags = []
        num_actual_comments = len(comments_to_use_or_pad_info)

        for i in range(self.num_comments_to_process):
            if i < num_actual_comments:
                comment_info = comments_to_use_or_pad_info[i]
                # Ensure they are tensors; if they are lists from JSON, convert them
                input_ids = torch.tensor(comment_info['input_ids'], dtype=torch.long)
                attention_mask = torch.tensor(comment_info['attention_mask'], dtype=torch.long)
                
                # Double check length, though pre-tokenization should handle this
                if input_ids.shape[0] != self.max_seq_length:
                    # This should ideally not happen if pre-tokenization was correct
                    logger.warning(f"Sample {idx}, comment {i}: input_ids length mismatch. Expected {self.max_seq_length}, got {input_ids.shape[0]}. Check pre-tokenization.")
                    # Quick fix: pad or truncate again (less ideal)
                    if input_ids.shape[0] > self.max_seq_length:
                        input_ids = input_ids[:self.max_seq_length]
                        attention_mask = attention_mask[:self.max_seq_length]
                    else:
                        padding_len = self.max_seq_length - input_ids.shape[0]
                        input_ids = F.pad(input_ids, (0, padding_len), value=self.pad_token_id)
                        attention_mask = F.pad(attention_mask, (0, padding_len), value=0)

                processed_comments_input_ids.append(input_ids)
                processed_comments_attention_mask.append(attention_mask)
                active_comment_flags.append(True)
            else: # Create padding comment tensors
                pad_input_ids = torch.full((self.max_seq_length,), self.pad_token_id, dtype=torch.long)
                pad_attention_mask = torch.zeros(self.max_seq_length, dtype=torch.long)
                processed_comments_input_ids.append(pad_input_ids)
                processed_comments_attention_mask.append(pad_attention_mask)
                active_comment_flags.append(False)
        
        input_ids_tensor = torch.stack(processed_comments_input_ids)
        attention_mask_tensor = torch.stack(processed_comments_attention_mask)
        comment_active_mask_tensor = torch.tensor(active_comment_flags, dtype=torch.bool)
        
        # Label processing (same as before)
        integer_labels = []
        for trait_name in self.trait_names:
            label = sample['labels'].get(trait_name) # Use .get() for safety
            if label is None:
                 logger.warning(f"Label for trait {trait_name} missing in sample {sample.get('id', idx)}. Using 0.")
                 label = 0
            elif not isinstance(label, int):
                try: label = int(label)
                except ValueError:
                    raise ValueError(f"Label for trait {trait_name} in sample {sample.get('id', idx)} is not int and cannot be cast: {label} ({type(label)})")
            if not (0 <= label < self.ordinal_values_per_trait):
                raise ValueError(f"Label {label} for trait {trait_name} in sample {sample.get('id', idx)} out of range [0, {self.ordinal_values_per_trait-1}]")
            integer_labels.append(label)
        labels_tensor = torch.tensor(integer_labels, dtype=torch.long)

        # Numerical features processing (same as before)
        numerical_features_list = []
        if self.numerical_feature_names:
            for fname in self.numerical_feature_names:
                val = sample.get(fname)
                if val is None:
                    # logger.warning(f"Numerical feature '{fname}' missing for sample id {sample.get('id', idx)}. Using 0.0.")
                    val = 0.0
                try:
                    numerical_features_list.append(float(val))
                except (ValueError, TypeError):
                    # logger.warning(f"Could not convert numerical feature '{fname}' (value: {val}) to float for sample id {sample.get('id', idx)}. Using 0.0.")
                    numerical_features_list.append(0.0)
        numerical_features_tensor = torch.tensor(numerical_features_list, dtype=torch.float)

        return (
            input_ids_tensor,
            attention_mask_tensor,
            comment_active_mask_tensor,
            numerical_features_tensor,
            labels_tensor
        )

# --- Optuna Objective Function (MODIFIED) ---
def objective(trial: optuna.trial.Trial,
              full_train_data: List[Dict],
              full_val_data: List[Dict],
              tokenizer: BertTokenizerFast,
              global_config: Dict,
              device: torch.device,
              num_epochs: int = 10):

    logger.info(f"Starting Optuna Trial {trial.number}")

    num_traits = len(global_config['TRAIT_NAMES'])
    ordinal_values_per_trait = global_config['ORDINAL_VALUES_PER_TRAIT']
    bert_model_name = global_config['BERT_MODEL_NAME']
    
    # Determine num_numerical_features from global_config
    numerical_feature_names_trial = global_config.get('NUMERICAL_FEATURE_NAMES', [])
    num_numerical_features_trial = len(numerical_feature_names_trial)

    # --- Suggest Hyperparameters ---
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    attention_hidden_dim = trial.suggest_categorical("attention_hidden_dim", [64, 128, 256])
    lr_bert = trial.suggest_float("lr_bert", 1e-6, 5e-4, log=True)
    lr_head = trial.suggest_float("lr_head", 5e-5, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    num_bert_layers_to_pool = trial.suggest_int("num_bert_layers_to_pool", 1, 4)
    n_comments_trial = trial.suggest_int("n_comments_to_process", 3, 15)
    num_unfrozen_bert_layers = trial.suggest_int("num_unfrozen_bert_layers", 0, 12) # 0 means freeze all BERT
    patience_early_stopping = trial.suggest_int("patience_early_stopping", 3, 6) # Increased upper bound
    scheduler_type = trial.suggest_categorical("scheduler_type", ["none", "linear_warmup"])
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.15) if scheduler_type != "none" else 0.0
    batch_size_trial = trial.suggest_categorical("batch_size", [8, 16, 32]) # Added 32

    numerical_embedding_dim_trial = 0
    if num_numerical_features_trial > 0:
        numerical_embedding_dim_trial = trial.suggest_categorical("numerical_embedding_dim", [32, 64, 128])


    logger.info(f"Trial {trial.number} - Suggested Parameters: {trial.params}")
    logger.info(f"Trial {trial.number} - Using {num_numerical_features_trial} numerical features.")


    train_dataset_trial = PersonalityDataset(
        data=full_train_data, tokenizer=tokenizer,
        max_seq_length=global_config['SEQ_LEN'],
        trait_names=global_config['TRAIT_NAMES'],
        ordinal_values_per_trait=global_config['ORDINAL_VALUES_PER_TRAIT'],
        num_comments_to_process=n_comments_trial,
        numerical_feature_names=numerical_feature_names_trial # New
    )
    val_dataset_trial = PersonalityDataset(
        data=full_val_data, tokenizer=tokenizer,
        max_seq_length=global_config['SEQ_LEN'],
        trait_names=global_config['TRAIT_NAMES'],
        ordinal_values_per_trait=global_config['ORDINAL_VALUES_PER_TRAIT'],
        num_comments_to_process=n_comments_trial,
        numerical_feature_names=numerical_feature_names_trial # New
    )
    train_loader_trial = DataLoader(train_dataset_trial, batch_size=batch_size_trial, shuffle=True)
    val_loader_trial = DataLoader(val_dataset_trial, batch_size=batch_size_trial, shuffle=False)

    model = PersonalityModelV2(
        bert_model_name=global_config['BERT_MODEL_NAME'],
        num_traits=num_traits,
        ordinal_values_per_trait=ordinal_values_per_trait,
        n_comments_to_process=n_comments_trial,
        dropout_rate=dropout_rate,
        attention_hidden_dim=attention_hidden_dim,
        num_bert_layers_to_pool=num_bert_layers_to_pool,
        num_numerical_features=num_numerical_features_trial, # New
        numerical_embedding_dim=numerical_embedding_dim_trial # New
    ).to(device)

    # BERT Layer Freezing (Unchanged from your version)
    for param in model.bert.parameters(): param.requires_grad = False
    if num_unfrozen_bert_layers > 0:
        if hasattr(model.bert, 'embeddings'): # Ensure embeddings exist
            for param in model.bert.embeddings.parameters(): param.requires_grad = True
        
        # Unfreeze transformer layers from the top
        # Corrected loop to ensure we don't go out of bounds if num_unfrozen > num_actual_layers
        actual_layers_to_unfreeze = min(num_unfrozen_bert_layers, model.bert.config.num_hidden_layers)
        for i in range(model.bert.config.num_hidden_layers - actual_layers_to_unfreeze, model.bert.config.num_hidden_layers):
            if i >= 0 : # Check layer index is valid
                 for param in model.bert.encoder.layer[i].parameters(): param.requires_grad = True
        
        if hasattr(model.bert, 'pooler') and model.bert.pooler is not None:
            for param in model.bert.pooler.parameters(): param.requires_grad = True
    
    logger.debug(f"Trial {trial.number} - BERT parameters requiring grad: "
                 f"{sum(p.numel() for p in model.bert.parameters() if p.requires_grad)}")


    # Optimizer Setup
    optimizer_grouped_parameters = []
    bert_params_to_tune = [p for p in model.bert.parameters() if p.requires_grad]
    if bert_params_to_tune and lr_bert > 0:
         optimizer_grouped_parameters.append({"params": bert_params_to_tune, "lr": lr_bert})
    
    head_params = list(model.attention_w.parameters()) + list(model.attention_v.parameters())
    for classifier_head in model.trait_classifiers:
        head_params.extend(list(classifier_head.parameters()))
    if model.uses_numerical_features: # Add numerical processor params if they exist
        head_params.extend(list(model.numerical_processor.parameters()))
    
    optimizer_grouped_parameters.append({"params": head_params, "lr": lr_head})
        
    if not optimizer_grouped_parameters or not any(group['params'] for group in optimizer_grouped_parameters):
        logger.warning(f"Trial {trial.number} - No parameters to optimize. Skipping training.")
        return 0.0

    optimizer = optim.AdamW(optimizer_grouped_parameters, weight_decay=weight_decay)
    
    # Scheduler (Unchanged from your version)
    if scheduler_type == "linear_warmup":
        num_training_steps = len(train_loader_trial) * num_epochs
        num_warmup_steps = int(num_training_steps * warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    else: # "none" or any other case
        scheduler = None

    best_trial_val_accuracy = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for batch_idx, batch_tuple in enumerate(train_loader_trial):
            # Unpack batch tuple correctly (numerical_features is new)
            input_ids, attention_m, comment_active_m, numerical_feats, labels_ord = [b.to(device) for b in batch_tuple]
            
            optimizer.zero_grad()
            # Pass numerical_feats to the model
            all_logits = model(input_ids, attention_m, comment_active_m, numerical_feats) 
            
            # Loss calculation (remains largely the same logic)
            logits_reshaped = all_logits.view(
                input_ids.shape[0], num_traits, model.num_binary_classifiers_per_trait
            )
            current_batch_loss = 0
            for i in range(num_traits): # Assuming num_traits is correct here
                trait_logits = logits_reshaped[:, i, :]
                trait_labels_ordinal = labels_ord[:, i]
                current_batch_loss += coral_style_loss_calculation(
                    trait_logits, trait_labels_ordinal, model.ordinal_values_per_trait, device
                )
            if num_traits > 0:
                final_loss = current_batch_loss / num_traits
                if final_loss.requires_grad:
                    final_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    if scheduler: scheduler.step()
                total_train_loss += final_loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader_trial) if len(train_loader_trial) > 0 else 0.0
        logger.info(f"Trial {trial.number}, Epoch {epoch+1}/{num_epochs} completed. Avg Train Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        current_epoch_val_loss = 0
        current_epoch_all_val_preds = []
        current_epoch_all_val_labels = []
        with torch.no_grad():
            for batch_tuple in val_loader_trial:
                input_ids, attention_m, comment_active_m, numerical_feats, labels_ord = [b.to(device) for b in batch_tuple]
                if input_ids.numel() == 0: continue
                
                all_logits = model(input_ids, attention_m, comment_active_m, numerical_feats) # Pass numerical_feats
                if all_logits.numel() == 0: continue
                
                logits_reshaped = all_logits.view(
                    input_ids.shape[0], num_traits, model.num_binary_classifiers_per_trait
                )
                batch_val_loss = 0
                for i in range(num_traits):
                    trait_logits = logits_reshaped[:, i, :]
                    trait_labels_ordinal = labels_ord[:, i]
                    batch_val_loss += coral_style_loss_calculation(
                        trait_logits, trait_labels_ordinal, model.ordinal_values_per_trait, device
                    )
                if num_traits > 0:
                    current_epoch_val_loss += (batch_val_loss / num_traits).item()
                
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
                if total_predictions_epoch > 0:
                    current_epoch_val_accuracy = correct_predictions_epoch / total_predictions_epoch
        
        logger.info(f"Trial {trial.number}, Epoch {epoch+1} Val Loss: {avg_val_loss_epoch:.4f}, Val Accuracy: {current_epoch_val_accuracy:.4f}")

        if current_epoch_val_accuracy > best_trial_val_accuracy:
            best_trial_val_accuracy = current_epoch_val_accuracy
            patience_counter = 0
            logger.debug(f"Trial {trial.number}, Epoch {epoch+1}: New best val_accuracy: {best_trial_val_accuracy:.4f}")
        else:
            patience_counter += 1
        
        trial.report(current_epoch_val_accuracy, epoch)
        if trial.should_prune():
            logger.info(f"Trial {trial.number} pruned by Optuna at epoch {epoch+1}.")
            # Clean up model and data loaders to free GPU memory
            del model, train_loader_trial, val_loader_trial, optimizer
            if scheduler: del scheduler
            torch.cuda.empty_cache()
            gc.collect()
            return best_trial_val_accuracy 
        
        if patience_counter >= patience_early_stopping:
            logger.info(f"Trial {trial.number} - Early stopping at epoch {epoch+1} (Patience: {patience_early_stopping}).")
            break
    
    logger.info(f"Trial {trial.number} finished. Best Val Accuracy for this trial: {best_trial_val_accuracy:.4f}")
    # Clean up model and data loaders to free GPU memory
    del model, train_loader_trial, val_loader_trial, optimizer
    if scheduler: del scheduler
    torch.cuda.empty_cache()
    gc.collect()
    return best_trial_val_accuracy


# --- Helper Functions (MODIFIED to handle numerical features if they exist) ---
# YOU WILL NEED TO MODIFY THESE to ensure numerical features are loaded into the dicts

def test_data_transform(path, numerical_feature_names: Optional[List[str]] = None):
    logger.info(f"Transforming test data from: {path}")
    if numerical_feature_names is None: numerical_feature_names = []
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        logger.error(f"Validation data file not found: {path}")
        raise

    # Label transformation (same as before)
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
        new_dict = {'id': row['id'], 'comments': comments, 'labels': labels}
        
        # Add numerical features
        for num_feat_name in numerical_feature_names:
            if num_feat_name in row:
                try:
                    new_dict[num_feat_name] = float(row[num_feat_name])
                except (ValueError, TypeError):
                    logger.warning(f"Could not convert numerical feature '{num_feat_name}' (value: {row[num_feat_name]}) to float for val ID {row['id']}. Using 0.0.")
                    new_dict[num_feat_name] = 0.0
            else:
                logger.warning(f"Numerical feature '{num_feat_name}' not found in val data for ID {row['id']}. Using 0.0.")
                new_dict[num_feat_name] = 0.0
        data.append(new_dict)
    logger.info(f"Finished transforming test data. {len(data)} samples processed.")
    return data

def transform_train_label(value, trait_name="<unknown>", sample_id="<unknown>"): # Unchanged
    if not isinstance(value, (int, float, np.number)): # Added np.number for robustness
        logger.warning(f"Non-numeric label value '{value}' ({type(value)}) for trait '{trait_name}' in sample '{sample_id}'. Defaulting to 0.")
        return 0
    if pd.isna(value): # Handle NaN
        logger.warning(f"NaN label value for trait '{trait_name}' in sample '{sample_id}'. Defaulting to 0.")
        return 0
    if value <= 0.33: return 0
    elif value <= 0.66: return 1
    else: return 2

def transform_train_data(path, numerical_feature_names: Optional[List[str]] = None):
    logger.info(f"Transforming training data from: {path}")
    if numerical_feature_names is None: numerical_feature_names = []
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
        author = author_original.copy() # Work on a copy
        author_id = author.get('id', f'<unknown_id_{len(transformed_data)}>') 
        
        # Labels (same as before)
        labels_original = author.get('labels', {})
        new_labels = {}
        trait_keys = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism', 'humility']
        for key in trait_keys:
            new_labels[key] = transform_train_label(labels_original.get(key, 0.0), trait_name=key, sample_id=author_id)
        author['labels'] = new_labels
        
        # Comments (same as before)
        if 'comments' in author and isinstance(author['comments'], list):
            author['comments'] = [str(c) if c is not None else "" for c in author['comments']]
        else:
            logger.warning(f"Comments missing or malformed for sample '{author_id}'. Using default empty comments.")
            author['comments'] = ["", "", ""] # Ensure it's always a list of 3 for consistency if needed

        # Add numerical features
        # IMPORTANT: This assumes numerical features are top-level keys in the JSON objects
        # within raw_train_data. Adjust if they are nested or named differently.
        for num_feat_name in numerical_feature_names:
            val = author.get(num_feat_name) # Get from the current author dict
            if val is None:
                logger.warning(f"Numerical feature '{num_feat_name}' missing for train sample id {author_id}. Using 0.0.")
                author[num_feat_name] = 0.0
            else:
                try:
                    author[num_feat_name] = float(val)
                except (ValueError, TypeError):
                    logger.warning(f"Could not convert numerical feature '{num_feat_name}' (value: {val}) to float for train sample id {author_id}. Using 0.0.")
                    author[num_feat_name] = 0.0
        
        transformed_data.append(author)
    logger.info(f"Finished transforming training data. {len(transformed_data)} samples processed.")
    return transformed_data


# --- Main Script (MODIFIED) ---
if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {DEVICE}")

    # --- Global Configuration ---
    GLOBAL_CONFIG = {
        'BERT_MODEL_NAME': "bert-base-uncased",
        'TRAIT_NAMES': ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism', 'humility'],
        'SEQ_LEN': 512,
        'ORDINAL_VALUES_PER_TRAIT': 3,
        # 'BATCH_SIZE': 16, # Optuna will tune this
        # DEFINE YOUR NUMERICAL FEATURE NAMES HERE (must match keys in your data dicts)
        # Example: If your TextFeatureExtractor creates 'txt_mean_words_per_sentence' etc.
        # AND you've added these to your JSON/CSV data that transform_train/test_data read.
        'NUMERICAL_FEATURE_NAMES': [
            'txt_mean_words_per_comment', 'txt_median_words_per_comment', 'txt_total_words',
            'txt_mean_sents_per_comment', 'txt_median_sents_per_comment', 'txt_total_sents',
            'txt_mean_words_per_sentence', 'txt_median_words_per_sentence',
            'txt_sents_per_comment_skew', 'txt_words_per_sentence_skew',
            'txt_punc_!_total', 'txt_punc_?_total', 'txt_punc_._total', 'txt_punc_,_total', # Example punc
            'txt_flesch_reading_ease_agg', 'txt_gunning_fog_agg',
            'txt_mean_word_len_overall', 'txt_ttr_overall',
            'txt_mean_sentiment_neg', 'txt_mean_sentiment_neu', 'txt_mean_sentiment_pos',
            'txt_mean_sentiment_compound', 'txt_std_sentiment_compound'
            # Add any other numerical column names you want to use
        ]
    }

    NUM_EPOCHS_PER_TRIAL = 25 
    N_OPTUNA_TRIALS = 50 # Increased for more exploration

    base_data_path = "." # Assuming data is in current dir or adjust path
    train_data_file_name = "humility_added_with_txt_features.json" # ASSUME YOU HAVE THIS FILE
    val_data_file_name = "val_data_with_txt_features.csv"       # ASSUME YOU HAVE THIS FILE
    
    # IMPORTANT: You need to first run your TextFeatureExtractor on your original
    # humility_added.json and val_data.csv to create these new files that include
    # the numerical features as columns/keys.

    train_data_path = os.path.join(base_data_path, train_data_file_name)
    val_data_path = os.path.join(base_data_path, val_data_file_name)

    logger.info(f"Attempting to load training data from: {os.path.abspath(train_data_path)}")
    logger.info(f"Attempting to load validation data from: {os.path.abspath(val_data_path)}")

    # Pass numerical feature names to transformation functions
    full_train_data = transform_train_data(train_data_path, GLOBAL_CONFIG['NUMERICAL_FEATURE_NAMES'])
    full_val_data = test_data_transform(val_data_path, GLOBAL_CONFIG['NUMERICAL_FEATURE_NAMES'])

    tokenizer = BertTokenizerFast.from_pretrained(GLOBAL_CONFIG['BERT_MODEL_NAME'])
    logger.info("Tokenizer loaded.")

    logger.info(f"Full train dataset size: {len(full_train_data)}, Full val dataset size: {len(full_val_data)}")
    if len(full_train_data) == 0 or len(full_val_data) == 0:
        logger.error("One of the datasets is empty. Please check data loading and paths.")
        exit()
    
    # Check if a sample contains expected numerical features
    if full_train_data and GLOBAL_CONFIG['NUMERICAL_FEATURE_NAMES']:
        sample_check = full_train_data[0]
        missing_keys_in_sample = [key for key in GLOBAL_CONFIG['NUMERICAL_FEATURE_NAMES'] if key not in sample_check]
        if missing_keys_in_sample:
            logger.warning(f"First training sample is missing these expected numerical features: {missing_keys_in_sample}. Ensure data transformation is correct.")
        else:
            logger.info(f"First training sample contains the expected numerical features. Example value for '{GLOBAL_CONFIG['NUMERICAL_FEATURE_NAMES'][0]}': {sample_check.get(GLOBAL_CONFIG['NUMERICAL_FEATURE_NAMES'][0])}")


    # Optuna Study (largely unchanged from your version, ensure it passes relevant parts of GLOBAL_CONFIG)
    logger.info(f"Starting Optuna study: {N_OPTUNA_TRIALS} trials, up to {NUM_EPOCHS_PER_TRIAL} epochs/trial.")
    
    study_name = "personality_bert_with_numerical_v1"
    storage_name = f"sqlite:///{study_name}.db"
    study = optuna.create_study(study_name=study_name,
                                direction="maximize",
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=5, n_min_trials=3), # Adjusted pruner
                                storage=storage_name,
                                load_if_exists=True)
    if study.trials: logger.info(f"Resuming existing study {study.study_name} with {len(study.trials)} trials.")
    
    try:
        study.optimize(
            lambda trial: objective(
                trial, full_train_data, full_val_data, tokenizer,
                GLOBAL_CONFIG, DEVICE, num_epochs=NUM_EPOCHS_PER_TRIAL
            ),
            n_trials=N_OPTUNA_TRIALS,
            gc_after_trial=True
        )
    except Exception as e:
        logger.exception("An error occurred during the Optuna study.")
        # Consider re-raising or handling more gracefully depending on needs
        # raise 

    logger.info("\n--- Optuna Study Finished ---")
    # ... (rest of your Optuna logging and saving results - unchanged) ...
    logger.info(f"Number of finished trials: {len(study.trials)}")
    
    if not study.trials: # Handle case where no trials completed (e.g. all errored early)
        logger.warning("No trials were completed in the study.")
    else:
        try:
            best_trial = study.best_trial
            logger.info(f"Best trial number: {best_trial.number}")
            logger.info(f"  Value (Validation Accuracy): {best_trial.value:.4f}")
            logger.info("  Params: ")
            for key, value in best_trial.params.items():
                logger.info(f"    {key}: {value}")

            study_df = study.trials_dataframe()
            study_df.to_csv(f"{study_name}_results.csv", index=False)
            logger.info(f"Optuna study results saved to {study_name}_results.csv")
        except Exception as e: # Catch specific exceptions if possible, e.g. if best_trial is None
            logger.error(f"Could not process or save Optuna study results: {e}")