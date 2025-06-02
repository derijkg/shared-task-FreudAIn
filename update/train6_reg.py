import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig, BertTokenizerFast, get_linear_schedule_with_warmup
from typing import Optional, Tuple, List, Dict, Union # Added Union
import random
from torch.utils.data import Dataset, DataLoader
# import json # Not needed for loading data directly in main if already loaded
# import pandas as pd # Not needed for loading data directly in main if already loaded
import optuna
import torch.optim as optim
import os
import logging
import gc
import numpy as np # For np.mean in evaluation
import json
import pandas as pd

# --- Setup Logger ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Regression Loss Function (NEW) ---
# We'll use nn.MSELoss directly in the training loop.

# --- PersonalityModelV3 (Regression and q_scores integration) ---
class PersonalityModelV3(nn.Module):
    def __init__(self,
                 bert_model_name: str,
                 num_traits: int,
                 n_comments_to_process: int = 3,
                 dropout_rate: float = 0.2,
                 attention_hidden_dim: int = 128,
                 num_bert_layers_to_pool: int = 4,
                 num_q_features_per_comment: int = 3, # For Q1, Q2, Q3 scores per comment
                 num_other_numerical_features: int = 0, # From sample['features'] excluding q_scores
                 numerical_embedding_dim: int = 64
                ):
        super().__init__()
        self.bert_config = BertConfig.from_pretrained(bert_model_name, output_hidden_states=True)
        self.bert = BertModel.from_pretrained(bert_model_name, config=self.bert_config)
        self.n_comments_to_process = n_comments_to_process
        self.num_bert_layers_to_pool = num_bert_layers_to_pool
        bert_hidden_size = self.bert.config.hidden_size
        self.num_q_features_per_comment = num_q_features_per_comment

        # Comment processing part (BERT embedding + q_scores)
        comment_feature_dim = bert_hidden_size + self.num_q_features_per_comment
        self.attention_w = nn.Linear(comment_feature_dim, attention_hidden_dim)
        self.attention_v = nn.Linear(attention_hidden_dim, 1, bias=False)
        
        self.dropout = nn.Dropout(dropout_rate)

        # Other numerical features processing part (from sample['features'])
        self.num_other_numerical_features = num_other_numerical_features
        self.uses_other_numerical_features = self.num_other_numerical_features > 0
        self.other_numerical_processor_output_dim = 0

        # Dimension of aggregated comment features (output of attention over comment_feature_dim)
        aggregated_comment_feature_dim = comment_feature_dim 
        combined_input_dim_for_heads = aggregated_comment_feature_dim

        if self.uses_other_numerical_features:
            self.other_numerical_processor_output_dim = numerical_embedding_dim
            self.other_numerical_processor = nn.Sequential(
                nn.Linear(self.num_other_numerical_features, self.other_numerical_processor_output_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
            combined_input_dim_for_heads += self.other_numerical_processor_output_dim
            logger.info(f"Model will use {self.num_other_numerical_features} other numerical features, processed to dim {self.other_numerical_processor_output_dim}.")
        else:
            logger.info("Model will NOT use other numerical features.")

        # Trait regression heads
        self.trait_regressors = nn.ModuleList()
        for _ in range(num_traits):
            self.trait_regressors.append(
                nn.Linear(combined_input_dim_for_heads, 1) # Output one value per trait
            )

    def _pool_bert_layers(self, all_hidden_states: Tuple[torch.Tensor, ...], attention_mask: torch.Tensor) -> torch.Tensor:
        # Assuming all_hidden_states contains embeddings for all layers
        # The last 'num_bert_layers_to_pool' layers are averaged.
        # Or, more commonly, take the [CLS] token embedding from the last few layers or just the last layer.
        # Your current pooling averages token embeddings for selected layers. Let's keep it for now.
        
        layers_to_pool = all_hidden_states[-self.num_bert_layers_to_pool:]
        pooled_outputs = []
        expanded_attention_mask = attention_mask.unsqueeze(-1).expand_as(layers_to_pool[0]) # (batch*n_comments, seq_len, hidden_size)
        
        for layer_hidden_states in layers_to_pool:
            # Masked average pooling
            sum_embeddings = torch.sum(layer_hidden_states * expanded_attention_mask, dim=1) # (batch*n_comments, hidden_size)
            sum_mask = expanded_attention_mask.sum(dim=1) # (batch*n_comments, hidden_size)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            pooled_outputs.append(sum_embeddings / sum_mask) # Element-wise division
            
        stacked_pooled_outputs = torch.stack(pooled_outputs, dim=0) # (num_pool_layers, batch*n_comments, hidden_size)
        mean_pooled_layers_embedding = torch.mean(stacked_pooled_outputs, dim=0) # (batch*n_comments, hidden_size)
        return mean_pooled_layers_embedding


    def forward(self,
                input_ids: torch.Tensor,      # (batch_size, n_comments, seq_len)
                attention_mask: torch.Tensor, # (batch_size, n_comments, seq_len)
                q_scores: torch.Tensor,       # (batch_size, n_comments, num_q_features)
                comment_active_mask: torch.Tensor, # (batch_size, n_comments)
                other_numerical_features: Optional[torch.Tensor] = None # (batch_size, num_other_num_features)
               ):
        batch_size = input_ids.shape[0]
        
        # Flatten for BERT: (batch_size * n_comments, seq_len)
        input_ids_flat = input_ids.view(-1, input_ids.shape[-1])
        attention_mask_flat = attention_mask.view(-1, attention_mask.shape[-1])
        
        bert_outputs = self.bert(input_ids=input_ids_flat, attention_mask=attention_mask_flat)
        # bert_last_hidden_state = bert_outputs.last_hidden_state # (batch*n_comments, seq_len, bert_hidden_size)
        # Pooled BERT embeddings for each comment
        # comment_bert_embeddings_flat = bert_last_hidden_state[:, 0, :] # Using [CLS] token
        comment_bert_embeddings_flat = self._pool_bert_layers(bert_outputs.hidden_states, attention_mask_flat)


        # Reshape back to (batch_size, n_comments, bert_hidden_size)
        comment_bert_embeddings = comment_bert_embeddings_flat.view(batch_size, self.n_comments_to_process, -1)
        
        # Concatenate q_scores with BERT embeddings for each comment
        # q_scores is (batch_size, n_comments, num_q_features)
        comment_features_with_q = torch.cat((comment_bert_embeddings, q_scores), dim=2)
        
        # Attention over combined comment features
        # comment_features_with_q shape: (batch_size, n_comments, bert_hidden_size + num_q_features)
        u = torch.tanh(self.attention_w(comment_features_with_q)) # (batch_size, n_comments, attention_hidden_dim)
        scores = self.attention_v(u).squeeze(-1) # (batch_size, n_comments)
        
        if comment_active_mask is not None:
            scores = scores.masked_fill(~comment_active_mask, -1e9) # Apply mask before softmax
            
        attention_weights = F.softmax(scores, dim=1) # (batch_size, n_comments)
        attention_weights_expanded = attention_weights.unsqueeze(-1) # (batch_size, n_comments, 1)
        
        # Weighted sum of comment_features_with_q
        aggregated_comment_features = torch.sum(attention_weights_expanded * comment_features_with_q, dim=1)
        # aggregated_comment_features shape: (batch_size, bert_hidden_size + num_q_features)

        final_features_for_heads = aggregated_comment_features
        if self.uses_other_numerical_features:
            if other_numerical_features is None or other_numerical_features.shape[1] != self.num_other_numerical_features:
                raise ValueError(
                    f"Other numerical features expected but not provided correctly. "
                    f"Expected {self.num_other_numerical_features}, got shape {other_numerical_features.shape if other_numerical_features is not None else 'None'}"
                )
            processed_other_numerical_features = self.other_numerical_processor(other_numerical_features)
            final_features_for_heads = torch.cat((aggregated_comment_features, processed_other_numerical_features), dim=1)
        
        combined_features_dropped = self.dropout(final_features_for_heads)
        
        trait_regression_outputs = []
        for regressor_head in self.trait_regressors:
            trait_regression_outputs.append(regressor_head(combined_features_dropped))
        
        # Concatenate outputs for all traits: (batch_size, num_traits)
        all_trait_outputs_raw = torch.cat(trait_regression_outputs, dim=1)
        
        # Apply sigmoid to constrain output to [0, 1] for regression
        all_trait_outputs_sigmoid = torch.sigmoid(all_trait_outputs_raw)
        
        return all_trait_outputs_sigmoid

    def predict_scores(self, outputs: torch.Tensor) -> torch.Tensor:
        # The forward pass already returns the sigmoid-activated scores
        return outputs


# --- PersonalityDatasetV3 (Consumes pre-processed data) ---
class PersonalityDatasetV3(Dataset):
    def __init__(self,
                 data: List[Dict],
                 trait_names: List[str],    # Order of traits for labels
                 n_comments_to_process: int,
                 # For numerical features from sample['features']
                 other_numerical_feature_names: List[str],
                 # For q_scores from sample['features']['q_scores']
                 num_q_features_per_comment: int = 3,
                 is_test_set: bool = False
                ):
        self.data = data
        self.trait_names_ordered = trait_names # e.g., ['Openness', 'Conscientiousness', ...]
        self.n_comments_to_process = n_comments_to_process
        self.other_numerical_feature_names = other_numerical_feature_names
        self.num_q_features_per_comment = num_q_features_per_comment
        self.is_test_set = is_test_set

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # 1. Pre-tokenized comments (input_ids, attention_mask)
        # These are expected to be already padded/truncated to seq_len
        # and stacked into (num_actual_comments, seq_len) tensors
        tokenized_info = sample['features']['comments_tokenized']
        
        # These are already tensors as per your sample data.
        # If they were lists, you'd use torch.tensor() here.
        all_input_ids = tokenized_info['input_ids']         # (actual_num_comments, seq_len)
        all_attention_mask = tokenized_info['attention_mask'] # (actual_num_comments, seq_len)
        # all_token_type_ids = tokenized_info['token_type_ids'] # Usually not needed for single-text tasks

        num_actual_comments = all_input_ids.shape[0]
        
        # Pad/truncate the list of comments to n_comments_to_process
        final_input_ids = torch.zeros((self.n_comments_to_process, all_input_ids.shape[1]), dtype=torch.long)
        final_attention_mask = torch.zeros((self.n_comments_to_process, all_attention_mask.shape[1]), dtype=torch.long)
        comment_active_flags = torch.zeros(self.n_comments_to_process, dtype=torch.bool)

        # Select comments (randomly if more, pad if fewer)
        indices_to_select = list(range(num_actual_comments))
        if num_actual_comments > self.n_comments_to_process:
            indices_to_select = random.sample(indices_to_select, self.n_comments_to_process)
            comments_to_fill = self.n_comments_to_process
        else:
            comments_to_fill = num_actual_comments
        
        for i in range(comments_to_fill):
            original_idx = indices_to_select[i]
            final_input_ids[i] = all_input_ids[original_idx]
            final_attention_mask[i] = all_attention_mask[original_idx]
            comment_active_flags[i] = True

        # 2. Q-Scores (per comment)
        # sample['features']['q_scores'] is List[List[float]]
        raw_q_scores = sample['features'].get('q_scores', []) # List of lists
        final_q_scores = torch.zeros((self.n_comments_to_process, self.num_q_features_per_comment), dtype=torch.float)
        
        num_actual_q_score_sets = len(raw_q_scores)
        q_scores_to_fill = min(num_actual_q_score_sets, self.n_comments_to_process)

        for i in range(q_scores_to_fill):
            # If we sampled comments, we need to pick the q_scores for the *selected* comments.
            # This assumes the order in raw_q_scores corresponds to the order in tokenized_info.
            original_idx = indices_to_select[i] if i < len(indices_to_select) else i
            if original_idx < num_actual_q_score_sets:
                 final_q_scores[i] = torch.tensor(raw_q_scores[original_idx][:self.num_q_features_per_comment], dtype=torch.float)
        
        # 3. Other Numerical Features (user-level, from sample['features'])
        other_numerical_features_list = []
        for fname in self.other_numerical_feature_names:
            val = sample['features'].get(fname, 0.0) # Default to 0.0 if missing
            try:
                other_numerical_features_list.append(float(val))
            except (ValueError, TypeError):
                logger.warning(f"Could not convert numerical feature '{fname}' (value: {val}) to float for sample id {sample.get('id', idx)}. Using 0.0.")
                other_numerical_features_list.append(0.0)
        other_numerical_features_tensor = torch.tensor(other_numerical_features_list, dtype=torch.float)

        # 4. Labels (for regression, 0-1 range)
        if not self.is_test_set:
            labels_dict = sample['labels'] # e.g., {'Openness': 1.0, ...}
            # Ensure labels are in the order defined by self.trait_names_ordered
            regression_labels = []
            for trait_key in self.trait_names_ordered: # Use the Pythonic keys here
                # Convert trait_key (e.g. 'openness') to the key in sample['labels'] (e.g. 'Openness')
                # Assuming sample['labels'] uses title case like 'Openness'
                label_val = labels_dict.get(trait_key.title(), labels_dict.get(trait_key, 0.0))
                try:
                    label_float = float(label_val)
                    if not (0.0 <= label_float <= 1.0):
                        # logger.warning(f"Label for {trait_key} ({label_float}) out of [0,1] range for sample {idx}. Clipping.")
                        label_float = np.clip(label_float, 0.0, 1.0)
                    regression_labels.append(label_float)
                except (ValueError, TypeError):
                    logger.error(f"Invalid label value for trait {trait_key}: {label_val}. Using 0.0. Sample ID: {sample.get('id', idx)}")
                    regression_labels.append(0.0)
            labels_tensor = torch.tensor(regression_labels, dtype=torch.float)
            
            return (
                final_input_ids,
                final_attention_mask,
                final_q_scores,
                comment_active_flags,
                other_numerical_features_tensor,
                labels_tensor
            )
        else: # Test set, no labels
            return (
                final_input_ids,
                final_attention_mask,
                final_q_scores,
                comment_active_flags,
                other_numerical_features_tensor
                # No labels_tensor for test set
            )

# --- Optuna Objective Function (MODIFIED for Regression) ---
def objective(trial: optuna.trial.Trial,
              train_data_list: List[Dict], # Direct list of dicts
              val_data_list: List[Dict],   # Direct list of dicts
              global_config: Dict,
              device: torch.device,
              num_epochs_per_trial: int = 10): # Renamed for clarity

    logger.info(f"Starting Optuna Trial {trial.number}")

    num_traits = len(global_config['TRAIT_NAMES'])
    other_numerical_feature_names_trial = global_config.get('OTHER_NUMERICAL_FEATURE_NAMES', [])
    num_other_numerical_features_trial = len(other_numerical_feature_names_trial)
    num_q_features_per_comment_trial = global_config.get('NUM_Q_FEATURES_PER_COMMENT', 3)

    # --- Suggest Hyperparameters ---
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.4) # Adjusted range
    attention_hidden_dim = trial.suggest_categorical("attention_hidden_dim", [128, 256, 512]) # Larger options
    lr_bert = trial.suggest_float("lr_bert", 5e-6, 1e-4, log=True) # Adjusted range
    lr_head = trial.suggest_float("lr_head", 1e-4, 1e-2, log=True) # Adjusted range
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True) # Adjusted range
    num_bert_layers_to_pool = trial.suggest_int("num_bert_layers_to_pool", 1, 4)
    n_comments_trial = trial.suggest_int("n_comments_to_process", 1, global_config.get('MAX_COMMENTS_TO_PROCESS_PHYSICAL', 3)) # Max based on data
    num_unfrozen_bert_layers = trial.suggest_int("num_unfrozen_bert_layers", 0, 6) # Fewer unfrozen layers often better
    patience_early_stopping = trial.suggest_int("patience_early_stopping", 3, 5)
    scheduler_type = trial.suggest_categorical("scheduler_type", ["none", "linear_warmup"])
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.05, 0.2) if scheduler_type != "none" else 0.0
    batch_size_trial = trial.suggest_categorical("batch_size", [8, 16]) # Kept smaller due to BERT

    other_numerical_embedding_dim_trial = 0
    if num_other_numerical_features_trial > 0:
        other_numerical_embedding_dim_trial = trial.suggest_categorical("other_numerical_embedding_dim", [32, 64])

    logger.info(f"Trial {trial.number} - Suggested Parameters: {trial.params}")

    train_dataset_trial = PersonalityDatasetV3(
        data=train_data_list,
        trait_names=global_config['TRAIT_NAMES_ORDERED'], # Use ordered list
        n_comments_to_process=n_comments_trial,
        other_numerical_feature_names=other_numerical_feature_names_trial,
        num_q_features_per_comment=num_q_features_per_comment_trial,
        is_test_set=False
    )
    val_dataset_trial = PersonalityDatasetV3(
        data=val_data_list,
        trait_names=global_config['TRAIT_NAMES_ORDERED'], # Use ordered list
        n_comments_to_process=n_comments_trial,
        other_numerical_feature_names=other_numerical_feature_names_trial,
        num_q_features_per_comment=num_q_features_per_comment_trial,
        is_test_set=False
    )
    train_loader_trial = DataLoader(train_dataset_trial, batch_size=batch_size_trial, shuffle=True, num_workers=2, pin_memory=True)
    val_loader_trial = DataLoader(val_dataset_trial, batch_size=batch_size_trial, shuffle=False, num_workers=2, pin_memory=True)

    model = PersonalityModelV3(
        bert_model_name=global_config['BERT_MODEL_NAME'],
        num_traits=num_traits,
        n_comments_to_process=n_comments_trial,
        dropout_rate=dropout_rate,
        attention_hidden_dim=attention_hidden_dim,
        num_bert_layers_to_pool=num_bert_layers_to_pool,
        num_q_features_per_comment=num_q_features_per_comment_trial,
        num_other_numerical_features=num_other_numerical_features_trial,
        numerical_embedding_dim=other_numerical_embedding_dim_trial
    ).to(device)

    # BERT Layer Freezing
    for name, param in model.bert.named_parameters(): param.requires_grad = False # Freeze all initially
    if num_unfrozen_bert_layers > 0:
        if hasattr(model.bert, 'embeddings'):
            for param in model.bert.embeddings.parameters(): param.requires_grad = True
        
        actual_layers_to_unfreeze = min(num_unfrozen_bert_layers, model.bert.config.num_hidden_layers)
        for i in range(model.bert.config.num_hidden_layers - actual_layers_to_unfreeze, model.bert.config.num_hidden_layers):
            if i >= 0:
                for param in model.bert.encoder.layer[i].parameters(): param.requires_grad = True
        
        if hasattr(model.bert, 'pooler') and model.bert.pooler is not None: # Though pooler is often not used for seq classification
            for param in model.bert.pooler.parameters(): param.requires_grad = True
    
    logger.debug(f"Trial {trial.number} - BERT params requiring grad: "
                 f"{sum(p.numel() for p in model.bert.parameters() if p.requires_grad)}")

    # Optimizer Setup
    optimizer_grouped_parameters = []
    bert_params_to_tune = [p for p in model.bert.parameters() if p.requires_grad]
    if bert_params_to_tune and lr_bert > 0:
         optimizer_grouped_parameters.append({"params": bert_params_to_tune, "lr": lr_bert, "weight_decay": 0.01}) # Different WD for BERT

    head_params = list(model.attention_w.parameters()) + list(model.attention_v.parameters())
    for regressor_head in model.trait_regressors:
        head_params.extend(list(regressor_head.parameters()))
    if model.uses_other_numerical_features:
        head_params.extend(list(model.other_numerical_processor.parameters()))
    
    optimizer_grouped_parameters.append({"params": head_params, "lr": lr_head, "weight_decay": weight_decay}) # Main WD for head
        
    if not any(pg['params'] for pg in optimizer_grouped_parameters if pg['params']): # Check if any group has params
        logger.warning(f"Trial {trial.number} - No parameters to optimize. Skipping training.")
        return float('inf') # Return high loss for minimization

    optimizer = optim.AdamW(optimizer_grouped_parameters) # WD applied per group
    
    scheduler = None
    if scheduler_type == "linear_warmup":
        num_training_steps = len(train_loader_trial) * num_epochs_per_trial
        num_warmup_steps = int(num_training_steps * warmup_ratio)
        if num_warmup_steps > 0 : # Ensure warmup steps > 0 if scheduler is used
             scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    # Regression loss
    loss_fn = nn.MSELoss().to(device) # Or nn.L1Loss()
    best_trial_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs_per_trial):
        model.train()
        total_train_loss = 0
        for batch_idx, batch_tuple in enumerate(train_loader_trial):
            input_ids, attention_m, q_s, comment_active_m, other_num_feats, labels_reg = [b.to(device) for b in batch_tuple]
            
            optimizer.zero_grad()
            predicted_scores = model(input_ids, attention_m, q_s, comment_active_m, other_num_feats)
            
            current_batch_loss = loss_fn(predicted_scores, labels_reg)
            
            if torch.isnan(current_batch_loss) or torch.isinf(current_batch_loss):
                logger.warning(f"Trial {trial.number}, Epoch {epoch+1}, Batch {batch_idx}: NaN or Inf loss detected. Skipping batch.")
                torch.cuda.empty_cache()
                continue

            current_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler: scheduler.step()
            total_train_loss += current_batch_loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader_trial) if len(train_loader_trial) > 0 else float('inf')
        logger.info(f"Trial {trial.number}, Epoch {epoch+1}/{num_epochs_per_trial} completed. Avg Train Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        current_epoch_val_loss = 0
        all_val_preds_epoch = []
        all_val_labels_epoch = []
        with torch.no_grad():
            for batch_tuple in val_loader_trial:
                input_ids, attention_m, q_s, comment_active_m, other_num_feats, labels_reg = [b.to(device) for b in batch_tuple]
                if input_ids.numel() == 0: continue
                
                predicted_scores = model(input_ids, attention_m, q_s, comment_active_m, other_num_feats)
                if predicted_scores.numel() == 0: continue
                
                batch_val_loss = loss_fn(predicted_scores, labels_reg)
                current_epoch_val_loss += batch_val_loss.item()
                all_val_preds_epoch.append(predicted_scores.cpu())
                all_val_labels_epoch.append(labels_reg.cpu())

        avg_val_loss_epoch = current_epoch_val_loss / len(val_loader_trial) if len(val_loader_trial) > 0 else float('inf')
        
        # Calculate MAE for logging (optional, but good for interpretability)
        val_mae = -1.0
        if all_val_labels_epoch:
            all_val_labels_cat = torch.cat(all_val_labels_epoch, dim=0)
            all_val_preds_cat = torch.cat(all_val_preds_epoch, dim=0)
            if all_val_labels_cat.numel() > 0:
                val_mae = F.l1_loss(all_val_preds_cat, all_val_labels_cat).item() # MAE

        logger.info(f"Trial {trial.number}, Epoch {epoch+1} Val Loss (MSE): {avg_val_loss_epoch:.4f}, Val MAE: {val_mae:.4f}")

        if avg_val_loss_epoch < best_trial_val_loss:
            best_trial_val_loss = avg_val_loss_epoch
            patience_counter = 0
            logger.debug(f"Trial {trial.number}, Epoch {epoch+1}: New best val_loss: {best_trial_val_loss:.4f}")
            # Could save best model for this trial here if needed
        else:
            patience_counter += 1
        
        trial.report(avg_val_loss_epoch, epoch) # Report validation loss to Optuna
        if trial.should_prune():
            logger.info(f"Trial {trial.number} pruned by Optuna at epoch {epoch+1}.")
            del model, train_loader_trial, val_loader_trial, optimizer, scheduler
            torch.cuda.empty_cache(); gc.collect()
            return best_trial_val_loss # Return the best loss achieved so far for this pruned trial
        
        if patience_counter >= patience_early_stopping:
            logger.info(f"Trial {trial.number} - Early stopping at epoch {epoch+1} (Patience: {patience_early_stopping}).")
            break
    
    logger.info(f"Trial {trial.number} finished. Best Val Loss (MSE) for this trial: {best_trial_val_loss:.4f}")
    del model, train_loader_trial, val_loader_trial, optimizer, scheduler
    torch.cuda.empty_cache(); gc.collect()
    return best_trial_val_loss


# --- Main Script Execution ---
if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {DEVICE}")

    # --- Assume train_data, val_data, test_data are already loaded as lists of dicts ---
    with open("train_data.json", 'r') as f:
        train_data = json.load(f) 
    with open("val_data.json", 'r') as f:
        val_data = json.load(f)
    with open("test_data.json", 'r') as f:
        test_data = json.load(f) # test_data will not have 'labels' key
    

    _trait_names_ordered_config = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Emotional stability', 'Humility']
    _other_numerical_features_config = [
        'mean_words_per_comment', 'median_words_per_comment', 'mean_sents_per_comment',
        'median_sents_per_comment', 'mean_words_per_sentence', 'median_words_per_sentence',
        'sents_per_comment_skew', 'words_per_sentence_skew', 'total_double_whitespace',
        'punc_em_total', 'punc_qm_total', 'punc_period_total', 'punc_comma_total',
        'punc_colon_total', 'punc_semicolon_total', 'flesch_reading_ease_agg',
        'gunning_fog_agg', 'mean_word_len_overall', 'ttr_overall',
        'mean_sentiment_neg', 'mean_sentiment_neu', 'mean_sentiment_pos',
        'mean_sentiment_compound', 'std_sentiment_compound'
    ]



    SEQ_LEN_CONFIG = 512


    # --- Global Configuration ---
    GLOBAL_CONFIG = {
        'BERT_MODEL_NAME': "bert-base-uncased", # Your original
        'TRAIT_NAMES_ORDERED': _trait_names_ordered_config, # Pythonic keys for consistency internally
        'TRAIT_NAMES': _trait_names_ordered_config, # For num_traits consistency; will use TRAIT_NAMES_ORDERED for actual label mapping
        'MAX_COMMENTS_TO_PROCESS_PHYSICAL': 3, # Max comments physically present in any sample for n_comments_trial suggestion
        'NUM_Q_FEATURES_PER_COMMENT': 3,
        'OTHER_NUMERICAL_FEATURE_NAMES': _other_numerical_features_config,
        # SEQ_LEN is implicitly handled by pre-tokenized data; PersonalityDataset doesn't need it.
        # PAD_TOKEN_ID also not needed by PersonalityDataset as padding is assumed done.
    }

    NUM_EPOCHS_PER_TRIAL_OPTUNA = 15 # Short epochs for mock runs
    N_OPTUNA_TRIALS = 20             # Few trials for mock runs

    if not train_data or not val_data:
        logger.error("Train or Validation data is empty. Please load your data.")
        exit()

    logger.info(f"Starting Optuna study: {N_OPTUNA_TRIALS} trials, up to {NUM_EPOCHS_PER_TRIAL_OPTUNA} epochs/trial.")
    
    study_name = "personality_regression_v1"
    storage_name = f"sqlite:///{study_name}.db"
    # For Optuna, we want to MINIMIZE the loss (e.g., MSE)
    study = optuna.create_study(study_name=study_name,
                                direction="minimize", # MINIMIZE for loss
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=1, n_min_trials=1), # Adjusted for few epochs/trials
                                storage=storage_name,
                                load_if_exists=True)
    if study.trials: logger.info(f"Resuming existing study {study.study_name} with {len(study.trials)} previous trials.")
    
    try:
        study.optimize(
            lambda trial: objective(
                trial, train_data, val_data,
                GLOBAL_CONFIG, DEVICE, num_epochs_per_trial=NUM_EPOCHS_PER_TRIAL_OPTUNA
            ),
            n_trials=N_OPTUNA_TRIALS,
            gc_after_trial=True
        )
    except Exception as e:
        logger.exception("An error occurred during the Optuna study.")

    logger.info("\n--- Optuna Study Finished ---")
    logger.info(f"Number of finished trials: {len(study.trials)}")
    
    if not study.trials:
        logger.warning("No trials were completed in the study.")
    else:
        try:
            # Ensure there's at least one completed (not failed/pruned early) trial
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if completed_trials:
                best_trial = study.best_trial
                logger.info(f"Best trial number: {best_trial.number}")
                logger.info(f"  Value (Validation Loss - MSE): {best_trial.value:.4f}") # This is now loss
                logger.info("  Params: ")
                for key, value in best_trial.params.items():
                    logger.info(f"    {key}: {value}")
            else:
                logger.warning("No trials completed successfully to determine the best trial.")

            study_df = study.trials_dataframe()
            study_df.to_csv(f"{study_name}_results.csv", index=False)
            logger.info(f"Optuna study results saved to {study_name}_results.csv")
        except Exception as e:
            logger.error(f"Could not process or save Optuna study results: {e}")

    # Example of how to use the model for prediction on test_data (after finding best hyperparameters)
    # You would typically load the best model weights here.
    # For now, let's just show how to run prediction with a newly initialized model (not trained properly here)
    if test_data and completed_trials: # Check if there are completed trials to get params
        logger.info("\n--- Example: Predicting on Test Data (using best trial's HPs if available) ---")
        best_params = study.best_trial.params if completed_trials else {} # Use best HPs or defaults if none
        
        test_dataset = PersonalityDatasetV3(
            data=test_data,
            trait_names=GLOBAL_CONFIG['TRAIT_NAMES_ORDERED'],
            n_comments_to_process=best_params.get("n_comments_to_process", 3),
            other_numerical_feature_names=GLOBAL_CONFIG['OTHER_NUMERICAL_FEATURE_NAMES'],
            num_q_features_per_comment=GLOBAL_CONFIG['NUM_Q_FEATURES_PER_COMMENT'],
            is_test_set=True
        )
        test_loader = DataLoader(test_dataset, batch_size=best_params.get("batch_size", 8), shuffle=False)

        # Initialize model with best HPs (or defaults)
        # NOTE: In a real scenario, you would load saved model weights from the best trial
        test_model = PersonalityModelV3(
            bert_model_name=GLOBAL_CONFIG['BERT_MODEL_NAME'],
            num_traits=len(GLOBAL_CONFIG['TRAIT_NAMES']),
            n_comments_to_process=best_params.get("n_comments_to_process", 3),
            dropout_rate=best_params.get("dropout_rate", 0.2),
            attention_hidden_dim=best_params.get("attention_hidden_dim", 128),
            num_bert_layers_to_pool=best_params.get("num_bert_layers_to_pool", 2),
            num_q_features_per_comment=GLOBAL_CONFIG['NUM_Q_FEATURES_PER_COMMENT'],
            num_other_numerical_features=len(GLOBAL_CONFIG['OTHER_NUMERICAL_FEATURE_NAMES']),
            numerical_embedding_dim=best_params.get("other_numerical_embedding_dim", 64) if GLOBAL_CONFIG['OTHER_NUMERICAL_FEATURE_NAMES'] else 0
        ).to(DEVICE)
        test_model.eval()

        all_test_predictions = []
        with torch.no_grad():
            for batch_tuple in test_loader:
                # Unpack assuming test_loader yields items for is_test_set=True
                input_ids, attention_m, q_s, comment_active_m, other_num_feats = [b.to(DEVICE) for b in batch_tuple]
                predicted_scores = test_model(input_ids, attention_m, q_s, comment_active_m, other_num_feats)
                all_test_predictions.append(predicted_scores.cpu().numpy())
        
        if all_test_predictions:
            final_test_predictions = np.concatenate(all_test_predictions, axis=0)
            logger.info(f"Shape of test predictions: {final_test_predictions.shape}") # (num_test_samples, num_traits)
            # You can now save these predictions or process them further
            # Example: Print first 5 predictions
            for i in range(min(5, len(final_test_predictions))):
                 pred_dict = {trait: round(score.item(), 4) for trait, score in zip(GLOBAL_CONFIG['TRAIT_NAMES_ORDERED'], final_test_predictions[i])}
                 logger.info(f"Test Sample {test_data[i]['id']} Predictions: {pred_dict}")