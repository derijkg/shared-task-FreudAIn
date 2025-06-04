import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizerFast, BertModel, get_linear_schedule_with_warmup
import numpy as np
from sklearn.model_selection import train_test_split # Not strictly needed for this script, but kept for completeness if functions are reused
import random
from typing import List, Dict, Optional, Tuple
import copy
import itertools # Not strictly needed for this script
import json
from collections import Counter # Not strictly needed for this script
import math
import pandas as pd # Not strictly needed for this script

# --- Configuration (from your original script) ---
BERT_MODEL_NAME = 'bert-base-uncased'
TRAIT_NAMES = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism', 'humility']
NUM_TRAITS = len(TRAIT_NAMES)
NUM_LEVELS = 3  # low, medium, high
ORDINAL_OUTPUTS_PER_TRAIT = NUM_LEVELS - 1
MAX_SEQ_LENGTH = 128
# NUM_NUMERICAL_FEATURES will be determined from data

# --- 1. Dataset Class (Unchanged from your script) ---
class PersonalityDataset(Dataset):
    def __init__(self,
                 data: List[Dict],
                 tokenizer: BertTokenizerFast,
                 max_seq_length: int,
                 trait_names: List[str],
                 num_comments_to_process: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.trait_names = trait_names
        self.num_comments_to_process = num_comments_to_process

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        user_comments_all = sample['comments']
        numerical_features = sample.get('numerical_features', [])

        # For overfitting, we might want deterministic comment selection,
        # but random sampling during training is fine if the dataset is tiny and epochs are many.
        # Let's stick to random sampling as per original to not change too much behavior.
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
                comment_text = "" # Pad with empty string
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
            label = sample['labels'][trait_name] # Assuming labels exist for overfitting data
            integer_labels.append(label)

        return {
            'input_ids': input_ids_tensor,
            'attention_mask': attention_mask_tensor,
            'comment_active_mask': comment_active_mask_tensor,
            'numerical_features': torch.tensor(numerical_features, dtype=torch.float),
            'labels': torch.tensor(integer_labels, dtype=torch.long)
        }

# --- 2. Model Class (Unchanged from your script) ---
class PersonalityModel(nn.Module):
    def __init__(self,
                 bert_model_name: str,
                 num_traits: int,
                 ordinal_outputs_per_trait: int,
                 num_numerical_features: int = 0,
                 n_comments_to_process: int = 3,
                 dropout_rate: float = 0.2,
                 attention_hidden_dim: int = 128):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.n_comments_to_process = n_comments_to_process
        self.ordinal_outputs_per_trait = ordinal_outputs_per_trait
        self.num_numerical_features = num_numerical_features

        bert_hidden_size = self.bert.config.hidden_size

        self.attention_w = nn.Linear(bert_hidden_size, attention_hidden_dim)
        self.attention_v = nn.Linear(attention_hidden_dim, 1, bias=False)

        self.feature_combiner_input_size = bert_hidden_size + self.num_numerical_features
        self.dropout = nn.Dropout(dropout_rate) # Keeping dropout to overfit the original model structure

        self.trait_classifiers = nn.ModuleList()
        for _ in range(num_traits):
            self.trait_classifiers.append(
                nn.Linear(self.feature_combiner_input_size, ordinal_outputs_per_trait)
            )

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                comment_active_mask: torch.Tensor,
                numerical_features: Optional[torch.Tensor] = None):

        batch_size = input_ids.shape[0]
        input_ids_flat = input_ids.view(-1, input_ids.shape[-1])
        attention_mask_flat = attention_mask.view(-1, attention_mask.shape[-1])

        outputs = self.bert(input_ids=input_ids_flat, attention_mask=attention_mask_flat)
        comment_embeddings_flat = outputs.pooler_output
        comment_embeddings = comment_embeddings_flat.view(batch_size, self.n_comments_to_process, -1)

        u = torch.tanh(self.attention_w(comment_embeddings))
        scores = self.attention_v(u).squeeze(-1)
        if comment_active_mask is not None:
            scores = scores.masked_fill(~comment_active_mask, -1e9) # Ensure mask is boolean
        attention_weights = F.softmax(scores, dim=1)
        attention_weights_expanded = attention_weights.unsqueeze(-1)
        aggregated_comment_embedding = torch.sum(attention_weights_expanded * comment_embeddings, dim=1)

        if numerical_features is not None and numerical_features.numel() > 0 and self.num_numerical_features > 0:
            if numerical_features.shape[1] != self.num_numerical_features:
                 print(f"Warning: numerical_features.shape[1] ({numerical_features.shape[1]}) "
                       f"does not match self.num_numerical_features ({self.num_numerical_features})")
            combined_features = torch.cat((aggregated_comment_embedding, numerical_features), dim=1)
        else:
            combined_features = aggregated_comment_embedding

        combined_features_dropped = self.dropout(combined_features)

        trait_specific_logits = []
        for classifier_head in self.trait_classifiers:
            trait_specific_logits.append(classifier_head(combined_features_dropped))

        all_logits = torch.cat(trait_specific_logits, dim=1)
        return all_logits

# --- 3. CORAL Loss Function (Unchanged from your script) ---
class MultiTaskCORALLoss(nn.Module):
    def __init__(self, num_traits: int, num_levels: int, device: torch.device, trait_importance_weights: Optional[List[float]] = None):
        super().__init__()
        self.num_traits = num_traits
        self.num_levels = num_levels
        self.ordinal_outputs_per_trait = num_levels - 1
        self.device = device

        if trait_importance_weights is not None:
            self.trait_importance_weights = torch.tensor(trait_importance_weights, dtype=torch.float, device=self.device)
            if len(self.trait_importance_weights) != num_traits:
                raise ValueError("Length of trait_importance_weights must match num_traits.")
        else:
            self.trait_importance_weights = None

    def forward(self, all_logits: torch.Tensor, true_labels_int: torch.Tensor) -> torch.Tensor:
        batch_size = all_logits.shape[0]
        total_loss = torch.tensor(0.0, device=self.device)
        logits_per_trait_view = all_logits.view(batch_size, self.num_traits, self.ordinal_outputs_per_trait)

        for i in range(self.num_traits):
            trait_logits = logits_per_trait_view[:, i, :]
            trait_labels_int = true_labels_int[:, i]

            levels_binary_targets = torch.zeros_like(trait_logits, device=self.device)
            for k in range(self.ordinal_outputs_per_trait):
                levels_binary_targets[:, k] = (trait_labels_int > k).float()

            loss_trait = F.binary_cross_entropy_with_logits(
                trait_logits, levels_binary_targets, reduction='mean'
            )

            if self.trait_importance_weights is not None:
                total_loss += loss_trait * self.trait_importance_weights[i]
            else:
                total_loss += loss_trait

        return total_loss / self.num_traits if self.num_traits > 0 else torch.tensor(0.0, device=self.device)

# --- Prediction Conversion (Unchanged from your script) ---
def convert_ordinal_logits_to_predictions(logits: torch.Tensor, num_traits: int, ordinal_outputs_per_trait: int, threshold: float = 0.5) -> torch.Tensor:
    batch_size = logits.shape[0]
    logits_per_trait = logits.view(batch_size, num_traits, ordinal_outputs_per_trait)
    probs = torch.sigmoid(logits_per_trait)
    predictions = (probs > threshold).long().sum(dim=2)
    return predictions

# --- 4. Training Loop (Unchanged from your script) ---
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, verbose=True):
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(data_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        comment_active_mask = batch['comment_active_mask'].to(device)
        numerical_features = batch['numerical_features'].to(device)
        labels_int = batch['labels'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, comment_active_mask, numerical_features)
        loss = loss_fn(logits, labels_int)
        total_loss += loss.item()

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()

        if verbose and (batch_idx + 1) % (len(data_loader) // 2 + 1) == 0 : # Print a few times per epoch
             print(f"  Batch {batch_idx + 1}/{len(data_loader)}, Current Batch Train Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
    # Removed per-epoch summary from here as it's handled in the calling loop.
    return avg_loss

# --- 5. Evaluation Function (evaluate_on_test_set from your script, slightly adapted for clarity) ---
def evaluate_on_overfit_set(model: PersonalityModel,
                            data_loader: DataLoader,
                            loss_fn: MultiTaskCORALLoss,
                            device: torch.device,
                            num_traits: int,
                            ordinal_outputs_per_trait: int,
                            trait_names: List[str]) -> Tuple[Optional[float], Optional[Dict[str, float]], Optional[float], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    model.eval()
    total_loss = 0
    all_predictions_list = []
    all_true_labels_list = []

    if not data_loader or len(data_loader) == 0:
        print("Data_loader is empty. Cannot evaluate.")
        return None, None, None, None

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            comment_active_mask = batch['comment_active_mask'].to(device)
            numerical_features = batch['numerical_features'].to(device)
            labels_int = batch['labels'].to(device)

            logits = model(input_ids, attention_mask, comment_active_mask, numerical_features)
            loss = loss_fn(logits, labels_int)
            total_loss += loss.item()

            predictions_batch = convert_ordinal_logits_to_predictions(
                logits.cpu(), num_traits, ordinal_outputs_per_trait
            )
            all_predictions_list.append(predictions_batch)
            all_true_labels_list.append(labels_int.cpu())

    avg_loss = total_loss / len(data_loader)

    all_predictions_tensor = torch.cat(all_predictions_list, dim=0)
    all_true_labels_tensor = torch.cat(all_true_labels_list, dim=0)

    trait_accuracies = {}
    for i in range(num_traits):
        correct_predictions = (all_predictions_tensor[:, i] == all_true_labels_tensor[:, i]).sum().item()
        total_samples_for_trait = all_true_labels_tensor.shape[0]
        accuracy = correct_predictions / total_samples_for_trait if total_samples_for_trait > 0 else 0
        trait_accuracies[trait_names[i]] = accuracy

    correct_sample_matches = (all_predictions_tensor == all_true_labels_tensor).all(dim=1).sum().item()
    total_samples = all_true_labels_tensor.shape[0]
    overall_exact_match_accuracy = correct_sample_matches / total_samples if total_samples > 0 else 0

    return avg_loss, trait_accuracies, overall_exact_match_accuracy, (all_predictions_tensor, all_true_labels_tensor)


# --- Function to Run Overfitting Experiment ---
def run_overfitting_experiment(
    overfit_data: List[Dict],
    device: torch.device,
    hyperparams: Dict,
    num_numerical_features: int
):
    print("\n--- Starting Overfitting Experiment ---")
    print(f"Using hyperparameters: {hyperparams}")
    print(f"Number of samples for overfitting: {len(overfit_data)}")
    if not overfit_data:
        print("Overfitting data is empty. Aborting.")
        return

    # --- Initialize Tokenizer, Dataset, DataLoader ---
    tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL_NAME)

    overfit_dataset = PersonalityDataset(
        data=overfit_data,
        tokenizer=tokenizer,
        max_seq_length=MAX_SEQ_LENGTH,
        trait_names=TRAIT_NAMES,
        num_comments_to_process=hyperparams['n_comments_to_process']
    )

    # For very small datasets, num_workers=0 is often best.
    overfit_dataloader_train = DataLoader(
        overfit_dataset,
        batch_size=hyperparams['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    # Dataloader for evaluation on the same set (no shuffle)
    overfit_dataloader_eval = DataLoader(
        overfit_dataset,
        batch_size=hyperparams['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # --- Initialize Model, Loss, Optimizer ---
    model = PersonalityModel(
        bert_model_name=BERT_MODEL_NAME,
        num_traits=NUM_TRAITS,
        ordinal_outputs_per_trait=ORDINAL_OUTPUTS_PER_TRAIT,
        num_numerical_features=num_numerical_features,
        n_comments_to_process=hyperparams['n_comments_to_process'],
        dropout_rate=hyperparams['dropout_rate'],
        attention_hidden_dim=hyperparams['attention_hidden_dim']
    ).to(device)

    loss_fn = MultiTaskCORALLoss(
        num_traits=NUM_TRAITS, num_levels=NUM_LEVELS, device=device,
        trait_importance_weights=[1.0] * NUM_TRAITS # Equal weights
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=hyperparams['learning_rate'], eps=1e-8)
    scheduler = None # No scheduler, train for fixed epochs

    print(f"\nTraining for {hyperparams['num_epochs_overfit']} epochs to overfit...")
    for epoch in range(hyperparams['num_epochs_overfit']):
        # verbose=False for train_epoch's batch prints, we'll print epoch summary
        avg_train_loss = train_epoch(
            model, overfit_dataloader_train, loss_fn, optimizer, device, scheduler, verbose=False
        )
        print(f"Overfitting Epoch {epoch + 1}/{hyperparams['num_epochs_overfit']}, Average Train Loss: {avg_train_loss:.6f}")

        # Evaluate on the same training data every few epochs
        if (epoch + 1) % hyperparams.get('eval_every_epochs', 5) == 0 or epoch == hyperparams['num_epochs_overfit'] - 1:
            print(f"\nEvaluating on the overfitting dataset (Epoch {epoch + 1}):")
            eval_loss, trait_accuracies, overall_accuracy, _ = evaluate_on_overfit_set(
                model, overfit_dataloader_eval, loss_fn, device, NUM_TRAITS, ORDINAL_OUTPUTS_PER_TRAIT, TRAIT_NAMES
            )
            if eval_loss is not None:
                print(f"  Overfit Set - Eval Loss: {eval_loss:.6f}")
                print(f"  Overfit Set - Overall Exact Match Accuracy: {overall_accuracy:.4f}")
                if trait_accuracies:
                    for trait, acc in trait_accuracies.items():
                        print(f"    {trait} Accuracy: {acc:.4f}")
                # Stop if perfect accuracy is achieved (or very high)
                if overall_accuracy >= 0.999:
                    print("Model has achieved near-perfect accuracy on the overfit set. Stopping early.")
                    break
            else:
                print("  Evaluation on overfit set failed.")

    print("\n--- Overfitting Experiment Finished ---")
    # You can save the model here if needed:
    # torch.save(model.state_dict(), "overfitted_model.pth")
    # print("Saved overfitted model state_dict to overfitted_model.pth")

# --- Main Execution Block for Overfitting ---
if __name__ == "__main__":
    # --- Overfitting Configuration ---
    PATH_TO_TRAIN_DATA_JSON = r"..\shared task\data\humility_added.json" # IMPORTANT: Update this path
    N_SAMPLES_FOR_OVERFIT = 50  # Number of samples to take for overfitting (e.g., 10-50)
    # Parameters for the overfitting run
    OVERFIT_PARAMS = {
        'learning_rate': 2e-5,        # Can be tuned, standard BERT LR
        'dropout_rate': 0.1,          # From your model's default/grid search
        'batch_size': 4,              # Small batch size for small data
        'attention_hidden_dim': 128,  # From your model's default/grid search
        'n_comments_to_process': 3,   # From your model's default/grid search
        'num_epochs_overfit': 200,    # Number of epochs to train for overfitting
        'eval_every_epochs': 10       # How often to evaluate on the overfit set
    }

    # --- Setup Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Seed for reproducibility (optional, but good practice) ---
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # --- Load Full Training Data ---
    try:
        with open(PATH_TO_TRAIN_DATA_JSON, 'r') as f:
            full_train_data = json.load(f)
        print(f"Successfully loaded {len(full_train_data)} samples from {PATH_TO_TRAIN_DATA_JSON}")
    except FileNotFoundError:
        print(f"Error: Training data file not found at {PATH_TO_TRAIN_DATA_JSON}")
        print("Please ensure the file exists and the path is correct.")
        # Create dummy data for demonstration if file not found
        print("Creating dummy data for demonstration purposes.")
        full_train_data = []
        for i in range(max(20, N_SAMPLES_FOR_OVERFIT)): # Ensure enough dummy data
            full_train_data.append({
                'user_id': f'dummy_user_{i}',
                'comments': [f"This is dummy comment one for user {i}.", f"This is dummy comment two for user {i}.", "Dummy comment three."],
                'labels': {trait: random.randint(0, NUM_LEVELS-1) for trait in TRAIT_NAMES},
                'numerical_features': [random.random() for _ in range(3)] # Assuming 3 numerical features if any
            })
        # To disable numerical features for dummy data, set to:
        # 'numerical_features': []

    if not full_train_data:
        print("No training data available. Exiting.")
        exit()

    # --- Select Small Subset for Overfitting ---
    if len(full_train_data) < N_SAMPLES_FOR_OVERFIT:
        print(f"Warning: Requested {N_SAMPLES_FOR_OVERFIT} samples, but only {len(full_train_data)} available. Using all.")
        overfit_subset = full_train_data
    else:
        # Take the first N samples for simplicity and determinism
        overfit_subset = full_train_data[:N_SAMPLES_FOR_OVERFIT]
    print(f"Selected {len(overfit_subset)} samples for the overfitting experiment.")

    # --- Determine Number of Numerical Features ---
    current_num_numerical_features = 0
    if overfit_subset and overfit_subset[0].get('numerical_features') is not None:
        current_num_numerical_features = len(overfit_subset[0]['numerical_features'])
        print(f"Number of numerical features detected: {current_num_numerical_features}")
    else:
        print("No 'numerical_features' key found or it's empty in the first sample of the subset. Assuming 0 numerical features.")

    # --- Run Overfitting ---
    run_overfitting_experiment(
        overfit_subset,
        device,
        OVERFIT_PARAMS,
        num_numerical_features=current_num_numerical_features
    )

    print("\n--- Overfitting Script Finished ---")