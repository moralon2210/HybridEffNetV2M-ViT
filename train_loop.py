import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import time
import datetime
import gc
import json

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data_utils.MetGBM_data_handler import MetGBMDataset
from utils.system_utils import set_seed
sys.path.append(os.path.dirname(os.getcwd()))

# Models
from models.classification_models import HybridVisionClassification

# Utils
from utils.model_utils import agg_predictions_per_tumor, get_model_preds

# Function to select and return the optimizer based on config
def pick_optimizer(fold_model, config):
    if config.model_params.optimizer == 'SGD':
        return torch.optim.SGD(fold_model.parameters(), lr=config.model_params.learning_rate,
                               momentum=config.model_params.momentum, weight_decay=config.model_params.weight_decay)
    elif config.model_params.optimizer == 'AdamW':
        return torch.optim.AdamW(fold_model.parameters(), lr=config.model_params.learning_rate,
                                 weight_decay=config.model_params.weight_decay)
    elif config.model_params.optimizer == 'RMSprop':
        return torch.optim.RMSprop(fold_model.parameters(), lr=config.model_params.learning_rate, alpha=0.9, eps=1e-7,
                                   momentum=config.model_params.momentum, weight_decay=config.model_params.weight_decay, centered=True)

# Function to initialize and return the model with the specified configuration
def load_model(config):
    fold_model = HybridVisionClassification(config, 1).to(config.device)
    return fold_model

# Function to compute the positive class weight for imbalanced data
def compute_pos_weight(train_target, fold, config):
    train_target_counts = train_target.value_counts()
    config.logger.info(f'Train target for fold {fold + 1}: {train_target_counts.to_dict()}')
    pos_weight = train_target_counts.loc[0] / train_target_counts.loc[1]
    return pos_weight

# Main training loop that runs across multiple folds
def main(folds, df_x_train, df_y_train, config, mode):
    start_time = time.time()
    print('\n')
    config.logger.info(f"Starting training for model {config.model_type_str}")

    # Dictionaries to store predictions and indices for each fold
    folds_oof_mean_preds_dict = {}
    folds_oof_tumor_label_dict = {}
    folds_valid_idx = {}
    folds_train_idx = {}
    folds_best_model = []

    # Loop through each fold
    for fold, (train_idx, valid_idx) in enumerate(folds):
        config.logger.info(f"Starting fold {fold + 1}")

        # Store training and validation indices
        folds_train_idx[fold + 1] = [int(idx) for idx in train_idx]
        folds_valid_idx[fold + 1] = [int(idx) for idx in valid_idx]

        # Variables to track best performance in the current fold
        best_roc = None
        best_model_state_dict = None
        patience_f = config.model_params.patience

        # Initialize model and optimizer
        fold_model = load_model(config)
        #fold_model = nn.DataParallel(fold_model, device_ids=[0, 2])
        optimizer = pick_optimizer(fold_model, config)

        # Learning rate scheduler
        scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='max', patience=config.model_params.scheduler_lr_patience,
                                      verbose=True, factor=config.model_params.scheduler_lr_factor)

        # Prepare training and validation datasets
        train_data = df_x_train.iloc[train_idx].reset_index(drop=True)
        train_target = df_y_train.iloc[train_idx].reset_index(drop=True)
        valid_data = df_x_train.iloc[valid_idx].reset_index(drop=True)
        valid_target = df_y_train.iloc[valid_idx].reset_index(drop=True)

        config.logger.info(f'Valid target for fold {fold + 1}: {valid_target.value_counts().to_dict()}')

        # Define loss function with optional class weights
        if config.model_params.with_class_weights:
            pos_weight = compute_pos_weight(train_target, fold, config)
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
            config.logger.info(f"Positive class weight: {pos_weight}")
        else:
            criterion = nn.BCEWithLogitsLoss()

        # Create datasets and dataloaders
        train_dataset = MetGBMDataset(train_data, config, config.train, train_target)
        valid_dataset = MetGBMDataset(valid_data, config, config.oof, valid_target)

        train_loader = DataLoader(train_dataset, batch_size=config.model_params.train_batch_size, shuffle=True,
                                  num_workers=config.model_params.num_workers)
        valid_loader = DataLoader(valid_dataset, batch_size=config.model_params.test_batch_size, shuffle=False,
                                  num_workers=config.model_params.num_workers)

        # Epoch loop
        for epoch in range(config.model_params.epochs):
            start_epoch_time = time.time()
            train_losses = 0

            # Set model to training mode
            fold_model.train()

            # Lists to store predictions and labels for each batch
            all_train_preds = []
            all_train_labels = []
            all_train_segs = []
            # Training loop
            for images, labels, segs in train_loader:
                images = images.to(config.device, dtype=torch.float32)
                labels = labels.to(config.device, dtype=torch.float32)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                out = fold_model(images)
                loss = criterion(out, labels.unsqueeze(1))

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                # Accumulate training loss
                train_losses += loss.item()

                # Store predictions and labels
                train_preds = torch.sigmoid(out).detach()
                all_train_preds.extend(train_preds.flatten().cpu().numpy().astype(float).tolist())
                all_train_labels.extend(labels.cpu().tolist())
                all_train_segs.extend(segs)
                
                # Clear GPU memory
                del images, labels, out, train_preds, segs

            # Evaluate on validation set
            with torch.no_grad():
                agg_mean_train_preds, train_tumor_labels, _ = agg_predictions_per_tumor(all_train_segs, all_train_preds, all_train_labels)
                train_acc = accuracy_score(train_tumor_labels, np.round(agg_mean_train_preds))

                valid_preds, valid_labels = get_model_preds(fold_model, valid_loader, len(valid_target), config.device, has_labels = True)
                agg_mean_valid_preds, valid_tumor_labels, _ = agg_predictions_per_tumor(valid_dataset.segs_list, valid_preds, valid_labels)

                valid_acc = accuracy_score(valid_tumor_labels, np.round(agg_mean_valid_preds))
                valid_f1_score = f1_score(valid_tumor_labels, np.round(agg_mean_valid_preds))
                valid_roc = roc_auc_score(valid_tumor_labels, agg_mean_valid_preds)

            # Log performance metrics
            config.logger.info(
                f"Epoch: {epoch + 1}/{config.model_params.epochs} | Loss: {train_losses:.4f} | Train Acc: {train_acc:.3f} | "
                f"Valid Acc: {valid_acc:.3f} | Valid ROC: {valid_roc:.3f} | Valid F1: {valid_f1_score:.3f}")

            # Step the learning rate scheduler
            scheduler.step(valid_roc)

            # Save the best model so far
            if not best_roc: # If best_roc = None
                best_roc = valid_roc
                best_model_state_dict = fold_model.state_dict()
                best_model_char = f"Fold{fold+1}_Epoch{epoch+1}_ValidAcc_{valid_acc:.3f}_ROC_{valid_roc:.3f}_f1_score_{valid_f1_score:.3f}"
                best_agg_mean_valid_preds  = agg_mean_valid_preds
                best_valid_tumor_labels = valid_tumor_labels
                continue

            if valid_roc > best_roc:
                best_roc = valid_roc
                best_model_state_dict = fold_model.state_dict()
                best_model_char = f"Fold{fold+1}_Epoch{epoch+1}_ValidAcc_{valid_acc:.3f}_ROC_{valid_roc:.3f}_f1_score_{valid_f1_score:.3f}"
                patience_f = config.model_params.patience
                best_agg_mean_valid_preds  = agg_mean_valid_preds
                best_valid_tumor_labels = valid_tumor_labels
                
            else:
                # Decrease patience due to no improvement in ROC
                patience_f = patience_f - 1
                if patience_f == 0:
                    config.logger.info(f"Early stopping due to no improvement | Best ROC: {best_roc:.3f}")
                    break

        # Save the best model for the current fold
        config.logger.info("Evaluating best model for the current fold...")

        if config.model_params.save_best_model:
            torch.save(best_model_state_dict, os.path.join(config.curr_run_results_path, best_model_char + ".pth"))

        # Load the best model to store for later use
        best_model = load_model(config)
        #best_model = nn.DataParallel(best_model, device_ids=[0, 2])
        best_model.load_state_dict(best_model_state_dict)
        # Move model to cpu to reduce memory usage
        best_model.to('cpu')

        # Append best model to list and store predictions
        folds_best_model.append(best_model)
        folds_oof_mean_preds_dict[fold + 1] = list(best_agg_mean_valid_preds.astype(float))
        folds_oof_tumor_label_dict[fold + 1] = list(best_valid_tumor_labels.astype(float))

        # Clean up to free memory
        del train_dataset, train_loader, valid_loader, best_model, fold_model, optimizer
        gc.collect()

    
    if config.global_params.save_train_val_idx:
        folds_idx = {'train': folds_train_idx, 'valid': folds_valid_idx}
        folds_idx_path = os.path.join(config.curr_run_results_path, 'folds_train_valid_idx.json')
        with open(folds_idx_path, "w") as f:
            # Save training and validation indices for reproducibility
            
            json.dump(folds_idx, f)

    # Log total training time
    elapsed = time.time() - start_time
    config.logger.info(f"\nTraining completed in {elapsed:.2f} seconds")

    return folds_oof_mean_preds_dict, folds_oof_tumor_label_dict, folds_best_model



