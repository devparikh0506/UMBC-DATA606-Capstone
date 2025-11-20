"""
Subject-Dependent Training with Cross-Validation

This script trains separate models for each subject using cross-validation
within each subject's data. This is the standard approach for BCI applications.
"""

from __future__ import annotations
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, cohen_kappa_score
from tqdm.auto import tqdm
import pandas as pd

from app.scripts.train import run_training
from app.utils.early_stopping import EarlyStopping
from app.utils.data_utils import get_all_epochs, MNEEpochsDataset


def train_subject_dependent_cv(
    data_loader,
    model_class,
    model_kwargs,
    subject_ids=None,
    n_splits=5,
    epochs=100,
    batch_size=64,
    lr=1e-3,
    weight_decay=1e-4,
    device=None,
    save_dir='outputs/subject_models',
    random_state=42,
):
    """
    Train subject-dependent models with cross-validation.
    
    For each subject:
    1. Load all training data for that subject
    2. Perform k-fold cross-validation
    3. Train final model on all data
    4. Save subject-specific model
    
    Parameters:
    -----------
    data_loader : EEGDataLoader
        Data loader instance
    model_class : nn.Module class
        Model class (e.g., EEGNet)
    model_kwargs : dict
        Keyword arguments for model initialization
    subject_ids : list, optional
        List of subject IDs to train. If None, uses all subjects.
    n_splits : int
        Number of folds for cross-validation
    epochs : int
        Maximum number of training epochs
    batch_size : int
        Batch size for training
    lr : float
        Learning rate
    weight_decay : float
        Weight decay for optimizer
    device : torch.device, optional
        Device to use. If None, auto-detects.
    save_dir : str
        Directory to save models
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    results : pd.DataFrame
        DataFrame with results for each subject and fold
    """
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if subject_ids is None:
        subject_ids = data_loader.list_subjects()
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Results storage
    all_results = []
    
    print(f"🚀 Starting Subject-Dependent Training with {n_splits}-Fold CV")
    print(f"📊 Subjects: {subject_ids}")
    print(f"💻 Device: {device}\n")
    
    for subject_id in tqdm(subject_ids, desc="Subjects"):
        print(f"\n{'='*60}")
        print(f"📌 Training Subject: {subject_id}")
        print(f"{'='*60}")
        
        # Load all epochs for this subject
        subject_epochs = get_all_epochs(data_loader, subject_id)
        
        if subject_epochs is None or len(subject_epochs) == 0:
            print(f"⚠️  No data found for {subject_id}, skipping...")
            continue
        
        # Extract data and labels
        eeg_data = subject_epochs.get_data(copy=False)  # (n_trials, n_channels, n_times)
        labels = subject_epochs.events[:, -1] - 1  # Convert to 0-indexed
        
        print(f"   Data shape: {eeg_data.shape}")
        print(f"   Labels: {np.bincount(labels)} (class distribution)")
        
        # Cross-validation
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        cv_results = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(eeg_data, labels)):
            print(f"\n   🔄 Fold {fold + 1}/{n_splits}")
            print(f"      Train: {len(train_idx)} trials, Val: {len(val_idx)} trials")
            
            # Split data
            X_train_fold = eeg_data[train_idx]
            y_train_fold = labels[train_idx]
            X_val_fold = eeg_data[val_idx]
            y_val_fold = labels[val_idx]
            
            # Create datasets
            train_dataset = MNEEpochsDataset(X_train_fold, y_train_fold)
            val_dataset = MNEEpochsDataset(X_val_fold, y_val_fold)
            
            # Initialize model (fresh model for each fold)
            model = model_class(**model_kwargs)
            model.to(device)
            
            # Optimizer and loss
            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay
            )
            criterion = nn.CrossEntropyLoss()
            
            # Learning rate scheduler
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=10
            )
            
            # Early stopping
            early_stopping = EarlyStopping(
                monitor='val_loss', 
                mode='min', 
                patience=50
            )
            
            # Train
            save_prefix = os.path.join(save_dir, f"{subject_id}_fold{fold+1}")
            model, history, best_path = run_training(
                model=model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                optimizer=optimizer,
                loss_fn=criterion,
                device=device,
                epochs=epochs,
                batch_size=batch_size,
                save_prefix=save_prefix,
                scheduler=lr_scheduler,
                early_stopping=early_stopping,
            )
            
            # Evaluate on validation set
            model.eval()
            val_preds = []
            val_labels = []
            
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device)
                    logits = model(x)
                    preds = logits.argmax(dim=1)
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(y.numpy())
            
            val_acc = accuracy_score(val_labels, val_preds)
            val_kappa = cohen_kappa_score(val_labels, val_preds)
            
            cv_results.append({
                'fold': fold + 1,
                'val_acc': val_acc,
                'val_kappa': val_kappa,
                'best_epoch': early_stopping.best_epoch if early_stopping else epochs,
            })
            
            all_results.append({
                'subject': subject_id,
                'fold': fold + 1,
                'val_acc': val_acc,
                'val_kappa': val_kappa,
                'best_epoch': early_stopping.best_epoch if early_stopping else epochs,
            })
            
            print(f"      ✅ Fold {fold + 1} - Val Acc: {val_acc:.4f}, Val Kappa: {val_kappa:.4f}")
        
        # Train final model on all data
        print(f"\n   🎯 Training Final Model on All Data")
        final_model = model_class(**model_kwargs)
        final_model.to(device)
        
        # Use 80/20 split for final model training
        from sklearn.model_selection import train_test_split
        X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
            eeg_data, labels, test_size=0.2, random_state=random_state, stratify=labels
        )
        
        train_dataset_final = MNEEpochsDataset(X_train_final, y_train_final)
        val_dataset_final = MNEEpochsDataset(X_val_final, y_val_final)
        
        optimizer_final = torch.optim.Adam(
            final_model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        criterion_final = nn.CrossEntropyLoss()
        lr_scheduler_final = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_final, mode='min', factor=0.5, patience=10
        )
        early_stopping_final = EarlyStopping(
            monitor='val_loss', 
            mode='min', 
            patience=50
        )
        
        save_prefix_final = os.path.join(save_dir, f"{subject_id}_final")
        final_model, history_final, best_path_final = run_training(
            model=final_model,
            train_dataset=train_dataset_final,
            val_dataset=val_dataset_final,
            optimizer=optimizer_final,
            loss_fn=criterion_final,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            save_prefix=save_prefix_final,
            scheduler=lr_scheduler_final,
            early_stopping=early_stopping_final,
        )
        
        # Summary for this subject
        cv_df = pd.DataFrame(cv_results)
        mean_acc = cv_df['val_acc'].mean()
        std_acc = cv_df['val_acc'].std()
        mean_kappa = cv_df['val_kappa'].mean()
        std_kappa = cv_df['val_kappa'].std()
        
        print(f"\n   📊 Subject {subject_id} CV Results:")
        print(f"      Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"      Kappa:    {mean_kappa:.4f} ± {std_kappa:.4f}")
        print(f"      Final model saved: {best_path_final}")
    
    # Overall summary
    results_df = pd.DataFrame(all_results)
    
    print(f"\n{'='*60}")
    print(f"📊 OVERALL RESULTS (Subject-Dependent)")
    print(f"{'='*60}")
    
    # Per-subject summary
    subject_summary = results_df.groupby('subject').agg({
        'val_acc': ['mean', 'std'],
        'val_kappa': ['mean', 'std'],
    }).round(4)
    
    print("\nPer-Subject Performance:")
    print(subject_summary)
    
    # Overall mean across all subjects and folds
    overall_acc = results_df['val_acc'].mean()
    overall_acc_std = results_df['val_acc'].std()
    overall_kappa = results_df['val_kappa'].mean()
    overall_kappa_std = results_df['val_kappa'].std()
    
    print(f"\nOverall Performance (across all subjects and folds):")
    print(f"  Accuracy: {overall_acc:.4f} ± {overall_acc_std:.4f}")
    print(f"  Kappa:    {overall_kappa:.4f} ± {overall_kappa_std:.4f}")
    
    # Save results
    results_path = os.path.join(save_dir, 'cv_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\n💾 Results saved to: {results_path}")
    
    return results_df


if __name__ == "__main__":
    # Example usage
    # Note: Import EEGDataLoader and EEGNet from your notebook or create separate module
    # Example:
    #   from notebooks.model_training import EEGDataLoader, EEGNet
    #   or create a separate models.py module
    
    # Initialize data loader
    # data_loader = EEGDataLoader()
    
    # Model configuration
    # model_kwargs = {
    #     'num_classes': 2,
    #     'num_channels': 3,
    #     'fs': 250,
    #     'dropout_rate': 0.5,
    #     'F1': 16,
    #     'D': 2,
    #     'k_temporal': int(250 * 0.5),
    #     'use_spatial_dropout': True,
    # }
    
    # Train all subjects
    # results = train_subject_dependent_cv(
    #     data_loader=data_loader,
    #     model_class=EEGNet,
    #     model_kwargs=model_kwargs,
    #     subject_ids=None,  # Train all subjects
    #     n_splits=5,
    #     epochs=100,
    #     batch_size=64,
    #     lr=1e-3,
    #     save_dir='outputs/subject_models',
    # )
    pass

