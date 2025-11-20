"""
Subject-Independent Training (Leave-One-Subject-Out Cross-Validation)

This script evaluates subject-independent performance by training on multiple
subjects and testing on a held-out subject. This is useful for research
and understanding generalizability.
"""

from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import accuracy_score, cohen_kappa_score
from tqdm.auto import tqdm
import pandas as pd

from app.scripts.train import run_training
from app.utils.early_stopping import EarlyStopping
from app.utils.data_utils import get_all_epochs, MNEEpochsDataset


def get_all_subjects_data(data_loader, subject_ids, get_all_epochs_func):
    """
    Load and concatenate data from multiple subjects.
    
    Parameters:
    -----------
    data_loader : EEGDataLoader
        Data loader instance
    subject_ids : list
        List of subject IDs to load
    get_all_epochs_func : callable
        Function to get all epochs for a subject
        
    Returns:
    --------
    all_data : np.ndarray
        Concatenated EEG data (n_trials, n_channels, n_times)
    all_labels : np.ndarray
        Concatenated labels
    subject_ids_per_trial : np.ndarray
        Subject ID for each trial (for tracking)
    """
    all_epochs_list = []
    subject_ids_per_trial = []
    
    for subject_id in subject_ids:
        epochs = get_all_epochs_func(data_loader, subject_id)
        if epochs is not None and len(epochs) > 0:
            all_epochs_list.append(epochs)
            # Track which subject each trial belongs to
            n_trials = len(epochs)
            subject_ids_per_trial.extend([subject_id] * n_trials)
    
    if not all_epochs_list:
        return None, None, None
    
    # Concatenate epochs
    from mne import concatenate_epochs
    combined_epochs = concatenate_epochs(all_epochs_list)
    
    eeg_data = combined_epochs.get_data(copy=False)
    labels = combined_epochs.events[:, -1] - 1  # 0-indexed
    
    return eeg_data, labels, np.array(subject_ids_per_trial)


def train_subject_independent_loso(
    data_loader,
    model_class,
    model_kwargs,
    subject_ids=None,
    epochs=100,
    batch_size=64,
    lr=1e-3,
    weight_decay=1e-4,
    device=None,
    save_dir='outputs/subject_independent',
    random_state=42,
):
    """
    Leave-One-Subject-Out (LOSO) Cross-Validation for subject-independent evaluation.
    
    For each subject:
    1. Train on all other subjects
    2. Test on held-out subject
    3. Report performance
    
    Parameters:
    -----------
    data_loader : EEGDataLoader
        Data loader instance
    model_class : nn.Module class
        Model class (e.g., EEGNet)
    model_kwargs : dict
        Keyword arguments for model initialization
    subject_ids : list, optional
        List of subject IDs. If None, uses all subjects.
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
        DataFrame with results for each test subject
    """
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if subject_ids is None:
        subject_ids = data_loader.list_subjects()
    
    os.makedirs(save_dir, exist_ok=True)
    
    results = []
    
    print(f"🚀 Starting Subject-Independent Training (Leave-One-Subject-Out)")
    print(f"📊 Subjects: {subject_ids}")
    print(f"💻 Device: {device}\n")
    
    for test_subject in tqdm(subject_ids, desc="Test Subjects"):
        print(f"\n{'='*60}")
        print(f"📌 Test Subject: {test_subject}")
        print(f"{'='*60}")
        
        # Get training subjects (all except test subject)
        train_subjects = [s for s in subject_ids if s != test_subject]
        print(f"   Training on subjects: {train_subjects}")
        
        # Load training data (all subjects except test)
        X_train, y_train, _ = get_all_subjects_data(
            data_loader, train_subjects, get_all_epochs
        )
        
        if X_train is None:
            print(f"⚠️  No training data found, skipping...")
            continue
        
        # Load test data (held-out subject)
        test_epochs = get_all_epochs(data_loader, test_subject)
        if test_epochs is None or len(test_epochs) == 0:
            print(f"⚠️  No test data found for {test_subject}, skipping...")
            continue
        
        X_test = test_epochs.get_data(copy=False)
        y_test = test_epochs.events[:, -1] - 1  # 0-indexed
        
        print(f"   Train: {X_train.shape[0]} trials")
        print(f"   Test:  {X_test.shape[0]} trials")
        print(f"   Train labels: {np.bincount(y_train)}")
        print(f"   Test labels:  {np.bincount(y_test)}")
        
        # Split training data into train/val (80/20)
        from sklearn.model_selection import train_test_split
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=0.2, random_state=random_state, stratify=y_train
        )
        
        # Create datasets
        train_dataset = MNEEpochsDataset(X_train_split, y_train_split)
        val_dataset = MNEEpochsDataset(X_val_split, y_val_split)
        test_dataset = MNEEpochsDataset(X_test, y_test)
        
        # Initialize model
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
        save_prefix = os.path.join(save_dir, f"trained_on_{'_'.join(train_subjects)}_test_{test_subject}")
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
        
        # Evaluate on test set
        model.eval()
        test_preds = []
        test_labels = []
        
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                logits = model(x)
                preds = logits.argmax(dim=1)
                test_preds.extend(preds.cpu().numpy())
                test_labels.extend(y.numpy())
        
        test_acc = accuracy_score(test_labels, test_preds)
        test_kappa = cohen_kappa_score(test_labels, test_preds)
        
        results.append({
            'test_subject': test_subject,
            'train_subjects': '_'.join(train_subjects),
            'n_train_trials': len(X_train),
            'n_test_trials': len(X_test),
            'test_acc': test_acc,
            'test_kappa': test_kappa,
            'best_epoch': early_stopping.best_epoch if early_stopping else epochs,
        })
        
        print(f"   ✅ Test Accuracy: {test_acc:.4f}")
        print(f"   ✅ Test Kappa:     {test_kappa:.4f}")
    
    # Summary
    results_df = pd.DataFrame(results)
    
    print(f"\n{'='*60}")
    print(f"📊 OVERALL RESULTS (Subject-Independent)")
    print(f"{'='*60}")
    
    mean_acc = results_df['test_acc'].mean()
    std_acc = results_df['test_acc'].std()
    mean_kappa = results_df['test_kappa'].mean()
    std_kappa = results_df['test_kappa'].std()
    
    print(f"\nOverall Performance (across all test subjects):")
    print(f"  Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  Kappa:    {mean_kappa:.4f} ± {std_kappa:.4f}")
    
    print("\nPer-Subject Performance:")
    print(results_df[['test_subject', 'test_acc', 'test_kappa']].to_string(index=False))
    
    # Save results
    results_path = os.path.join(save_dir, 'loso_results.csv')
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
    
    # Train subject-independent models
    # results = train_subject_independent_loso(
    #     data_loader=data_loader,
    #     model_class=EEGNet,
    #     model_kwargs=model_kwargs,
    #     subject_ids=None,  # Use all subjects
    #     epochs=100,
    #     batch_size=64,
    #     lr=1e-3,
    #     save_dir='outputs/subject_independent',
    # )
    pass

