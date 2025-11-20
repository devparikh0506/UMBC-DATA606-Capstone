"""Data utility functions for preprocessing and standardization."""

import numpy as np
import mne
from typing import Tuple, Dict


def standardize_signals(X_train: np.ndarray, X_test: np.ndarray, 
                        eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Standardize EEG signals using per-electrode z-score normalization.
    
    Args:
        X_train: Training data of shape (n_trials, n_channels, n_times)
        X_test: Test data of shape (n_trials, n_channels, n_times)
        eps: Small epsilon to avoid division by zero
        
    Returns:
        X_train_std: Standardized training data
        X_test_std: Standardized test data
        stats: Dictionary with mean and std per channel
    """
    mean_per_channel = np.mean(X_train, axis=(0, 2), keepdims=True)
    std_per_channel = np.std(X_train, axis=(0, 2), keepdims=True)
    std_per_channel = np.maximum(std_per_channel, eps)
    
    X_train_std = (X_train - mean_per_channel) / std_per_channel
    X_test_std = (X_test - mean_per_channel) / std_per_channel
    
    stats = {
        'mean': mean_per_channel.squeeze(),
        'std': std_per_channel.squeeze()
    }
    
    return X_train_std, X_test_std, stats


def standardize_signals_using_stats(X: np.ndarray, stats: Dict, 
                                   eps: float = 1e-8) -> np.ndarray:
    """
    Standardize signals using pre-computed statistics.
    
    Args:
        X: Data of shape (n_trials, n_channels, n_times)
        stats: Dictionary with 'mean' and 'std' arrays of shape (n_channels,)
        eps: Small epsilon to avoid division by zero
        
    Returns:
        X_std: Standardized data
    """
    mean = stats['mean']
    std = np.maximum(stats['std'], eps)
    
    mean = mean.reshape(1, -1, 1)
    std = std.reshape(1, -1, 1)
    
    X_std = (X - mean) / std
    return X_std


def get_all_epochs(loader, subject):
    """
    Fetch all epochs for a given subject (training sessions only).
    
    Args:
        loader: EEGDataLoader instance
        subject: Subject ID (e.g., 'B01')
        
    Returns:
        Concatenated epochs for all training sessions, or None if no epochs found
    """
    all_epochs = []
    for session in loader.list_sessions(subject):
        data = loader.get_data(subject, session, 'T')
        if data['epochs'] is not None and len(data['epochs']) > 0:
            all_epochs.append(data['epochs'])
    
    if all_epochs:
        return mne.concatenate_epochs(all_epochs)
    return None

