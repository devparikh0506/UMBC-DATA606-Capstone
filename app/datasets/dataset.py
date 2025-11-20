"""PyTorch dataset wrapper for MNE epochs data."""

import numpy as np


class MNEEpochsDataset:
    """Dataset wrapper for MNE epochs data."""
    
    def __init__(self, X, y):
        """
        Args:
            X: EEG data array of shape (n_trials, n_channels, n_times)
            y: Labels array of shape (n_trials,)
        """
        self.X = X
        self.y = y
        
        if isinstance(self.X, np.ndarray):
            self.X = self.X.astype(np.float32)
        if isinstance(self.y, np.ndarray):
            self.y = self.y.astype(np.int64)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        return x, y

