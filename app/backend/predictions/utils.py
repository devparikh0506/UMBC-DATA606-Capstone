"""
Utilities for model loading, data preprocessing, and inference.
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import mne
from pathlib import Path
from django.conf import settings

# Add parent directories to path to import EEGNet and EEGDataLoader
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

# Import from notebooks (we'll need to extract these classes)
# For now, we'll define them here based on the notebook code


class EEGNet(nn.Module):
    """
    EEGNet: Compact Convolutional Neural Network for EEG Classification.
    
    Architecture (Lawhern et al., 2018):
    - Block 1: Temporal Conv → Depthwise Conv → Separable Conv → AvgPool
    - Block 2: Depthwise Conv → Separable Conv → AvgPool
    - Classification: Dense layer with dropout
    
    Matches implementation from training.ipynb notebook.
    """
    def __init__(self, 
                 num_channels=3,
                 num_classes=2,
                 F1=8,
                 F2=16,
                 D=2,
                 kernel_length=64,
                 pool_time=4,
                 pool_space=1,
                 dropout_rate=0.5):
        super().__init__()
        
        # Block 1: Temporal Convolution
        self.conv_temporal = nn.Conv2d(1, F1, (1, kernel_length), 
                                       padding=(0, kernel_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        
        # Block 1: Depthwise Convolution (spatial)
        self.conv_spatial = nn.Conv2d(F1, D * F1, (num_channels, 1), 
                                      groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(D * F1)
        self.elu1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((pool_space, pool_time))
        self.drop1 = nn.Dropout(dropout_rate)
        
        # Block 2: Separable Convolution (depthwise + pointwise)
        self.conv_separable_depth = nn.Conv2d(D * F1, D * F1, (1, 16),
                                               groups=D * F1, padding=(0, 8), bias=False)
        self.conv_separable_point = nn.Conv2d(D * F1, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.elu2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout_rate)
        
        # Classification - use global average pooling for flexibility
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(F2, num_classes, bias=True)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, channels, time) or (batch, 1, channels, time)
        
        Returns:
            output: Class logits of shape (batch, num_classes)
        """
        # Handle input shape: (batch, channels, time) -> (batch, 1, channels, time)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # Block 1: Temporal Convolution
        x = self.conv_temporal(x)
        x = self.bn1(x)
        
        # Block 1: Depthwise Spatial Convolution
        x = self.conv_spatial(x)
        x = self.bn2(x)
        x = self.elu1(x)
        x = self.pool1(x)
        x = self.drop1(x)
        
        # Block 2: Separable Convolution
        x = self.conv_separable_depth(x)
        x = self.conv_separable_point(x)
        x = self.bn3(x)
        x = self.elu2(x)
        x = self.pool2(x)
        x = self.drop2(x)
        
        # Global average pooling: (batch, F2, 1, time) -> (batch, F2, 1, 1)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # (batch, F2)
        
        # Classification
        output = self.classifier(x)
        
        return output


class EEGDataLoader:
    """
    EEGDataLoader for BCI Competition IV-2b dataset.
    
    Matches preprocessing pipeline from training notebook:
    - Bandpass filtering: 4-40 Hz
    - Notch filtering: 50 Hz
    - Epoch extraction: [2.5, 7.0] seconds (1125 samples at 250 Hz)
    - Per-electrode standardization support
    """
    def __init__(self, data_folder, file_filters=None, 
                 raw_channels=['EEG:Cz', 'EEG:C3', 'EEG:C4'], 
                 channel_fun=lambda x: x.replace('EEG:', ''), 
                 filtered=True,
                 apply_standardization=True,
                 filter_range=(4, 40),
                 apply_notch=True,
                 notch_freq=50):
        import re
        self.filename_pattern = r"^(B\d{2})(\d{2})([A-Za-z])\.gdf$"
        self.data_folder = data_folder
        self.file_filters = file_filters
        self.data_dir_info = self._scan_data_dir_(self.file_filters)
        self._data_cache = {}
        self.filter = filtered
        self.apply_standardization = apply_standardization
        self.filter_range = filter_range
        self.apply_notch = apply_notch
        self.notch_freq = notch_freq
        self.raw_channels = raw_channels
        if channel_fun:
            self.channels = [channel_fun(ch) for ch in self.raw_channels]
        else:
            self.channels = self.raw_channels
        self.channel_mapping = {raw: ch for raw, ch in zip(self.raw_channels, self.channels)}
        # Cache for standardization statistics
        self._standardization_stats = {}
    
    def _scan_data_dir_(self, filters=None):
        import re
        files = [f for f in os.listdir(self.data_folder) if f.endswith('.gdf')]
        info = []
        for filename in files:
            match = re.match(self.filename_pattern, filename)
            if match:
                subj, sess, stype = match.groups()
                if filters:
                    if 'subject_ids' in filters and subj not in filters['subject_ids']:
                        continue
                    if 'session_ids' in filters and sess not in filters['session_ids']:
                        continue
                    if 'session_types' in filters and stype not in filters['session_types']:
                        continue
                info.append({
                    "subject_id": subj,
                    "session_id": sess,
                    "session_type": stype,
                    "filename": filename
                })
        return info
    
    def get_data(self, subject_id, session_id, session_type, reload=False):
        key = (subject_id, session_id, session_type)
        if not reload and key in self._data_cache:
            return self._data_cache[key]
        data = self._load_eeg_data_(subject_id, session_id, session_type)
        self._data_cache[key] = data
        return data
    
    def _load_eeg_data_(self, subject_id, session_id, session_type):
        """
        Load and preprocess EEG data matching training pipeline exactly.
        
        Preprocessing steps (must match training.ipynb/CTNet notebook):
        1. Load GDF file
        2. Rename channels (EEG:Cz -> Cz, EEG:C3 -> C3, EEG:C4 -> C4)
        3. Set EOG channel types
        4. Set montage (standard_1020)
        5. Filter: 4-40 Hz bandpass + 50 Hz notch (matching training)
        6. Channel selection: Cz, C3, C4 (in that order)
        7. Epoch extraction: tmin=2.5, tmax=7.0 (1125 samples at 250 Hz), baseline=None
        
        Following BCI Competition IV-2b protocol:
        - Training: Sessions 1, 2, 3 (type 'T')
        - Test: Sessions 4, 5 (type 'E')
        - Epoch window: [2.5, 7.0] seconds relative to cue (4.5 seconds = 1125 samples)
        """
        file_path = os.path.join(self.data_folder, f"{subject_id}{session_id}{session_type}.gdf")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        raw = mne.io.read_raw_gdf(file_path, preload=True, verbose=False)
        raw.rename_channels(self.channel_mapping)
        for ch in raw.ch_names:
            if ch.startswith('EOG'):
                raw.set_channel_types({ch: 'eog'})
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, match_case=False)
        
        # Preprocessing: Matching training notebook
        if self.filter:
            # Use configurable filter range (default: 4-40 Hz)
            low_freq, high_freq = self.filter_range
            raw.filter(low_freq, high_freq, verbose=False)
            if self.apply_notch:
                raw.notch_filter(self.notch_freq, verbose=False)
        
        available_channels = []
        for ch in self.channels:
            if ch in raw.ch_names:
                available_channels.append(ch)
            elif f"EEG:{ch}" in raw.ch_names:
                available_channels.append(f"EEG:{ch}")
        if not available_channels:
            raise ValueError(f"None of the requested channels {self.channels} found")
        
        raw.pick_channels(available_channels)
        picks = mne.pick_channels(raw.info["ch_names"], include=available_channels)
        
        events, event_dict = mne.events_from_annotations(raw)
        
        # Only training sessions (T) are supported
        if session_type != 'T':
            return {'raw': raw, 'epochs': None, 'epochs_per_run': None}
        
        try:
            start_trial_code = event_dict['768']
            left_code = event_dict['769']
            right_code = event_dict['770']
            rejected_code = event_dict['1023']
            
            trial_starts = events[events[:, 2] == start_trial_code][:, 0]
            rejected_starts = set(events[events[:, 2] == rejected_code][:, 0])
            mi_events = events[np.isin(events[:, 2], [left_code, right_code])]
            
            valid_mi_events = []
            for mi_event in mi_events:
                trial_start = trial_starts[trial_starts <= mi_event[0]]
                if len(trial_start) > 0:
                    trial_start = trial_start.max()
                    if trial_start not in rejected_starts:
                        valid_mi_events.append(mi_event)
            
            if not valid_mi_events:
                return {'raw': raw, 'epochs': None, 'epochs_per_run': None}
            
            valid_mi_events = np.array(valid_mi_events)
            standard_event_id = {'left': 1, 'right': 2}
            remapped_events = valid_mi_events.copy()
            remapped_events[:, 2][remapped_events[:, 2] == left_code] = standard_event_id['left']
            remapped_events[:, 2][remapped_events[:, 2] == right_code] = standard_event_id['right']
        except KeyError as e:
            return {'raw': raw, 'epochs': None, 'epochs_per_run': None}
        
        # Create epochs matching training parameters exactly
        # Paper: BCI IV-2b uses [2.5, 7] seconds relative to cue
        # tmin=2.5, tmax=7.0 (4.5 seconds = 1125 samples at 250Hz)
        # This must match training.ipynb/CTNet notebook epoching parameters
        epochs = mne.Epochs(
            raw, remapped_events, standard_event_id,
            picks=picks, tmin=2.5, tmax=7.0,
            preload=True, baseline=None, verbose=False
        )
        
        return {
            'raw': raw,
            'epochs': epochs,
            'epochs_per_run': None,  # Not used in backend but kept for compatibility
            'event_dict': event_dict
        }
    
    def list_subjects(self):
        """List all available subjects."""
        return sorted(set(x["subject_id"] for x in self.data_dir_info))
    
    def list_sessions(self, subject_id):
        """List all available sessions for a subject."""
        return sorted(set(x["session_id"] for x in self.data_dir_info if x["subject_id"] == subject_id))
    
    def list_session_types(self, subject_id, session_id):
        """List all available session types for a subject and session."""
        return sorted(set(x["session_type"] for x in self.data_dir_info
                         if x["subject_id"] == subject_id and x["session_id"] == session_id))
    
    def _standardize_data(self, X_train, X_test, subject_id=None, eps=1e-8):
        """
        Standardize data using training statistics only (prevents data leakage).
        
        Args:
            X_train: Training data of shape (n_trials, n_channels, n_times)
            X_test: Test data of shape (n_trials, n_channels, n_times)
            subject_id: Subject ID for caching statistics (optional)
            eps: Small epsilon to avoid division by zero
            
        Returns:
            X_train_std: Standardized training data
            X_test_std: Standardized test data
            stats: Dictionary with mean and std per channel
        """
        # Compute mean and std per channel across all training trials
        mean_per_channel = np.mean(X_train, axis=(0, 2), keepdims=True)
        std_per_channel = np.std(X_train, axis=(0, 2), keepdims=True)
        
        # Add epsilon to avoid division by zero
        std_per_channel = np.maximum(std_per_channel, eps)
        
        # Standardize training data
        X_train_std = (X_train - mean_per_channel) / std_per_channel
        
        # Standardize test data using training statistics (prevents data leakage)
        X_test_std = (X_test - mean_per_channel) / std_per_channel
        
        # Store statistics
        stats = {
            'mean': mean_per_channel.squeeze(),  # (n_channels,)
            'std': std_per_channel.squeeze()      # (n_channels,)
        }
        
        # Cache statistics if subject_id provided
        if subject_id:
            self._standardization_stats[subject_id] = stats
        
        return X_train_std, X_test_std, stats
    
    def standardize_signals_using_stats(self, X, stats, eps=1e-8):
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
        
        # Reshape for broadcasting
        mean = mean.reshape(1, -1, 1)
        std = std.reshape(1, -1, 1)
        
        X_std = (X - mean) / std
        return X_std


def load_model(subject_id, device='cpu'):
    """
    Load PyTorch EEGNet model for a subject.
    
    Model configuration matches training.ipynb:
    - F1=8, F2=16, D=2, kernel_length=64
    - dropout_rate=0.5
    - Input: (batch, 3, 1125) - 3 channels, 1125 time points (4.5 seconds at 250 Hz)
    """
    model_path = os.path.join(settings.MODELS_FOLDER, subject_id, 'best_model.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Match training notebook model configuration
    model = EEGNet(
        num_classes=2,
        num_channels=3,
        F1=8,
        F2=16,
        D=2,
        kernel_length=64,
        pool_time=4,
        pool_space=1,
        dropout_rate=0.5,
    )
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def get_standardization_stats(subject_id):
    """
    Get standardization statistics for a subject from training data.
    
    This loads training sessions (01T, 02T, 03T) and computes statistics
    to match training preprocessing.
    
    Returns:
        stats: Dictionary with 'mean' and 'std' arrays of shape (n_channels,)
               Returns None if training data not available
    """
    try:
        # Load training data to compute standardization statistics
        data_loader = EEGDataLoader(
            data_folder=str(settings.DATA_FOLDER),
            raw_channels=['EEG:Cz', 'EEG:C3', 'EEG:C4'],
            channel_fun=lambda x: x.replace('EEG:', ''),
            filtered=True,
            apply_standardization=False,  # We'll compute stats manually
            filter_range=(4, 40),
            apply_notch=True,
            notch_freq=50
        )
        
        # Load training sessions (01T, 02T, 03T)
        train_epochs_list = []
        for session_id in ['01', '02', '03']:
            try:
                data = data_loader.get_data(subject_id, session_id, 'T')
                if data['epochs'] is not None and len(data['epochs']) > 0:
                    train_epochs_list.append(data['epochs'])
            except (FileNotFoundError, Exception):
                continue
        
        if not train_epochs_list:
            return None
        
        # Concatenate training epochs
        train_epochs = mne.concatenate_epochs(train_epochs_list) if len(train_epochs_list) > 1 else train_epochs_list[0]
        X_train = train_epochs.get_data(copy=False)  # (n_trials, n_channels, n_times)
        
        # Compute standardization statistics (per channel)
        mean_per_channel = np.mean(X_train, axis=(0, 2))  # (n_channels,)
        std_per_channel = np.std(X_train, axis=(0, 2))    # (n_channels,)
        
        stats = {
            'mean': mean_per_channel,
            'std': np.maximum(std_per_channel, 1e-8)  # Add epsilon
        }
        
        return stats
        
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Could not load standardization stats for {subject_id}: {e}")
        return None


def load_evaluation_data(subject_id, run_id, metadata_only=False, use_db=True):
    """
    Load training session data.
    
    Supports:
    - Training runs: 01T, 02T, 03T
    
    Args:
        subject_id: Subject ID (e.g., 'B01')
        run_id: Run ID (e.g., '01T', '02T', '03T')
        metadata_only: If True, return only metadata
        use_db: If True, try to load from database first, then fallback to file
    
    Returns:
        If metadata_only=True: dict with metadata
        If metadata_only=False: (eeg_data, labels) tuple
    """
    # Only training sessions are supported
    if not run_id.endswith('T'):
        raise ValueError(f"Invalid run_id format: {run_id}. Only training runs (01T, 02T, 03T) are supported")
    
    session_type = 'T'
    
    # Try database first if enabled (now works for all run types)
    if use_db:
        try:
            from api.models import Subject, EvaluationRun, TrialData
            
            subject = Subject.objects.get(subject_id=subject_id)
            run = EvaluationRun.objects.get(subject=subject, run_id=run_id)
            
            if metadata_only:
                return {
                    'run_id': run.run_id,
                    'session_id': run.session_id,
                    'session_type': run.session_type,
                    'n_trials': run.n_trials,
                    'n_channels': run.n_channels,
                    'n_times': run.n_times,
                    'sampling_rate': run.sampling_rate,
                    'duration_per_trial': run.duration_per_trial,
                }
            
            # Load trial data from database
            trials = TrialData.objects.filter(run=run).order_by('trial_index')
            if not trials.exists():
                raise ValueError(f"No trial data found in database for {subject_id}/{run_id}")
            
            eeg_data = np.array([trial.get_data() for trial in trials])
            # Labels are already 0-indexed (0=left, 1=right) as stored in database
            labels = np.array([trial.ground_truth for trial in trials])
            
            return eeg_data, labels
            
        except (Subject.DoesNotExist, EvaluationRun.DoesNotExist):
            # Fallback to file loading
            pass
        except Exception as e:
            # Log error but fallback to file loading
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Error loading from database: {e}. Falling back to file loading.")
            pass
    
    # Load from file using updated preprocessing pipeline
    # Matching training notebook: filter_range=(4, 40), apply_notch=True, notch_freq=50
    data_loader = EEGDataLoader(
        data_folder=str(settings.DATA_FOLDER),
        raw_channels=['EEG:Cz', 'EEG:C3', 'EEG:C4'],
        channel_fun=lambda x: x.replace('EEG:', ''),
        filtered=True,
        apply_standardization=False,  # Standardization done separately if needed
        filter_range=(4, 40),  # Bandpass filter: 4-40 Hz
        apply_notch=True,      # Notch filter: 50 Hz
        notch_freq=50
    )
    session_id = run_id[:2]  # Extract session number from "03T", "04E", etc.
    
    data = data_loader.get_data(subject_id, session_id, session_type)
    
    if data['epochs'] is None or len(data['epochs']) == 0:
        raise ValueError(f"No epochs found for {subject_id} {run_id}")
    
    epochs = data['epochs']
    eeg_data = epochs.get_data(copy=False)  # (n_trials, n_channels, n_times)
    labels = epochs.events[:, -1] - 1  # Convert to 0-indexed (1->0, 2->1)
    
    # Verify data shape matches expected (1125 time points from [2.5, 7.0] window)
    expected_timepoints = 1125  # 4.5 seconds at 250 Hz
    if eeg_data.shape[2] != expected_timepoints:
        raise ValueError(
            f"Unexpected time points: {eeg_data.shape[2]} (expected {expected_timepoints}). "
            f"Preprocessing may not match training pipeline."
        )
    
    if metadata_only:
        return {
            'n_trials': len(eeg_data),
            'n_channels': eeg_data.shape[1],
            'n_times': eeg_data.shape[2],
            'sampling_rate': epochs.info['sfreq'],
            'duration_per_trial': eeg_data.shape[2] / epochs.info['sfreq'],
        }
    
    return eeg_data, labels


def create_sliding_windows(data, window_size=1125, step=1125):
    """
    Create sliding windows from trial data.
    
    Args:
        data: numpy array of shape (n_trials, n_channels, n_times)
        window_size: number of time samples per window (default 1125 = 4.5s at 250Hz)
                     Must match model input size from training
        step: number of samples to slide (default 1125 = non-overlapping windows)
    
    Returns:
        windows: numpy array of shape (n_windows, n_channels, window_size)
        window_times: list of (trial_idx, start, center, end) tuples
    """
    n_trials, n_channels, n_times = data.shape
    windows = []
    window_times = []
    
    for trial_idx in range(n_trials):
        trial_data = data[trial_idx]  # (n_channels, n_times)
        start = 0
        while start + window_size <= n_times:
            window = trial_data[:, start:start + window_size]  # (n_channels, window_size)
            windows.append(window)
            center = start + window_size // 2
            window_times.append((trial_idx, start, center, start + window_size))
            start += step
    
    # Convert to numpy array: (n_windows, n_channels, window_size)
    return np.array(windows), window_times


def predict_window(model, window_data, device='cpu', stats=None):
    """
    Run inference on a single window.
    
    Args:
        model: PyTorch model
        window_data: numpy array of shape (n_channels, window_size) - 1125 time points
        device: device to run inference on
        stats: Standardization statistics dict with 'mean' and 'std' arrays (optional)
               If provided, will standardize the window_data before inference
    
    Returns:
        prediction: class index (0 or 1)
        confidence: probability of predicted class
        probabilities: array of probabilities for both classes
    """
    # Standardize if stats provided (matching training preprocessing)
    if stats is not None:
        mean = stats['mean'].reshape(-1, 1)  # (n_channels, 1)
        std = np.maximum(stats['std'], 1e-8).reshape(-1, 1)  # (n_channels, 1)
        window_data = (window_data - mean) / std
    
    # Reshape to (1, n_channels, window_size) for model input
    # Model expects (batch, channels, time) = (1, 3, 1125)
    x = torch.tensor(window_data, dtype=torch.float32).unsqueeze(0)  # (1, n_channels, window_size)
    x = x.to(device)
    
    with torch.no_grad():
        logits = model(x)
        probabilities = torch.softmax(logits, dim=1)
        confidence, prediction = torch.max(probabilities, dim=1)
    
    return prediction.item(), confidence.item(), probabilities[0].cpu().numpy()
