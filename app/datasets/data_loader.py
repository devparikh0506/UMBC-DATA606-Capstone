"""
EEG Data Loader for BCI Competition IV-2b Dataset

Loads and preprocesses EEG data from GDF files.
Only uses training sessions (T) - evaluation sessions (E) are excluded.
"""

import os
import re
import mne
import numpy as np
from typing import Dict, Optional, Tuple


class EEGDataLoader:
    """
    EEGDataLoader for BCI Competition IV-2b dataset.
    
    Only loads training sessions (01T, 02T, 03T).
    Evaluation sessions (04E, 05E) are excluded.
    """
    
    def __init__(self, 
                 data_folder='../data/BCICIV_2b_gdf', 
                 file_filters=None, 
                 raw_channels=['EEG:Cz', 'EEG:C3', 'EEG:C4'], 
                 channel_fun=lambda x: x.replace('EEG:', ''), 
                 filtered=True,
                 apply_standardization=True,
                 filter_range=(4, 40),
                 apply_notch=True,
                 notch_freq=50):
        """
        Initialize EEGDataLoader.
        
        Args:
            data_folder: Path to folder containing GDF files
            file_filters: Optional filters for file selection
            raw_channels: List of raw channel names
            channel_fun: Function to transform channel names
            filtered: Whether to apply filtering
            apply_standardization: Whether to apply electrode-wise standardization
            filter_range: Tuple (low_freq, high_freq) for bandpass filter
            apply_notch: Whether to apply notch filter
            notch_freq: Frequency for notch filter (default 50 Hz)
        """
        self.filename_pattern = r"^(B\d{2})(\d{2})([A-Za-z])\.gdf$"
        self.data_folder = data_folder
        self.file_filters = file_filters or {}
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
        self._standardization_stats = {}
    
    def _scan_data_dir_(self, filters=None):
        """Scan data directory and return file information."""
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
    
    def list_subjects(self):
        """List all available subjects."""
        return sorted(set(x["subject_id"] for x in self.data_dir_info))
    
    def list_sessions(self, subject_id):
        """List all available sessions for a subject (training only)."""
        sessions = sorted(set(x["session_id"] for x in self.data_dir_info 
                             if x["subject_id"] == subject_id and x["session_type"] == 'T'))
        return sessions
    
    def list_session_types(self, subject_id, session_id):
        """List session types for a subject and session (returns ['T'] only)."""
        types = sorted(set(x["session_type"] for x in self.data_dir_info
                          if x["subject_id"] == subject_id and x["session_id"] == session_id))
        return [t for t in types if t == 'T']
    
    def get_data(self, subject_id, session_id, session_type='T', reload=False):
        """
        Get data for a subject/session.
        
        Args:
            subject_id: Subject ID (e.g., 'B01')
            session_id: Session ID (e.g., '01')
            session_type: Session type (only 'T' supported)
            reload: Whether to reload from disk (ignore cache)
            
        Returns:
            Dictionary with 'raw', 'epochs', 'epochs_per_run', 'event_dict'
        """
        if session_type != 'T':
            raise ValueError("Only training sessions (T) are supported")
        
        key = (subject_id, session_id, session_type)
        if not reload and key in self._data_cache:
            return self._data_cache[key]
        
        data = self._load_eeg_data_(subject_id, session_id, session_type)
        self._data_cache[key] = data
        return data
    
    def _load_eeg_data_(self, subject_id, session_id, session_type):
        """
        Load and preprocess EEG data from GDF file.
        
        Only supports training sessions (T) with class labels (769, 770).
        """
        if session_type != 'T':
            return {'raw': None, 'epochs': None, 'epochs_per_run': None}
        
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
        
        if self.filter:
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
            raise ValueError(f"None of the requested channels {self.channels} found in {file_path}")
        
        raw.pick_channels(available_channels)
        picks = mne.pick_channels(raw.info["ch_names"], include=available_channels)
        
        events, event_dict = mne.events_from_annotations(raw)
        
        try:
            start_trial_code = event_dict['768']
            left_code = event_dict['769']
            right_code = event_dict['770']
            rejected_code = event_dict['1023']
            run_start_code = event_dict['32766']
        except KeyError:
            return {'raw': raw, 'epochs': None, 'epochs_per_run': None}
        
        trial_starts = events[events[:, 2] == start_trial_code][:, 0]
        rejected_starts = set(events[events[:, 2] == rejected_code][:, 0])
        mi_events = events[np.isin(events[:, 2], [left_code, right_code])]
        run_starts = events[events[:, 2] == run_start_code][:, 0]
        run_starts = np.append(run_starts, [raw.n_times])
        
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
        
        epochs = mne.Epochs(
            raw, remapped_events, standard_event_id,
            picks=picks, tmin=2.5, tmax=7.0,
            preload=True, baseline=None, verbose=False
        )
        
        epochs_per_run = {}
        for i in range(len(run_starts) - 1):
            run_start = run_starts[i]
            run_end = run_starts[i + 1]
            run_events = remapped_events[(remapped_events[:, 0] >= run_start) &
                                         (remapped_events[:, 0] < run_end)]
            if len(run_events) > 0:
                run_epochs = mne.Epochs(
                    raw, run_events, standard_event_id,
                    picks=picks, tmin=2.5, tmax=7.0,
                    preload=True, baseline=None, verbose=False
                )
                epochs_per_run[i + 1] = run_epochs
        
        return {
            'raw': raw,
            'epochs': epochs,
            'epochs_per_run': epochs_per_run,
            'event_dict': event_dict
        }
    
    def _standardize_data(self, X_train, X_test, subject_id=None, eps=1e-8):
        """Standardize data using training statistics only."""
        mean_per_channel = np.mean(X_train, axis=(0, 2), keepdims=True)
        std_per_channel = np.std(X_train, axis=(0, 2), keepdims=True)
        std_per_channel = np.maximum(std_per_channel, eps)
        
        X_train_std = (X_train - mean_per_channel) / std_per_channel
        X_test_std = (X_test - mean_per_channel) / std_per_channel
        
        stats = {
            'mean': mean_per_channel.squeeze(),
            'std': std_per_channel.squeeze()
        }
        
        if subject_id:
            self._standardization_stats[subject_id] = stats
        
        return X_train_std, X_test_std, stats

