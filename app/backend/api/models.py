from django.db import models
import numpy as np
import json


class Subject(models.Model):
    """Subject model to store subject information."""
    subject_id = models.CharField(max_length=10, unique=True, primary_key=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['subject_id']
    
    def __str__(self):
        return self.subject_id


class EvaluationRun(models.Model):
    """
    Run model to store run metadata (training runs only).
    
    Supports:
    - Training runs: 01T, 02T, 03T
    """
    subject = models.ForeignKey(Subject, on_delete=models.CASCADE, related_name='runs')
    run_id = models.CharField(max_length=10)  # e.g., "01T", "02T", "03T"
    session_id = models.CharField(max_length=10)  # e.g., "01", "02", "03"
    session_type = models.CharField(max_length=1, default='T')  # Always 'T' for training
    filename = models.CharField(max_length=100)
    n_trials = models.IntegerField()
    n_channels = models.IntegerField()
    n_times = models.IntegerField()
    sampling_rate = models.FloatField()
    duration_per_trial = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['subject', 'run_id']
        ordering = ['subject', 'session_id']
    
    def __str__(self):
        return f"{self.subject.subject_id} - {self.run_id}"
    
    @property
    def is_training(self):
        """Check if this is a training run."""
        return self.session_type == 'T'


class TrialData(models.Model):
    """Trial data model to store preprocessed EEG trial data."""
    run = models.ForeignKey(EvaluationRun, on_delete=models.CASCADE, related_name='trials')
    trial_index = models.IntegerField()  # Index within the run
    ground_truth = models.IntegerField()  # 0 for left, 1 for right
    # Store data as JSON (for small datasets) or reference to file
    # For large datasets, consider storing as binary or using a file storage system
    data_json = models.TextField()  # JSON representation of (n_channels, n_times) array
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['run', 'trial_index']
        ordering = ['run', 'trial_index']
        indexes = [
            models.Index(fields=['run', 'trial_index']),
        ]
    
    def __str__(self):
        return f"{self.run} - Trial {self.trial_index}"
    
    def get_data(self):
        """Get trial data as numpy array."""
        data_list = json.loads(self.data_json)
        return np.array(data_list)
    
    def set_data(self, data_array):
        """Set trial data from numpy array."""
        self.data_json = json.dumps(data_array.tolist())

