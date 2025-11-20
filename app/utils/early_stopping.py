from __future__ import annotations
import copy
import math

class EarlyStopping:
    """
    monitor in {'val_loss','val_acc'}
    mode: 'min' for loss, 'max' for acc
    """
    def __init__(self, monitor='val_loss', mode='min', patience=50, min_delta=0.0):
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = math.inf if mode == 'min' else -math.inf
        self.best_epoch = 0
        self.best_state_dict = None
        self.best_metrics = None
        self.counter = 0
        self.early_stop = False

    def _is_better(self, current):
        if self.mode == 'min':
            return (self.best_score - current) > self.min_delta
        else:
            return (current - self.best_score) > self.min_delta

    def step(self, metrics: dict, model, epoch: int):
        current = metrics[self.monitor]
        if self._is_better(current):
            self.best_score = current
            self.best_epoch = epoch
            self.best_state_dict = copy.deepcopy(model.state_dict())
            self.best_metrics = metrics.copy()
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False
