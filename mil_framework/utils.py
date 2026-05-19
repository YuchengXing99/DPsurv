"""
Shared utilities for the MIL training pipeline.

seed_torch   — reproducible seeding across Python / NumPy / PyTorch.
EarlyStopping — patience-based early stopping with checkpoint tracking.
"""

import random
import numpy as np
import torch


def seed_torch(seed: int = 1) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """
    Stop training when a monitored metric stops improving.

    Args:
        patience:    epochs to wait after last improvement
        min_epochs:  minimum epochs before stopping is allowed
        mode:        'min' (lower is better) or 'max' (higher is better)
        delta:       minimum change to qualify as improvement
    """

    def __init__(self, patience: int = 10, min_epochs: int = 5, mode: str = 'min', delta: float = 0.0):
        self.patience = patience
        self.min_epochs = min_epochs
        self.mode = mode
        self.delta = delta
        self.best = None
        self.counter = 0
        self.best_epoch = 0

    def __call__(self, metric: float, epoch: int) -> bool:
        """
        Returns True if training should stop.
        """
        if self.best is None:
            self.best = metric
            self.best_epoch = epoch
            return False

        if self.mode == 'min':
            improved = metric < self.best - self.delta
        else:
            improved = metric > self.best + self.delta

        if improved:
            self.best = metric
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1

        if epoch < self.min_epochs:
            return False
        return self.counter >= self.patience
