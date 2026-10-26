"""
Stage 2 Multi-Class Classification Training Pipeline

This package provides:
- Modular loss functions (Focal, GP, Hybrid)
- Configurable trainer with calibration support
- Data loading and augmentation
- Result comparison and aggregation
"""

__version__ = "1.0.0"
__author__ = "Alpamys Pentayev"

from .config import TrainerConfig
from .losses import FocalLoss, GPLoss, HybridLoss
from .models import build_model
from .trainer import Trainer
from .data import get_dataloaders

__all__ = [
    "TrainerConfig",
    "FocalLoss",
    "GPLoss",
    "HybridLoss",
    "build_model",
    "Trainer",
    "get_dataloaders",
]
