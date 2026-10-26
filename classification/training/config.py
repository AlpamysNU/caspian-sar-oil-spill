"""
TrainerConfig dataclass for Stage 2 Multi-Class Classification Training Pipeline

This module defines the configuration contract for the training pipeline,
including model selection, loss configuration, hyperparameters, and paths.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional
import yaml


@dataclass
class TrainerConfig:
    """
    Configuration for the Stage 2 multi-class patch classification trainer.

    5 classes: Sea, Oil, Look-alike, Ship, Land
    Primary metric: Oil (Class 1) recall
    """

    # Model & Loss choices
    model_name: str = "effnet_b2"  # {effnet_b0, effnet_b2, convnext_tiny, resnet50, mobilenet_v3_large, swin_tiny}
    loss_type: str = "focal"        # {focal, gp, hybrid, bce}

    # Training hyperparameters
    epochs: int = 100
    batch_size: int = 256
    lr: float = 8e-4
    seed: int = 42
    num_workers: int = 4
    pin_memory: bool = True

    # Focal Loss parameters
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25

    # GP Loss (Guided Precision) parameters
    gp_margin: float = 0.1

    # Hybrid Loss weights
    focal_weight: float = 1.0
    gp_weight: float = 0.1
    ce_weight: float = 1.0

    # Directory paths (relative to the project root)
    parquet_path: str = "../OTHER_OIL_SPILL_DATASETS/OIL_SPILL_Segmentation/MULTICLASSIFICATION/m4d_stage2_ovr_index.parquet"
    patches_dir: str = "../OTHER_OIL_SPILL_DATASETS/OIL_SPILL_Segmentation/MULTICLASSIFICATION/patches_npy"
    weights_dir: str = "results/weights"
    runs_dir: str = "results/runs"
    run_name: str = "default_run"

    # System & calibration
    device: str = "cuda"
    calibrate_after: bool = True

    # Valid choices (class variables)
    VALID_MODELS = {"effnet_b0", "effnet_b2", "convnext_tiny", "resnet50", "mobilenet_v3_large", "swin_tiny", "effnet_b2_cbam", "effnet_b2_cbam_transformer"}
    VALID_LOSSES = {"focal", "gp", "hybrid", "bce"}

    def __post_init__(self):
        """Validate model and loss type choices."""
        if self.model_name not in self.VALID_MODELS:
            raise ValueError(
                f"Invalid model_name '{self.model_name}'. "
                f"Choose from: {self.VALID_MODELS}"
            )

        if self.loss_type not in self.VALID_LOSSES:
            raise ValueError(
                f"Invalid loss_type '{self.loss_type}'. "
                f"Choose from: {self.VALID_LOSSES}"
            )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TrainerConfig":
        """
        Load configuration from a YAML file.

        Args:
            yaml_path: Path to YAML config file (relative or absolute)

        Returns:
            TrainerConfig instance populated from YAML

        Raises:
            FileNotFoundError: If YAML file does not exist
            ValueError: If YAML contains invalid model_name or loss_type
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        # Filter to only TrainerConfig fields to avoid unexpected kwargs
        valid_fields = {
            "model_name", "loss_type", "epochs", "batch_size", "lr", "seed",
            "num_workers", "pin_memory", "focal_gamma", "focal_alpha",
            "gp_margin", "focal_weight", "gp_weight", "ce_weight",
            "parquet_path", "patches_dir", "weights_dir", "runs_dir", "run_name",
            "device", "calibrate_after"
        }

        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)

    def to_dict(self) -> dict:
        """
        Convert config to dictionary.

        Returns:
            Dictionary representation of the config (excludes class variables)
        """
        return asdict(self)

    def __repr__(self) -> str:
        """Return a readable string representation."""
        return (
            f"TrainerConfig(\n"
            f"  model_name={self.model_name!r},\n"
            f"  loss_type={self.loss_type!r},\n"
            f"  epochs={self.epochs},\n"
            f"  batch_size={self.batch_size},\n"
            f"  lr={self.lr},\n"
            f"  device={self.device!r}\n"
            f")"
        )
