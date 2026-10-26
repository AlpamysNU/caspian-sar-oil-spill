"""
Data loading and augmentation utilities (Stage 2 pipeline).

Handles M4D-SAR patches, train/val/test splits, and augmentation pipelines.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import warnings


class Stage2OvRDataset(Dataset):
    """PyTorch Dataset for Stage 2 multi-hot classification.

    Loads SAR patches and 5-class binary labels from parquet index.
    Supports data augmentation and normalization.

    Classes (multi-hot labels):
        0: Sea
        1: Oil
        2: Look-alike
        3: Ship
        4: Land
    """

    def __init__(self,
                 parquet_path: str,
                 patches_dir: str,
                 indices: Optional[np.ndarray] = None,
                 augment: bool = False,
                 normalize: bool = True):
        """Initialize dataset.

        Args:
            parquet_path: Path to m4d_stage2_ovr_index.parquet
            patches_dir: Path to directory containing patch_*.npy files
            indices: Optional subset of indices to use (for train/val/test splits)
            augment: If True, apply data augmentation (rotation, flip, etc.)
            normalize: If True, normalize patch to zero mean, unit variance

        Raises:
            FileNotFoundError: If parquet or patches_dir does not exist
        """
        parquet_path = Path(parquet_path)
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        patches_dir = Path(patches_dir)
        if not patches_dir.exists():
            raise FileNotFoundError(f"Patches directory not found: {patches_dir}")

        # Load parquet index
        self.df = pd.read_parquet(parquet_path)

        # Apply indices subset if provided (for train/val/test splits)
        if indices is not None:
            self.df = self.df.iloc[indices].reset_index(drop=True)

        self.patches_dir = patches_dir
        self.augment = augment
        self.normalize = normalize

        # Verify expected columns
        expected_cols = ['patch_id', 'label_sea', 'label_oil', 'label_look_alike', 'label_ship', 'label_land']
        missing_cols = [col for col in expected_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Parquet missing expected columns: {missing_cols}")

    def __len__(self) -> int:
        """Return number of patches."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (patch_tensor, labels_tensor).

        Returns:
            patch_tensor: shape (1, 256, 256) — SAR patch, 1 channel
            labels_tensor: shape (5,) — multi-hot binary labels [sea, oil, lookalike, ship, land]

        Raises:
            FileNotFoundError: If NPY file for patch does not exist
        """
        row = self.df.iloc[idx]
        patch_id = row['patch_id']

        # Construct path to NPY file
        # Patches may be in train/ or test/ subdirectories
        npy_path = self.patches_dir / f"{patch_id}.npy"

        if not npy_path.exists():
            # Try subdirectories (train/ or test/)
            for subdir in ['train', 'test']:
                npy_path_alt = self.patches_dir / subdir / f"{patch_id}.npy"
                if npy_path_alt.exists():
                    npy_path = npy_path_alt
                    break

        if not npy_path.exists():
            raise FileNotFoundError(
                f"NPY file not found for patch {patch_id} at {npy_path} "
                f"(searched in {self.patches_dir} and subdirectories)"
            )

        # Load patch (expected shape: (256, 256))
        patch = np.load(npy_path, allow_pickle=False)

        # Verify shape
        if patch.shape != (256, 256):
            warnings.warn(f"Patch {patch_id} has unexpected shape {patch.shape}, expected (256, 256)")

        # Convert to float32
        patch = patch.astype(np.float32)

        # Apply normalization if requested
        if self.normalize:
            mean = patch.mean()
            std = patch.std()
            patch = (patch - mean) / (std + 1e-7)

        # Apply augmentation if requested
        if self.augment:
            patch = self._augment(patch)

        # Add channel dimension: (256, 256) -> (1, 256, 256)
        patch = np.expand_dims(patch, axis=0)

        # Make a copy to ensure C-contiguous memory layout (avoids negative strides)
        patch = np.ascontiguousarray(patch)

        # Convert to torch tensor
        patch_tensor = torch.from_numpy(patch).float()

        # Extract multi-hot labels
        labels = np.array([
            row['label_sea'],
            row['label_oil'],
            row['label_look_alike'],
            row['label_ship'],
            row['label_land']
        ], dtype=np.int64)

        labels_tensor = torch.from_numpy(labels).long()

        return patch_tensor, labels_tensor

    def _augment(self, patch: np.ndarray) -> np.ndarray:
        """Apply random augmentation to patch.

        Args:
            patch: SAR patch array, shape (256, 256)

        Returns:
            Augmented patch, same shape
        """
        # Random rotation ±10 degrees
        if np.random.rand() > 0.5:
            k = np.random.choice([1, 3])  # 90 or 270 degrees (90*k)
            # Use np.rot90 which is more efficient
            # k=1: 90 deg, k=2: 180 deg, k=3: 270 deg
            if np.random.rand() > 0.5:
                patch = np.rot90(patch, k=k)

        # Random horizontal flip (50% probability)
        if np.random.rand() > 0.5:
            patch = np.fliplr(patch)

        # Random vertical flip (50% probability)
        if np.random.rand() > 0.5:
            patch = np.flipud(patch)

        # Add small Gaussian noise
        if np.random.rand() > 0.5:
            noise = np.random.normal(0, 0.01, patch.shape)
            patch = patch + noise

        return patch


def get_dataloaders(config,
                   train_ratio: float = 0.8,
                   val_ratio: float = 0.1,
                   test_ratio: float = 0.1,
                   seed: int = 42) -> Dict[str, DataLoader]:
    """Create train/val/test DataLoaders from parquet index and patches.

    Loads the parquet file, performs deterministic train/val/test split,
    and creates DataLoaders with appropriate settings.

    Args:
        config: TrainerConfig with parquet_path, patches_dir, batch_size, num_workers, pin_memory
        train_ratio: Fraction for training (default 0.8)
        val_ratio: Fraction for validation (default 0.1)
        test_ratio: Fraction for testing (default 0.1)
        seed: Random seed for reproducible split

    Returns:
        dict with keys: 'train', 'val', 'test'
        Each value is a DataLoader with appropriate shuffle and worker settings

    Raises:
        FileNotFoundError: If parquet or patches directory does not exist
        ValueError: If ratios don't sum to 1.0
    """
    # Validate ratios
    ratio_sum = train_ratio + val_ratio + test_ratio
    if not np.isclose(ratio_sum, 1.0, atol=1e-6):
        raise ValueError(f"Train/val/test ratios must sum to 1.0, got {ratio_sum}")

    # Load parquet to get total size
    parquet_path = Path(config.parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    n_total = len(df)

    # Set random seed for reproducibility
    np.random.seed(seed)

    # Create shuffled indices
    indices = np.arange(n_total)
    np.random.shuffle(indices)

    # Calculate split points
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    # Create datasets
    train_dataset = Stage2OvRDataset(
        parquet_path=str(config.parquet_path),
        patches_dir=str(config.patches_dir),
        indices=train_indices,
        augment=True,
        normalize=True
    )

    val_dataset = Stage2OvRDataset(
        parquet_path=str(config.parquet_path),
        patches_dir=str(config.patches_dir),
        indices=val_indices,
        augment=False,
        normalize=True
    )

    test_dataset = Stage2OvRDataset(
        parquet_path=str(config.parquet_path),
        patches_dir=str(config.patches_dir),
        indices=test_indices,
        augment=False,
        normalize=True
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
