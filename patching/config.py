"""
Configuration for patch extraction and reconstruction.
"""

from dataclasses import dataclass
from typing import Literal
import yaml
from pathlib import Path


@dataclass
class PatchConfig:
    """Configuration for patch extraction and reconstruction.

    Attributes:
        patch_size: Patch width/height in pixels (256, 384, or 512).
        stride: Stride between patch centers in pixels.
        overlap: Overlap ratio between 0 and 1 (calculated as 1 - stride/patch_size).
        window_type: Type of windowing function for reconstruction ("hann" or "uniform").
        pad_mode: Padding mode for border patches ("reflect", "edge", or "constant").
        skip_border: If True, don't extract patches that extend beyond scene bounds.
        name: Unique identifier for this configuration (e.g., "ps512_overlap50").
    """

    patch_size: int
    stride: int
    overlap: float
    window_type: Literal["hann", "uniform"] = "hann"
    pad_mode: Literal["reflect", "edge", "constant"] = "reflect"
    skip_border: bool = False
    name: str = ""

    def __post_init__(self):
        """Validate configuration and compute name if not provided."""
        # Validate patch size
        if self.patch_size not in (256, 384, 512):
            raise ValueError(f"patch_size must be 256, 384, or 512, got {self.patch_size}")

        # Validate overlap consistency
        if not 1 <= self.stride <= self.patch_size:
            raise ValueError('stride must be between 1 and patch_size')
        computed_overlap = 1.0 - (self.stride / self.patch_size)
        expected_overlap = round(computed_overlap, 2)
        actual_overlap = round(self.overlap, 2)
        if abs(expected_overlap - actual_overlap) > 0.01:
            raise ValueError(
                f"Inconsistent stride/overlap: stride={self.stride} implies "
                f"overlap={expected_overlap:.2f}, but got {self.overlap:.2f}"
            )

        # Validate window type
        if self.window_type not in ("hann", "uniform"):
            raise ValueError(f"window_type must be 'hann' or 'uniform', got {self.window_type}")

        # Auto-generate name if not provided
        if not self.name:
            overlap_pct = int(self.overlap * 100)
            self.name = f"ps{self.patch_size}_overlap{overlap_pct}"

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "patch_size": self.patch_size,
            "stride": self.stride,
            "overlap": self.overlap,
            "window_type": self.window_type,
            "pad_mode": self.pad_mode,
            "skip_border": self.skip_border,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PatchConfig":
        """Create from dictionary."""
        return cls(**d)


def load_patching_configs(yaml_path: str) -> dict:
    """Load patch configurations from YAML file.

    Args:
        yaml_path: Path to patching.yaml.

    Returns:
        Dictionary mapping config names to PatchConfig objects.

    Raises:
        FileNotFoundError: If YAML file does not exist.
        ValueError: If configuration is invalid.
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    configs = {}
    for cfg_name, cfg_data in data.get("patch_configs", {}).items():
        try:
            config = PatchConfig(
                patch_size=cfg_data["patch_size"],
                stride=cfg_data["stride"],
                overlap=cfg_data["overlap"],
                window_type=cfg_data.get("window_type", "hann"),
                name=cfg_name,
            )
            configs[cfg_name] = config
        except (KeyError, ValueError) as e:
            raise ValueError(f"Invalid config '{cfg_name}': {e}")

    return configs


# Preset configurations (for convenience)
CONFIG_PS256 = PatchConfig(
    patch_size=256,
    stride=128,
    overlap=0.5,
    window_type="hann",
    pad_mode="reflect",
    skip_border=False,
    name="ps256_overlap50",
)

CONFIG_PS384 = PatchConfig(
    patch_size=384,
    stride=192,
    overlap=0.5,
    window_type="hann",
    pad_mode="reflect",
    skip_border=False,
    name="ps384_overlap50",
)

CONFIG_PS512 = PatchConfig(
    patch_size=512,
    stride=256,
    overlap=0.5,
    window_type="hann",
    pad_mode="reflect",
    skip_border=False,
    name="ps512_overlap50",
)

CONFIG_PS512_DENSE = PatchConfig(
    patch_size=512,
    stride=128,
    overlap=0.75,
    window_type="hann",
    pad_mode="reflect",
    skip_border=False,
    name="ps512_overlap75",
)

ALL_CONFIGS = {
    "ps256_overlap50": CONFIG_PS256,
    "ps384_overlap50": CONFIG_PS384,
    "ps512_overlap50": CONFIG_PS512,
    "ps512_overlap75": CONFIG_PS512_DENSE,
}
