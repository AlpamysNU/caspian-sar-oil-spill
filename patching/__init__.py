"""
Patching system for M4D-SAR-OilSpill dataset.

Provides reversible patch extraction, scene reconstruction, and validation.
"""

from .config import PatchConfig
from .core import PatchExtractor, SceneReconstructor
from .validator import ReversibilityValidator

__all__ = [
    "PatchConfig",
    "PatchExtractor",
    "SceneReconstructor",
    "ReversibilityValidator",
]
