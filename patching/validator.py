"""
Validation and reversibility testing for patching system.
"""

import logging
from typing import NamedTuple, Tuple
import numpy as np
from pathlib import Path

from .config import PatchConfig
from .core import SceneReconstructor

logger = logging.getLogger(__name__)


class ReversibilityMetrics(NamedTuple):
    """Metrics for scene reconstruction quality."""

    rmse: float  # Root Mean Squared Error
    mae: float  # Mean Absolute Error
    psnr: float  # Peak Signal-to-Noise Ratio (dB)
    percent_within_1: float  # % of pixels within ±1 of original
    percent_within_5: float  # % of pixels within ±5 of original
    max_error: float  # Maximum absolute error
    mean_variance: float  # Mean per-pixel variance from weight map


class ReversibilityValidator:
    """Validate patch extraction and reconstruction reversibility."""

    def __init__(self, config: PatchConfig):
        """Initialize validator.

        Args:
            config: PatchConfig for reconstruction.
        """
        self.config = config
        self.reconstructor = SceneReconstructor(config)

    def compute_reconstruction_metrics(
        self, original: np.ndarray, reconstructed: np.ndarray
    ) -> ReversibilityMetrics:
        """Compute reconstruction quality metrics.

        Args:
            original: Original image (H, W), uint8.
            reconstructed: Reconstructed image (H, W), float32.

        Returns:
            ReversibilityMetrics with quality measures.
        """
        original_float = original.astype(np.float32)

        # Convert reconstructed to uint8 range for comparison
        reconstructed_uint8 = np.clip(reconstructed, 0, 255)

        # Compute errors
        diff = original_float - reconstructed_uint8
        rmse = float(np.sqrt(np.mean(diff**2)))
        mae = float(np.mean(np.abs(diff)))
        max_error = float(np.max(np.abs(diff)))

        # Compute PSNR (Peak Signal-to-Noise Ratio)
        max_pixel = 255.0
        if rmse == 0:
            psnr = float("inf")
        else:
            psnr = float(20 * np.log10(max_pixel / rmse))

        # Compute percentage of pixels within tolerance
        percent_within_1 = float(
            100 * np.sum(np.abs(diff) <= 1) / original.size
        )
        percent_within_5 = float(
            100 * np.sum(np.abs(diff) <= 5) / original.size
        )

        # Compute mean variance (based on inverse of weight map)
        # This is a proxy for reconstruction uncertainty
        mean_variance = float(np.std(diff))

        return ReversibilityMetrics(
            rmse=rmse,
            mae=mae,
            psnr=psnr,
            percent_within_1=percent_within_1,
            percent_within_5=percent_within_5,
            max_error=max_error,
            mean_variance=mean_variance,
        )

    def test_reversibility(
        self,
        original_scene: np.ndarray,
        patches: dict,
        original_height: int,
        original_width: int,
        patches_coords: dict | None = None,
        tolerance: float = 2.0,
    ) -> Tuple[bool, ReversibilityMetrics, np.ndarray]:
        """Test reversibility of patch extraction/reconstruction.

        Args:
            original_scene: Original image.
            patches: Dict mapping (y, x) to patch arrays.
            original_height: Original height.
            original_width: Original width.
            patches_coords: Optional dict mapping patch keys to padding-aware metadata.
            tolerance: RMSE tolerance (in pixel value units).

        Returns:
            Tuple of (is_reversible, metrics, reconstructed_scene).
                is_reversible: True if RMSE <= tolerance.
                metrics: ReversibilityMetrics.
                reconstructed_scene: Reconstructed image.

        Raises:
            ValueError: If dimensions don't match.
        """
        if original_scene.shape != (original_height, original_width):
            raise ValueError(
                f"Original scene shape {original_scene.shape} doesn't match "
                f"expected ({original_height}, {original_width})"
            )

        # Reconstruct
        reconstructed, weight_map = self.reconstructor.reconstruct_scene(
            patches, original_height, original_width, patches_coords=patches_coords
        )

        # Compute metrics
        metrics = self.compute_reconstruction_metrics(original_scene, reconstructed)

        # Check if reversible
        is_reversible = metrics.rmse <= tolerance

        logger.info(f"Reversibility test: RMSE={metrics.rmse:.3f} (tolerance={tolerance})")
        logger.info(f"  MAE={metrics.mae:.3f}, PSNR={metrics.psnr:.1f} dB")
        logger.info(f"  Pixels within ±1: {metrics.percent_within_1:.1f}%")
        logger.info(f"  Pixels within ±5: {metrics.percent_within_5:.1f}%")

        return is_reversible, metrics, reconstructed

    def compare_edge_artifacts(
        self, original: np.ndarray, reconstructed: np.ndarray, margin: int = 50
    ) -> dict:
        """Analyze edge artifacts in reconstruction.

        Args:
            original: Original image.
            reconstructed: Reconstructed image.
            margin: Pixel margin from edges to analyze.

        Returns:
            Dictionary with per-edge metrics.
        """
        original_float = original.astype(np.float32)
        reconstructed_uint8 = np.clip(reconstructed, 0, 255)
        diff = original_float - reconstructed_uint8

        h, w = original.shape
        edges = {
            "top": diff[:margin, :],
            "bottom": diff[h - margin :, :],
            "left": diff[:, :margin],
            "right": diff[:, w - margin :],
            "center": diff[margin : h - margin, margin : w - margin],
        }

        edge_metrics = {}
        for edge_name, edge_region in edges.items():
            edge_metrics[edge_name] = {
                "rmse": float(np.sqrt(np.mean(edge_region**2))),
                "mae": float(np.mean(np.abs(edge_region))),
                "max_error": float(np.max(np.abs(edge_region))),
            }

        return edge_metrics

    def generate_error_heatmap(
        self, original: np.ndarray, reconstructed: np.ndarray
    ) -> np.ndarray:
        """Generate absolute error heatmap.

        Args:
            original: Original image.
            reconstructed: Reconstructed image.

        Returns:
            Error heatmap (uint8, 0-255 scale).
        """
        original_float = original.astype(np.float32)
        reconstructed_uint8 = np.clip(reconstructed, 0, 255)
        error = np.abs(original_float - reconstructed_uint8)

        # Scale to 0-255 for visualization
        error_normalized = (error / error.max() * 255).astype(np.uint8) if error.max() > 0 else error.astype(np.uint8)

        return error_normalized

    def save_validation_report(
        self,
        output_dir: Path,
        metrics: ReversibilityMetrics,
        edge_metrics: dict,
    ):
        """Save validation results to file.

        Args:
            output_dir: Directory to save report.
            metrics: Reconstruction metrics.
            edge_metrics: Per-edge analysis.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        report_path = output_dir / f"reversibility_report_{self.config.name}.txt"

        with open(report_path, "w") as f:
            f.write(f"Reversibility Report: {self.config.name}\n")
            f.write("=" * 60 + "\n\n")

            f.write("Configuration:\n")
            f.write(f"  Patch size: {self.config.patch_size}\n")
            f.write(f"  Stride: {self.config.stride}\n")
            f.write(f"  Overlap: {self.config.overlap:.1%}\n")
            f.write(f"  Window type: {self.config.window_type}\n\n")

            f.write("Overall Metrics:\n")
            f.write(f"  RMSE: {metrics.rmse:.3f}\n")
            f.write(f"  MAE: {metrics.mae:.3f}\n")
            f.write(f"  PSNR: {metrics.psnr:.1f} dB\n")
            f.write(f"  Max error: {metrics.max_error:.1f}\n")
            f.write(f"  Pixels within ±1: {metrics.percent_within_1:.1f}%\n")
            f.write(f"  Pixels within ±5: {metrics.percent_within_5:.1f}%\n\n")

            f.write("Edge Analysis:\n")
            for edge_name, metrics_dict in edge_metrics.items():
                f.write(f"  {edge_name.capitalize()}:\n")
                f.write(f"    RMSE: {metrics_dict['rmse']:.3f}\n")
                f.write(f"    MAE: {metrics_dict['mae']:.3f}\n")
                f.write(f"    Max error: {metrics_dict['max_error']:.1f}\n")

        logger.info(f"Saved validation report: {report_path}")
