"""
Example: Test reversibility of patching system on a single M4D scene.

This script demonstrates:
1. Loading a raw M4D scene
2. Extracting patches with different sizes (256, 384, 512)
3. Reconstructing scenes from patches
4. Computing reconstruction metrics
5. Validating reversibility

Usage:
    python example_reversibility_test.py \
        --scene-path <path/to/scene.png> \
        --mask-path <path/to/mask.png> \
        --output-dir <output/dir> \
        --all-configs
"""

import argparse
import logging
from pathlib import Path
import numpy as np
from typing import Dict, Tuple
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from patching import (
    PatchConfig,
    PatchExtractor,
    SceneReconstructor,
    ReversibilityValidator,
)
from patching.config import ALL_CONFIGS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_scene(scene_path: Path) -> np.ndarray:
    """Load a scene image from file.

    Supports PNG, NPY, etc.
    """
    scene_path = Path(scene_path)
    if not scene_path.exists():
        raise FileNotFoundError(f"Scene file not found: {scene_path}")

    if scene_path.suffix == ".npy":
        scene = np.load(scene_path)
    else:
        from PIL import Image

        scene = np.array(Image.open(scene_path))

    # Convert to grayscale if needed
    if scene.ndim == 3:
        scene = scene[0]

    logger.info(f"Loaded scene: shape {scene.shape}, dtype {scene.dtype}")
    return scene


def load_mask(mask_path: Path) -> np.ndarray:
    """Load a segmentation mask.

    Expected: 5-channel (HxWx5) one-hot encoding.
    """
    mask_path = Path(mask_path)
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    if mask_path.suffix == ".npy":
        mask = np.load(mask_path)
    else:
        from PIL import Image

        mask = np.array(Image.open(mask_path))

    logger.info(f"Loaded mask: shape {mask.shape}, dtype {mask.dtype}")

    # Ensure 5-channel format
    if mask.ndim == 2:
        raise ValueError(f"Mask must be 3D (HxWx5), got {mask.shape}")
    if mask.shape[2] != 5:
        logger.warning(
            f"Mask has {mask.shape[2]} channels, expected 5. "
            f"Will use first 5 channels."
        )
        mask = mask[:, :, :5]

    return mask


def extract_and_validate(
    config: PatchConfig,
    scene: np.ndarray,
    mask: np.ndarray,
    scene_id: str,
    output_dir: Path,
) -> Tuple[bool, dict]:
    """Extract patches and validate reversibility.

    Args:
        config: Patch configuration
        scene: Original scene array
        mask: Segmentation mask
        scene_id: Scene identifier
        output_dir: Output directory

    Returns:
        Tuple of (is_reversible, metrics_dict)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing config: {config.name}")
    logger.info(f"{'='*60}")

    # Create extractor
    config_output_dir = output_dir / config.name
    extractor = PatchExtractor(config, config_output_dir)

    # Extract real patches via the production extractor so padding/border logic is tested.
    logger.info("Extracting patches...")
    scene_path = output_dir / f"{scene_id}_scene.npy"
    mask_path = output_dir / f"{scene_id}_mask.npy"
    np.save(scene_path, scene)
    np.save(mask_path, mask)

    extracted_metadata = extractor.extract_patches_from_scene(
        scene_path=scene_path,
        mask_path=mask_path,
        scene_id=scene_id,
        split="test",
        show_progress=False,
    )

    patches = {}
    patches_coords = {}
    for metadata in extracted_metadata:
        patch = np.load(extractor.patches_dir / f"{metadata.patch_id}.npy")
        key = (metadata.y_orig, metadata.x_orig)
        patches[key] = patch
        patches_coords[key] = metadata

    logger.info(f"Extracted {len(patches)} patches")

    # Reconstruct
    logger.info("Reconstructing scene...")
    reconstructor = SceneReconstructor(config)
    reconstructed, weight_map = reconstructor.reconstruct_scene(
        patches, scene.shape[0], scene.shape[1], patches_coords=patches_coords
    )

    # Validate
    logger.info("Computing reconstruction metrics...")
    validator = ReversibilityValidator(config)
    is_reversible, metrics, _ = validator.test_reversibility(
        scene,
        patches,
        scene.shape[0],
        scene.shape[1],
        patches_coords=patches_coords,
        tolerance=2.0,
    )

    # Edge analysis
    edge_metrics = validator.compare_edge_artifacts(scene, reconstructed, margin=50)

    # Save report
    validator.save_validation_report(
        output_dir, metrics, edge_metrics
    )

    # Compile results
    results = {
        "config_name": config.name,
        "is_reversible": is_reversible,
        "metrics": metrics._asdict(),
        "edge_metrics": edge_metrics,
        "num_patches": len(patches),
    }

    return is_reversible, results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test patching system reversibility on M4D scene"
    )
    parser.add_argument(
        "--scene-path",
        type=Path,
        required=True,
        help="Path to raw SAR scene image",
    )
    parser.add_argument(
        "--mask-path",
        type=Path,
        required=True,
        help="Path to segmentation mask (5-channel)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./test_reversibility_output"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--config",
        type=str,
        choices=list(ALL_CONFIGS.keys()),
        help="Single config to test (default: test all)",
    )
    parser.add_argument(
        "--all-configs",
        action="store_true",
        help="Test all configurations (256, 384, 512, 512_dense)",
    )
    parser.add_argument(
        "--save-error-maps",
        action="store_true",
        help="Save error heatmaps as PNG",
    )

    args = parser.parse_args()

    # Load scene and mask
    logger.info(f"Loading scene from {args.scene_path}")
    scene = load_scene(args.scene_path)

    logger.info(f"Loading mask from {args.mask_path}")
    mask = load_mask(args.mask_path)

    # Validate scene/mask
    if scene.shape[:2] != mask.shape[:2]:
        raise ValueError(
            f"Scene shape {scene.shape} doesn't match mask shape {mask.shape[:2]}"
        )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Determine which configs to test
    configs_to_test = {}
    if args.config:
        configs_to_test = {args.config: ALL_CONFIGS[args.config]}
    elif args.all_configs:
        configs_to_test = ALL_CONFIGS
    else:
        # Default: test all
        configs_to_test = ALL_CONFIGS

    # Run tests
    results_list = []
    for cfg_name, config in configs_to_test.items():
        is_reversible, results = extract_and_validate(
            config, scene, mask, "test_scene", output_dir
        )
        results_list.append(results)

    # Summary report
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}\n")

    all_reversible = True
    for results in results_list:
        config_name = results["config_name"]
        is_reversible = results["is_reversible"]
        metrics = results["metrics"]
        status = "✓ PASS" if is_reversible else "✗ FAIL"

        logger.info(f"{status} | {config_name}")
        logger.info(f"  RMSE: {metrics['rmse']:.3f} | PSNR: {metrics['psnr']:.1f} dB")
        logger.info(
            f"  Pixels within ±1: {metrics['percent_within_1']:.1f}% | "
            f"within ±5: {metrics['percent_within_5']:.1f}%"
        )
        logger.info(f"  Patches: {results['num_patches']} | Max error: {metrics['max_error']:.1f}")
        logger.info("")

        if not is_reversible:
            all_reversible = False

    # Final status
    logger.info(f"{'='*60}")
    if all_reversible:
        logger.info("✓ All configurations PASS reversibility test")
        return 0
    else:
        logger.warning("✗ Some configurations FAILED reversibility test")
        return 1


if __name__ == "__main__":
    exit(main())
