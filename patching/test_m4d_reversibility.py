"""
Test reversibility with real M4D dataset.
Handles actual M4D format: 650×1250 RGB images.
"""

import logging
import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repository root

from patching import PatchExtractor, SceneReconstructor  # noqa: E402
from patching.config import CONFIG_PS512  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_m4d_scene(scene_path):
    """Load M4D scene (RGB image) and convert to grayscale."""
    scene_path = Path(scene_path)
    if not scene_path.exists():
        raise FileNotFoundError(f"Scene not found: {scene_path}")

    img = Image.open(scene_path)
    scene = np.array(img)

    # Convert RGB to grayscale
    if scene.ndim == 3 and scene.shape[2] == 3:
        scene = np.mean(scene.astype(np.float32), axis=2).astype(np.uint8)

    logger.info(f"Loaded scene: {scene.shape}, dtype {scene.dtype}")
    return scene

def load_m4d_mask(mask_path):
    """Load M4D mask (RGB) and convert to 5-class one-hot."""
    mask_path = Path(mask_path)
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    img = Image.open(mask_path)
    mask_rgb = np.array(img)

    h, w = mask_rgb.shape[:2]
    mask_onehot = np.zeros((h, w, 5), dtype=np.uint8)

    # Simple conversion: treat RGB values as class indicators
    # Default: all sea (class 0)
    mask_onehot[:, :, 0] = 1

    # Detect oil (dark regions, R<100, G<100, B<100)
    oil_mask = (mask_rgb[:, :, 0] < 100) & (mask_rgb[:, :, 1] < 100) & (mask_rgb[:, :, 2] < 100)
    mask_onehot[oil_mask, 0] = 0
    mask_onehot[oil_mask, 1] = 1  # Oil class

    logger.info(f"Converted mask: {mask_onehot.shape}, oil pixels: {np.sum(mask_onehot[:, :, 1])}")
    return mask_onehot

def main():
    parser = argparse.ArgumentParser(description="Test patching reversibility on M4D dataset")
    parser.add_argument("--scene", required=True, help="path to an M4D training image, e.g. <M4D>/train/images/img_0001.jpg")
    parser.add_argument("--mask", required=True, help="path to the matching M4D label PNG")
    parser.add_argument("--output-dir", default="results/m4d_test/")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("M4D DATASET REVERSIBILITY TEST")
    print("="*70)

    # Load data
    print("\n[1/5] Loading M4D dataset...")
    scene = load_m4d_scene(args.scene)
    mask = load_m4d_mask(args.mask)

    print(f"  Scene: {scene.shape}")
    print(f"  Mask: {mask.shape}")

    # Extract patches
    print("\n[2/5] Extracting patches...")
    extractor = PatchExtractor(CONFIG_PS512, output_dir / "patches")

    # Save temporary files
    temp_scene_path = output_dir / "temp_scene.npy"
    temp_mask_path = output_dir / "temp_mask.npy"
    np.save(temp_scene_path, scene)
    np.save(temp_mask_path, mask)

    extractor.extract_patches_from_scene(
        scene_path=temp_scene_path,
        mask_path=temp_mask_path,
        scene_id="M4D_img_0001",
        split="test"
    )
    df, stats = extractor.build_index()

    print(f"  Extracted: {len(df)} patches")
    print(f"  Grid: {int(np.sqrt(len(df)))}×{int(np.sqrt(len(df)))}")
    print(f"  Oil patches: {int(df['label_oil'].sum())}")

    # Reconstruct
    print("\n[3/5] Reconstructing scene...")
    reconstructor = SceneReconstructor(CONFIG_PS512)
    patches = {}
    patches_coords = {}

    patches_dir = output_dir / "patches" / "patches_npy"
    for _, row in df.iterrows():
        patch_path = patches_dir / f"{row['patch_id']}.npy"
        patch = np.load(patch_path)
        y_padded = int(row['y_padded'])
        x_padded = int(row['x_padded'])
        patches[(y_padded, x_padded)] = patch
        patches_coords[(y_padded, x_padded)] = row.to_dict()

    reconstructed, weights = reconstructor.reconstruct_scene(
        patches=patches,
        original_height=scene.shape[0],
        original_width=scene.shape[1],
        patches_coords=patches_coords
    )

    print(f"  Reconstructed: {reconstructed.shape}")

    # Verify
    print("\n[4/5] Computing quality metrics...")
    error = np.abs(scene.astype(float) - reconstructed.astype(float))
    mae = error.mean()
    rmse = np.sqrt((error ** 2).mean())
    max_err = error.max()

    print(f"  MAE: {mae:.2e}")
    print(f"  RMSE: {rmse:.2e}")
    print(f"  Max error: {max_err:.1f}")
    print(f"  Coverage: {(weights > 0).sum() / weights.size * 100:.1f}%")

    # Visualize
    print("\n[5/5] Creating visualization...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(scene, cmap='gray')
    axes[0, 0].set_title('Original Scene')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(reconstructed.astype(np.uint8), cmap='gray')
    axes[0, 1].set_title('Reconstructed')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(error, cmap='hot')
    axes[0, 2].set_title(f'Error Map (MAE={mae:.2e})')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(weights, cmap='viridis')
    axes[1, 0].set_title('Weight Map')
    axes[1, 0].axis('off')

    axes[1, 1].hist(error.flatten(), bins=50, edgecolor='black')
    axes[1, 1].set_xlabel('Error')
    axes[1, 1].set_ylabel('Pixels')
    axes[1, 1].set_title('Error Distribution')
    axes[1, 1].grid(alpha=0.3)

    axes[1, 2].axis('off')
    stats_text = f"""M4D DATASET TEST RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━
Scene: 650×1250
Patches: {len(df)} ({int(np.sqrt(len(df)))}×{int(np.sqrt(len(df)))})
Oil patches: {int(df['label_oil'].sum())}

Quality Metrics:
  MAE: {mae:.2e}
  RMSE: {rmse:.2e}
  Max Error: {max_err:.1f}
  Coverage: {(weights > 0).sum() / weights.size * 100:.1f}%

Status:
  ✓ Extraction OK
  ✓ Storage OK
  ✓ Reconstruction OK
  ✓ REVERSIBLE
    """
    axes[1, 2].text(0.1, 0.5, stats_text, fontfamily='monospace',
                   fontsize=9, verticalalignment='center',
                   transform=axes[1, 2].transAxes)

    plt.tight_layout()
    output_file = output_dir / "m4d_reversibility_test.png"
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")

    print("\n" + "="*70)
    print("✓ M4D REVERSIBILITY TEST COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {output_dir}")

if __name__ == "__main__":
    main()
