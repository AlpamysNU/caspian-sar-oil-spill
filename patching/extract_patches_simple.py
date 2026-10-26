#!/usr/bin/env python3
"""
Simple script to extract patches from a JPG image.
Usage: python extract_patches_simple.py --image image.jpg --output patches_dir/
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import pandas as pd
import json

# Import using absolute imports
try:
    from patching.core import PatchExtractor
    from patching.config import CONFIG_PS512, PatchConfig
except ImportError:
    # If running as script
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from patching.core import PatchExtractor
    from patching.config import CONFIG_PS512, PatchConfig


M4D_RGB_COLORS = {
    "sea": np.array([0, 0, 0], dtype=np.uint8),
    "oil": np.array([0, 255, 255], dtype=np.uint8),
    "look_alike": np.array([255, 0, 0], dtype=np.uint8),
    "ship": np.array([128, 64, 0], dtype=np.uint8),
    "land": np.array([0, 153, 0], dtype=np.uint8),
}


def is_oil(rgb):
    return np.all(rgb == M4D_RGB_COLORS["oil"], axis=-1)


def is_look_alike(rgb):
    return np.all(rgb == M4D_RGB_COLORS["look_alike"], axis=-1)


def is_land(rgb):
    return np.all(rgb == M4D_RGB_COLORS["land"], axis=-1)


def is_sea(rgb):
    return np.all(rgb == M4D_RGB_COLORS["sea"], axis=-1)


def is_ship(rgb):
    """Robust brown-color rule matching the reference step1.py."""
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    return (
        (r >= 60) & (r <= 170) &
        (g >= 30) & (g <= 100) &
        (b <= 15)
    )


def remap_rgb_mask_to_one_hot(mask_rgb):
    """Convert an M4D RGB label image to 5-channel one-hot mask."""
    h, w = mask_rgb.shape[:2]
    mask = np.zeros((h, w, 5), dtype=np.uint8)

    sea = is_sea(mask_rgb)
    oil = is_oil(mask_rgb)
    look_alike = is_look_alike(mask_rgb)
    ship = is_ship(mask_rgb)
    land = is_land(mask_rgb)

    mask[sea, 0] = 1
    mask[oil, 1] = 1
    mask[look_alike, 2] = 1
    mask[ship, 3] = 1
    mask[land, 4] = 1

    unlabeled = ~(sea | oil | look_alike | ship | land)
    if np.any(unlabeled):
        mask[unlabeled, 0] = 1

    return mask


def one_hot_mask_to_rgb(mask_patch):
    """Render a 5-channel one-hot mask using the reference M4D palette."""
    h, w = mask_patch.shape[:2]
    mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    mask_rgb[:] = M4D_RGB_COLORS["sea"]

    mask_rgb[mask_patch[:, :, 1] > 0] = M4D_RGB_COLORS["oil"]
    mask_rgb[mask_patch[:, :, 2] > 0] = M4D_RGB_COLORS["look_alike"]
    mask_rgb[mask_patch[:, :, 3] > 0] = M4D_RGB_COLORS["ship"]
    mask_rgb[mask_patch[:, :, 4] > 0] = M4D_RGB_COLORS["land"]
    return mask_rgb

def jpg_to_grayscale(jpg_path):
    """Load JPG and convert to grayscale."""
    img = Image.open(jpg_path)
    arr = np.array(img)

    # If RGB, convert to grayscale
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = np.mean(arr.astype(np.float32), axis=2).astype(np.uint8)

    return arr

def save_patches_as_png(patches_npy_dir, output_png_dir):
    """Save all NPY patches as PNG images for visual inspection."""
    output_png_dir = Path(output_png_dir)
    output_png_dir.mkdir(exist_ok=True)

    patch_files = sorted(output_png_dir.parent.glob("patches_npy/*.npy"))

    for patch_path in patch_files:
        patch_data = np.load(patch_path)
        patch_name = patch_path.stem
        png_path = output_png_dir / f"{patch_name}.png"

        # Save as PNG
        Image.fromarray(patch_data.astype(np.uint8)).save(png_path)

    return len(patch_files)

def create_patch_preview_grid(patches_npy_dir, output_file):
    """Create a grid preview of all patches."""
    import matplotlib.pyplot as plt

    patch_files = sorted(Path(patches_npy_dir).glob("*.npy"))

    if not patch_files:
        return False

    # Determine grid size
    num_patches = len(patch_files)
    grid_cols = int(np.ceil(np.sqrt(num_patches)))
    grid_rows = int(np.ceil(num_patches / grid_cols))

    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(15, 15))
    axes = axes.flatten() if num_patches > 1 else [axes]

    for idx, patch_path in enumerate(patch_files):
        patch_data = np.load(patch_path)
        axes[idx].imshow(patch_data, cmap='gray')
        axes[idx].set_title(patch_path.stem, fontsize=8)
        axes[idx].axis('off')

    # Hide unused subplots
    for idx in range(num_patches, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    plt.close()

    return True

def load_mask_from_dataset(image_path, dataset_labels_dir=None):
    """Load mask from M4D dataset or create dummy if not found."""
    image_name = Path(image_path).stem

    # Try to find mask in dataset
    if dataset_labels_dir is None:
        # Auto-detect dataset path from image path
        if 'oil-spill-detection-dataset' in str(image_path):
            dataset_labels_dir = Path(image_path).parent.parent / 'labels'
        else:
            dataset_labels_dir = None

    if dataset_labels_dir:
        mask_path = Path(dataset_labels_dir) / f"{image_name}.png"
        if mask_path.exists():
            print(f"  ✓ Loading mask from dataset: {mask_path.name}")
            mask_rgb = np.array(Image.open(mask_path))
            return remap_rgb_mask_to_one_hot(mask_rgb)

    # Fallback: create dummy mask
    print(f"  ⚠️  No mask found for {image_name}, creating dummy (all sea)")
    h, w = 650, 1250  # Default M4D size
    mask = np.zeros((h, w, 5), dtype=np.uint8)
    mask[:, :, 0] = 1  # All pixels are "sea" (class 0)
    return mask

def main():
    parser = argparse.ArgumentParser(description="Extract patches from JPG image")
    parser.add_argument("--image", required=True, help="Path to input JPG image")
    parser.add_argument("--output", required=True, help="Output directory for patches")
    parser.add_argument("--patch-size", type=int, default=512, help="Patch size (256/384/512)")
    parser.add_argument("--stride", type=int, default=256, help="Stride between patches")
    parser.add_argument("--save-png", action="store_true", help="Also save patches as PNG images")
    parser.add_argument("--save-preview", action="store_true", help="Create preview grid of all patches")
    args = parser.parse_args()

    image_path = Path(args.image)
    output_dir = Path(args.output)

    if not image_path.exists():
        print(f"❌ Error: Image not found: {image_path}")
        return 1

    print("\n" + "="*70)
    print("PATCH EXTRACTION")
    print("="*70)

    # Load image
    print(f"\n[1/4] Loading image: {image_path.name}")
    try:
        scene = jpg_to_grayscale(image_path)
        print(f"  ✓ Loaded: {scene.shape} (H×W)")
        print(f"  Range: {scene.min()}-{scene.max()}")
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return 1

    # Load mask from dataset
    print(f"\n[2/4] Loading segmentation mask...")
    mask = load_mask_from_dataset(image_path)
    print(f"  ✓ Loaded: {mask.shape} (H×W×5 classes)")

    # Create configuration
    config = CONFIG_PS512
    if args.patch_size != 512 or args.stride != 256:
        config = PatchConfig(
            patch_size=args.patch_size,
            stride=args.stride,
            overlap=1.0 - (args.stride / args.patch_size),
            name=f"ps{args.patch_size}_s{args.stride}"
        )

    print(f"\n[3/4] Extracting patches...")
    print(f"  Config: {config.name}")
    print(f"  Patch size: {config.patch_size}×{config.patch_size}")
    print(f"  Stride: {config.stride}")

    # Extract patches
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save temporary files
    temp_scene_path = output_dir / "temp_scene.npy"
    temp_mask_path = output_dir / "temp_mask.npy"
    np.save(temp_scene_path, scene)
    np.save(temp_mask_path, mask)

    try:
        extractor = PatchExtractor(config, output_dir)
        extractor.extract_patches_from_scene(
            scene_path=temp_scene_path,
            mask_path=temp_mask_path,
            scene_id=image_path.stem,
            split="test",
            show_progress=True
        )
        df, stats = extractor.build_index()

        # Также сохранить патчи маски
        if args.save_png:
            print(f"\n[BONUS] Saving mask patches as PNG...")
            mask_png_dir = output_dir / "mask_patches_png"
            mask_png_dir.mkdir(exist_ok=True)

            # Загрузить маску
            try:
                mask_full = np.load(temp_mask_path)

                # Для каждого патча извлечь его из маски
                for _, row in df.iterrows():
                    patch_id = row['patch_id']
                    y_padded = int(row['y_padded'])
                    x_padded = int(row['x_padded'])
                    ps = int(row['patch_size'])

                    # Проверка границ для безопасности
                    h, w = mask_full.shape[:2]
                    y_end = min(y_padded + ps, h)
                    x_end = min(x_padded + ps, w)

                    # Извлечь патч маски (все 5 каналов)
                    mask_patch_h = y_end - y_padded
                    mask_patch_w = x_end - x_padded

                    # Создать маску патча полного размера с padding нулями если нужно
                    mask_patch = np.zeros((ps, ps, 5), dtype=np.uint8)
                    mask_patch[:mask_patch_h, :mask_patch_w, :] = mask_full[y_padded:y_end, x_padded:x_end, :]

                    # Отрисовать маску патча в той же палитре, что и исходный M4D label PNG.
                    mask_rgb = one_hot_mask_to_rgb(mask_patch)

                    # Сохранить маску
                    Image.fromarray(mask_rgb).save(mask_png_dir / f"{patch_id}_mask.png")

                print(f"  ✓ Saved {len(df)} mask PNG files to: {mask_png_dir}/")
            except Exception as e:
                print(f"  ⚠️  Warning: Could not save mask patches: {e}")

        print(f"  ✓ Extracted {len(df)} patches")
        grid_size = int(np.sqrt(len(df)))
        print(f"  Grid: {grid_size}×{grid_size}")

        # Clean up temp files
        temp_scene_path.unlink()
        temp_mask_path.unlink()

    except Exception as e:
        print(f"❌ Error extracting patches: {e}")
        return 1

    # Print summary
    print(f"\n[4/4] Saving metadata...")
    print(f"  ✓ Patches saved to: {output_dir / 'patches_npy'}/")
    print(f"  ✓ Index saved to: {output_dir / f'index_{config.name}.parquet'}")
    print(f"  ✓ Metadata saved to: {output_dir / f'metadata_{config.name}.json'}")

    # Save patches as PNG if requested
    if args.save_png:
        print(f"\n[BONUS] Saving patches as PNG...")
        png_dir = output_dir / "patches_png"
        num_saved = save_patches_as_png(output_dir / "patches_npy", png_dir)
        print(f"  ✓ Saved {num_saved} PNG files to: {png_dir}/")

    # Create preview grid if requested
    if args.save_preview:
        print(f"\n[BONUS] Creating patch preview grid...")
        preview_file = output_dir / f"patches_preview_grid.png"
        if create_patch_preview_grid(output_dir / "patches_npy", preview_file):
            print(f"  ✓ Preview grid saved to: {preview_file}")

    print("\n" + "="*70)
    print("✓ EXTRACTION COMPLETE")
    print("="*70)
    print(f"\nNext step: Reconstruct patches with:")
    print(f"  python reconstruct_scene_simple.py --patches-dir {output_dir}/")
    print("="*70 + "\n")

    return 0

if __name__ == "__main__":
    sys.exit(main())
