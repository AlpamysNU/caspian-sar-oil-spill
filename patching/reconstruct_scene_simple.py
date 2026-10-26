#!/usr/bin/env python3
"""
Simple script to reconstruct scene from extracted patches.
Usage: python reconstruct_scene_simple.py --patches-dir patches_dir/
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Import using absolute imports
try:
    from patching.core import SceneReconstructor
    from patching.config import PatchConfig
except ImportError:
    # If running as script
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from patching.core import SceneReconstructor
    from patching.config import PatchConfig

def main():
    parser = argparse.ArgumentParser(description="Reconstruct scene from patches")
    parser.add_argument("--patches-dir", required=True, help="Directory containing patches and metadata")
    parser.add_argument("--output", help="Output image file (optional)")
    args = parser.parse_args()

    patches_dir = Path(args.patches_dir)

    if not patches_dir.exists():
        print(f"❌ Error: Directory not found: {patches_dir}")
        return 1

    print("\n" + "="*70)
    print("SCENE RECONSTRUCTION")
    print("="*70)

    # Find metadata files
    parquet_files = list(patches_dir.glob("index_*.parquet"))
    if not parquet_files:
        print(f"❌ Error: No index_*.parquet found in {patches_dir}")
        return 1

    index_file = parquet_files[0]
    config_name = index_file.stem.replace("index_", "")

    print(f"\n[1/4] Loading metadata...")
    try:
        df = pd.read_parquet(index_file)
        print(f"  ✓ Loaded index: {len(df)} patches")
        print(f"  Config: {config_name}")

        # Extract config from metadata
        config = PatchConfig(
            patch_size=int(df.iloc[0]['patch_size']),
            stride=int(df.iloc[0]['stride']),
            overlap=float(df.iloc[0]['patch_size'] - df.iloc[0]['stride']) / df.iloc[0]['patch_size'],
            name=config_name
        )
    except Exception as e:
        print(f"❌ Error loading metadata: {e}")
        return 1

    # Load patches
    print(f"\n[2/4] Loading patches...")
    patches_npy_dir = patches_dir / "patches_npy"
    if not patches_npy_dir.exists():
        print(f"❌ Error: patches_npy directory not found in {patches_dir}")
        return 1

    try:
        patches = {}
        patches_coords = {}

        for _, row in df.iterrows():
            patch_id = row['patch_id']
            patch_path = patches_npy_dir / f"{patch_id}.npy"

            if not patch_path.exists():
                print(f"  ⚠️  Warning: Patch not found: {patch_path}")
                continue

            patch = np.load(patch_path)
            y_padded = int(row['y_padded'])
            x_padded = int(row['x_padded'])

            patches[(y_padded, x_padded)] = patch
            patches_coords[(y_padded, x_padded)] = row.to_dict()

        print(f"  ✓ Loaded {len(patches)} patches")

    except Exception as e:
        print(f"❌ Error loading patches: {e}")
        return 1

    # Reconstruct
    print(f"\n[3/4] Reconstructing scene...")
    try:
        reconstructor = SceneReconstructor(config)

        # Recover original scene size from padded grid geometry.
        row0 = df.iloc[0]
        pad_top = int(row0['pad_top'])
        pad_bottom = int(row0['pad_bottom'])
        pad_left = int(row0['pad_left'])
        pad_right = int(row0['pad_right'])

        padded_height = int(df['y_padded'].max()) + config.patch_size
        padded_width = int(df['x_padded'].max()) + config.patch_size
        original_height = padded_height - pad_top - pad_bottom
        original_width = padded_width - pad_left - pad_right

        reconstructed, weights = reconstructor.reconstruct_scene(
            patches=patches,
            original_height=original_height,
            original_width=original_width,
            patches_coords=patches_coords
        )

        print(f"  ✓ Reconstructed: {reconstructed.shape}")

    except Exception as e:
        print(f"❌ Error reconstructing: {e}")
        return 1

    # Compute metrics
    print(f"\n[4/4] Computing quality metrics...")
    coverage = (weights > 0).sum() / weights.size * 100

    print(f"  Coverage: {coverage:.1f}%")
    print(f"  Range: {reconstructed.min()}-{reconstructed.max()}")

    # Save reconstruction
    output_path = args.output or patches_dir / "reconstructed.png"
    output_path = Path(output_path)

    try:
        Image.fromarray(reconstructed.astype(np.uint8)).save(output_path)
        print(f"  ✓ Saved: {output_path}")
    except Exception as e:
        print(f"⚠️  Warning: Could not save image: {e}")

    # Create comparison visualization
    print(f"\nCreating comparison visualization...")
    viz_path = output_path.parent / f"{output_path.stem}_comparison.png"

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(reconstructed, cmap='gray')
    axes[0].set_title('Reconstructed Scene')
    axes[0].axis('off')

    axes[1].imshow(weights, cmap='viridis')
    axes[1].set_title('Weight Map (Hann Windowing)')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(viz_path, dpi=100, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {viz_path}")

    print("\n" + "="*70)
    print("✓ RECONSTRUCTION COMPLETE")
    print("="*70)
    print(f"\nResults:")
    print(f"  Reconstructed image: {output_path}")
    print(f"  Comparison: {viz_path}")
    print("="*70 + "\n")

    return 0

if __name__ == "__main__":
    sys.exit(main())
