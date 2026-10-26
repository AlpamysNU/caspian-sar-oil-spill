#!/usr/bin/env python3
"""Build training-ready M4D patches and a single Parquet index.

This command defaults to official train and prefixes scene IDs with the split name. This prevents
train/img_0001 and test/img_0001 from overwriting each other.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from patching.config import PatchConfig
    from patching.core import PatchExtractor
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from patching.config import PatchConfig
    from patching.core import PatchExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract all M4D scenes into training-ready NPY patches."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Dataset directory containing train/ and test/",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument('--splits', nargs='+', choices=('train', 'test'), default=['train'])
    parser.add_argument("--patch-size", type=int, choices=(256, 384, 512), default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and report patch count/storage without writing patches",
    )
    parser.add_argument(
        "--limit-per-split",
        type=int,
        default=None,
        help="Process only the first N scenes per split (for a smoke test)",
    )
    return parser.parse_args()


def patch_count(height: int, width: int, patch_size: int, stride: int) -> int:
    pad_h = patch_size - height if height < patch_size else (
        stride - (height - patch_size) % stride
    ) % stride
    pad_w = patch_size - width if width < patch_size else (
        stride - (width - patch_size) % stride
    ) % stride
    rows = (height + pad_h - patch_size) // stride + 1
    cols = (width + pad_w - patch_size) // stride + 1
    return rows * cols


def collect_scenes(dataset_root: Path, limit: int | None, splits=('train',)) -> dict[str, list[tuple[Path, Path]]]:
    scenes: dict[str, list[tuple[Path, Path]]] = {}
    if limit is not None and limit < 1:
        raise ValueError('limit must be positive')
    for split in dict.fromkeys(splits):
        image_dir = dataset_root / split / "images"
        label_dir = dataset_root / split / "labels"
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(
                f"Expected {image_dir} and {label_dir} to be directories"
            )

        image_paths = sorted(image_dir.glob("*.jpg"))
        if not image_paths:
            raise ValueError(f'No images in {image_dir}')
        if {p.stem for p in image_paths} != {p.stem for p in label_dir.glob('*.png')}:
            raise ValueError(f'Image/mask identifiers do not match in {split}')
        if limit is not None:
            image_paths = image_paths[:limit]
        pairs = []
        for image_path in image_paths:
            label_path = label_dir / f"{image_path.stem}.png"
            if not label_path.is_file():
                raise FileNotFoundError(f"Missing label for {image_path}: {label_path}")
            pairs.append((image_path, label_path))
        scenes[split] = pairs
    return scenes


def existing_ancestor(path: Path) -> Path:
    candidate = path.resolve()
    while not candidate.exists():
        candidate = candidate.parent
    return candidate


def estimate(scenes: dict[str, list[tuple[Path, Path]]], patch_size: int, stride: int) -> dict:
    split_counts = {}
    total_patches = 0
    dimensions = set()
    for split, pairs in scenes.items():
        count = 0
        for image_path, label_path in pairs:
            with Image.open(image_path) as image:
                width, height = image.size
            with Image.open(label_path) as label:
                if label.size != (width, height):
                    raise ValueError(f'Shape mismatch: {label_path}')
            dimensions.add((height, width))
            count += patch_count(height, width, patch_size, stride)
        split_counts[split] = count
        total_patches += count

    # A uint8 .npy contains the raw pixels plus a small format header.
    estimated_npy_bytes = total_patches * (patch_size * patch_size + 128)
    return {
        "scene_count": sum(len(pairs) for pairs in scenes.values()),
        "scene_count_by_split": {key: len(value) for key, value in scenes.items()},
        "scene_dimensions": [list(value) for value in sorted(dimensions)],
        "patch_count": total_patches,
        "patch_count_by_split": split_counts,
        "estimated_npy_bytes": estimated_npy_bytes,
        "estimated_npy_gb": estimated_npy_bytes / 1e9,
        "estimated_npy_gib": estimated_npy_bytes / (1024 ** 3),
    }


def ensure_clean_output(output: Path) -> None:
    if not output.exists():
        return
    mixed_outputs = list(output.glob("index_*.parquet"))
    patches_dir = output / "patches_npy"
    if mixed_outputs or (patches_dir.exists() and next(patches_dir.glob("*.npy"), None)):
        raise FileExistsError(
            f"Output already contains a patch dataset: {output}. "
            "Choose a new output directory to avoid mixing experiments."
        )


def main() -> int:
    args = parse_args()
    if args.stride < 1 or args.stride > args.patch_size:
        raise ValueError("stride must be between 1 and patch_size")

    overlap = 1.0 - args.stride / args.patch_size
    config = PatchConfig(
        patch_size=args.patch_size,
        stride=args.stride,
        overlap=overlap,
        window_type="hann",
        pad_mode="reflect",
        skip_border=False,
        name=f"ps{args.patch_size}_overlap{int(round(overlap * 100)):02d}",
    )
    scenes = collect_scenes(args.dataset_root, args.limit_per_split, args.splits)
    report = estimate(scenes, args.patch_size, args.stride)
    disk = shutil.disk_usage(existing_ancestor(args.output))
    report.update(
        {
            "dataset_root": str(args.dataset_root.resolve()),
            "output": str(args.output.resolve()),
            "config": config.to_dict(),
            "disk_free_bytes": disk.free,
            "disk_free_gib": disk.free / (1024 ** 3),
            "recommended_free_gib": report["estimated_npy_gib"] * 1.20,
        }
    )
    print(json.dumps(report, indent=2))

    if args.dry_run:
        return 0
    if disk.free < report["estimated_npy_bytes"] * 1.20:
        raise OSError(
            "Insufficient free space: keep at least 20% headroom above the NPY estimate"
        )
    ensure_clean_output(args.output)

    args.output.mkdir(parents=True, exist_ok=True)
    extractor = PatchExtractor(config, args.output)
    started = time.monotonic()
    completed_scenes = 0
    manifest = []

    with tempfile.TemporaryDirectory(prefix="m4d-patching-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        for split in scenes:
            for image_path, label_path in scenes[split]:
                # Explicit L conversion avoids treating RGB rows as channels.
                grayscale = np.asarray(Image.open(image_path).convert("L"), dtype=np.uint8)
                manifest.append({
                    'scene_id': f'{split}_{image_path.stem}', 'split': split,
                    'group_id': f'{split}_{image_path.stem}',
                    'group_basis': 'image ID; acquisition and incident grouping unavailable',
                    'height': grayscale.shape[0], 'width': grayscale.shape[1],
                    'image_path': str(image_path.resolve()), 'mask_path': str(label_path.resolve()),
                    'image_sha256': hashlib.sha256(image_path.read_bytes()).hexdigest(),
                    'mask_sha256': hashlib.sha256(label_path.read_bytes()).hexdigest(),
                })
                temp_scene = temp_dir / f"{split}_{image_path.stem}.npy"
                np.save(temp_scene, grayscale)

                extractor.extract_patches_from_scene(
                    scene_path=temp_scene,
                    mask_path=label_path,
                    scene_id=f"{split}_{image_path.stem}",
                    split=split,
                    image_pad_mode="reflect",
                    mask_pad_mode="constant",
                    show_progress=False,
                )
                temp_scene.unlink()
                completed_scenes += 1
                if completed_scenes % 25 == 0 or completed_scenes == report["scene_count"]:
                    elapsed = time.monotonic() - started
                    rate = completed_scenes / elapsed if elapsed else 0.0
                    remaining = (report["scene_count"] - completed_scenes) / rate if rate else 0.0
                    print(
                        f"Scenes {completed_scenes}/{report['scene_count']} | "
                        f"{rate:.2f} scenes/s | ETA {remaining / 60:.1f} min",
                        flush=True,
                    )

    dataframe, metadata = extractor.build_index()
    if len(dataframe) != report['patch_count'] or not dataframe.patch_id.is_unique:
        raise AssertionError('Unexpected patch count or duplicate identifiers')
    import pandas as pd
    pd.DataFrame(manifest).to_csv(args.output / 'scene_manifest.csv', index=False)
    elapsed = time.monotonic() - started
    build_report = {
        **report,
        "actual_patch_count": len(dataframe),
        "elapsed_seconds": elapsed,
        "metadata": metadata,
    }
    report_path = args.output / "build_report.json"
    report_path.write_text(json.dumps(build_report, indent=2) + "\n", encoding="utf-8")
    print(f"Completed {len(dataframe)} patches in {elapsed / 60:.1f} min")
    print(f"Index: {args.output / ('index_' + config.name + '.parquet')}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
