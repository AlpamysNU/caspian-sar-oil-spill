#!/usr/bin/env python3
"""Inspect one patch-size/overlap configuration on one M4D scene.

The script intentionally runs a single configuration per invocation so patch
size and overlap can be studied as separate experimental factors.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image
from scipy.signal import windows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualise extraction and reconstruction for one configuration."
    )
    parser.add_argument("--image", type=Path, required=True, help="Input scene image")
    parser.add_argument("--patch-size", type=int, choices=(256, 384, 512), required=True)
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.0,
        help="Overlap fraction in [0, 1); use 0.5 for 50%%",
    )
    parser.add_argument(
        "--window",
        choices=("uniform", "hann"),
        default="uniform",
        help="Blending window; uniform is the neutral reconstruction control",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def validate_overlap(patch_size: int, overlap: float) -> int:
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    stride_float = patch_size * (1.0 - overlap)
    stride = int(round(stride_float))
    if stride < 1 or not math.isclose(stride_float, stride, abs_tol=1e-9):
        raise ValueError(
            "overlap must produce an integer stride: "
            f"patch_size={patch_size}, overlap={overlap}, stride={stride_float}"
        )
    return stride


def symmetric_padding(length: int, patch_size: int, stride: int) -> tuple[int, int]:
    if length < patch_size:
        required = patch_size - length
    else:
        required = (stride - (length - patch_size) % stride) % stride
    before = required // 2
    return before, required - before


def blending_window(patch_size: int, kind: str) -> np.ndarray:
    if kind == "uniform":
        return np.ones((patch_size, patch_size), dtype=np.float64)
    one_d = windows.hann(patch_size, sym=False)
    return np.clip(np.outer(one_d, one_d), 1e-6, None)


def run_experiment(
    image_path: Path,
    patch_size: int,
    overlap: float,
    window_kind: str,
    output_dir: Path,
) -> dict:
    stride = validate_overlap(patch_size, overlap)
    scene = np.asarray(Image.open(image_path).convert("L"), dtype=np.float64)
    height, width = scene.shape

    pad_top, pad_bottom = symmetric_padding(height, patch_size, stride)
    pad_left, pad_right = symmetric_padding(width, patch_size, stride)
    padded = np.pad(
        scene,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="reflect",
    )

    y_positions = list(range(0, padded.shape[0] - patch_size + 1, stride))
    x_positions = list(range(0, padded.shape[1] - patch_size + 1, stride))
    coords = [(y, x) for y in y_positions for x in x_positions]
    patches = [padded[y : y + patch_size, x : x + patch_size] for y, x in coords]

    window = blending_window(patch_size, window_kind)
    reconstructed_sum = np.zeros_like(padded, dtype=np.float64)
    weights = np.zeros_like(padded, dtype=np.float64)
    for patch, (y, x) in zip(patches, coords):
        reconstructed_sum[y : y + patch_size, x : x + patch_size] += patch * window
        weights[y : y + patch_size, x : x + patch_size] += window

    if np.any(weights == 0):
        raise RuntimeError("Patch grid left uncovered pixels")
    reconstructed_padded = reconstructed_sum / weights
    reconstructed = reconstructed_padded[
        pad_top : pad_top + height, pad_left : pad_left + width
    ]
    weights_cropped = weights[pad_top : pad_top + height, pad_left : pad_left + width]
    error = reconstructed - scene

    output_dir.mkdir(parents=True, exist_ok=True)
    config_id = f"ps{patch_size}_overlap{int(round(overlap * 100)):02d}"
    reconstruction_path = output_dir / f"{config_id}_reconstructed.png"
    panel_path = output_dir / f"{config_id}_inspection.png"
    metrics_path = output_dir / f"{config_id}_metrics.json"
    Image.fromarray(np.clip(np.rint(reconstructed), 0, 255).astype(np.uint8)).save(
        reconstruction_path
    )

    preview_count = min(12, len(patches))
    preview_cols = 4
    preview_rows = math.ceil(preview_count / preview_cols)
    fig = plt.figure(figsize=(18, 14), constrained_layout=True)
    grid = fig.add_gridspec(3, 3, height_ratios=(1, 1, 1.15))

    ax_original = fig.add_subplot(grid[0, 0])
    ax_original.imshow(scene, cmap="gray", vmin=0, vmax=255)
    ax_original.set_title("Original scene")
    ax_original.axis("off")

    ax_grid = fig.add_subplot(grid[0, 1])
    ax_grid.imshow(scene, cmap="gray", vmin=0, vmax=255)
    for y, x in coords:
        rect = Rectangle(
            (x - pad_left, y - pad_top),
            patch_size,
            patch_size,
            fill=False,
            edgecolor="#ffcc00",
            linewidth=0.8,
            alpha=0.8,
        )
        ax_grid.add_patch(rect)
    ax_grid.set_xlim(0, width)
    ax_grid.set_ylim(height, 0)
    ax_grid.set_title(f"Patch grid: {len(y_positions)} x {len(x_positions)} = {len(patches)}")
    ax_grid.axis("off")

    ax_reconstructed = fig.add_subplot(grid[0, 2])
    ax_reconstructed.imshow(reconstructed, cmap="gray", vmin=0, vmax=255)
    ax_reconstructed.set_title("Reconstructed scene")
    ax_reconstructed.axis("off")

    ax_error = fig.add_subplot(grid[1, 0])
    error_limit = max(float(np.max(np.abs(error))), 1e-12)
    image_error = ax_error.imshow(error, cmap="coolwarm", vmin=-error_limit, vmax=error_limit)
    ax_error.set_title("Signed reconstruction error")
    ax_error.axis("off")
    fig.colorbar(image_error, ax=ax_error, fraction=0.046, pad=0.04)

    ax_weights = fig.add_subplot(grid[1, 1])
    image_weights = ax_weights.imshow(weights_cropped, cmap="viridis")
    ax_weights.set_title(f"Weight map ({window_kind})")
    ax_weights.axis("off")
    fig.colorbar(image_weights, ax=ax_weights, fraction=0.046, pad=0.04)

    ax_hist = fig.add_subplot(grid[1, 2])
    ax_hist.hist(error.ravel(), bins=50, color="#4472c4", edgecolor="white")
    ax_hist.set_title("Error distribution")
    ax_hist.set_xlabel("Reconstructed - original")
    ax_hist.set_ylabel("Pixels")
    ax_hist.grid(alpha=0.25)

    preview_grid = grid[2, :].subgridspec(preview_rows, preview_cols)
    for index in range(preview_rows * preview_cols):
        axis = fig.add_subplot(preview_grid[index // preview_cols, index % preview_cols])
        if index < preview_count:
            y, x = coords[index]
            axis.imshow(patches[index], cmap="gray", vmin=0, vmax=255)
            axis.set_title(f"#{index + 1}: padded y={y}, x={x}", fontsize=9)
        axis.axis("off")

    fig.suptitle(
        f"{image_path.name} | patch={patch_size}, stride={stride}, "
        f"overlap={overlap:.0%}, window={window_kind}",
        fontsize=15,
    )
    fig.savefig(panel_path, dpi=150)
    plt.close(fig)

    metrics = {
        "image": str(image_path),
        "image_shape": [height, width],
        "patch_size": patch_size,
        "stride": stride,
        "overlap": overlap,
        "window": window_kind,
        "grid_shape": [len(y_positions), len(x_positions)],
        "patch_count": len(patches),
        "padding": {
            "top": pad_top,
            "bottom": pad_bottom,
            "left": pad_left,
            "right": pad_right,
        },
        "coverage_min": int(np.min(np.rint(weights_cropped))) if window_kind == "uniform" else float(weights_cropped.min()),
        "coverage_max": int(np.max(np.rint(weights_cropped))) if window_kind == "uniform" else float(weights_cropped.max()),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "max_abs_error": float(np.max(np.abs(error))),
        "inspection_figure": str(panel_path),
        "reconstruction": str(reconstruction_path),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def main() -> int:
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")
    metrics = run_experiment(
        image_path=args.image,
        patch_size=args.patch_size,
        overlap=args.overlap,
        window_kind=args.window,
        output_dir=args.output_dir,
    )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
