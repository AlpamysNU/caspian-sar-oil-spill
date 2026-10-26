#!/usr/bin/env python3
"""Audit M4D patch geometry and ideal Stage 2 workload on official train."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from PIL import Image
from scipy import ndimage

CLASS_NAMES = {0: "Sea", 1: "Oil", 2: "Look-alike", 3: "Ship", 4: "Land"}
CONFIGS = ((256, 128), (384, 192), (512, 256), (512, 128))


def decode_m4d_mask(mask_rgb: np.ndarray) -> np.ndarray:
    """Decode an RGB M4D mask using the same class rules as PatchExtractor."""
    if mask_rgb.ndim != 3 or mask_rgb.shape[2] != 3:
        raise ValueError(f"Expected an RGB mask, received shape {mask_rgb.shape}")
    sea = np.all(mask_rgb == (0, 0, 0), axis=-1)
    oil = np.all(mask_rgb == (0, 255, 255), axis=-1)
    look_alike = np.all(mask_rgb == (255, 0, 0), axis=-1)
    land = np.all(mask_rgb == (0, 153, 0), axis=-1)
    red, green, blue = (mask_rgb[..., index] for index in range(3))
    ship = ((red >= 60) & (red <= 170) & (green >= 30) & (green <= 100) & (blue <= 15))
    class_masks = (sea, oil, look_alike, ship, land)
    assignment_count = np.sum(np.stack(class_masks, axis=0), axis=0)
    if np.any(assignment_count != 1):
        invalid = mask_rgb[assignment_count != 1]
        sample = np.unique(invalid.reshape(-1, 3), axis=0)[:10].tolist()
        raise ValueError(f"Mask has {invalid.shape[0]} unknown or ambiguous pixels; sample RGB values: {sample}")
    decoded = np.empty(mask_rgb.shape[:2], dtype=np.uint8)
    for class_id, class_mask in enumerate(class_masks):
        decoded[class_mask] = class_id
    return decoded


def load_mask(path: Path) -> np.ndarray:
    """Load and strictly decode an M4D RGB label image."""
    return decode_m4d_mask(np.asarray(Image.open(path).convert("RGB")))


def padding_and_positions(length: int, patch_size: int, stride: int) -> tuple[int, int, list[int]]:
    """Return symmetric padding and padded-grid start positions for one axis."""
    if length < 1 or patch_size < 1 or stride < 1 or stride > patch_size:
        raise ValueError("Lengths and patch size must be positive; stride must not exceed patch size")
    total_padding = patch_size - length if length < patch_size else (stride - (length - patch_size) % stride) % stride
    before = total_padding // 2
    after = total_padding - before
    return before, after, list(range(0, length + total_padding - patch_size + 1, stride))


def iter_original_windows(height: int, width: int, patch_size: int, stride: int) -> tuple[tuple[int, int, int, int], Iterator[tuple[int, int, int, int, int, int]]]:
    """Yield padded starts and clipped original-space bounds for every grid window."""
    pad_top, pad_bottom, y_positions = padding_and_positions(height, patch_size, stride)
    pad_left, pad_right, x_positions = padding_and_positions(width, patch_size, stride)

    def windows() -> Iterator[tuple[int, int, int, int, int, int]]:
        for y_padded in y_positions:
            top = max(0, y_padded - pad_top)
            bottom = min(height, y_padded - pad_top + patch_size)
            for x_padded in x_positions:
                left = max(0, x_padded - pad_left)
                right = min(width, x_padded - pad_left + patch_size)
                yield y_padded, x_padded, top, bottom, left, right

    return (pad_top, pad_bottom, pad_left, pad_right), windows()


def audit_scene(mask: np.ndarray, patch_size: int, stride: int) -> dict[str, Any]:
    """Measure oracle-selected windows and coverage in original-scene coordinates."""
    height, width = mask.shape
    padding, windows = iter_original_windows(height, width, patch_size, stride)
    oil = mask == 1
    selected_scene_coverage = np.zeros(mask.shape, dtype=bool)
    selected_oil_coverage = np.zeros(mask.shape, dtype=bool)
    total_patches = selected_patches = all_valid_area = selected_valid_area = 0
    repeated_oil_coverage = 0
    for _, _, top, bottom, left, right in windows:
        valid_area = (bottom - top) * (right - left)
        oil_view = oil[top:bottom, left:right]
        oil_pixels = int(oil_view.sum())
        total_patches += 1
        all_valid_area += valid_area
        if oil_pixels == 0:
            continue
        selected_patches += 1
        selected_valid_area += valid_area
        repeated_oil_coverage += oil_pixels
        selected_scene_coverage[top:bottom, left:right] = True
        selected_oil_coverage[top:bottom, left:right] |= oil_view
    oil_pixels = int(oil.sum())
    unique_oil_coverage = int(selected_oil_coverage.sum())
    if unique_oil_coverage != oil_pixels:
        raise AssertionError(f"Oracle selection retained {unique_oil_coverage}/{oil_pixels} Oil pixels")
    padded_area_per_patch = patch_size * patch_size
    return {
        "pad_top": padding[0], "pad_bottom": padding[1], "pad_left": padding[2], "pad_right": padding[3],
        "patches_total": total_patches, "oil_positive_patches": selected_patches,
        "all_padded_area_with_repetition_px": total_patches * padded_area_per_patch,
        "selected_padded_area_with_repetition_px": selected_patches * padded_area_per_patch,
        "all_valid_area_with_repetition_px": all_valid_area,
        "selected_valid_area_with_repetition_px": selected_valid_area,
        "selected_unique_scene_area_px": int(selected_scene_coverage.sum()),
        "oil_pixels": oil_pixels, "oil_pixel_coverage_unique_px": unique_oil_coverage,
        "oil_pixel_coverage_repeated_px": repeated_oil_coverage,
    }


def component_rows(scene_id: str, mask: np.ndarray) -> list[dict[str, Any]]:
    """Return 8-connected component geometry for Oil, Look-alike, and Ship."""
    rows = []
    structure = np.ones((3, 3), dtype=int)
    for class_id in (1, 2, 3):
        labels, _ = ndimage.label(mask == class_id, structure=structure)
        for component_id, component_slice in enumerate(ndimage.find_objects(labels), start=1):
            if component_slice is None:
                continue
            y_slice, x_slice = component_slice
            component = labels[component_slice] == component_id
            rows.append({
                "scene_id": scene_id, "class": CLASS_NAMES[class_id], "component_id": component_id,
                "area_px": int(component.sum()), "bbox_h": y_slice.stop - y_slice.start,
                "bbox_w": x_slice.stop - x_slice.start,
                "bbox_max": max(y_slice.stop - y_slice.start, x_slice.stop - x_slice.start),
            })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dictionaries to CSV with a stable header."""
    if not rows:
        raise ValueError(f"No rows available for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    train = args.dataset_root / "train"
    image_paths = sorted((train / "images").glob("*.jpg"))
    if args.limit is not None:
        image_paths = image_paths[:args.limit]
    if not image_paths:
        raise FileNotFoundError(f"No train images found below {train}")
    args.output.mkdir(parents=True, exist_ok=True)
    per_scene_rows: list[dict[str, Any]] = []
    components: list[dict[str, Any]] = []
    totals = {f"{size}_{stride}": {
        "patch_size": size, "stride": stride, "scene_count": 0, "patches_total": 0,
        "oil_positive_patches": 0, "all_padded_area_with_repetition_px": 0,
        "selected_padded_area_with_repetition_px": 0, "all_valid_area_with_repetition_px": 0,
        "selected_valid_area_with_repetition_px": 0, "selected_unique_scene_area_px": 0,
        "oil_pixels": 0, "oil_pixel_coverage_unique_px": 0, "oil_pixel_coverage_repeated_px": 0,
    } for size, stride in CONFIGS}
    for image_path in image_paths:
        label_path = train / "labels" / f"{image_path.stem}.png"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing mask for {image_path.name}: {label_path}")
        mask = load_mask(label_path)
        components.extend(component_rows(image_path.stem, mask))
        for patch_size, stride in CONFIGS:
            config_id = f"{patch_size}_{stride}"
            metrics = audit_scene(mask, patch_size, stride)
            per_scene_rows.append({"scene_id": image_path.stem, "patch_size": patch_size, "stride": stride, **metrics})
            totals[config_id]["scene_count"] += 1
            for key in totals[config_id]:
                if key not in {"patch_size", "stride", "scene_count"}:
                    totals[config_id][key] += metrics[key]
    configurations = {}
    for config_id, values in totals.items():
        total_patches = values["patches_total"]
        selected_patches = values["oil_positive_patches"]
        all_valid_area = values["all_valid_area_with_repetition_px"]
        selected_valid_area = values["selected_valid_area_with_repetition_px"]
        oil_pixels = values["oil_pixels"]
        configurations[config_id] = {**values,
            "accepted_fraction": selected_patches / total_patches,
            "potential_patch_workload_reduction_fraction": 1 - selected_patches / total_patches,
            "potential_valid_area_reduction_fraction": 1 - selected_valid_area / all_valid_area,
            "oil_pixel_retention_fraction": values["oil_pixel_coverage_unique_px"] / oil_pixels if oil_pixels else None,
        }
    summary = {
        "scope": "official M4D train split only", "scene_count": len(image_paths),
        "oracle_rule": "Select every patch whose valid original-scene area contains at least one Oil pixel.",
        "interpretation": "Upper bound on patch workload reduction for this oracle rule; not measured GPU savings.",
        "configurations": configurations,
    }
    write_csv(args.output / "patch_workload_by_scene.csv", per_scene_rows)
    write_csv(args.output / "components.csv", components)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output / "README.md").write_text(
        "# Stage 2 geometry audit\n\nThis audit uses the official M4D train split only. Windows follow the symmetric padded grids used by the patch extractor. The oracle selects every patch whose valid original-scene area contains at least one Oil pixel. Reported reductions are workload upper bounds for this rule and are not measured GPU savings.\n\nSee `summary.json`, `patch_workload_by_scene.csv`, and `components.csv`.\n",
        encoding="utf-8")
    print(f"Audited {len(image_paths)} scenes across {len(CONFIGS)} patch grids.")


if __name__ == "__main__":
    main()
