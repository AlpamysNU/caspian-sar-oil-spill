#!/usr/bin/env python3
"""Measure 8-connected M4D components against actual padded patch grids."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage

from stage2_geometry_audit import load_mask, padding_and_positions

CLASS_NAMES = {1: 'Oil', 2: 'Look-alike', 3: 'Ship'}
CONFIGS = ((256, 128), (384, 192), (512, 256), (512, 128))


def padded_positions(length: int, patch_size: int, stride: int) -> tuple[int, list[int]]:
    before, _, positions = padding_and_positions(length, patch_size, stride)
    return before, positions


def area_group(area: int) -> str:
    if area <= 8:
        return '1-8'
    if area <= 24:
        return '9-24'
    return '>=25'


def component_coverage(labels: np.ndarray, component_id: int, component_slice: tuple[slice, slice], patch_size: int, stride: int) -> tuple[bool, float, float, int]:
    height, width = labels.shape
    pad_top, y_positions = padded_positions(height, patch_size, stride)
    pad_left, x_positions = padded_positions(width, patch_size, stride)
    y_slice, x_slice = component_slice
    area = int((labels[y_slice, x_slice] == component_id).sum())
    counts = []
    for y in y_positions:
        top, bottom = y - pad_top, y - pad_top + patch_size
        if bottom <= y_slice.start or top >= y_slice.stop:
            continue
        for x in x_positions:
            left, right = x - pad_left, x - pad_left + patch_size
            if right <= x_slice.start or left >= x_slice.stop:
                continue
            intersect = labels[max(top, y_slice.start):min(bottom, y_slice.stop), max(left, x_slice.start):min(right, x_slice.stop)]
            count = int((intersect == component_id).sum())
            if count:
                counts.append(count)
    maximum = max(counts, default=0)
    return maximum == area, maximum / area, sum(counts) / area, len(counts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()
    labels_dir = args.dataset_root / 'train' / 'labels'
    paths = sorted(labels_dir.glob('*.png'))
    if args.limit:
        paths = paths[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    structure = np.ones((3, 3), dtype=int)
    for path in paths:
        mask = load_mask(path)
        for class_id, class_name in CLASS_NAMES.items():
            labels, count = ndimage.label(mask == class_id, structure=structure)
            for component_id, component_slice in enumerate(ndimage.find_objects(labels), start=1):
                if component_slice is None:
                    continue
                y_slice, x_slice = component_slice
                area = int((labels[component_slice] == component_id).sum())
                base = {'scene_id': path.stem, 'class': class_name, 'component_id': component_id, 'area_px': area, 'area_group': area_group(area), 'bbox_h': y_slice.stop - y_slice.start, 'bbox_w': x_slice.stop - x_slice.start, 'bbox_max': max(y_slice.stop - y_slice.start, x_slice.stop - x_slice.start)}
                for patch_size, stride in CONFIGS:
                    complete, maximum, repetition, windows = component_coverage(labels, component_id, component_slice, patch_size, stride)
                    rows.append({**base, 'patch_size': patch_size, 'stride': stride, 'fully_contained_in_window': int(complete), 'max_fraction_in_one_window': maximum, 'mean_pixel_repetition': repetition, 'windows_touching_component': windows})
    fields = list(rows[0])
    with (args.output / 'component_window_coverage.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['class'], row['patch_size'], row['stride'], row['area_group'])].append(row)
    summary = {'scene_count': len(paths), 'metrics': []}
    for key, values in sorted(grouped.items()):
        summary['metrics'].append({'class': key[0], 'patch_size': key[1], 'stride': key[2], 'area_group': key[3], 'component_count': len(values), 'fully_contained_fraction': float(np.mean([value['fully_contained_in_window'] for value in values])), 'mean_max_fraction_in_one_window': float(np.mean([value['max_fraction_in_one_window'] for value in values])), 'mean_pixel_repetition': float(np.mean([value['mean_pixel_repetition'] for value in values]))})
    (args.output / 'component_window_coverage_summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote {len(rows)} component-grid records for {len(paths)} scenes.')


if __name__ == '__main__':
    main()
