"""E1(b) -- does native-512 resolution alone recover thin Oil slicks?

Two-way (plus one reference) comparison, scored on the SAME ground-truth Oil
components, of:

    baseline    SegFormer-B0 / 256, patch 256 stride 128   results/oof_gpu_full_v1
    ca_stem     + Coordinate Attention at stem, 256         results/oof_segformer_b0_ca_256
    native512   SegFormer-B0 / 512, patch 512 stride 256    results/oof_native512_full_v1

against the acceptance criterion set for E1 in the 2026-09-22 error-map
handoff: detection rate by mean-width bin, not pooled Oil IoU.

This is intentionally separate from the three-model error map of the working archive.
That analysis' cross-model consensus block (fn3/fp3, suspicious_fp/fn) hardcodes
the three CA-ablation variants by name and reuses the baseline's
`_coverage.npy`, which encodes window multiplicity for patch 256 / stride 128.
native512 has a different patch/stride, so its window-overlap geometry differs
and must not be folded into that consensus logic or the 73-scene label-review
list. The helper `component_geometry` below is copied unchanged from it.

No inference is run. Everything is post-hoc from the saved scene rasters.
Read-only outside --output.
"""
import argparse
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

from segmentation import paths
from segmentation.evaluation.common import CONN8, OIL, read_prediction

WIDTH_BINS = [0, 4, 8, 16, 32, np.inf]
WIDTH_LABELS = ['<4', '4-8', '8-16', '16-32', '>=32']


def component_geometry(comp_mask, target, radius=5):
    """Shape and neighbourhood of one ground-truth Oil component.

    mean_width_px    twice the mean inscribed radius -- how thick the slick is
    elongation       major/minor axis ratio of the pixel scatter
    lookalike_ring   share of the ring just outside the component labelled
                     Look-alike, i.e. is the slick sitting in a look-alike field
    """
    ys, xs = np.nonzero(comp_mask)
    y0, y1 = max(int(ys.min()) - radius - 1, 0), min(int(ys.max()) + radius + 2, comp_mask.shape[0])
    x0, x1 = max(int(xs.min()) - radius - 1, 0), min(int(xs.max()) + radius + 2, comp_mask.shape[1])
    sub = comp_mask[y0:y1, x0:x1]
    distance = ndimage.distance_transform_edt(np.pad(sub, 1))[1:-1, 1:-1]
    mean_width = float(2 * distance[sub].mean())

    coords = np.stack([ys - ys.mean(), xs - xs.mean()])
    if coords.shape[1] > 1:
        eigen = np.linalg.eigvalsh(np.cov(coords))
        minor = max(float(eigen[0]), 1e-9)
        elongation = float(np.sqrt(max(float(eigen[1]), 0.0) / minor))
    else:
        elongation = 1.0

    ring = ndimage.binary_dilation(sub, CONN8, iterations=radius) & ~sub
    ring_target = target[y0:y1, x0:x1][ring]
    lookalike_ring = float((ring_target == 2).mean()) if ring_target.size else 0.0
    sea_ring = float((ring_target == 0).mean()) if ring_target.size else 0.0
    return {'mean_width_px': mean_width, 'elongation': elongation,
            'lookalike_ring_share': lookalike_ring, 'sea_ring_share': sea_ring}


RUNS = {
    'baseline': 'oof_gpu_full_v1',
    'ca_stem': 'oof_segformer_b0_ca_256',
    'native512': 'oof_native512_full_v1',
}


def analyse_scene(task, results_root, labels_root):
    scene, fold = task
    labels_root, results_root = Path(labels_root), Path(results_root)
    target = np.asarray(Image.open(labels_root / f'{scene.removeprefix("train_")}.png'))
    if target.ndim != 2:
        raise ValueError(f'Unexpected label shape for {scene}')

    # GT components are built once from the reference mask and every model is
    # scored against this SAME set -- a model is never allowed to grow its own
    # component graph, or detection rates across models would not be comparable.
    gt_oil = target == OIL
    gt_comp, n_gt = ndimage.label(gt_oil, structure=CONN8)
    gt_areas = np.bincount(gt_comp.ravel(), minlength=n_gt + 1)[1:] if n_gt else np.array([], dtype=np.int64)
    geometry = {cid: component_geometry(gt_comp == cid, target) for cid in range(1, n_gt + 1)}

    detected = {}
    for model, run in RUNS.items():
        pred = read_prediction(results_root / run, fold, scene)
        if pred.shape != target.shape:
            raise ValueError(f'Shape mismatch: {model} predicts {pred.shape} for '
                              f'{scene} fold{fold:02d}, target is {target.shape}')
        pred_oil = pred == OIL
        detected_any = np.zeros(n_gt + 1, dtype=bool)
        if n_gt:
            hit = np.bincount(np.asarray(gt_comp[pred_oil]).ravel(), minlength=n_gt + 1)
            hit[0] = 0
            detected_any[1:] = hit[1:] >= 1
        detected[model] = detected_any

    rows = []
    for cid in range(1, n_gt + 1):
        rows.append({
            'scene_id': scene, 'fold': fold, 'component_id': cid,
            'area_px': int(gt_areas[cid - 1]),
            'mean_width_px': geometry[cid]['mean_width_px'],
            **{f'detected_{model}': bool(detected[model][cid]) for model in RUNS},
        })
    return rows


def validate_against_existing(components, existing_path):
    """Cross-check GT geometry and baseline/ca_stem detection against the
    already-published 3-variant analysis, so a plumbing mistake here (wrong
    fold, wrong scene, wrong prediction file) fails loudly instead of quietly
    producing a plausible-looking wrong number."""
    existing = pd.read_csv(existing_path).set_index(['scene_id', 'component_id'])
    merged = components.set_index(['scene_id', 'component_id']).join(
        existing[['area_px', 'mean_width_px', 'baseline', 'ca_stem']], rsuffix='_ref', how='left')

    if merged['area_px_ref'].isna().any():
        missing = merged[merged['area_px_ref'].isna()].index.tolist()[:5]
        raise ValueError(f'{merged["area_px_ref"].isna().sum()} components here are absent from '
                          f'{existing_path} (e.g. {missing}) -- scene/fold mismatch')
    full_run = len(components) == len(existing)
    if full_run and len(merged) != len(existing):
        raise ValueError(f'Component count mismatch on a full run: {len(merged)} here vs '
                          f'{len(existing)} in {existing_path}')
    if not np.array_equal(merged['area_px'].to_numpy(), merged['area_px_ref'].to_numpy()):
        raise ValueError('area_px disagrees with the existing per-component CSV')
    if not np.allclose(merged['mean_width_px'].to_numpy(), merged['mean_width_px_ref'].to_numpy()):
        raise ValueError('mean_width_px disagrees with the existing per-component CSV')
    if not np.array_equal(merged['detected_baseline'].to_numpy(), merged['baseline'].to_numpy()):
        raise ValueError('detected_baseline disagrees with the existing per-component CSV')
    if not np.array_equal(merged['detected_ca_stem'].to_numpy(), merged['ca_stem'].to_numpy()):
        raise ValueError('detected_ca_stem disagrees with the existing per-component CSV')

    if full_run:
        for label in WIDTH_LABELS:
            here = (pd.cut(components.mean_width_px, WIDTH_BINS, labels=WIDTH_LABELS) == label).sum()
            ref = (pd.cut(existing.mean_width_px, WIDTH_BINS, labels=WIDTH_LABELS) == label).sum()
            if here != ref:
                raise ValueError(f'Width bin {label}: {here} components here vs {ref} in {existing_path}')
    print(f'Validated against {existing_path.name}: {len(merged)} components, '
          f'geometry and baseline/ca_stem detection match exactly'
          + ('.' if full_run else ' (partial run, count/bin totals not cross-checked).'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    paths.add_root_argument(parser)
    parser.add_argument('--results-root', type=Path, default=paths.RESULTS)
    parser.add_argument('--labels-root', type=Path,
                        default=paths.DATASET / 'train/labels_1D')
    parser.add_argument('--output', type=Path,
                        default=paths.RESULTS / 'error_analysis_3models')
    parser.add_argument('--validate-against', type=Path,
                        default=paths.RESULTS / 'error_analysis_3models/oil_component_agreement.csv')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    baseline_dir = args.results_root / RUNS['baseline']
    tasks = []
    for fold in range(1, 6):
        for path in sorted((baseline_dir / f'fold{fold:02d}').glob('*_target_rgb.png')):
            tasks.append((path.name.removesuffix('_target_rgb.png'), fold))
    if args.limit:
        tasks = tasks[:args.limit]
    print(f'{len(tasks)} scenes x {len(RUNS)} runs: {list(RUNS)}')

    worker = partial(analyse_scene, results_root=args.results_root, labels_root=args.labels_root)
    rows = []
    with Pool(args.workers) as pool:
        for done, scene_rows in enumerate(pool.imap_unordered(worker, tasks, chunksize=4), 1):
            rows.extend(scene_rows)
            if done % 100 == 0:
                print(f'  {done}/{len(tasks)}')

    components = pd.DataFrame(rows)
    components['width_bin'] = pd.cut(components.mean_width_px, WIDTH_BINS, labels=WIDTH_LABELS)

    args.output.mkdir(parents=True, exist_ok=True)
    components.to_csv(args.output / 'native512_width_bin_detection.csv', index=False)

    if args.validate_against.is_file():
        validate_against_existing(components, args.validate_against)
    else:
        print(f'Warning: {args.validate_against} not found, skipping cross-check.')

    summary = components.groupby('width_bin', observed=True).agg(
        n_components=('component_id', 'size'),
        **{f'detection_rate_{model}': (f'detected_{model}', 'mean') for model in RUNS},
    ).reindex(WIDTH_LABELS)
    summary.to_csv(args.output / 'native512_width_bin_summary.csv')
    print('\nDetection rate by mean slick width (E1 acceptance criterion):')
    print(summary.to_string())

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(WIDTH_LABELS))
    width = 0.25
    colors = {'baseline': '#8ab4d8', 'ca_stem': '#2f7d54', 'native512': '#d1495b'}
    for i, model in enumerate(RUNS):
        ax.bar(x + (i - 1) * width, summary[f'detection_rate_{model}'] * 100, width,
               label=model, color=colors[model])
    ax.set_xticks(x, [f'{label}\nn={int(n) if pd.notna(n) else 0}'
                      for label, n in zip(WIDTH_LABELS, summary.n_components)])
    ax.set_xlabel('mean slick width, px')
    ax.set_ylabel('detection rate, %')
    ax.set_title('Does native-512 resolution recover thin slicks? (E1 acceptance criterion)')
    ax.legend(frameon=False)
    fig.tight_layout()
    figures_dir = args.output / 'figures'
    figures_dir.mkdir(exist_ok=True)
    fig.savefig(figures_dir / 'fig_native512_width_bins.png', bbox_inches='tight')
    print(f'\n  figures/fig_native512_width_bins.png')

    print('\nInterpretation guide (not automated -- judge by eye against these n):')
    print('  native-512 should raise detection rate in the <4 and 4-8 px bins')
    print('  WITHOUT a material drop in the wider bins; a gain confined to one')
    print('  bin with small n is weak evidence for choosing native-512 as E1.')


if __name__ == '__main__':
    main()
