"""Oracle gate upper bound (2026-10-03): how much could a perfect binary 'Oil in this window?' gate add?

For every scene and arm the saved OOF prediction is re-scored after removing Oil (relabelled Sea) where an ideal gate
would have closed the path: (a) scene oracle: scenes without ground-truth Oil; (b) window oracle: pixels covered only by
256 px windows (stride 128, grid x = -15 + 128k, y = -59 + 128j as in Stage 4) that contain no ground-truth Oil.
A real gate can only do worse (it also drops true Oil) -- this is a ceiling for precision/IoU, recall cannot rise.
Reads saved predictions only; writes `oracle_gate.csv` (pooled metrics, all scenes) and a per-scene count table.
"""
import argparse
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from segmentation.evaluation.common import OIL, read_prediction
from segmentation import paths

WIN, STRIDE, X0, Y0 = 256, 128, -15, -59


def window_keep_mask(gt_oil):
    h, w = gt_oil.shape
    keep = np.zeros((h, w), bool)
    for y in range(Y0 - STRIDE * 4, h, STRIDE):
        for x in range(X0 - STRIDE * 4, w, STRIDE):
            ys, ye, xs, xe = max(y, 0), min(y + WIN, h), max(x, 0), min(x + WIN, w)
            if ys < ye and xs < xe and gt_oil[ys:ye, xs:xe].any():
                keep[ys:ye, xs:xe] = True
    return keep


def analyse(task, results_root, labels_root, runs):
    scene, fold = task
    gt = np.asarray(Image.open(Path(labels_root) / f'{scene.removeprefix("train_")}.png')) == OIL
    keep_win = window_keep_mask(gt)
    keep = {'none': np.ones_like(gt), 'scene': np.full(gt.shape, gt.any()), 'window': keep_win}
    rows = []
    for arm, run in runs.items():
        pred = read_prediction(Path(results_root) / run, fold, scene) == OIL
        for gate, k in keep.items():
            p = pred & k
            rows.append({'scene_id': scene, 'arm': arm, 'gate': gate, 'tp': int((p & gt).sum()), 'fp': int((p & ~gt).sum()),
                         'fn': int((~p & gt).sum()), 'closed_px': int((~k).sum())})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    paths.add_root_argument(ap)
    ap.add_argument('--results-root', type=Path, default=paths.RESULTS)
    ap.add_argument('--labels-root', type=Path, default=paths.DATASET / 'train/labels_1D')
    ap.add_argument('--output', type=Path, default=paths.RESULTS / 'error_analysis_b2_seed_replicate/oracle_gate')
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    runs = {'baseline': 'oof_gpu_full_v1', 'segformer_b0_shift': 'oof_stage3_b0_shift_256',
            'segformer_b2_shift': 'oof_stage3_b2_shift_256', 'segformer_b2_shift_s43': 'oof_stage3_b2_shift_seed43_256'}
    base = args.results_root / runs['segformer_b2_shift']
    tasks = [(p.name.removesuffix('_prediction.npy'), fold) for fold in range(1, 6)
             for p in sorted((base / f'fold{fold:02d}').glob('*_prediction.npy'))]
    with Pool(args.workers) as pool:
        rows = [r for part in pool.map(partial(analyse, results_root=args.results_root, labels_root=args.labels_root,
                                               runs=runs), tasks, chunksize=4) for r in part]
    df = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output / 'oracle_gate_by_scene.csv', index=False)
    out = []
    for gate in ['none', 'scene', 'window']:
        for arm in runs:
            s = df[(df.arm == arm) & (df.gate == gate)][['tp', 'fp', 'fn', 'closed_px']].sum()
            out.append({'arm': arm, 'gate': gate, 'oil_recall': s.tp / (s.tp + s.fn), 'oil_precision': s.tp / (s.tp + s.fp),
                        'oil_iou': s.tp / (s.tp + s.fp + s.fn), 'fp_px': int(s.fp)})
    res = pd.DataFrame(out)
    res.to_csv(args.output / 'oracle_gate.csv', index=False)
    print(len(tasks), 'scenes')
    print(res.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
