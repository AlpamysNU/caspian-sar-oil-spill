"""Oracle window-class upper bound (2026-10-03): what if a window classifier always knew which of Oil / Look-alike a window contains?

Same window grid as Stage 4 (256 px, stride 128, x = -15 + 128k, y = -59 + 128j). A pixel may carry class c in {Oil, Look-alike}
only if at least one covering window contains ground-truth c. A predicted Oil/Look-alike pixel that is not allowed is relabelled to
the other of the two when that one is allowed, otherwise to Sea. Variants: `gate_oil` (only Oil is checked, relabelled to Sea; same as
oracle_gate.py 'window'), `type` (both classes, with the swap). A real classifier can only do worse; the segmenter keeps its
own boundaries. Reads saved predictions only; writes `oracle_window_class.csv` and a per-scene table.
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

LA, SEA = 2, 0
WIN, STRIDE, X0, Y0 = 256, 128, -15, -59


def allowed_mask(gt_class):
    h, w = gt_class.shape
    allowed = np.zeros((h, w), bool)
    for y in range(Y0 - STRIDE * 4, h, STRIDE):
        for x in range(X0 - STRIDE * 4, w, STRIDE):
            ys, ye, xs, xe = max(y, 0), min(y + WIN, h), max(x, 0), min(x + WIN, w)
            if ys < ye and xs < xe and (gt_class[ys:ye, xs:xe]).any():
                allowed[ys:ye, xs:xe] = True
    return allowed


def counts(pred, gt, cls):
    p, g = pred == cls, gt == cls
    return int((p & g).sum()), int((p & ~g).sum()), int((~p & g).sum())


def analyse(task, results_root, labels_root, runs):
    scene, fold = task
    gt = np.asarray(Image.open(Path(labels_root) / f'{scene.removeprefix("train_")}.png'))
    ok_oil, ok_la = allowed_mask(gt == OIL), allowed_mask(gt == LA)
    rows = []
    for arm, run in runs.items():
        pred = read_prediction(Path(results_root) / run, fold, scene)
        variants = {'none': pred}
        g = pred.copy()
        g[(pred == OIL) & ~ok_oil] = SEA
        variants['gate_oil'] = g
        t = pred.copy()
        bad_oil, bad_la = (pred == OIL) & ~ok_oil, (pred == LA) & ~ok_la
        t[bad_oil] = np.where(ok_la[bad_oil], LA, SEA)
        t[bad_la] = np.where(ok_oil[bad_la], OIL, SEA)
        variants['type'] = t
        for name, v in variants.items():
            row = {'scene_id': scene, 'arm': arm, 'variant': name}
            for cls, label in ((OIL, 'oil'), (LA, 'la')):
                tp, fp, fn = counts(v, gt, cls)
                row.update({f'{label}_tp': tp, f'{label}_fp': fp, f'{label}_fn': fn})
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    paths.add_root_argument(ap)
    ap.add_argument('--results-root', type=Path, default=paths.RESULTS)
    ap.add_argument('--labels-root', type=Path, default=paths.DATASET / 'train/labels_1D')
    ap.add_argument('--output', type=Path, default=paths.RESULTS / 'error_analysis_b2_seed_replicate/oracle_window_class')
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
    df.to_csv(args.output / 'oracle_window_class_by_scene.csv', index=False)
    out = []
    for variant in ['none', 'gate_oil', 'type']:
        for arm in runs:
            s = df[(df.arm == arm) & (df.variant == variant)].drop(columns=['scene_id', 'arm', 'variant']).sum()
            out.append({'arm': arm, 'variant': variant,
                        'oil_recall': s.oil_tp / (s.oil_tp + s.oil_fn), 'oil_precision': s.oil_tp / (s.oil_tp + s.oil_fp),
                        'oil_iou': s.oil_tp / (s.oil_tp + s.oil_fp + s.oil_fn),
                        'la_recall': s.la_tp / (s.la_tp + s.la_fn), 'la_precision': s.la_tp / (s.la_tp + s.la_fp),
                        'la_iou': s.la_tp / (s.la_tp + s.la_fp + s.la_fn)})
    res = pd.DataFrame(out)
    res.to_csv(args.output / 'oracle_window_class.csv', index=False)
    print(len(tasks), 'scenes')
    print(res.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
