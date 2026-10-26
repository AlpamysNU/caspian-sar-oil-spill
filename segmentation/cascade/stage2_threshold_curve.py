"""Stage-3 result as a function of the Stage-2 window threshold, split by true window content (2026-10-03, step 1 of the soft-probability check).

Same pixel emulation as stage2_window_accuracy.py (stored patch geometry, saved Stage-3 OOF label maps) but the window decision is
`prob >= t` over a grid of t instead of one operating point, for the two Stage-2 models. Variants: `gate_oil` (Oil kept only where a
covering window has p_oil >= t; low t = suppress only clearly unlikely Oil) and `type` (Oil and Look-alike both checked with the same t,
swap to the other class if allowed, else Sea). Counts are also split by the true content of the window whose centre cell contains the pixel
(neither / oil_only / la_only / both). Nothing is tuned; the whole curve is reported. Writes `stage2_threshold_curve.csv` and
`stage2_threshold_curve_by_content.csv`.
"""
import argparse
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from segmentation.evaluation.common import OIL, read_prediction
from segmentation.cascade.stage2_window_accuracy import LA, SEA, SIZE, STAGE2, allowed, geometry, load_stage2
from segmentation import paths

ARMS = {'segformer_b0_shift': 'oof_stage3_b0_shift_256', 'segformer_b2_shift': 'oof_stage3_b2_shift_256',
        'segformer_b2_shift_s43': 'oof_stage3_b2_shift_seed43_256'}
THRESHOLDS = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
CONTENT = ['neither', 'oil_only', 'la_only', 'both']


def content_map(shape, origins, codes):
    m = np.zeros(shape, np.int8)
    for (y, x), c in zip(origins, codes):
        m[max(y, 0):y + SIZE, max(x, 0):x + SIZE] = c
    for (y, x), c in zip(origins, codes):
        m[max(y + SIZE // 4, 0):y + 3 * SIZE // 4, max(x + SIZE // 4, 0):x + 3 * SIZE // 4] = c
    return m


def counts(v, gt, strat):
    out = []
    for cls in (OIL, LA):
        p, g = v == cls, gt == cls
        for m in (p & g, p & ~g, ~p & g):
            out.append(np.bincount(strat[m], minlength=4))
    return np.stack(out)   # 6 x 4: oil tp, fp, fn, la tp, fp, fn


def analyse(task, results_root, labels_root, windows):
    scene, fold = task
    gt = np.asarray(Image.open(Path(labels_root) / f'{scene.removeprefix("train_")}.png'))
    preds = {arm: read_prediction(Path(results_root) / run, fold, scene) for arm, run in ARMS.items()}
    rows = []
    for model, w in windows.items():
        w = w[w.scene_id == scene]
        if w.empty:
            continue
        origins = [geometry(p) for p in w.patch_id]
        po, pl = w.prob_oil.to_numpy(), w.prob_look_alike.to_numpy()
        codes = (w.true_oil.to_numpy() == 1).astype(int) + 2 * (w.true_look_alike.to_numpy() == 1)
        strat = content_map(gt.shape, origins, codes)
        for arm, pred in preds.items():
            rows.append(('none', -1.0, model, arm, counts(pred, gt, strat)))
        for t in THRESHOLDS:
            ok_oil, ok_la = allowed(gt.shape, origins, po >= t), allowed(gt.shape, origins, pl >= t)
            bad_oil_any, bad_la_any = ~ok_oil, ~ok_la
            for arm, pred in preds.items():
                g = pred.copy()
                g[(pred == OIL) & bad_oil_any] = SEA
                bad_oil, bad_la = (pred == OIL) & bad_oil_any, (pred == LA) & bad_la_any
                ty = pred.copy()
                ty[bad_oil] = np.where(ok_la[bad_oil], LA, SEA)
                ty[bad_la] = np.where(ok_oil[bad_la], OIL, SEA)
                rows.append(('gate_oil', t, model, arm, counts(g, gt, strat)))
                rows.append(('type', t, model, arm, counts(ty, gt, strat)))
    return rows


def metrics(c):
    oil_tp, oil_fp, oil_fn, la_tp, la_fp, la_fn = c
    d = lambda a, b: a / b if b else np.nan
    return {'oil_recall': d(oil_tp, oil_tp + oil_fn), 'oil_precision': d(oil_tp, oil_tp + oil_fp),
            'oil_iou': d(oil_tp, oil_tp + oil_fp + oil_fn), 'la_recall': d(la_tp, la_tp + la_fn),
            'la_precision': d(la_tp, la_tp + la_fp), 'la_iou': d(la_tp, la_tp + la_fp + la_fn)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    paths.add_root_argument(ap)
    ap.add_argument('--stage2-root', type=Path, default=paths.STAGE2_RESULTS / 'controlled_runs_fix1')
    ap.add_argument('--results-root', type=Path, default=paths.RESULTS)
    ap.add_argument('--labels-root', type=Path, default=paths.DATASET / 'train/labels_1D')
    ap.add_argument('--output', type=Path, default=paths.RESULTS / 'error_analysis_b2_seed_replicate/stage2_window_accuracy')
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    windows = {m: load_stage2(args.stage2_root, m) for m in STAGE2}
    base = args.results_root / ARMS['segformer_b2_shift']
    tasks = [(p.name.removesuffix('_prediction.npy'), fold) for fold in range(1, 6)
             for p in sorted((base / f'fold{fold:02d}').glob('*_prediction.npy'))]
    total = {}
    with Pool(args.workers) as pool:
        for part in pool.imap_unordered(partial(analyse, results_root=args.results_root, labels_root=args.labels_root,
                                                windows=windows), tasks, chunksize=4):
            for variant, t, model, arm, c in part:
                key = (variant, t, model, arm)
                total[key] = total.get(key, 0) + c
    pooled, bycontent = [], []
    for (variant, t, model, arm), c in total.items():
        key = {'variant': variant, 'threshold': t, 'stage2': model, 'arm': arm}
        pooled.append({**key, **metrics(c.sum(axis=1))})
        for i, name in enumerate(CONTENT):
            bycontent.append({**key, 'content': name, 'oil_tp': c[0, i], 'oil_fp': c[1, i], 'oil_fn': c[2, i], **metrics(c[:, i])})
    args.output.mkdir(parents=True, exist_ok=True)
    pooled, bycontent = pd.DataFrame(pooled), pd.DataFrame(bycontent)
    pooled.to_csv(args.output / 'stage2_threshold_curve.csv', index=False)
    bycontent.to_csv(args.output / 'stage2_threshold_curve_by_content.csv', index=False)
    pd.set_option('display.width', 200)
    for model in STAGE2:
        s = pooled[(pooled.arm == 'segformer_b2_shift') & (pooled.stage2 == model)]
        print(f'\n== {model}, B2 seed 42 (no gate: {metrics(total[("none", -1.0, model, "segformer_b2_shift")].sum(axis=1))})')
        print(s.sort_values(['variant', 'threshold']).round(4).to_string(index=False))
    print(f'{len(tasks)} scenes')


if __name__ == '__main__':
    main()
