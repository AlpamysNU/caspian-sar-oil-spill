"""Real Stage-2 window classifiers vs the oracle ceilings (2026-10-03).

Part 1: window-level accuracy of the multi-class Stage-2 classifiers (EfficientNet-B2, ConvNeXt-Tiny, 256 px, five-fold OOF
`val_predictions.parquet`, 45,090 windows) for Oil and Look-alike: AP, ROC-AUC, precision/recall at the stored operational
Oil threshold and at 0.5, and the window content table (neither / Oil only / Look-alike only / both).
Part 2: the oracle rules of oracle_window_class.py driven by the *predicted* window labels instead of the true ones, applied to
the saved Stage-3 OOF predictions through the stored patch geometry (patch_id `<scene>_y<Y>_x<X>`, 256 px). No threshold is tuned.
Writes `stage2_window_accuracy.csv`, `stage2_window_content.csv` and `stage2_real_vs_oracle.csv`.
"""
import argparse
import re
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def average_precision_score(y, p):
    order = np.argsort(-p, kind='mergesort')
    y = y[order]
    tp = np.cumsum(y)
    return float((tp / np.arange(1, len(y) + 1))[y == 1].mean())


def roc_auc_score(y, p):
    ranks = pd.Series(p).rank().to_numpy()
    n1, n0 = int(y.sum()), len(y) - int(y.sum())
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

from segmentation.evaluation.common import OIL, read_prediction
from segmentation import paths

LA, SEA, SIZE = 2, 0, 256
STAGE2 = {'convnext_tiny': 'convnext_tiny_p256_fold{f:02d}_seed42', 'effnet_b2': 'effnet_b2_p256_fold{f:02d}_seed42'}
STAGE3 = {'baseline': 'oof_gpu_full_v1', 'segformer_b0_shift': 'oof_stage3_b0_shift_256',
          'segformer_b2_shift': 'oof_stage3_b2_shift_256', 'segformer_b2_shift_s43': 'oof_stage3_b2_shift_seed43_256'}


def load_stage2(root, model):
    return pd.concat([pd.read_parquet(root / STAGE2[model].format(f=f) / 'val_predictions.parquet') for f in range(1, 6)],
                     ignore_index=True)


def window_tables(root):
    acc, content = [], []
    for model in STAGE2:
        v = load_stage2(root, model)
        for cls in ('oil', 'look_alike'):
            y, p = v[f'true_{cls}'].to_numpy(), v[f'prob_{cls}'].to_numpy()
            for rule, pred in (('stored_threshold' if cls == 'oil' else 'stored_pred', v[f'pred_{cls}'].to_numpy().astype(bool)),
                               ('prob>=0.5', p >= .5)):
                tp, fp, fn = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum()), int((~pred & (y == 1)).sum())
                acc.append({'model': model, 'class': cls, 'rule': rule, 'windows': len(v), 'positives': int(y.sum()),
                            'recall': tp / (tp + fn), 'precision': tp / (tp + fp), 'ap': average_precision_score(y, p),
                            'roc_auc': roc_auc_score(y, p), 'predicted_positive': int(pred.sum())})
        truth = np.select([(v.true_oil == 1) & (v.true_look_alike == 1), v.true_oil == 1, v.true_look_alike == 1],
                          ['both', 'oil_only', 'la_only'], 'neither')
        pred = np.select([(v.pred_oil == 1) & (v.pred_look_alike == 1), v.pred_oil == 1, v.pred_look_alike == 1],
                         ['both', 'oil_only', 'la_only'], 'neither')
        ct = pd.crosstab(pd.Series(truth, name='true'), pd.Series(pred, name='pred'))
        ct.insert(0, 'model', model)
        content.append(ct.reset_index())
    return pd.DataFrame(acc), pd.concat(content, ignore_index=True)


def geometry(patch_id):
    m = re.search(r'_y(-?\d+)_x(-?\d+)$', patch_id)
    return int(m.group(1)), int(m.group(2))


def allowed(shape, origins, flags):
    ok = np.zeros(shape, bool)
    for (y, x), f in zip(origins, flags):
        if f:
            ok[max(y, 0):y + SIZE, max(x, 0):x + SIZE] = True
    return ok


def analyse(task, results_root, labels_root, runs, windows):
    scene, fold = task
    gt = np.asarray(Image.open(Path(labels_root) / f'{scene.removeprefix("train_")}.png'))
    rows = []
    for model, w in windows.items():
        w = w[w.scene_id == scene]
        if w.empty:
            continue
        origins = [geometry(p) for p in w.patch_id]
        for mode, (oil_flag, la_flag) in {'real_th': (w.pred_oil.to_numpy() == 1, w.pred_look_alike.to_numpy() == 1),
                                          'real_05': (w.prob_oil.to_numpy() >= .5, w.prob_look_alike.to_numpy() >= .5),
                                          'oracle': (w.true_oil.to_numpy() == 1, w.true_look_alike.to_numpy() == 1)}.items():
            ok_oil, ok_la = allowed(gt.shape, origins, oil_flag), allowed(gt.shape, origins, la_flag)
            for arm, run in runs.items():
                pred = read_prediction(Path(results_root) / run, fold, scene)
                g, t = pred.copy(), pred.copy()
                g[(pred == OIL) & ~ok_oil] = SEA
                bad_oil, bad_la = (pred == OIL) & ~ok_oil, (pred == LA) & ~ok_la
                t[bad_oil] = np.where(ok_la[bad_oil], LA, SEA)
                t[bad_la] = np.where(ok_oil[bad_la], OIL, SEA)
                for variant, v in (('gate_oil', g), ('type', t)):
                    row = {'stage2': model, 'mode': mode, 'arm': arm, 'variant': variant}
                    for cls, label in ((OIL, 'oil'), (LA, 'la')):
                        p_, g_ = v == cls, gt == cls
                        row.update({f'{label}_tp': int((p_ & g_).sum()), f'{label}_fp': int((p_ & ~g_).sum()),
                                    f'{label}_fn': int((~p_ & g_).sum())})
                    rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    paths.add_root_argument(ap)
    ap.add_argument('--stage2-root', type=Path, default=paths.STAGE2_RESULTS / 'controlled_runs_fix1')
    ap.add_argument('--results-root', type=Path, default=paths.RESULTS)
    ap.add_argument('--labels-root', type=Path, default=paths.DATASET / 'train/labels_1D')
    ap.add_argument('--output', type=Path, default=paths.RESULTS / 'error_analysis_b2_seed_replicate/stage2_window_accuracy')
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    acc, content = window_tables(args.stage2_root)
    acc.to_csv(args.output / 'stage2_window_accuracy.csv', index=False)
    content.to_csv(args.output / 'stage2_window_content.csv', index=False)
    print(acc.round(4).to_string(index=False))
    print(content.to_string(index=False))
    windows = {m: load_stage2(args.stage2_root, m) for m in STAGE2}
    base = args.results_root / STAGE3['segformer_b2_shift']
    tasks = [(p.name.removesuffix('_prediction.npy'), fold) for fold in range(1, 6)
             for p in sorted((base / f'fold{fold:02d}').glob('*_prediction.npy'))]
    with Pool(args.workers) as pool:
        rows = [r for part in pool.map(partial(analyse, results_root=args.results_root, labels_root=args.labels_root,
                                               runs=STAGE3, windows=windows), tasks, chunksize=4) for r in part]
    df = pd.DataFrame(rows)
    out = []
    for keys, s in df.groupby(['stage2', 'mode', 'arm', 'variant']):
        s = s[['oil_tp', 'oil_fp', 'oil_fn', 'la_tp', 'la_fp', 'la_fn']].sum()
        out.append({**dict(zip(['stage2', 'mode', 'arm', 'variant'], keys)),
                    'oil_recall': s.oil_tp / (s.oil_tp + s.oil_fn), 'oil_precision': s.oil_tp / (s.oil_tp + s.oil_fp),
                    'oil_iou': s.oil_tp / (s.oil_tp + s.oil_fp + s.oil_fn),
                    'la_iou': s.la_tp / (s.la_tp + s.la_fp + s.la_fn)})
    res = pd.DataFrame(out)
    res.to_csv(args.output / 'stage2_real_vs_oracle.csv', index=False)
    print(res[res.arm == 'segformer_b2_shift'].round(4).to_string(index=False))


if __name__ == '__main__':
    main()
