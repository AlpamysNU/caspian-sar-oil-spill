"""Scoring of the one-time official-test evaluation (segmentation/inference/test_split.py).

Primary result: the 'ensemble5' prediction (mean softmax of the five fold models) of each architecture, pooled over the 110 test
scenes; 95% intervals from a scene bootstrap (2000 draws, seed 0), paired between models on the same resampled scenes.
Secondary: the five single fold models (mean and standard deviation of the pooled metrics).
Metrics: Oil recall first, then Oil precision, Oil IoU, Ship IoU, Look-alike IoU, mIoU; share of Oil called Look-alike.
Writes test_metrics.csv, test_paired.csv and test_summary.txt next to the confusion tables.
"""

import numpy as np
import pandas as pd

from segmentation import paths

ROOT = paths.RESULTS / 'test_split_eval'
ARCHS = {'segformer_b2_hrda_c': 'HRDA-C', 'segformer_b2': 'B2', 'deeplabv3plus_resnet50': 'DeepLabV3+'}
CLASSES = ['Sea', 'Oil', 'Look-alike', 'Ship', 'Land']
COLS = [f'c{i}' for i in range(25)]
OIL_OOF = {'HRDA-C': 0.6603, 'B2': 0.6185, 'DeepLabV3+': 0.5793}   # pooled out-of-fold Oil IoU on the training scenes


def metrics(vec):
    cm = np.asarray(vec, dtype=np.float64).reshape(5, 5)
    tp = np.diag(cm)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    iou = np.where(tp + fp + fn > 0, tp / np.maximum(tp + fp + fn, 1), np.nan)
    return {'oil_recall': tp[1] / max(cm[1].sum(), 1), 'oil_precision': tp[1] / max(cm[:, 1].sum(), 1), 'oil_iou': iou[1],
            'ship_iou': iou[3], 'lookalike_iou': iou[2], 'miou': np.nanmean(iou),
            'oil_to_lookalike': cm[1, 2] / max(cm[1].sum(), 1)}


def main():
    tables = {}
    for arch, label in ARCHS.items():
        path = ROOT / arch / 'confusion_by_scene.csv'
        if path.is_file():
            tables[label] = pd.read_csv(path)
    rng = np.random.default_rng(0)
    rows, lines = [], []
    ens = {}
    for label, df in tables.items():
        e = df[df.prediction == 'ensemble5'].set_index('scene').sort_index()
        ens[label] = e[COLS].values
        m = metrics(e[COLS].values.sum(0))
        n = len(e)
        draws = [metrics(ens[label][rng.integers(0, n, n)].sum(0)) for _ in range(2000)]
        for k, v in m.items():
            lo, hi = np.nanpercentile([d[k] for d in draws], [2.5, 97.5])
            rows.append({'model': label, 'prediction': 'ensemble5', 'metric': k, 'value': v, 'lo': lo, 'hi': hi, 'n_scenes': n})
        folds = [metrics(df[df.prediction == f'fold{i}'][COLS].values.sum(0)) for i in range(1, 6)]
        for k in m:
            vals = [f[k] for f in folds]
            rows.append({'model': label, 'prediction': 'single_folds_mean', 'metric': k, 'value': float(np.mean(vals)),
                         'lo': float(np.min(vals)), 'hi': float(np.max(vals)), 'n_scenes': n})
        lines.append(f'{label}: Oil recall {m["oil_recall"]:.4f}, precision {m["oil_precision"]:.4f}, Oil IoU {m["oil_iou"]:.4f} '
                     f'(OOF on training scenes {OIL_OOF[label]:.4f}), Ship IoU {m["ship_iou"]:.4f}, Look-alike IoU {m["lookalike_iou"]:.4f}, '
                     f'mIoU {m["miou"]:.4f}, Oil->Look-alike {m["oil_to_lookalike"]:.4f}')
    pd.DataFrame(rows).to_csv(ROOT / 'test_metrics.csv', index=False)
    paired = []
    for a, b in (('HRDA-C', 'B2'), ('HRDA-C', 'DeepLabV3+'), ('B2', 'DeepLabV3+')):
        if a in ens and b in ens:
            A, B = ens[a], ens[b]
            n = len(A)
            base = {k: metrics(A.sum(0))[k] - metrics(B.sum(0))[k] for k in metrics(A.sum(0))}
            draws = []
            for _ in range(2000):
                r = rng.integers(0, n, n)
                x, y = metrics(A[r].sum(0)), metrics(B[r].sum(0))
                draws.append({k: x[k] - y[k] for k in x})
            for k in base:
                lo, hi = np.nanpercentile([d[k] for d in draws], [2.5, 97.5])
                paired.append({'model': a, 'minus': b, 'metric': k, 'diff': base[k], 'lo': lo, 'hi': hi})
            lines.append(f'{a} minus {b}: Oil IoU {base["oil_iou"]:+.4f} [{[p for p in paired if p["model"] == a and p["minus"] == b and p["metric"] == "oil_iou"][0]["lo"]:+.4f}, '
                         f'{[p for p in paired if p["model"] == a and p["minus"] == b and p["metric"] == "oil_iou"][0]["hi"]:+.4f}]')
    pd.DataFrame(paired).to_csv(ROOT / 'test_paired.csv', index=False)
    (ROOT / 'test_summary.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
