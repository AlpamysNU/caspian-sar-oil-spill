"""Calibration of the pixel-level Oil probability of HRDA-C (window-averaged softmax) and Oil probability heat maps (2026-10-15).

Input: results/oof_stage3_b2_hrda_c_shift_probs_256 (OOF, saved with --save-probs: Oil and Look-alike probabilities per scene, float16).
Reliability table (per probability bin: pixels, mean predicted Oil probability, observed Oil share in the ORIGINAL labels), Brier score and
expected calibration error (ECE) pooled, per fold, and without the 73 flagged scenes. Heat maps: SAR | labels | argmax prediction | Oil probability.
No tuning on these data: the figures describe the model as it is. Writes into results/error_analysis_hrda_c/calibration/.
"""
from multiprocessing import Pool

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from segmentation import paths

ROOT = paths.DATASET
OOF = paths.RESULTS / 'oof_stage3_b2_hrda_c_shift_probs_256'
OUT = paths.RESULTS / 'error_analysis_hrda_c/calibration'
EDGES = np.array([0, .01, .02, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .98, .99, 1.0 + 1e-6])
ALPHA = 0.8
COLORS = np.array([[0, 0, 0], [0, 255, 255], [255, 0, 0], [153, 76, 0], [0, 153, 0]], dtype=np.uint8)  # Sea, Oil (cyan), Look-alike (red), Ship (brown), Land (green): the dataset mask colours


def scene_bins(task):
    fold, path = task
    scene = path.name.removesuffix('_probs_oil_la.npy').removeprefix('train_')
    p = np.load(path)[0].astype(np.float64)
    y = (np.asarray(Image.open(ROOT / f'train/labels_1D/{scene}.png')) == 1)
    b = np.clip(np.digitize(p.ravel(), EDGES) - 1, 0, len(EDGES) - 2)
    k = len(EDGES) - 1
    return {'scene': scene, 'fold': fold, 'n': np.bincount(b, minlength=k), 'sp': np.bincount(b, weights=p.ravel(), minlength=k),
            'sy': np.bincount(b, weights=y.ravel().astype(float), minlength=k), 'brier': float(((p - y) ** 2).sum())}


def table(rows):
    n = sum(r['n'] for r in rows); sp = sum(r['sp'] for r in rows); sy = sum(r['sy'] for r in rows)
    ok = n > 0
    t = pd.DataFrame({'bin_lo': EDGES[:-1], 'bin_hi': np.minimum(EDGES[1:], 1.0), 'pixels': n, 'mean_pred': np.where(ok, sp / np.maximum(n, 1), np.nan),
                      'observed_oil': np.where(ok, sy / np.maximum(n, 1), np.nan)})
    ece = float((n / n.sum() * np.abs(t.mean_pred.fillna(0) - t.observed_oil.fillna(0))).sum())
    return t, ece, sum(r['brier'] for r in rows) / n.sum()


def calibration():
    tasks = [(f, p) for f in range(1, 6) for p in sorted((OOF / f'fold{f:02d}').glob('*_probs_oil_la.npy'))]
    with Pool(8) as pool:
        rows = pool.map(scene_bins, tasks, chunksize=8)
    flags = pd.read_csv(paths.RESULTS / 'error_analysis_3models/scene_comparison_wide.csv', usecols=['scene_id', 'suspicious_label']).set_index('scene_id').suspicious_label.astype(bool)
    flagged = lambda s: bool(flags.get(s, flags.get('train_' + s, False)))
    subsets = {'all': rows, 'no_flagged73': [r for r in rows if not flagged(r['scene'])]}
    for f in range(1, 6):
        subsets[f'fold{f}'] = [r for r in rows if r['fold'] == f]
    summary = []
    for name, rs in subsets.items():
        t, ece, brier = table(rs)
        t.to_csv(OUT / f'reliability_{name}.csv', index=False)
        summary.append({'subset': name, 'scenes': len(rs), 'ece': ece, 'brier': brier})
    pd.DataFrame(summary).to_csv(OUT / 'calibration_summary.csv', index=False)
    t, _, _ = table(rows)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    ax[0].plot([0, 1], [0, 1], 'k--', lw=1, label='perfect calibration')
    ax[0].plot(t.mean_pred, t.observed_oil, 'o-', color='#0b7a75', label='HRDA-C (pooled OOF)')
    ax[0].set_xlabel('Mean predicted Oil probability (bin)'); ax[0].set_ylabel('Observed Oil share in original labels'); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[0].set_title('Reliability of the pixel-level Oil probability')
    ax[1].bar(range(len(t)), t.pixels, color='#0b7a75'); ax[1].set_yscale('log')
    ax[1].set_xticks(range(len(t))); ax[1].set_xticklabels([f'{lo:g}' for lo in t.bin_lo], rotation=70, fontsize=8)
    ax[1].set_xlabel('Predicted Oil probability, bin lower edge'); ax[1].set_ylabel('Pixels'); ax[1].set_title('Pixels per bin')
    fig.tight_layout(); fig.savefig(OUT / 'reliability_oil.png', dpi=130); plt.close(fig)
    print(pd.DataFrame(summary).round(5).to_string(index=False)); print(t.round(4).to_string(index=False))


def overlay(gray, lab):
    rgb = np.repeat(gray[..., None], 3, axis=2).astype(float)
    m = (lab > 0)[..., None]
    return np.where(m, (1 - ALPHA) * rgb + ALPHA * COLORS[lab], rgb).astype(np.uint8)


def crop(labs, shape, margin=60, size=300):
    m = np.zeros(shape, bool)
    for l in labs:
        m |= np.isin(l, (1, 2, 3))
    if not m.any():
        return slice(0, shape[0]), slice(0, shape[1])
    ys, xs = np.where(m)
    def fit(lo, hi, n):
        lo, hi = lo - margin, hi + margin
        if hi - lo < size:
            c = (lo + hi) // 2; lo, hi = c - size // 2, c + size // 2
        return max(lo, 0), min(hi, n)
    y0, y1 = fit(ys.min(), ys.max(), shape[0]); x0, x1 = fit(xs.min(), xs.max(), shape[1])
    return slice(y0, y1), slice(x0, x1)


def heatmaps():
    sel = pd.read_csv(paths.RESULTS / 'error_analysis_hrda_c/figures/scene_selection.csv')
    groups = {'gains_and_loss': ['img_0468', 'img_0490', 'img_0646'], 'random_seed0': list(sel[sel.group == 'random_seed0'].scene)}
    for key, scenes in groups.items():
        fig, axes = plt.subplots(len(scenes), 4, figsize=(17, 4.3 * len(scenes)))
        for r, sc in enumerate(scenes):
            gray = np.asarray(Image.open(ROOT / f'train/images/{sc}.jpg').convert('L'))
            gt = np.asarray(Image.open(ROOT / f'train/labels_1D/{sc}.png'))
            hit = next(OOF.glob(f'fold*/train_{sc}_prediction.npy'))
            pred = np.load(hit); prob = np.load(str(hit).replace('_prediction.npy', '_probs_oil_la.npy'))[0].astype(np.float32)
            s = crop([gt, pred], gray.shape)
            axes[r, 0].imshow(gray[s], cmap='gray'); axes[r, 0].set_title(f'{sc}  SAR')
            axes[r, 1].imshow(overlay(gray, gt)[s]); axes[r, 1].set_title('Original labels')
            axes[r, 2].imshow(overlay(gray, pred)[s]); axes[r, 2].set_title('HRDA-C prediction (argmax)')
            axes[r, 3].imshow(gray[s], cmap='gray')
            im = axes[r, 3].imshow(prob[s], cmap='magma', vmin=0, vmax=1, alpha=0.65); axes[r, 3].set_title('Oil probability (HRDA-C)')
            for a in axes[r]:
                a.axis('off')
            fig.colorbar(im, ax=axes[r, 3], fraction=0.046, pad=0.02)
        fig.suptitle('Oil probability heat maps of HRDA-C (window-averaged softmax, OOF)', fontsize=12)
        fig.tight_layout(); fig.savefig(OUT / f'oil_probability_{key}.png', dpi=110); plt.close(fig)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    calibration()
    heatmaps()
