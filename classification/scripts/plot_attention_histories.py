"""Training curves and run statistics of the controlled Stage-2 runs (plain, CBAM, CBAM+Transformer, ConvNeXt-Tiny).

Reads history.csv/metrics.json copied to results/attention_histories (rsync from the HPC run root); writes a figure and a table.
Run: python classification/scripts/plot_attention_histories.py   (from the repository root)
"""
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[2]))   # repository root
from segmentation import paths  # noqa: E402
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = paths.STAGE2_RESULTS / 'attention_histories'
OUT = paths.STAGE2_RESULTS / 'attention_analysis_2026-10-18'
OUT.mkdir(parents=True, exist_ok=True)
MODELS = [('effnet_b2', 'EfficientNet-B2', '#0072B2'), ('effnet_b2_cbam', 'B2 + CBAM', '#E69F00'),
          ('effnet_b2_cbam_transformer', 'B2 + CBAM + Transformer', '#D55E00'), ('convnext_tiny', 'ConvNeXt-Tiny', '#009E73')]
SIZES = (256, 512)


def load(model, size):
    runs = []
    for fold in range(1, 6):
        d = ROOT / f'{model}_p{size}_fold{fold:02d}_seed42'
        runs.append((pd.read_csv(d / 'history.csv'), json.load(open(d / 'metrics.json'))))
    return runs


plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5})
fig, axes = plt.subplots(4, 2, figsize=(6.5, 7.6), sharex=True, sharey='row')
rows = []
for c, size in enumerate(SIZES):
    for model, label, color in MODELS:
        runs = load(model, size)
        h = [r[0] for r in runs]
        n = min(len(x) for x in h)
        stack = lambda col: np.stack([x[col].to_numpy()[:n] for x in h])
        ep = np.arange(1, n + 1)
        for r, (col, title) in enumerate([('oil_f2', 'Validation Oil F2'),
                                          ('oil_ap', 'Validation Oil AP'),
                                          ('validation_loss', 'Validation loss'), ('train_loss', 'Training loss')]):
            v = stack(col); m = v.mean(0)
            axes[r, c].plot(ep, m, color=color, label=label, lw=1.6)
            axes[r, c].fill_between(ep, v.min(0), v.max(0), color=color, alpha=.12)
            axes[r, c].set_ylabel(title) if c == 0 else None
            if r == 0: axes[r, c].set_title(f'{size} px windows')
        best = [r[1]['best_epoch'] for r in runs]
        f2 = [x['oil_f2'].max() for x in h]
        last = [x['oil_f2'].iloc[-1] for x in h]
        vl = stack('validation_loss'); tl = stack('train_loss')
        rows.append(dict(size=size, model=label, best_epoch_mean=np.mean(best), best_epoch_min=min(best), best_epoch_max=max(best),
                         f2_best_epoch=float(np.mean([r[1]['best']['selected']['f2'] for r in runs])),
                         f2_max_any_epoch=float(np.mean(f2)), f2_last_epoch=float(np.mean(last)),
                         val_loss_min=float(vl.min(1).mean()), val_loss_last=float(vl[:, -1].mean()), train_loss_last=float(tl[:, -1].mean()),
                         epoch_seconds=float(np.mean([x['epoch_seconds'].mean() for x in h])),
                         params_trainable=int(runs[0][1]['best']['trainable_parameters'])))
for ax in axes[-1]:
    ax.set_xlabel('Epoch')
for ax in axes.ravel():
    ax.grid(alpha=.25)
axes[0, 0].legend(fontsize=6.5, frameon=False, loc='lower right')
for ax, L in zip(axes.T.ravel(), 'ACEGBDFH'):
    pass
for i, ax in enumerate(axes.ravel()):
    ax.text(0.03, 0.95, 'ABCDEFGH'[i], transform=ax.transAxes, fontsize=10, fontweight='bold', va='top', ha='left', bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none', alpha=0.85))
fig.tight_layout()
fig.savefig(OUT / 'training_curves.png', dpi=300)
t = pd.DataFrame(rows)
t.to_csv(OUT / 'run_statistics.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(t.round(4).to_string(index=False))
