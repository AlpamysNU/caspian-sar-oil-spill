"""Pooled metrics and paired scene bootstrap of the out-of-fold (OOF) predictions, for the four comparisons of the thesis.

    python -m segmentation.evaluation.bootstrap revised            # B2 trained on revised labels vs original labels (+ seed 43 noise floor)
    python -m segmentation.evaluation.bootstrap hrda_c             # HRDA-C vs B2 (seed 42, seed 43, two-seed ensemble)
    python -m segmentation.evaluation.bootstrap hrda_d             # HRDA-D vs HRDA-C and B2
    python -m segmentation.evaluation.bootstrap ensemble_deeplab   # DeepLabV3+, ensembles and flip-TTA vs B2 seed 42

Each OOF directory holds `fold01..fold05/<scene>_prediction.npy`. A 5x5 confusion matrix is computed per scene against the original
labels (`labels_1D`) and, where a comparison needs it, against the revised labels, and cached as `by_scene_<model>.csv`. The 95%
intervals come from 2,000 resamples of scenes (seed 0), identical for both models of a pair. Bars (Oil IoU gain above the seed noise
of 0.004, interval above 0) were fixed before scoring; this script only reports the numbers.
"""
import argparse
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image

from segmentation import paths

WIND = ['img_0216', 'img_0537', 'img_0556', 'img_0444', 'img_0948']   # scenes with possible wind-farm lattices, kept as Ship
COLS = [f'c{i}' for i in range(25)]
REVISED = paths.RESULTS / 'label_review/revised_manual/labels_1D_revised_full'
FLAGS = paths.RESULTS / 'error_analysis_3models/scene_comparison_wide.csv'   # 73 label-review scenes of the Stage-3 error map
OOF = {
    'b2_s42': 'oof_stage3_b2_shift_256', 'b2_s43': 'oof_stage3_b2_shift_seed43_256', 'b0_shift': 'oof_stage3_b0_shift_256',
    'deeplab': 'oof_stage3_deeplabv3plus_resnet50_shift_256', 'hrda_c': 'oof_stage3_b2_hrda_c_shift_256',
    'hrda_d': 'oof_stage3_b2_hrda_d_shift_256', 'A0': 'oof_ens_a0', 'A_tta': 'oof_ens_a', 'B_ens2': 'oof_ens_b',
    'C_ens2_tta': 'oof_ens_c', 'D_ens4_tta': 'oof_ens_d',
    'orig_b2_s42': 'oof_stage3_b2_shift_256', 'orig_b2_s43': 'oof_stage3_b2_shift_seed43_256',
    'rev_b2_s42': 'oof_stage3_b2_shift_revised_256', 'rev_b2_s43': 'oof_stage3_b2_shift_revised_seed43_256',
}


def cm(gt, pred):
    return np.bincount(gt.ravel().astype(np.int64) * 5 + pred.ravel(), minlength=25)


def scene_rows(task):
    fold, path = task
    scene = path.name.removesuffix('_prediction.npy').removeprefix('train_')
    pred = np.load(path)
    gts = {'orig': np.asarray(Image.open(paths.DATASET / f'train/labels_1D/{scene}.png')), 'rev': np.asarray(Image.open(REVISED / f'{scene}.png'))}
    return [{'scene': scene, 'fold': fold, 'gt': g, **{f'c{i}': v for i, v in enumerate(cm(gt, pred))}} for g, gt in gts.items()]


def collect(model, cache_dir):
    """Per-scene confusion rows (original and revised labels) of one OOF directory, cached in `cache_dir`."""
    cache = cache_dir / f'by_scene_{model}.csv'
    if cache.exists():
        return pd.read_csv(cache)
    base = paths.RESULTS / OOF[model]
    tasks = [(f, p) for f in range(1, 6) for p in sorted((base / f'fold{f:02d}').glob('*_prediction.npy'))]
    with Pool(8) as pool:
        rows = [r for part in pool.imap_unordered(scene_rows, tasks, chunksize=8) for r in part]
    df = pd.DataFrame(rows).sort_values(['gt', 'scene'])
    df.to_csv(cache, index=False)
    return df


def metrics(m):
    m = m.reshape(5, 5).astype(float)
    tp = np.diag(m); gt = m.sum(1); pr = m.sum(0)
    iou = tp / np.maximum(gt + pr - tp, 1)
    return {'oil_recall': tp[1] / gt[1], 'oil_precision': tp[1] / pr[1], 'oil_iou': iou[1], 'look_alike_iou': iou[2], 'ship_iou': iou[3], 'ship_recall': tp[3] / gt[3],
            'miou': iou.mean(), 'oil_to_sea': m[1, 0] / gt[1], 'sea_to_oil': m[0, 1] / gt[0]}


def interval(values):
    return np.percentile(values, [2.5, 97.5])


def paired_rows(A, B, ii, extra):
    """Rows of a paired bootstrap of every metric: A and B are per-scene confusion arrays, ii the resample indices."""
    pa, pb = metrics(A.sum(0)), metrics(B.sum(0))
    d = {k: [] for k in pa}
    for r in ii:
        x, y = metrics(A[r].sum(0)), metrics(B[r].sum(0))
        for k in d:
            d[k].append(x[k] - y[k])
    rows = []
    for k, v in d.items():
        lo, hi = interval(v)
        rows.append({**extra, 'metric': k, 'a': pa[k], 'b': pb[k], 'diff': pa[k] - pb[k], 'lo': lo, 'hi': hi})
    return rows


def per_scene(models, cache_dir):
    data = {k: collect(k, cache_dir) for k in models}
    return {(k, g): df[df['gt'] == g].set_index('scene').sort_index() for k, df in data.items() for g in ('orig', 'rev')}


def print_table(df, metric_list, **where):
    pd.set_option('display.width', 200)
    for k in metric_list:
        mask = df.metric == k
        for col, val in where.items():
            mask &= df[col] == val
        print(k)
        print(df[mask].drop(columns=['metric']).round(4).to_string(index=False))


# ----------------------------------------------------------------------------- revised-label experiment
def run_revised():
    out = paths.RESULTS / 'error_analysis_b2_revised_labels'
    out.mkdir(parents=True, exist_ok=True)
    models = ['orig_b2_s42', 'orig_b2_s43', 'rev_b2_s42', 'rev_b2_s43']
    table = {}
    for (k, g), d in per_scene(models, out).items():
        assert len(d) == 1002, (k, g, len(d))
        table[(k, g)] = d
    rows = []
    for (k, g), d in table.items():
        rows.append({'model': k, 'labels': g, 'subset': 'all', **metrics(d[COLS].sum().values)})
        rows.append({'model': k, 'labels': g, 'subset': 'no_windfarm5', **metrics(d.drop(index=WIND)[COLS].sum().values)})
    pd.DataFrame(rows).to_csv(out / 'pooled_metrics.csv', index=False)
    rng = np.random.default_rng(0)
    scenes = table[('orig_b2_s42', 'orig')].index
    rng.integers(0, len(scenes), size=(2000, len(scenes)))   # one unused draw: keeps the random stream of the reported numbers
    comps = [('orig', 'rev_b2_s42', 'orig_b2_s42'), ('rev', 'rev_b2_s42', 'orig_b2_s42'), ('orig', 'orig_b2_s43', 'orig_b2_s42'),
             ('orig', 'rev_b2_s42', 'orig_b2_s43'), ('orig', 'rev_b2_s43', 'orig_b2_s43'), ('rev', 'rev_b2_s43', 'orig_b2_s43'),
             ('orig', 'rev_b2_s43', 'rev_b2_s42')]
    boot = []
    for subset in ('all', 'no_windfarm5'):
        keep = np.array([s not in WIND for s in scenes]) if subset != 'all' else np.ones(len(scenes), bool)
        for g, a, b in comps:
            A, B = table[(a, g)][COLS].values[keep], table[(b, g)][COLS].values[keep]
            ii = rng.integers(0, len(A), size=(2000, len(A)))
            boot += paired_rows(A, B, ii, {'subset': subset, 'labels': g, 'model': a, 'minus': b})
    pd.DataFrame(boot).to_csv(out / 'paired_bootstrap.csv', index=False)
    # two-seed mean: (rev s42 - orig s42 + rev s43 - orig s43) / 2 with the same scene resamples
    seedmean = []
    for subset in ('all', 'no_windfarm5'):
        keep = np.array([x not in WIND for x in scenes]) if subset != 'all' else np.ones(len(scenes), bool)
        for g in ('orig', 'rev'):
            M = {m: table[(m, g)][COLS].values[keep] for m in ('orig_b2_s42', 'orig_b2_s43', 'rev_b2_s42', 'rev_b2_s43')}
            ii = np.random.default_rng(1).integers(0, keep.sum(), size=(2000, keep.sum()))

            def mean_gain(m):
                return {k: (m['rev_b2_s42'][k] - m['orig_b2_s42'][k] + m['rev_b2_s43'][k] - m['orig_b2_s43'][k]) / 2 for k in m['rev_b2_s42']}
            pt = mean_gain({n: metrics(a.sum(0)) for n, a in M.items()})
            dist = {k: [] for k in pt}
            for r in ii:
                for k, v in mean_gain({n: metrics(a[r].sum(0)) for n, a in M.items()}).items():
                    dist[k].append(v)
            for k, v in dist.items():
                lo, hi = interval(v)
                seedmean.append({'subset': subset, 'labels': g, 'metric': k, 'diff_mean_of_two_seeds': pt[k], 'lo': lo, 'hi': hi})
    pd.DataFrame(seedmean).to_csv(out / 'two_seed_mean_revised_minus_original.csv', index=False)
    # missed scenes: scenes with ground-truth Oil and no predicted Oil pixel overlapping it
    miss = []
    for (k, g), d in table.items():
        has = d[[f'c{5 + j}' for j in range(5)]].sum(1) > 0
        miss.append({'model': k, 'labels': g, 'scenes_with_oil': int(has.sum()), 'missed_zero_oil_tp': int((has & (d['c6'] == 0)).sum())})
    pd.DataFrame(miss).to_csv(out / 'missed_scenes.csv', index=False)
    print('done', out)


# ----------------------------------------------------------------------------- HRDA-C / HRDA-D against B2 (and each other)
def run_hrda(name, models, pairs, headline):
    out = paths.RESULTS / f'error_analysis_{name}'
    out.mkdir(parents=True, exist_ok=True)
    cache = paths.RESULTS / 'error_analysis_ensemble_deeplab'   # the by_scene tables are shared with the ensemble comparison
    cache.mkdir(parents=True, exist_ok=True)
    T = {k: collect(k, cache).query("gt == 'orig'").set_index('scene').sort_index() for k in models}
    scenes = T[models[0]].index
    flags = pd.read_csv(FLAGS, usecols=['scene_id', 'suspicious_label']).set_index('scene_id').suspicious_label.astype(bool)
    flagged = np.array([bool(flags.get(sc, flags.get('train_' + sc, False))) for sc in scenes])
    folds = T[models[0]]['fold'].values
    rng = np.random.default_rng(0)
    subsets = {'all': np.ones(len(scenes), bool), 'no_flagged73': ~flagged}
    for f in range(1, 6):
        subsets[f'fold{f}'] = folds == f
    rows = []
    for a, b in pairs:
        for subset, keep in subsets.items():
            A, B = T[a][COLS].values[keep], T[b][COLS].values[keep]
            ii = rng.integers(0, len(A), size=(2000, len(A)))
            rows += paired_rows(A, B, ii, {'model': a, 'minus': b, 'subset': subset, 'n_scenes': int(keep.sum())})
    df = pd.DataFrame(rows)
    df.to_csv(out / 'paired_bootstrap.csv', index=False)
    print(f'flagged scenes matched: {int(flagged.sum())}')
    metric_list = ('oil_iou', 'oil_recall', 'oil_precision', 'look_alike_iou', 'ship_iou', 'miou', 'oil_to_sea', 'sea_to_oil')
    print_table(df[['model', 'minus', 'subset', 'n_scenes', 'metric', 'a', 'b', 'diff', 'lo', 'hi']], metric_list, minus=headline)


# ----------------------------------------------------------------------------- DeepLabV3+, ensembles and flip-TTA
def run_ensemble_deeplab():
    out = paths.RESULTS / 'error_analysis_ensemble_deeplab'
    out.mkdir(parents=True, exist_ok=True)
    models = ['b2_s42', 'b2_s43', 'b0_shift', 'deeplab', 'A0', 'A_tta', 'B_ens2', 'C_ens2_tta', 'D_ens4_tta']
    table = per_scene(models, out)
    scenes = table[('b2_s42', 'orig')].index
    comps = [('orig', m, 'b2_s42') for m in ('b0_shift', 'deeplab', 'A0', 'A_tta', 'B_ens2', 'C_ens2_tta', 'D_ens4_tta')]
    comps += [('orig', 'B_ens2', 'b2_s43'), ('orig', 'C_ens2_tta', 'B_ens2'), ('orig', 'C_ens2_tta', 'A_tta'),
              ('rev', 'D_ens4_tta', 'b2_s42'), ('rev', 'D_ens4_tta', 'C_ens2_tta'), ('rev', 'B_ens2', 'b2_s42')]
    rng = np.random.default_rng(0)
    ii = rng.integers(0, len(scenes), size=(2000, len(scenes)))
    rows = []
    for g, a, b in comps:
        rows += paired_rows(table[(a, g)][COLS].values, table[(b, g)][COLS].values, ii, {'labels': g, 'model': a, 'minus': b})
    df = pd.DataFrame(rows)
    df.to_csv(out / 'paired_bootstrap.csv', index=False)
    for k in ('oil_iou', 'oil_recall', 'oil_precision', 'look_alike_iou', 'ship_iou'):
        print(k)
        print(df[df.metric == k][['labels', 'model', 'minus', 'a', 'b', 'diff', 'lo', 'hi']].round(4).to_string(index=False))


COMPARISONS = {
    'revised': run_revised,
    'hrda_c': lambda: run_hrda('hrda_c', ['b2_s42', 'b2_s43', 'hrda_c', 'B_ens2'],
                               [('hrda_c', 'b2_s42'), ('hrda_c', 'b2_s43'), ('hrda_c', 'B_ens2')], 'b2_s42'),
    'hrda_d': lambda: run_hrda('hrda_d', ['b2_s42', 'b2_s43', 'hrda_c', 'hrda_d'],
                               [('hrda_d', 'hrda_c'), ('hrda_d', 'b2_s42'), ('hrda_d', 'b2_s43')], 'hrda_c'),
    'ensemble_deeplab': run_ensemble_deeplab,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    paths.add_root_argument(parser)
    parser.add_argument('comparison', choices=sorted(COMPARISONS))
    COMPARISONS[parser.parse_args().comparison]()


if __name__ == '__main__':
    main()
