"""Audit and describe the 20 controlled development runs, without test evaluation.

Run from any directory: python3 classification/scripts/analyze_controlled_runs.py
Only reads original artifacts; writes a separate analysis directory. Dependencies:
numpy, pandas, pyarrow, matplotlib. Checkpoint tensors are not deserialized.
"""
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[2]))   # repository root
from segmentation import paths  # noqa: E402
from pathlib import Path
import argparse
import hashlib
import json
import zipfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
CLASSES = ['oil', 'look_alike', 'land', 'sea', 'ship']
COLORS = ['#0072B2', '#56B4E9', '#D55E00', '#CC79A7']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binary(y, pred):
    y, pred = np.asarray(y, bool), np.asarray(pred, bool)
    tp, fp, fn, tn = [int(x.sum()) for x in
                      [y & pred, ~y & pred, y & ~pred, ~y & ~pred]]
    return dict(n=len(y), positives=int(y.sum()), tp=tp, fp=fp, fn=fn, tn=tn,
                precision=tp/max(tp+fp, 1), recall=tp/max(tp+fn, 1),
                f1=2*tp/max(2*tp+fp+fn, 1), f2=5*tp/max(5*tp+fp+4*fn, 1),
                routed_fraction=float(pred.mean()), prevalence=float(y.mean()))


def pr(y, score, minimum=.9):
    y, score = np.asarray(y, np.int64), np.asarray(score)
    order = np.argsort(-score, kind='stable')
    ends = np.r_[np.flatnonzero(np.diff(score[order])), len(y)-1]
    tp = np.cumsum(y[order])[ends]
    recall, precision = tp/y.sum(), tp/(ends+1)
    f2 = 5*tp/(4*y.sum()+ends+1)
    good = np.flatnonzero(recall >= minimum)
    best = max(good, key=lambda j: (f2[j], precision[j], recall[j], score[order[ends[j]]]))
    ap = float(np.dot(np.diff(np.r_[0, recall]), precision))
    return float(score[order[ends[best]]]), ap, recall, precision


def md_table(frame):
    def fmt(x):
        return f'{x:.4f}' if isinstance(x, (float, np.floating)) else str(x)
    return '\n'.join(['| ' + ' | '.join(map(str, frame.columns)) + ' |',
                      '| ' + ' | '.join(['---']*len(frame.columns)) + ' |'] +
                     ['| ' + ' | '.join(fmt(x) for x in row) + ' |'
                      for row in frame.itertuples(index=False, name=None)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=paths.STAGE2_RESULTS/'controlled_runs_fix1')
    parser.add_argument('--output', type=Path, default=paths.STAGE2_RESULTS/'controlled_analysis_2026-09-13')
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = paths.STAGE2_RESULTS/'grouped_folds_v1/scene_folds.csv'
    audit_path = manifest_path.with_name('audit.json')
    manifest = pd.read_csv(manifest_path)
    frozen = json.loads(audit_path.read_text())
    assert sha(manifest_path) == frozen['manifest_sha256']
    assert set(manifest.split) == {'train'} and not manifest.scene_id.duplicated().any()
    for col in ['scene_id', 'group_id', 'image_sha256']:
        assert manifest.groupby(col).fold.nunique().max() == 1
    indices = {}
    expected_runs = {f'{model}_p{size}_fold{fold:02d}_seed42'
                     for model in ['effnet_b2', 'convnext_tiny'] for size in [256, 512] for fold in range(1, 6)}
    assert {p.name for p in args.runs.iterdir() if p.is_dir() and not p.name.startswith('smoke_')} == expected_runs
    for size in [256, 512]:
        root = paths.STAGE2_RESULTS/f'training_patches/ps{size}_overlap50'
        for path in [root/f'index_ps{size}_overlap50.parquet', root/'scene_manifest.csv', root/'verification.json']:
            matching = [v for k, v in frozen['input_sha256'].items()
                        if Path(k).parts[-2:] == path.parts[-2:]]
            assert matching == [sha(path)], path
        index = pd.read_parquet(root/f'index_ps{size}_overlap50.parquet')
        assert set(index.split) == {'train'} and not index.patch_id.duplicated().any()
        indices[size] = index.merge(manifest[['scene_id', 'fold', 'group_id']], on='scene_id', validate='many_to_one')
    rows, histories, audits, all_pred, all_config, epochs_audit = [], [], [], {}, [], []
    for model in ['effnet_b2', 'convnext_tiny']:
        for size in [256, 512]:
            key = f'{model}_p{size}'
            frames = []
            for fold in range(1, 6):
                name = f'{key}_fold{fold:02d}_seed42'
                root = args.runs/name
                required = ['config.json', 'metrics.json', 'history.json', 'history.csv',
                            'last.pt', 'best_operational.pt', 'best_pr_auc.pt',
                            'val_predictions.parquet', 'oil_false_negatives.parquet', 'oil_false_positives.parquet']
                assert all((root/f).is_file() and (root/f).stat().st_size for f in required), name
                for f in ['last.pt', 'best_operational.pt', 'best_pr_auc.pt']:
                    with zipfile.ZipFile(root/f) as z:
                        assert len(z.infolist()) > 10  # Central directory only; no tensor/CRC audit.
                config = json.loads((root/'config.json').read_text())
                all_config.append(config)
                c = config['config']
                m = json.loads((root/'metrics.json').read_text())
                h = json.loads((root/'history.json').read_text())
                assert (c['model'], c['patch_size'], c['fold'], c['seed']) == (model, size, fold, 42)
                assert c['epochs'] == 50 and c['adaptation_epochs'] == 5
                assert not c['smoke'] and not c['max_train_batches'] and not c['max_val_batches']
                assert c['batch_size']*c['accumulation_steps'] == 64
                assert m['status'] == 'complete' and m['completed_epochs'] == 50 and not m['smoke']
                assert len(h) == 50 and [x['epoch'] for x in h] == list(range(1, 51))
                csv_history = pd.read_csv(root/'history.csv')
                assert len(csv_history) == 50
                for field, values in [('oil_f2', [x['selected']['f2'] for x in h]),
                                      ('train_loss', [x['train_loss'] for x in h]),
                                      ('validation_loss', [x['validation_loss'] for x in h])]:
                    np.testing.assert_allclose(csv_history[field], values)
                    assert np.isfinite(values).all()
                assert (np.diff([0]+[x['optimizer_steps'] for x in h]) > 0).all()
                logs = list(args.runs.glob(name+'_job_*.log'))
                assert logs and any('Exit: 0' in p.read_text().splitlines()[-1] for p in logs)
                b = m['best']
                assert b == h[m['best_epoch']-1]
                best = max(h, key=lambda x: (x['selected']['f2'], x['selected']['ap'],
                           x['selected']['precision'], -x['oil']['fn'], -x['validation_loss']))
                assert best['epoch'] == m['best_epoch']
                expected = indices[size].query('fold == @fold').set_index('patch_id')
                train = indices[size].query('fold != @fold')
                counts = train[[f'label_{cl}' for cl in CLASSES]].sum().to_numpy()
                np.testing.assert_allclose(config['train_class_counts'], counts)
                np.testing.assert_allclose(config['pos_weight'], (len(train)-counts)/counts, rtol=1e-6)
                assert config['class_names'] == CLASSES
                assert config['source_hashes']['scene_folds.csv'] == sha(manifest_path)
                assert config['source_hashes']['fold_audit.json'] == sha(audit_path)
                for e in range(1, 51):
                    ep = root/f'validation_epoch_{e:03d}.parquet'
                    assert pq.read_metadata(ep).num_rows == len(expected)
                    epochs_audit.append(dict(run=name, epoch=e, rows=len(expected), bytes=ep.stat().st_size))
                df = pd.read_parquet(root/'val_predictions.parquet')
                pd.testing.assert_frame_equal(df, pd.read_parquet(root/f'validation_epoch_{m["best_epoch"]:03d}.parquet'))
                assert not df.patch_id.duplicated().any() and set(df.patch_id) == set(expected.index)
                assert (df.fold == fold).all() and (df.patch_size == size).all() and (df.epoch == m['best_epoch']).all()
                meta = expected.loc[df.patch_id]
                assert np.array_equal(df.scene_id.to_numpy(), meta.scene_id.to_numpy())
                for cl in CLASSES:
                    np.testing.assert_array_equal(df[f'true_{cl}'], meta[f'label_{cl}'])
                    p = df[f'prob_{cl}'].to_numpy()
                    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
                    logits = df[f'logit_{cl}'].to_numpy(dtype=float)
                    assert np.isfinite(logits).all()
                    np.testing.assert_allclose(p, 1/(1+np.exp(-np.clip(logits, -700, 700))), atol=1e-7, rtol=1e-5)
                    threshold = b['selected']['threshold'] if cl == 'oil' else .5
                    np.testing.assert_array_equal(df[f'pred_{cl}'], p >= threshold)
                    cm = binary(df[f'true_{cl}'], df[f'pred_{cl}'])
                    for field in ['precision', 'recall', 'f1', 'fn', 'fp']:
                        np.testing.assert_allclose(cm[field], b[cl][field], atol=1e-7)
                threshold, ap, _, _ = pr(df.true_oil, df.prob_oil)
                np.testing.assert_allclose([threshold, ap], [b['selected']['threshold'], b['selected']['ap']], atol=2e-7)
                cm = binary(df.true_oil, df.pred_oil)
                np.testing.assert_allclose(cm['f2'], b['selected']['f2'])
                np.testing.assert_allclose(df.oil_threshold, threshold)
                for suffix, mask in [('negatives', (df.true_oil == 1) & (df.pred_oil == 0)),
                                     ('positives', (df.true_oil == 0) & (df.pred_oil == 1))]:
                    saved = pd.read_parquet(root/f'oil_false_{suffix}.parquet')
                    pd.testing.assert_frame_equal(saved.reset_index(drop=True), df[mask].reset_index(drop=True))
                t95, _, _, _ = pr(df.true_oil, df.prob_oil, .95)
                cm95 = binary(df.true_oil, df.prob_oil >= t95)
                ref = binary(df.true_oil, df.prob_oil >= .5)
                df['pred_oil_r95'] = (df.prob_oil >= t95).astype(int)
                for col in ['group_id', 'pixels_oil', 'valid_pixels', 'pad_ratio', 'is_border', 'y_orig', 'x_orig']:
                    df[col] = meta[col].to_numpy()
                df['oil_fraction'] = df.pixels_oil/df.valid_pixels
                df['configuration'] = key
                frames.append(df)
                final = h[-1]
                row = dict(configuration=key, model=model, patch_size=size, fold=fold, best_epoch=m['best_epoch'],
                           threshold=threshold, ap=ap, **cm, macro_f1=b['macro_f1'],
                           threshold_r95=t95, recall_r95=cm95['recall'], precision_r95=cm95['precision'],
                           f2_r95=cm95['f2'], routed_fraction_r95=cm95['routed_fraction'],
                           recall_at_05=ref['recall'], precision_at_05=ref['precision'],
                           last_f2=final['selected']['f2'], best_minus_last_f2=cm['f2']-final['selected']['f2'],
                           last_train_loss=final['train_loss'], last_val_loss=final['validation_loss'],
                           best_val_loss=b['validation_loss'], min_val_loss_epoch=min(h, key=lambda x:x['validation_loss'])['epoch'],
                           epoch_gpu_hours=m['epoch_gpu_hours'], parameters=config['parameters'],
                           peak_gpu_gib=max(x['peak_cuda_allocated_bytes'] for x in h)/2**30,
                           amp_skipped_steps=final['amp_skipped_steps'], optimizer_steps=final['optimizer_steps'],
                           validation_pipeline_pps=np.mean([x['validation_pipeline_patches_per_second'] for x in h[5:]]))
                row['best_ap_epoch'] = max(h, key=lambda x:x['selected']['ap'])['epoch']
                row['max_epoch_ap'] = max(x['selected']['ap'] for x in h)
                rows.append(row)
                for x in h:
                    histories.append(dict(configuration=key, fold=fold, epoch=x['epoch'], train_loss=x['train_loss'],
                                          validation_loss=x['validation_loss'], f2=x['selected']['f2'], ap=x['selected']['ap'],
                                          recall=x['selected']['recall'], precision=x['selected']['precision'],
                                          threshold=x['selected']['threshold'], routed_fraction=x['routed_fraction'],
                                          amp_skipped_steps=x['amp_skipped_steps']))
                audits.append(dict(run=name, passed=True, files=len(list(root.iterdir())),
                                   validation_rows=len(df), validation_scenes=df.scene_id.nunique(),
                                   epoch_exports=50, failure_json_present=(root/'failure.json').exists()))
            joined = pd.concat(frames, ignore_index=True)
            assert not joined.patch_id.duplicated().any() and set(joined.patch_id) == set(indices[size].patch_id)
            assert joined.groupby('scene_id').fold.nunique().max() == 1
            all_pred[key] = joined
            joined.to_parquet(out/f'oof_{key}.parquet', index=False)
            joined[joined.true_oil != joined.pred_oil].to_csv(out/f'oil_errors_{key}.csv', index=False)
    # Identity across deployments, and a transparent check against current local source.
    hashes = all_config[0]['source_hashes']
    assert all(c['source_hashes'] == hashes for c in all_config)
    fixed = ['seed', 'epochs', 'adaptation_epochs', 'weight_decay', 'gradient_clip', 'stem_lr',
             'head_adaptation_lr', 'backbone_lr', 'head_lr', 'recall_min', 'recall_target',
             'speckle_std', 'intensity_min', 'intensity_max']
    for field in fixed:
        assert len({c['config'][field] for c in all_config}) == 1, field
    for size in [256, 512]:
        a, b = [all_pred[f'{m}_p{size}'].sort_values('patch_id') for m in ['effnet_b2', 'convnext_tiny']]
        for col in ['patch_id', 'scene_id', 'fold'] + [f'true_{cl}' for cl in CLASSES]:
            np.testing.assert_array_equal(a[col], b[col])
    runs, history = pd.DataFrame(rows), pd.DataFrame(histories)
    runs.to_csv(out/'per_run.csv', index=False)
    history.to_csv(out/'learning_history.csv', index=False)
    pd.DataFrame(audits).to_csv(out/'artifact_audit.csv', index=False)
    pd.DataFrame(epochs_audit).to_csv(out/'epoch_export_inventory.csv', index=False)
    summary = []
    for key, g in runs.groupby('configuration'):
        row = dict(configuration=key, folds=len(g), worst_recall=g.recall.min(), total_fn=g.fn.sum(),
                   total_fp=g.fp.sum(), total_epoch_gpu_hours=g.epoch_gpu_hours.sum(),
                   best_epochs=','.join(g.sort_values('fold').best_epoch.astype(str)))
        for col in ['f2', 'recall', 'precision', 'ap', 'macro_f1', 'routed_fraction', 'f2_r95',
                    'recall_r95', 'precision_r95', 'routed_fraction_r95', 'recall_at_05',
                    'best_minus_last_f2', 'peak_gpu_gib', 'validation_pipeline_pps']:
            row[col+'_mean'], row[col+'_sd'] = g[col].mean(), g[col].std(ddof=1)
        summary.append(row)
    summary = pd.DataFrame(summary).sort_values('f2_mean', ascending=False)
    summary.to_csv(out/'configuration_summary.csv', index=False)
    pooled, thresholds, classes, errors, scenes, screening, missed, fp_labels = [], [], [], [], [], [], [], []
    for key, df in all_pred.items():
        for cl in CLASSES:
            cm = binary(df[f'true_{cl}'], df[f'pred_{cl}'])
            classes.append(dict(configuration=key, class_name=cl, **cm, ap=pr(df[f'true_{cl}'], df[f'prob_{cl}'])[1]))
        for policy, pred in [('fold_r90', df.pred_oil), ('fold_r95', df.pred_oil_r95), ('fixed_05', df.prob_oil >= .5)]:
            pooled.append(dict(configuration=key, policy=policy, **binary(df.true_oil, pred)))
            positive_routing = df.assign(routed_positive=(pred & (df.true_oil == 1)).astype(int)).groupby('scene_id').agg(
                oil_patches=('true_oil', 'sum'), routed_oil_patches=('routed_positive', 'sum'))
            oil_scenes = positive_routing[positive_routing.oil_patches > 0]
            lost = oil_scenes[oil_scenes.routed_oil_patches == 0]
            screening.append(dict(configuration=key, policy=policy, oil_scenes=len(oil_scenes),
                                  scenes_all_oil_patches_missed=len(lost), fraction_with_any_oil_patch_routed=1-len(lost)/len(oil_scenes)))
            for scene, s in lost.iterrows():
                missed.append(dict(configuration=key, policy=policy, scene_id=scene, oil_patches=int(s.oil_patches)))
        fp = df[(df.true_oil == 0) & (df.pred_oil == 1)]
        for cl in CLASSES[1:]:
            fp_labels.append(dict(configuration=key, true_class=cl, oil_fp=len(fp),
                                  fp_with_class=int(fp[f'true_{cl}'].sum()), fraction=float(fp[f'true_{cl}'].mean())))
        for target in [.90, .95]:
            t, ap, _, _ = pr(df.true_oil, df.prob_oil, target)
            per_fold = [binary(g.true_oil, g.prob_oil >= t)['recall'] for _, g in df.groupby('fold')]
            thresholds.append(dict(configuration=key, recall_constraint=target, exploratory_pooled_threshold=t,
                                   ap=ap, **binary(df.true_oil, df.prob_oil >= t), worst_fold_recall=min(per_fold)))
        positive = df[df.true_oil == 1].copy()
        positive['oil_area_bin'] = pd.cut(positive.oil_fraction, [0, .001, .01, .05, .2, 1.000001],
                                                labels=['<=0.1%', '0.1-1%', '1-5%', '5-20%', '>20%'])
        for bucket, g in positive.groupby('oil_area_bin', observed=True):
            errors.append(dict(configuration=key, oil_area_bin=str(bucket), oil_patches=len(g),
                               fn=int((g.pred_oil == 0).sum()), recall=g.pred_oil.mean(), recall_r95=g.pred_oil_r95.mean()))
        for scene, g in df.groupby('scene_id'):
            scenes.append(dict(configuration=key, scene_id=scene, group_id=g.group_id.iloc[0], fold=g.fold.iloc[0],
                               **binary(g.true_oil, g.pred_oil)))
    pd.DataFrame(pooled).to_csv(out/'pooled_operating_points.csv', index=False)
    pd.DataFrame(thresholds).to_csv(out/'exploratory_pooled_thresholds.csv', index=False)
    pd.DataFrame(classes).to_csv(out/'per_class_pooled.csv', index=False)
    pd.DataFrame(errors).to_csv(out/'oil_area_error_analysis.csv', index=False)
    pd.DataFrame(screening).to_csv(out/'scene_screening.csv', index=False)
    pd.DataFrame(missed).to_csv(out/'scenes_all_oil_patches_missed.csv', index=False)
    pd.DataFrame(fp_labels).to_csv(out/'false_positive_label_composition.csv', index=False)
    scene_table = pd.DataFrame(scenes)
    scene_table.sort_values('fn', ascending=False).to_csv(out/'per_scene_errors.csv', index=False)
    # Paired cluster bootstrap: same frozen duplicate-image groups and folds for both models.
    # Conditional descriptive intervals only: no refitting, no correction for epoch/threshold selection.
    rng = np.random.default_rng(42)
    boot_rows = []
    for size in [256, 512]:
        agg = []
        for model in ['effnet_b2', 'convnext_tiny']:
            g = scene_table[scene_table.configuration == f'{model}_p{size}']
            agg.append(g.groupby(['fold', 'group_id'])[['tp', 'fp', 'fn']].sum().sort_index())
        assert agg[0].index.equals(agg[1].index)
        diffs = []
        for _ in range(2000):
            ix = np.concatenate([rng.choice(np.flatnonzero(agg[0].index.get_level_values('fold') == f),
                                            size=sum(agg[0].index.get_level_values('fold') == f), replace=True)
                                 for f in range(1, 6)])
            vals = []
            for a in agg:
                tp, fp, fn = a.to_numpy()[ix].sum(0)
                vals.append(5*tp/(5*tp+fp+4*fn))
            diffs.append(vals[1]-vals[0])
        delta = runs.query('patch_size == @size').pivot(index='fold', columns='model', values='f2')
        d = delta.convnext_tiny-delta.effnet_b2
        observed = []
        for a in agg:
            tp, fp, fn = a.sum().to_numpy()
            observed.append(5*tp/(5*tp+fp+4*fn))
        boot_rows.append(dict(patch_size=size, contrast='ConvNeXt minus EfficientNet', mean_fold_f2_difference=d.mean(),
                              folds_convnext_wins=int((d > 0).sum()), observed_pooled_f2_difference=observed[1]-observed[0],
                              conditional_pooled_f2_delta_low=np.quantile(diffs, .025),
                              conditional_pooled_f2_delta_high=np.quantile(diffs, .975), bootstrap_replicates=2000))
    pd.DataFrame(boot_rows).to_csv(out/'paired_architecture_comparison.csv', index=False)
    audit = dict(full_runs=20, excluded_smoke_directories=sum(p.is_dir() for p in args.runs.glob('smoke_*fold*')),
                 checks='Completion, matrix, source/fold hashes, train-only index hashes, fold labels, class weights, best-epoch ranking, predictions and confusion metrics, false-error exports, OOF coverage, all epoch Parquet footers',
                 source_hashes=hashes, local_source_matches={f:sha(BASE/'training'/f)==v for f,v in hashes.items() if f.endswith('.py')},
                 validation_scenes=len(manifest), duplicate_image_groups=manifest.group_id.nunique(),
                 checkpoint_check='All 60 checkpoint ZIP central directories readable; tensors and full-file CRC not checked',
                 no_test_evaluation=True, selection_bias='Epochs and thresholds selected using these validation folds. OOF here is development evidence, not unbiased nested CV.',
                 bootstrap='2000 paired duplicate-image-group resamples stratified by fold, seed 42; conditional on fitted/selected models, thresholds and predictions; not a confirmatory significance test')
    audit['analysis_script_sha256'] = sha(Path(__file__))
    (out/'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    plot(out, runs, history, all_pred, pd.DataFrame(errors))
    report = '# Controlled Stage 2: development analysis\n\n'
    report += 'All 20 full runs passed the implemented artifact checks; 50 epochs each. Four smoke runs excluded. See audit.json for scope.\n\n'
    report += 'Best operational checkpoint per fold; mean and sample SD over five folds. AP is non-interpolated average precision.\n\n'
    report += md_table(summary[['configuration','f2_mean','f2_sd','recall_mean','worst_recall','precision_mean','ap_mean','routed_fraction_mean','best_epochs']])
    report += '\n\n## Recall 0.95 operating target (same checkpoints)\n\n'
    report += md_table(summary[['configuration','recall_r95_mean','precision_r95_mean','f2_r95_mean','routed_fraction_r95_mean']])
    report += '\n\n## Scenes where all oil-positive windows are rejected\n\n'
    report += md_table(pd.DataFrame(screening).query('policy != "fixed_05"'))
    report += '\n\nThis is a screening diagnostic on development scenes, not spill-instance recall or pixel coverage.\n'
    report += '\n\n## Interpretation limits\n\n'
    report += ('Epoch and threshold selection reused validation labels; these are development estimates with selection optimism. '
               'No independent test evaluation or nested model selection was performed. Different patch sizes have different grids, '
               'class prevalence and sample counts; their patch metrics compare the complete grid/context configuration and do not isolate architecture or context. '
               'False-negative patch counts are not missed-spill counts. Routed patch fractions are not segmentation speedups or spatial coverage. '
               'Pooled thresholds are exploratory: a fold-model threshold is not automatically valid for a refit or ensemble. '
               'Only seed 42 was trained. Fold SD is not a confidence interval. The cluster bootstrap is conditional and does not include training/selection uncertainty. '
               'Exact duplicate image groups are kept together; acquisition/incident-level independence is unavailable.\n\n')
    report += 'Timing covers training, validation and prediction export, excluding initialization, checkpoint I/O and lost partial epochs. Validation pipeline throughput includes data loading and metrics; it is not isolated model inference throughput.\n\n'
    report += 'Reproduce: `python3 classification/scripts/analyze_controlled_runs.py`\n'
    (out/'README.md').write_text(report)
    print(summary[['configuration','f2_mean','f2_sd','recall_mean','precision_mean','ap_mean','best_epochs']].to_string(index=False))
    print('Analysis written to', out)


def plot(out, runs, history, predictions, errors):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    keys = list(predictions)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, metric, title in zip(axes.flat, ['f2', 'ap', 'precision', 'recall'], ['Oil F2', 'Oil AP', 'Oil precision', 'Oil recall']):
        for j, key in enumerate(keys):
            values = runs[runs.configuration == key][metric]
            ax.scatter(np.full(5, j)+np.linspace(-.09,.09,5), values, color=COLORS[j], s=30)
            ax.errorbar(j, values.mean(), yerr=values.std(ddof=1), color='black', fmt='_', capsize=5, markersize=18)
        ax.set_xticks(range(4), [k.replace('_p','\n').replace('convnext_tiny','ConvNeXt').replace('effnet_b2','EffNet-B2') for k in keys])
        ax.set_title(title); ax.grid(axis='y', alpha=.2)
    fig.suptitle('Selected checkpoints: five development folds (mean ± SD)')
    save(fig, out/'fold_comparison')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, key, color in zip(axes.flat, keys, COLORS):
        g = history[history.configuration == key]
        for metric, style, label in [('train_loss','-','Train'),('validation_loss','--','Validation')]:
            p = g.pivot(index='epoch', columns='fold', values=metric)
            ax.plot(p.index, p.mean(axis=1), style, color=color, label=label)
            ax.fill_between(p.index, p.mean(axis=1)-p.std(axis=1), p.mean(axis=1)+p.std(axis=1), color=color, alpha=.1)
        ax.axvline(5.5, color='gray', linestyle=':', linewidth=1)
        ax.set_title(key); ax.set_xlabel('Epoch'); ax.set_ylabel('Weighted BCE'); ax.legend()
    fig.suptitle('Learning curves: mean ± SD; full fine-tuning begins at epoch 6')
    save(fig, out/'learning_losses')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for key, color in zip(keys, COLORS):
        g = history[history.configuration == key]
        for ax, metric in zip(axes, ['f2','ap']):
            p = g.pivot(index='epoch', columns='fold', values=metric)
            ax.plot(p.index,p.mean(axis=1),color=color,label=key)
            ax.fill_between(p.index,p.mean(axis=1)-p.std(axis=1),p.mean(axis=1)+p.std(axis=1),color=color,alpha=.1)
            ax.set_xlabel('Epoch'); ax.set_ylabel('Oil '+metric.upper()); ax.grid(alpha=.2)
    axes[0].legend(fontsize=8); fig.suptitle('Validation metrics; threshold reselected each epoch')
    save(fig, out/'learning_metrics')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, size in zip(axes, [256,512]):
        for key, color in zip(keys,COLORS):
            if not key.endswith(str(size)): continue
            df=predictions[key]; _,ap,r,p=pr(df.true_oil,df.prob_oil)
            ax.step(r,p,where='post',color=color,label=f'{key}: pooled AP={ap:.3f}')
        ax.axhline(df.true_oil.mean(),color='gray',linestyle=':',label='Prevalence')
        ax.set(xlabel='Recall',ylabel='Precision',title=f'{size} px: pooled development predictions',xlim=(0,1),ylim=(0,1))
        ax.legend(fontsize=8); ax.grid(alpha=.2)
    save(fig,out/'precision_recall')
    fig, ax = plt.subplots(figsize=(10,5),constrained_layout=True)
    bins=['<=0.1%','0.1-1%','1-5%','5-20%','>20%']
    for key,color in zip(keys,COLORS):
        g=errors[errors.configuration==key].set_index('oil_area_bin').reindex(bins)
        ax.plot(bins,g.recall,'o-',color=color,label=key)
    ax.set(xlabel='Oil pixels / valid pixels in positive patch',ylabel='Oil recall',ylim=(0,1.02),title='Missed oil by relative oil area (selected fold thresholds)')
    ax.legend(fontsize=8); ax.grid(alpha=.2)
    save(fig,out/'oil_area_recall')


def save(fig, stem):
    fig.savefig(stem.with_suffix('.png'),dpi=180)
    fig.savefig(stem.with_suffix('.pdf'))
    plt.close(fig)


if __name__ == '__main__':
    main()
