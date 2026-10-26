"""Evaluate two Stage-2/Stage-3 cascades on fold-aligned 256 px OOF scenes.

Stage-2 routing is read from the frozen OOF prediction tables produced by the
controlled classifiers. SegFormer is run only for routed windows. Final Oil
metrics use every original scene pixel: a pixel outside the union of routed
windows is a negative Oil prediction and therefore remains a false negative
when its target is Oil. Five-class metrics are reported separately on routed
coverage because gate rejection does not imply the semantic class Sea.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from torch.utils.data import DataLoader

from segmentation import paths
from segmentation.cascade.b0_baseline import model_from_pretrained
from segmentation.cascade.infer_b0_oof import (DISPLAY_PALETTE, UNCOVERED_COLOR, add_patch, colorize_mask,
                                               metrics, native_backend, synchronize, validate_checkpoint)
from segmentation.data import CLASSES, Patches, atomic_save, load_tables, read_mask, sha256


CONFIGURATIONS = ('effnet_b2_p256', 'convnext_tiny_p256', 'effnet_b2_p512', 'convnext_tiny_p512')
POLICIES = ('fold_r95', 'fold_r90', 'fixed_05')


def nullable_mean_sd(values):
    """Undefined ratios stay null and do not count as zero-valued folds."""
    defined = [value for value in values if value is not None]
    return dict(mean=float(np.mean(defined)) if defined else None,
                sample_sd=float(np.std(defined, ddof=1)) if len(defined) > 1 else None,
                defined_folds=len(defined))


def route_mask(frame, policy):
    """Return the frozen routing decision without retuning a threshold."""
    if policy == 'fold_r95':
        if 'pred_oil_r95' not in frame:
            raise ValueError('Routing table lacks pred_oil_r95')
        return frame.pred_oil_r95.to_numpy(dtype=bool)
    if policy == 'fold_r90':
        return frame.pred_oil.to_numpy(dtype=bool)
    if policy == 'fixed_05':
        return frame.prob_oil.to_numpy(dtype=float) >= .5
    raise ValueError(f'Unknown routing policy: {policy}')


def align_routing(validation, routing, fold, configuration, policy):
    """Validate one-to-one identity/geometry and attach routing metadata."""
    required = {'patch_id', 'scene_id', 'fold', 'patch_size', 'y_orig', 'x_orig',
                'valid_pixels', 'prob_oil', 'pred_oil'}
    missing = required-set(routing.columns)
    if missing:
        raise ValueError(f'Routing table missing columns: {sorted(missing)}')
    if routing.patch_id.duplicated().any() or validation.patch_id.duplicated().any():
        raise ValueError('Duplicate patch IDs')
    size = int(configuration.rsplit('_p', 1)[1])
    if set(routing.fold.unique()) != {fold} or set(routing.patch_size.unique()) != {size}:
        raise ValueError('Routing fold or patch size mismatch')
    if set(routing.configuration.unique()) != {configuration}:
        raise ValueError('Routing configuration mismatch')
    if set(validation.patch_id) != set(routing.patch_id):
        raise ValueError('Routing and Stage-3 validation patches differ')
    route = routing.set_index('patch_id').loc[validation.patch_id].reset_index()
    for column in ('scene_id', 'y_orig', 'x_orig', 'valid_pixels'):
        expected = validation[column].to_numpy()
        actual = route[column].to_numpy()
        if not np.array_equal(expected, actual):
            raise ValueError(f'Routing geometry mismatch: {column}')
    result = validation.copy().reset_index(drop=True)
    result['gate_prob_oil'] = route.prob_oil.to_numpy(dtype=np.float32)
    result['routed'] = route_mask(route, policy)
    result['gate_policy'] = policy
    return result


def oil_metrics(target, prediction):
    """Binary Oil metrics on the complete original scene domain."""
    true = target == 1
    pred = prediction == 1
    tp = int(np.count_nonzero(true & pred))
    fp = int(np.count_nonzero(~true & pred))
    fn = int(np.count_nonzero(true & ~pred))
    tn = int(np.count_nonzero(~true & ~pred))
    def ratio(a, b):
        return float(a/b) if b else None
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                iou=ratio(tp, tp+fp+fn), precision=ratio(tp, tp+fp),
                recall=ratio(tp, tp+fn), f1=ratio(2*tp, 2*tp+fp+fn),
                support=tp+fn)


def evaluate_cascade_scene(target, prediction, coverage):
    """Evaluate final Oil everywhere and semantic labels only where routed."""
    covered = coverage > 0
    oil = target == 1
    labels, count = ndimage.label(oil, structure=np.ones((3, 3), dtype=np.uint8))
    hit = set(np.unique(labels[oil & (prediction == 1)]).tolist())-{0}
    covered_components = set(np.unique(labels[oil & covered]).tolist())-{0}
    cm = np.bincount((target[covered].astype(np.int64)*5+prediction[covered]).ravel(),
                     minlength=25).reshape(5, 5)
    return dict(oil_full_domain=oil_metrics(target, prediction),
                covered_five_class=metrics(cm),
                scene_pixels=int(target.size), covered_pixels=int(covered.sum()),
                rejected_pixels=int((~covered).sum()),
                coverage_fraction=float(covered.mean()),
                oil_pixels=int(oil.sum()), oil_pixels_covered=int((oil & covered).sum()),
                oil_area_retention=float((oil & covered).sum()/oil.sum()) if oil.any() else None,
                oil_components_total=int(count),
                oil_components_with_routed_coverage=len(covered_components),
                oil_components_missed=int(count-len(hit)),
                oil_scene_missed=bool(count and not hit))


def infer_cascade_scene(model, frame, args, masks, device, output):
    scene_id = str(frame.iloc[0].scene_id)
    target = read_mask(masks[scene_id])
    total = np.zeros((5, *target.shape), dtype=np.float32)
    coverage = np.zeros(target.shape, dtype=np.uint16)
    accepted = frame[frame.routed].copy()
    rows = accepted.to_dict('records')
    model_seconds = 0.0
    started = time.perf_counter()
    if rows:
        loader = DataLoader(native_backend(getattr(args, 'patch_size', 256)).Patches(accepted, args.data_root, masks),
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=device.type == 'cuda')
        offset = 0
        with torch.inference_mode():
            for images, _ in loader:
                synchronize(device)
                tick = time.perf_counter()
                images = images.to(device)
                with torch.autocast(device_type=device.type,
                                    enabled=args.amp and device.type == 'cuda'):
                    logits = model(pixel_values=images).logits
                    logits = F.interpolate(logits, size=images.shape[-2:], mode='bilinear',
                                           align_corners=False)
                probabilities = logits.float().softmax(1).cpu().numpy()
                synchronize(device)
                model_seconds += time.perf_counter()-tick
                for probability in probabilities:
                    add_patch(total, coverage, probability, rows[offset])
                    offset += 1
        if offset != len(rows):
            raise RuntimeError('Incomplete routed-window inference')
    covered = coverage > 0
    total /= np.maximum(coverage, 1)[None]
    prediction = total.argmax(0).astype(np.uint8)
    prediction[~covered] = 255
    report = evaluate_cascade_scene(target, prediction, coverage)
    report.update(scene_id=scene_id, patches_total=len(frame), patches_routed=len(accepted),
                  routed_fraction=float(len(accepted)/len(frame)),
                  segmentation_model_seconds=model_seconds,
                  processing_seconds=time.perf_counter()-started)
    Image.fromarray(prediction).save(output/f'{scene_id}_prediction.png')
    Image.fromarray(colorize_mask(prediction)).save(output/f'{scene_id}_prediction_rgb.png')
    Image.fromarray(colorize_mask(target)).save(output/f'{scene_id}_target_rgb.png')
    np.save(output/f'{scene_id}_coverage.npy', coverage, allow_pickle=False)
    if args.save_probabilities:
        np.save(output/f'{scene_id}_probabilities.npy', total, allow_pickle=False)
    return report


def summarize_reports(reports):
    oil = {key: sum(r['oil_full_domain'][key] for r in reports)
           for key in ('tp', 'fp', 'fn', 'tn')}
    tp, fp, fn, tn = (oil[k] for k in ('tp', 'fp', 'fn', 'tn'))
    def ratio(a, b):
        return float(a/b) if b else None
    oil.update(iou=ratio(tp, tp+fp+fn), precision=ratio(tp, tp+fp),
               recall=ratio(tp, tp+fn), f1=ratio(2*tp, 2*tp+fp+fn),
               support=tp+fn)
    cm = np.sum([r['covered_five_class']['confusion_matrix'] for r in reports], axis=0)
    return dict(scenes=len(reports), oil_full_domain=oil,
                covered_five_class=metrics(cm),
                scene_pixels=sum(r['scene_pixels'] for r in reports),
                covered_pixels=sum(r['covered_pixels'] for r in reports),
                oil_pixels=sum(r['oil_pixels'] for r in reports),
                oil_pixels_covered=sum(r['oil_pixels_covered'] for r in reports),
                patches_total=sum(r['patches_total'] for r in reports),
                patches_routed=sum(r['patches_routed'] for r in reports),
                oil_scenes_missed=sum(r['oil_scene_missed'] for r in reports),
                oil_components_missed=sum(r['oil_components_missed'] for r in reports),
                segmentation_model_seconds=sum(r['segmentation_model_seconds'] for r in reports),
                processing_seconds=sum(r['processing_seconds'] for r in reports))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    paths.add_root_argument(parser)
    parser.add_argument('--data-root', type=Path, default=paths.STAGE2_RESULTS / 'training_patches/ps256_overlap50')
    parser.add_argument('--folds-dir', type=Path, default=paths.STAGE2_RESULTS / 'grouped_folds_v1')
    parser.add_argument('--masks-root', type=Path, default=paths.DATASET / 'train/labels')
    parser.add_argument('--segmentation-checkpoints-root', type=Path, required=True)
    parser.add_argument('--routing-root', type=Path, required=True,
                        help='folder with the frozen Stage-2 OOF routing tables oof_<configuration>.parquet')
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--baseline-summary', type=Path,
                        help='Completed ungated OOF summary used for automatic deltas')
    parser.add_argument('--configurations', nargs='+', choices=CONFIGURATIONS,
                        default=None)
    parser.add_argument('--patch-size', type=int, choices=(256, 512), default=256)
    parser.add_argument('--policy', choices=POLICIES, default='fold_r95')
    parser.add_argument('--folds', nargs='+', type=int, choices=range(1, 6),
                        default=[1, 2, 3, 4, 5])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--save-probabilities', action='store_true')
    parser.add_argument('--max-scenes-per-fold', type=int)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.configurations is None:
        args.configurations = [f'{m}_p{args.patch_size}' for m in ('effnet_b2', 'convnext_tiny')]
    if any(not c.endswith(f'_p{args.patch_size}') for c in args.configurations):
        parser.error('Configuration and native patch size disagree')
    if args.batch_size < 1 or args.workers < 0 or len(set(args.folds)) != len(args.folds):
        parser.error('Invalid batch size, workers or duplicate folds')
    if len(set(args.configurations)) != len(args.configurations):
        parser.error('Duplicate configurations')
    if args.max_scenes_per_fold is not None and args.max_scenes_per_fold < 1:
        parser.error('max-scenes-per-fold must be positive')

    hashes = {'fold_manifest_sha256': sha256(args.folds_dir/'scene_folds.csv'),
              'scene_manifest_sha256': sha256(args.data_root/'scene_manifest.csv'),
              'patch_index_sha256': sha256(next(args.data_root.glob('index_*.parquet')))}
    prepared = []
    routing_hashes = {}
    for fold in args.folds:
        _, validation, masks = native_backend(args.patch_size).load_tables(args.data_root, args.folds_dir, fold,
                                           args.masks_root)
        checkpoint = (args.segmentation_checkpoints_root/
                      f'segformer_b0_p{args.patch_size}_fold{fold:02d}_seed{args.seed}'/'best.pt')
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        validate_checkpoint(state, fold, args.seed, hashes, args.patch_size)
        record = dict(fold=fold, checkpoint=str(checkpoint.resolve()),
                      checkpoint_sha256=sha256(checkpoint), epoch=state['epoch'])
        routes = {}
        for configuration in args.configurations:
            route_path = args.routing_root/f'oof_{configuration}.parquet'
            complete = pd.read_parquet(route_path)
            routing_hashes[str(route_path)] = sha256(route_path)
            route = complete[complete.fold == fold].copy()
            routes[configuration] = align_routing(validation, route, fold,
                                                   configuration, args.policy)
        prepared.append((validation, masks, state, record, routes))
        counts = ', '.join(f'{c}={int(routes[c].routed.sum())}/{len(validation)}'
                           for c in args.configurations)
        print(f'Audit fold {fold}: {validation.scene_id.nunique()} scenes; {counts}',
              flush=True)
    if args.baseline_summary is not None:
        baseline = json.loads(args.baseline_summary.read_text())
        if baseline.get('status') != 'complete' or not baseline.get('full_oof'):
            raise ValueError('Expected completed full OOF baseline')
        baseline_folds = {f['fold']: f for f in baseline['folds']}
        for _, _, _, record, _ in prepared:
            reference = baseline_folds[record['fold']]
            if reference['sha256'] != record['checkpoint_sha256'] or reference['uncovered_pixels'] != 0:
                raise ValueError('Baseline checkpoint or coverage mismatch')
            for key, digest in hashes.items():
                if reference['identity'].get(key) != digest:
                    raise ValueError('Baseline data identity mismatch')
    if args.dry_run:
        print('Data, checkpoint and routing audit passed; no inference was run.')
        return
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu or a GPU environment')
    args.output_root.mkdir(parents=True, exist_ok=False)
    manifest = dict(arguments={k: str(v) if isinstance(v, Path) else v
                               for k, v in vars(args).items()},
                    source_sha256=sha256(__file__), classes=CLASSES,
                    routing_sha256=routing_hashes,
                    checkpoints=[item[3] for item in prepared],
                    configurations=args.configurations, policy=args.policy,
                    routing='frozen Stage-2 OOF probabilities; no threshold retuning',
                    rejected_region='prediction ID 255; negative for full-domain binary Oil metrics',
                    five_class_scope='union of routed windows only',
                    reconstruction='uniform mean of FP32 softmax probabilities over routed windows',
                    display_palette=dict(zip(CLASSES, DISPLAY_PALETTE)),
                    uncovered_color=UNCOVERED_COLOR)
    atomic_save(manifest, args.output_root/'manifest.json')
    all_summaries = {}
    wall_start = time.perf_counter()
    for configuration in args.configurations:
        fold_summaries = []
        for validation, masks, state, record, routes in prepared:
            model = model_from_pretrained(state['model_config'],
                              coordinate_attention=state['identity'].get('coordinate_attention', False))
            model.load_state_dict(state['model_state'], strict=True)
            model.to(device).eval()
            directory = args.output_root/configuration/f'fold{record["fold"]:02d}'
            directory.mkdir(parents=True)
            routed = routes[configuration]
            scene_ids = sorted(routed.scene_id.unique())
            if args.max_scenes_per_fold:
                scene_ids = scene_ids[:args.max_scenes_per_fold]
            selected = routed[routed.scene_id.isin(scene_ids)].sort_values(['scene_id', 'patch_id'])
            selected[['patch_id', 'scene_id', 'fold', 'y_orig', 'x_orig',
                      'valid_pixels', 'gate_prob_oil', 'routed', 'gate_policy']].to_csv(
                          directory/'patches.csv', index=False)
            reports = []
            for scene_id, frame in selected.groupby('scene_id', sort=True):
                report = infer_cascade_scene(model, frame, args, masks, device, directory)
                report.update(fold=record['fold'], configuration=configuration,
                              checkpoint_sha256=record['checkpoint_sha256'])
                atomic_save(report, directory/f'{scene_id}_metrics.json')
                reports.append(report)
                print(f'{configuration} fold {record["fold"]}: {len(reports)}/{len(scene_ids)} '
                      f'{scene_id} Oil IoU={report["oil_full_domain"]["iou"]}', flush=True)
            summary = {**record, **summarize_reports(reports)}
            summary['routed_fraction'] = summary['patches_routed']/summary['patches_total']
            summary['scene_coverage_fraction'] = summary['covered_pixels']/summary['scene_pixels']
            summary['oil_area_retention'] = (summary['oil_pixels_covered']/summary['oil_pixels']
                                             if summary['oil_pixels'] else None)
            atomic_save(summary, directory/'summary.json')
            fold_summaries.append(summary)
            del model
        macro = {}
        for metric in ('iou', 'precision', 'recall', 'f1'):
            values = [f['oil_full_domain'][metric] for f in fold_summaries]
            macro[metric] = nullable_mean_sd(values)
        pooled_reports = dict(status='complete',
                              full_oof=set(args.folds) == set(range(1, 6)) and
                              args.max_scenes_per_fold is None,
                              configuration=configuration, policy=args.policy,
                              folds=fold_summaries, fold_macro_oil_full_domain=macro)
        totals = {key: sum(f['oil_full_domain'][key] for f in fold_summaries)
                  for key in ('tp', 'fp', 'fn', 'tn')}
        tp, fp, fn, tn = (totals[k] for k in ('tp', 'fp', 'fn', 'tn'))
        def ratio(a, b):
            return float(a/b) if b else None
        totals.update(iou=ratio(tp, tp+fp+fn), precision=ratio(tp, tp+fp),
                      recall=ratio(tp, tp+fn), f1=ratio(2*tp, 2*tp+fp+fn))
        pooled_reports['pooled_oil_full_domain'] = totals
        pooled_reports['patches_total'] = sum(f['patches_total'] for f in fold_summaries)
        pooled_reports['patches_routed'] = sum(f['patches_routed'] for f in fold_summaries)
        pooled_reports['routed_fraction'] = (pooled_reports['patches_routed']/
                                             pooled_reports['patches_total'])
        pooled_reports['segmentation_model_seconds'] = sum(
            f['segmentation_model_seconds'] for f in fold_summaries)
        atomic_save(pooled_reports, args.output_root/configuration/'summary.json')
        all_summaries[configuration] = pooled_reports
    comparisons = None
    if args.baseline_summary is not None:
        baseline = json.loads(args.baseline_summary.read_text())
        if not baseline.get('full_oof') or baseline.get('status') != 'complete':
            raise ValueError('Baseline summary must be a complete full OOF run')
        baseline_fold = {f['fold']: f for f in baseline['folds']}
        if set(baseline_fold) != set(range(1, 6)):
            raise ValueError('Baseline summary must contain folds 1..5')
        comparisons = {}
        for configuration, result in all_summaries.items():
            fold_rows = []
            for cascade_fold in result['folds']:
                base = baseline_fold[cascade_fold['fold']]['classes']['oil']
                cascade = cascade_fold['oil_full_domain']
                fold_rows.append(dict(
                    fold=cascade_fold['fold'],
                    **{f'baseline_{m}': base[m] for m in ('iou', 'precision', 'recall', 'f1')},
                    **{f'cascade_{m}': cascade[m] for m in ('iou', 'precision', 'recall', 'f1')},
                    **{f'delta_{m}': cascade[m]-base[m] for m in ('iou', 'precision', 'recall', 'f1')},
                    routed_fraction=cascade_fold['routed_fraction'],
                    scene_coverage_fraction=cascade_fold['scene_coverage_fraction'],
                    oil_area_retention=cascade_fold['oil_area_retention']))
            pooled_base = baseline['pooled']['classes']['oil']
            pooled_cascade = result['pooled_oil_full_domain']
            comparisons[configuration] = dict(
                folds=fold_rows,
                pooled={m: dict(baseline=pooled_base[m], cascade=pooled_cascade[m],
                                delta=pooled_cascade[m]-pooled_base[m])
                        for m in ('iou', 'precision', 'recall', 'f1')})
        atomic_save(comparisons, args.output_root/'comparison_to_ungated.json')
    atomic_save(dict(status='complete', policy=args.policy,
                     full_oof=all(v['full_oof'] for v in all_summaries.values()),
                     configurations=all_summaries,
                     comparison_to_ungated=comparisons,
                     wall_seconds=time.perf_counter()-wall_start,
                     classifier_runtime_seconds=None,
                     timing_note='Routing uses saved OOF classifier outputs; rerun classifiers separately for end-to-end timing.'),
                args.output_root/'summary.json')


if __name__ == '__main__':
    main()
