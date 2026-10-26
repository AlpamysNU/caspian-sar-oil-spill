"""Fold-aligned SegFormer-B0 inference, uniform probability stitching and scene metrics (ungated baseline OOF).

Source restored on 2026-10-26 from the compiled module of the original run (bytecode, names and constants of all 16 code objects match the
original); comments of the original source are not recoverable. Both native backends (256 px and 512 px windows) are included.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from torch.utils.data import DataLoader

from segmentation import paths
from segmentation.cascade.b0_baseline import PRETRAINED, model_from_pretrained
from segmentation.data import CLASSES, Patches, atomic_save, load_tables, read_mask, sha256


def native_backend(size):
    if size == 512:
        from segmentation.cascade import b0_native512 as backend
        return backend
    if size == 256:
        from segmentation.cascade import b0_baseline as backend
        return backend
    raise ValueError('Unsupported native size')


DISPLAY_PALETTE = ((0, 0, 0), (0, 255, 255), (255, 0, 0), (128, 64, 0), (0, 153, 0))
UNCOVERED_COLOR = (255, 0, 255)


def colorize_mask(mask):
    """Map class IDs to RGB for viewing; magenta marks uncovered pixels."""
    if mask.ndim != 2 or not np.isin(mask, [0, 1, 2, 3, 4, 255]).all():
        raise ValueError('Expected a 2D class-ID mask containing 0–4 or 255')
    palette = np.zeros((256, 3), dtype=np.uint8)
    palette[:5] = DISPLAY_PALETTE
    palette[255] = UNCOVERED_COLOR
    return palette[mask.astype(np.uint8)]


def metrics(cm):
    cm = np.asarray(cm, dtype=np.int64)
    tp = cm.diagonal()
    true, pred = cm.sum(1), cm.sum(0)

    def ratio(a, b):
        return float(a / b) if b else None

    classes = {name: dict(iou=ratio(tp[i], true[i] + pred[i] - tp[i]),
                          precision=ratio(tp[i], pred[i]),
                          recall=ratio(tp[i], true[i]),
                          f1=ratio(2 * tp[i], true[i] + pred[i]),
                          support=int(true[i]))
               for i, name in enumerate(CLASSES)}
    values = [v['iou'] for v in classes.values() if v['iou'] is not None]
    return dict(confusion_matrix=cm.tolist(), classes=classes,
                miou=float(np.mean(values)) if values else None)


def add_patch(total, coverage, probabilities, row):
    """Clip native window to scene extent; padding never contributes."""
    h, w = coverage.shape
    y, x = int(row['y_orig']), int(row['x_orig'])
    size = probabilities.shape[-1]
    y0, y1, x0, x1 = max(0, y), min(h, y + size), max(0, x), min(w, x + size)
    if y1 <= y0 or x1 <= x0 or (y1 - y0) * (x1 - x0) != int(row['valid_pixels']):
        raise ValueError(f"Invalid geometry: {row['patch_id']}")
    if not np.isfinite(probabilities).all():
        raise ValueError('Nonfinite probabilities')
    total[:, y0:y1, x0:x1] += probabilities[:, y0 - y:y1 - y, x0 - x:x1 - x]
    coverage[y0:y1, x0:x1] += 1


def evaluate_scene(target, prediction, coverage):
    valid = coverage > 0
    cm = np.bincount((target[valid].astype(np.int64) * 5 + prediction[valid]).ravel(), minlength=25).reshape(5, 5)
    oil = target == 1
    components, count = ndimage.label(oil, structure=np.ones((3, 3)))
    eligible = set(np.unique(components[valid & oil]).tolist())
    hit = set(np.unique(components[valid & oil & (prediction == 1)]).tolist())
    return {**metrics(cm),
            'covered_pixels': int(valid.sum()),
            'uncovered_pixels': int((~valid).sum()),
            'oil_components_total': int(count),
            'oil_components_evaluable': len(eligible),
            'oil_components_missed': len(eligible - hit),
            'oil_components_uncovered': int(count) - len(eligible),
            'oil_scene_missed': bool(eligible and not hit)}


def validate_checkpoint(state, fold, seed, hashes, patch_size=256):
    identity = state['identity']
    expected = dict(fold=fold, seed=seed, model=PRETRAINED, classes=list(CLASSES),
                    normalization='uint8 / 255; one SAR channel', **hashes)
    for key, value in expected.items():
        if identity.get(key) != value:
            raise ValueError(f'Checkpoint identity mismatch: {key}, fold {fold}')
    if tuple(state['classes']) != CLASSES or not 1 <= state['epoch'] <= identity['epochs']:
        raise ValueError('Invalid checkpoint classes/epoch')
    if state['model_config']['num_channels'] != 1:
        raise ValueError('Expected single-channel model')
    if patch_size == 512 and (identity.get('patch_size'), identity.get('stride')) != (512, 256):
        raise ValueError('Expected native 512 checkpoint identity')
    if patch_size == 256 and identity.get('patch_size', 256) != 256:
        raise ValueError('Expected native 256 checkpoint identity')


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def infer_scene(model, frame, args, masks, device, output):
    scene_id = str(frame.iloc[0].scene_id)
    target = read_mask(masks[scene_id])
    total = np.zeros((5, *target.shape), dtype=np.float32)
    coverage = np.zeros(target.shape, dtype=np.uint16)
    rows = frame.to_dict('records')
    backend = native_backend(getattr(args, 'patch_size', 256))
    loader = DataLoader(backend.Patches(frame, args.data_root, masks), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=device.type == 'cuda')
    offset, model_seconds = 0, 0.0
    synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        for images, _ in loader:
            synchronize(device)
            tick = time.perf_counter()
            images = images.to(device)
            with torch.autocast(device_type=device.type, enabled=args.amp and device.type == 'cuda'):
                logits = model(pixel_values=images).logits
                logits = F.interpolate(logits, size=images.shape[-2:], mode='bilinear', align_corners=False)
            probabilities = logits.float().softmax(1).cpu().numpy()
            synchronize(device)
            model_seconds += time.perf_counter() - tick
            for prob in probabilities:
                add_patch(total, coverage, prob, rows[offset])
                offset += 1
    if offset != len(rows):
        raise RuntimeError('Incomplete patch inference')
    valid = coverage > 0
    if not valid.all():
        raise ValueError('Ungated scene has uncovered pixels')
    total /= np.maximum(coverage, 1)[None]
    prediction = total.argmax(0).astype(np.uint8)
    prediction[~valid] = 255
    report = evaluate_scene(target, prediction, coverage)
    report.update(scene_id=scene_id, patches=len(rows), model_seconds=model_seconds,
                  processing_seconds=time.perf_counter() - start)
    Image.fromarray(prediction).save(output / f'{scene_id}_prediction.png')
    Image.fromarray(colorize_mask(prediction)).save(output / f'{scene_id}_prediction_rgb.png')
    Image.fromarray(colorize_mask(target)).save(output / f'{scene_id}_target_rgb.png')
    np.save(output / f'{scene_id}_coverage.npy', coverage, allow_pickle=False)
    if args.save_probabilities:
        np.save(output / f'{scene_id}_probabilities.npy', total, allow_pickle=False)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    paths.add_root_argument(p)
    p.add_argument('--data-root', type=Path, default=paths.STAGE2_RESULTS / 'training_patches/ps256_overlap50', help='use training_patches/ps512_overlap50 with --patch-size 512')
    p.add_argument('--folds-dir', type=Path, default=paths.STAGE2_RESULTS / 'grouped_folds_v1')
    p.add_argument('--masks-root', type=Path, default=paths.DATASET / 'train/labels')
    p.add_argument('--checkpoints-root', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--folds', nargs='+', type=int, choices=range(1, 6), default=[1, 2, 3, 4, 5])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--patch-size', type=int, choices=(256, 512), default=256)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--amp', action='store_true', help='CUDA autocast; default FP32')
    p.add_argument('--save-probabilities', action='store_true', help='Save float32 (5,H,W) arrays')
    p.add_argument('--max-scenes-per-fold', type=int, help='Smoke subset; never reported as full OOF')
    p.add_argument('--dry-run', action='store_true', help='Audit data and checkpoints without inference')
    args = p.parse_args()
    if args.batch_size < 1 or args.workers < 0 or len(set(args.folds)) != len(args.folds):
        p.error('Invalid batch size, workers or duplicate folds')
    if args.max_scenes_per_fold is not None and args.max_scenes_per_fold < 1:
        p.error('max-scenes-per-fold must be positive')
    hashes = {
        'fold_manifest_sha256': sha256(args.folds_dir / 'scene_folds.csv'),
        'scene_manifest_sha256': sha256(args.data_root / 'scene_manifest.csv'),
        'patch_index_sha256': sha256(next(args.data_root.glob('index_*.parquet')))}
    prepared = []
    for fold in args.folds:
        _, val, masks = native_backend(args.patch_size).load_tables(args.data_root, args.folds_dir, fold, args.masks_root)
        checkpoint = args.checkpoints_root / f'segformer_b0_p{args.patch_size}_fold{fold:02d}_seed{args.seed}' / 'best.pt'
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        validate_checkpoint(state, fold, args.seed, hashes, args.patch_size)
        record = dict(fold=fold, checkpoint=str(checkpoint.resolve()), sha256=sha256(checkpoint), epoch=state['epoch'],
                      identity=state['identity'])
        prepared.append((val, masks, state, record))
        print(f"Audit fold {fold}: {val.scene_id.nunique()} scenes, {len(val)} patches, epoch {state['epoch']}", flush=True)
    if args.dry_run:
        print('Data/checkpoint audit passed; model forward not tested.')
        return
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu or a GPU environment')

    args.output_root.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        checkpoints=[v[3] for v in prepared],
        source_sha256=sha256(__file__),
        trainer_sha256=sha256(native_backend(args.patch_size).__file__),
        torch_version=str(torch.__version__),
        classes=CLASSES,
        display_palette=dict(zip(CLASSES, DISPLAY_PALETTE)),
        uncovered_color=UNCOVERED_COLOR,
        reconstruction='uniform mean of FP32 softmax probabilities; argmax after stitching',
        component_connectivity=8,
        component_hit='at least one correctly predicted covered oil pixel')
    atomic_save(manifest, args.output_root / 'manifest.json')
    fold_reports = []
    start = time.perf_counter()
    for val, masks, state, record in prepared:
        model = model_from_pretrained(state['model_config'], coordinate_attention=state['identity'].get('coordinate_attention', False))
        model.load_state_dict(state['model_state'], strict=True)
        model.to(device).eval()
        directory = args.output_root / f"fold{record['fold']:02d}"
        directory.mkdir()
        ids = sorted(val.scene_id.unique())
        if args.max_scenes_per_fold:
            ids = ids[:args.max_scenes_per_fold]
        selected = val[val.scene_id.isin(ids)].sort_values(['scene_id', 'patch_id'])
        selected.to_csv(directory / 'patches.csv', index=False)
        reports = []
        for scene_id, frame in selected.groupby('scene_id', sort=True):
            report = infer_scene(model, frame, args, masks, device, directory)
            report.update(fold=record['fold'], checkpoint_sha256=record['sha256'])
            atomic_save(report, directory / f'{scene_id}_metrics.json')
            reports.append(report)
            print(f"Fold {record['fold']}: {len(reports)}/{len(ids)} {scene_id} Oil IoU={report['classes']['oil']['iou']}", flush=True)
        cm = np.sum([r['confusion_matrix'] for r in reports], axis=0)
        summary = {**record, **metrics(cm),
                   'scenes': len(reports),
                   'oil_scenes_missed': sum(r['oil_scene_missed'] for r in reports),
                   'oil_components_missed': sum(r['oil_components_missed'] for r in reports),
                   'uncovered_pixels': sum(r['uncovered_pixels'] for r in reports)}
        atomic_save(summary, directory / 'summary.json')
        fold_reports.append(summary)
        del model
    macro = {}
    for metric in ('iou', 'precision', 'recall', 'f1'):
        values = [r['classes']['oil'][metric] for r in fold_reports]
        values = [v for v in values if v is not None]
        macro['oil_' + metric] = dict(mean=float(np.mean(values)) if values else None,
                                      sample_sd=float(np.std(values, ddof=1)) if len(values) > 1 else None)
    values = [r['miou'] for r in fold_reports if r['miou'] is not None]
    macro['miou'] = dict(mean=float(np.mean(values)) if values else None,
                         sample_sd=float(np.std(values, ddof=1)) if len(values) > 1 else None)
    atomic_save(dict(status='complete',
                     full_oof=set(args.folds) == set(range(1, 6)) and args.max_scenes_per_fold is None,
                     folds=fold_reports, fold_macro=macro,
                     pooled=metrics(np.sum([r['confusion_matrix'] for r in fold_reports], axis=0)),
                     wall_seconds=time.perf_counter() - start),
                args.output_root / 'summary.json')


if __name__ == '__main__':
    main()
