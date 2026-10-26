"""Five-fold OOF scene reconstruction for the Stage-3 models of `models.py`.

Windows of 256 px with stride 128, uniform probability averaging over overlapping windows, one `summary.json` per fold and for the run
(the scoring scripts read them). The model is built by the SPEC of `models.py` (`build(config)`), the checkpoint is
`<--checkpoints-root>/<arch>_p256_foldNN_seed42/<--checkpoint>`.
`--checkpoint best.pt` is the baseline rule (max Oil IoU); `best_balsel.pt` is the balanced rule.
Use a NEW --output-root per (arch, checkpoint) pair.
By default only `<scene>_prediction.npy` and the summaries are written (about 0.8 GB per run instead of 2.3 GB):
the colourised PNGs are tiny and the window-count `_coverage.npy` maps are the same for every model with these windows
(the scoring scripts read them from the baseline OOF folder only). `--save-rgb` / `--save-coverage` restore them.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from segmentation import paths
from segmentation.data import CLASSES, Patches, load_tables, read_mask
from segmentation.models import SPECS


def colorize(mask):
    palette = np.array([[0, 0, 0], [0, 255, 255], [255, 0, 0], [128, 64, 0], [0, 153, 0]], dtype=np.uint8)
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    valid = (mask >= 0) & (mask < len(palette))
    out[valid] = palette[mask[valid]]
    out[mask == 255] = [255, 0, 255]
    return out


def scores(cm):
    cm = np.asarray(cm, dtype=np.int64)
    result = {}
    ious = []
    for i, name in enumerate(CLASSES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        union = tp + fp + fn
        result[name] = None if union == 0 else float(tp / union)
        if union:
            ious.append(tp / union)
    result['miou'] = float(np.mean(ious)) if ious else None
    return result


def accumulate(cm, target, pred):
    valid = target != 255
    pairs = target[valid].astype(np.int64) * 5 + pred[valid].astype(np.int64)
    cm += np.bincount(pairs, minlength=25).reshape(5, 5)


def window_weight(kind, size=256):
    """Per-window weight map for the probability average: None = uniform (the original reconstruction), 'cosine' =
    separable sine taper sin(pi (k + 0.5) / size) per axis, strictly positive (no division by zero at scene borders), which
    lowers the weight of window borders so that the set of contributing windows changes smoothly instead of in steps."""
    if kind == 'uniform':
        return None
    ramp = np.sin(np.pi * (np.arange(size) + 0.5) / size)
    return np.outer(ramp, ramp)


@torch.inference_mode()
def infer_fold(args, spec, fold, device):
    _, frame, masks = load_tables(args.data_root, args.folds_dir, fold, args.masks_root)
    if args.debug_windows:   # TEST ONLY: the first N windows of the fold
        frame = frame.head(args.debug_windows)
    checkpoint = args.checkpoints_root / f'{spec["name"]}_p256_fold{fold:02d}_seed{args.seed}' / args.checkpoint
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state.get('arch') != spec['name']:
        raise ValueError(f'Checkpoint arch {state.get("arch")!r} is not {spec["name"]!r}: {checkpoint}')
    model = spec['build'](state['model_config'])
    model.load_state_dict(state['model_state'], strict=True)
    model.to(device).eval()
    if 'dataset_factory' in spec:   # two-scale models: windows plus scene context (context.py)
        dataset = spec['dataset_factory'](frame, args.data_root, masks, None, False, 0, 0.5, args.scenes_root)
    else:
        dataset = Patches(frame, args.data_root, masks, False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=device.type == 'cuda')
    weight = window_weight(args.window_weight)
    probabilities, counts, weight_sums, row_pos = {}, {}, {}, 0
    for batch_index, (images, _) in enumerate(loader, 1):
        _, logits = model(images.to(device, non_blocking=True))
        probs = F.interpolate(logits, size=(256, 256), mode='bilinear', align_corners=False).float().softmax(1).cpu().numpy()
        for prob in probs:
            row = frame.iloc[row_pos]
            scene = row.scene_id
            if scene not in probabilities:
                shape = read_mask(masks[scene]).shape
                probabilities[scene] = np.zeros((5, *shape), dtype=np.float64)
                counts[scene] = np.zeros(shape, dtype=np.uint16)
                weight_sums[scene] = np.zeros(shape, dtype=np.float64)
            y, x = int(row.y_orig), int(row.x_orig)
            h, w = read_mask(masks[scene]).shape
            y0, y1, x0, x1 = max(y, 0), min(y + 256, h), max(x, 0), min(x + 256, w)
            window = weight[y0 - y:y1 - y, x0 - x:x1 - x] if weight is not None else 1.0
            probabilities[scene][:, y0:y1, x0:x1] += prob[:, y0 - y:y1 - y, x0 - x:x1 - x] * window
            weight_sums[scene][y0:y1, x0:x1] += window
            counts[scene][y0:y1, x0:x1] += 1
            row_pos += 1
        if batch_index == 1 or batch_index % args.progress_every == 0 or batch_index == len(loader):
            print(f'fold {fold}: batch {batch_index}/{len(loader)} ({100.0 * batch_index / len(loader):.1f}%)', flush=True)
    fold_dir = args.output_root / f'fold{fold:02d}'
    fold_dir.mkdir(parents=True)
    cm = np.zeros((5, 5), dtype=np.int64)
    reports = []
    for scene in sorted(probabilities):
        target = read_mask(masks[scene])
        covered = counts[scene] > 0
        pred = np.full(target.shape, 255, dtype=np.uint8)
        mean_prob = np.zeros_like(probabilities[scene])
        denominator = counts[scene] if weight is None else weight_sums[scene]   # uniform: window count, as before
        mean_prob[:, covered] = probabilities[scene][:, covered] / denominator[covered]
        pred[covered] = mean_prob[:, covered].argmax(0).astype(np.uint8)
        accumulate(cm, np.where(pred == 255, 255, target), pred)   # same as before when fully covered
        np.save(fold_dir / f'{scene}_prediction.npy', pred)
        if args.save_probs:
            np.save(fold_dir / f'{scene}_probs_oil_la.npy', mean_prob[[1, 2]].astype(np.float16))
        if args.save_rgb:
            Image.fromarray(colorize(pred)).save(fold_dir / f'{scene}_prediction_rgb.png')
            Image.fromarray(colorize(target)).save(fold_dir / f'{scene}_target_rgb.png')
        if args.save_coverage:   # identical for every model with the same windows; the scoring scripts read the baseline's
            np.save(fold_dir / f'{scene}_coverage.npy', counts[scene])
        reports.append({'scene_id': scene, 'covered_fraction': float(covered.mean()),
                        'oil_pixels': int((target == 1).sum()), 'oil_pixels_covered': int(((target == 1) & covered).sum())})
    summary = {'fold': fold, 'checkpoint': str(checkpoint.resolve()), 'checkpoint_epoch': state['epoch'],
               'scenes': len(reports), 'window_weight': args.window_weight, 'confusion_matrix': cm.tolist(), 'metrics': scores(cm), 'scene_reports': reports}
    (fold_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return cm, summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    paths.add_root_argument(p)
    p.add_argument('--arch', required=True, choices=sorted(SPECS))
    p.add_argument('--data-root', type=Path, default=paths.STAGE2_RESULTS / 'training_patches/ps256_overlap50')
    p.add_argument('--folds-dir', type=Path, default=paths.STAGE2_RESULTS / 'grouped_folds_v1')
    p.add_argument('--masks-root', type=Path, default=paths.DATASET / 'train/labels')
    p.add_argument('--checkpoints-root', type=Path, required=True)
    p.add_argument('--scenes-root', type=Path, default=paths.DATASET / 'train/images', help='original scene jpgs (two-scale models only)')
    p.add_argument('--checkpoint', choices=('best.pt', 'best_balsel.pt', 'last.pt'), default='best.pt')
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    p.add_argument('--progress-every', type=int, default=25)
    p.add_argument('--save-rgb', action='store_true', help='also write the colourised prediction/target PNGs (off by default)')
    p.add_argument('--save-coverage', action='store_true', help='also write per-scene window-count maps (off by default, 1.6 GB per run)')
    p.add_argument('--save-probs', action='store_true',
                   help='also save the window-averaged Oil and Look-alike probabilities per scene as float16 <scene>_probs_oil_la.npy (2 x H x W, about 3 GB per run; off by default)')
    p.add_argument('--window-weight', choices=('uniform', 'cosine'), default='uniform',
                   help='probability average over overlapping windows: uniform (default, all earlier OOF runs) or cosine taper')
    p.add_argument('--folds', type=int, nargs='+', choices=range(1, 6), default=[1, 2, 3, 4, 5])
    p.add_argument('--debug-windows', type=int, default=0, help='TEST ONLY: use the first N windows per fold')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    spec = SPECS[args.arch]
    args.checkpoints_root, args.output_root = paths.resolve(args.checkpoints_root), paths.resolve(args.output_root)
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f'Output directory is not empty: {args.output_root}')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; use --device cpu')
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        for fold in range(1, 6):
            _, frame, _ = load_tables(args.data_root, args.folds_dir, fold, args.masks_root)
            print(f'fold {fold}: {len(frame)} patches, checkpoint='
                  f'{args.checkpoints_root / (spec["name"] + f"_p256_fold{fold:02d}_seed{args.seed}") / args.checkpoint}')
        return
    device = torch.device(args.device)
    total, summaries = np.zeros((5, 5), dtype=np.int64), []
    for fold in args.folds:
        cm, summary = infer_fold(args, spec, fold, device)
        total += cm
        summaries.append(summary)
        print(json.dumps({'fold': fold, 'metrics': summary['metrics']}), flush=True)
    root = {'status': 'complete', 'full_oof': args.folds == [1, 2, 3, 4, 5] and not args.debug_windows, 'classes': list(CLASSES), 'architecture_variant': spec['name'], 'window_weight': args.window_weight,
            'checkpoint': args.checkpoint, 'metrics': scores(total), 'confusion_matrix': total.tolist(), 'folds': summaries}
    (args.output_root / 'summary.json').write_text(json.dumps(root, indent=2))
    print(json.dumps(root['metrics'], indent=2))


if __name__ == '__main__':
    main()
