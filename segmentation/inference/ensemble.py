"""Five-fold OOF scene reconstruction for an ENSEMBLE of Stage-3 checkpoints, with optional flip test-time augmentation.

Inference only, no training. Same windowing, probability averaging over overlapping windows, outputs and `summary.json`
layout as `oof.py` (so the scoring scripts read it unchanged). For every window the softmax probabilities of
each member checkpoint (and each TTA view of it) are averaged with equal weight; the window average over the scene then
follows `--window-weight` as before.

Members are given as `--member CKPT_ROOT[:SEED]` (repeatable); the checkpoint of fold NN is
`CKPT_ROOT/<arch>_p256_foldNN_seed<SEED>/<--checkpoint>`, so every member predicts each scene only with the model that
did NOT train on it (out-of-fold). `--tta none` = one view; `--tta flip` = identity, horizontal flip, vertical flip and
both (rotation by 180 degrees); the prediction of a flipped view is flipped back before averaging.
Use a NEW --output-root per (members, tta) configuration.
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
from segmentation.inference.oof import accumulate, colorize, scores, window_weight
from segmentation.models import SPECS

TTA_VIEWS = {'none': [()], 'flip': [(), (3,), (2,), (2, 3)]}   # tensor dims to flip, (N, C, H, W)


def parse_member(text):
    root, _, seed = text.partition(':')
    return Path(root), int(seed) if seed else 42


def member_probabilities(models, images, views):
    """Mean softmax probability (N, 5, 256, 256) over all models and TTA views; each view is undone before averaging."""
    total, n = 0.0, 0
    for model in models:
        for dims in views:
            x = torch.flip(images, dims) if dims else images
            _, logits = model(x)
            probs = F.interpolate(logits, size=(256, 256), mode='bilinear', align_corners=False).float().softmax(1)
            total = total + (torch.flip(probs, dims) if dims else probs)
            n += 1
    return total / n


@torch.inference_mode()
def infer_fold(args, spec, members, fold, device):
    _, frame, masks = load_tables(args.data_root, args.folds_dir, fold, args.masks_root)
    if args.debug_windows:   # TEST ONLY
        frame = frame.head(args.debug_windows)
    models, checkpoints = [], []
    for root, seed in members:
        checkpoint = root / f'{spec["name"]}_p256_fold{fold:02d}_seed{seed}' / args.checkpoint
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if state.get('arch') != spec['name']:
            raise ValueError(f'Checkpoint arch {state.get("arch")!r} is not {spec["name"]!r}: {checkpoint}')
        model = spec['build'](state['model_config'])
        model.load_state_dict(state['model_state'], strict=True)
        models.append(model.to(device).eval())
        checkpoints.append(str(checkpoint.resolve()))
    views = TTA_VIEWS[args.tta]
    loader = DataLoader(Patches(frame, args.data_root, masks, False), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=device.type == 'cuda')
    weight = window_weight(args.window_weight)
    probabilities, counts, weight_sums, row_pos = {}, {}, {}, 0
    for batch_index, (images, _) in enumerate(loader, 1):
        probs = member_probabilities(models, images.to(device, non_blocking=True), views).cpu().numpy()
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
        denominator = counts[scene] if weight is None else weight_sums[scene]
        mean_prob[:, covered] = probabilities[scene][:, covered] / denominator[covered]
        pred[covered] = mean_prob[:, covered].argmax(0).astype(np.uint8)
        accumulate(cm, np.where(pred == 255, 255, target), pred)
        np.save(fold_dir / f'{scene}_prediction.npy', pred)
        if args.save_probs:
            np.save(fold_dir / f'{scene}_probs_oil_la.npy', mean_prob[[1, 2]].astype(np.float16))
        if args.save_rgb:
            Image.fromarray(colorize(pred)).save(fold_dir / f'{scene}_prediction_rgb.png')
            Image.fromarray(colorize(target)).save(fold_dir / f'{scene}_target_rgb.png')
        reports.append({'scene_id': scene, 'covered_fraction': float(covered.mean()),
                        'oil_pixels': int((target == 1).sum()), 'oil_pixels_covered': int(((target == 1) & covered).sum())})
    summary = {'fold': fold, 'checkpoints': checkpoints, 'tta': args.tta, 'scenes': len(reports),
               'window_weight': args.window_weight, 'confusion_matrix': cm.tolist(), 'metrics': scores(cm),
               'scene_reports': reports}
    (fold_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    del models
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return cm, summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    paths.add_root_argument(p)
    p.add_argument('--arch', default='segformer_b2', choices=sorted(SPECS))
    p.add_argument('--member', action='append', required=True, metavar='CKPT_ROOT[:SEED]',
                   help='training output root of one ensemble member, optionally with its seed (default 42); repeatable')
    p.add_argument('--tta', choices=sorted(TTA_VIEWS), default='none')
    p.add_argument('--data-root', type=Path, default=paths.STAGE2_RESULTS / 'training_patches/ps256_overlap50')
    p.add_argument('--folds-dir', type=Path, default=paths.STAGE2_RESULTS / 'grouped_folds_v1')
    p.add_argument('--masks-root', type=Path, default=paths.DATASET / 'train/labels')
    p.add_argument('--checkpoint', choices=('best.pt', 'best_balsel.pt', 'last.pt'), default='best.pt')
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    p.add_argument('--progress-every', type=int, default=25)
    p.add_argument('--save-rgb', action='store_true')
    p.add_argument('--save-probs', action='store_true', help='also save Oil and Look-alike probabilities as float16 (about 3 GB)')
    p.add_argument('--window-weight', choices=('uniform', 'cosine'), default='uniform')
    p.add_argument('--folds', type=int, nargs='+', choices=range(1, 6), default=[1, 2, 3, 4, 5])
    p.add_argument('--debug-windows', type=int, default=0, help='TEST ONLY: use the first N windows per fold')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    members = [parse_member(m) for m in args.member]
    spec = SPECS[args.arch]
    args.output_root = paths.resolve(args.output_root)
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f'Output directory is not empty: {args.output_root}')
    if args.dry_run:
        for fold in args.folds:
            for root, seed in members:
                path = root / f'{spec["name"]}_p256_fold{fold:02d}_seed{seed}' / args.checkpoint
                print(f'fold {fold}: {path} exists={path.exists()}')
        return
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; use --device cpu')
    args.output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    total, summaries = np.zeros((5, 5), dtype=np.int64), []
    for fold in args.folds:
        cm, summary = infer_fold(args, spec, members, fold, device)
        total += cm
        summaries.append(summary)
        print(json.dumps({'fold': fold, 'metrics': summary['metrics']}), flush=True)
    root = {'status': 'complete', 'full_oof': args.folds == [1, 2, 3, 4, 5] and not args.debug_windows,
            'classes': list(CLASSES), 'architecture_variant': spec['name'], 'members': args.member, 'tta': args.tta,
            'window_weight': args.window_weight, 'checkpoint': args.checkpoint, 'metrics': scores(total),
            'confusion_matrix': total.tolist(), 'folds': summaries}
    (args.output_root / 'summary.json').write_text(json.dumps(root, indent=2))
    print(json.dumps(root['metrics'], indent=2))


if __name__ == '__main__':
    main()
