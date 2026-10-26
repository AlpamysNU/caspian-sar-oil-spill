"""Train one Stage-3 model on one fold of the grouped five-fold protocol:

    python -m segmentation.train --model segformer_b2_hrda_c --fold 1 --shift-aug --output-root results/stage3_segformer_b2_hrda_c_shift_256

`--model` selects a SPEC of `models.py`. Inputs default to the layout under `--project-root` (see `paths.py`); the official
test split is never loaded here. Both checkpoint rules come out of one run (`best.pt` = max Oil IoU, `best_balsel.pt` =
balanced rule of `common.select_epoch`), plus `last.pt`, `history.json` and `checkpoint_selection_report.json`.
No `--resume`: a restart needs a new output root.
"""
import argparse
import fcntl
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from segmentation import paths
from segmentation.common import (AMP_DTYPES, AugPatches, TopKCheckpoints, choose_epochs, class_ious, load_baseline_reference,
                                 run_epoch)
from segmentation.data import CLASSES, atomic_save, load_tables, seed_worker, sha256, verify_patch_masks
from segmentation.models import SPECS


def main():
    chosen = argparse.ArgumentParser(add_help=False)
    chosen.add_argument('--model', choices=sorted(SPECS))
    name = chosen.parse_known_args()[0].model
    if name is None:   # no model yet: a plain --help lists them
        argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args(
            [a for a in sys.argv[1:] if a in ('-h', '--help')] or ['--help'])
    spec = SPECS[name]
    parser = argparse.ArgumentParser(description=spec['description'], formatter_class=argparse.RawDescriptionHelpFormatter)
    paths.add_root_argument(parser)
    parser.add_argument('--model', choices=sorted(SPECS), required=True)
    parser.add_argument('--data-root', type=Path, default=paths.STAGE2_RESULTS / 'training_patches/ps256_overlap50')
    parser.add_argument('--folds-dir', type=Path, default=paths.STAGE2_RESULTS / 'grouped_folds_v1')
    parser.add_argument('--masks-root', type=Path, default=paths.DATASET / 'train/labels', help='Original PNG masks named img_NNNN.png')
    parser.add_argument('--output-root', required=True, type=Path, help='A new dedicated root for this model (relative to the project root)')
    parser.add_argument('--patch-masks-root', type=Path, help='Verified precomputed mask bundle')
    parser.add_argument('--scenes-root', type=Path, default=paths.DATASET / 'train/images', help='Original scene jpgs (two-scale models)')
    parser.add_argument('--baseline-oof-summary', type=Path, default=paths.RESULTS / 'oof_gpu_full_v1/summary.json',
                        help='per-fold reference of the B0 baseline for the balanced rule')
    parser.add_argument('--fold', type=int, choices=range(1, 6), required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=spec['lr'])
    parser.add_argument('--amp-dtype', choices=sorted(AMP_DTYPES), default=spec.get('amp_dtype', 'float16'))
    parser.add_argument('--shift-aug', action='store_true', help='Random reflect-padded translation of the window')
    parser.add_argument('--shift-max', type=int, default=64)
    parser.add_argument('--shift-prob', type=float, default=0.5)
    parser.add_argument('--keep-top-k', type=int, default=25)
    parser.add_argument('--debug-samples', type=int, default=0,
                        help='TEST ONLY: train/validate on N Oil-positive + N other windows per partition')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='One real GPU train/validation batch; no checkpoint')
    args = parser.parse_args()
    if not args.output_root.is_absolute():
        args.output_root = args.project_root / args.output_root
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or args.keep_top_k < 1:
        parser.error('epochs/batch-size/keep-top-k must be positive; workers must be nonnegative')
    if not 0 <= args.shift_prob <= 1:
        parser.error('shift-prob must be in [0, 1]')
    shift_max = args.shift_max if args.shift_aug else 0

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train, val, masks = load_tables(args.data_root, args.folds_dir, args.fold, args.masks_root)
    print(json.dumps({'arch': spec['name'], 'fold': args.fold, 'train_scenes': train.scene_id.nunique(),
                      'val_scenes': val.scene_id.nunique(), 'train_patches': len(train), 'val_patches': len(val),
                      'shift_aug': bool(shift_max)}, indent=2))
    patch_masks = None
    if args.patch_masks_root is not None:
        patch_masks = verify_patch_masks(args.patch_masks_root, args.data_root,
                                         pd.concat([train, val], ignore_index=True))
        print(f'Precomputed masks verified: {len(patch_masks)}', flush=True)
    if 'dataset_factory' in spec:
        train_ds = spec['dataset_factory'](train, args.data_root, masks, patch_masks, True, shift_max, args.shift_prob, args.scenes_root)
        val_ds = spec['dataset_factory'](val, args.data_root, masks, patch_masks, False, 0, 0.5, args.scenes_root)
    else:
        train_ds = AugPatches(train, args.data_root, masks, patch_masks, True, shift_max, args.shift_prob)
        val_ds = AugPatches(val, args.data_root, masks, patch_masks)
    for dataset in (train_ds, val_ds):
        for index in (0, len(dataset) // 2, len(dataset) - 1):
            dataset[index]
    print('Mask/geometry sample audit passed')
    if args.dry_run:
        return
    if args.debug_samples:
        def subset(ds, frame):
            oil = frame.index[frame.label_oil == 1].tolist()[:args.debug_samples]
            rest = frame.index[frame.label_oil != 1].tolist()[:args.debug_samples]
            return Subset(ds, oil + rest)
        train_ds, val_ds = subset(train_ds, train), subset(val_ds, val)
        print(f'DEBUG: {len(train_ds)} train / {len(val_ds)} val windows', flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required for training')
    device = torch.device('cuda')
    amp_dtype = AMP_DTYPES[args.amp_dtype]
    scaler = torch.cuda.amp.GradScaler(enabled=amp_dtype == torch.float16)

    if args.smoke:
        train_ids = train.index[train.label_oil == 1].tolist()[:args.batch_size]
        val_ids = val.index[val.label_oil == 1].tolist()[:args.batch_size]
        if not train_ids or not val_ids:
            raise ValueError('Smoke probe needs Oil-positive patches in both partitions')
        probe = spec['build']().to(device)
        probe_optimizer = torch.optim.AdamW(spec['param_groups'](probe, args.lr), lr=args.lr, weight_decay=0.01)
        probe_train = DataLoader(Subset(train_ds, train_ids), batch_size=len(train_ids), num_workers=0)
        probe_val = DataLoader(Subset(val_ds, val_ids), batch_size=len(val_ids), num_workers=0)
        train_loss, _, steps = run_epoch(probe, probe_train, probe_optimizer, scaler, device, amp_dtype)
        val_loss, cm, _ = run_epoch(probe, probe_val, None, scaler, device, amp_dtype)
        print(json.dumps({'smoke': 'passed', 'arch': spec['name'], 'train_loss': train_loss, 'val_loss': val_loss,
                          'optimizer_updates': steps['optimizer_updates'], 'val_oil_pixels': int(cm[1].sum()),
                          'parameters_m': sum(p.numel() for p in probe.parameters()) / 1e6}), flush=True)
        return

    reference_all = load_baseline_reference(args.baseline_oof_summary)
    if args.fold not in reference_all:
        raise ValueError(f'No baseline reference for fold {args.fold}')
    baseline_reference = reference_all[args.fold]
    run_dir = args.output_root / f'{spec["name"]}_p256_fold{args.fold:02d}_seed{args.seed}'
    run_dir.mkdir(parents=True, exist_ok=True)
    run_lock = (run_dir / '.lock').open('a')
    try:
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError(f'Another process owns this run: {run_dir}') from None
    if any(path.name != '.lock' for path in run_dir.iterdir()):
        raise FileExistsError('Run already contains outputs; use a new output root (no --resume support)')

    augmentation = 'horizontal/vertical flip p=0.5; uniform rot90'
    if shift_max:
        augmentation += f'; reflect-padded shift up to {shift_max} px, p={args.shift_prob}'
    model = spec['build']().to(device)
    identity = {
        'arch': spec['name'], 'model': spec['pretrained'], 'fold': args.fold, 'seed': args.seed,
        'epochs': args.epochs, 'batch_size': args.batch_size, 'workers': args.workers, 'classes': list(CLASSES),
        'loss': spec['loss'], 'normalization': 'uint8 / 255; one SAR channel', 'augmentation': augmentation,
        'shift_aug': bool(shift_max), 'optimizer': 'AdamW', 'lr': args.lr, 'weight_decay': 0.01,
        'param_groups': [{'lr': g['lr'], 'parameters': sum(p.numel() for p in g['params'])}
                         for g in spec['param_groups'](model, args.lr)],
        'scheduler': 'cosine', 'eta_min': 1e-6, 'gradient_clip': 1.0, 'amp_dtype': args.amp_dtype,
        'checkpoints': 'best.pt = max Oil IoU (baseline rule); best_balsel.pt = lexicographic_oil_plateau_ship_guardrail_v1',
        'keep_top_k': args.keep_top_k, 'source_sha256': sha256(spec['file']), 'common_sha256': sha256(Path(__file__).with_name('common.py')),
        'baseline_oof_summary_sha256': sha256(args.baseline_oof_summary), 'torch_version': str(torch.__version__),
        'fold_manifest_sha256': sha256(args.folds_dir / 'scene_folds.csv'),
        'scene_manifest_sha256': sha256(args.data_root / 'scene_manifest.csv'),
        'patch_index_sha256': sha256(next(args.data_root.glob('index_*.parquet'))),
    }
    if args.patch_masks_root is not None:
        identity['patch_masks_manifest_sha256'] = sha256(args.patch_masks_root / 'manifest.json')
    atomic_save(identity, run_dir / 'config.json')

    train_generator = torch.Generator().manual_seed(args.seed)
    val_generator = torch.Generator().manual_seed(args.seed + 1)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True,
                              worker_init_fn=seed_worker, generator=train_generator, drop_last=False)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True,
                            worker_init_fn=seed_worker, generator=val_generator)
    optimizer = torch.optim.AdamW(spec['param_groups'](model, args.lr), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    top = TopKCheckpoints(run_dir / 'epoch_checkpoints', args.keep_top_k)
    history = []
    for epoch in range(1, args.epochs + 1):
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        start = time.monotonic()
        train_loss, _, steps = run_epoch(model, train_loader, optimizer, scaler, device, amp_dtype)
        torch.cuda.synchronize(device)
        train_seconds = time.monotonic() - start
        val_loss, cm, _ = run_epoch(model, val_loader, None, scaler, device, amp_dtype)
        torch.cuda.synchronize(device)
        iou = class_ious(cm)
        if iou[1] is None:
            raise ValueError('Validation has no Oil support or predictions')
        metrics = {'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
                   'lr': optimizer.param_groups[0]['lr'], **steps, 'train_seconds': train_seconds,
                   'epoch_seconds': time.monotonic() - start,
                   'peak_gpu_memory_gib': torch.cuda.max_memory_reserved(device) / 1024 ** 3,
                   'confusion_matrix': cm.tolist(), 'iou': dict(zip(CLASSES, iou)),
                   'miou': float(np.mean([v for v in iou if v is not None]))}
        history.append(metrics)
        scheduler.step()
        top.offer(epoch, iou[1], model.state_dict())
        atomic_save(history, run_dir / 'history.json')
        print(json.dumps(metrics), flush=True)

    best, chosen, extra = choose_epochs(history, set(top.kept), baseline_reference)
    config = model.config_dict()

    def export(name, epoch):
        atomic_save({'model_state': top.load(epoch), 'model_config': config, 'identity': identity,
                     'epoch': epoch, 'classes': CLASSES, 'arch': spec['name']}, run_dir / name, tensor=True)

    export('best.pt', best['epoch'])
    export('best_balsel.pt', chosen['epoch'])
    atomic_save({'model_state': {k: v.detach().cpu() for k, v in model.state_dict().items()}, 'model_config': config,
                 'identity': identity, 'epoch': args.epochs, 'classes': CLASSES, 'arch': spec['name']},
                run_dir / 'last.pt', tensor=True)
    report = {'rule_best_oil': {'epoch': best['epoch'], 'iou': best['iou']},
              'rule_balsel': {'rule': 'lexicographic_oil_plateau_ship_guardrail_v1', 'epoch': chosen['epoch'],
                              'iou': chosen['iou'], **extra},
              'baseline_reference': baseline_reference}
    atomic_save(report, run_dir / 'checkpoint_selection_report.json')
    print(json.dumps(report, indent=2))
    top.cleanup()


if __name__ == '__main__':
    main()
