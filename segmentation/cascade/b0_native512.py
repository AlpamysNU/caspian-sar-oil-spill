"""SegFormer-B0 trainer on native 512 px windows (stride 256): the native-512 arm of the cascade experiment (E1(b)).

Inputs are native 512x512 SAR patches, original RGB masks and the frozen grouped_folds_v1 manifest (patch index `ps512_overlap50`). The
official test split is never loaded here. Checkpoints are written as `segformer_b0_p512_foldNN_seed42/best.pt`. It is the "native backend"
of `infer_b0_oof.py --patch-size 512` and `infer_cascade.py --patch-size 512`.
"""
import argparse
import fcntl
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from segmentation import paths as project_paths
from segmentation.data import CLASSES, atomic_save, read_mask, restore_rng, rng_state, seed_worker, sha256

PRETRAINED = 'nvidia/segformer-b0-finetuned-ade-512-512'


PRETRAINED = 'nvidia/segformer-b0-finetuned-ade-512-512'


def load_tables(data_root, folds_dir, fold, masks_root):
    data_root, folds_dir, masks_root = map(Path, (data_root, folds_dir, masks_root))
    audit = json.loads((folds_dir / 'audit.json').read_text())
    if fold not in range(1, audit['n_folds'] + 1):
        raise ValueError('Invalid fold')
    folds_path = folds_dir / 'scene_folds.csv'
    if sha256(folds_path) != audit['manifest_sha256']:
        raise ValueError('Frozen fold manifest hash mismatch')
    indexes = list(data_root.glob('index_*.parquet'))
    if len(indexes) != 1:
        entries = sorted(path.name for path in data_root.iterdir()) if data_root.is_dir() else []
        raise ValueError(
            f'Expected exactly one native patch index in {data_root.resolve()}; '
            f'found {len(indexes)}: {[str(path) for path in indexes]}; '
            f'directory_exists={data_root.is_dir()}, entries={entries[:30]}')
    for path in (indexes[0], data_root / 'scene_manifest.csv', data_root / 'verification.json'):
        hashes = [digest for name, digest in audit['input_sha256'].items()
                  if Path(name).parts[-2:] == path.parts[-2:]]
        if len(hashes) != 1 or sha256(path) != hashes[0]:
            raise ValueError(f'Frozen data hash mismatch: {path}')
    frame = pd.read_parquet(indexes[0])
    scenes = pd.read_csv(folds_path)
    scene_manifest = pd.read_csv(data_root / 'scene_manifest.csv')
    if set(frame.split) != {'train'} or set(scenes.split) != {'train'}:
        raise ValueError('Official test data must not enter fold training')
    if len(frame) != 8016 or set(frame.patch_size) != {512} or set(frame.stride) != {256}:
        raise ValueError('Expected frozen 512/256 patch grid')
    if frame.patch_id.duplicated().any() or scenes.scene_id.duplicated().any():
        raise ValueError('Duplicate patch/scene IDs')
    if set(frame.scene_id) != set(scenes.scene_id) or set(scene_manifest.scene_id) != set(scenes.scene_id):
        raise ValueError('Scene manifests disagree')
    for key in ('scene_id', 'group_id', 'image_sha256'):
        if set(scenes.loc[scenes.fold == fold, key]) & set(scenes.loc[scenes.fold != fold, key]):
            raise ValueError(f'Fold leakage: {key}')
    mask_paths = {}
    for row in scene_manifest.itertuples():
        path = masks_root / (row.scene_id.removeprefix('train_') + '.png')
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256(path) != row.mask_sha256:
            raise ValueError(f'Original mask hash mismatch: {row.scene_id}')
        mask_paths[row.scene_id] = path
    frame = frame.merge(scenes[['scene_id', 'fold']], on='scene_id', validate='many_to_one')
    return frame[frame.fold != fold].reset_index(drop=True), frame[frame.fold == fold].reset_index(drop=True), mask_paths


def crop_patch_mask(mask, row):
    y, x = int(row['y_orig']), int(row['x_orig'])
    height, width = mask.shape
    y0, y1 = max(y, 0), min(y + 512, height)
    x0, x1 = max(x, 0), min(x + 512, width)
    if y0 >= y1 or x0 >= x1:
        raise ValueError('Patch has no valid pixels')
    target = np.full((512, 512), 255, dtype=np.uint8)
    target[y0-y:y1-y, x0-x:x1-x] = mask[y0:y1, x0:x1]
    if int(np.sum(target != 255)) != int(row['valid_pixels']):
        raise ValueError(f'Valid-pixel geometry mismatch: {row["patch_id"]}')
    counts = np.bincount(target[target != 255].ravel(), minlength=5)
    for class_id, name in enumerate(CLASSES):
        if int(counts[class_id]) != int(row[f'pixels_{name}']):
            raise ValueError(f'Mask class-count mismatch: {row["patch_id"]}, {name}')
    return target


def patch_mask_contract(data_root):
    data_root = Path(data_root)
    return {
        'version': 1, 'classes': list(CLASSES), 'ignore_index': 255,
        'shape': [512, 512], 'dtype': 'uint8',
        'patch_index_sha256': sha256(next(data_root.glob('index_*.parquet'))),
        'scene_manifest_sha256': sha256(data_root / 'scene_manifest.csv'),
    }


def verify_patch_masks(root, data_root, frame):
    """Audit the whole bundle once per launch, never inside the epoch loop."""
    root = Path(root).resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['contract'] != patch_mask_contract(data_root):
        raise ValueError('Precomputed masks belong to a different dataset or class mapping')
    files = manifest['files']
    if len(frame) != len(files) or set(frame.patch_id) != set(files):
        raise ValueError('Precomputed mask patch IDs do not match the frozen index')
    paths = {}
    for patch_id, digest in files.items():
        path = (root / 'masks_npy' / f'{patch_id}.npy').resolve()
        if not path.is_relative_to(root / 'masks_npy'):
            raise ValueError('Mask path escapes bundle')
        if sha256(path) != digest:
            raise ValueError(f'Precomputed mask hash mismatch: {patch_id}')
        paths[patch_id] = path
    return paths


class Patches(Dataset):
    def __init__(self, frame, data_root, mask_paths, augment=False, patch_masks=None):
        self.rows = frame.to_dict('records')
        self.data_root = Path(data_root).resolve()
        self.mask_paths = mask_paths
        self.augment = augment
        self.patch_masks = patch_masks

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = (self.data_root / row['patch_path']).resolve()
        if not path.is_relative_to(self.data_root):
            raise ValueError('Patch path escapes data root')
        image = np.load(path, allow_pickle=False)
        if image.shape != (512, 512) or image.dtype != np.uint8:
            raise ValueError(f'Invalid SAR patch: {path}')
        if self.patch_masks is None:
            mask = read_mask(self.mask_paths[row['scene_id']])
            target = crop_patch_mask(mask, row)
        else:
            target = np.load(self.patch_masks[row['patch_id']], allow_pickle=False)
            if target.shape != (512, 512) or target.dtype != np.uint8:
                raise ValueError(f'Invalid cached mask: {row["patch_id"]}')
        if self.augment:
            if random.random() < 0.5:
                image, target = np.fliplr(image), np.fliplr(target)
            if random.random() < 0.5:
                image, target = np.flipud(image), np.flipud(target)
            k = random.randrange(4)
            image, target = np.rot90(image, k), np.rot90(target, k)
        return torch.from_numpy(np.ascontiguousarray(image)).float().unsqueeze(0) / 255.0, torch.from_numpy(np.ascontiguousarray(target)).long()


def model_from_pretrained(config=None):
    from transformers import SegformerConfig, SegformerForSemanticSegmentation
    if config is not None:
        return SegformerForSemanticSegmentation(SegformerConfig.from_dict(config))
    labels = dict(enumerate(CLASSES))
    model = SegformerForSemanticSegmentation.from_pretrained(
        PRETRAINED, num_labels=5, ignore_mismatched_sizes=True,
        id2label=labels, label2id={name: i for i, name in labels.items()})
    old = model.segformer.encoder.patch_embeddings[0].proj
    new = torch.nn.Conv2d(1, old.out_channels, old.kernel_size, old.stride, old.padding, bias=old.bias is not None)
    with torch.no_grad():
        new.weight.copy_(old.weight.mean(dim=1, keepdim=True))
        if old.bias is not None:
            new.bias.copy_(old.bias)
    model.segformer.encoder.patch_embeddings[0].proj = new
    model.config.num_channels = 1
    model.config.semantic_loss_ignore_index = 255
    return model


def run_epoch(model, loader, optimizer, scaler, device):
    training = optimizer is not None
    model.train(training)
    confusion = torch.zeros((5, 5), dtype=torch.int64)
    total_loss = 0.0
    total_pixels = 0
    updates = skipped = 0
    for image, target in loader:
        image, target = image.to(device), target.to(device)
        valid = target != 255
        pixels = int(valid.sum())
        if not pixels:
            raise ValueError('Batch has no valid pixels')
        with torch.set_grad_enabled(training), torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            logits = model(pixel_values=image).logits
            logits = F.interpolate(logits, size=target.shape[-2:], mode='bilinear', align_corners=False)
            loss = F.cross_entropy(logits, target, ignore_index=255)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite segmentation loss')
        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradients_finite = torch.stack([
                torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
            ]).all()
            if gradients_finite:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            elif not scaler.is_enabled():
                raise FloatingPointError('Nonfinite gradients without AMP recovery')
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < previous_scale:
                skipped += 1
            else:
                updates += 1
        total_loss += float(loss.detach()) * pixels
        total_pixels += pixels
        pred = logits.detach().argmax(1)
        pairs = target[valid] * 5 + pred[valid]
        confusion += torch.bincount(pairs.cpu(), minlength=25).reshape(5, 5)
    if not total_pixels or (training and not updates):
        raise RuntimeError('Epoch has no valid pixels or successful optimizer updates')
    return total_loss / total_pixels, confusion, {'optimizer_updates': updates, 'amp_skipped_steps': skipped}


def main():
    parser = argparse.ArgumentParser()
    project_paths.add_root_argument(parser)
    parser.add_argument('--data-root', type=Path, default=project_paths.STAGE2_RESULTS / 'training_patches/ps512_overlap50')
    parser.add_argument('--folds-dir', type=Path, default=project_paths.STAGE2_RESULTS / 'grouped_folds_v1')
    parser.add_argument('--masks-root', type=Path, default=project_paths.DATASET / 'train/labels', help='Original PNG masks named img_NNNN.png')
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--patch-masks-root', type=Path, required=True,
                        help='Verified precomputed mask bundle; original masks still audited at startup')
    parser.add_argument('--fold', type=int, choices=range(1, 6), required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--stop-after-epoch', type=int, default=None,
                        help='Stop after this epoch while keeping the full --epochs LR schedule; resumable')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='One real GPU train/validation batch; no checkpoint')
    parser.add_argument('--resume', action='store_true', help='Continue this fold from last.pt')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error('epochs/batch-size must be positive; workers must be nonnegative')
    if args.stop_after_epoch is not None and not 1 <= args.stop_after_epoch <= args.epochs:
        parser.error('stop-after-epoch must be between 1 and epochs')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    train, val, masks = load_tables(args.data_root, args.folds_dir, args.fold, args.masks_root)
    print(json.dumps({'fold': args.fold, 'train_scenes': train.scene_id.nunique(), 'val_scenes': val.scene_id.nunique(), 'train_patches': len(train), 'val_patches': len(val)}, indent=2))
    patch_masks = None
    if args.patch_masks_root is not None:
        patch_masks = verify_patch_masks(args.patch_masks_root, args.data_root,
                                         pd.concat([train, val], ignore_index=True))
        print(f'Precomputed masks verified: {len(patch_masks)}', flush=True)
    train_ds = Patches(train, args.data_root, masks, True, patch_masks)
    val_ds = Patches(val, args.data_root, masks, patch_masks=patch_masks)
    for dataset in (train_ds, val_ds):
        for index in (0, len(dataset)//2, len(dataset)-1):
            dataset[index]
    print('Mask/geometry sample audit passed')
    if args.dry_run:
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required for training')
    device = torch.device('cuda')
    if args.smoke:
        # Include Oil-positive windows so the probe exercises the target class.
        train_ids = train.index[train.label_oil == 1].tolist()[:args.batch_size]
        val_ids = val.index[val.label_oil == 1].tolist()[:args.batch_size]
        if not train_ids or not val_ids:
            raise ValueError('Smoke probe needs Oil-positive patches in both partitions')
        probe_model = model_from_pretrained().to(device)
        probe_optimizer = torch.optim.AdamW(probe_model.parameters(), lr=6e-5)
        probe_scaler = torch.cuda.amp.GradScaler()
        probe_train = DataLoader(Subset(train_ds, train_ids), batch_size=len(train_ids), num_workers=0)
        probe_val = DataLoader(Subset(val_ds, val_ids), batch_size=len(val_ids), num_workers=0)
        train_loss, _, updates = run_epoch(probe_model, probe_train, probe_optimizer, probe_scaler, device)
        val_loss, cm, _ = run_epoch(probe_model, probe_val, None, probe_scaler, device)
        print(json.dumps({'smoke': 'passed', 'train_loss': train_loss, 'val_loss': val_loss,
                          'optimizer_updates': updates['optimizer_updates'],
                          'val_oil_pixels': int(cm[1].sum())}), flush=True)
        return
    run_dir = args.output_root / f'segformer_b0_p512_fold{args.fold:02d}_seed{args.seed}'
    run_dir.mkdir(parents=True, exist_ok=True)
    # Keep this descriptor alive until main exits, including direct Python launches.
    run_lock = (run_dir / '.lock').open('a')
    try:
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError(f'Another process owns this run: {run_dir}') from None
    identity = {
        'model': PRETRAINED, 'fold': args.fold, 'seed': args.seed,
        'patch_size': 512, 'stride': 256,
        'epochs': args.epochs, 'batch_size': args.batch_size, 'workers': args.workers,
        'classes': list(CLASSES), 'loss': 'cross_entropy_ignore_padding_255',
        'normalization': 'uint8 / 255; one SAR channel',
        'augmentation': 'horizontal/vertical flip p=0.5; uniform rot90',
        'optimizer': 'AdamW', 'lr': 6e-5, 'weight_decay': 0.01,
        'scheduler': 'cosine', 'eta_min': 1e-6, 'gradient_clip': 1.0,
        'source_sha256': sha256(__file__), 'torch_version': str(torch.__version__),
        'fold_manifest_sha256': sha256(args.folds_dir / 'scene_folds.csv'),
        'scene_manifest_sha256': sha256(args.data_root / 'scene_manifest.csv'),
        'patch_index_sha256': sha256(next(args.data_root.glob('index_*.parquet'))),
    }
    if args.patch_masks_root is not None:
        identity['patch_masks_manifest_sha256'] = sha256(args.patch_masks_root / 'manifest.json')
    state = None
    if args.resume:
        state = torch.load(run_dir / 'last.pt', map_location='cpu', weights_only=False)
        if state.get('identity') != identity:
            raise ValueError('Resume configuration/code/data differ; use a new output root')
    elif any(path.name != '.lock' for path in run_dir.iterdir()):
        raise FileExistsError('Run already contains outputs; use --resume or a new output root')
    train_generator = torch.Generator().manual_seed(args.seed)
    val_generator = torch.Generator().manual_seed(args.seed + 1)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=True, worker_init_fn=seed_worker, generator=train_generator)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=args.workers,
                            pin_memory=True, worker_init_fn=seed_worker, generator=val_generator)
    model = model_from_pretrained(state['model_config'] if state else None).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-5, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler()
    best = -1.0
    history = []
    best_weights = None
    best_epoch = 0
    start_epoch = 1
    if state is not None:
        model.load_state_dict(state['model_state'])
        optimizer.load_state_dict(state['optimizer_state'])
        scheduler.load_state_dict(state['scheduler_state'])
        scaler.load_state_dict(state['scaler_state'])
        best = state['best_oil_iou']
        start_epoch = state['epoch'] + 1
        history = state['history']
        best_weights, best_epoch = state['best_model_state'], state['best_epoch']
        train_generator.set_state(state['train_generator'])
        val_generator.set_state(state['val_generator'])
        restore_rng(state['rng'])
    atomic_save(identity, run_dir / 'config.json')
    # last.pt is authoritative if a job stopped while exporting history/best.
    if state is not None:
        atomic_save(history, run_dir / 'history.json')
        atomic_save({'model_state': best_weights, 'model_config': model.config.to_dict(),
                     'identity': identity, 'epoch': best_epoch, 'classes': CLASSES},
                    run_dir / 'best.pt', tensor=True)
    final_epoch = args.stop_after_epoch or args.epochs
    for epoch in range(start_epoch, final_epoch + 1):
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        epoch_start = time.monotonic()
        train_loss, _, steps = run_epoch(model, train_loader, optimizer, scaler, device)
        torch.cuda.synchronize(device)
        train_seconds = time.monotonic() - epoch_start
        val_loss, cm, _ = run_epoch(model, val_loader, None, scaler, device)
        torch.cuda.synchronize(device)
        epoch_seconds = time.monotonic() - epoch_start
        intersection = cm.diag().double()
        union = cm.sum(0) + cm.sum(1) - cm.diag()
        iou = [float(intersection[i] / union[i]) if union[i] else None for i in range(5)]
        if iou[1] is None:
            raise ValueError('Validation has no Oil support or predictions')
        metrics = {'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
                   'lr': optimizer.param_groups[0]['lr'], **steps,
                   'train_seconds': train_seconds, 'epoch_seconds': epoch_seconds,
                   'peak_gpu_memory_gib': torch.cuda.max_memory_reserved(device) / 1024**3,
                   'confusion_matrix': cm.tolist(), 'iou': dict(zip(CLASSES, iou)),
                   'miou': float(np.mean([v for v in iou if v is not None]))}
        history.append(metrics)
        if iou[1] > best:
            best = iou[1]
            best_weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch
        scheduler.step()
        checkpoint = {'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
                      'scheduler_state': scheduler.state_dict(), 'scaler_state': scaler.state_dict(),
                      'epoch': epoch, 'identity': identity, 'model_config': model.config.to_dict(),
                      'best_oil_iou': best, 'best_epoch': best_epoch, 'best_model_state': best_weights,
                      'history': history, 'rng': rng_state(),
                      'train_generator': train_generator.get_state(), 'val_generator': val_generator.get_state()}
        atomic_save(checkpoint, run_dir / 'last.pt', tensor=True)
        atomic_save(history, run_dir / 'history.json')
        atomic_save({'model_state': best_weights, 'model_config': model.config.to_dict(),
                     'identity': identity, 'epoch': best_epoch, 'classes': CLASSES},
                    run_dir / 'best.pt', tensor=True)
        print(json.dumps(metrics), flush=True)


if __name__ == '__main__':
    main()
