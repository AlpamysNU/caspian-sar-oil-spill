"""Building blocks of the Stage-3 trainer: model helpers, the shifted-window dataset, the epoch loop, the checkpoint
store and the two checkpoint rules.

The data contract of the validated B0 trainers is kept unchanged: 256 px SAR windows, the grouped five-fold manifest,
original masks audited at start-up, optional verified precomputed patch masks, AdamW + cosine schedule, gradient
clipping 1.0. A model is an `nn.Module` whose `forward(image, target=None) -> (loss | None, logits_at_input_resolution)`.

  * `--shift-aug` (see `train.py`): random translation of the window by up to `--shift-max` px with reflect padding of
    image and labels. It is not real scene context; it only decorrelates slick position from the window grid.
  * Both checkpoint rules come out of one run: `best.pt` = highest Oil IoU, and `best_balsel.pt` = the balanced rule
    `select_epoch` (Oil-IoU plateau, Ship-IoU guardrail, balanced score), with `checkpoint_selection_report.json` for
    both. To bound disk use only the `--keep-top-k` highest-Oil-IoU epochs are kept until the end.
"""
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from segmentation.data import CLASSES, Patches, atomic_save

AMP_DTYPES = {'float16': torch.float16, 'bfloat16': torch.bfloat16}


# ----------------------------------------------------------------------------- model helpers
def first_conv_to_single_channel(model):
    """Replace the first 3-channel Conv2d by a 1-channel one initialised with the channel mean."""
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and module.in_channels == 3:
            new = torch.nn.Conv2d(1, module.out_channels, module.kernel_size, module.stride, module.padding,
                                  bias=module.bias is not None)
            with torch.no_grad():
                new.weight.copy_(module.weight.mean(dim=1, keepdim=True))
                if module.bias is not None:
                    new.bias.copy_(module.bias)
            *path, leaf = name.split('.')
            parent = model
            for part in path:
                parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
            if leaf.isdigit():
                parent[int(leaf)] = new
            else:
                setattr(parent, leaf, new)
            return name
    raise ValueError('No 3-channel Conv2d found')


class SegformerStep(torch.nn.Module):
    """HF SegFormer + cross-entropy; used by SegFormer-B2 and by the B0 + shift control."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image, target=None):
        logits = self.model(pixel_values=image).logits
        logits = F.interpolate(logits, size=image.shape[-2:], mode='bilinear', align_corners=False)
        loss = F.cross_entropy(logits, target, ignore_index=255) if target is not None else None
        return loss, logits

    def config_dict(self):
        return self.model.config.to_dict()


def build_segformer(pretrained_id, config=None):
    from transformers import SegformerConfig, SegformerForSemanticSegmentation
    if config is not None:
        return SegformerStep(SegformerForSemanticSegmentation(SegformerConfig.from_dict(config)))
    labels = dict(enumerate(CLASSES))
    model = SegformerForSemanticSegmentation.from_pretrained(
        pretrained_id, num_labels=5, ignore_mismatched_sizes=True,
        id2label=labels, label2id={name: i for i, name in labels.items()})
    first_conv_to_single_channel(model)
    model.config.num_channels = 1
    model.config.semantic_loss_ignore_index = 255
    return SegformerStep(model)


# ----------------------------------------------------------------------------- data
def shift_reflect(image, target, dy, dx, max_shift):
    """Translate by (dy, dx) with reflect padding; `max_shift` bounds |dy|, |dx| and the padding."""
    if dy == 0 and dx == 0:
        return image, target
    pad = ((max_shift, max_shift), (max_shift, max_shift))
    image, target = np.pad(image, pad, mode='reflect'), np.pad(target, pad, mode='reflect')
    h, w = image.shape[0] - 2 * max_shift, image.shape[1] - 2 * max_shift
    rows = slice(max_shift - dy, max_shift - dy + h)
    cols = slice(max_shift - dx, max_shift - dx + w)
    return image[rows, cols], target[rows, cols]


class AugPatches(Patches):
    """Patches with the base flips/rot90 and an optional reflect-padded shift (train only)."""

    def __init__(self, frame, data_root, mask_paths, patch_masks=None, train=False, shift_max=0, shift_prob=0.5):
        super().__init__(frame, data_root, mask_paths, False, patch_masks)
        if shift_max and not 0 < shift_max < 256:
            raise ValueError('shift_max must be in 1..255')
        self.train, self.shift_max, self.shift_prob = train, shift_max, shift_prob

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        if not self.train:
            return image, target
        image, target = image[0].numpy(), target.numpy()
        if random.random() < 0.5:
            image, target = np.fliplr(image), np.fliplr(target)
        if random.random() < 0.5:
            image, target = np.flipud(image), np.flipud(target)
        k = random.randrange(4)
        image, target = np.rot90(image, k), np.rot90(target, k)
        if self.shift_max and random.random() < self.shift_prob:
            image, target = shift_reflect(image, target, random.randint(-self.shift_max, self.shift_max),
                                          random.randint(-self.shift_max, self.shift_max), self.shift_max)
        return (torch.from_numpy(np.ascontiguousarray(image)).float().unsqueeze(0),
                torch.from_numpy(np.ascontiguousarray(target)).long())


# ----------------------------------------------------------------------------- loop
def run_epoch(model, loader, optimizer, scaler, device, amp_dtype):
    training = optimizer is not None
    model.train(training)
    confusion = torch.zeros((5, 5), dtype=torch.int64)
    total_loss, total_pixels, updates, skipped = 0.0, 0, 0, 0
    for image, target in loader:
        image, target = image.to(device), target.to(device)
        valid = target != 255
        pixels = int(valid.sum())
        if not pixels:
            raise ValueError('Batch has no valid pixels')
        with torch.set_grad_enabled(training), torch.autocast(device_type=device.type, dtype=amp_dtype,
                                                              enabled=device.type == 'cuda'):
            loss, logits = model(image, target)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite segmentation loss')
        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradients_finite = torch.stack([
                torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all()
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


def class_ious(cm):
    intersection = cm.diag().double()
    union = cm.sum(0) + cm.sum(1) - cm.diag()
    return [float(intersection[i] / union[i]) if union[i] else None for i in range(5)]


class TopKCheckpoints:
    """Keeps the k highest-Oil-IoU epoch states on disk; everything else is discarded as it is evicted."""

    def __init__(self, directory, k):
        self.directory, self.k, self.kept = Path(directory), k, {}
        self.directory.mkdir()

    def offer(self, epoch, oil_iou, state):
        if len(self.kept) >= self.k and oil_iou <= min(v[0] for v in self.kept.values()):
            return
        path = self.directory / f'epoch{epoch:03d}.pt'
        atomic_save({key: value.detach().cpu().clone() for key, value in state.items()}, path, tensor=True)
        self.kept[epoch] = (oil_iou, path)
        if len(self.kept) > self.k:
            worst = min(self.kept, key=lambda e: self.kept[e][0])
            self.kept.pop(worst)[1].unlink()

    def load(self, epoch):
        return torch.load(self.kept[epoch][1], map_location='cpu', weights_only=True)

    def cleanup(self):
        for _, path in self.kept.values():
            path.unlink(missing_ok=True)
        self.directory.rmdir()


def choose_epochs(history, kept_epochs, baseline_reference):
    """Baseline rule (first epoch with max Oil IoU) and balanced rule, restricted to epochs still on disk."""
    best = None
    for entry in history:
        if best is None or entry['iou']['oil'] > best['iou']['oil']:
            best = entry
    chosen, guardrail_failed, n_candidates, n_valid = select_epoch(history, baseline_reference)
    truncated = chosen['epoch'] not in kept_epochs
    if truncated:
        chosen, guardrail_failed, _, n_valid = select_epoch(
            [e for e in history if e['epoch'] in kept_epochs], baseline_reference)
    return best, chosen, {'guardrail_failed': guardrail_failed, 'n_oil_plateau_candidates': n_candidates,
                          'n_ship_guardrail_valid': n_valid, 'selection_truncated': truncated}


# ----------------------------------------------------------------------------- checkpoint selection
OIL_PLATEAU_FRACTION = 0.98
SHIP_GUARDRAIL_FRACTION = 0.80
SCORE_WEIGHTS = {'oil': 0.5, 'ship': 0.25, 'look_alike': 0.25}


def load_baseline_reference(summary_path):
    """Per-fold {oil, ship, look_alike} IoU from a baseline OOF `summary.json`
    (e.g. oof_gpu_full_v1/summary.json); used by `train.py` for the balanced checkpoint rule."""
    summary = json.loads(Path(summary_path).read_text())
    return {
        f['fold']: {name: f['classes'][name]['iou'] for name in ('oil', 'ship', 'look_alike')}
        for f in summary['folds']
    }


def select_epoch(history, baseline_ref_fold):
    max_oil = max(e['iou']['oil'] for e in history)
    candidates = [e for e in history if e['iou']['oil'] >= OIL_PLATEAU_FRACTION * max_oil]
    ship_floor = SHIP_GUARDRAIL_FRACTION * baseline_ref_fold['ship']
    valid = [e for e in candidates if e['iou']['ship'] >= ship_floor]
    guardrail_failed = not valid
    pool = valid if valid else candidates

    def score(epoch):
        if not valid:
            return epoch['iou']['ship']
        return sum(weight * epoch['iou'][name] / baseline_ref_fold[name]
                   for name, weight in SCORE_WEIGHTS.items())

    chosen = max(pool, key=score)
    return chosen, guardrail_failed, len(candidates), len(valid)
