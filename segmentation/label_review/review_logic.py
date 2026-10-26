"""Pure (GUI-free) logic of the manual label review (napari): loading, validation, saving into the REVISED copy only, change log.

Safety rules enforced here (2026-10-04): the original dataset is never written (the output directory must not be inside it); the revised map has the same
shape as the SAR image and the original mask, dtype uint8, values in 0..4; a scene is written only if it differs from the original; every saved scene gets one
row in the log (changed pixels per transition, category, note, UTC time) and a before/after overlay PNG for the audit.
"""
import csv
import datetime
from pathlib import Path

import numpy as np
from PIL import Image

CLASSES = {0: 'sea', 1: 'oil', 2: 'look_alike', 3: 'ship', 4: 'land'}
CATEGORIES = ['no_change', 'ship_added', 'class_fixed', 'slick_added', 'slick_removed', 'contour_fixed', 'land_fixed', 'ambiguous', 'other']
COLORS = {1: (0, 255, 255), 2: (255, 0, 0), 3: (255, 150, 0), 4: (0, 200, 0)}


def scene_name(scene):
    return scene.strip().removeprefix('train_')


def load_scene(root, name, out_dir):
    """Return (image uint8, original labels, current revised labels). The revised copy is resumed if it exists, otherwise it starts as the original."""
    image = np.asarray(Image.open(root / f'train/images/{name}.jpg').convert('L'))
    original = np.asarray(Image.open(root / f'train/labels_1D/{name}.png'))
    saved = Path(out_dir) / 'labels_1D_revised' / f'{name}.png'
    revised = np.asarray(Image.open(saved)) if saved.exists() else original.copy()
    if not (image.shape == original.shape == revised.shape):
        raise ValueError(f'{name}: shape mismatch {image.shape} {original.shape} {revised.shape}')
    return image, original, revised


def validate(image, original, revised):
    """Raise on anything that could corrupt the label geometry; return the list of warnings."""
    revised = np.asarray(revised)
    if revised.shape != original.shape or revised.shape != image.shape:
        raise ValueError(f'shape {revised.shape} differs from the image {image.shape}')
    if revised.dtype != np.uint8:
        raise ValueError(f'dtype {revised.dtype}, expected uint8')
    bad = set(np.unique(revised).tolist()) - set(CLASSES)
    if bad:
        raise ValueError(f'values outside 0..4: {sorted(bad)}')
    warnings = []
    changed = revised != original
    if changed.mean() > 0.30:
        warnings.append(f'{100 * changed.mean():.0f}% of the pixels changed')
    return warnings


def transitions(original, revised):
    """Counts of changed pixels per (from, to) class pair, e.g. {'sea->ship': 120}."""
    changed = revised != original
    out = {}
    for a, b in set(zip(original[changed].tolist(), revised[changed].tolist())):
        out[f'{CLASSES[a]}->{CLASSES[b]}'] = int(((original == a) & (revised == b)).sum())
    return out


def overlay(image, original, revised, path):
    """Side-by-side audit image: SAR with original contours | SAR with revised contours (changed pixels in white)."""
    from scipy import ndimage
    def draw(labels, mark=None):
        a = np.stack([image] * 3, -1).copy()
        for k, c in COLORS.items():
            m = labels == k
            a[m & ~ndimage.binary_erosion(m)] = c
        if mark is not None:
            a[mark & ~ndimage.binary_erosion(mark)] = (255, 255, 255)
        return a
    left, right = draw(original), draw(revised, revised != original)
    Image.fromarray(np.concatenate([left, np.full((left.shape[0], 6, 3), 255, np.uint8), right], 1)).save(path)


def save(root, out_dir, name, image, original, revised, category, note, log_path):
    root, out_dir = Path(root).resolve(), Path(out_dir).resolve()
    if out_dir == root or root in out_dir.parents:
        raise ValueError('refusing to write inside the original dataset')
    warnings = validate(image, original, revised)
    changed = int((revised != original).sum())
    target = out_dir / 'labels_1D_revised' / f'{name}.png'
    if changed:
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.asarray(revised, dtype=np.uint8)).save(target)
        (out_dir / 'overlays').mkdir(exist_ok=True)
        overlay(image, original, np.asarray(revised), out_dir / 'overlays' / f'{name}.png')
    elif target.exists():
        target.unlink()   # reverted to the original: no revised file for this scene
    new = not Path(log_path).exists()
    with open(log_path, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['utc', 'scene_id', 'category', 'changed_px', 'transitions', 'note', 'warnings'])
        w.writerow([datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'), f'train_{name}', category, changed,
                    ';'.join(f'{k}:{v}' for k, v in transitions(original, np.asarray(revised)).items()), note, '|'.join(warnings)])
    return changed, warnings
