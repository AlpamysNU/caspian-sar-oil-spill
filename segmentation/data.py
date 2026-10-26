"""Data helpers shared by every Stage-3 script: class order, mask decoding, the fold tables and the 256 px patch dataset.

Inputs are the native 256x256 SAR patches, the original RGB masks and the frozen grouped_folds_v1 manifest.
The official test split is never loaded here.
"""
import hashlib
import json
import random
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


PALETTE = ((0, 0, 0), (0, 255, 255), (255, 0, 0), None, (0, 153, 0))
CLASSES = ('sea', 'oil', 'look_alike', 'ship', 'land')



def atomic_save(value, path, tensor=False):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    if tensor:
        torch.save(value, temporary)
    else:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(temporary, path)


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


@lru_cache(maxsize=16)
def read_mask(path):
    with Image.open(path) as source:
        return decode_mask(np.asarray(source.convert('RGB')))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decode_mask(rgb):
    out = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    for index, color in enumerate(PALETTE):
        if color is not None:
            out[np.all(rgb == color, axis=2)] = index
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    ship = (r >= 60) & (r <= 170) & (g >= 30) & (g <= 100) & (b <= 15)
    out[ship] = 3
    if np.any(out == 255):
        raise ValueError(f'Unknown RGB mask pixels: {int(np.sum(out == 255))}')
    return out


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
    if len(frame) != 45090 or set(frame.patch_size) != {256} or set(frame.stride) != {128}:
        raise ValueError('Expected frozen 256/128 patch grid')
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
    y0, y1 = max(y, 0), min(y + 256, height)
    x0, x1 = max(x, 0), min(x + 256, width)
    if y0 >= y1 or x0 >= x1:
        raise ValueError('Patch has no valid pixels')
    target = np.full((256, 256), 255, dtype=np.uint8)
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
        'shape': [256, 256], 'dtype': 'uint8',
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
        if image.shape != (256, 256) or image.dtype != np.uint8:
            raise ValueError(f'Invalid SAR patch: {path}')
        if self.patch_masks is None:
            mask = read_mask(self.mask_paths[row['scene_id']])
            target = crop_patch_mask(mask, row)
        else:
            target = np.load(self.patch_masks[row['patch_id']], allow_pickle=False)
            if target.shape != (256, 256) or target.dtype != np.uint8:
                raise ValueError(f'Invalid cached mask: {row["patch_id"]}')
        if self.augment:
            if random.random() < 0.5:
                image, target = np.fliplr(image), np.fliplr(target)
            if random.random() < 0.5:
                image, target = np.flipud(image), np.flipud(target)
            k = random.randrange(4)
            image, target = np.rot90(image, k), np.rot90(target, k)
        return torch.from_numpy(np.ascontiguousarray(image)).float().unsqueeze(0) / 255.0, torch.from_numpy(np.ascontiguousarray(target)).long()
