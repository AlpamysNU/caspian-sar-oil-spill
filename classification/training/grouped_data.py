"""Official-train loaders using a frozen scene-level fold manifest.

This module is independent of the historical random-patch-split API in data.py.
"""
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

CLASS_NAMES = ('oil', 'look_alike', 'land', 'sea', 'ship')
TARGET_COLUMNS = [f'label_{name}' for name in CLASS_NAMES]


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class GroupedPatchDataset(Dataset):
    """Read native uint8 SAR arrays and explicit float multi-label targets.

    An optional tensor transform is permitted only for training. Its complete
    policy must be registered by the caller. No augmentation is enabled silently.
    Row metadata remain accessible through ``df`` in validation iteration order.
    """

    def __init__(self, frame, root, patch_size, role, transform=None):
        if role not in ('train', 'validation'):
            raise ValueError('Role must be train or validation')
        if role != 'train' and transform is not None:
            raise ValueError('Validation transforms are forbidden')
        self.df = frame.reset_index(drop=True).copy()
        self.root = Path(root).resolve()
        self.patch_size = patch_size
        self.transform = transform
        self.role = role
        labels = self.df[TARGET_COLUMNS].to_numpy(dtype=np.float32)
        if not np.isin(labels, (0, 1)).all():
            raise ValueError('Targets must be finite binary values')
        self.targets = torch.from_numpy(labels)
        self.paths = []
        self.cache = None
        for relative in self.df.patch_path:
            path = (self.root / relative).resolve()
            if Path(relative).is_absolute() or not path.is_relative_to(self.root):
                raise ValueError('Patch path must remain inside its dataset directory')
            if not path.is_file():
                raise FileNotFoundError(path)
            self.paths.append(path)

    def __len__(self):
        return len(self.df)

    def preload(self):
        """Read native uint8 patches once; conversions/augmentation use copies."""
        cache = np.empty((len(self), self.patch_size, self.patch_size), dtype=np.uint8)
        for i, path in enumerate(self.paths):
            array = np.load(path, allow_pickle=False)
            if array.dtype != np.uint8 or array.shape != cache.shape[1:]:
                raise ValueError(f'Invalid native uint8 patch: {path}')
            cache[i] = array
        cache.flags.writeable = False
        self.cache = cache

    def __getitem__(self, index):
        array = (self.cache[index] if self.cache is not None
                 else np.load(self.paths[index], allow_pickle=False))
        if array.dtype != np.uint8 or array.shape != (self.patch_size, self.patch_size):
            raise ValueError(f'Invalid native uint8 patch: {self.paths[index]}')
        x = torch.from_numpy(array.astype(np.float32)).unsqueeze(0) / 255.0
        if self.transform is not None:
            x = self.transform(x)
        if x.shape != (1, self.patch_size, self.patch_size) or x.dtype != torch.float32 or not torch.isfinite(x).all():
            raise ValueError('Transform returned invalid tensor')
        return x.contiguous(), self.targets[index].clone()

    def positive_weights(self):
        if self.role != 'train':
            raise ValueError('Class weights must be computed from training only')
        positives = self.targets.sum(dim=0)
        if torch.any((positives == 0) | (positives == len(self))):
            raise ValueError('Training class lacks positives or negatives')
        return (len(self) - positives) / positives


def load_fold_datasets(dataset_root, folds_dir, fold, train_transform=None):
    root, folds_dir = Path(dataset_root), Path(folds_dir)
    audit = json.loads((folds_dir / 'audit.json').read_text())
    manifest_path = folds_dir / 'scene_folds.csv'
    if fingerprint(manifest_path) != audit['manifest_sha256']:
        raise ValueError('Fold manifest checksum mismatch')
    if isinstance(fold, bool) or not isinstance(fold, int) or fold not in range(1, audit['n_folds'] + 1):
        raise ValueError('Invalid validation fold')
    sources = list(root.glob('index_*.parquet'))
    if len(sources) != 1:
        raise ValueError('Expected one patch index')
    # Match source fingerprints independently of the machine-specific root path.
    for path in (sources[0], root / 'scene_manifest.csv', root / 'verification.json'):
        candidates = [digest for name, digest in audit['input_sha256'].items()
                      if Path(name).parts[-2:] == path.parts[-2:]]
        if len(candidates) != 1 or fingerprint(path) != candidates[0]:
            raise ValueError(f'Dataset differs from frozen fold inputs: {path}')
    frame = pd.read_parquet(sources[0])
    manifest = pd.read_csv(manifest_path)
    if manifest.scene_id.duplicated().any() or frame.patch_id.duplicated().any():
        raise ValueError('Duplicate identifiers')
    if set(frame.split) != {'train'} or set(manifest.split) != {'train'}:
        raise ValueError('Official test is forbidden in development loaders')
    if set(frame.scene_id) != set(manifest.scene_id):
        raise ValueError('Incomplete scene coverage')
    for key in ('scene_id', 'group_id', 'image_sha256'):
        if set(manifest.loc[manifest.fold == fold, key]) & set(manifest.loc[manifest.fold != fold, key]):
            raise ValueError(f'Fold leakage: {key}')
    frame = frame.merge(manifest[['scene_id', 'fold']], on='scene_id', validate='many_to_one')
    sizes = frame.patch_size.unique()
    if len(sizes) != 1 or sizes[0] not in (256, 384, 512):
        raise ValueError('Unsupported or mixed patch sizes')
    datasets = {}
    for role, mask in [('train', frame.fold != fold), ('validation', frame.fold == fold)]:
        if not mask.any():
            raise ValueError('Empty partition')
        datasets[role] = GroupedPatchDataset(frame[mask], root, int(sizes[0]), role,
                                           train_transform if role == 'train' else None)
    return datasets


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def make_fold_loaders(dataset_root, folds_dir, fold, batch_size=32,
                      num_workers=0, seed=42, train_transform=None, cache_ram=False):
    """Return train/validation loaders; caller must seed main-process RNGs.

    Persist loader.generator state in trainer checkpoints for resumed shuffling.
    Worker RNGs are seeded on every iterator creation; persistent workers are off.
    """
    datasets = load_fold_datasets(dataset_root, folds_dir, fold, train_transform)
    if cache_ram:
        import time
        started = time.perf_counter()
        for dataset in datasets.values():
            dataset.preload()
        print(json.dumps(dict(cache='ram_uint8', patches=sum(len(d) for d in datasets.values()),
                              bytes=sum(d.cache.nbytes for d in datasets.values()),
                              preload_seconds=time.perf_counter()-started)), flush=True)
    return {role: DataLoader(dataset, batch_size=batch_size, shuffle=role == 'train',
                            num_workers=num_workers, worker_init_fn=seed_worker,
                            **({'multiprocessing_context': 'fork'} if cache_ram and num_workers else {}),
                            generator=torch.Generator().manual_seed(seed), drop_last=False)
            for role, dataset in datasets.items()}
