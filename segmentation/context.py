"""Scene context for the HRDA-inspired two-scale arm (2026-10-15): reflect crops of the original scene and the paired dataset.

A window of the frozen 256/128 grid (row `y_orig`, `x_orig`, possibly negative) gets a context view of the SAME scene: a native
512x512 crop with the same centre, reflect-padded beyond the scene border exactly like the window grid itself, and reduced to
256x256 by 2x2 area averaging. The central 256x256 of the native crop is the stored window (tested), so the central 128x128 of the
reduced context covers exactly the window. Nothing is read from any other scene.
"""
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from segmentation.common import AugPatches, shift_reflect
from segmentation.data import Patches, sha256

CONTEXT = 512


def reflect_index(i, n):
    """Index of position `i` after infinite reflect padding of an axis of length n (period 2(n-1), edge not repeated):
    equals numpy.pad(mode='reflect') for any pad width."""
    if n == 1:
        return np.zeros_like(i)
    period = 2 * (n - 1)
    m = np.mod(i, period)
    return np.where(m <= n - 1, m, period - m)


def reflect_crop(scene, top, left, size):
    """size x size crop whose top-left corner is (top, left) in scene coordinates; outside the scene it is reflect-padded."""
    rows = reflect_index(np.arange(top, top + size), scene.shape[0])
    cols = reflect_index(np.arange(left, left + size), scene.shape[1])
    return scene[np.ix_(rows, cols)]


def reduce2(array):
    """2x2 area average (exact for even sizes)."""
    h, w = array.shape
    return array.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))


@lru_cache(maxsize=32)
def read_scene(path):
    return np.asarray(Image.open(path).convert('L'), dtype=np.uint8)   # same conversion as the patch builder (core.py)


def scene_paths(scenes_root, data_root, scene_ids):
    """scene_id (train_img_NNNN) -> jpg path, each file checked against `image_sha256` of the frozen scene manifest."""
    manifest = pd.read_csv(Path(data_root) / 'scene_manifest.csv').set_index('scene_id')
    paths = {}
    for scene in sorted(set(scene_ids)):
        path = Path(scenes_root) / (scene.removeprefix('train_') + '.jpg')
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256(path) != manifest.loc[scene, 'image_sha256']:
            raise ValueError(f'Scene image hash mismatch: {scene}')
        paths[scene] = str(path)
    return paths


class ContextPatches(AugPatches):
    """AugPatches that returns a 2-channel image: [detail window, context reduced to 256]; target = the window's labels.

    Augmentation draws (flips, rot90, shift) are applied to detail, target and context together. The shift moves the window as in
    the control arms (reflect-padded detail and target); the context is cut at the shifted centre from the real scene."""

    def __init__(self, frame, data_root, mask_paths, patch_masks=None, train=False, shift_max=0, shift_prob=0.5, scenes_root=None):
        super().__init__(frame, data_root, mask_paths, patch_masks, train, shift_max, shift_prob)
        if scenes_root is None:
            raise ValueError('scenes_root is required')
        self.scene_files = scene_paths(scenes_root, data_root, frame.scene_id)
        self.margin = shift_max if train else 0

    def context_source(self, index):
        row = self.rows[index]
        size = CONTEXT + 2 * self.margin
        offset = (size - 256) // 2
        scene = read_scene(self.scene_files[row['scene_id']])
        return reflect_crop(scene, int(row['y_orig']) - offset, int(row['x_orig']) - offset, size).astype(np.float32) / 255.0

    def __getitem__(self, index):
        detail, target = Patches.__getitem__(self, index)
        detail, target = detail[0].numpy(), target.numpy()
        big = self.context_source(index)
        if self.train:
            if random.random() < 0.5:
                detail, target, big = np.fliplr(detail), np.fliplr(target), np.fliplr(big)
            if random.random() < 0.5:
                detail, target, big = np.flipud(detail), np.flipud(target), np.flipud(big)
            k = random.randrange(4)
            detail, target, big = np.rot90(detail, k), np.rot90(target, k), np.rot90(big, k)
            dy = dx = 0
            if self.shift_max and random.random() < self.shift_prob:
                dy, dx = random.randint(-self.shift_max, self.shift_max), random.randint(-self.shift_max, self.shift_max)
                detail, target = shift_reflect(detail, target, dy, dx, self.shift_max)
            big = big[self.margin - dy:self.margin - dy + CONTEXT, self.margin - dx:self.margin - dx + CONTEXT]
        image = np.stack([np.ascontiguousarray(detail), reduce2(np.ascontiguousarray(big))]).astype(np.float32)
        return torch.from_numpy(image), torch.from_numpy(np.ascontiguousarray(target)).long()
