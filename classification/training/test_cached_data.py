"""CPU checks for optional native-patch RAM caching."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from grouped_data import GroupedPatchDataset, TARGET_COLUMNS, make_fold_loaders


class CachedDataTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        rows = []
        for i in range(4):
            name = f'{i}.npy'
            np.save(self.root / name, np.full((512, 512), i * 63, dtype=np.uint8))
            rows.append(dict(patch_path=name, **{k: (i + j) % 2 for j, k in enumerate(TARGET_COLUMNS)}))
        self.frame = pd.DataFrame(rows)

    def dataset(self, role='train', transform=None):
        return GroupedPatchDataset(self.frame, self.root, 512, role, transform)

    def test_identical_and_no_further_file_reads(self):
        dataset = self.dataset()
        expected = [dataset[i] for i in range(len(dataset))]
        dataset.preload()
        self.assertFalse(dataset.cache.flags.writeable)
        with patch('grouped_data.np.load', side_effect=AssertionError('Unexpected disk read')):
            for i, (x, y) in enumerate(expected):
                actual_x, actual_y = dataset[i]
                self.assertTrue(torch.equal(x, actual_x))
                self.assertTrue(torch.equal(y, actual_y))

    def test_transform_does_not_mutate_cache_or_targets(self):
        dataset = self.dataset(transform=lambda x: x.add_(torch.rand(())))
        torch.manual_seed(42)
        expected, _ = dataset[1]
        dataset.preload()
        original = dataset.cache.copy()
        torch.manual_seed(42)
        actual, labels = dataset[1]
        labels.zero_()
        self.assertTrue(torch.equal(expected, actual))
        np.testing.assert_array_equal(original, dataset.cache)
        self.assertGreater(dataset.targets[1].sum().item(), 0)

    def test_invalid_native_patch_rejected(self):
        for array in (np.zeros((512, 512), dtype=np.float32), np.zeros((256, 256), dtype=np.uint8)):
            np.save(self.root / '0.npy', array)
            with self.assertRaisesRegex(ValueError, 'Invalid native uint8 patch'):
                self.dataset().preload()

    def test_worker_batches_match_disk_loader(self):
        def datasets(*args):
            return {role: self.dataset(role) for role in ('train', 'validation')}
        with patch('grouped_data.load_fold_datasets', side_effect=datasets):
            disk = make_fold_loaders(self.root, self.root, 1, batch_size=2, num_workers=2)
            cached = make_fold_loaders(self.root, self.root, 1, batch_size=2, num_workers=2, cache_ram=True)
        for role in disk:
            for (x, y), (cx, cy) in zip(disk[role], cached[role]):
                self.assertTrue(torch.equal(x, cx))
                self.assertTrue(torch.equal(y, cy))


if __name__ == '__main__':
    unittest.main()
