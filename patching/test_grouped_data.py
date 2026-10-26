"""Regression checks for fold grouping and loader target/input semantics."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


HERE = Path(__file__).resolve().parent
data = module('grouped_data', HERE.parent / 'classification/training/grouped_data.py')
folds = module('build_grouped_folds', HERE / 'build_grouped_folds.py')


class Contracts(unittest.TestCase):
    def test_group_integrity_and_reproducibility(self):
        groups = np.repeat(np.arange(20).astype(str), 2)
        features = np.ones((40, 3))
        a = folds.assign_folds(features, groups)
        np.testing.assert_array_equal(a, folds.assign_folds(features, groups))
        self.assertEqual(set(a), {1, 2, 3, 4, 5})
        for group in set(groups):
            self.assertEqual(len(set(a[groups == group])), 1)

    def test_native_pixels_and_named_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for size in (256, 384, 512):
                pixels = np.arange(size * size, dtype=np.uint8).reshape(size, size)
                np.save(root / 'patch.npy', pixels)
                row = {'patch_path': 'patch.npy', 'label_ship': 1, 'label_sea': 0,
                       'label_land': 1, 'label_look_alike': 0, 'label_oil': 1}
                ds = data.GroupedPatchDataset(pd.DataFrame([row]), root, size, 'validation')
                x, y = ds[0]
                torch.testing.assert_close(x[0], torch.from_numpy(pixels.astype(np.float32)) / 255)
                self.assertEqual(y.tolist(), [1, 0, 1, 0, 1])
                self.assertEqual(y.dtype, torch.float32)
                with self.assertRaises(ValueError):
                    ds.positive_weights()
                with self.assertRaises(ValueError):
                    data.GroupedPatchDataset(pd.DataFrame([row]), root, size, 'validation', lambda x: x)
            np.save(root / 'patch.npy', np.zeros((2, 2), dtype=np.uint8))
            with self.assertRaises(ValueError):
                ds[0]


if __name__ == '__main__':
    unittest.main()
