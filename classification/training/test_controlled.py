"""CPU regression tests for operational metrics and complete epoch resume."""
from dataclasses import replace
from pathlib import Path
import random
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.trainer import ControlledConfig, Trainer, seed_all
from training.operational_metrics import select_threshold, report


class ToyData(Dataset):
    def __init__(self, augment=False):
        self.augment = augment
        self.targets = torch.tensor([[i % 2, (i+1) % 2, i % 2, 1, 0] for i in range(11)], dtype=torch.float32)
        self.df = pd.DataFrame(dict(patch_id=[f'p{i}' for i in range(11)], scene_id=[f's{i}' for i in range(11)]))

    def __len__(self):
        return 11

    def __getitem__(self, i):
        x = torch.full((1, 8, 8), (i+1)/12)
        if self.augment:
            x = x + (torch.rand(()) + random.random() + np.random.rand())*.01
        return x, self.targets[i]

    def positive_weights(self):
        return torch.ones(5)


def loaders(batch_size=3, augment=True):
    return {role: DataLoader(ToyData(augment and role == 'train'), batch_size=batch_size,
                            shuffle=role == 'train', generator=torch.Generator().manual_seed(42))
            for role in ('train', 'validation')}


class TinyModel(nn.Module):
    def __init__(self, dropout=True):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 3, 1)
        self.body = nn.Sequential(nn.Conv2d(3, 3, 1), nn.ReLU())
        self.fc = nn.Sequential(nn.Dropout(.2 if dropout else 0), nn.Linear(3, 5))

    def forward(self, x):
        return self.fc(self.body(self.conv1(x)).mean((2, 3)))


class ControlledTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_ap_and_tied_thresholds(self):
        selected = select_threshold([1, 0, 1, 0], [.9, .8, .7, .6])
        self.assertAlmostEqual(selected['ap'], (1 + 2/3)/2)
        self.assertEqual(selected['threshold'], .7)
        tied = select_threshold([1, 0, 1, 0], [.5]*4)
        self.assertEqual(tied['recall'], 1.)
        self.assertEqual(tied['ap'], .5)
        self.assertAlmostEqual(tied['f2'], 5*.5/(4*.5+1))
        for y, p in [([0, 0], [.1, .2]), ([1, 0], [float('nan'), .2])]:
            with self.assertRaises(ValueError):
                select_threshold(y, p)

    def test_threshold_matches_bruteforce(self):
        rng = np.random.default_rng(42)
        for _ in range(30):
            y = rng.integers(0, 2, 100)
            p = rng.integers(0, 20, 100)/20
            candidates = []
            for threshold in np.unique(p):
                pred = p >= threshold
                tp = (pred & (y == 1)).sum()
                recall, precision = tp/y.sum(), tp/pred.sum()
                if recall >= .9:
                    candidates.append((5*tp/(4*y.sum()+pred.sum()), precision, recall, threshold))
            expected = max(candidates)
            actual = select_threshold(y, p)
            self.assertEqual(actual['threshold'], expected[3])

    def test_resume_identical_across_unfreeze(self):
        with tempfile.TemporaryDirectory() as directory:
            config = ControlledConfig(model='test', device='cpu', smoke=True, epochs=3,
                                      adaptation_epochs=1, batch_size=3, accumulation_steps=2,
                                      num_workers=0, output_root=str(Path(directory)/'full'))
            seed_all(9)
            full = Trainer(config, model=TinyModel(), loaders=loaders())
            full.fit()
            paused_config = replace(config, output_root=str(Path(directory)/'resume'))
            seed_all(9)
            paused = Trainer(paused_config, model=TinyModel(), loaders=loaders())
            paused.fit(stop_after=1)
            restored = Trainer(paused_config, resume='auto', model=TinyModel(), loaders=loaders())
            restored.fit()
            for name, value in full.model.state_dict().items():
                torch.testing.assert_close(value, restored.model.state_dict()[name], rtol=0, atol=0)
            self.assertEqual(full.best_key, restored.best_key)
            self.assertEqual(full.steps, restored.steps)
            self.assertEqual(full.scheduler, restored.scheduler)
            pd.testing.assert_frame_equal(pd.read_parquet(full.output/'val_predictions.parquet'),
                                          pd.read_parquet(restored.output/'val_predictions.parquet'))
            for expected in ('last.pt', 'best_operational.pt', 'best_pr_auc.pt', 'history.csv',
                             'oil_false_negatives.parquet', 'oil_false_positives.parquet', 'metrics.json'):
                self.assertTrue((restored.output/expected).exists())
            with self.assertRaisesRegex(ValueError, 'contract mismatch'):
                Trainer(replace(paused_config, head_lr=2e-4), resume=str(restored.output/'last.pt'),
                        model=TinyModel(), loaders=loaders())

    def test_accumulation_matches_large_batch_with_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            models = []
            for batch, accumulation in [(6, 1), (3, 2)]:
                seed_all(9)
                c = ControlledConfig(model='test', device='cpu', smoke=True, epochs=3, adaptation_epochs=1,
                                     batch_size=batch, accumulation_steps=accumulation, num_workers=0,
                                     output_root=str(Path(directory)/str(batch)))
                trainer = Trainer(c, model=TinyModel(False), loaders=loaders(batch, False))
                trainer.set_phase(2)
                trainer.train_epoch()
                self.assertEqual(trainer.steps, 2)
                models.append(trainer.model.state_dict())
            for key in models[0]:
                torch.testing.assert_close(models[0][key], models[1][key], atol=1e-7, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
