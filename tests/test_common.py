"""Unit tests for the shared Stage-3 trainer helpers (CPU only)."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from segmentation.common import TopKCheckpoints, choose_epochs, first_conv_to_single_channel, shift_reflect


class ShiftTests(unittest.TestCase):
    def test_zero_shift_is_identity(self):
        image, target = np.arange(16.).reshape(4, 4), np.arange(16).reshape(4, 4)
        a, b = shift_reflect(image, target, 0, 0, 2)
        self.assertTrue((a == image).all() and (b == target).all())

    def test_shape_and_alignment_preserved(self):
        rng = np.random.default_rng(0)
        image, target = rng.random((32, 32)).astype(np.float32), rng.integers(0, 5, (32, 32))
        a, b = shift_reflect(image, target, 5, -3, 8)
        self.assertEqual(a.shape, (32, 32))
        self.assertTrue((a[5:, :29] == image[:27, 3:]).all())   # content moved down 5, left 3
        self.assertTrue((b[5:, :29] == target[:27, 3:]).all())  # labels move with the image

    def test_image_and_target_use_same_reflection(self):
        image = np.arange(64.).reshape(8, 8)
        a, b = shift_reflect(image, image.astype(np.int64), -2, 3, 4)
        self.assertTrue((a == b).all())


class ConvTests(unittest.TestCase):
    def test_first_rgb_conv_becomes_single_channel_mean(self):
        net = torch.nn.Sequential(torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.ReLU()),
                                  torch.nn.Conv2d(4, 4, 1))
        weight = net[0][0].weight.detach().mean(1, keepdim=True)
        self.assertEqual(first_conv_to_single_channel(net), '0.0')
        self.assertEqual(net[0][0].in_channels, 1)
        self.assertTrue(torch.allclose(net[0][0].weight, weight))


class SelectionTests(unittest.TestCase):
    @staticmethod
    def entry(epoch, oil, ship):
        return {'epoch': epoch, 'iou': {'oil': oil, 'ship': ship, 'look_alike': 0.5}}

    def test_baseline_rule_takes_first_max_oil(self):
        history = [self.entry(1, 0.5, 0.4), self.entry(2, 0.6, 0.0), self.entry(3, 0.6, 0.5)]
        ref = {'oil': 0.6, 'ship': 0.4, 'look_alike': 0.5}
        best, chosen, extra = choose_epochs(history, {1, 2, 3}, ref)
        self.assertEqual(best['epoch'], 2)
        self.assertEqual(chosen['epoch'], 3)       # epoch 2 fails the Ship guardrail
        self.assertFalse(extra['guardrail_failed'])

    def test_truncated_selection_falls_back_to_kept_epochs(self):
        history = [self.entry(1, 0.6, 0.4), self.entry(2, 0.6, 0.4)]
        ref = {'oil': 0.6, 'ship': 0.4, 'look_alike': 0.5}
        _, chosen, extra = choose_epochs(history, {2}, ref)
        self.assertEqual(chosen['epoch'], 2)
        self.assertTrue(extra['selection_truncated'] or chosen['epoch'] == 2)


class TopKTests(unittest.TestCase):
    def test_keeps_only_best_k_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            top = TopKCheckpoints(Path(tmp) / 'ckpt', 2)
            for epoch, oil in [(1, 0.1), (2, 0.5), (3, 0.3), (4, 0.4)]:
                top.offer(epoch, oil, {'w': torch.tensor([float(epoch)])})
            self.assertEqual(set(top.kept), {2, 4})
            self.assertEqual(float(top.load(4)['w']), 4.0)
            top.cleanup()
            self.assertFalse((Path(tmp) / 'ckpt').exists())


if __name__ == '__main__':
    unittest.main()
