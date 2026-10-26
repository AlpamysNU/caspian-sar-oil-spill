"""CPU tests of the ensemble / flip-TTA averaging used by segmentation/inference/ensemble.py."""
import unittest

import torch

from segmentation.inference.ensemble import TTA_VIEWS, member_probabilities, parse_member


class PixelModel(torch.nn.Module):
    """1x1 convolution: flip-equivariant, so flip TTA must reproduce the single view exactly."""
    def __init__(self, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.conv = torch.nn.Conv2d(3, 5, 1)

    def forward(self, x):
        return None, self.conv(x)


class EnsembleTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.images = torch.randn(2, 3, 256, 256)

    def test_flip_tta_equals_single_view_for_equivariant_model(self):
        model = PixelModel(1)
        single = member_probabilities([model], self.images, TTA_VIEWS['none'])
        flipped = member_probabilities([model], self.images, TTA_VIEWS['flip'])
        self.assertTrue(torch.allclose(single, flipped, atol=1e-5))

    def test_ensemble_is_mean_of_members(self):
        a, b = PixelModel(1), PixelModel(2)
        both = member_probabilities([a, b], self.images, TTA_VIEWS['none'])
        mean = (member_probabilities([a], self.images, TTA_VIEWS['none']) + member_probabilities([b], self.images, TTA_VIEWS['none'])) / 2
        self.assertTrue(torch.allclose(both, mean, atol=1e-6))
        self.assertTrue(torch.allclose(both.sum(1), torch.ones(2, 256, 256), atol=1e-5))

    def test_parse_member(self):
        self.assertEqual(parse_member('a/b:43')[1], 43)
        self.assertEqual(parse_member('a/b')[1], 42)


if __name__ == '__main__':
    unittest.main()
