"""CPU tests for the two-scale (HRDA-inspired) arm: context geometry on synthetic and real scenes, augmentation consistency, fusion alignment."""
import random
import unittest

import numpy as np
import pandas as pd
import torch

import segmentation.context as hc
from segmentation.models import TwoScaleStep, central_upsample

from segmentation import paths

DATA = paths.STAGE2_RESULTS / 'training_patches/ps256_overlap50'
SCENES = paths.DATASET / 'train/images'


class ReflectTests(unittest.TestCase):
    def test_matches_numpy_pad_including_pad_larger_than_scene(self):
        rng = np.random.default_rng(0)
        for shape in [(40, 70), (5, 9), (2, 3)]:
            scene = rng.integers(0, 255, shape, dtype=np.uint8)
            p = 37
            padded = np.pad(scene, p, mode='reflect')
            crop = hc.reflect_crop(scene, -p, -p, max(shape) + 2 * p)
            h, w = padded.shape
            self.assertTrue((crop[:h, :w] == padded).all(), shape)

    def test_reduce2_is_area_mean(self):
        a = np.arange(16, dtype=np.float32).reshape(4, 4)
        self.assertEqual(hc.reduce2(a).tolist(), [[2.5, 4.5], [10.5, 12.5]])


class RealDataTests(unittest.TestCase):
    @unittest.skipUnless((DATA / 'index_ps256_overlap50.parquet').is_file() and SCENES.is_dir(), 'local patch bundle or scenes missing')
    def test_central_crop_of_context_is_the_stored_window(self):
        frame = pd.read_parquet(DATA / 'index_ps256_overlap50.parquet')
        rows = pd.concat([frame[frame.is_border].sample(40, random_state=0), frame[~frame.is_border].sample(10, random_state=0)])
        for row in rows.itertuples():
            scene = hc.read_scene(str(SCENES / (row.scene_id.removeprefix('train_') + '.jpg')))
            big = hc.reflect_crop(scene, row.y_orig - 128, row.x_orig - 128, 512)
            stored = np.load(DATA / row.patch_path)
            self.assertTrue((big[128:384, 128:384] == stored).all(), row.patch_id)


class FakePatches(hc.ContextPatches):
    """ContextPatches on a synthetic scene (skips file and mask I/O)."""

    def __init__(self, scene, train, shift_max=0):
        self.rows = [{'scene_id': 's', 'y_orig': 100, 'x_orig': -20}]
        self.train, self.shift_max, self.shift_prob, self.margin = train, shift_max, 1.0, shift_max if train else 0
        self.scene = scene

    def context_source(self, index):
        row = self.rows[index]
        size = hc.CONTEXT + 2 * self.margin
        off = (size - 256) // 2
        return hc.reflect_crop(self.scene, row['y_orig'] - off, row['x_orig'] - off, size).astype(np.float32) / 255.0


def fake_patch_getitem(self, index):
    row = self.rows[index]
    win = hc.reflect_crop(self.scene, row['y_orig'], row['x_orig'], 256).astype(np.float32) / 255.0
    return torch.from_numpy(win).unsqueeze(0), torch.from_numpy((win * 4).astype(np.int64))   # target is a function of the pixel


class AugmentationTests(unittest.TestCase):
    def setUp(self):
        self.scene = np.random.default_rng(1).integers(0, 255, (700, 900), dtype=np.uint8)
        self._orig = hc.Patches.__getitem__
        hc.Patches.__getitem__ = fake_patch_getitem

    def tearDown(self):
        hc.Patches.__getitem__ = self._orig

    def test_eval_centre_of_context_is_the_window(self):
        image, _ = FakePatches(self.scene, False)[0]
        self.assertEqual(tuple(image.shape), (2, 256, 256))
        # exact check: reduce2 of the window equals the central 128x128 of the context
        window = image[0].numpy()
        self.assertTrue(np.allclose(image[1, 64:192, 64:192].numpy(), hc.reduce2(window), atol=1e-6))

    def test_train_augmentation_keeps_detail_target_and_context_aligned(self):
        ds = FakePatches(self.scene, True, shift_max=64)
        for seed in range(30):
            random.seed(seed)
            image, target = ds[0]
            self.assertEqual(tuple(image.shape), (2, 256, 256))
            # target = floor(4 * pixel) of the detail: reflect shift and flips must move image and target together
            self.assertTrue(((image[0] * 4).long() == target).all() or np.abs(((image[0] * 4) - target.float()).numpy()).max() <= 1)

    def test_unshifted_augmented_context_centre_matches_detail(self):
        ds = FakePatches(self.scene, True, shift_max=64)
        ds.shift_prob = 0.0
        for seed in range(20):
            random.seed(seed)
            image, _ = ds[0]
            self.assertTrue(np.allclose(image[1, 64:192, 64:192].numpy(), hc.reduce2(image[0].numpy()), atol=1e-6))

    def test_shifted_context_is_cut_at_the_shifted_centre(self):
        class Fixed:   # no flips, k = 0, shift (dy, dx) = (+20, -30)
            draws = iter([0.9, 0.9, 0.0])   # flip, flip (>= 0.5: off), shift (< prob: on)
            ints = iter([20, -30])

            @classmethod
            def random(cls):
                return next(cls.draws)

            @staticmethod
            def randrange(n):
                return 0

            @classmethod
            def randint(cls, a, b):
                return next(cls.ints)

        ds = FakePatches(self.scene, True, shift_max=64)
        real = hc.random
        hc.random = Fixed
        try:
            image, _ = ds[0]
        finally:
            hc.random = real
        row = ds.rows[0]
        # the window content moved by (+dy, +dx), i.e. the window now looks at scene position (y - dy, x - dx); the context is centred there
        expected = hc.reduce2(hc.reflect_crop(self.scene, row['y_orig'] - 20 - 128, row['x_orig'] + 30 - 128, 512).astype(np.float32) / 255.0)
        self.assertTrue(np.allclose(image[1].numpy(), expected, atol=1e-6))
        # and the shifted detail window agrees with it in the interior where no reflected margin is involved
        self.assertTrue(np.allclose(image[0, 100:156, 100:156].numpy(), hc.reflect_crop(self.scene, row['y_orig'] - 20 + 100, row['x_orig'] + 30 + 100, 56).astype(np.float32) / 255.0, atol=1e-6))


class StubOut:
    def __init__(self, logits):
        self.logits = logits


class StubBackbone(torch.nn.Module):
    """Stands in for SegFormer: 1/4-resolution logits, 5 classes; no transformers dependency."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 5, 4, stride=4)

    def forward(self, pixel_values):
        return StubOut(self.conv(pixel_values))


class FusionTests(unittest.TestCase):
    def test_central_upsample_alignment(self):
        # a ramp over the 64x64 logit map: output pixel r of the 256 window must sample logit coordinate 16 + (r + 0.5) / 8 - 0.5
        ramp = torch.arange(64.).view(1, 1, 64, 1).expand(1, 1, 64, 64).contiguous()
        out = central_upsample(ramp, (256, 256))[0, 0, :, 0]
        expected = 16 + (torch.arange(256) + 0.5) / 8 - 0.5
        self.assertTrue(torch.allclose(out[8:-8], expected[8:-8], atol=1e-4))

    def test_forward_shapes_loss_and_fusion_is_mean(self):
        torch.manual_seed(0)
        model = TwoScaleStep(StubBackbone())
        image = torch.rand(2, 2, 256, 256)
        target = torch.randint(0, 5, (2, 256, 256)); target[:, :10] = 255
        loss, fused = model(image, target)
        self.assertEqual(tuple(fused.shape), (2, 5, 256, 256))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))
        none_loss, fused2 = model(image)
        self.assertIsNone(none_loss)
        self.assertTrue(torch.allclose(fused, fused2))

    def test_context_channel_changes_the_prediction(self):
        model = TwoScaleStep(StubBackbone()).eval()
        image = torch.rand(1, 2, 256, 256)
        other = image.clone(); other[:, 1, 64:192, 64:192] += 0.5
        self.assertFalse(torch.allclose(model(image)[1], model(other)[1]))
        outside = image.clone(); outside[:, 1, :32, :32] += 0.5   # context outside the window footprint must not change the output
        self.assertTrue(torch.allclose(model(image)[1], model(outside)[1]))


if __name__ == '__main__':
    unittest.main()


class ScaleAttentionTests(unittest.TestCase):
    def test_zero_init_equals_the_fixed_fusion_of_variant_c(self):
        from segmentation.models import ScaleAttentionStep, hrda_d_groups as param_groups
        torch.manual_seed(0)
        backbone = StubBackbone()
        fixed, learned = TwoScaleStep(backbone).eval(), ScaleAttentionStep(backbone).eval()
        image = torch.rand(2, 2, 256, 256)
        self.assertTrue(torch.allclose(fixed(image)[1], learned(image)[1], atol=1e-6))
        groups = param_groups(learned, 6e-5)
        self.assertTrue(np.allclose([g['lr'] for g in groups], [6e-5, 6e-4]))
        self.assertEqual(sum(len(g['params']) for g in groups), len(list(learned.parameters())))

    def test_attention_gets_gradient_and_can_move_the_weight(self):
        from segmentation.models import ScaleAttentionStep
        torch.manual_seed(0)
        model = ScaleAttentionStep(StubBackbone())
        image = torch.rand(2, 2, 256, 256)
        target = torch.randint(0, 5, (2, 256, 256)); target[:, :7] = 255
        loss, fused = model(image, target)
        loss.backward()
        self.assertEqual(tuple(fused.shape), (2, 5, 256, 256))
        self.assertTrue(model.attention[2].weight.grad.abs().sum() > 0)
        with torch.no_grad():
            model.attention[2].bias.fill_(20.0)   # a -> 1: only the context branch remains
        model.eval()
        _, only_context = model(image)
        self.assertTrue(torch.allclose(only_context, central_upsample(model.backbone(pixel_values=image[:, 1:]).logits, (256, 256)), atol=1e-4))
