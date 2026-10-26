"""CPU tests of the window weighting used by segmentation/inference/oof.py."""
import unittest

import numpy as np

from segmentation.inference.oof import window_weight


def reconstruct(kind, probs, origins, shape, size=256):
    """Same accumulation as infer_fold for per-window probability maps `probs` (5, size, size) at `origins`."""
    weight = window_weight(kind, size)
    total, counts, wsum = np.zeros((5, *shape)), np.zeros(shape), np.zeros(shape)
    for prob, (y, x) in zip(probs, origins):
        y0, y1, x0, x1 = max(y, 0), min(y + size, shape[0]), max(x, 0), min(x + size, shape[1])
        w = weight[y0 - y:y1 - y, x0 - x:x1 - x] if weight is not None else 1.0
        total[:, y0:y1, x0:x1] += prob[:, y0 - y:y1 - y, x0 - x:x1 - x] * w
        wsum[y0:y1, x0:x1] += w
        counts[y0:y1, x0:x1] += 1
    return total / (counts if weight is None else wsum), counts


class WindowWeightTests(unittest.TestCase):
    def test_uniform_is_none_and_cosine_positive_symmetric(self):
        self.assertIsNone(window_weight('uniform'))
        w = window_weight('cosine')
        self.assertEqual(w.shape, (256, 256))
        self.assertTrue((w > 0).all())
        self.assertTrue(np.allclose(w, w[::-1, ::-1]))
        self.assertGreater(w[128, 128], 100 * w[0, 0])

    def test_constant_probability_is_reproduced_on_the_centred_grid(self):
        shape, size = (650, 1250), 256   # origins as in the real index: centred, partly outside the scene
        origins = [(-59 + 128 * i, -15 + 128 * j) for i in range(5) for j in range(9)]
        prob = np.full((5, size, size), 0.2)
        for kind in ('uniform', 'cosine'):
            mean, counts = reconstruct(kind, [prob] * len(origins), origins, shape)
            self.assertTrue((counts > 0).all(), kind)
            self.assertTrue(np.allclose(mean, 0.2), kind)

    def test_taper_moves_a_window_disagreement_off_the_step(self):
        # two windows overlapping by half: A says class 0, B class 1. Uniform: the decision flips exactly where the set of
        # windows changes; cosine: the flip happens where the taper weights cross, not at a window border.
        size, shape = 64, (64, 96)
        a, b = np.zeros((5, size, size)), np.zeros((5, size, size))
        a[0], b[1] = 1.0, 1.0
        for kind in ('uniform', 'cosine'):
            mean, _ = reconstruct(kind, [a, b], [(0, 0), (0, 32)], shape, size)
            flip = np.flatnonzero(mean[:, 32].argmax(0)[1:] != mean[:, 32].argmax(0)[:-1])
            self.assertEqual(len(flip), 1)
            if kind == 'uniform':
                self.assertIn(flip[0], (31, 63))   # a window border (x = 32 or x = 64 after the shift of one pixel)


if __name__ == '__main__':
    unittest.main()
