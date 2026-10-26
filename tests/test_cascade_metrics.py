"""CPU tests of the scene metrics and patch stitching of the baseline / cascade inference."""
import unittest

import numpy as np

from segmentation.cascade.infer_b0_oof import add_patch, evaluate_scene, metrics, native_backend


class SceneMetricsTests(unittest.TestCase):
    def test_perfect_prediction(self):
        target = np.zeros((6, 8), np.uint8)
        target[1:3, 1:4] = 1
        report = evaluate_scene(target, target.copy(), np.ones(target.shape, np.uint16))
        self.assertEqual(report['classes']['oil']['iou'], 1.0)
        self.assertEqual(report['oil_components_total'], 1)
        self.assertFalse(report['oil_scene_missed'])
        self.assertEqual(report['uncovered_pixels'], 0)

    def test_missed_component_and_uncovered_pixels(self):
        target = np.zeros((6, 8), np.uint8)
        target[1:3, 1:3] = 1
        target[4:6, 5:8] = 1
        prediction = np.zeros_like(target)
        coverage = np.ones(target.shape, np.uint16)
        coverage[:, 4:] = 0   # the second component is not covered by any window
        report = evaluate_scene(target, prediction, coverage)
        self.assertEqual(report['oil_components_total'], 2)
        self.assertEqual(report['oil_components_evaluable'], 1)
        self.assertEqual(report['oil_components_missed'], 1)
        self.assertEqual(report['oil_components_uncovered'], 1)
        self.assertTrue(report['oil_scene_missed'])
        self.assertEqual(report['uncovered_pixels'], 24)

    def test_metrics_undefined_class_is_none(self):
        cm = np.zeros((5, 5), np.int64)
        cm[0, 0] = 10
        out = metrics(cm)
        self.assertEqual(out['classes']['sea']['iou'], 1.0)
        self.assertIsNone(out['classes']['oil']['iou'])
        self.assertEqual(out['miou'], 1.0)


class StitchingTests(unittest.TestCase):
    def test_window_is_clipped_to_the_scene(self):
        total, coverage = np.zeros((5, 6, 6), np.float32), np.zeros((6, 6), np.uint16)
        row = {'y_orig': -2, 'x_orig': -2, 'valid_pixels': 4 * 4, 'patch_id': 'p'}
        add_patch(total, coverage, np.ones((5, 6, 6), np.float32), row)
        self.assertEqual(int(coverage.sum()), 16)
        self.assertEqual(float(total[0].sum()), 16.0)

    def test_wrong_geometry_is_rejected(self):
        total, coverage = np.zeros((5, 6, 6), np.float32), np.zeros((6, 6), np.uint16)
        with self.assertRaises(ValueError):
            add_patch(total, coverage, np.ones((5, 6, 6), np.float32), {'y_orig': 0, 'x_orig': 0, 'valid_pixels': 35, 'patch_id': 'p'})


class BackendTests(unittest.TestCase):
    def test_both_native_backends_load(self):
        self.assertTrue(hasattr(native_backend(256), 'model_from_pretrained'))
        self.assertTrue(hasattr(native_backend(512), 'Patches'))
        with self.assertRaises(ValueError):
            native_backend(384)


if __name__ == '__main__':
    unittest.main()
