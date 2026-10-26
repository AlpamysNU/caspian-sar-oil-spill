"""Regression tests for the Stage 2 geometry audit."""
from __future__ import annotations

import unittest
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage2_geometry_audit import audit_scene, decode_m4d_mask, iter_original_windows


class Stage2GeometryAuditTests(unittest.TestCase):
    def test_empty_scene_has_no_selected_coverage(self):
        result = audit_scene(np.zeros((650,1250),dtype=np.uint8),512,256)
        self.assertEqual(result['selected_unique_scene_area_px'],0)
        self.assertEqual(result['oil_positive_patches'],0)

    def test_small_scene_padding_excludes_artificial_pixels(self):
        result = audit_scene(np.ones((7,11),dtype=np.uint8),256,128)
        self.assertEqual(result['patches_total'],1)
        self.assertEqual(result['all_valid_area_with_repetition_px'],77)
        self.assertEqual(result['oil_pixel_coverage_unique_px'],77)

    def test_extractor_matches_audit_for_small_and_normal_scenes(self):
        import tempfile
        sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
        from patching.core import PatchExtractor
        from patching.config import PatchConfig
        for shape in ((7,11),(650,1250)):
            labels = np.zeros((*shape,5),dtype=np.uint8)
            labels[...,0] = 1
            labels[-1,-1,0] = 0
            labels[-1,-1,1] = 1
            for size,stride in ((256,128),(384,192),(512,256),(512,128)):
                with tempfile.TemporaryDirectory() as name:
                    root=Path(name)
                    np.save(root/'image.npy',np.zeros(shape,dtype=np.uint8))
                    np.save(root/'labels.npy',labels)
                    extractor=PatchExtractor(PatchConfig(size,stride,1-stride/size),root/'out')
                    rows=extractor.extract_patches_from_scene(root/'image.npy',root/'labels.npy','scene',show_progress=False)
                    result=audit_scene(labels[...,1],size,stride)
                    self.assertEqual(sum(r.label_oil for r in rows),result['oil_positive_patches'])
                    self.assertEqual(sum(r.pixels_oil for r in rows),result['oil_pixel_coverage_repeated_px'])

    def test_padding_matches_512_grid(self) -> None:
        padding, windows = iter_original_windows(650, 1250, 512, 256)
        self.assertEqual(padding, (59, 59, 15, 15))
        self.assertEqual(len(list(windows)), 8)

    def test_oracle_retains_oil_at_centre_edges_and_corners(self) -> None:
        mask = np.zeros((650, 1250), dtype=np.uint8)
        points = ((0, 0), (0, 1249), (649, 0), (649, 1249), (500, 1000))
        for row, column in points:
            mask[row, column] = 1
        metrics = audit_scene(mask, 256, 128)
        _, windows = iter_original_windows(650, 1250, 256, 128)
        expected = sum(np.any(mask[top:bottom, left:right] == 1) for _, _, top, bottom, left, right in windows)
        self.assertEqual(metrics["oil_positive_patches"], expected)
        self.assertEqual(metrics["oil_pixel_coverage_unique_px"], len(points))

    def test_single_interior_pixel_is_counted_in_four_windows(self) -> None:
        mask = np.zeros((650, 1250), dtype=np.uint8)
        mask[500, 1000] = 1
        metrics = audit_scene(mask, 256, 128)
        self.assertEqual(metrics["oil_positive_patches"], 4)
        self.assertEqual(metrics["oil_pixel_coverage_repeated_px"], 4)

    def test_decoder_accepts_ship_colour_range(self) -> None:
        rgb = np.array([[[0, 0, 0], [0, 255, 255], [255, 0, 0], [100, 50, 5], [0, 153, 0]]], dtype=np.uint8)
        np.testing.assert_array_equal(decode_m4d_mask(rgb), [[0, 1, 2, 3, 4]])

    def test_decoder_rejects_unknown_colours(self) -> None:
        with self.assertRaises(ValueError):
            decode_m4d_mask(np.array([[[1, 2, 3]]], dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
