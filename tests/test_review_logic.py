import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from segmentation.label_review import review_logic as r


class ReviewLogicTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / 'data'
        (self.root / 'train/images').mkdir(parents=True)
        (self.root / 'train/labels_1D').mkdir(parents=True)
        img = (np.random.default_rng(0).random((40, 60)) * 255).astype(np.uint8)
        lab = np.zeros((40, 60), np.uint8); lab[5:10, 5:20] = 2; lab[30:33, 40:44] = 4
        Image.fromarray(img).save(self.root / 'train/images/img_0001.jpg', quality=100)
        Image.fromarray(lab).save(self.root / 'train/labels_1D/img_0001.png')
        self.out = Path(self.tmp.name) / 'out'; self.out.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_roundtrip_and_log(self):
        image, original, revised = r.load_scene(self.root, 'img_0001', self.out)
        revised = revised.copy(); revised[30:33, 40:44] = 3; revised[5:10, 5:20] = 1
        changed, warnings = r.save(self.root, self.out, 'img_0001', image, original, revised, 'ship_added', 'test', self.out / 'log.csv')
        self.assertEqual(changed, 12 + 75)
        saved = np.asarray(Image.open(self.out / 'labels_1D_revised/img_0001.png'))
        self.assertTrue((saved == revised).all() and saved.dtype == np.uint8 and saved.shape == original.shape)
        self.assertEqual(r.transitions(original, revised), {'land->ship': 12, 'look_alike->oil': 75})
        self.assertTrue((self.out / 'overlays/img_0001.png').exists())
        # resume: the saved revision is loaded back
        _, _, again = r.load_scene(self.root, 'img_0001', self.out)
        self.assertTrue((again == revised).all())
        # reverting to the original removes the revised file
        r.save(self.root, self.out, 'img_0001', image, original, original, 'no_change', '', self.out / 'log.csv')
        self.assertFalse((self.out / 'labels_1D_revised/img_0001.png').exists())

    def test_rejects_bad_values_shape_and_dataset_dir(self):
        image, original, revised = r.load_scene(self.root, 'img_0001', self.out)
        bad = revised.copy(); bad[0, 0] = 7
        with self.assertRaises(ValueError): r.validate(image, original, bad)
        with self.assertRaises(ValueError): r.validate(image, original, revised[:-1])
        with self.assertRaises(ValueError): r.validate(image, original, revised.astype(np.int32))
        with self.assertRaises(ValueError): r.save(self.root, self.root / 'train', 'img_0001', image, original, revised, 'x', '', self.out / 'l.csv')


if __name__ == '__main__':
    unittest.main()
