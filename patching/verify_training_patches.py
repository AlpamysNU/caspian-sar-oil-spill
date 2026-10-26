#!/usr/bin/env python3
"""Independently check every patch, class count, and reconstructed source pixel."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def verify(output):
    if not __debug__:
        raise RuntimeError('Verification requires enabled assertions; do not use python -O')
    manifest = pd.read_csv(output / 'scene_manifest.csv')
    indices = list(output.glob('index_*.parquet'))
    assert len(indices) == 1
    index = pd.read_parquet(indices[0])
    assert index.patch_id.is_unique
    assert set(index.scene_id) == set(manifest.scene_id)
    assert {p.stem for p in (output / 'patches_npy').glob('*.npy')} == set(index.patch_id)
    for number, scene in enumerate(manifest.itertuples(), start=1):
        image_path, mask_path = Path(scene.image_path), Path(scene.mask_path)
        assert hashlib.sha256(image_path.read_bytes()).hexdigest() == scene.image_sha256
        assert hashlib.sha256(mask_path.read_bytes()).hexdigest() == scene.mask_sha256
        image = np.asarray(Image.open(image_path).convert('L'))
        # Independently supplied label IDs avoid reusing the extraction RGB decoder.
        labels = np.asarray(Image.open(mask_path.parent.parent / 'labels_1D' / mask_path.name))
        assert image.shape == labels.shape == (scene.height, scene.width)
        rows = index[index.scene_id == scene.scene_id]
        first = rows.iloc[0]
        size, stride = int(first.patch_size), int(first.stride)
        h, w = image.shape
        dh = max(size-h, 0) if h < size else (-(h-size)) % stride
        dw = max(size-w, 0) if w < size else (-(w-size)) % stride
        pt, pl = dh//2, dw//2
        expected = {(y,x) for y in range(0,h+dh-size+1,stride) for x in range(0,w+dw-size+1,stride)}
        assert set(zip(rows.y_padded, rows.x_padded)) == expected
        padded = np.pad(image, ((pt,dh-pt),(pl,dw-pl)), mode='reflect')
        reconstructed = np.zeros_like(image, dtype=np.float64)
        coverage = np.zeros_like(image, dtype=np.int32)
        for row in rows.itertuples():
            assert row.split == scene.split
            assert (row.pad_top,row.pad_bottom,row.pad_left,row.pad_right) == (pt,dh-pt,pl,dw-pl)
            y,x = row.y_padded,row.x_padded
            assert (row.y_orig,row.x_orig) == (y-pt,x-pl)
            patch = np.load(output / row.patch_path, allow_pickle=False)
            assert patch.dtype == np.uint8 and patch.shape == (size,size)
            np.testing.assert_array_equal(patch,padded[y:y+size,x:x+size])
            t,b,l,r = max(0,y-pt),min(h,y-pt+size),max(0,x-pl),min(w,x-pl+size)
            counts = np.bincount(labels[t:b,l:r].ravel(),minlength=5)
            for cls,n in zip(('sea','oil','look_alike','ship','land'), counts):
                assert getattr(row,'pixels_'+cls) == n
                assert getattr(row,'label_'+cls) == int(n>0)
            assert row.valid_pixels == (b-t)*(r-l)
            assert row.padding_pixels == size*size-row.valid_pixels
            assert abs(row.pad_ratio-row.padding_pixels/(size*size)) < 1e-12
            reconstructed[t:b,l:r] += patch[t-(y-pt):b-(y-pt),l-(x-pl):r-(x-pl)]
            coverage[t:b,l:r] += 1
        assert np.all(coverage > 0)
        np.testing.assert_array_equal(reconstructed / coverage, image)
        if number % 100 == 0:
            print(f'Verified {number}/{len(manifest)} scenes in {output.name}', flush=True)
    result = {'scenes':len(manifest),'patches':len(index),'oil_positive_patches':int(index.label_oil.sum()),
              'all_patch_pixels_and_labels_checked':True,'uniform_reconstruction_max_error':0,
              'splits':sorted(index.split.unique().tolist())}
    (output/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(output, result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    verify(parser.parse_args().output)
