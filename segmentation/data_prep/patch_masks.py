"""Patch-mask bundles for training.

    python -m segmentation.data_prep.patch_masks build --output-root <dir> [--verify]
    python -m segmentation.data_prep.patch_masks from-labels --labels-dir <dir of 1D maps> --output-root <dir>

  build        precomputed 256 px masks (stride 128 grid) for all five frozen folds from the original masks, verified against every scene
  from-labels  the same bundle format from a directory of 1D class maps (for example the revised labels of the label review)

Inputs default to the layout under --project-root (stage2_results/ and oil-spill-detection-dataset/); a bundle is passed to
segmentation.train with --patch-masks-root.
"""


import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from segmentation import paths
from segmentation.data import atomic_save, crop_patch_mask, load_tables, patch_mask_contract, read_mask, sha256, verify_patch_masks



# ----------------------------------------------------------------------------- build


DOC_build = """Build/verify native 256/128 masks for all five frozen development folds."""


def main_build():
    parser = argparse.ArgumentParser(description=DOC_build)
    paths.add_root_argument(parser)
    parser.add_argument('--data-root', type=Path, default=paths.STAGE2_RESULTS / 'training_patches/ps256_overlap50')
    parser.add_argument('--folds-dir', type=Path, default=paths.STAGE2_RESULTS / 'grouped_folds_v1')
    parser.add_argument('--masks-root', type=Path, default=paths.DATASET / 'train/labels')
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--verify', action='store_true',
                        help='Verify existing bundle against every original scene mask')
    args = parser.parse_args()
    start = time.monotonic()
    train, val, originals = load_tables(args.data_root, args.folds_dir, 1, args.masks_root)
    frame = pd.concat([train, val], ignore_index=True).sort_values(['scene_id', 'patch_id'])
    if args.verify:
        verify_patch_masks(args.output_root, args.data_root, frame)
    else:
        # Never overwrite a completed or interrupted build.
        args.output_root.mkdir(parents=True, exist_ok=False)
        (args.output_root / 'masks_npy').mkdir()
    files = {}
    padded = 0
    for scene_number, (scene_id, rows) in enumerate(frame.groupby('scene_id', sort=True), 1):
        mask = read_mask(originals[scene_id])
        for row in rows.to_dict('records'):
            target = crop_patch_mask(mask, row)
            padded += int(np.count_nonzero(target == 255))
            path = args.output_root / 'masks_npy' / f'{row["patch_id"]}.npy'
            if not path.resolve().is_relative_to((args.output_root / 'masks_npy').resolve()):
                raise ValueError('Patch ID escapes output directory')
            if not args.verify:
                temporary = path.with_suffix('.npy.tmp')
                with temporary.open('wb') as output:
                    np.save(output, target, allow_pickle=False)
                os.replace(temporary, path)
            saved = np.load(path, allow_pickle=False)
            if saved.dtype != np.uint8 or not np.array_equal(saved, target):
                raise ValueError(f'Saved mask differs from original crop: {path}')
            files[row['patch_id']] = sha256(path)
        if scene_number % 100 == 0:
            print(f'{scene_number}/{len(originals)} scenes; {len(files)} masks verified', flush=True)
    if not args.verify:
        atomic_save({'contract': patch_mask_contract(args.data_root), 'files': files},
                    args.output_root / 'manifest.json')
    report = {'status': 'verified', 'patches': len(files), 'scenes': len(originals),
              'padding_pixels': padded, 'elapsed_seconds': time.monotonic() - start,
              'manifest_sha256': sha256(args.output_root / 'manifest.json')}
    atomic_save(report, args.output_root / 'verification.json')
    print(json.dumps(report, indent=2), flush=True)



# ----------------------------------------------------------------------------- from-labels


DOC_from_labels = """Build a patch-mask bundle (256/128 grid, same format as `build`) from a directory of 1D scene label maps (2026-10-04).

Intended for the contour-tightened labels (labels_1D_tight033) and to be run where the frozen data live (the HPC project root): it reads the frozen patch
index through the same hash-verified load_tables() as the trainer. Differences against `build`: scene masks come from
<labels-dir>/<img_XXXX>.png (class indices already decoded; build_tight_labels.py verified they equal the decoded RGB masks) and the per-patch class-count check
against the frozen index is skipped (the counts differ on purpose); the valid-pixel geometry check is kept. The bundle contract (index and scene manifest
hashes, class mapping) is the trainer's own, so `segmentation.train` accepts the bundle through --patch-masks-root without code changes.
--self-test --labels-dir <ORIGINAL labels_1D> --reference-bundle <existing bundle> compares the first --n-scenes scenes bit for bit with the existing bundle and
writes nothing; run it before the real build."""


def crop(mask, row):
    y, x = int(row['y_orig']), int(row['x_orig'])
    height, width = mask.shape
    y0, y1, x0, x1 = max(y, 0), min(y + 256, height), max(x, 0), min(x + 256, width)
    if y0 >= y1 or x0 >= x1:
        raise ValueError('Patch has no valid pixels')
    target = np.full((256, 256), 255, dtype=np.uint8)
    target[y0 - y:y1 - y, x0 - x:x1 - x] = mask[y0:y1, x0:x1]
    if int(np.sum(target != 255)) != int(row['valid_pixels']):
        raise ValueError(f'Valid-pixel geometry mismatch: {row["patch_id"]}')
    return target


def main_from_labels():
    p = argparse.ArgumentParser(description=DOC_from_labels)
    paths.add_root_argument(p)
    p.add_argument('--data-root', type=Path, default=paths.STAGE2_RESULTS / 'training_patches/ps256_overlap50')
    p.add_argument('--folds-dir', type=Path, default=paths.STAGE2_RESULTS / 'grouped_folds_v1')
    p.add_argument('--masks-root', type=Path, default=paths.DATASET / 'train/labels', help='original train masks (enumerates scenes, as in `build`)')
    p.add_argument('--labels-dir', required=True, type=Path)
    p.add_argument('--output-root', required=True, type=Path)
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--reference-bundle', type=Path, default=None)
    p.add_argument('--n-scenes', type=int, default=20)
    a = p.parse_args()
    if a.self_test and a.reference_bundle is None:
        raise SystemExit('--self-test needs --reference-bundle')
    start = time.monotonic()
    train, val, _ = load_tables(a.data_root, a.folds_dir, 1, a.masks_root)
    frame = pd.concat([train, val], ignore_index=True).sort_values(['scene_id', 'patch_id'])
    if not a.self_test:
        a.output_root.mkdir(parents=True, exist_ok=False)   # never overwrite
        (a.output_root / 'masks_npy').mkdir()
    files, checked = {}, 0
    for n, (scene_id, rows) in enumerate(frame.groupby('scene_id', sort=True), 1):
        if a.self_test and n > a.n_scenes:
            break
        mask = np.asarray(Image.open(a.labels_dir / f'{scene_id.removeprefix("train_")}.png'))
        for row in rows.to_dict('records'):
            target = crop(mask, row)
            if a.self_test:
                if not np.array_equal(np.load(a.reference_bundle / 'masks_npy' / f'{row["patch_id"]}.npy'), target):
                    raise SystemExit(f'SELF-TEST FAILED: {row["patch_id"]}')
                checked += 1
                continue
            path = a.output_root / 'masks_npy' / f'{row["patch_id"]}.npy'
            tmp = path.with_suffix('.npy.tmp')
            with tmp.open('wb') as f:
                np.save(f, target, allow_pickle=False)
            os.replace(tmp, path)
            files[row['patch_id']] = sha256(path)
        if n % 100 == 0:
            print(f'{n} scenes', flush=True)
    if a.self_test:
        print(f'self-test passed: {checked} patches of {a.n_scenes} scenes identical to the existing bundle')
        return
    atomic_save({'contract': patch_mask_contract(a.data_root), 'files': files}, a.output_root / 'manifest.json')
    atomic_save({'status': 'built_from_labels', 'labels_dir': str(a.labels_dir), 'patches': len(files),
                 'elapsed_seconds': time.monotonic() - start, 'manifest_sha256': sha256(a.output_root / 'manifest.json')},
                a.output_root / 'provenance.json')
    print('built', len(files), 'patches')

COMMANDS = {
    'build': main_build,
    'from-labels': main_from_labels,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print('usage: python -m segmentation.data_prep.patch_masks <command> [options]\n\ncommands:\n  ' + '\n  '.join(COMMANDS))
        raise SystemExit(0 if len(sys.argv) > 1 and sys.argv[1] in ('-h', '--help') else 2)
    command = sys.argv.pop(1)
    sys.argv[0] = f'segmentation.data_prep.patch_masks {command}'
    COMMANDS[command]()


if __name__ == '__main__':
    main()
