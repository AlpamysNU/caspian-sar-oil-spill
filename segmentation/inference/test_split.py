"""One-time evaluation of the final Stage-3 models on the 110 official test scenes of M4D-SAR-OilSpill (2026-10-20).

No training, no tuning, no threshold choice: the five fold models (best.pt, seed 42) of each architecture are applied as they are.
Scenes are reconstructed exactly like the OOF scenes: grey conversion, reflect padding of 59 / 15 px, 256 px windows on the
frozen grid (starts -59 + 128 k, -15 + 128 k), uniform average of the softmax over overlapping windows. The two-scale model gets
the 512 px reflect crop around each window reduced to 256 px (context.py).

Per scene and per prediction (each single fold model, and the mean softmax of the five = the 'ensemble5' prediction) the 5x5
confusion matrix is written to <output-root>/<arch>/confusion_by_scene.csv; scoring is in segmentation/evaluation/test_split.py.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from segmentation import paths
from segmentation.context import reduce2, reflect_crop
from segmentation.models import SPECS

WINDOW, STRIDE, PAD_Y, PAD_X = 256, 128, 59, 15


def grid(h, w):
    ys = list(range(-PAD_Y, h + PAD_Y - WINDOW + 1, STRIDE))
    xs = list(range(-PAD_X, w + PAD_X - WINDOW + 1, STRIDE))
    return [(y, x) for y in ys for x in xs]


def load_models(spec, checkpoints_root, device):
    models = []
    for fold in range(1, 6):
        path = checkpoints_root / f'{spec["name"]}_p256_fold{fold:02d}_seed42' / 'best.pt'
        try:
            state = torch.load(path, map_location='cpu', weights_only=False)
        except TypeError:   # torch < 1.13 has no weights_only
            state = torch.load(path, map_location='cpu')
        if state.get('arch') != spec['name']:
            raise ValueError(f'Checkpoint arch {state.get("arch")!r} is not {spec["name"]!r}: {path}')
        model = spec['build'](state['model_config'])
        model.load_state_dict(state['model_state'], strict=True)
        models.append(model.to(device).eval())
    return models


@torch.inference_mode()
def predict(models, grey, two_scale, device, batch_size):
    """-> list of float32 [5, H, W] scene probabilities, one per model (uniform average over windows)."""
    h, w = grey.shape
    padded = np.pad(grey, ((PAD_Y, PAD_Y), (PAD_X, PAD_X)), mode='reflect')
    windows = grid(h, w)
    probs = [np.zeros((5, h, w), dtype=np.float32) for _ in models]
    count = np.zeros((h, w), dtype=np.float32)
    for start in range(0, len(windows), batch_size):
        chunk = windows[start:start + batch_size]
        tiles = []
        for y, x in chunk:
            detail = padded[y + PAD_Y:y + PAD_Y + WINDOW, x + PAD_X:x + PAD_X + WINDOW].astype(np.float32) / 255.0
            if two_scale:
                off = (512 - WINDOW) // 2
                context = reduce2(reflect_crop(grey, y - off, x - off, 512).astype(np.float32) / 255.0)
                tiles.append(np.stack([detail, context]))
            else:
                tiles.append(detail[None])
        batch = torch.from_numpy(np.stack(tiles)).to(device)
        for index, model in enumerate(models):
            _, logits = model(batch)
            p = F.interpolate(logits, size=(WINDOW, WINDOW), mode='bilinear', align_corners=False).float().softmax(1).cpu().numpy()
            for (y, x), tile in zip(chunk, p):
                y0, y1, x0, x1 = max(y, 0), min(y + WINDOW, h), max(x, 0), min(x + WINDOW, w)
                probs[index][:, y0:y1, x0:x1] += tile[:, y0 - y:y1 - y, x0 - x:x1 - x]
        for y, x in chunk:
            count[max(y, 0):min(y + WINDOW, h), max(x, 0):min(x + WINDOW, w)] += 1
    return [p / count for p in probs]


def confusion(target, pred):
    valid = target != 255
    pairs = target[valid].astype(np.int64) * 5 + pred[valid].astype(np.int64)
    return np.bincount(pairs, minlength=25)


def main():
    p = argparse.ArgumentParser()
    paths.add_root_argument(p)
    p.add_argument('--arch', required=True, choices=sorted(SPECS))
    p.add_argument('--checkpoints-root', type=Path, required=True)
    p.add_argument('--test-root', type=Path, default=paths.DATASET / 'test', help='the official test split of M4D')
    p.add_argument('--output-root', type=Path, default=paths.RESULTS / 'test_split_eval')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    p.add_argument('--limit', type=int, default=0, help='TEST ONLY: first N scenes')
    p.add_argument('--save-predictions', action='store_true', help='save the ensemble class map of every scene (uint8 npy)')
    args = p.parse_args()
    device = torch.device(args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu')
    spec = SPECS[args.arch]
    args.checkpoints_root, args.output_root, args.test_root = map(paths.resolve, (args.checkpoints_root, args.output_root, args.test_root))
    models = load_models(spec, args.checkpoints_root, device)
    two_scale = 'dataset_factory' in spec
    out = args.output_root / args.arch
    out.mkdir(parents=True, exist_ok=True)
    scenes = sorted(args.test_root.joinpath('images').glob('img_*.jpg'))
    if args.limit:
        scenes = scenes[:args.limit]
    rows = []
    for number, path in enumerate(scenes, 1):
        grey = np.asarray(Image.open(path).convert('L'), dtype=np.uint8)
        # class-index file of the dataset: the RGB label PNGs of some test scenes contain compression colours (img_0016, 0032, 0038, 0054, ...);
        # on all scenes where the RGB colours decode cleanly, labels_1D is identical to the decoded RGB labels (checked)
        target = np.asarray(Image.open(args.test_root / 'labels_1D' / (path.stem + '.png')), dtype=np.uint8)
        if target.shape != grey.shape:
            raise ValueError(f'Shape mismatch {path.stem}: {target.shape} vs {grey.shape}')
        probs = predict(models, grey, two_scale, device, args.batch_size)
        ensemble = np.mean(probs, axis=0)
        predictions = {f'fold{i + 1}': pr.argmax(0) for i, pr in enumerate(probs)}
        predictions['ensemble5'] = ensemble.argmax(0)
        for name, pred in predictions.items():
            rows.append({'scene': path.stem, 'prediction': name, **{f'c{i}': int(v) for i, v in enumerate(confusion(target, pred))}})
        if args.save_predictions:
            np.save(out / f'{path.stem}_ensemble5.npy', predictions['ensemble5'].astype(np.uint8))
        if number == 1 or number % 10 == 0 or number == len(scenes):
            print(f'{args.arch}: scene {number}/{len(scenes)}', flush=True)
    pd.DataFrame(rows).to_csv(out / 'confusion_by_scene.csv', index=False)


if __name__ == '__main__':
    main()
