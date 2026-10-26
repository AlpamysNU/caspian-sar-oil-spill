"""Run the five fold models on a whole grey scene (any size): windows of 256 px with stride 128 on a grid that ends at the border.

Used for the raw Sentinel-1 scenes and for external checks. No training and no tuning: the fold models (`best.pt`) are applied as they
are and their softmax outputs are averaged. The two-scale model gets, besides each window, the 512 px reflect crop around the same
centre reduced to 256 px by 2x2 averaging (`segmentation.context`). Pixels are scaled by 1/255, as in training.
"""
import numpy as np
import torch

from segmentation.context import reduce2, reflect_crop
from segmentation.models import SPECS

WINDOW, STRIDE = 256, 128


def window_starts(n):
    starts = list(range(0, n - WINDOW + 1, STRIDE))
    return starts if starts[-1] + WINDOW == n else starts + [n - WINDOW]


def load_models(checkpoints_root, folds, device, arch='segformer_b2_hrda_c'):
    """The `best.pt` checkpoints of folds `folds` under `<checkpoints_root>/<arch>_p256_foldNN_seed42/`."""
    spec = SPECS[arch]
    models = []
    for fold in folds:
        path = checkpoints_root / f'{spec["name"]}_p256_fold{fold:02d}_seed42' / 'best.pt'
        state = torch.load(path, map_location='cpu', weights_only=False)
        if state.get('arch') != spec['name']:
            raise ValueError(f'Unexpected checkpoint arch {state.get("arch")!r}: {path}')
        model = spec['build'](state['model_config'])
        model.load_state_dict(state['model_state'], strict=True)
        models.append(model.to(device).eval())
    return models


@torch.inference_mode()
def predict(models, image, device, batch_size):
    """image: uint8 [H, W] -> mean softmax over models and windows, float32 [5, H, W] (two-scale models)."""
    h, w = image.shape
    pairs = []
    for y in window_starts(h):
        for x in window_starts(w):
            detail = image[y:y + WINDOW, x:x + WINDOW].astype(np.float32) / 255.0
            offset = (512 - WINDOW) // 2
            context = reduce2(reflect_crop(image, y - offset, x - offset, 512).astype(np.float32) / 255.0)
            pairs.append((y, x, np.stack([detail, context])))
    prob = np.zeros((5, h, w), dtype=np.float64)
    count = np.zeros((h, w), dtype=np.float64)
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start:start + batch_size]
        batch = torch.from_numpy(np.stack([c[2] for c in chunk])).to(device)
        mean = None
        for model in models:
            _, logits = model(batch)
            p = logits.float().softmax(1)
            mean = p if mean is None else mean + p
        mean = (mean / len(models)).cpu().numpy()
        for (y, x, _), p in zip(chunk, mean):
            prob[:, y:y + WINDOW, x:x + WINDOW] += p
            count[y:y + WINDOW, x:x + WINDOW] += 1
    return (prob / count).astype(np.float32)
