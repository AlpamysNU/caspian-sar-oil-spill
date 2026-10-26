"""Small helpers shared by the analysis scripts that read saved out-of-fold prediction maps."""
import numpy as np
from PIL import Image

OIL = 1
CONN8 = np.ones((3, 3), dtype=bool)


def read_prediction(run_dir, fold, scene):
    """Class map of one OOF scene: `<run_dir>/fold0F/<scene>_prediction.npy` (or the older `.png`)."""
    npy = run_dir / f'fold{fold:02d}' / f'{scene}_prediction.npy'
    if npy.is_file():
        return np.load(npy)
    png = run_dir / f'fold{fold:02d}' / f'{scene}_prediction.png'
    if png.is_file():
        return np.asarray(Image.open(png))
    raise FileNotFoundError(f'No prediction for {scene} in {run_dir}')
