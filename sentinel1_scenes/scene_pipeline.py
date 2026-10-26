"""From a raw Sentinel-1 GRD scene to a HRDA-C class map

Working folder: <project root>/sentinel1/ (scene folders, raw/, out/, optional osm/platforms.json). Checkpoints: <project root>/results/stage3_*.

    python -m sentinel1_scenes.scene_pipeline run --scene-dir <scene> --x0 .. --y0 .. --tag <tag>   calibrated preprocessing + HRDA-C on a crop (recipes A and B of Section 5.6)
    python -m sentinel1_scenes.scene_pipeline compare-models --tag <tag> --model <name>             B2 / DeepLabV3+ on the grey crop of a run
    python -m sentinel1_scenes.scene_pipeline sensitivity                                            one-parameter changes of the recipe on one fixed region
    python -m sentinel1_scenes.scene_pipeline compare-sensitivity                                    class areas and Oil IoU of those runs against the baseline
    python -m sentinel1_scenes.scene_pipeline grey / annotation-info                                 earlier uncalibrated grey conversion; incidence angle and calibration LUT of a scene
"""


import argparse
import glob
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.interpolate import LinearNDInterpolator

from segmentation import paths
from segmentation.inference.scene import load_models, predict, WINDOW, window_starts
from segmentation.models import SPECS



# ----------------------------------------------------------------------------- grey


DOC_grey = """Raw Sentinel-1 GRD VV (uint16 DN) -> 8-bit grey scene that mimics the M4D training images.

Recipe estimated from measurements on the M4D scenes (texture_probe.py, scale_probe*.py), NOT from documentation:
  - pixel: 2x2 mean of intensity (DN^2), about 20 m (lag-1 speckle correlation 0.30, as in M4D; raw 10 m gives 0.6)
  - scale: logarithmic (dB), about 22 grey levels per dB (noise-based estimate 24, land-sea contrast estimate 20)
  - level: the scene median is mapped to the M4D median sea grey level (96); clip to 0..255, no-data 0
Input is the raw uint16 file written by:  gdal_translate -of ENVI raw/vv_full.tif raw/vv_full.bin  (the python GDAL binding crashes here)
Run:  python3 preprocess_scene.py raw/vv_full.bin 25743 16709 out/scene_20m.png"""


WEB = paths.PROJECT_ROOT / 'sentinel1'


grey_POOL, LEVELS_PER_DB, SEA_GREY = 2, 22.0, 96.0


def main_grey():
    src, width, height, dst = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    raw = np.memmap(src, dtype='<u2', mode='r', shape=(height, width))
    h, w = height // grey_POOL * grey_POOL, width // grey_POOL * grey_POOL
    inten = np.empty((h // grey_POOL, w // grey_POOL), np.float32)
    ok = np.empty((h // grey_POOL, w // grey_POOL), bool)
    STRIP = 1024
    for y in range(0, h, STRIP):
        rows = min(STRIP, h - y)
        dn = np.asarray(raw[y:y + rows, :w], dtype=np.float32)
        k = rows // grey_POOL
        inten[y // grey_POOL:y // grey_POOL + k] = (dn ** 2).reshape(k, grey_POOL, w // grey_POOL, grey_POOL).mean((1, 3))
        ok[y // grey_POOL:y // grey_POOL + k] = (dn > 0).reshape(k, grey_POOL, w // grey_POOL, grey_POOL).all((1, 3))
    db = 10 * np.log10(np.maximum(inten, 1.0))
    ref = float(np.median(db[ok]))
    grey = np.clip(SEA_GREY + LEVELS_PER_DB * (db - ref), 1, 255)
    grey[~ok] = 0
    Image.fromarray(grey.astype(np.uint8)).save(dst)
    json.dump({'source': src, 'pool': grey_POOL, 'levels_per_db': LEVELS_PER_DB, 'sea_grey': SEA_GREY,
               'scene_median_db_of_DN2': ref, 'shape_hw': list(grey.shape), 'valid_fraction': float(ok.mean())},
              open(dst.replace('.png', '.json'), 'w'), indent=1)
    print('saved', dst, grey.shape, 'median dB(DN^2)', round(ref, 2), 'valid', round(float(ok.mean()), 3))



# ----------------------------------------------------------------------------- annotation-info


DOC_annotation_info = """Incidence angle, calibration LUT and sea backscatter profile across range for the scene (no GDAL needed)."""


def main_annotation_info():
    A = glob.glob(str(WEB / 'raw/meta/annotation/s1*-vv-*.xml'))[0]
    C = glob.glob(str(WEB / 'raw/meta/annotation/calibration/calibration-*-vv-*.xml'))[0]
    root = ET.parse(A).getroot()
    info = root.find('imageAnnotation/imageInformation')
    print('lines', info.findtext('numberOfLines'), 'samples', info.findtext('numberOfSamples'),
          'pixel spacing range/azimuth', info.findtext('rangePixelSpacing'), info.findtext('azimuthPixelSpacing'),
          'range looks', info.findtext('rangeProcessing/numberOfLooks') if info.find('rangeProcessing') is not None else '',
          'incidence mid', info.findtext('incidenceAngleMidSwath'))
    print('product', root.findtext('adsHeader/mode'), root.findtext('adsHeader/polarisation'), root.findtext('adsHeader/startTime'))
    pts = [(int(p.findtext('line')), int(p.findtext('pixel')), float(p.findtext('latitude')), float(p.findtext('longitude')),
            float(p.findtext('incidenceAngle'))) for p in root.iter('geolocationGridPoint')]
    g = np.array(pts)
    print('geolocation grid points', len(g), 'lines', g[:, 0].min(), g[:, 0].max(), 'pixels', g[:, 1].min(), g[:, 1].max())
    print('incidence angle: min %.1f max %.1f' % (g[:, 4].min(), g[:, 4].max()))
    # incidence angle vs pixel column (mean over lines)
    cols = np.unique(g[:, 1])
    inc_col = np.array([g[g[:, 1] == c, 4].mean() for c in cols])
    print('incidence by pixel column (every ~10th):', [(int(c), round(float(a), 1)) for c, a in zip(cols[::max(len(cols)//10, 1)], inc_col[::max(len(cols)//10, 1)])])
    np.save(WEB / 'out/inc_cols.npy', np.c_[cols, inc_col])
    # calibration: sigmaNought vectors
    cr = ET.parse(C).getroot()
    vecs = cr.findall('calibrationVectorList/calibrationVector')
    sn = []
    for v in vecs:
        pix = np.array(v.findtext('pixel').split(), int); s = np.array(v.findtext('sigmaNought').split(), float)
        sn.append((int(v.findtext('line')), pix, s))
    mid = sn[len(sn)//2]
    print('calibration vectors', len(vecs), 'sigmaNought LUT at pixel 0/mid/end of the middle vector:',
          [round(float(mid[2][i]), 1) for i in (0, len(mid[2])//2, -1)], 'pixels', int(mid[1][0]), int(mid[1][-1]))
    np.save(WEB / 'out/sigma_lut_mid.npy', np.c_[mid[1], mid[2]])



# ----------------------------------------------------------------------------- run


DOC_run = """Calibrated preprocessing + HRDA-C on a crop of any downloaded scene directory (VV .bin + meta/ annotation).

Example:  python -m sentinel1_scenes.scene_pipeline run --scene-dir sentinel1/scene_1011 --x0 19800 --y0 4800 --w 5000 --h 3900 --tag s1011
Outputs (sentinel1/out/<tag>_*): grey_<variant>.npy, pred_<variant>.npy, snr_db.npy (float16), overview.png, summary.json.
Variants: 'raw' (no noise subtraction) and 'denoised' (thermal noise subtracted, floor 10 dB below the noise level).
ASSUMPTION (not measured): sea sigma0 -19.5 dB -> grey 96, 22 grey levels per dB, 2x2 intensity pooling (about 20 m)."""


run_POOL, LEVELS, SIGMA_REF_DB, GREY_REF = 2, 22.0, -19.5, 96.0


CLASSES = ['Sea', 'Oil', 'Look-alike', 'Ship', 'Land']


COLOURS = np.array([(0, 0, 0), (0, 255, 255), (255, 0, 0), (153, 76, 0), (0, 153, 0)], np.uint8)


def main_run():
    p = argparse.ArgumentParser(); paths.add_root_argument(p)
    p.add_argument('--scene-dir', type=Path, required=True)
    p.add_argument('--x0', type=int, required=True); p.add_argument('--y0', type=int, required=True)
    p.add_argument('--w', type=int, default=5000); p.add_argument('--h', type=int, default=3900)
    p.add_argument('--tag', required=True)
    p.add_argument('--norm', choices=('median', 'abs'), default='median', help='median: crop median sigma0 -> grey 96 (M4D-like); abs: sigma0 -19.5 dB -> 96')
    p.add_argument('--sea-from', type=Path, help='pred_raw .npy of an earlier run on the SAME crop: the median reference is then taken over pixels that are not Land there (20 px margin), M4D-like sea level when land dominates the crop')
    p.add_argument('--variants', nargs='+', default=['raw', 'denoised'])
    p.add_argument('--save-probs', action='store_true', help='also save the Oil and Look-alike probability maps (float16)')
    p.add_argument('--levels', type=float, default=22.0, help='grey levels per dB (default 22, estimated from M4D); lower values leave very dark areas unclipped')
    p.add_argument('--grey-ref', type=float, default=96.0, help='grey level of the sea reference (default 96; the median sea of the M4D train scenes is about 134)')
    p.add_argument('--edge', type=int, default=150, help='raw pixels at the scene border to mask out (border noise / no-data)')
    args = p.parse_args()
    sd, tag = args.scene_dir, args.tag
    X0, Y0, W, H = args.x0, args.y0, args.w // run_POOL * run_POOL, args.h // run_POOL * run_POOL

    hdr = (sd / 'vv_full.hdr').read_text()
    RAW_W, RAW_H = int(re.search(r'samples\s*=\s*(\d+)', hdr).group(1)), int(re.search(r'lines\s*=\s*(\d+)', hdr).group(1))
    X0, Y0 = max(X0, args.edge), max(Y0, args.edge)
    W = min(W, RAW_W - args.edge - X0) // run_POOL * run_POOL
    H = min(H, RAW_H - args.edge - Y0) // run_POOL * run_POOL
    print(f'crop clamped inside the scene border: x {X0}..{X0 + W}, y {Y0}..{Y0 + H}', flush=True)
    ann = ET.parse(glob.glob(str(sd / 'meta/annotation/s1*-vv-*.xml'))[0]).getroot()
    cal = ET.parse(glob.glob(str(sd / 'meta/annotation/calibration/calibration-*-vv-*.xml'))[0]).getroot()
    noi = ET.parse(glob.glob(str(sd / 'meta/annotation/calibration/noise-*-vv-*.xml'))[0]).getroot()


    def vectors_to_crop(vectors, key, rows, cols):
        lines = np.array([int(v.findtext('line')) for v in vectors], float)
        per_vec = np.stack([np.interp(cols, np.array(v.findtext('pixel').split(), float),
                                      np.array(v.findtext(key).split(), float)) for v in vectors])
        idx = np.clip(np.searchsorted(lines, rows) - 1, 0, len(lines) - 2)
        t = np.clip((rows - lines[idx]) / (lines[idx + 1] - lines[idx]), 0, 1)[:, None]
        return per_vec[idx] * (1 - t) + per_vec[idx + 1] * t


    rows, cols = np.arange(Y0, Y0 + H), np.arange(X0, X0 + W)
    sigma_lut = vectors_to_crop(cal.findall('calibrationVectorList/calibrationVector'), 'sigmaNought', rows, cols)
    noise = vectors_to_crop(noi.findall('noiseRangeVectorList/noiseRangeVector'), 'noiseRangeLut', rows, cols)
    for v in noi.findall('noiseAzimuthVectorList/noiseAzimuthVector'):
        a, b = int(v.findtext('firstRangeSample')), int(v.findtext('lastRangeSample'))
        sel = (cols >= a) & (cols <= b)
        if sel.any():
            noise[:, sel] *= np.interp(rows, np.array(v.findtext('line').split(), float), np.array(v.findtext('noiseAzimuthLut').split(), float))[:, None]

    raw = np.memmap(sd / 'vv_full.bin', dtype='<u2', mode='r', shape=(RAW_H, RAW_W))
    dn = np.asarray(raw[Y0:Y0 + H, X0:X0 + W], dtype=np.float64)
    sh = (H // run_POOL, run_POOL, W // run_POOL, run_POOL)
    valid = (dn > 0).reshape(sh).all((1, 3))
    inner = ((rows >= args.edge) & (rows < RAW_H - args.edge))[:, None] & ((cols >= args.edge) & (cols < RAW_W - args.edge))[None, :]
    border = ~inner.reshape(sh).all((1, 3))
    valid &= ~border
    print(f'border strip masked: {float(border.mean()) * 100:.1f} % of the crop', flush=True)
    inten, noise_p, lut2 = (a.reshape(sh).mean((1, 3)) for a in (dn ** 2, noise, sigma_lut ** 2))
    snr = inten / np.maximum(noise_p, 1e-9)
    snr_db = 10 * np.log10(np.maximum(snr, 1e-6))
    (WEB / 'out').mkdir(parents=True, exist_ok=True)
    np.save(WEB / f'out/{tag}_snr_db.npy', snr_db.astype(np.float16))

    # incidence angle of the crop from the geolocation grid
    g = np.array([[float(q.findtext(k)) for k in ('line', 'pixel', 'latitude', 'longitude', 'incidenceAngle')] for q in ann.iter('geolocationGridPoint')])
    inc = LinearNDInterpolator(g[:, [1, 0]], g[:, 4])
    lonlat = LinearNDInterpolator(g[:, [1, 0]], g[:, [3, 2]])
    print(f'crop incidence {float(inc(X0, Y0 + H / 2)):.1f}..{float(inc(X0 + W, Y0 + H / 2)):.1f} deg; '
          f'corners lon/lat TL {lonlat(X0, Y0).round(3)} BR {lonlat(X0 + W, Y0 + H).round(3)}', flush=True)
    sigma_db_raw = 10 * np.log10(np.median((inten / lut2)[valid]))
    print(f'median sigma0 {sigma_db_raw:.1f} dB, median SNR {np.median(snr_db[valid]):.1f} dB, SNR<3 dB: {float((snr[valid] < 2).mean()) * 100:.1f} %', flush=True)


    sea_ref = valid
    if args.sea_from is not None:
        from scipy import ndimage
        land = np.load(args.sea_from) == 4
        assert land.shape == valid.shape, 'sea-from map must come from the same crop'
        sea_ref = valid & ~ndimage.binary_dilation(land, iterations=20)
        print(f'sea reference: {float(sea_ref.mean()) * 100:.1f} % of the crop (land and 20 px margin excluded)', flush=True)


    def to_grey(sigma0):
        db = 10 * np.log10(np.maximum(sigma0, 1e-6))
        ref = float(np.median(db[sea_ref])) if args.norm == 'median' else SIGMA_REF_DB
        out = np.clip(args.grey_ref + args.levels * (db - ref), 1, 255)
        out[~valid] = 0
        return out.astype(np.uint8)


    variants = {'raw': to_grey(inten / lut2), 'denoised': to_grey(np.maximum(inten - noise_p, 0.1 * noise_p) / lut2)}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = load_models(paths.RESULTS / 'stage3_segformer_b2_hrda_c_shift_256', [1, 2, 3, 4, 5], device)

    # OSM platforms/wells in crop pixel coordinates (scene pixel -> lon/lat is the grid; invert by interpolating lon/lat -> pixel)
    to_px = LinearNDInterpolator(g[:, [3, 2]], g[:, [1, 0]])
    points = []
    for e in (json.load(open(WEB / 'osm/platforms.json'))['elements'] if (WEB / 'osm/platforms.json').exists() else []):
        c = e.get('center') or e
        px = to_px(c['lon'], c['lat'])
        if px is None or np.isnan(px).any(): continue
        x, y = (px[0] - X0) / run_POOL, (px[1] - Y0) / run_POOL
        if -100 <= x < W / run_POOL + 100 and -100 <= y < H / run_POOL + 100:
            points.append((e['tags'].get('man_made'), x, y))
    print('OSM objects in crop:', len(points), flush=True)

    summary = {'norm': args.norm, 'edge_px': args.edge, 'tag': tag, 'scene_dir': str(sd), 'crop_raw': [X0, Y0, W, H], 'median_sigma0_db': sigma_db_raw,
               'median_snr_db': float(np.median(snr_db[valid])), 'low_snr_pct': float((snr[valid] < 2).mean() * 100),
               'osm_objects': len(points)}
    preds = {}
    for name, grey in {k: v for k, v in variants.items() if k in args.variants}.items():
        t = time.time()
        prob = predict(models, grey, device, 8)
        pred = prob.argmax(0).astype(np.uint8)
        if args.save_probs:
            np.save(WEB / f'out/{tag}_prob_oil_la_{name}.npy', np.asarray(prob[1:3], dtype=np.float16))
        pred[border] = 0
        preds[name] = pred
        np.save(WEB / f'out/{tag}_grey_{name}.npy', grey); np.save(WEB / f'out/{tag}_pred_{name}.npy', pred)
        low = snr < 2
        summary[name] = {'seconds': round(time.time() - t), 'grey_mean': float(grey.mean()),
                         'share_pct': {c: round(float((pred == i).mean()) * 100, 2) for i, c in enumerate(CLASSES)},
                         'oil_km2': round(float((pred == 1).sum()) * 0.0004, 1), 'lookalike_km2': round(float((pred == 2).sum()) * 0.0004, 1),
                         'oil_in_low_snr_pct': round(float(low[pred == 1].mean()) * 100, 1) if (pred == 1).any() else None,
                         'lookalike_in_low_snr_pct': round(float(low[pred == 2].mean()) * 100, 1) if (pred == 2).any() else None}
        print(name, json.dumps(summary[name]), flush=True)
        json.dump(summary, open(WEB / f'out/{tag}_summary.json', 'w'), indent=1)

    # overview: grey | prediction (raw variant) | grey with SNR<3 dB tinted; platforms circled; halved for size
    grey, pred = variants['raw'], preds['raw']
    rgb = np.stack([grey] * 3, -1).copy()
    rgb[snr < 2] = (0.5 * rgb[snr < 2] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)
    panel = np.concatenate([np.stack([grey] * 3, -1), COLOURS[pred], rgb], 1)
    img = Image.fromarray(panel)
    d = ImageDraw.Draw(img)
    Wp = W // run_POOL
    for kind, x, y in points:
        r = 14 if kind == 'offshore_platform' else 6
        for off in (0, Wp, 2 * Wp):
            d.ellipse([x + off - r, y - r, x + off + r, y + r], outline=(0, 255, 255), width=3)
    img.resize((img.width // 2, img.height // 2), Image.LANCZOS).save(WEB / f'out/{tag}_overview.png')
    print('done')



# ----------------------------------------------------------------------------- compare-models


DOC_compare_models = """Run single-scale Stage-3 models (five fold checkpoints averaged) on the grey crop saved by `run`, same 256 px windows with stride 128.
Example:  python -m sentinel1_scenes.scene_pipeline compare-models --tag c210808w --model b2_revised_s42
Writes sentinel1/out/<tag>_<model>_pred.npy (uint8 classes) and _prob_oil_la.npy (float16, Oil and Look-alike)."""


R = paths.RESULTS


MODELS = {  # name -> (module, checkpoint root, seed)
    'b2_revised_s42': ('segformer_b2', R / 'stage3_segformer_b2_shift_revised_256', 42),
    'b2_revised_s43': ('segformer_b2', R / 'stage3_segformer_b2_shift_revised_seed43_256', 43),
    'b2_original_s42': ('segformer_b2', R / 'stage3_segformer_b2_shift_256', 42),
    'deeplabv3plus': ('deeplabv3plus_resnet50', R / 'stage3_deeplabv3plus_resnet50_shift_256', 42),
}


def main_compare_models():
    p = argparse.ArgumentParser(); paths.add_root_argument(p); p.add_argument('--tag', required=True); p.add_argument('--model', required=True, choices=sorted(MODELS)); p.add_argument('--batch', type=int, default=8)
    a = p.parse_args()
    module, root, seed = MODELS[a.model]
    spec = SPECS[module]
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = []
    for fold in range(1, 6):
        st = torch.load(root / f'{spec["name"]}_p256_fold{fold:02d}_seed{seed}' / 'best.pt', map_location='cpu', weights_only=False)
        m = spec['build'](st['model_config']); m.load_state_dict(st['model_state'], strict=True); models.append(m.to(dev).eval())
    grey = np.load(WEB / f'out/{a.tag}_grey_raw.npy'); h, w = grey.shape
    pos = [(y, x) for y in window_starts(h) for x in window_starts(w)]
    prob = np.zeros((5, h, w)); cnt = np.zeros((h, w))
    with torch.inference_mode():
        for s in range(0, len(pos), a.batch):
            ch = pos[s:s + a.batch]
            x = torch.from_numpy(np.stack([grey[y:y + WINDOW, x_:x_ + WINDOW].astype(np.float32) / 255.0 for y, x_ in ch])[:, None]).to(dev)
            mean = 0
            for m in models:
                _, lg = m(x)
                if lg.shape[-1] != WINDOW:
                    lg = torch.nn.functional.interpolate(lg, size=(WINDOW, WINDOW), mode='bilinear', align_corners=False)
                mean = mean + lg.float().softmax(1)
            mean = (mean / len(models)).cpu().numpy()
            for (y, x_), pr in zip(ch, mean):
                prob[:, y:y + WINDOW, x_:x_ + WINDOW] += pr; cnt[y:y + WINDOW, x_:x_ + WINDOW] += 1
    prob = (prob / cnt).astype(np.float32); pred = prob.argmax(0).astype(np.uint8)
    old = np.load(WEB / f'out/{a.tag}_pred_raw.npy'); pred[old == 0] = np.where(pred[old == 0] == 0, 0, pred[old == 0])   # no masking change
    np.save(WEB / f'out/{a.tag}_{a.model}_pred.npy', pred); np.save(WEB / f'out/{a.tag}_{a.model}_prob_oil_la.npy', prob[1:3].astype(np.float16))
    print(a.model, {c: round(float((pred == i).mean()) * 100, 2) for i, c in enumerate(['Sea', 'Oil', 'Look-alike', 'Ship', 'Land'])})



# ----------------------------------------------------------------------------- sensitivity


DOC_sensitivity = """Sensitivity of the HRDA-C prediction to the preprocessing recipe, one parameter at a time, on one fixed raw region.

Example:  python -m sentinel1_scenes.scene_pipeline sensitivity
Reads sentinel1/raw/vv_full.bin (uint16 DN, 25743 x 16709), writes out/sens_<name>_pred.npy (uint8 class map) and out/sensitivity.json.
Baseline: pool 2x2 (about 20 m), 22 grey levels per dB, scene median -> grey 96."""


RAW_W, RAW_H = 25743, 16709


def main_sensitivity():
    REF_DB = json.load(open(WEB / 'out/scene_20m.json'))['scene_median_db_of_DN2']
    X0, Y0, W, H = 3000, 8400, 4998, 3900  # raw-pixel region (the vortex crop of the 20 m scene, x 1500 y 4200 2500x1950)

    # name: (pool, grey levels per dB, level shift in dB of the sea reference)
    VARIANTS = {
        'base': (2, 22.0, 0.0),
        'lv18': (2, 18.0, 0.0),
        'lv26': (2, 26.0, 0.0),
        'pool3': (3, 22.0, 0.0),
        'dark2': (2, 22.0, -2.0),   # scene appears 2 dB darker than the reference, so everything is mapped brighter
        'bright2': (2, 22.0, +2.0),
    }


    def make_grey(raw, pool, levels, shift_db):
        h, w = H // pool * pool, W // pool * pool
        dn = np.asarray(raw[Y0:Y0 + h, X0:X0 + w], dtype=np.float32)
        inten = (dn ** 2).reshape(h // pool, pool, w // pool, pool).mean((1, 3))
        ok = (dn > 0).reshape(h // pool, pool, w // pool, pool).all((1, 3))
        db = 10 * np.log10(np.maximum(inten, 1.0))
        grey = np.clip(96.0 + levels * (db - REF_DB - shift_db), 1, 255)
        grey[~ok] = 0
        return grey.astype(np.uint8)


    (WEB / 'out').mkdir(parents=True, exist_ok=True)
    raw = np.memmap(WEB / 'raw/vv_full.bin', dtype='<u2', mode='r', shape=(RAW_H, RAW_W))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = load_models(paths.RESULTS / 'stage3_segformer_b2_hrda_c_shift_256', [1, 2, 3, 4, 5], device)
    summary = {}
    for name, (pool, levels, shift) in VARIANTS.items():
        grey = make_grey(raw, pool, levels, shift)
        t = time.time()
        pred = predict(models, grey, device, 8).argmax(0).astype(np.uint8)
        np.save(WEB / f'out/sens_{name}_pred.npy', pred)
        px_km2 = (10 * pool / 1000) ** 2
        summary[name] = {'pool': pool, 'levels_per_db': levels, 'shift_db': shift, 'shape': list(pred.shape),
                         'grey_mean': float(grey.mean()), 'seconds': round(time.time() - t),
                         'share_pct': {c: round(float((pred == i).mean()) * 100, 2)
                                       for i, c in enumerate(['Sea', 'Oil', 'Look-alike', 'Ship', 'Land'])},
                         'oil_km2': round(float((pred == 1).sum()) * px_km2, 1),
                         'lookalike_km2': round(float((pred == 2).sum()) * px_km2, 1)}
        print(name, summary[name], flush=True)
        json.dump(summary, open(WEB / 'out/sensitivity.json', 'w'), indent=1)



# ----------------------------------------------------------------------------- compare-sensitivity


DOC_compare_sensitivity = """Compare the sensitivity runs with the baseline: class areas, Oil IoU, how much baseline Oil/Look-alike moves between classes.
pool3 is on a coarser grid: nearest-neighbour upsampled to the 20 m grid."""


OUT = WEB / 'out'


def main_compare_sensitivity():
    summary = json.load(open(OUT / 'sensitivity.json'))
    base = np.load(OUT / 'sens_base_pred.npy')


    def to_base_grid(pred):
        if pred.shape == base.shape:
            return pred
        ys = (np.arange(base.shape[0]) * pred.shape[0] / base.shape[0]).astype(int)
        xs = (np.arange(base.shape[1]) * pred.shape[1] / base.shape[1]).astype(int)
        return pred[np.ix_(ys, xs)]


    def iou(a, b):
        return float((a & b).sum() / max((a | b).sum(), 1))


    rows = []
    for name in summary:
        pred = to_base_grid(np.load(OUT / f'sens_{name}_pred.npy'))
        bo, bl = base == 1, base == 2
        rows.append({
            'variant': name, 'oil_km2': summary[name]['oil_km2'], 'lookalike_km2': summary[name]['lookalike_km2'],
            'oil_iou_vs_base': round(iou(pred == 1, bo), 3),
            'dark_iou_vs_base (Oil|Look-alike)': round(iou((pred == 1) | (pred == 2), bo | bl), 3),
            'base_oil_kept_oil': round(float((pred[bo] == 1).mean()), 3),
            'base_oil_to_lookalike': round(float((pred[bo] == 2).mean()), 3),
            'base_oil_to_sea': round(float((pred[bo] == 0).mean()), 3),
            'base_lookalike_to_oil': round(float((pred[bl] == 1).mean()), 3),
            'pixel_agreement_all_classes': round(float((pred == base).mean()), 3),
        })
    print(json.dumps(rows, indent=1))
    json.dump(rows, open(OUT / 'sensitivity_comparison.json', 'w'), indent=1)

    # side-by-side figure: baseline and each variant (Oil red, Look-alike yellow, Ship blue, Land green, Sea dark)
    colours = np.array([(40, 40, 40), (255, 0, 0), (255, 220, 0), (0, 120, 255), (0, 160, 0)], np.uint8)
    tiles = [colours[to_base_grid(np.load(OUT / f'sens_{n}_pred.npy'))][::3, ::3] for n in summary]
    cols = 3
    rows_img = [np.concatenate(tiles[i:i + cols] + [np.zeros_like(tiles[0])] * (cols - len(tiles[i:i + cols])), 1)
                for i in range(0, len(tiles), cols)]
    Image.fromarray(np.concatenate(rows_img, 0)).save(OUT / 'sensitivity_grid.png')
    print('variants in order:', list(summary))

COMMANDS = {
    'grey': main_grey,
    'annotation-info': main_annotation_info,
    'run': main_run,
    'compare-models': main_compare_models,
    'sensitivity': main_sensitivity,
    'compare-sensitivity': main_compare_sensitivity,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print('usage: python -m sentinel1_scenes.scene_pipeline <command> [options]\n\ncommands:\n  ' + '\n  '.join(COMMANDS))
        raise SystemExit(0 if len(sys.argv) > 1 and sys.argv[1] in ('-h', '--help') else 2)
    command = sys.argv.pop(1)
    sys.argv[0] = f'sentinel1_scenes.scene_pipeline {command}'
    COMMANDS[command]()


if __name__ == '__main__':
    main()
