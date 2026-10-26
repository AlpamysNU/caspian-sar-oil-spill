"""Reference statistics of the M4D scenes behind the preprocessing recipe

The recipe of the raw-scene application (2x2 pooling, about 22 grey levels per dB, sea level) is estimated from these measurements, not taken from documentation.
texture, scale-cv and scale-skew compare M4D sea patches with a raw GRD (need GDAL and <project root>/sentinel1/raw/vv_full.tif).
"""


import glob
import json
import sys

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.ndimage import binary_erosion, gaussian_filter
from scipy.stats import skew

from segmentation import paths



# ----------------------------------------------------------------------------- texture


DOC_texture = """Compare speckle texture of M4D training scenes (Sea pixels) with the raw GRD VV at several downsampling factors."""


WEB = paths.PROJECT_ROOT / 'sentinel1'


def main_texture():
    from osgeo import gdal
    D = str(paths.DATASET / 'train') + '/'

    def acf(x, valid):
        """lag-1..3 autocorrelation of the residual after subtracting a local mean, along x (range-like) and y."""
        x = x.astype(np.float64)
        r = x - gaussian_filter(x, 12)
        out = {}
        for ax, name in ((1, 'x'), (0, 'y')):
            vals = []
            for lag in (1, 2, 3):
                a = np.take(r, range(0, r.shape[ax] - lag), axis=ax); b = np.take(r, range(lag, r.shape[ax]), axis=ax)
                va = np.take(valid, range(0, r.shape[ax] - lag), axis=ax) & np.take(valid, range(lag, r.shape[ax]), axis=ax)
                vals.append(float((a * b)[va].mean() / (r[valid] ** 2).mean()))
            out[name] = [round(v, 2) for v in vals]
        return out

    print('--- M4D training scenes (Sea pixels, 8-bit values)')
    res = []
    for f in sorted(glob.glob(D + 'images/*.jpg'))[::25][:30]:
        im = np.array(Image.open(f).convert('L')).astype(np.float32)
        lab = np.array(Image.open(f.replace('images', 'labels_1D').replace('.jpg', '.png')))
        sea = binary_erosion(lab == 0, iterations=15)
        if sea.mean() < 0.6: continue
        a = acf(im, sea); res.append((a['x'], a['y'], im[sea].std()))
    print('mean lag1-3 x:', np.mean([r[0] for r in res], 0).round(2), ' y:', np.mean([r[1] for r in res], 0).round(2),
          ' sea std (8-bit):', np.mean([r[2] for r in res]).round(1), ' n scenes', len(res))

    print('--- raw GRD VV, 20*log10(DN), sea crops')
    ds = gdal.Open(str(WEB / 'raw/vv_full.tif')); b = ds.GetRasterBand(1)
    for name, (x0, y0) in {'bottom sea': (13000, 12500), 'left sea': (1500, 2500)}.items():
        crop = b.ReadAsArray(x0, y0, 3750, 1950).astype(np.float32)
        for k in (1, 2, 3):
            c = crop[:crop.shape[0] // k * k, :crop.shape[1] // k * k].reshape(crop.shape[0] // k, k, crop.shape[1] // k, k).mean((1, 3))
            db = 20 * np.log10(np.maximum(c, 1))
            a = acf(db, np.ones_like(db, bool))
            print(f'{name:11s} mean-pool {k}x ({10*k} m): x {a["x"]}  y {a["y"]}  std dB {db.std():.2f}')



# ----------------------------------------------------------------------------- scale-cv


DOC_scale_cv = """Is the training intensity scale linear (std grows with mean) or logarithmic (std constant)? Sea patches only."""


def main_scale_cv():
    from osgeo import gdal
    D = str(paths.DATASET / 'train') + '/'

    def patch_stats(img, sea, p=32):
        m, s = [], []
        for y in range(0, img.shape[0] - p, p):
            for x in range(0, img.shape[1] - p, p):
                if sea[y:y+p, x:x+p].all():
                    w = img[y:y+p, x:x+p].astype(np.float64); m.append(w.mean()); s.append(w.std())
        return np.array(m), np.array(s)

    M, S = [], []
    for f in sorted(glob.glob(D + 'images/*.jpg'))[::12]:
        im = np.array(Image.open(f).convert('L')); lab = np.array(Image.open(f.replace('images', 'labels_1D').replace('.jpg', '.png')))
        m, s = patch_stats(im, lab == 0)
        keep = (m > 20) & (m < 235)
        M += list(m[keep]); S += list(s[keep])
    M, S = np.array(M), np.array(S)
    print(f'M4D sea patches: n={len(M)}  corr(mean,std)={np.corrcoef(M,S)[0,1]:.2f}  std/mean (CV) median={np.median(S/M):.2f}')
    for lo, hi in ((20, 60), (60, 100), (100, 140), (140, 180), (180, 235)):
        k = (M >= lo) & (M < hi)
        if k.sum() > 20: print(f'  patch mean {lo:3d}-{hi:3d}: n={k.sum():5d} median std={np.median(S[k]):5.1f}  CV={np.median(S[k]/M[k]):.2f}')

    # raw GRD: same statistic for amplitude (DN), intensity and dB, after 2x2 mean-pooling, on sea crops
    ds = gdal.Open(str(WEB / 'raw/vv_full.tif')); b = ds.GetRasterBand(1)
    crop = b.ReadAsArray(1500, 2500, 3750, 1950).astype(np.float32)
    c = crop[:1950, :3750].reshape(975, 2, 1875, 2).mean((1, 3))
    for name, v in (('amplitude DN', c), ('dB', 20*np.log10(np.maximum(c, 1)))):
        pm = [v[y:y+32, x:x+32].mean() for y in range(0, 940, 32) for x in range(0, 1840, 32)]
        ps = [v[y:y+32, x:x+32].std() for y in range(0, 940, 32) for x in range(0, 1840, 32)]
        print(f'raw {name:12s}: CV median {np.median(np.array(ps)/np.array(pm)):.3f}  std median {np.median(ps):.2f}')



# ----------------------------------------------------------------------------- scale-skew


DOC_scale_skew = None


def main_scale_skew():
    from osgeo import gdal
    D = str(paths.DATASET / 'train') + '/'
    def stats(img, ok, lo=None, hi=None, p=32):
        cv, sk = [], []
        for y in range(0, img.shape[0]-p, p):
            for x in range(0, img.shape[1]-p, p):
                if ok is None or ok[y:y+p, x:x+p].all():
                    w = img[y:y+p, x:x+p].astype(np.float64)
                    if lo is not None and not (lo < w.mean() < hi): continue
                    cv.append(w.std()/w.mean()); sk.append(skew(w.ravel()))
        return np.median(cv), np.median(sk), len(cv)
    cvs = []
    for f in sorted(glob.glob(D+'images/*.jpg'))[::12]:
        im = np.array(Image.open(f).convert('L')); lab = np.array(Image.open(f.replace('images','labels_1D').replace('.jpg','.png')))
        cvs.append(stats(im, lab == 0, 60, 140)[:2])
    print('M4D sea patches mean 60-140: CV %.2f skew %.2f' % tuple(np.nanmedian(np.array(cvs, float), 0)))
    ds = gdal.Open(str(WEB / 'raw/vv_full.tif')); b = ds.GetRasterBand(1)
    for nm, (x0, y0) in {'bottom sea': (13000, 12500), 'left sea': (1500, 2500)}.items():
        crop = b.ReadAsArray(x0, y0, 3750, 1950).astype(np.float64)
        for k in (1, 2):
            c = crop.reshape(1950//k, k, 3750//k, k).mean((1, 3)) if k > 1 else crop
            for tag, v in (('amplitude', c), ('intensity', c**2)):
                cv, sk, n = stats(v, None)
                print(f'{nm:11s} {k}x pool {tag:9s}: CV {cv:.2f} skew {sk:.2f}')



# ----------------------------------------------------------------------------- speckle


DOC_speckle = """Speckle level of open sea: M4D train images (windows of 64x64 px with only Sea labels, at least 40 px from any other class) against the grey crops of the
raw Sentinel-1 scenes. Statistics per window after removing a plane: std of the grey level (divided by the assumed 22 levels/dB gives dB) and the lag-1 autocorrelation
along rows and columns. ADDITIONAL ANALYSIS (recipe check for the raw-scene application)."""


def main_speckle():
    R = str(paths.DATASET / 'train') + '/'
    O = str(WEB / 'out') + '/'
    B = 64

    def stats(win):
        y, x = np.mgrid[:B, :B]; A = np.stack([np.ones(B * B), y.ravel(), x.ravel()], 1)
        c, *_ = np.linalg.lstsq(A, win.ravel(), rcond=None); r = (win.ravel() - A @ c).reshape(B, B)
        ac = lambda a, b: float((a * b).mean() / (r.var() + 1e-9))
        return r.std(), ac(r[:, :-1], r[:, 1:]), ac(r[:-1], r[1:])

    def windows(img, ok, step=B, limit=None, rng=np.random.default_rng(0)):
        out = []
        for y in range(0, img.shape[0] - B, step):
            for x in range(0, img.shape[1] - B, step):
                if ok[y:y + B, x:x + B].all() and 25 < img[y:y + B, x:x + B].mean() < 235:
                    out.append(stats(img[y:y + B, x:x + B].astype(np.float64)))
        return out

    res = {}
    m4 = []
    for f in sorted(glob.glob(R + 'labels_1D/*.png'))[::4]:
        lab = np.asarray(Image.open(f)); img = np.asarray(Image.open(R + 'images/' + f.split('/')[-1].replace('.png', '.jpg')).convert('L'))
        ok = ~ndimage.binary_dilation(lab != 0, iterations=40) & (img > 0)
        m4 += windows(img, ok, step=96)
    a = np.array(m4); res['M4D sea'] = {'windows': len(a), 'std_grey_median': round(float(np.median(a[:, 0])), 2), 'std_grey_p10_p90': [round(float(np.percentile(a[:, 0], q)), 2) for q in (10, 90)],
                                       'lag1_row': round(float(np.median(a[:, 1])), 3), 'lag1_col': round(float(np.median(a[:, 2])), 3)}
    for tag in ('c220325s', 'c230706s'):
        g = np.load(O + f'{tag}_grey_raw.npy'); pred = np.load(O + f'{tag}_pred_raw.npy')
        ok = ~ndimage.binary_dilation(pred != 0, iterations=40) & (g > 0)
        w = np.array(windows(g[::1, ::1], ok, step=192))
        res[f'{tag} (recipe: pool 2, 22 levels/dB)'] = {'windows': len(w), 'std_grey_median': round(float(np.median(w[:, 0])), 2), 'std_grey_p10_p90': [round(float(np.percentile(w[:, 0], q)), 2) for q in (10, 90)],
                                                        'lag1_row': round(float(np.median(w[:, 1])), 3), 'lag1_col': round(float(np.median(w[:, 2])), 3)}
        for f in (2, 3, 4):
            h, wd = (g.shape[0] // f) * f, (g.shape[1] // f) * f
            gp = g[:h, :wd].reshape(h // f, f, wd // f, f).mean(axis=(1, 3)); okp = ok[:h, :wd].reshape(h // f, f, wd // f, f).all(axis=(1, 3))
            w = np.array(windows(gp, okp, step=96))
            res[f'{tag} + extra pool {f}'] = {'windows': len(w), 'std_grey_median': round(float(np.median(w[:, 0])), 2), 'lag1_row': round(float(np.median(w[:, 1])), 3), 'lag1_col': round(float(np.median(w[:, 2])), 3)}
    print(json.dumps(res, indent=1)); json.dump(res, open(O + 'speckle_match.json', 'w'), indent=1)



# ----------------------------------------------------------------------------- grey-by-class


DOC_grey_by_class = """Grey level of Oil and Look-alike pixels in the M4D train images (jpg grey), overall and for large components (> 20 km2 at an assumed 20 m pixel),
relative to the sea median of the same scene. ADDITIONAL ANALYSIS for the Novorossiysk case."""


def main_grey_by_class():
    R = str(paths.DATASET / 'train') + '/'
    PX_KM2 = 0.02 * 0.02
    acc = {k: [] for k in ('Oil', 'Look-alike', 'Oil_large', 'Look-alike_large', 'Sea_median')}
    for f in sorted(glob.glob(R + 'labels_1D/*.png')):
        lab = np.asarray(Image.open(f)); img = np.asarray(Image.open(R + 'images/' + f.split('/')[-1].replace('.png', '.jpg')).convert('L'))
        sea = lab == 0
        if sea.sum() < 1000: continue
        acc['Sea_median'].append(float(np.median(img[sea])))
        for c, name in ((1, 'Oil'), (2, 'Look-alike')):
            m = lab == c
            if not m.any(): continue
            acc[name].append(img[m][::7])
            cc, n = ndimage.label(m, structure=np.ones((3, 3)))
            big = np.isin(cc, [i + 1 for i, s in enumerate(ndimage.sum(m, cc, range(1, n + 1))) if s * PX_KM2 > 20])
            if big.any(): acc[name + '_large'].append(img[big][::7])
    res = {'sea_median_grey_median_over_scenes': float(np.median(acc['Sea_median']))}
    for k in ('Oil', 'Look-alike', 'Oil_large', 'Look-alike_large'):
        v = np.concatenate(acc[k]); res[k] = {'pixels_sampled': int(v.size), 'grey_mean': round(float(v.mean()), 1), 'grey_median': float(np.median(v)),
            'share_grey_le_10_pct': round(100 * float((v <= 10).mean()), 1), 'share_grey_le_30_pct': round(100 * float((v <= 30).mean()), 1)}
    print(json.dumps(res, indent=1)); json.dump(res, open(str(WEB / str(WEB / 'out/novorossiysk_2021/') ) + 'm4d_grey_by_class.json', 'w'), indent=1)



# ----------------------------------------------------------------------------- component-sizes


DOC_component_sizes = """Component sizes of Oil and Look-alike in the M4D train labels (1,002 scenes, 650x1250 px), 8-connected, with a 20 m pixel assumed
(the pixel size is estimated, not given by the dataset), to compare with the 73 km2 Novorossiysk patch."""


def main_component_sizes():
    R = str(paths.DATASET / 'train/labels_1D') + '/'
    PX_KM2 = 0.02 * 0.02
    out = {1: [], 2: []}
    for f in sorted(glob.glob(R + '*.png')):
        a = np.asarray(Image.open(f))
        for c in (1, 2):
            lab, n = ndimage.label(a == c, structure=np.ones((3, 3)))
            if n:
                out[c] += (ndimage.sum(np.ones_like(lab), lab, range(1, n + 1)) * PX_KM2).tolist()
    res = {}
    for c, name in ((1, 'Oil'), (2, 'Look-alike')):
        s = np.array(out[c])
        res[name] = {'components': len(s), 'median_km2': float(np.median(s)), 'p90_km2': float(np.percentile(s, 90)), 'max_km2': float(s.max()),
                     **{f'n_gt_{t}_km2': int((s > t).sum()) for t in (5, 20, 50)}}
    print(json.dumps(res, indent=1))
    json.dump(res, open(str(WEB / str(WEB / 'out/novorossiysk_2021/') ) + 'm4d_component_sizes.json', 'w'), indent=1)

COMMANDS = {
    'texture': main_texture,
    'scale-cv': main_scale_cv,
    'scale-skew': main_scale_skew,
    'speckle': main_speckle,
    'grey-by-class': main_grey_by_class,
    'component-sizes': main_component_sizes,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print('usage: python -m sentinel1_scenes.m4d_reference <command> [options]\n\ncommands:\n  ' + '\n  '.join(COMMANDS))
        raise SystemExit(0 if len(sys.argv) > 1 and sys.argv[1] in ('-h', '--help') else 2)
    command = sys.argv.pop(1)
    sys.argv[0] = f'sentinel1_scenes.m4d_reference {command}'
    COMMANDS[command]()


if __name__ == '__main__':
    main()
