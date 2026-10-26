"""Case studies and figures of Section 5.6 (raw scenes)

    python -m sentinel1_scenes.case_studies dark-area [scene_dir]     area of the dark patch of the Novorossiysk scene of 2021-08-08 (sigma0 below thresholds)
    python -m sentinel1_scenes.case_studies patch-shrink --tag <tag>  does the size of the patch decide the Look-alike call?
    python -m sentinel1_scenes.case_studies stages-figure --stage compute|plot --tag <tag>
    python -m sentinel1_scenes.case_studies raw-cases-figure [s|m]    three raw cases x SAR + three models (recipe A or B)
"""


import argparse
import glob
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from scipy.interpolate import LinearNDInterpolator

from segmentation import paths
from segmentation.inference.scene import load_models, predict, WINDOW, window_starts
from segmentation.models import SPECS



# ----------------------------------------------------------------------------- dark-area


DOC_dark_area = """Area of the dark patch (sigma0 VV below a threshold, 100 m mean in dB as in the GEE script) in the Novorossiysk scene of 2021-08-08.
Polygon of the GEE script: lon 37.311..37.623, lat 44.513..44.794. Calibrated sigma0 (sigma-nought LUT), thermal noise not subtracted, 10 m pixels.
Writes out/novorossiysk_2021/dark_patch.png and dark_patch.json."""


WEB = paths.PROJECT_ROOT / 'sentinel1'


def main_dark_area():
    sd = sys.argv[1] if len(sys.argv) > 1 else str(WEB / 'case_2021_novorossiysk/scene_210808')
    ann = ET.parse(glob.glob(sd + '/meta/annotation/s1*-vv-*.xml')[0]).getroot()
    cal = ET.parse(glob.glob(sd + '/meta/annotation/calibration/calibration-*-vv-*.xml')[0]).getroot()
    g = np.array([[float(q.findtext(k)) for k in ('line', 'pixel', 'latitude', 'longitude')] for q in ann.iter('geolocationGridPoint')])
    to_line = LinearNDInterpolator(g[:, 2:4], g[:, 0]); to_pix = LinearNDInterpolator(g[:, 2:4], g[:, 1])
    la0, la1, lo0, lo1 = 44.513, 44.794, 37.311, 37.623
    cn = [(la, lo) for la in (la0, la1) for lo in (lo0, lo1)]
    ys = [float(to_line(*c)) for c in cn]; xs = [float(to_pix(*c)) for c in cn]
    Y0, Y1, X0, X1 = int(min(ys)) - 300, int(max(ys)) + 300, int(min(xs)) - 300, int(max(xs)) + 300
    hdr = open(sd + '/vv_full.hdr').read()
    import re
    W_, H_ = int(re.search(r'samples\s*=\s*(\d+)', hdr).group(1)), int(re.search(r'lines\s*=\s*(\d+)', hdr).group(1))
    Y0, X0, Y1, X1 = max(Y0, 0), max(X0, 0), min(Y1, H_), min(X1, W_)
    raw = np.memmap(sd + '/vv_full.bin', dtype='<u2', mode='r', shape=(H_, W_))
    dn = np.asarray(raw[Y0:Y1, X0:X1], np.float32)
    vec = cal.findall('calibrationVectorList/calibrationVector')
    lines = np.array([int(v.findtext('line')) for v in vec], float)
    rows, cols = np.arange(Y0, Y1), np.arange(X0, X1)
    per = np.stack([np.interp(cols, np.array(v.findtext('pixel').split(), float), np.array(v.findtext('sigmaNought').split(), float)) for v in vec])
    idx = np.clip(np.searchsorted(lines, rows) - 1, 0, len(lines) - 2); t = np.clip((rows - lines[idx]) / (lines[idx + 1] - lines[idx]), 0, 1)[:, None]
    lut = (per[idx] * (1 - t) + per[idx + 1] * t).astype(np.float32)
    db = 10 * np.log10(np.maximum(dn ** 2 / lut ** 2, 1e-8)); db[dn == 0] = np.nan
    db = np.where(np.isnan(db), -50, db)
    sm = ndimage.uniform_filter(db, 10)                       # 100 m square mean of the dB values (10 m pixels)
    # geographic mask of the polygon
    Y, X = np.mgrid[Y0:Y1:10, X0:X1:10]
    # approximate polygon membership with the geolocation grid (lat/lon of each coarse pixel)
    to_lat = LinearNDInterpolator(g[:, [1, 0]], g[:, 2]); to_lon = LinearNDInterpolator(g[:, [1, 0]], g[:, 3])
    lat, lon = to_lat(X, Y), to_lon(X, Y)
    inpoly = (lat >= la0) & (lat <= la1) & (lon >= lo0) & (lon <= lo1)
    sm10 = sm[::10, ::10][:inpoly.shape[0], :inpoly.shape[1]]
    dn10 = dn[::10, ::10][:inpoly.shape[0], :inpoly.shape[1]]
    sea = inpoly & (dn10 > 0)
    pix_km2 = 0.01 * 0.01 * 100                                # one coarse sample = 100 m x 100 m = 0.01 km2
    res = {'polygon_km2': float(inpoly.sum() * pix_km2), 'scene': sd}
    for thr in (-20.0, -21.5, -23.0):
        m = sea & (sm10 < thr)
        lab, n = ndimage.label(m); sizes = ndimage.sum(m, lab, range(1, n + 1)) if n else []
        res[f'area_km2_below_{thr}'] = float(m.sum() * pix_km2); res[f'largest_component_km2_below_{thr}'] = float(max(sizes) * pix_km2) if n else 0.0
    m = sea & (sm10 < -21.5)
    res['mean_sigma0_db_inside'] = float(np.nanmean(sm10[m])); res['mean_sigma0_db_outside_sea'] = float(np.nanmean(sm10[sea & ~m & (sm10 > -21.5)]))
    res['median_sigma0_db_outside_sea'] = float(np.median(sm10[sea & ~m]))
    print(json.dumps(res, indent=1)); json.dump(res, open(str(WEB / 'out/novorossiysk_2021/dark_patch.json'), 'w'), indent=1)
    img = np.clip((sm10 + 30) / 25, 0, 1); rgb = np.stack([img] * 3, -1); rgb[m] = np.array([0.9, 0.1, 0.1]) * 0.6 + rgb[m] * 0.4
    Image.fromarray((rgb * 255).astype(np.uint8)).save(str(WEB / 'out/novorossiysk_2021/dark_patch.png'))
    np.save(str(WEB / 'out/novorossiysk_2021/dark_patch_mask.npy'), m)
    np.save(str(WEB / 'out/novorossiysk_2021/dark_patch_origin.npy'), np.array([Y0, X0]))



# ----------------------------------------------------------------------------- patch-shrink


DOC_patch_shrink = """Does the size of the Novorossiysk patch decide the Look-alike call? The grey crop around the patch is shrunk by f x f mean pooling (patch area / f^2,
same darkness); the control smooths the speckle by an f x f mean filter but keeps the size. Three models, five folds averaged.
ADDITIONAL ANALYSIS. Output: sentinel1/out/novorossiysk_2021/patch_shrink_test_<tag>.json"""


R = paths.RESULTS


SINGLES = {'b2_original_s42': ('segformer_b2', R / 'stage3_segformer_b2_shift_256', 42),
           'deeplabv3plus': ('deeplabv3plus_resnet50', R / 'stage3_deeplabv3plus_resnet50_shift_256', 42)}


def load_single(name, dev):
    module, root, seed = SINGLES[name]
    spec = SPECS[module]
    models = []
    for fold in range(1, 6):
        st = torch.load(root / f'{spec["name"]}_p256_fold{fold:02d}_seed{seed}' / 'best.pt', map_location='cpu', weights_only=False)
        m = spec['build'](st['model_config']); m.load_state_dict(st['model_state'], strict=True); models.append(m.to(dev).eval())
    return models


@torch.inference_mode()
def predict_single(models, img, dev, batch=8):
    h, w = img.shape
    pos = [(y, x) for y in window_starts(h) for x in window_starts(w)]
    prob = np.zeros((5, h, w)); cnt = np.zeros((h, w))
    for s in range(0, len(pos), batch):
        ch = pos[s:s + batch]
        x = torch.from_numpy(np.stack([img[y:y + WINDOW, x_:x_ + WINDOW].astype(np.float32) / 255.0 for y, x_ in ch])[:, None]).to(dev)
        mean = 0
        for m in models:
            _, lg = m(x)
            if lg.shape[-1] != WINDOW:
                lg = torch.nn.functional.interpolate(lg, size=(WINDOW, WINDOW), mode='bilinear', align_corners=False)
            mean = mean + lg.float().softmax(1)
        mean = (mean / len(models)).cpu().numpy()
        for (y, x_), pr in zip(ch, mean):
            prob[:, y:y + WINDOW, x_:x_ + WINDOW] += pr; cnt[y:y + WINDOW, x_:x_ + WINDOW] += 1
    return prob / cnt


OUT = WEB / 'out'


def main_patch_shrink():
    p = argparse.ArgumentParser(); paths.add_root_argument(p); p.add_argument('--tag', default='c210808s'); a = p.parse_args()
    grey = np.load(OUT / f'{a.tag}_grey_raw.npy')
    M = np.load(OUT / 'novorossiysk_2021/dark_patch_mask_grid20m.npy')
    ys, xs = np.nonzero(M); m = 350
    y0, y1, x0, x1 = max(ys.min() - m, 0), min(ys.max() + m, grey.shape[0]), max(xs.min() - m, 0), min(xs.max() + m, grey.shape[1])
    g, mk = grey[y0:y1, x0:x1].astype(np.float32), M[y0:y1, x0:x1]
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hrda = load_models(paths.RESULTS / 'stage3_segformer_b2_hrda_c_shift_256', [1, 2, 3, 4, 5], dev)
    singles = {n: load_single(n, dev) for n in SINGLES}

    def pool(arr, f):
        h, w = (arr.shape[0] // f) * f, (arr.shape[1] // f) * f
        return arr[:h, :w].reshape(h // f, f, w // f, f).mean(axis=(1, 3))

    variants = [('size x1', g, mk)]
    for f in (2, 3, 4):
        variants.append((f'size /{f * f} (pool {f})', pool(g, f), pool(mk.astype(np.float32), f) > 0.5))
    for f in (2, 4):
        variants.append((f'smoothed {f}, same size', ndimage.uniform_filter(g, f), mk))
    res = {}
    for name, img, mask in variants:
        u8 = np.clip(np.round(img), 0, 255).astype(np.uint8)
        row = {'apparent_patch_km2_at_20m_pixel': round(float(mask.sum() * 4e-4), 1), 'crop_px': list(img.shape)}
        preds = {'HRDA-C': predict(hrda, u8, dev, 8).argmax(0)}
        for n, ms in singles.items():
            preds[n] = predict_single(ms, u8, dev).argmax(0)
        for n, pr in preds.items():
            row[n] = {'oil_pct': round(100 * float((pr[mask] == 1).mean()), 1), 'la_pct': round(100 * float((pr[mask] == 2).mean()), 1)}
        res[name] = row; print(name, json.dumps(row), flush=True)
    json.dump(res, open(OUT / f'novorossiysk_2021/patch_shrink_test_{a.tag}.json', 'w'), indent=1)



# ----------------------------------------------------------------------------- stages-figure


DOC_stages_figure = """Stage-by-stage figure of the scene pipeline on a saved crop (default: Novorossiysk, tag c210808w).
Reads sentinel1/out/<tag>_{grey_raw,pred_raw,prob_oil_la_raw}.npy (made by `scene_pipeline run --save-probs`), runs HRDA-C (five folds) on one window.
A grey crop with window grid, B overlap count, C-E one window (detail, 512 px context, prediction), F Look-alike probability, G final classes."""


COLOURS = np.array([(0, 0, 0), (0, 255, 255), (255, 0, 0), (153, 76, 0), (0, 153, 0)], np.uint8)


def main_stages_figure():
    p = argparse.ArgumentParser(); paths.add_root_argument(p); p.add_argument('--tag', default='c210808w'); p.add_argument('--stage', choices=('compute', 'plot'), required=True, help='compute: model on one window (venv with torch); plot: figure (python with matplotlib)')
    p.add_argument('--center', help='y,x in grey pixels: use the grid window whose centre is nearest (default: most contrasty mostly-Look-alike window)')
    p.add_argument('--out', type=Path, default=WEB / str(WEB / 'out/novorossiysk_2021/pipeline_stages.png'))
    a = p.parse_args()
    grey = np.load(WEB / f'out/{a.tag}_grey_raw.npy'); pred = np.load(WEB / f'out/{a.tag}_pred_raw.npy')
    prob = np.load(WEB / f'out/{a.tag}_prob_oil_la_raw.npy').astype(np.float32)
    h, w = grey.shape
    ys, xs = window_starts(h), window_starts(w)
    count = np.zeros((h, w), np.uint8)
    for y in ys:
        for x in xs:
            count[y:y + WINDOW, x:x + WINDOW] += 1
    # example window: among windows that are mostly Look-alike (mean P > 0.5), the one with the most grey-level contrast (patch edge)
    cand = [(y, x) for y in ys for x in xs if prob[1, y:y + WINDOW, x:x + WINDOW].mean() > 0.5]
    best = max(cand, key=lambda t: grey[t[0]:t[0] + WINDOW, t[1]:t[1] + WINDOW].astype(np.float32).std())
    if a.center:
        cy, cx = (int(v) for v in a.center.split(','))
        best = min(((y, x) for y in ys for x in xs), key=lambda t: (t[0] + WINDOW / 2 - cy) ** 2 + (t[1] + WINDOW / 2 - cx) ** 2)
    y0, x0 = best
    cache = WEB / f'out/{a.tag}_stages_window.npz'
    if a.stage == 'compute':
        import torch
        from segmentation.context import reduce2, reflect_crop
        from segmentation.inference.scene import load_models
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        models = load_models(paths.RESULTS / 'stage3_segformer_b2_hrda_c_shift_256', [1, 2, 3, 4, 5], dev)
        detail = grey[y0:y0 + WINDOW, x0:x0 + WINDOW].astype(np.float32) / 255.0
        off = (512 - WINDOW) // 2
        context = reduce2(reflect_crop(grey, y0 - off, x0 - off, 512).astype(np.float32) / 255.0)
        with torch.inference_mode():
            batch = torch.from_numpy(np.stack([detail, context])[None]).to(dev)
            pw = sum(m(batch)[1].float().softmax(1) for m in models).div(len(models))[0].cpu().numpy()
        win_pred = pw.argmax(0)

        np.savez(cache, detail=detail, context=context, pw=pw, best=np.array(best))
        print('saved', cache, 'window', best, 'max P(Oil) in window', round(float(pw[1].max()), 3))
        raise SystemExit
    z = np.load(cache); detail, context, pw = z['detail'], z['context'], z['pw']; win_pred = pw.argmax(0); off = (512 - WINDOW) // 2
    y0, x0 = (int(v) for v in z['best'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Rectangle
    plt.rcParams.update({'font.size': 8})
    fig = plt.figure(figsize=(6.5, 7.6))
    gs = GridSpec(3, 6, figure=fig, height_ratios=[1.15, 0.8, 1.15], hspace=0.12, wspace=0.12)
    axes = {'A': fig.add_subplot(gs[0, 0:3]), 'B': fig.add_subplot(gs[0, 3:6]), 'C': fig.add_subplot(gs[1, 0:2]), 'D': fig.add_subplot(gs[1, 2:4]),
            'E': fig.add_subplot(gs[1, 4:6]), 'F': fig.add_subplot(gs[2, 0:3]), 'G': fig.add_subplot(gs[2, 3:6])}
    ax = axes['A']; ax.imshow(grey, cmap='gray', vmin=0, vmax=255)
    for y in ys: ax.axhline(y, color='c', lw=0.25, alpha=0.5)
    for x in xs: ax.axvline(x, color='c', lw=0.25, alpha=0.5)
    ax.add_patch(Rectangle((x0, y0), WINDOW, WINDOW, fill=False, ec='red', lw=1.2))
    ax.add_patch(Rectangle((x0 - off, y0 - off), 512, 512, fill=False, ec='yellow', lw=0.9, ls='--'))
    im = axes['B'].imshow(count, cmap='viridis')
    cb = fig.colorbar(im, cax=axes['B'].inset_axes([0.55, 0.07, 0.4, 0.035]), orientation='horizontal'); cb.set_label('windows per pixel', color='white', fontsize=7)
    cb.ax.tick_params(colors='white', labelsize=7); cb.ax.xaxis.set_ticks_position('top'); cb.ax.xaxis.set_label_position('top')
    axes['C'].imshow(detail, cmap='gray', vmin=0, vmax=0.5)
    axes['D'].imshow(context, cmap='gray', vmin=0, vmax=0.5)
    axes['E'].imshow(COLOURS[win_pred])
    im = axes['F'].imshow(prob[1], cmap='magma', vmin=0, vmax=1)
    cb = fig.colorbar(im, cax=axes['F'].inset_axes([0.55, 0.07, 0.4, 0.035]), orientation='horizontal'); cb.set_label('P(Look-alike)', color='white', fontsize=7)
    cb.ax.tick_params(colors='white', labelsize=7); cb.ax.xaxis.set_ticks_position('top'); cb.ax.xaxis.set_label_position('top')
    axes['G'].imshow(COLOURS[pred])
    for k, ax in axes.items():
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.02, 0.97, k, transform=ax.transAxes, va='top', ha='left', fontsize=10, fontweight='bold', color='white',
                bbox=dict(fc='black', ec='none', alpha=0.6, pad=1.5))
    from matplotlib.patches import Patch
    axes['G'].legend(handles=[Patch(fc=COLOURS[i] / 255, label=c) for i, c in enumerate(['Sea', 'Oil', 'Look-alike', 'Ship', 'Land'])], loc='lower left', fontsize=6, framealpha=0.8, ncol=1)
    fig.savefig(a.out, dpi=300, bbox_inches='tight')
    print('saved', a.out, 'window', best, 'grid', len(ys), 'x', len(xs), '=', len(ys) * len(xs), 'windows; example window class shares',
          {c: round(float((win_pred == i).mean()), 3) for i, c in enumerate(['Sea', 'Oil', 'Look-alike', 'Ship', 'Land'])},
          'max P(Oil) in window', round(float(pw[1].max()), 3))



# ----------------------------------------------------------------------------- raw-cases-figure


DOC_raw_cases_figure = """Chapter 5 figure: three raw Sentinel-1 cases (rows) x SAR + three models (columns), sea-normalised runs.
Reads sentinel1/out/<tag>_grey_raw.npy and the prediction maps of `scene_pipeline run` / `scene_pipeline compare-models`.
Rows: Novorossiysk 8 Aug 2021 (c210808s), Kashagan 25 Mar 2022 (c220325s), Peshnoi 6 Jul 2023 (c230706s).
Pass 'm' as the argument for recipe B (tags ending in m, file ch5_raw_scene_cases_recipe_b); default 's' is recipe A."""


def main_raw_cases_figure():
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    OUT = WEB / 'out'
    FIG = paths.RESULTS / 'thesis_figures'
    COLOURS = np.array([(0, 0, 0), (0, 255, 255), (255, 0, 0), (153, 76, 0), (0, 153, 0)], np.uint8)
    PRED = ['pred_raw', 'b2_original_s42_pred', 'deeplabv3plus_pred']

    m = np.load(OUT / 'novorossiysk_2021/dark_patch_mask.npy'); Y0, X0 = np.load(OUT / 'novorossiysk_2021/dark_patch_origin.npy')
    ii, jj = np.nonzero(m)
    gy, gx = (Y0 + 10 * ii - 450) // 2, (X0 + 10 * jj - 3400) // 2
    cy, cx = int((gy.min() + gy.max()) / 2), int((gx.min() + gx.max()) / 2)
    half_nov = int(max(gy.max() - gy.min(), gx.max() - gx.min()) / 2) + 80
    SUF = sys.argv[1] if len(sys.argv) > 1 else 's'
    cases = [(f'c210808{SUF}', cy, cx, half_nov), (f'c220325{SUF}', 1250, 1250, 300), (f'c230706{SUF}', 1370, 1300, 560)]
    NAME = 'ch5_raw_scene_cases' if SUF == 's' else 'ch5_raw_scene_cases_recipe_b'

    plt.rcParams.update({'font.size': 8})
    fig, axes = plt.subplots(3, 4, figsize=(6.5, 5.0), gridspec_kw=dict(wspace=0.03, hspace=0.03))
    letters = iter('ABCDEFGHIJKL')
    for r, (tag, y, x, h) in enumerate(cases):
        grey = np.load(OUT / f'{tag}_grey_raw.npy')
        ys, xs = slice(max(y - h, 0), min(y + h, grey.shape[0])), slice(max(x - h, 0), min(x + h, grey.shape[1]))
        axes[r, 0].imshow(grey[ys, xs], cmap='gray', vmin=0, vmax=255)
        for c, name in enumerate(PRED):
            axes[r, c + 1].imshow(COLOURS[np.load(OUT / f'{tag}_{name}.npy')][ys, xs])
        for c in range(4):
            a = axes[r, c]; a.set_xticks([]); a.set_yticks([])
            a.text(0.03, 0.96, next(letters), transform=a.transAxes, va='top', ha='left', fontsize=9, fontweight='bold', color='white',
                   bbox=dict(fc='black', ec='none', alpha=0.6, pad=1.2))
        # 5 km scale bar (one pooled pixel = 20 m -> 250 px)
        axes[r, 0].plot([20, 270], [ (ys.stop - ys.start) - 25] * 2, color='white', lw=2)
    fig.savefig(FIG / f'{NAME}.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG / f'{NAME}.pdf', bbox_inches='tight')
    print('saved', FIG / f'{NAME}.png', 'novorossiysk half-size', half_nov)

COMMANDS = {
    'dark-area': main_dark_area,
    'patch-shrink': main_patch_shrink,
    'stages-figure': main_stages_figure,
    'raw-cases-figure': main_raw_cases_figure,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print('usage: python -m sentinel1_scenes.case_studies <command> [options]\n\ncommands:\n  ' + '\n  '.join(COMMANDS))
        raise SystemExit(0 if len(sys.argv) > 1 and sys.argv[1] in ('-h', '--help') else 2)
    command = sys.argv.pop(1)
    sys.argv[0] = f'sentinel1_scenes.case_studies {command}'
    COMMANDS[command]()


if __name__ == '__main__':
    main()
