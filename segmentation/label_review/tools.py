"""Label-review helpers (E4): one entry point, one sub-command per tool.

    python -m segmentation.label_review.tools <command> [options]     (every command accepts --project-root)

The napari editor itself is review_napari.py; the validated write path (revised copy only, change log, overlays) is review_logic.py.
The original dataset is never written by any of these commands.

  suspicious     rank training scenes by two label-review suspicions (displaced Oil mask, unlabelled ship candidates)
  ship-sheet     numbered ship-candidate map and contact sheet for one scene
  add-ships      add hand-picked ships to the revised copy of one scene
  snap           one-scene snap proposal for Oil components (threshold between mask and surrounding sea)
  lattice        test whether a group of Ship objects is a regular lattice (wind turbines) or a random scatter (vessels)
  review-sheet   blind review sheet for one scene: SAR, SAR with the original mask, zoomed crops
  change-masks   colour masks of only the changed pixels of the revised scenes
  export-rgb     colour (RGB) copy of the revised class maps for viewing
  click-ships    matplotlib tool to click bright point targets that carry no Ship label
  apply-clicks   turn the clicks into Ship masks in the revised copy
"""


import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi, ndimage
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree

from segmentation import paths
from segmentation.label_review import review_logic as rl



# ----------------------------------------------------------------------------- suspicious


DOC_suspicious = """Rank training scenes by two label-review suspicions (E4 helper, 2026-10-05). Read-only on the dataset; writes one CSV.

1. Displaced Oil mask: for every Oil component (>= --min-oil px) the mask is shifted by up to --max-shift px and the shift that puts it on the darkest
   region is searched (FFT cross-correlation of the mask with the local darkness map). Reported: best shift, darkness contrast at zero shift and at the best
   shift. A displaced mask has a large gain and a shift of several pixels; a mask on empty sea has a contrast near zero at every shift.
2. Unlabelled ship candidates: bright compact blobs (smoothed high-pass image above --z-ship noise sigmas, 3..--max-ship px) farther than 6 px from any
   Ship label and not inside Land.

Scenes are processed one at a time (small RAM). Only train/ is read; the official test set is not touched."""


OIL, suspicious_SHIP, suspicious_LAND = 1, 3, 4


def darkness(img):
    """Relative darkness against the large-scale background after light smoothing (speckle suppression)."""
    g = ndi.gaussian_filter(img, 2.0)
    bg = ndi.gaussian_filter(img, 25.0)
    return (bg - g) / (bg + 1.0)


def oil_shift(dark, comp, max_shift, margin=8):
    """Best (dy, dx, score_best, score_zero) for one Oil component; score = mean darkness inside the shifted mask minus the crop mean."""
    ys, xs = np.nonzero(comp)
    pad = max_shift + margin
    y0, y1 = max(0, ys.min() - pad), min(comp.shape[0], ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(comp.shape[1], xs.max() + 1 + pad)
    d = dark[y0:y1, x0:x1]
    d = d - d.mean()
    m = comp[y0:y1, x0:x1].astype(np.float32)
    corr = fftconvolve(d, m[::-1, ::-1], mode='same') / m.sum()      # corr[y, x]: mean darkness inside the mask moved so that its centre is at (y, x)
    cy, cx = (np.array(corr.shape) - 1) / 2.0
    cy, cx = int(round(cy)), int(round(cx))
    win = corr[max(0, cy - max_shift): cy + max_shift + 1, max(0, cx - max_shift): cx + max_shift + 1]
    oy, ox = max(0, cy - max_shift), max(0, cx - max_shift)
    by, bx = np.unravel_index(np.argmax(win), win.shape)
    return by + oy - cy, bx + ox - cx, float(win[by, bx]), float(corr[cy, cx])


def ship_candidates(img, label, z_ship, max_ship):
    hp = img - ndi.gaussian_filter(img, 15.0)
    s = ndi.gaussian_filter(hp, 1.0)
    valid = label != suspicious_LAND
    sigma = 1.4826 * np.median(np.abs(s[valid] - np.median(s[valid]))) + 1e-6
    z = s / sigma
    cand = (z > z_ship) & valid
    lab, n = ndi.label(cand)
    if n == 0:
        return 0, 0.0
    near_ship = ndi.binary_dilation(label == suspicious_SHIP, iterations=6)
    near_land = ndi.binary_dilation(label == suspicious_LAND, iterations=3)
    area = ndi.sum(cand, lab, range(1, n + 1))
    peak = ndi.maximum(z, lab, range(1, n + 1))
    hit_ship = ndi.maximum(near_ship, lab, range(1, n + 1))
    hit_land = ndi.maximum(near_land, lab, range(1, n + 1))
    keep = (area >= 3) & (area <= max_ship) & (hit_ship == 0) & (hit_land == 0)
    return int(keep.sum()), float(peak[keep].max()) if keep.any() else 0.0


def main_suspicious():
    ap = argparse.ArgumentParser()
    paths.add_root_argument(ap)
    ap.add_argument('--root', type=Path, default=paths.DATASET)
    ap.add_argument('--out', type=Path, default=paths.RESULTS / 'label_review/suspicion/suspicion_scores.csv')
    ap.add_argument('--scenes', type=Path, help='optional text file with scene names (img_0156 per line); default: all train scenes')
    ap.add_argument('--min-oil', type=int, default=150)
    ap.add_argument('--max-shift', type=int, default=30)
    ap.add_argument('--z-ship', type=float, default=7.0)
    ap.add_argument('--max-ship', type=int, default=600)
    a = ap.parse_args()
    names = [l.strip().replace('train_', '') for l in open(a.scenes) if l.strip()] if a.scenes else sorted(p.stem for p in (a.root / 'train/images').glob('*.jpg'))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in names:
        img = np.asarray(Image.open(a.root / f'train/images/{n}.jpg').convert('L')).astype(np.float32)
        lab = np.asarray(Image.open(a.root / f'train/labels_1D/{n}.png'))
        dark = darkness(img)
        comps, k = ndi.label(lab == OIL)
        worst = dict(dy=0, dx=0, mag=0.0, s0=0.0, sb=0.0, gain=0.0, area=0)
        n_comp = 0
        for i in range(1, k + 1):
            comp = comps == i
            area = int(comp.sum())
            if area < a.min_oil:
                continue
            n_comp += 1
            dy, dx, sb, s0 = oil_shift(dark, comp, a.max_shift)
            gain = sb - s0
            if gain > worst['gain']:
                worst = dict(dy=dy, dx=dx, mag=float(np.hypot(dy, dx)), s0=s0, sb=sb, gain=gain, area=area)
        n_ship, zmax = ship_candidates(img, lab, a.z_ship, a.max_ship)
        rows.append(dict(scene=n, oil_px=int((lab == OIL).sum()), oil_components=n_comp, worst_comp_px=worst['area'], shift_dy=worst['dy'], shift_dx=worst['dx'],
                         shift_mag=round(worst['mag'], 1), contrast_zero=round(worst['s0'], 4), contrast_best=round(worst['sb'], 4), gain=round(worst['gain'], 4),
                         ship_candidates=n_ship, ship_zmax=round(zmax, 1), labelled_ship_px=int((lab == suspicious_SHIP).sum())))
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'{len(rows)} scenes -> {a.out}')



# ----------------------------------------------------------------------------- ship-sheet


DOC_ship_sheet = """Numbered ship-candidate map and contact sheet for one scene (E4 assisted-review helper; restored 2026-10-06 from the rule of the `suspicious` command).

python -m segmentation.label_review.tools ship-sheet <scene> [--z 5.0] [--max-px 600] [--max-n 40]   (second pass for ships the first pass missed: --z 3.6)

Candidate rule: high-pass image (minus Gaussian 15) smoothed with sigma 1, divided by the robust noise sigma (MAD, Land excluded); blobs above --z of
3..--max-px px, not within 6 px of a Ship label and not within 3 px of Land. The Ship/Land mask is the revised label if the scene has a revised file,
otherwise the original. Writes <scene>_ship_candidates_map.png (whole scene, numbered boxes) and ..._sheet.png (crops, same numbers, with y,x of the peak
to pass to `add-ships`) to results/label_review/suspicion/. Read-only on the dataset; adds nothing to the revised copy."""


ship_sheet_SHIP, ship_sheet_LAND = 3, 4


def candidates(img, label, z_thr, max_px):
    hp = img - ndi.gaussian_filter(img, 15.0)
    s = ndi.gaussian_filter(hp, 1.0)
    valid = label != ship_sheet_LAND
    sigma = 1.4826 * np.median(np.abs(s[valid] - np.median(s[valid]))) + 1e-6
    z = s / sigma
    lab, n = ndi.label((z > z_thr) & valid)
    out = []
    if n == 0:
        return out
    near_ship = ndi.binary_dilation(label == ship_sheet_SHIP, iterations=6)
    near_land = ndi.binary_dilation(label == ship_sheet_LAND, iterations=3)
    idx = range(1, n + 1)
    area = ndi.sum(lab > 0, lab, idx)
    peak = ndi.maximum(z, lab, idx)
    pos = ndi.maximum_position(z, lab, idx)
    hs = ndi.maximum(near_ship, lab, idx)
    hl = ndi.maximum(near_land, lab, idx)
    for i in range(n):
        if 3 <= area[i] <= max_px and hs[i] == 0 and hl[i] == 0:
            out.append(dict(y=int(pos[i][0]), x=int(pos[i][1]), z=float(peak[i]), area=int(area[i])))
    return sorted(out, key=lambda c: -c['z'])


def stretch(a):
    lo, hi = np.percentile(a, [1, 99.7])
    return (np.clip((a - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8)


def main_ship_sheet():
    ap = argparse.ArgumentParser()
    paths.add_root_argument(ap)
    ap.add_argument('scene')
    ap.add_argument('--root', type=Path, default=paths.DATASET)
    ap.add_argument('--revised', type=Path, default=paths.RESULTS / 'label_review/revised_manual/labels_1D_revised')
    ap.add_argument('--out', type=Path, default=paths.RESULTS / 'label_review/suspicion')
    ap.add_argument('--z', type=float, default=5.0)
    ap.add_argument('--max-px', type=int, default=600)
    ap.add_argument('--max-n', type=int, default=40)
    ap.add_argument('--exclude', type=int, nargs='*', default=[], help='candidate numbers (of the full list at this --z) to drop from the picture; the other candidates keep their numbers')
    ap.add_argument('--half', type=int, default=32, help='half size of each crop in the sheet')
    a = ap.parse_args()
    n = a.scene.replace('train_', '')
    img = np.asarray(Image.open(a.root / f'train/images/{n}.jpg').convert('L')).astype(np.float32)
    rev = a.revised / f'train_{n}.png'
    if not rev.exists():
        rev = a.revised / f'{n}.png'
    src = rev if rev.exists() else a.root / f'train/labels_1D/{n}.png'
    label = np.asarray(Image.open(src))
    cands = candidates(img, label, a.z, a.max_px)
    if len(cands) > a.max_n:
        print(f'WARNING: {len(cands)} candidates found, only the {a.max_n} strongest are shown (raise --max-n)')
    cands = cands[:a.max_n]
    ids = [i for i in range(1, len(cands) + 1) if i not in set(a.exclude)]
    print(f'{n}: label from {src.name} ({"revised" if rev.exists() else "original"}), {len(ids)} candidates at z > {a.z}')
    for i in ids:
        c = cands[i - 1]
        print(f'  #{i:2d} y{c["y"]} x{c["x"]} z {c["z"]:.1f} area {c["area"]}')
    a.out.mkdir(parents=True, exist_ok=True)
    tag = ('' if a.z == 5.0 else f'_z{a.z:g}') + ('_filtered' if a.exclude else '')

    g = stretch(img)
    rgb = np.stack([g] * 3, -1)
    rgb[label == ship_sheet_SHIP] = (rgb[label == ship_sheet_SHIP] * 0.4 + np.array([150, 90, 30]) * 0.6).astype(np.uint8)
    m = Image.fromarray(rgb)
    d = ImageDraw.Draw(m)
    for i in ids:
        c = cands[i - 1]
        d.rectangle([c['x'] - 10, c['y'] - 10, c['x'] + 10, c['y'] + 10], outline=(255, 255, 0))
        d.text((c['x'] + 12, c['y'] - 12), str(i), fill=(255, 255, 0))
    m.save(a.out / f'{n}_ship_candidates{tag}_map.png')

    if ids:
        h, cols = a.half, 6
        size = 2 * h * 2  # crops are shown at 2x
        rows = (len(ids) + cols - 1) // cols
        sheet = Image.new('RGB', (cols * size, rows * (size + 14)), (20, 20, 20))
        pad = np.pad(g, h, mode='edge')
        for k_, i in enumerate(ids):
            c = cands[i - 1]
            crop = pad[c['y']:c['y'] + 2 * h, c['x']:c['x'] + 2 * h]
            t = Image.fromarray(crop).convert('RGB').resize((size, size), Image.NEAREST)
            td = ImageDraw.Draw(t)
            td.rectangle([size // 2 - 12, size // 2 - 12, size // 2 + 12, size // 2 + 12], outline=(255, 255, 0))
            r, k = divmod(k_, cols)
            sheet.paste(t, (k * size, r * (size + 14) + 14))
            ImageDraw.Draw(sheet).text((k * size + 2, r * (size + 14) + 1), f'#{i} y{c["y"]} x{c["x"]} z{c["z"]:.0f}', fill=(255, 255, 255))
        sheet.save(a.out / f'{n}_ship_candidates{tag}_sheet.png')
    print('written to', a.out)



# ----------------------------------------------------------------------------- add-ships


DOC_add_ships = """Add hand-picked ships to the revised copy of one scene (E4 helper, 2026-10-06).

python -m segmentation.label_review.tools add-ships <scene> y,x [y,x ...] [--note ...]   (positions of the bright objects; the mask is the connected region of the high-pass image above
max(3.0, --thr-frac (0.4) x local peak) sigma within 14 px of the peak (peak searched within 6 px of the given position), +1 px, only where the revised map is Sea, or Sea/Look-alike/Land with --over-lookalike/--over-land).
The change is saved through review_logic.save (same validation, overlay and log row as the napari tool; category ship_added)."""


def main_add_ships():
    ap = argparse.ArgumentParser()
    paths.add_root_argument(ap)
    ap.add_argument('scene')
    ap.add_argument('points', nargs='+', help='y,x of each ship')
    ap.add_argument('--root', type=Path, default=paths.DATASET)
    ap.add_argument('--out', type=Path, default=paths.RESULTS / 'label_review/revised_manual')
    ap.add_argument('--over-lookalike', action='store_true', help='also write Ship over Look-alike pixels (author decision, img_0871)')
    ap.add_argument('--over-land', action='store_true', help='also write Ship over Land pixels (ships labelled as Land, author decision, img_0216)')
    ap.add_argument('--thr-frac', type=float, default=0.4, help='mask threshold as a fraction of the local peak z (default 0.4; raise for bright objects with halo or streaks)')
    ap.add_argument('--note', default='author-picked ships added from a candidate sheet')
    a = ap.parse_args()
    n = a.scene.replace('train_', '')
    image, original, revised = rl.load_scene(a.root, n, a.out)
    img = image.astype(np.float32)
    hp = img - ndi.gaussian_filter(img, 15.0)
    s = ndi.gaussian_filter(hp, 1.0)
    valid = original != 4
    z = s / (1.4826 * np.median(np.abs(s[valid] - np.median(s[valid]))))
    yy, xx = np.mgrid[:original.shape[0], :original.shape[1]]
    new = revised.copy()
    for p in a.points:
        y, x = (int(v) for v in p.split(','))
        py, px = np.unravel_index(np.argmax(np.where((yy - y) ** 2 + (xx - x) ** 2 <= 36, z, -1e9)), z.shape)
        t = max(3.0, a.thr_frac * z[py, px])
        lab, _ = ndi.label((z > t) & ((yy - py) ** 2 + (xx - px) ** 2 <= 14 ** 2))
        if lab[py, px] == 0:
            raise SystemExit(f'ship at (y{y},x{x}): peak z {z[py, px]:.1f} is below the threshold {t:.1f}, nothing added (noisy sigma; use a smoothed-intensity mask for bright objects); nothing saved')
        m = ndi.binary_fill_holes(ndi.binary_dilation(lab == lab[py, px], structure=np.ones((3, 3)), iterations=1)) & np.isin(revised, (0, 2) * a.over_lookalike + (0,) * (not a.over_lookalike) + (4,) * a.over_land)
        new[m] = 3
        print(f'ship at (y{y},x{x}): peak (y{py},x{px}) z {z[py, px]:.1f} thr {t:.1f} area {int(m.sum())} px')
    ch, w = rl.save(a.root, a.out, n, image, original, new, 'ship_added', a.note, a.out / 'revision_log_manual.csv')
    print('changed vs original', ch, rl.transitions(original, new), w)



# ----------------------------------------------------------------------------- snap


DOC_snap = """One-scene snap proposal for the label review (E4 helper, 2026-10-05). Read-only on the dataset and on the revised labels.

For every component of --cls (default 1 = Oil) with at least --min-px pixels the threshold is set per component half-way between the median smoothed
brightness inside the mask and the median in a ring 8-20 px outside it (Sea pixels only); the snapped component is the connected dark region of the
smoothed image under that threshold, restricted to --grow px around the original mask, with holes filled. Components below --min-px are left unchanged.
Output: <out>/<scene>_snap.png (original cyan, snap magenta, full image and one crop per big component), the thresholds on stdout and, with
--save-proposal, a full class map <out>/proposals/<scene>.png for 'Use proposal' in review_napari.py (original labels, snapped class replaced; ships,
land and other classes are never overwritten). --record DECISION [--note ...] appends the decision of the author to <out>/decisions.csv.
The proposal is a starting point; nothing is written into the revised labels by this script."""


def edge(m):
    return m & ~ndi.binary_erosion(m)


def snap_class(img, lab, cls, min_px, sigma, grow):
    g = ndi.gaussian_filter(img, sigma)
    comps, k = ndi.label(lab == cls)
    snap, info = np.zeros(lab.shape, bool), []
    for i in range(1, k + 1):
        c = comps == i
        if c.sum() < min_px:
            continue
        ring = ndi.binary_dilation(c, iterations=20) & ~ndi.binary_dilation(c, iterations=8) & (lab == 0)
        if not ring.any():
            continue
        inside, bg = float(np.median(g[c])), float(np.median(g[ring]))
        t = (inside + bg) / 2
        s = (g < t) & ndi.binary_dilation(c, iterations=grow)
        dl, _ = ndi.label(s)
        ids = np.unique(dl[c & s])
        s = ndi.binary_fill_holes(np.isin(dl, ids[ids > 0]))
        dist = ndi.distance_transform_edt(~edge(s))[edge(c)]
        info.append(dict(comp=i, y=int(np.nonzero(c)[0].mean()), x=int(np.nonzero(c)[1].mean()), orig=int(c.sum()), snap=int(s.sum()), inside=inside, bg=bg,
                         thr=t, iou=float((s & c).sum() / (s | c).sum()), dist_med=float(np.median(dist)), dist_p90=float(np.percentile(dist, 90))))
        snap |= s
    return snap, info


def main_snap():
    ap = argparse.ArgumentParser()
    paths.add_root_argument(ap)
    ap.add_argument('scene')
    ap.add_argument('--root', type=Path, default=paths.DATASET)
    ap.add_argument('--out', type=Path, default=paths.RESULTS / 'label_review/suspicion')
    ap.add_argument('--cls', type=int, default=1)
    ap.add_argument('--min-px', type=int, default=300)
    ap.add_argument('--sigma', type=float, default=2.0)
    ap.add_argument('--grow', type=int, default=8)
    ap.add_argument('--save-proposal', action='store_true')
    ap.add_argument('--record', help='author decision: no_change | accepted | corrected | rejected')
    ap.add_argument('--note', default='')
    ap.add_argument('--log-no-change', action='store_true', help='also add a no_change row to revised_manual/revision_log_manual.csv (marks the scene as reviewed)')
    a = ap.parse_args()
    n = a.scene.replace('train_', '')
    img = np.asarray(Image.open(a.root / f'train/images/{n}.jpg').convert('L')).astype(np.float32)
    lab = np.asarray(Image.open(a.root / f'train/labels_1D/{n}.png'))
    snap, info = snap_class(img, lab, a.cls, a.min_px, a.sigma, a.grow)
    for d in info:
        print(f"comp {d['comp']} (y{d['y']},x{d['x']}): inside {d['inside']:.0f} bg {d['bg']:.0f} thr {d['thr']:.0f} | orig {d['orig']} snap {d['snap']} "
              f"IoU {d['iou']:.2f} boundary dist median {d['dist_med']:.1f} p90 {d['dist_p90']:.1f}")
    a.out.mkdir(parents=True, exist_ok=True)
    rgb = np.stack([img] * 3, -1).astype(np.uint8)
    rgb[edge(lab == a.cls)] = (0, 255, 255)
    rgb[edge(snap)] = (255, 0, 255)
    Image.fromarray(rgb).save(a.out / f'{n}_snap.png')
    if a.save_proposal:
        new = lab.copy()
        comps, k = ndi.label(lab == a.cls)
        for i in range(1, k + 1):
            if (comps == i).sum() >= a.min_px:
                new[comps == i] = 0
        new[snap & (new == 0)] = a.cls
        (a.out / 'proposals').mkdir(exist_ok=True)
        Image.fromarray(new.astype(np.uint8)).save(a.out / 'proposals' / f'{n}.png')
        print('proposal saved:', a.out / 'proposals' / f'{n}.png', 'differs from the original in', int((new != lab).sum()), 'px')
    if a.record:
        f = a.out / 'decisions.csv'
        new_file = not f.exists()
        with open(f, 'a', newline='') as o:
            w = csv.writer(o)
            if new_file:
                w.writerow(['utc', 'scene', 'cls', 'sigma', 'grow', 'min_px', 'thresholds', 'median_iou', 'decision', 'note'])
            w.writerow([datetime.now(timezone.utc).isoformat(timespec='seconds'), n, a.cls, a.sigma, a.grow, a.min_px,
                        ';'.join(f"c{d['comp']}:{d['thr']:.0f}" for d in info), f"{np.median([d['iou'] for d in info]):.2f}" if info else '', a.record, a.note])
        print('decision recorded in', f)
    if a.log_no_change:
        lf = paths.RESULTS / 'label_review/revised_manual/revision_log_manual.csv'
        with open(lf, 'a', newline='') as o:
            csv.writer(o).writerow([datetime.now(timezone.utc).isoformat(timespec='seconds'), f'train_{n}', 'no_change', 0, '', a.note, ''])
        print('no_change row added to', lf.name)



# ----------------------------------------------------------------------------- lattice


DOC_lattice = """Is a group of Ship objects a regular lattice (fixed structures such as wind turbines) or a random scatter (vessels)?

For a scene, takes the Ship components of the revised label (or only the pixels changed against the original, --added-only),
and tests the point pattern of their centroids:
  1. nearest-neighbour distance (median, coefficient of variation) and the Clark-Evans ratio (1 = random, > 1 = regular);
  2. the dominant neighbour vector: all neighbour difference vectors within --radius px are folded to one half plane and the
     densest bin is taken as the lattice basis v1; v2 is the densest vector not parallel to v1;
  3. the share of points that have a partner at +-v1 (and +-v2) within --tol px, against the same share for random points
     (Monte Carlo, same count, same bounding box, --trials draws): a lattice gives a share far above the random range.
Read-only: writes only a CSV line per scene and an optional picture. Pixel size is not known from the dataset, distances are in px."""


def centroids(mask):
    lab, n = ndi.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return np.zeros((0, 2))
    return np.array(ndi.center_of_mass(mask, lab, range(1, n + 1)))  # (y, x)


def basis(points, radius, tol):
    tree = cKDTree(points)
    vecs = []
    for i, j in tree.query_pairs(radius):
        v = points[j] - points[i]
        if v[1] < 0 or (v[1] == 0 and v[0] < 0):
            v = -v
        vecs.append(v)
    vecs = np.array(vecs)
    if len(vecs) == 0:
        return None, None, 0
    # densest vector: count of vectors within tol of each candidate vector
    vt = cKDTree(vecs)
    counts = np.array([len(x) for x in vt.query_ball_point(vecs, tol)])
    v1 = vecs[counts.argmax()]
    v1 = vecs[vt.query_ball_point(v1, tol)].mean(0)
    # second basis vector: densest, not parallel to v1 (|cross| / |v1||v| > 0.3)
    cross = np.abs(vecs[:, 0] * v1[1] - vecs[:, 1] * v1[0]) / (np.linalg.norm(vecs, axis=1) * np.linalg.norm(v1) + 1e-9)
    ok = cross > 0.3
    if not ok.any():
        return v1, None, int(counts.max())
    idx = np.where(ok)[0][counts[ok].argmax()]
    v2 = vecs[vt.query_ball_point(vecs[idx], tol)].mean(0)
    return v1, v2, int(counts.max())


def partner_share(points, v, tol):
    if v is None:
        return 0.0
    tree = cKDTree(points)
    hit = 0
    for p in points:
        a = tree.query_ball_point(p + v, tol)
        b = tree.query_ball_point(p - v, tol)
        hit += bool(a or b)
    return hit / len(points)


def main_lattice():
    ap = argparse.ArgumentParser()
    paths.add_root_argument(ap)
    ap.add_argument('scenes', nargs='+', help='img_XXXX')
    ap.add_argument('--root', type=Path, default=paths.DATASET)
    ap.add_argument('--rev', type=Path, default=paths.RESULTS / 'label_review/revised_manual/labels_1D_revised')
    ap.add_argument('--added-only', action='store_true', help='only Ship pixels that differ from the original label')
    ap.add_argument('--radius', type=float, default=400)
    ap.add_argument('--tol', type=float, default=4.0)
    ap.add_argument('--trials', type=int, default=200)
    ap.add_argument('--out', type=Path, default=paths.RESULTS / 'label_review/suspicion/lattice_check.csv')
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    rows = []
    for s in a.scenes:
        rev = np.asarray(Image.open(a.rev / f'{s}.png'))
        orig = np.asarray(Image.open(a.root / f'train/labels_1D/{s}.png'))
        ship = rev == 3
        if a.added_only:
            ship &= orig != 3
        pts = centroids(ship)
        n = len(pts)
        h, w = rev.shape
        if n < 6:
            print(s, 'too few objects', n)
            continue
        d, _ = cKDTree(pts).query(pts, k=2)
        nn = d[:, 1]
        y0, x0 = pts.min(0)
        y1, x1 = pts.max(0)
        area = max((y1 - y0) * (x1 - x0), 1.0)
        clark = nn.mean() / (0.5 / np.sqrt(n / area))
        v1, v2, c1 = basis(pts, a.radius, a.tol)
        s1, s2 = partner_share(pts, v1, a.tol), partner_share(pts, v2, a.tol)
        # random reference: same number of points in the same box; the same procedure incl. fitting the basis
        r1, r2 = [], []
        for _ in range(a.trials):
            rp = np.column_stack([rng.uniform(y0, y1, n), rng.uniform(x0, x1, n)])
            rv1, rv2, _ = basis(rp, a.radius, a.tol)
            r1.append(partner_share(rp, rv1, a.tol))
            r2.append(partner_share(rp, rv2, a.tol))
        r1, r2 = np.array(r1), np.array(r2)
        ang = lambda v: float(np.degrees(np.arctan2(v[0], v[1])) % 180) if v is not None else float('nan')
        norm = lambda v: float(np.linalg.norm(v)) if v is not None else float('nan')
        row = dict(scene=s, n=n, nn_median=float(np.median(nn)), nn_cv=float(nn.std() / nn.mean()), clark_evans=float(clark),
                   v1_len=norm(v1), v1_angle=ang(v1), share1=s1, rand1_p95=float(np.percentile(r1, 95)), rand1_max=float(r1.max()),
                   v2_len=norm(v2), v2_angle=ang(v2), share2=s2, rand2_p95=float(np.percentile(r2, 95)), rand2_max=float(r2.max()))
        rows.append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})
    if rows:
        import csv
        a.out.parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, 'w', newline='') as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)



# ----------------------------------------------------------------------------- review-sheet


DOC_review_sheet = """Blind review sheet for one scene (2026-10-03): SAR | SAR with the original mask (Oil cyan, Look-alike red, Ship orange, Land green) |
zoomed crops of the labelled regions (SAR left, mask overlay right). No model prediction is shown (first, blind pass)."""


PAL = {1: (0, 255, 255), 2: (255, 0, 0), 3: (255, 150, 0), 4: (0, 200, 0)}


def overlay(gray, gt, alpha=0.55):
    a = np.stack([gray] * 3, -1).astype(float)
    for k, c in PAL.items():
        m = gt == k
        a[m] = (1 - alpha) * a[m] + alpha * np.array(c)
    return a.astype(np.uint8)


def contour(gray, gt):
    a = np.stack([gray] * 3, -1).copy()
    for k, c in PAL.items():
        m = gt == k
        edge = m & ~ndimage.binary_erosion(m)
        a[edge] = c
    return a


def sheet(root, scene, out, crops=3, zoom=4, pad=30):
    gray = np.asarray(Image.open(root / f'train/images/{scene}.jpg').convert('L'))
    gt = np.asarray(Image.open(root / f'train/labels_1D/{scene}.png'))
    full = Image.new('RGB', (2 * gray.shape[1] + 10, gray.shape[0]), 'white')
    full.paste(Image.fromarray(np.stack([gray] * 3, -1)), (0, 0))
    full.paste(Image.fromarray(contour(gray, gt)), (gray.shape[1] + 10, 0))
    lab, n = ndimage.label(gt > 0, structure=np.ones((3, 3)))
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    tiles = []
    for i in np.argsort(-sizes[1:])[:crops] + 1:
        ys, xs = np.where(lab == i)
        y0, y1, x0, x1 = max(ys.min() - pad, 0), min(ys.max() + pad, gray.shape[0]), max(xs.min() - pad, 0), min(xs.max() + pad, gray.shape[1])
        a, b = gray[y0:y1, x0:x1], overlay(gray, gt)[y0:y1, x0:x1]
        w, h = a.shape[1] * zoom, a.shape[0] * zoom
        t = Image.new('RGB', (2 * w + 6, h), 'white')
        t.paste(Image.fromarray(np.stack([a] * 3, -1)).resize((w, h), Image.NEAREST), (0, 0))
        t.paste(Image.fromarray(b).resize((w, h), Image.NEAREST), (w + 6, 0))
        tiles.append(t)
    W = max([full.width] + [t.width for t in tiles])
    H = full.height + sum(t.height + 8 for t in tiles) + 24
    canvas = Image.new('RGB', (W, H), 'white')
    d = ImageDraw.Draw(canvas)
    d.text((4, 4), f'{scene}   classes px: ' + ', '.join(f'{n}={int((gt == k).sum())}' for k, n in ((1, "oil"), (2, "look-alike"), (3, "ship"), (4, "land"))), fill='black')
    canvas.paste(full, (0, 20))
    y = 20 + full.height + 8
    for t in tiles:
        canvas.paste(t, (0, y)); y += t.height + 8
    canvas.save(out)


def main_review_sheet():
    ap = argparse.ArgumentParser()
    paths.add_root_argument(ap)
    ap.add_argument('scenes', nargs='+')
    ap.add_argument('--root', type=Path, default=paths.DATASET)
    ap.add_argument('--out', type=Path, default=paths.RESULTS / 'label_review/sheets_pilot')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for s in a.scenes:
        sheet(a.root, s.removeprefix('train_'), a.out / f'{s.removeprefix("train_")}.png')



# ----------------------------------------------------------------------------- change-masks


DOC_change_masks = """Colour masks of ONLY the changed pixels of the manually revised scenes (review of 2026-10-05).

For every scene in <out>/labels_1D_revised/ writes <out>/change_masks/<scene>.png with two panels:
left = changed pixels on grey, right = the same on the SAR image. Colour = the NEW class of the changed pixel, in the dataset's label colours:
sea black, oil cyan, look-alike red, ship brown, land green. Changed pixels are dilated by --grow px for display only
(small ships are a few pixels wide); the label files are never touched. Read-only on the dataset and on the revised labels."""


# the dataset's own RGB label colours (checked on labels/*.png); changed-to-sea pixels are therefore black, shown on a grey background
change_masks_COLOURS = {0: (0, 0, 0), 1: (0, 255, 255), 2: (255, 0, 0), 3: (153, 76, 0), 4: (0, 153, 0)}


BACKGROUND = (110, 110, 110)


def main_change_masks():
    ap = argparse.ArgumentParser()
    paths.add_root_argument(ap)
    ap.add_argument('--root', type=Path, default=paths.DATASET)
    ap.add_argument('--out', type=Path, default=paths.RESULTS / 'label_review/revised_manual')
    ap.add_argument('--grow', type=int, default=3, help='display-only dilation of the changed pixels in px')
    a = ap.parse_args()
    dst = a.out / 'change_masks'
    dst.mkdir(exist_ok=True)
    for f in sorted((a.out / 'labels_1D_revised').glob('*.png')):
        name = f.stem
        image = np.asarray(Image.open(a.root / f'train/images/{name}.jpg').convert('RGB'))
        original = np.asarray(Image.open(a.root / f'train/labels_1D/{name}.png'))
        revised = np.asarray(Image.open(f))
        changed = revised != original
        mask = np.full(image.shape, BACKGROUND, np.uint8)
        painted = np.zeros(changed.shape, bool)
        for cls, rgb in change_masks_COLOURS.items():
            sel = changed & (revised == cls)
            if a.grow:
                sel = ndi.binary_dilation(sel, iterations=a.grow) & ~(ndi.binary_dilation(changed & (revised != cls), iterations=a.grow) & ~sel) | sel
            mask[sel] = rgb
            painted |= sel
        over = image.copy()
        over[painted] = (0.35 * over[painted] + 0.65 * mask[painted]).astype(np.uint8)
        sep = np.full((image.shape[0], 6, 3), 255, np.uint8)
        Image.fromarray(np.concatenate([mask, sep, over], axis=1)).save(dst / f'{name}.png')
        print(name, int(changed.sum()), 'px changed')



# ----------------------------------------------------------------------------- export-rgb


DOC_export_rgb = """Colour (RGB) copy of the revised class maps for viewing in any image viewer (E4 helper, 2026-10-06).

labels_1D_revised/<scene>.png holds class numbers 0-4 (looks almost black in a normal viewer). This writes <out>/labels_RGB_revised/<scene>.png in the
dataset's own label colours (sea black, oil cyan, look-alike red, ship brown, land green) for every revised scene; existing files in that folder are
regenerated from the 1D maps. Read-only on the dataset and on the 1D maps."""


export_rgb_COLOURS = np.array([(0, 0, 0), (0, 255, 255), (255, 0, 0), (153, 76, 0), (0, 153, 0)], np.uint8)


def main_export_rgb():
    ap = argparse.ArgumentParser()
    paths.add_root_argument(ap)
    ap.add_argument('--out', type=Path, default=paths.RESULTS / 'label_review/revised_manual')
    a = ap.parse_args()
    dst = a.out / 'labels_RGB_revised'
    dst.mkdir(exist_ok=True)
    files = sorted((a.out / 'labels_1D_revised').glob('*.png'))
    for f in files:
        Image.fromarray(export_rgb_COLOURS[np.asarray(Image.open(f))]).save(dst / f.name)
    print(len(files), 'files ->', dst)



# ----------------------------------------------------------------------------- click-ships


DOC_click_ships = """Manual ship marking (label review, 2026-10-04): click bright point targets that carry no Ship label.

Usage:  python3 click_ships.py scenes.txt [--out results/label_review/ship_clicks.csv]
scenes.txt: one scene per line (img_0877 or train_img_0877). Window controls: the matplotlib toolbar zoom/pan work as usual (switch the
zoom/pan tool OFF before clicking). Keys: left click = add a ship at the cursor, u = undo last click, n = save and next scene,
p = previous scene, q = save and quit. The original mask is drawn as thin contours (Oil cyan, Look-alike red, Ship orange, Land green);
existing Ship labels are orange, your clicks are yellow crosses. Already finished scenes are skipped (--redo to revisit).
The output CSV (scene_id, x, y in original pixel coordinates) is turned into masks by apply_ship_clicks.py."""


COL = {1: (0, 1, 1), 2: (1, 0, 0), 3: (1, 0.6, 0), 4: (0, 0.8, 0)}


def contours(gt):
    rgba = np.zeros(gt.shape + (4,))
    for k, c in COL.items():
        m = gt == k
        edge = m & ~ndimage.binary_erosion(m)
        rgba[edge] = (*c, 1)
    return rgba


def load_clicks(path):
    out = {}
    if path.exists():
        for r in csv.DictReader(open(path)):
            if r['x'] != '':
                out.setdefault(r['scene_id'], []).append((int(r['x']), int(r['y'])))
            else:
                out.setdefault(r['scene_id'], [])
    return out


def save_clicks(path, clicks):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['scene_id', 'x', 'y'])
        for s, pts in clicks.items():
            if not pts:
                w.writerow([s, '', ''])   # scene reviewed, no ship to add
            for x, y in pts:
                w.writerow([s, x, y])


def main_click_ships():
    ap = argparse.ArgumentParser(description=DOC_click_ships, formatter_class=argparse.RawDescriptionHelpFormatter)
    paths.add_root_argument(ap)
    ap.add_argument('scenes', type=Path)
    ap.add_argument('--root', type=Path, default=paths.DATASET)
    ap.add_argument('--out', type=Path, default=paths.RESULTS / 'label_review/ship_clicks.csv')
    ap.add_argument('--redo', action='store_true')
    a = ap.parse_args()
    scenes = [('train_' + l.strip().removeprefix('train_')) for l in a.scenes.read_text().split() if l.strip()]
    clicks = load_clicks(a.out)
    todo = [s for s in scenes if a.redo or s not in clicks]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    i = 0
    while 0 <= i < len(todo):
        s = todo[i]
        name = s.removeprefix('train_')
        gray = np.asarray(Image.open(a.root / f'train/images/{name}.jpg').convert('L'))
        gt = np.asarray(Image.open(a.root / f'train/labels_1D/{name}.png'))
        pts = list(clicks.get(s, []))
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.imshow(gray, cmap='gray', interpolation='nearest')
        ax.imshow(contours(gt), interpolation='nearest')
        marks, = ax.plot([p[0] for p in pts], [p[1] for p in pts], 'y+', ms=14, mew=2)
        state = {'move': 0}

        def title():
            ax.set_title(f'{s}  [{i + 1}/{len(todo)}]  ships clicked: {len(pts)}   (click=add, u=undo, n=next, p=prev, q=quit)')
            marks.set_data([p[0] for p in pts], [p[1] for p in pts]); fig.canvas.draw_idle()

        def on_click(ev):
            if ev.inaxes is ax and ev.button == 1 and fig.canvas.toolbar.mode == '' and ev.xdata is not None:
                pts.append((int(round(ev.xdata)), int(round(ev.ydata)))); title()

        def on_key(ev):
            if ev.key == 'u' and pts:
                pts.pop(); title()
            elif ev.key in ('n', 'p', 'q'):
                state['move'] = {'n': 1, 'p': -1, 'q': 99}[ev.key]; plt.close(fig)

        fig.canvas.mpl_connect('button_press_event', on_click)
        fig.canvas.mpl_connect('key_press_event', on_key)
        title(); plt.show()
        clicks[s] = pts
        save_clicks(a.out, clicks)
        if state['move'] == 99 or state['move'] == 0:
            break
        i += state['move']
    print('saved', a.out)



# ----------------------------------------------------------------------------- apply-clicks


DOC_apply_clicks = """Turn manual ship clicks into Ship masks in a REVISED copy of the 1D labels (label review, 2026-10-04).

Reads ship_clicks.csv (scene_id, x, y). For each click: the bright core around the click (pixels of the 3x3-smoothed image above local median + --frac x (peak - median), where the peak is >= --min-level and
>= 50 above the 41x41 local median, 8-connected to the peak, within --radius px), dilated by one pixel, becomes class 3 (Ship) where the original label is Sea (0), or Land (4)
when that Land component is small (< --small-land px, a mislabelled target rather than a coast); Oil and Look-alike pixels are never overwritten. The
original dataset is never touched: changed scenes are written to --out/labels_1D_revised/<name>.png (only changed scenes) and every click is logged in
--out/ship_revision_log.csv (scene, x, y, pixels added from Sea, from Land, area, skipped reason). Dry-run by default; use --write to save masks."""


CONN8 = np.ones((3, 3), int)


def ship_core(img, x, y, radius=8, frac=0.5, min_level=170, min_contrast=50, max_area=400):
    h, w = img.shape
    yy, xx = np.ogrid[:h, :w]
    circle = (yy - y) ** 2 + (xx - x) ** 2 <= radius ** 2
    sm = ndimage.uniform_filter(img.astype(float), 3)
    if not circle.any():
        return None
    peak = sm[circle].max()
    y0, y1, x0, x1 = max(y - 20, 0), min(y + 21, h), max(x - 20, 0), min(x + 21, w)
    med = np.median(sm[y0:y1, x0:x1])
    if peak < min_level or peak - med < min_contrast:
        return None
    py, px = np.unravel_index(np.argmax(np.where(circle, sm, -1)), sm.shape)
    lab, _ = ndimage.label((sm >= med + frac * (peak - med)) & circle, structure=CONN8)
    core = lab == lab[py, px]
    core = ndimage.binary_dilation(core, structure=CONN8)
    return core if core.sum() <= max_area else None


def apply(img, gt, clicks, small_land=1500, **kw):
    out, rows = gt.copy(), []
    land_lab, nl = ndimage.label(gt == 4, structure=CONN8)
    land_size = np.bincount(land_lab.ravel(), minlength=nl + 1)
    for x, y in clicks:
        core = ship_core(img, x, y, **kw)
        if core is None:
            rows.append((x, y, 0, 0, 0, 'no bright core'))
            continue
        from_sea = core & (out == 0)
        land_ids = np.unique(land_lab[core & (gt == 4)])
        small = [i for i in land_ids if i > 0 and land_size[i] < small_land]
        from_land = core & np.isin(land_lab, small) if small else np.zeros_like(core)
        for i in small:   # a whole small Land component is a mislabelled target
            from_land |= land_lab == i
        out[from_sea] = 3
        out[from_land] = 3
        rows.append((x, y, int(from_sea.sum()), int(from_land.sum()), int(core.sum()), ''))
    return out, rows


def main_apply_clicks():
    ap = argparse.ArgumentParser(description=DOC_apply_clicks)
    paths.add_root_argument(ap)
    ap.add_argument('--clicks', type=Path, default=paths.RESULTS / 'label_review/ship_clicks.csv')
    ap.add_argument('--root', type=Path, default=paths.DATASET)
    ap.add_argument('--out', type=Path, default=paths.RESULTS / 'label_review/revised')
    ap.add_argument('--radius', type=int, default=8)
    ap.add_argument('--frac', type=float, default=0.5)
    ap.add_argument('--min-level', type=float, default=170)
    ap.add_argument('--small-land', type=int, default=1500)
    ap.add_argument('--write', action='store_true')
    a = ap.parse_args()
    clicks = {}
    for r in csv.DictReader(open(a.clicks)):
        if r['x'] != '':
            clicks.setdefault(r['scene_id'], []).append((int(r['x']), int(r['y'])))
    log = []
    for s, pts in clicks.items():
        name = s.removeprefix('train_')
        img = np.asarray(Image.open(a.root / f'train/images/{name}.jpg').convert('L'))
        gt = np.asarray(Image.open(a.root / f'train/labels_1D/{name}.png'))
        new, rows = apply(img, gt, pts, small_land=a.small_land, radius=a.radius, frac=a.frac, min_level=a.min_level)
        log += [(s, *r) for r in rows]
        if a.write and (new != gt).any():
            (a.out / 'labels_1D_revised').mkdir(parents=True, exist_ok=True)
            Image.fromarray(new).save(a.out / 'labels_1D_revised' / f'{name}.png')
    if a.write:
        a.out.mkdir(parents=True, exist_ok=True)
        with open(a.out / 'ship_revision_log.csv', 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['scene_id', 'x', 'y', 'px_from_sea', 'px_from_land', 'core_area', 'note']); w.writerows(log)
    for r in log:
        print(r)

COMMANDS = {
    'suspicious': main_suspicious,
    'ship-sheet': main_ship_sheet,
    'add-ships': main_add_ships,
    'snap': main_snap,
    'lattice': main_lattice,
    'review-sheet': main_review_sheet,
    'change-masks': main_change_masks,
    'export-rgb': main_export_rgb,
    'click-ships': main_click_ships,
    'apply-clicks': main_apply_clicks,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print('usage: python -m segmentation.label_review.tools <command> [options]\n\ncommands:\n  ' + '\n  '.join(COMMANDS))
        raise SystemExit(0 if len(sys.argv) > 1 and sys.argv[1] in ('-h', '--help') else 2)
    command = sys.argv.pop(1)
    sys.argv[0] = f'segmentation.label_review.tools {command}'
    COMMANDS[command]()


if __name__ == '__main__':
    main()
