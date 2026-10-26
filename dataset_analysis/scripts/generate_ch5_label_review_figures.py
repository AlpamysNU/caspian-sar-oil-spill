"""Chapter 5 figures on the label review: (1) before/after examples, (2) possible wind-farm lattices.
Original labels come from the dataset, revised labels from the parallel copy (results/label_review)."""
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[2]))   # repository root
from segmentation import paths
import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from PIL import Image

D = paths.DATASET / "train"
R = paths.RESULTS / "label_review/revised_manual/labels_RGB_revised"
LEG = [("Sea", "#000000"), ("Oil", "#00FFFF"), ("Look-alike", "#FF0000"), ("Ship", "#994C00"), ("Land", "#009900")]


def load(s):
    img = np.array(Image.open(D / "images" / f"img_{s}.jpg").convert("L"))
    o = np.array(Image.open(D / "labels" / f"img_{s}.png").convert("RGB"))
    r = np.array(Image.open(R / f"img_{s}.png").convert("RGB"))
    return img, o, r


def crop_box(ch, margin, H, W, min_size=220):
    ys, xs = np.where(ch)
    y0, y1, x0, x1 = ys.min() - margin, ys.max() + margin, xs.min() - margin, xs.max() + margin
    for lo, hi, n in ((y0, y1, H), (x0, x1, W)):
        pass
    h, w = y1 - y0, x1 - x0
    if h < min_size: y0 -= (min_size - h) // 2; y1 = y0 + min_size
    if w < min_size: x0 -= (min_size - w) // 2; x1 = x0 + min_size
    y0, x0 = max(0, y0), max(0, x0); y1, x1 = min(H, y1), min(W, x1)
    return y0, y1, x0, x1


def best_window(ch, size):
    """Top-left corner of the size x size window that contains most changed pixels (integral image)."""
    H, W = ch.shape
    c = np.pad(ch.astype(np.int32).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    best, pos = -1, (0, 0)
    for y in range(0, H - size + 1, 10):
        for x in range(0, W - size + 1, 10):
            v = c[y + size, x + size] - c[y, x + size] - c[y + size, x] + c[y, x]
            if v > best: best, pos = v, (y, x)
    return pos


def examples(out):
    rows = [("0730", "Missing ships added (scene without any Ship label)", 300),
            ("0065", "False Ship label removed, shifted Oil mask corrected", None),
            ("0275", "Displaced Oil mask corrected", None),
            ("0279", "Look-alike pieces on ship wakes changed to Oil", 260)]
    crops = []
    for s, title, win in rows:
        img, o, r = load(s); H, W = img.shape; ch = (o != r).any(2)
        if win:
            y0, x0 = best_window(ch, win); box = (y0, y0 + win, x0, x0 + win)
        else:
            box = crop_box(ch, 25, H, W)
        crops.append((s, title, img, o, r, ch, box))
    ratios = [(b[1] - b[0]) / (b[3] - b[2]) for *_, b in crops]
    fig = plt.figure(figsize=(6.5, sum(1.6 * min(max(q, 0.45), 1.25) for q in ratios) + 0.4), facecolor="white")
    gs = fig.add_gridspec(len(rows), 4, height_ratios=[min(max(q, 0.45), 1.25) for q in ratios], wspace=0.04, hspace=0.18)
    for i, (s, title, img, o, r, ch, (y0, y1, x0, x1)) in enumerate(crops):
        sl = (slice(y0, y1), slice(x0, x1))
        ov = np.stack([img] * 3, -1).astype(float) / 255
        ov[ch] = [1, 0, 1]
        panels = [(np.stack([img] * 3, -1)[sl], "SAR image"), (o[sl], "Original label"), (r[sl], "Revised label"), (ov[sl], "Changed pixels")]
        for j, (im, t) in enumerate(panels):
            a = fig.add_subplot(gs[i, j]); a.imshow(im, interpolation="nearest"); a.set_xticks([]); a.set_yticks([])
            if i == 0: a.set_title(t, fontsize=8.5)
            if j == 0:
                a.text(0.03, 0.97, "ABCD"[i], transform=a.transAxes, fontsize=11, fontweight="bold", va="top", ha="left", color="white", bbox=dict(boxstyle="square,pad=0.15", fc="black", ec="none", alpha=0.75))
    fig.legend(handles=[Patch(fc=c, ec="grey", label=n) for n, c in LEG[:4]] + [Patch(fc="#FF00FF", label="changed")], loc="lower center", ncol=5, frameon=False, fontsize=8, handlelength=1.2, columnspacing=1.0)
    fig.savefig(out / "ch5_label_review_examples.png", dpi=250, bbox_inches="tight"); fig.savefig(out / "ch5_label_review_examples.pdf", bbox_inches="tight")


def lattices(out):
    fig = plt.figure(figsize=(6.5, 4.2), facecolor="white")
    gs = fig.add_gridspec(2, 3, width_ratios=[2.0, 1, 1], wspace=0.05, hspace=0.12)
    for i, (s, win) in enumerate([("0556", 320), ("0216", 220)]):
        img, o, r = load(s); H, W = img.shape; ch = (o != r).any(2)
        y0, x0 = best_window(ch, win)
        g3 = np.stack([img] * 3, -1)
        a = fig.add_subplot(gs[i, 0]); a.imshow(g3, interpolation="nearest")
        a.add_patch(Rectangle((x0, y0), win, win, fill=False, ec="#FFD400", lw=2)); a.set_xticks([]); a.set_yticks([])
        a.text(0.02, 0.97, "AB"[i], transform=a.transAxes, fontsize=11, fontweight="bold", ha="left", va="top", color="white", bbox=dict(boxstyle="square,pad=0.15", fc="black", ec="none", alpha=0.75))
        sl = (slice(y0, y0 + win), slice(x0, x0 + win))
        b = fig.add_subplot(gs[i, 1]); b.imshow(g3[sl], interpolation="nearest"); b.set_xticks([]); b.set_yticks([])
        c = fig.add_subplot(gs[i, 2]); c.imshow(r[sl], interpolation="nearest"); c.set_xticks([]); c.set_yticks([])
        if i == 0:
            a.set_title("Scene", fontsize=8.5); b.set_title("Zoom, SAR image", fontsize=8.5); c.set_title("Zoom, revised label", fontsize=8.5)
    fig.savefig(out / "ch5_wind_farm_lattices.png", dpi=250, bbox_inches="tight"); fig.savefig(out / "ch5_wind_farm_lattices.pdf", bbox_inches="tight")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, default=paths.PROJECT_ROOT / "dataset_analysis_output" / "results__ch4/figures")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True); examples(a.out); lattices(a.out)
