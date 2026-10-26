"""Chapter 4 schematic figures: (1) the four-stage cascade, (2) the 256 px / stride 128 window grid on a real scene, (3) the two-scale model built on SegFormer-B2.
No performance numbers are drawn; everything is geometry or definition."""
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[2]))   # repository root
from segmentation import paths
import argparse
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.patches

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from matplotlib.colors import ListedColormap
from PIL import Image

ROOT = paths.DATASET / "train"
MASK_COLORS = ["#000000", "#00FFFF", "#FF0000", "#994C00", "#009900"]  # Sea, Oil, Look-alike, Ship, Land


def cascade(out, scene="img_0047", pred_path=paths.RESULTS / "oof_stage3_b2_hrda_c_shift_256/fold03/train_img_0047_prediction.npy"):
    """Academic version: panels with the real scene, the window grid, the gate decision, the segmentation and the merged map."""
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix"})
    img = np.array(Image.open(ROOT / "images" / f"{scene}.jpg").convert("L"))
    gt = np.array(Image.open(ROOT / "labels_1D" / f"{scene}.png")) if (ROOT / "labels_1D").exists() else None
    pred = np.load(pred_path)
    H, W = img.shape
    ys = [-59 + 128 * k for k in range(5)]; xs = [-15 + 128 * k for k in range(9)]
    cmap = ListedColormap(MASK_COLORS)
    # illustrative gate: a window is accepted when its labels contain Oil
    acc = np.zeros((5, 9), bool)
    if gt is not None:
        for i, y in enumerate(ys):
            for j, x in enumerate(xs):
                sub = gt[max(y, 0):max(y + 256, 0), max(x, 0):max(x + 256, 0)]
                acc[i, j] = (sub == 1).any()
    fig, ax = plt.subplots(1, 5, figsize=(7.4, 2.3), facecolor="white")
    titles = ["Input scene", "Stage 1\nWindows", "Stage 2\nClassification gate", "Stage 3\nSegmentation", "Stage 4\nScene reconstruction"]
    sub_t = ["SAR intensity,\n1250 $\\times$ 650 px", "256 $\\times$ 256 px,\nstride 128 px (45 windows)",
             "Oil window?\n(recall-first threshold)", "five classes per pixel,\naccepted windows only", "average of overlapping\nwindow probabilities"]
    for a, t, st in zip(ax, titles, sub_t):
        a.set_xticks([]); a.set_yticks([]); a.set_xlim(-30, W + 30); a.set_ylim(H + 70, -70)
        for sp in a.spines.values(): sp.set_linewidth(0.6); sp.set_color("#555")
        a.set_title(t, fontsize=7.5, pad=4, fontweight="bold" if t.startswith("Stage") else "normal")
        a.set_xlabel(st, fontsize=6.5, labelpad=4)
    ext = (0, W, H, 0)
    for a in ax[:3]: a.imshow(img, cmap="gray", extent=ext, vmin=0, vmax=255)
    def draw_grid(a, ec, lw, alpha):
        for y in ys:
            for x in xs: a.add_patch(Rectangle((x, y), 256, 256, fill=False, ec=ec, lw=lw, alpha=alpha))
    draw_grid(ax[1], "#E6A800", 0.5, 0.9)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            ax[2].add_patch(Rectangle((x, y), 256, 256, fill=acc[i, j], fc="#2C7FB8" if acc[i, j] else "none", alpha=0.35 if acc[i, j] else 0.7,
                                       ec="#2C7FB8" if acc[i, j] else "#BBBBBB", lw=0.8 if acc[i, j] else 0.4))
    # stage 3: prediction only inside accepted windows
    m = np.zeros((H, W), bool)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            if acc[i, j]: m[max(y, 0):max(y + 256, 0), max(x, 0):max(x + 256, 0)] = True
    rgb = np.full((H, W, 3), 0.93)
    colors = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 0], [0.6, 0.298, 0], [0, 0.6, 0]])
    rgb[m] = colors[np.clip(pred[m], 0, 4)]
    ax[3].imshow(rgb, extent=ext, interpolation="nearest")
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            if acc[i, j]: ax[3].add_patch(Rectangle((x, y), 256, 256, fill=False, ec="#2C7FB8", lw=0.5))
    ax[4].imshow(colors[np.clip(pred, 0, 4)], extent=ext, interpolation="nearest")
    for k in range(4):
        fig.patches.append(matplotlib.patches.FancyArrowPatch(ax[k].get_position().corners()[0] if False else (0, 0), (0, 0)))
    fig.patches.clear()
    fig.tight_layout(w_pad=1.6)
    for k in range(4):
        p0 = ax[k].get_position(); p1 = ax[k + 1].get_position()
        xm0, xm1 = p0.x1 + 0.004, p1.x0 - 0.004; ymid = (p0.y0 + p0.y1) / 2
        fig.add_artist(matplotlib.patches.FancyArrowPatch((xm0, ymid), (xm1, ymid), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=8, lw=0.9, color="#222"))
    fig.savefig(out / "ch4_cascade_scheme.png", dpi=300, bbox_inches="tight"); fig.savefig(out / "ch4_cascade_scheme.pdf", bbox_inches="tight")
    plt.rcParams.update(plt.rcParamsDefault)


def grid(out, scene):
    """Window grid on the padded canvas, plus one border window and one interior window cut out."""
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix"})
    img = np.array(Image.open(ROOT / "images" / f"{scene}.jpg").convert("L"))
    H, W = img.shape
    PT, PL, PH, PW = 59, 15, 768, 1280  # symmetric padding: top/bottom 59, left/right 15
    canvas = np.full((PH, PW), np.nan); canvas[PT:PT + H, PL:PL + W] = img
    ys = [128 * k for k in range(5)]; xs = [128 * k for k in range(9)]  # window origins on the padded canvas
    cm = plt.get_cmap("gray").copy(); cm.set_bad("#D9D9D9")
    fig = plt.figure(figsize=(7.0, 3.3), facecolor="white")
    gs = fig.add_gridspec(2, 2, width_ratios=[3.4, 1], height_ratios=[1, 1], wspace=0.08, hspace=0.3)
    A = fig.add_subplot(gs[:, 0]); A.imshow(np.ma.masked_invalid(canvas), cmap=cm, vmin=0, vmax=255, interpolation="nearest")
    for y in ys:
        for x in xs: A.add_patch(Rectangle((x - 0.5, y - 0.5), 256, 256, fill=False, ec="#E6A800", lw=0.5, alpha=0.85))
    border, interior = (0, 0), (2, 4)
    for (i, j), c in ((border, "#1B9E3B"), (interior, "#D6249F")):
        A.add_patch(Rectangle((xs[j] - 0.5, ys[i] - 0.5), 256, 256, fill=False, ec=c, lw=2.2))
    A.add_patch(Rectangle((PL - 0.5, PT - 0.5), W, H, fill=False, ec="#222", lw=0.9, ls="--"))
    A.set_xlim(-0.5, PW - 0.5); A.set_ylim(PH - 0.5, -0.5)
    A.set_xticks([128 * k + 128 for k in range(9)]); A.set_xticklabels([str(k + 1) for k in range(9)], fontsize=6.5)
    A.set_yticks([128 * k + 128 for k in range(5)]); A.set_yticklabels([str(k + 1) for k in range(5)], fontsize=6.5)
    A.set_xlabel("Window column", fontsize=7.5); A.set_ylabel("Window row", fontsize=7.5)
    tag = lambda ax_, L, x=0.02, h="left": ax_.text(x, 0.97, L, transform=ax_.transAxes, fontsize=10, fontweight="bold", va="top", ha=h, color="white", bbox=dict(boxstyle="square,pad=0.15", fc="black", ec="none", alpha=0.75))
    tag(A, "A", 0.99, "right")
    A.text(PL + 8, PT + H - 12, "scene 1250 $\\times$ 650 px (dashed)", fontsize=6.5, color="#fff", bbox=dict(fc="#000", alpha=0.5, ec="none", pad=1.5))
    A.text(330, 40, "padding (grey)", fontsize=6.5, color="#222")
    for k, ((i, j), c, t) in enumerate((((0, 0), "#1B9E3B", "B"),
                                        ((2, 4), "#D6249F", "C"))):
        a = fig.add_subplot(gs[k, 1]); crop = canvas[ys[i]:ys[i] + 256, xs[j]:xs[j] + 256]
        a.imshow(np.ma.masked_invalid(crop), cmap=cm, vmin=0, vmax=255, interpolation="nearest")
        for sp in a.spines.values(): sp.set_edgecolor(c); sp.set_linewidth(2)
        a.set_xticks([]); a.set_yticks([]); tag(a, t)
        a.set_xlabel("256 $\\times$ 256 px", fontsize=7)
    fig.savefig(out / "ch4_patch_grid_example.png", dpi=300, bbox_inches="tight"); fig.savefig(out / "ch4_patch_grid_example.pdf", bbox_inches="tight")
    plt.rcParams.update(plt.rcParamsDefault)


def two_scale(out, scene="img_0047", pred_path=paths.RESULTS / "oof_stage3_b2_hrda_c_shift_256/fold03/train_img_0047_prediction.npy", win=(128, 738)):
    """Two-scale model (HRDA-C) on SegFormer-B2 with the notation of the text: inputs x_d and x_c, shared network f_theta, logits z_d and z_c,
    upsampling and cropping operators, fused logits z and the training loss. Inputs and the final prediction are real; the blocks between them are drawn."""
    img = np.array(Image.open(ROOT / "images" / f"{scene}.jpg").convert("L"))
    pred = np.load(pred_path)
    y0, x0 = win
    detail = img[y0:y0 + 256, x0:x0 + 256]
    big = np.pad(img, 256, mode="reflect")[y0 - 128 + 256:y0 + 384 + 256, x0 - 128 + 256:x0 + 384 + 256]   # 512 px crop, window at its centre
    context = big.reshape(256, 2, 256, 2).mean(axis=(1, 3))
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix"})
    Wi, Hi = 7.2, 3.3
    fig = plt.figure(figsize=(Wi, Hi), facecolor="white")
    bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, 1); bg.set_ylim(0, 1); bg.axis("off")
    def panel(rect, im, letter, cmap="gray", **kw):
        a = fig.add_axes(rect); a.imshow(im, cmap=cmap, interpolation="nearest", **kw); a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values(): sp.set_linewidth(0.6); sp.set_color("#555")
        a.text(0.04, 0.96, letter, transform=a.transAxes, fontsize=9, fontweight="bold", va="top", ha="left", color="white", bbox=dict(boxstyle="square,pad=0.15", fc="black", ec="none", alpha=0.75))
        return a
    txt = dict(ha="center", va="center", fontsize=7.5)
    side = 0.82; wf, hf = side / Wi, side / Hi
    yt, yb = 0.58, 0.20            # lower edges of the detail (top) and context (bottom) rows
    A = panel([0.01, 0.42, 0.17, 0.17 * Wi / (img.shape[1] / img.shape[0]) / Hi], img, "A", vmin=0, vmax=255)
    A.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), 256, 256, fill=False, ec="#E6A800", lw=1.3))
    A.add_patch(Rectangle((x0 - 128 - 0.5, y0 - 128 - 0.5), 512, 512, fill=False, ec="white", lw=1.0, ls="--"))
    A.set_xlim(-0.5, img.shape[1] - 0.5); A.set_ylim(img.shape[0] - 0.5, -0.5)
    bg.text(0.095, 0.385, "scene $I$", **txt)
    xin = 0.225
    panel([xin, yt, wf, hf], detail, "B", vmin=0, vmax=255); panel([xin, yb, wf, hf], context, "C", vmin=0, vmax=255)
    bg.text(xin + wf / 2, yt + hf + 0.04, "$x_d$: window, $256\\times256$", **txt)
    bg.text(xin + wf / 2, yb - 0.045, "$x_c$: context $512\\times512$,\n reduced by $2\\times2$ averaging", **txt)
    def box(x, y, w, h, t, fc="#EEF3F8", ec="#2C7FB8", ls="-", fs=7.5):
        bg.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.003,rounding_size=0.01", fc=fc, ec=ec, lw=0.9, ls=ls)); bg.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs)
    def arrow(p0, p1, lab=None, off=0.03):
        bg.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=8, lw=0.9, color="#222"))
        if lab: bg.text((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2 + off, lab, ha="center", va="bottom", fontsize=6.5)
    xe, we = 0.385, 0.125
    box(xe, yb, we, yt + hf - yb, "$f_{\\theta}$\n\nSegFormer-B2\n(MiT-B2 encoder,\nall-MLP decode\nhead)\n\nshared weights\n$\\theta$ for both\nviews", fc="#F4F4F4", ec="#555")
    xo = 0.565
    yd, yc = yt + hf / 2, yb + hf / 2
    box(xo, yd - 0.08, 0.135, 0.16, "$z_d\\in\\mathbb{R}^{5\\times64\\times64}$\n$\\tilde z_d=\\uparrow_4 z_d$")
    box(xo, yc - 0.08, 0.135, 0.16, "$z_c\\in\\mathbb{R}^{5\\times64\\times64}$\n$\\tilde z_c=\\uparrow_8\\,\\mathrm{crop}(z_c)$")
    arrow((xin + wf + 0.004, yd), (xe - 0.003, yd)); arrow((xin + wf + 0.004, yc), (xe - 0.003, yc))
    arrow((xe + we + 0.003, yd), (xo - 0.003, yd)); arrow((xe + we + 0.003, yc), (xo - 0.003, yc))
    arrow((0.185, 0.55), (xin - 0.004, yd - 0.05)); arrow((0.185, 0.50), (xin - 0.004, yc + 0.05))
    xm = 0.775
    box(xm, 0.45, 0.085, 0.14, "$z=\\frac{1}{2}\\,(\\tilde z_d+\\tilde z_c)$", fc="white", ec="#222", fs=7.5)
    arrow((xo + 0.135 + 0.003, yd), (xm + 0.02, 0.593)); arrow((xo + 0.135 + 0.003, yc), (xm + 0.02, 0.447))
    D = panel([0.885, 0.40, 0.105, 0.105 * Wi / Hi], pred[y0:y0 + 256, x0:x0 + 256], "D", cmap=ListedColormap(MASK_COLORS), vmin=0, vmax=4)
    bg.text(0.9375, 0.40 + 0.105 * Wi / Hi + 0.045, "$\\hat y=\\arg\\max\\,z$", **txt)
    arrow((xm + 0.085 + 0.003, 0.52), (0.882, 0.52))
    bg.add_patch(FancyBboxPatch((0.225, 0.005), 0.77, 0.05, boxstyle="round,pad=0.002,rounding_size=0.008", fc="white", ec="#888", lw=0.7, ls="--"))
    bg.text(0.61, 0.03, "training only:  $\\mathcal{L}=\\mathrm{CE}(z,y)+\\frac{1}{2}\\mathrm{CE}(\\tilde z_d,y)+\\frac{1}{2}\\mathrm{CE}(\\tilde z_c,y)$,  $y$: label of the window;  inference uses $z$ only", ha="center", va="center", fontsize=7.3)
    fig.savefig(out / "ch4_two_scale_model.png", dpi=300, bbox_inches="tight"); fig.savefig(out / "ch4_two_scale_model.pdf", bbox_inches="tight")
    plt.close(fig); plt.rcParams.update(plt.rcParamsDefault)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, default=paths.PROJECT_ROOT / "dataset_analysis_output" / "results__ch4/figures"); ap.add_argument("--scene", default="img_0047")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True); cascade(a.out); grid(a.out, a.scene); two_scale(a.out)
