"""Chapter 4 figure: how the class statistics change from scenes to 256x256 windows (stride 128).

Inputs: the verified patch index of the 1,002 official-train scenes and the scene manifest.
Outputs: PNG/PDF figure and a CSV with the numbers shown (dataset_analysis/results__ch4/figures).
"""
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[2]))   # repository root
from segmentation import paths
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASSES = ["sea", "oil", "look_alike", "ship", "land"]
LABELS = ["Sea", "Oil", "Look-alike", "Ship", "Land"]
COL_SCENE, COL_WIN, COL_512 = "#4C78A8", "#E08A1E", "#3A9E6F"


def window_stats(idx):
    pos = idx[idx["pixels_oil"] > 0]
    neg = idx[idx["pixels_oil"] == 0]
    cls_pix = np.array([idx["pixels_" + c].sum() for c in CLASSES], float)
    return {"n": len(idx), "pres": np.array([(idx["pixels_" + c] > 0).mean() * 100 for c in CLASSES]),
            "share": 100 * cls_pix / cls_pix.sum(), "frac": 100 * pos["pixels_oil"] / pos["valid_pixels"],
            "n_pos": len(pos), "la_pos": (pos["pixels_look_alike"] > 0).mean() * 100, "la_neg": (neg["pixels_look_alike"] > 0).mean() * 100,
            "border": 100 * idx["is_border"].mean(),
            "pad": 100 * idx["padding_pixels"].sum() / (idx["valid_pixels"].sum() + idx["padding_pixels"].sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch-root", type=Path, default=paths.STAGE2_RESULTS / "training_patches")
    ap.add_argument("--scene-csv", type=Path, default=paths.PROJECT_ROOT / "dataset_analysis_output" / "results__ch3/class_pixels_per_image.csv")
    ap.add_argument("--out", type=Path, default=paths.PROJECT_ROOT / "dataset_analysis_output" / "results__ch4/figures")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    st = {}
    for ps in (256, 512):
        idx = pd.read_csv(a.patch_root / f"ps{ps}_overlap50" / f"index_ps{ps}_overlap50.csv")
        idx = idx[idx["split"] == "train"]
        assert idx["scene_id"].nunique() == 1002
        st[ps] = window_stats(idx)
    sc = pd.read_csv(a.scene_csv).iloc[:1002]  # the per-image table lists the 1,002 train scenes first, then the test scenes
    assert list(sc["filename"].iloc[[0, -1]]) == ["img_0001.png", "img_1002.png"]
    colmap = {"sea": "Sea (Ocean Surface)", "oil": "Oil Spill", "look_alike": "Look-alike", "ship": "Ship", "land": "Land"}
    scene_pix = np.array([sc[colmap[c] + "_pixels"].sum() for c in CLASSES], float)
    scene_share = 100 * scene_pix / scene_pix.sum()
    scene_pres = np.array([(sc[colmap[c] + "_pixels"] > 0).mean() * 100 for c in CLASSES])

    tab = pd.DataFrame({"class": LABELS, "scene_presence_%": scene_pres, "win256_presence_%": st[256]["pres"], "win512_presence_%": st[512]["pres"],
                        "scene_pixel_share_%": scene_share, "win256_pixel_share_%": st[256]["share"], "win512_pixel_share_%": st[512]["share"]})
    tab.to_csv(a.out / "ch4_patching_statistics.csv", index=False)
    summ = {f"{k}_{ps}": v for ps in (256, 512) for k, v in {"windows": st[ps]["n"], "windows_per_scene": st[ps]["n"] / 1002, "oil_positive_windows": st[ps]["n_pos"],
            "oil_positive_rate_%": 100 * st[ps]["n_pos"] / st[ps]["n"], "median_oil_fraction_%": float(np.median(st[ps]["frac"])),
            "look_alike_given_oil_%": st[ps]["la_pos"], "look_alike_given_no_oil_%": st[ps]["la_neg"],
            "border_window_rate_%": st[ps]["border"], "padding_fraction_%": st[ps]["pad"]}.items()}
    pd.Series(summ).to_csv(a.out / "ch4_patching_summary.csv", header=["value"])
    print(tab.round(3).to_string(index=False)); print(pd.Series(summ).round(3).to_string())

    plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.6), facecolor="white")
    x = np.arange(5); w = 0.26
    series = [("Scenes (1,002)", scene_pres, scene_share, COL_SCENE), ("Windows 256 px (45,090)", st[256]["pres"], st[256]["share"], COL_WIN),
              ("Windows 512 px (8,016)", st[512]["pres"], st[512]["share"], COL_512)]
    for k, (title, yl, log, which) in enumerate([("Presence of each class", "Units containing the class (%)", False, 1),
                                                  ("Share of labelled pixels", "Share of pixels (%, log scale)", True, 2)]):
        a_ = ax[0, k]
        for m, (lab, pres, share, col) in enumerate(series):
            vals = pres if which == 1 else share
            bars = a_.bar(x + (m - 1) * w, vals, w, color=col, label=lab)
            for r in bars:
                v = r.get_height()
                a_.text(r.get_x() + r.get_width() / 2, v * (1.12 if log else 1) + (0 if log else 1.2), f"{v:.2f}" if v < 10 else f"{v:.0f}",
                        ha="center", va="bottom", fontsize=5.5, rotation=90 if log else 0)
        a_.set_xticks(x); a_.set_xticklabels(LABELS); a_.set_ylabel(yl)
        if log: a_.set_yscale("log"); a_.set_ylim(0.01, 600)
        else: a_.set_ylim(0, 115)
        if k == 0: a_.legend(frameon=False, loc="upper right", fontsize=7)
    c = ax[1, 0]
    for ps, col in ((256, COL_WIN), (512, COL_512)):
        s_ = np.sort(st[ps]["frac"].values); y_ = np.arange(1, len(s_) + 1) / len(s_)
        c.plot(s_, y_, color=col, lw=2, label=f"{ps} px (n = {st[ps]['n_pos']:,}; median {np.median(s_):.2f}%)")
    c.set_xscale("log"); c.set_xlabel("Oil fraction of the valid window area (%, log scale)"); c.set_ylabel("Cumulative fraction of windows")
    c.set_ylim(0, 1.02); c.legend(frameon=False, loc="upper left", fontsize=7)
    d = ax[1, 1]; xx = np.arange(2); ww = 0.34
    for m, (ps, col) in enumerate(((256, COL_WIN), (512, COL_512))):
        vals = [st[ps]["la_pos"], st[ps]["la_neg"]]
        bars = d.bar(xx + (m - 0.5) * ww, vals, ww, color=col, label=f"{ps} px")
        for r in bars: d.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.8, f"{r.get_height():.1f}%", ha="center", fontsize=7.5)
    d.set_xticks(xx); d.set_xticklabels(["Oil-positive\nwindows", "Oil-negative\nwindows"])
    d.set_ylabel("Windows with Look-alike (%)"); d.set_ylim(0, 52); d.legend(frameon=False, fontsize=7.5)
    for a_, L in zip(ax.flat, "ABCD"):
        a_.text(-0.2, 1.05, L, transform=a_.transAxes, fontsize=12, fontweight="bold")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(a.out / f"ch4_patching_statistics.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")


if __name__ == "__main__":
    main()
