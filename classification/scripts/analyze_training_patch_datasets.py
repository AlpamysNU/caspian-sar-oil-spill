#!/usr/bin/env python3
"""Summarize verified Stage 2 patch datasets and create Chapter 4 figures.

The script reads only index files and verification reports. Pixel totals are
repetition-weighted training pixels because overlapping patches repeat source
pixels; they must not be interpreted as unique scene area.
"""
from __future__ import annotations

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[2]))   # repository root
from segmentation import paths
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-stage2-patch-analysis")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CLASSES = ("sea", "oil", "look_alike", "ship", "land")
ACADEMIC_COLORS = ("#0072B2", "#E69F00", "#009E73")
CONFIGS = ((256, 128), (384, 192), (512, 256))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse the three verified official-train patch datasets.")
    parser.add_argument("--input-root", type=Path, default=paths.STAGE2_RESULTS / "training_patches")
    parser.add_argument("--output", type=Path, default=Path("stage2_results/training_patch_analysis"))
    return parser.parse_args()


def load_configuration(input_root: Path, patch_size: int, stride: int) -> tuple[pd.DataFrame, dict]:
    name = f"ps{patch_size}_overlap{int(round((1 - stride / patch_size) * 100)):02d}"
    directory = input_root / name
    index_path = directory / f"index_{name}.parquet"
    verification_path = directory / "verification.json"
    if not index_path.is_file() or not verification_path.is_file():
        raise FileNotFoundError(f"Expected verified dataset files in {directory}")
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if not verification.get("all_patch_pixels_and_labels_checked") or verification.get("uniform_reconstruction_max_error") != 0:
        raise ValueError(f"Dataset verification is incomplete: {directory}")
    index = pd.read_parquet(index_path)
    if len(index) != verification["patches"] or set(index["split"]) != {"train"}:
        raise ValueError(f"Unexpected index contents: {directory}")
    return index, verification


def percentile_rows(index: pd.DataFrame, name: str) -> list[dict]:
    positive = index.loc[index.label_oil == 1].copy()
    oil_fraction = positive.pixels_oil / positive.valid_pixels
    percentiles = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
    rows = []
    for percentile in percentiles:
        rows.append({
            "configuration": name,
            "percentile": percentile,
            "oil_pixels": float(positive.pixels_oil.quantile(percentile)),
            "oil_fraction_of_valid_area": float(oil_fraction.quantile(percentile)),
        })
    return rows


def academic_axis(axis: plt.Axes, label: str) -> None:
    """Style and label one panel for print-oriented academic output."""
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_linewidth(0.8)
    axis.grid(False)
    axis.tick_params(direction="out", width=0.8, length=3)
    axis.text(-0.14, 1.06, label, transform=axis.transAxes, fontsize=9, fontweight="bold", va="top")


def save_academic_figure(fig: plt.Figure, output: Path, stem: str) -> None:
    """Write a high-resolution raster preview and an editable vector figure."""
    fig.savefig(output / f"{stem}.png", dpi=450, bbox_inches="tight", facecolor="white")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_academic_figures(
    summary: pd.DataFrame,
    cooccurrence: pd.DataFrame,
    oil_fractions: dict[str, pd.Series],
    output: Path,
) -> None:
    """Create compact double-column figures with neutral panel titles."""
    plt.rcParams.update({
        "font.family": "Liberation Sans",
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    names = summary.configuration.tolist()
    x = np.arange(len(names))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25))
    fig.subplots_adjust(left=0.09, right=0.99, top=0.88, bottom=0.18, wspace=0.34)
    bars = axes[0].bar(x, summary.patches_per_scene, color=ACADEMIC_COLORS[0], width=0.58)
    axes[0].bar_label(bars, labels=[f"{value:.0f}" for value in summary.patches_per_scene], padding=2, fontsize=6.5)
    axes[0].set(xticks=x, xticklabels=names, xlabel="Patch size / stride (px)",
                ylabel="Evaluations per scene (count)", title="Stage 2 patch evaluations")
    axes[0].set_ylim(0, summary.patches_per_scene.max() * 1.2)
    academic_axis(axes[0], "a")
    stage2 = summary.stage2_input_megapixels_per_scene
    stage3 = summary.oracle_stage3_input_megapixels_per_scene
    axes[1].bar(x, stage2, color=ACADEMIC_COLORS[0], width=0.58, label="Stage 2: all patches")
    axes[1].bar(x, stage3, bottom=stage2, color=ACADEMIC_COLORS[1], width=0.58, label="Stage 3: oracle Oil-positive")
    for position, first, second in zip(x, stage2, stage3):
        axes[1].text(position, first / 2, f"{first:.2f}", ha="center", va="center", color="white", fontsize=6.5)
        axes[1].text(position, first + second / 2, f"{second:.2f}", ha="center", va="center", fontsize=6.5)
    axes[1].set(xticks=x, xticklabels=names, xlabel="Patch size / stride (px)",
                ylabel="Input pixels per scene (MP)", title="Native-input pixel-volume proxy")
    axes[1].legend(frameon=False, loc="upper right", handlelength=1.2)
    academic_axis(axes[1], "b")
    save_academic_figure(fig, output, "01_patch_scale_and_class_presence")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25))
    fig.subplots_adjust(left=0.09, right=0.99, top=0.88, bottom=0.18, wspace=0.34)
    positive_rate = summary.oil_positive_patch_rate * 100
    bars = axes[0].bar(x, positive_rate, color=ACADEMIC_COLORS, width=0.58)
    axes[0].bar_label(bars, labels=[f"{value:.1f}%" for value in positive_rate], padding=2, fontsize=6.5)
    axes[0].set(xticks=x, xticklabels=names, xlabel="Patch size / stride (px)",
                ylabel="Patches containing Oil (%)", title="Oil-positive patch frequency")
    axes[0].set_ylim(0, 55)
    academic_axis(axes[0], "a")
    for name, color in zip(names, ACADEMIC_COLORS):
        values = np.sort(oil_fractions[name].to_numpy() * 100)
        cumulative = np.arange(1, len(values) + 1) / len(values)
        axes[1].step(values, cumulative, where="post", color=color, linewidth=1.4, label=name)
    axes[1].set_xscale("log")
    axes[1].set(xlabel="Oil fraction of valid patch area (%)", ylabel="Cumulative probability",
                title="Oil occupancy in Oil-positive patches")
    axes[1].set_ylim(0, 1)
    axes[1].legend(title="Patch / stride (px)", title_fontsize=6.5, frameon=False, loc="lower right")
    academic_axis(axes[1], "b")
    save_academic_figure(fig, output, "02_valid_pixel_composition_and_padding")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25))
    fig.subplots_adjust(left=0.09, right=0.99, top=0.88, bottom=0.18, wspace=0.34)
    indexed = cooccurrence.set_index("configuration").loc[names]
    for label, axis, column, title, ylabel in (
        ("a", axes[0], "look_alike_given_oil", "Oil-positive patches", "Also containing Look-alike (%)"),
        ("b", axes[1], "look_alike_given_no_oil", "Oil-negative patches", "Containing Look-alike (%)"),
    ):
        values = indexed[column] * 100
        bars = axis.bar(x, values, color=ACADEMIC_COLORS, width=0.58)
        axis.bar_label(bars, labels=[f"{value:.1f}%" for value in values], padding=2, fontsize=6.5)
        axis.set(xticks=x, xticklabels=names, xlabel="Patch size / stride (px)", ylabel=ylabel, title=title)
        axis.set_ylim(0, max(values) * 1.25)
        academic_axis(axis, label)
    save_academic_figure(fig, output, "03_oil_positive_patch_occupancy_ecdf")


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary_rows, class_rows, quantile_rows, cooccurrence_rows, oil_fractions = [], [], [], [], {}
    scene_count = None
    for patch_size, stride in CONFIGS:
        name = f"{patch_size} / {stride}"
        index, verification = load_configuration(args.input_root, patch_size, stride)
        if scene_count is None:
            scene_count = verification["scenes"]
        elif scene_count != verification["scenes"]:
            raise ValueError("Configurations do not cover the same number of scenes")
        valid_pixels = int(index.valid_pixels.sum())
        padding_pixels = int(index.padding_pixels.sum())
        oil_positive = index.label_oil.astype(bool)
        look_alike_positive = index.label_look_alike.astype(bool)
        oil_fraction_positive = index.loc[oil_positive, "pixels_oil"] / index.loc[oil_positive, "valid_pixels"]
        summary_rows.append({
            "configuration": name, "patch_size": patch_size, "stride": stride,
            "overlap": 1 - stride / patch_size, "scene_count": scene_count,
            "patch_count": len(index), "patches_per_scene": len(index) / scene_count,
            "border_patch_count": int(index.is_border.sum()), "border_patch_rate": float(index.is_border.mean()),
            "valid_pixels": valid_pixels, "padding_pixels": padding_pixels,
            "padding_fraction_of_stored_pixels": padding_pixels / (valid_pixels + padding_pixels),
            "stage2_input_megapixels_per_scene": len(index) * patch_size ** 2 / scene_count / 1e6,
            "oracle_stage3_input_megapixels_per_scene": int(oil_positive.sum()) * patch_size ** 2 / scene_count / 1e6,
            "oil_positive_patch_count": int(oil_positive.sum()),
            "oil_positive_patch_rate": float(oil_positive.mean()),
            "median_oil_fraction_positive_patches": float(oil_fraction_positive.median()),
        })
        for class_name in CLASSES:
            pixels = int(index[f"pixels_{class_name}"].sum())
            positives = int(index[f"label_{class_name}"].sum())
            class_rows.append({
                "configuration": name, "class_name": class_name,
                "positive_patch_count": positives, "positive_patch_rate": positives / len(index),
                "valid_pixel_count_repetition_weighted": pixels, "valid_pixel_fraction": pixels / valid_pixels,
            })
        quantile_rows.extend(percentile_rows(index, name))
        oil_fractions[name] = oil_fraction_positive
        cooccurrence_rows.append({
            "configuration": name,
            "oil_and_look_alike_patch_count": int((oil_positive & look_alike_positive).sum()),
            "look_alike_given_oil": float(look_alike_positive[oil_positive].mean()),
            "look_alike_given_no_oil": float(look_alike_positive[~oil_positive].mean()),
        })

    summary = pd.DataFrame(summary_rows)
    classes = pd.DataFrame(class_rows)
    quantiles = pd.DataFrame(quantile_rows)
    cooccurrence = pd.DataFrame(cooccurrence_rows)
    summary.to_csv(args.output / "patch_dataset_summary.csv", index=False)
    classes.to_csv(args.output / "patch_class_statistics.csv", index=False)
    quantiles.to_csv(args.output / "oil_positive_patch_quantiles.csv", index=False)
    cooccurrence.to_csv(args.output / "oil_look_alike_cooccurrence.csv", index=False)
    write_academic_figures(summary, cooccurrence, oil_fractions, args.output)
    metadata = {
        "scope": "Three verified official M4D train patch datasets only",
        "configurations": [f"{size}/{stride}" for size, stride in CONFIGS],
        "source": "Patch index Parquet files and verification.json files",
        "pixel_accounting": "Pixel totals are repetition-weighted across overlapping patches; they are not unique original-scene area.",
        "oil_occupancy": "Computed only for patches containing at least one Oil pixel, using valid non-padding area.",
        "figures": ["01_patch_scale_and_class_presence", "02_valid_pixel_composition_and_padding", "03_oil_positive_patch_occupancy_ecdf"],
        "formats": ["PNG at 450 dpi", "vector PDF with editable TrueType text"],
        "workload_proxy": "Native input pixels per scene; it is not FLOPs, GPU time, energy, or monetary cost.",
        "oracle_stage3_rule": "A patch is routed when its valid area contains at least one ground-truth Oil pixel.",
    }
    (args.output / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Wrote analysis to {args.output}")


if __name__ == "__main__":
    main()
