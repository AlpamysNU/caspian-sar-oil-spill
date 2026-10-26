"""
Chapter 3 dataset figures generator for M4D-SAR-OilSpill.

This script creates the revised figure set proposed for the dataset-description
section of Chapter 3. It is intentionally separate from analyze_dataset.py so
the original pipeline remains unchanged.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

try:
    import seaborn as sns
except ImportError:  # pragma: no cover - optional dependency
    sns = None

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None

try:
    from scipy import ndimage
except Exception:  # pragma: no cover - scipy may be unavailable
    ndimage = None

from utils import (
    CLASS_NAMES,
    compute_class_distribution,
    remap_rgb_to_1d,
    validate_dataset,
)


CLASS_COLOR_MAP = {
    0: (0, 0, 0),       # Sea Surface
    1: (0, 255, 255),   # Oil Spill
    2: (255, 0, 0),     # Look-alike
    3: (153, 76, 0),    # Ship
    4: (0, 153, 0)      # Land
}

CLASS_HEX_MAP = {
    class_id: "#{:02x}{:02x}{:02x}".format(*rgb)
    for class_id, rgb in CLASS_COLOR_MAP.items()
}

TARGET_CLASSES = [1, 2, 3, 4]
RNG = np.random.default_rng(42)


def setup_publication_style() -> None:
    """Configure plotting defaults for publication-quality output."""
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    if sns is not None:
        sns.set_style("whitegrid")
        sns.set_palette("husl")
    else:
        plt.style.use("default")
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["figure.autolayout"] = True


def plot_grouped_bars(
    ax: plt.Axes,
    df: pd.DataFrame,
    category_col: str,
    value_col: str,
    hue_col: str,
    category_order: List[str],
    hue_order: List[str],
    colors: List[str],
) -> None:
    """Fallback grouped bar plot when seaborn is unavailable."""
    x = np.arange(len(category_order))
    width = 0.8 / max(len(hue_order), 1)

    for idx, hue_value in enumerate(hue_order):
        values = []
        for category in category_order:
            row = df[(df[category_col] == category) & (df[hue_col] == hue_value)]
            values.append(float(row.iloc[0][value_col]) if not row.empty else 0.0)
        offset = (idx - (len(hue_order) - 1) / 2.0) * width
        ax.bar(x + offset, values, width=width, color=colors[idx], label=hue_value, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(category_order)


def plot_grouped_boxplots(
    ax: plt.Axes,
    df: pd.DataFrame,
    class_order: List[str],
    split_order: List[str],
    colors: List[str],
) -> None:
    """Fallback grouped boxplot when seaborn is unavailable."""
    base_positions = np.arange(len(class_order)) * 3.0
    offsets = [-0.45, 0.45]

    for split_idx, split_name in enumerate(split_order):
        split_data = []
        positions = []
        for class_idx, class_name in enumerate(class_order):
            values = df[(df["Class"] == class_name) & (df["Split"] == split_name)]["Area (%)"].to_numpy()
            if values.size == 0:
                values = np.array([np.nan])
            split_data.append(values)
            positions.append(base_positions[class_idx] + offsets[split_idx])

        artists = ax.boxplot(
            split_data,
            positions=positions,
            widths=0.7,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
        )
        for patch in artists["boxes"]:
            patch.set_facecolor(colors[split_idx])
            patch.set_alpha(0.75)

    ax.set_xticks(base_positions)
    ax.set_xticklabels(class_order)


def plot_heatmap(
    ax: plt.Axes,
    data: np.ndarray,
    x_labels: List[str],
    y_labels: List[str],
    title: str,
    cbar_label: str,
    annotate: bool = False,
    fmt: str = ".1f",
    cmap: str = "YlOrRd",
    vmin: float = None,
    vmax: float = None,
) -> None:
    """Render heatmap with matplotlib only."""
    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    if annotate:
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                ax.text(col, row, format(data[row, col], fmt), ha="center", va="center", fontsize=8)


def plot_ecdf(ax: plt.Axes, values: np.ndarray, label: str, color: str, linewidth: float = 2.0) -> None:
    """Plot an empirical cumulative distribution function."""
    values = np.asarray(values)
    if values.size == 0:
        return
    x = np.sort(values)
    y = np.arange(1, values.size + 1) / float(values.size)
    ax.plot(x, y, label=label, color=color, linewidth=linewidth)


def load_dataset(dataset_root: Union[str, Path]) -> Dict:
    """Load the dataset using PIL so the new script stays self-contained."""
    dataset_root = Path(dataset_root)
    validation = validate_dataset(dataset_root)
    if not validation["valid"]:
        raise ValueError("Dataset validation failed:\n" + "\n".join(validation["errors"]))

    dataset = {
        "train": _load_split(dataset_root / "train"),
        "test": _load_split(dataset_root / "test"),
        "metadata": {
            "train_count": int(validation["train_scenes"]),
            "test_count": int(validation["test_scenes"]),
            "total_count": int(validation["total_scenes"]),
            "dataset_root": str(dataset_root),
        },
    }
    return dataset


def _load_split(split_dir: Path) -> List[Dict]:
    """Load a dataset split without requiring OpenCV."""
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    image_files = sorted(
        set(images_dir.glob("*.jpg")) |
        set(images_dir.glob("*.jpeg")) |
        set(images_dir.glob("*.png")) |
        set(images_dir.glob("*.tif")) |
        set(images_dir.glob("*.tiff"))
    )

    scenes = []
    for img_path in image_files:
        image = np.array(Image.open(img_path).convert("L"))
        mask_path = labels_dir / f"{img_path.stem}.png"
        if not mask_path.exists():
            raise ValueError(f"Missing mask for image: {img_path.name}")
        mask_rgb = np.array(Image.open(mask_path).convert("RGB"))
        mask_1d = remap_rgb_to_1d(mask_rgb)
        scenes.append({
            "image": image,
            "mask": mask_1d,
            "name": img_path.stem,
        })
    return scenes


def connected_component_areas(binary_mask: np.ndarray) -> List[int]:
    """Return connected-component areas using cv2 or scipy fallback."""
    if cv2 is not None:
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary_mask.astype(np.uint8), connectivity=8)
        return [int(stats[label_idx, cv2.CC_STAT_AREA]) for label_idx in range(1, num_labels)]

    if ndimage is None:
        raise RuntimeError("Connected components require either cv2 or scipy.ndimage")

    structure = np.ones((3, 3), dtype=np.uint8)
    labeled, num_labels = ndimage.label(binary_mask, structure=structure)
    if num_labels == 0:
        return []
    component_ids = np.arange(1, num_labels + 1)
    areas = ndimage.sum(binary_mask, labeled, index=component_ids)
    return [int(area) for area in np.asarray(areas).tolist() if int(area) > 0]


def mask_1d_to_rgb(mask_1d: np.ndarray) -> np.ndarray:
    """Convert class IDs back to RGB colors for visualization."""
    rgb = np.zeros((mask_1d.shape[0], mask_1d.shape[1], 3), dtype=np.uint8)
    for class_id, color in CLASS_COLOR_MAP.items():
        rgb[mask_1d == class_id] = color
    return rgb


def compute_scene_metrics(dataset_data: Dict) -> pd.DataFrame:
    """Build a scene-level metrics table for plotting and selection."""
    rows = []

    for split_name in ["train", "test"]:
        for scene in dataset_data[split_name]:
            mask = scene["mask"]
            dist = compute_class_distribution(mask)
            total_pixels = int(mask.size)
            present_classes = [class_id for class_id, count in dist.items() if count > 0]

            row = {
                "split": split_name,
                "scene_name": scene["name"],
                "total_pixels": total_pixels,
                "num_present_classes": len(present_classes),
                "non_sea_present_count": sum(1 for class_id in TARGET_CLASSES if dist[class_id] > 0),
            }

            for class_id in range(5):
                count = int(dist[class_id])
                row[f"class_{class_id}_pixels"] = count
                row[f"class_{class_id}_pct"] = count / total_pixels * 100.0
                row[f"class_{class_id}_present"] = int(count > 0)

            rows.append(row)

    return pd.DataFrame(rows)


def select_representative_scenes(
    dataset_data: Dict,
    metrics_df: pd.DataFrame
) -> List[Tuple[str, Dict]]:
    """
    Select five representative train scenes for Figure 3.1.

    The selection is heuristic but deterministic:
    - typical oil scene: oil area closest to median among oil-positive scenes
    - difficult look-alike scene: both oil and look-alike present with strong overlap
    - ship scene: highest ship area among ship-positive scenes
    - land scene: highest land area among land-positive scenes
    - mixed-complexity scene: most non-sea classes present, then highest combined minority area
    """
    train_rows = metrics_df[metrics_df["split"] == "train"].copy()
    train_lookup = {scene["name"]: scene for scene in dataset_data["train"]}
    selected_names = []

    def pick_row(candidates: pd.DataFrame, score_columns: List[str], ascending: List[bool]) -> pd.Series:
        remaining = candidates[~candidates["scene_name"].isin(selected_names)].copy()
        if remaining.empty:
            remaining = candidates.copy()
        return remaining.sort_values(score_columns, ascending=ascending).iloc[0]

    oil_rows = train_rows[train_rows["class_1_present"] == 1].copy()
    oil_median = float(oil_rows["class_1_pct"].median())
    oil_rows["median_distance"] = (oil_rows["class_1_pct"] - oil_median).abs()
    typical_oil = pick_row(
        oil_rows,
        ["median_distance", "class_2_present", "class_4_present"],
        [True, True, True],
    )
    selected_names.append(typical_oil["scene_name"])

    look_rows = train_rows[
        (train_rows["class_1_present"] == 1) &
        (train_rows["class_2_present"] == 1)
    ].copy()
    look_rows["oil_look_balance"] = np.minimum(look_rows["class_1_pct"], look_rows["class_2_pct"])
    difficult_look = pick_row(
        look_rows,
        ["oil_look_balance", "class_1_pct", "class_2_pct"],
        [False, False, False],
    )
    selected_names.append(difficult_look["scene_name"])

    ship_rows = train_rows[train_rows["class_3_present"] == 1].copy()
    ship_scene = pick_row(
        ship_rows,
        ["class_3_pct", "class_2_pct", "class_4_pct"],
        [False, False, False],
    )
    selected_names.append(ship_scene["scene_name"])

    land_rows = train_rows[train_rows["class_4_present"] == 1].copy()
    land_scene = pick_row(
        land_rows,
        ["class_4_pct", "class_2_pct"],
        [False, False],
    )
    selected_names.append(land_scene["scene_name"])

    mixed_rows = train_rows[train_rows["non_sea_present_count"] >= 3].copy()
    mixed_rows["combined_target_pct"] = (
        mixed_rows["class_1_pct"] +
        mixed_rows["class_2_pct"] +
        mixed_rows["class_3_pct"] +
        mixed_rows["class_4_pct"]
    )
    mixed_scene = pick_row(
        mixed_rows,
        ["non_sea_present_count", "combined_target_pct"],
        [False, False],
    )
    selected_names.append(mixed_scene["scene_name"])

    scene_specs = [
        ("Typical oil spill scene", train_lookup[typical_oil["scene_name"]]),
        ("Oil spill / look-alike confusion", train_lookup[difficult_look["scene_name"]]),
        ("Ship-containing scene", train_lookup[ship_scene["scene_name"]]),
        ("Coastal scene with land", train_lookup[land_scene["scene_name"]]),
        ("Mixed-complexity scene", train_lookup[mixed_scene["scene_name"]]),
    ]
    return scene_specs


def render_representative_panel(scene_specs, output_dir: Union[str, Path]) -> None:
    """Draw the SAR + ground-truth mask panel; scene_specs is a list of (label, {"name", "image", "mask"}).

    Layout: no suptitle (the caption is in the thesis), a left-aligned title above each SAR image, enough vertical
    space between rows so that titles never touch the previous row, and a legend with the real mask colours.
    """
    from matplotlib.patches import Patch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n = len(scene_specs)
    fig, axes = plt.subplots(n, 2, figsize=(11, 3.3 * n), gridspec_kw={"hspace": 0.32, "wspace": 0.04})
    for row_idx, (label, scene) in enumerate(scene_specs):
        present = [CLASS_NAMES[c] for c in TARGET_CLASSES if np.any(scene["mask"] == c)]
        present_text = ", ".join(present) if present else "Sea only"
        ax_img, ax_mask = axes[row_idx, 0], axes[row_idx, 1]
        ax_img.imshow(scene["image"], cmap="gray")
        ax_img.set_title(f"{chr(65 + row_idx)}. {label} ({scene['name']})", fontsize=11, fontweight="bold", loc="left", pad=6)
        ax_mask.imshow(mask_1d_to_rgb(scene["mask"]))
        ax_mask.set_title(f"Mask classes: {present_text}", fontsize=10, loc="left", pad=6)
        ax_img.axis("off")
        ax_mask.axis("off")
    handles = [Patch(facecolor=CLASS_HEX_MAP[c], edgecolor="0.5", label=CLASS_NAMES[c]) for c in sorted(CLASS_COLOR_MAP)]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=11, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.045)
    plt.savefig(output_dir / "01_representative_scenes_and_masks.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_representative_scenes(
    dataset_data: Dict,
    metrics_df: pd.DataFrame,
    output_dir: Union[str, Path]
) -> None:
    """Create a curated SAR + mask panel for Chapter 3."""
    setup_publication_style()
    render_representative_panel(select_representative_scenes(dataset_data, metrics_df), output_dir)


def representative_scenes_from_files(dataset_root: Union[str, Path], names: List[Tuple[str, str]], split: str = "train"):
    """Load only the named scenes (cheap alternative to load_dataset) for re-drawing the panel."""
    root = Path(dataset_root) / split
    specs = []
    for label, name in names:
        image = np.asarray(Image.open(root / "images" / f"{name}.jpg").convert("L"))
        mask = np.asarray(Image.open(root / "labels_1D" / f"{name}.png"))
        specs.append((label, {"name": name, "image": image, "mask": mask}))
    return specs


def plot_pixel_class_distribution(
    metrics_df: pd.DataFrame,
    output_dir: Union[str, Path]
) -> None:
    """Plot pixel-level distribution with a minority-class panel."""
    setup_publication_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for split_name in ["train", "test"]:
        split_rows = metrics_df[metrics_df["split"] == split_name]
        total_pixels = split_rows["total_pixels"].sum()
        for class_id in range(5):
            pixels = split_rows[f"class_{class_id}_pixels"].sum()
            rows.append({
                "Split": split_name.capitalize(),
                "Class": CLASS_NAMES[class_id],
                "Pixels (%)": pixels / total_pixels * 100.0,
                "IsMinority": class_id != 0,
            })

    df = pd.DataFrame(rows)
    df_minor = df[df["IsMinority"]].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.1, 1.0]})
    if sns is not None:
        sns.barplot(data=df, x="Class", y="Pixels (%)", hue="Split", palette="Set2", ax=axes[0])
        sns.barplot(data=df_minor, x="Class", y="Pixels (%)", hue="Split", palette="Set2", ax=axes[1])
    else:
        plot_grouped_bars(
            axes[0], df, "Class", "Pixels (%)", "Split",
            [CLASS_NAMES[class_id] for class_id in range(5)],
            ["Train", "Test"],
            ["#66c2a5", "#fc8d62"],
        )
        plot_grouped_bars(
            axes[1], df_minor, "Class", "Pixels (%)", "Split",
            [CLASS_NAMES[class_id] for class_id in TARGET_CLASSES],
            ["Train", "Test"],
            ["#66c2a5", "#fc8d62"],
        )
    axes[0].set_title("All classes")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Pixels (%)")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].set_title("Minority classes only")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Pixels (%)")
    axes[1].tick_params(axis="x", rotation=20)

    handles, labels = axes[0].get_legend_handles_labels()
    if sns is not None and axes[0].legend_ is not None:
        axes[0].legend_.remove()
    axes[1].legend(handles, labels, title="Split")
    fig.suptitle("Pixel-level class distribution in train and test splits", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "02_pixel_class_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_scene_class_presence(metrics_df: pd.DataFrame, output_dir: Union[str, Path]) -> None:
    """Plot the percentage of scenes containing each class."""
    setup_publication_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for split_name in ["train", "test"]:
        split_rows = metrics_df[metrics_df["split"] == split_name]
        total_scenes = len(split_rows)
        for class_id in range(5):
            present_count = split_rows[f"class_{class_id}_present"].sum()
            rows.append({
                "Split": split_name.capitalize(),
                "Class": CLASS_NAMES[class_id],
                "Scenes (%)": present_count / total_scenes * 100.0,
                "Scene Count": int(present_count),
            })

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(11, 6))
    if sns is not None:
        sns.barplot(data=df, x="Class", y="Scenes (%)", hue="Split", palette="Set2", ax=ax)
    else:
        plot_grouped_bars(
            ax, df, "Class", "Scenes (%)", "Split",
            [CLASS_NAMES[class_id] for class_id in range(5)],
            ["Train", "Test"],
            ["#66c2a5", "#fc8d62"],
        )
    ax.set_title("Scene-level class presence across the dataset", fontsize=16, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Scenes containing class (%)")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="Split")
    plt.tight_layout()
    plt.savefig(output_dir / "03_scene_class_presence.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_per_scene_area_distribution(metrics_df: pd.DataFrame, output_dir: Union[str, Path]) -> None:
    """Plot non-zero per-scene area distributions for minority classes."""
    setup_publication_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for split_name in ["train", "test"]:
        split_rows = metrics_df[metrics_df["split"] == split_name]
        for class_id in TARGET_CLASSES:
            pct_series = split_rows[f"class_{class_id}_pct"]
            for pct in pct_series[pct_series > 0]:
                rows.append({
                    "Split": split_name.capitalize(),
                    "Class": CLASS_NAMES[class_id],
                    "Area (%)": float(pct),
                })

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(12, 6))
    if sns is not None:
        sns.boxplot(data=df, x="Class", y="Area (%)", hue="Split", palette="Set2", ax=ax, showfliers=False)
    else:
        plot_grouped_boxplots(
            ax, df,
            [CLASS_NAMES[class_id] for class_id in TARGET_CLASSES],
            ["Train", "Test"],
            ["#66c2a5", "#fc8d62"],
        )
    ax.set_title("Per-scene area distribution of minority classes", fontsize=16, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Area per scene (%)")
    ax.set_yscale("log")
    ax.legend(title="Split")
    plt.tight_layout()
    plt.savefig(output_dir / "04_per_scene_area_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_class_cooccurrence(metrics_df: pd.DataFrame, output_dir: Union[str, Path]) -> None:
    """Plot a conditional class co-occurrence matrix at scene level."""
    setup_publication_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    presence = np.zeros((5, len(metrics_df)), dtype=np.uint8)
    for class_id in range(5):
        presence[class_id] = metrics_df[f"class_{class_id}_present"].to_numpy(dtype=np.uint8)

    conditional = np.zeros((5, 5), dtype=np.float64)
    for row_class in range(5):
        row_present = presence[row_class].sum()
        for col_class in range(5):
            if row_present == 0:
                conditional[row_class, col_class] = 0.0
            else:
                both = np.sum((presence[row_class] == 1) & (presence[col_class] == 1))
                conditional[row_class, col_class] = both / row_present * 100.0

    df = pd.DataFrame(
        conditional,
        index=[CLASS_NAMES[class_id] for class_id in range(5)],
        columns=[CLASS_NAMES[class_id] for class_id in range(5)],
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    if sns is not None:
        sns.heatmap(df, annot=True, fmt=".1f", cmap="YlOrRd", cbar_kws={"label": "Conditional probability (%)"}, ax=ax)
        ax.set_title("Class co-occurrence matrix at scene level", fontsize=16, fontweight="bold")
    else:
        plot_heatmap(
            ax,
            df.to_numpy(),
            list(df.columns),
            list(df.index),
            "Class co-occurrence matrix at scene level",
            "Conditional probability (%)",
            annotate=True,
            fmt=".1f",
        )
    ax.set_xlabel("Class B")
    ax.set_ylabel("Class A\n(Probability that B appears when A appears)")
    plt.tight_layout()
    plt.savefig(output_dir / "05_class_cooccurrence_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()


def compute_binned_occupancy(mask: np.ndarray, class_id: int, bin_size: int) -> np.ndarray:
    """Compute occupancy ratio per spatial bin for a single class in one scene."""
    height, width = mask.shape
    heat_h = height // bin_size
    heat_w = width // bin_size
    trimmed = mask[:heat_h * bin_size, :heat_w * bin_size]
    reshaped = trimmed.reshape(heat_h, bin_size, heat_w, bin_size)
    counts = np.sum(reshaped == class_id, axis=(1, 3))
    return counts / float(bin_size * bin_size)


def plot_normalized_spatial_heatmaps(dataset_data: Dict, output_dir: Union[str, Path]) -> None:
    """Plot normalized spatial heatmaps for Oil Spill and Look-alike."""
    setup_publication_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_classes = [1, 2]
    bin_size = 10
    heatmaps = {}
    counts = {}

    for class_id in selected_classes:
        aggregate = None
        num_scenes = 0
        for scene in dataset_data["train"]:
            mask = scene["mask"]
            if not np.any(mask == class_id):
                continue
            occupancy = compute_binned_occupancy(mask, class_id, bin_size)
            if aggregate is None:
                aggregate = np.zeros_like(occupancy, dtype=np.float64)
            aggregate += occupancy
            num_scenes += 1

        if aggregate is None or num_scenes == 0:
            raise ValueError(f"No scenes found for class {CLASS_NAMES[class_id]}")

        heatmaps[class_id] = aggregate / num_scenes * 100.0
        counts[class_id] = num_scenes

    vmax = max(float(np.max(heatmaps[class_id])) for class_id in selected_classes)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, class_id in zip(axes, selected_classes):
        title = f"{CLASS_NAMES[class_id]} (train scenes with class, n={counts[class_id]})"
        if sns is not None:
            sns.heatmap(
                heatmaps[class_id],
                cmap="YlOrRd",
                vmin=0.0,
                vmax=vmax,
                cbar_kws={"label": "Average occupancy within bin (%)"},
                ax=ax,
            )
            ax.set_title(title, fontweight="bold")
        else:
            plot_heatmap(
                ax,
                heatmaps[class_id],
                [str(i) for i in range(heatmaps[class_id].shape[1])],
                [str(i) for i in range(heatmaps[class_id].shape[0])],
                title,
                "Average occupancy within bin (%)",
                annotate=False,
                vmin=0.0,
                vmax=vmax,
            )
        ax.set_xlabel("Scene width (10-pixel bins)")
        ax.set_ylabel("Scene height (10-pixel bins)")

    fig.suptitle("Normalized spatial heatmaps for key classes", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "06_normalized_spatial_heatmaps.png", dpi=300, bbox_inches="tight")
    plt.close()


def sample_class_intensities(
    dataset_data: Dict,
    class_ids: List[int],
    max_scenes: int = 120,
    max_pixels_per_class: int = 120000
) -> Dict[int, np.ndarray]:
    """Sample SAR intensities for selected classes from the train split."""
    sampled = {class_id: [] for class_id in class_ids}
    train_scenes = dataset_data["train"]

    if len(train_scenes) > max_scenes:
        indices = np.linspace(0, len(train_scenes) - 1, max_scenes, dtype=int)
        scenes = [train_scenes[index] for index in indices]
    else:
        scenes = train_scenes

    for scene in scenes:
        image = scene["image"]
        mask = scene["mask"]
        for class_id in class_ids:
            pixels = image[mask == class_id]
            if pixels.size > 0:
                sampled[class_id].append(pixels)

    out = {}
    for class_id, chunks in sampled.items():
        if not chunks:
            out[class_id] = np.array([], dtype=np.uint8)
            continue
        values = np.concatenate(chunks).astype(np.uint8)
        if values.size > max_pixels_per_class:
            indices = RNG.choice(values.size, size=max_pixels_per_class, replace=False)
            values = values[indices]
        out[class_id] = values
    return out


def plot_intensity_pairwise_comparison(dataset_data: Dict, output_dir: Union[str, Path]) -> None:
    """Plot ECDF-based pairwise SAR intensity comparisons."""
    setup_publication_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    intensities = sample_class_intensities(dataset_data, class_ids=[0, 1, 2])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    panel_specs = [
        (axes[0], [1, 2], "Oil Spill vs Look-alike"),
        (axes[1], [1, 0], "Oil Spill vs Sea Surface"),
    ]

    for ax, class_ids, title in panel_specs:
        for class_id in class_ids:
            label = f"{CLASS_NAMES[class_id]} (n={len(intensities[class_id]):,})"
            if sns is not None:
                sns.ecdfplot(
                    intensities[class_id],
                    ax=ax,
                    label=label,
                    color=CLASS_HEX_MAP[class_id],
                    linewidth=2.0,
                )
            else:
                plot_ecdf(ax, intensities[class_id], label, CLASS_HEX_MAP[class_id], linewidth=2.0)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("SAR intensity (0-255)")
        ax.set_ylabel("ECDF")
        ax.legend()

    fig.suptitle("SAR intensity comparison between key classes", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "07_intensity_pairwise_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def extract_component_areas(dataset_data: Dict, class_ids: List[int]) -> pd.DataFrame:
    """Extract connected-component areas per class from all scenes."""
    rows = []

    for split_name in ["train", "test"]:
        for scene in dataset_data[split_name]:
            mask = scene["mask"]
            for class_id in class_ids:
                binary = (mask == class_id).astype(np.uint8)
                if binary.sum() == 0:
                    continue

                for area in connected_component_areas(binary):
                    if area <= 0:
                        continue
                    rows.append({
                        "Split": split_name.capitalize(),
                        "Class": CLASS_NAMES[class_id],
                        "Area (pixels)": area,
                        "log10 Area": np.log10(area),
                    })

    return pd.DataFrame(rows)


def plot_object_size_distribution(dataset_data: Dict, output_dir: Union[str, Path]) -> None:
    """Plot connected-component object size distributions."""
    setup_publication_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = extract_component_areas(dataset_data, class_ids=[1, 2, 3])
    fig, ax = plt.subplots(figsize=(12, 6))
    if sns is not None:
        sns.ecdfplot(
            data=df,
            x="Area (pixels)",
            hue="Class",
            palette=[CLASS_HEX_MAP[1], CLASS_HEX_MAP[2], CLASS_HEX_MAP[3]],
            ax=ax,
        )
    else:
        for class_id in [1, 2, 3]:
            class_name = CLASS_NAMES[class_id]
            plot_ecdf(
                ax,
                df[df["Class"] == class_name]["Area (pixels)"].to_numpy(),
                class_name,
                CLASS_HEX_MAP[class_id],
                linewidth=2.0,
            )
        ax.legend()
    ax.set_xscale("log")
    ax.set_title("Object size distribution of annotated regions", fontsize=16, fontweight="bold")
    ax.set_xlabel("Connected-component area (pixels, log scale)")
    ax.set_ylabel("ECDF")
    plt.tight_layout()
    plt.savefig(output_dir / "08_object_size_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()


def generate_ch3_summary(metrics_df: pd.DataFrame, output_path: Union[str, Path]) -> None:
    """Save a compact JSON summary for the Chapter 3 figure pipeline."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "train_scenes": int((metrics_df["split"] == "train").sum()),
        "test_scenes": int((metrics_df["split"] == "test").sum()),
        "scene_presence": {},
    }

    for split_name in ["train", "test"]:
        split_rows = metrics_df[metrics_df["split"] == split_name]
        total_scenes = len(split_rows)
        summary["scene_presence"][split_name] = {
            CLASS_NAMES[class_id]: {
                "count": int(split_rows[f"class_{class_id}_present"].sum()),
                "percent": round(float(split_rows[f"class_{class_id}_present"].mean() * 100.0), 4),
            }
            for class_id in range(5)
        }
        summary[f"{split_name}_total_scenes"] = total_scenes

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)


def main(dataset_root: Union[str, Path], output_dir: Union[str, Path]) -> None:
    """Run the full Chapter 3 figure generation pipeline."""
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Chapter 3 Dataset Figures Pipeline")
    print("=" * 72)

    print("\n[1/10] Loading dataset...")
    dataset_data = load_dataset(dataset_root)

    print("\n[2/10] Computing scene-level metrics...")
    metrics_df = compute_scene_metrics(dataset_data)

    print("\n[3/10] Generating representative scene panel...")
    plot_representative_scenes(dataset_data, metrics_df, figures_dir)

    print("\n[4/10] Generating pixel-level class distribution...")
    plot_pixel_class_distribution(metrics_df, figures_dir)

    print("\n[5/10] Generating scene-level class presence...")
    plot_scene_class_presence(metrics_df, figures_dir)

    print("\n[6/10] Generating per-scene area distribution...")
    plot_per_scene_area_distribution(metrics_df, figures_dir)

    print("\n[7/10] Generating class co-occurrence matrix...")
    plot_class_cooccurrence(metrics_df, figures_dir)

    print("\n[8/10] Generating normalized spatial heatmaps...")
    plot_normalized_spatial_heatmaps(dataset_data, figures_dir)

    print("\n[9/10] Generating intensity and object-size figures...")
    plot_intensity_pairwise_comparison(dataset_data, figures_dir)
    plot_object_size_distribution(dataset_data, figures_dir)

    print("\n[10/10] Saving compact Chapter 3 summary...")
    generate_ch3_summary(metrics_df, output_dir / "ch3_figure_summary.json")

    print("\nPipeline complete.")
    print(f"Output directory: {output_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate revised Chapter 3 dataset figures for M4D-SAR-OilSpill."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to dataset root containing train/ and test/"
    )
    parser.add_argument(
        "--output",
        default="./results_ch3",
        help="Output directory for Chapter 3 figures"
    )

    args = parser.parse_args()
    main(args.input, args.output)
