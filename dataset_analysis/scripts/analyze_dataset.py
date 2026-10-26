"""
Dataset Analysis Pipeline - Part 1: Data Loading & Statistics
Part 2: Visualization Setup & Class Examples

Provides functions for:
- Loading the original M4D-SAR-OilSpill dataset with validation
- Computing class distribution statistics per split
- Setting up Seaborn for publication-quality visualizations
- Generating class example visualizations
"""

import os
import json
import argparse
import math
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Union, Any
from tqdm import tqdm

from utils import (
    validate_dataset,
    compute_class_distribution,
    CLASS_NAMES,
    remap_rgb_to_1d
)


# ============================================================================
# CONSTANTS
# ============================================================================

# Color mapping for visualization consistency
CLASS_COLOR_MAP = {
    0: (0, 0, 0),           # Black → Sea Surface
    1: (0, 255, 255),       # Cyan → Oil Spill
    2: (255, 0, 0),         # Red → Look-alike
    3: (153, 76, 0),        # Brown → Ship
    4: (0, 153, 0)          # Green → Land
}


# ============================================================================
# TASK 3: DATA LOADING & STATISTICS
# ============================================================================

def load_original_dataset(dataset_root: Union[str, Path]) -> Dict:
    """
    Load M4D-SAR-OilSpill dataset with validation.

    Loads all training and test scenes from disk, validates dataset structure,
    and returns a structured dict with image/mask pairs and metadata.

    Parameters
    ----------
    dataset_root : str or Path
        Path to dataset root directory containing train/ and test/ splits.

    Returns
    -------
    Dict
        Dictionary with keys:
        - 'train': list of {'image': np.ndarray, 'mask': np.ndarray, 'name': str}
        - 'test': list of {'image': np.ndarray, 'mask': np.ndarray, 'name': str}
        - 'metadata': dict with 'train_count', 'test_count', 'total_count'

    Raises
    ------
    ValueError
        If dataset validation fails.

    Examples
    --------
    >>> data = load_original_dataset("../oil-spill-detection-dataset")
    >>> len(data['train'])
    1002
    >>> len(data['test'])
    110
    """
    dataset_root = Path(dataset_root)

    # Validate dataset first
    print("Validating dataset structure...")
    validation = validate_dataset(dataset_root)

    if not validation['valid']:
        errors = "\n".join(validation['errors'])
        raise ValueError(f"Dataset validation failed:\n{errors}")

    print(f"✓ Validation passed: {validation['total_scenes']} scenes found")

    # Load training split
    print("\nLoading training split...")
    train_data = _load_split(
        dataset_root / "train",
        split_name="train",
        expected_count=validation['train_scenes']
    )

    # Load test split
    print("\nLoading test split...")
    test_data = _load_split(
        dataset_root / "test",
        split_name="test",
        expected_count=validation['test_scenes']
    )

    # Assemble return dict
    dataset = {
        'train': train_data,
        'test': test_data,
        'metadata': {
            'train_count': len(train_data),
            'test_count': len(test_data),
            'total_count': len(train_data) + len(test_data),
            'dataset_root': str(dataset_root)
        }
    }

    return dataset


def _load_split(split_dir: Path, split_name: str, expected_count: int) -> List[Dict]:
    """
    Load image/mask pairs from a single split directory.

    Parameters
    ----------
    split_dir : Path
        Path to split directory (e.g., train/ or test/)
    split_name : str
        Name of split for logging ("train" or "test")
    expected_count : int
        Expected number of scenes (for consistency check)

    Returns
    -------
    List[Dict]
        List of scene dicts with keys: 'image', 'mask', 'name'
    """
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels_1D"  # ✅ FIXED: Use labels_1D instead of labels (RGB)

    # Get image files
    image_files = sorted(set(images_dir.glob("*.jpg")) | set(images_dir.glob("*.jpeg")))

    if len(image_files) != expected_count:
        raise ValueError(
            f"{split_name.upper()} split: expected {expected_count} images, "
            f"found {len(image_files)}"
        )

    scenes = []
    pbar = tqdm(image_files, desc=f"Loading {split_name}", unit="scene")

    for img_path in pbar:
        # Load image (SAR grayscale)
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")

        # Load mask (1D labels - already in correct format)
        mask_path = labels_dir / (img_path.stem + ".png")
        if not mask_path.exists():
            raise ValueError(f"Missing mask for image: {img_path.name}")

        # Load 1D mask directly (grayscale, class indices 0-4)
        mask_1d = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_1d is None:
            raise ValueError(f"Failed to load mask: {mask_path}")

        scenes.append({
            'image': image,
            'mask': mask_1d,
            'name': img_path.stem
        })

    return scenes


def compute_class_statistics(dataset_data: Dict) -> Dict:
    """
    Compute class distribution statistics for train and test splits.

    Aggregates pixel counts per class across all scenes in each split,
    computes percentages, and calculates imbalance ratios.

    Parameters
    ----------
    dataset_data : Dict
        Dataset dict from load_original_dataset with 'train' and 'test' keys.

    Returns
    -------
    Dict
        Statistics dict with keys:
        - 'train': dict with class stats for training split
        - 'test': dict with class stats for test split
        - 'summary': overall summary statistics

        Each split dict has keys per class:
        - 'pixel_count': total pixels of class
        - 'pixel_percent': percentage of total pixels
        - 'scene_count': number of scenes containing class
        - 'imbalance_ratio': (sea_pixels / class_pixels) ratio

    Examples
    --------
    >>> stats = compute_class_statistics(data)
    >>> stats['train']['Oil Spill']['pixel_percent']
    0.83
    """
    stats = {
        'train': {},
        'test': {},
        'summary': {}
    }

    for split_name in ['train', 'test']:
        split_data = dataset_data[split_name]
        class_pixels = {class_id: 0 for class_id in range(5)}
        class_scene_counts = {class_id: 0 for class_id in range(5)}

        # Aggregate statistics
        for scene in split_data:
            mask = scene['mask']
            dist = compute_class_distribution(mask)

            for class_id, count in dist.items():
                class_pixels[class_id] += count
                if count > 0:
                    class_scene_counts[class_id] += 1

        # Compute percentages and imbalance ratios
        total_pixels = sum(class_pixels.values())

        for class_id in range(5):
            class_name = CLASS_NAMES[class_id]
            pixel_count = class_pixels[class_id]
            pixel_pct = (pixel_count / total_pixels * 100) if total_pixels > 0 else 0

            # Imbalance ratio: sea surface pixels / class pixels
            sea_pixels = class_pixels[0]
            if pixel_count > 0:
                if sea_pixels > 0:
                    imbalance = round(sea_pixels / pixel_count, 2)
                else:
                    imbalance = 0.0  # Edge case: no sea pixels in dataset
            else:
                imbalance = float('inf')

            stats[split_name][class_name] = {
                'pixel_count': pixel_count,
                'pixel_percent': round(pixel_pct, 4),
                'scene_count': class_scene_counts[class_id],
                'imbalance_ratio': round(imbalance, 2) if imbalance != float('inf') else float('inf')
            }

        # Add summary for split
        stats[split_name]['_metadata'] = {
            'total_pixels': total_pixels,
            'total_scenes': len(split_data)
        }

    # Global summary
    train_total = stats['train']['_metadata']['total_pixels']
    test_total = stats['test']['_metadata']['total_pixels']

    stats['summary'] = {
        'train_total_pixels': train_total,
        'test_total_pixels': test_total,
        'combined_total_pixels': train_total + test_total,
        'train_scenes': stats['train']['_metadata']['total_scenes'],
        'test_scenes': stats['test']['_metadata']['total_scenes'],
        'combined_scenes': (
            stats['train']['_metadata']['total_scenes'] +
            stats['test']['_metadata']['total_scenes']
        )
    }

    return stats


# ============================================================================
# TASK 4: VISUALIZATION SETUP & CLASS EXAMPLES
# ============================================================================

def setup_seaborn_style() -> None:
    """
    Configure Seaborn and Matplotlib for publication-quality figures.

    Sets:
    - Figure DPI: 300
    - Style: whitegrid with light background
    - Font sizes: title (14), labels (11), ticks (10)
    - Legend and grid styling
    """
    # Set DPI for high resolution
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300

    # Seaborn style
    sns.set_style("whitegrid")
    sns.set_palette("husl")

    # Font sizes
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10

    # Grid and spines
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False

    # Figure layout
    plt.rcParams['figure.autolayout'] = True


def plot_class_examples(
    dataset_data: Dict,
    output_dir: Union[str, Path],
    examples_per_class: int = 5
) -> None:
    """
    Generate visualization of class examples: SAR + RGB mask + 1D labels.

    Creates a figure with one row per class, showing:
    - Column 1: Original SAR image (grayscale)
    - Column 2: RGB color mask
    - Column 3: 1D class labels (one-hot encoded colors)

    Parameters
    ----------
    dataset_data : Dict
        Dataset dict from load_original_dataset.
    output_dir : str or Path
        Output directory for figure (created if missing).
    examples_per_class : int, default=5
        Currently unused (reserved for future multi-example display).
        Currently shows the first scene containing each class.

    Returns
    -------
    None
        Saves figure as 01_class_examples.png to output_dir.

    Notes
    -----
    Currently shows first scene containing each class (examples_per_class
    parameter is for future extension).
    Uses setup_seaborn_style() for consistent formatting.
    """
    setup_seaborn_style()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find first scene with each class
    train_data = dataset_data['train']
    class_examples = {class_id: None for class_id in range(5)}

    for scene in train_data:
        mask = scene['mask']
        unique_classes = np.unique(mask)
        for class_id in unique_classes:
            if class_examples[class_id] is None:
                class_examples[class_id] = scene
        if all(v is not None for v in class_examples.values()):
            break

    # Check if all classes found
    missing_classes = [CLASS_NAMES[i] for i in range(5) if class_examples[i] is None]
    if missing_classes:
        raise ValueError(
            f"Could not find examples for classes: {missing_classes}. "
            f"Please ensure all 5 classes are present in the training dataset."
        )

    # Create figure: 5 rows (classes) × 3 cols (SAR + RGB mask + 1D labels)
    fig, axes = plt.subplots(5, 3, figsize=(12, 16))

    for class_id in range(5):
        scene = class_examples[class_id]
        if scene is None:
            print(f"Warning: No example found for class {class_id} ({CLASS_NAMES[class_id]})")
            continue

        image = scene['image']
        mask_1d = scene['mask']
        name = scene['name']

        # Column 1: SAR image
        ax = axes[class_id, 0]
        ax.imshow(image, cmap='gray')
        ax.set_title(f"{CLASS_NAMES[class_id]}", fontweight='bold')
        ax.set_ylabel(f"SAR Image ({image.shape[0]}×{image.shape[1]})")
        ax.axis('off')

        # Column 2: RGB mask overlay on image
        ax = axes[class_id, 1]
        # Recreate RGB mask from 1D labels for visualization
        mask_rgb = _mask_1d_to_rgb(mask_1d)
        ax.imshow(mask_rgb)
        ax.set_ylabel("RGB Mask")
        ax.axis('off')

        # Column 3: 1D class labels (color-coded)
        ax = axes[class_id, 2]
        # Show class label distribution
        label_counts = np.bincount(mask_1d.flatten(), minlength=5)
        # Use CLASS_COLOR_MAP for consistency
        colors = ['#{:02x}{:02x}{:02x}'.format(*rgb) for rgb in CLASS_COLOR_MAP.values()]
        ax.bar(range(5), label_counts, color=colors, alpha=0.7)
        ax.set_xticks(range(5))
        ax.set_xticklabels([f"C{i}" for i in range(5)], fontsize=9)
        ax.set_ylabel("Pixel Count")
        ax.set_title(name, fontsize=9)

    plt.tight_layout()
    output_path = output_dir / "01_class_examples.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Generated: {output_path}")
    plt.close()


def _mask_1d_to_rgb(mask_1d: np.ndarray) -> np.ndarray:
    """
    Convert 1D class labels back to RGB color map for visualization.

    Parameters
    ----------
    mask_1d : np.ndarray
        Class label array of shape (H, W) with values 0-4.

    Returns
    -------
    np.ndarray
        RGB array of shape (H, W, 3) with color values 0-255.
    """
    h, w = mask_1d.shape
    mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, rgb in CLASS_COLOR_MAP.items():
        mask_rgb[mask_1d == class_id] = rgb

    return mask_rgb


# ============================================================================
# TASK 5: DISTRIBUTION & HEATMAP VISUALIZATIONS
# ============================================================================

def plot_class_distribution(stats: Dict, output_dir: Union[str, Path]) -> None:
    """
    Generate barplot of class distribution across train and test splits.

    Creates a grouped barplot showing pixel percentage for each class,
    with separate bars for train and test splits.

    Parameters
    ----------
    stats : Dict
        Statistics dict from compute_class_statistics.
    output_dir : str or Path
        Output directory for figure (created if missing).

    Returns
    -------
    None
        Saves figure as 02_class_distribution.png to output_dir.

    Notes
    -----
    Uses Seaborn barplot with 'Set2' palette.
    X-axis: 5 classes, Y-axis: Pixels (%).
    """
    setup_seaborn_style()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data for plotting
    data_list = []
    for split_name in ['train', 'test']:
        for class_id in range(5):
            class_name = CLASS_NAMES[class_id]
            pixel_pct = stats[split_name][class_name]['pixel_percent']
            data_list.append({
                'Class': class_name,
                'Pixels (%)': pixel_pct,
                'Split': split_name.capitalize()
            })

    df = pd.DataFrame(data_list)

    # Create barplot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df, x='Class', y='Pixels (%)', hue='Split', palette='Set2', ax=ax)

    ax.set_title('Class Distribution: Train vs Test', fontsize=14, fontweight='bold')
    ax.set_xlabel('Class', fontsize=11)
    ax.set_ylabel('Pixels (%)', fontsize=11)
    ax.legend(title='Split', fontsize=10)

    plt.tight_layout()
    output_path = output_dir / "02_class_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Generated: {output_path}")
    plt.close()


def plot_spatial_heatmap(dataset_data: Dict, output_dir: Union[str, Path]) -> None:
    """
    Generate spatial heatmaps showing class pixel distribution across scenes.

    Divides 1250×650 scenes into 10×10 pixel bins and counts pixels per class per bin.
    Creates 5 separate heatmaps (one per class).

    Parameters
    ----------
    dataset_data : Dict
        Dataset dict from load_original_dataset.
    output_dir : str or Path
        Output directory for figures (created if missing).

    Returns
    -------
    None
        Saves 5 figures: 04_spatial_heatmap_class0.png ... class4.png to output_dir.

    Notes
    -----
    Method: Divide scenes into 10×10 pixel bins, count pixels per class.
    Results in 65×125 heatmap grid for each class.
    Colormap: 'YlOrRd'.
    Uses tqdm progress bar for scene iteration.
    Optimized: uses vectorized numpy operations for performance.
    """
    setup_seaborn_style()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Scene dimensions from M4D-SAR-OilSpill: 1250×650 (width × height)
    scene_height, scene_width = 650, 1250  # H=650, W=1250
    bin_size = 10
    heatmap_h = scene_height // bin_size  # 65
    heatmap_w = scene_width // bin_size   # 125

    # Initialize heatmaps for each class
    heatmaps = {class_id: np.zeros((heatmap_h, heatmap_w)) for class_id in range(5)}

    # Aggregate spatial statistics (using train set with sampling for efficiency)
    # Sample scenes to make computation tractable while preserving spatial patterns
    all_scenes = dataset_data['train']
    # Use all scenes if < 300, else sample 300 scenes uniformly
    if len(all_scenes) > 300:
        sample_indices = np.linspace(0, len(all_scenes) - 1, 300, dtype=int)
        sampled_scenes = [all_scenes[i] for i in sample_indices]
    else:
        sampled_scenes = all_scenes

    pbar = tqdm(sampled_scenes, desc="Computing spatial heatmaps", unit="scene")

    for scene in pbar:
        mask = scene['mask']

        # Reshape mask to separate bins: shape becomes (heatmap_h, bin_size, heatmap_w, bin_size)
        mask_reshaped = mask.reshape(heatmap_h, bin_size, heatmap_w, bin_size)

        # For each class, count occurrences in each bin (vectorized)
        for class_id in range(5):
            # Count where mask == class_id, summed over bin dimensions
            counts = np.sum(mask_reshaped == class_id, axis=(1, 3))
            heatmaps[class_id] += counts

    # Generate figure per class
    for class_id in range(5):
        class_name = CLASS_NAMES[class_id]
        heatmap = heatmaps[class_id]

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(heatmap, cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Pixel Count'})

        ax.set_title(f'Spatial Distribution: {class_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Scene Width (10-pixel bins)', fontsize=11)
        ax.set_ylabel('Scene Height (10-pixel bins)', fontsize=11)

        plt.tight_layout()
        output_path = output_dir / f"04_spatial_heatmap_class{class_id}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Generated: {output_path}")
        plt.close()


# ============================================================================
# TASK 6: PIXEL INTENSITY & IMBALANCE VISUALIZATIONS
# ============================================================================

def plot_pixel_intensity_distribution(dataset_data: Dict, output_dir: Union[str, Path]) -> None:
    """
    Generate histogram of pixel intensity per class with KDE overlay.

    Creates a single figure with overlaid histograms for all 5 classes,
    showing intensity distribution (0-255) with kernel density estimation.

    Parameters
    ----------
    dataset_data : Dict
        Dataset dict from load_original_dataset.
    output_dir : str or Path
        Output directory for figure (created if missing).

    Returns
    -------
    None
        Saves figure as 05_pixel_intensity_distribution.png to output_dir.

    Notes
    -----
    X-axis: Intensity (0-255), Y-axis: Density.
    hue='Class' with 5 colors, kde=True.
    Palette: 'husl'.
    Uses sampling to manage memory: samples up to 100K pixels per class.
    """
    setup_seaborn_style()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect intensity data per class (with sampling for efficiency)
    # Use train set only with scene sampling to keep computation tractable
    intensity_dict = {class_id: [] for class_id in range(5)}
    all_scenes = dataset_data['train']

    # Sample scenes uniformly if more than 100
    if len(all_scenes) > 100:
        sample_indices = np.linspace(0, len(all_scenes) - 1, 100, dtype=int)
        sampled_scenes = [all_scenes[i] for i in sample_indices]
    else:
        sampled_scenes = all_scenes

    pbar = tqdm(sampled_scenes, desc="Collecting pixel intensities", unit="scene")
    for scene in pbar:
        image = scene['image']
        mask = scene['mask']

        for class_id in range(5):
            # Get pixels belonging to this class (vectorized)
            class_pixels = image[mask == class_id]
            intensity_dict[class_id].extend(class_pixels.flatten().tolist())

    # Create DataFrame from collected intensities (sample if too large)
    intensity_list = []
    for class_id in range(5):
        class_name = CLASS_NAMES[class_id]
        intensities = intensity_dict[class_id]

        # Sample if more than 100K pixels per class
        if len(intensities) > 100000:
            sample_indices = np.random.choice(len(intensities), 100000, replace=False)
            intensities = [intensities[i] for i in sample_indices]

        for intensity in intensities:
            intensity_list.append({
                'Intensity': int(intensity),
                'Class': class_name
            })

    df = pd.DataFrame(intensity_list)

    # Create histogram with KDE
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(data=df, x='Intensity', hue='Class', kde=True, palette='husl', ax=ax)

    ax.set_title('Pixel Intensity Distribution by Class', fontsize=14, fontweight='bold')
    ax.set_xlabel('Intensity (0-255)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.legend(title='Class', fontsize=10)

    plt.tight_layout()
    output_path = output_dir / "05_pixel_intensity_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Generated: {output_path}")
    plt.close()


def plot_imbalance_violin(stats: Dict, output_dir: Union[str, Path]) -> None:
    """
    Generate violin plot showing class imbalance across train and test splits.

    Creates a violin plot with classes on X-axis, pixel percentage on Y-axis,
    and separate violins for train and test splits.

    Parameters
    ----------
    stats : Dict
        Statistics dict from compute_class_statistics.
    output_dir : str or Path
        Output directory for figure (created if missing).

    Returns
    -------
    None
        Saves figure as 06_class_imbalance_violin.png to output_dir.

    Notes
    -----
    X-axis: Class, Y-axis: Pixels (%).
    hue='Split' (Train/Test).
    Palette: 'muted'.
    """
    setup_seaborn_style()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data for plotting
    data_list = []
    for split_name in ['train', 'test']:
        for class_id in range(5):
            class_name = CLASS_NAMES[class_id]
            pixel_pct = stats[split_name][class_name]['pixel_percent']
            data_list.append({
                'Class': class_name,
                'Pixels (%)': pixel_pct,
                'Split': split_name.capitalize()
            })

    df = pd.DataFrame(data_list)

    # Create violin plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=df, x='Class', y='Pixels (%)', hue='Split', palette='muted', ax=ax)

    ax.set_title('Class Imbalance: Train vs Test', fontsize=14, fontweight='bold')
    ax.set_xlabel('Class', fontsize=11)
    ax.set_ylabel('Pixels (%)', fontsize=11)
    ax.legend(title='Split', fontsize=10)

    plt.tight_layout()
    output_path = output_dir / "06_class_imbalance_violin.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Generated: {output_path}")
    plt.close()


# ============================================================================
# TASK 7: TABLE GENERATION & JSON EXPORT
# ============================================================================

def generate_statistics_tables(stats: Dict, output_dir: Union[str, Path]) -> None:
    """
    Generate CSV, Markdown, and LaTeX statistics tables from stats dict.

    Creates three table formats for class distribution statistics:
    - CSV: machine-readable format
    - Markdown: for documentation/notes
    - LaTeX: for dissertation Chapter 3

    Parameters
    ----------
    stats : Dict
        Statistics dict from compute_class_statistics with structure:
        - stats['train'/'test'][class_name] with keys: pixel_count, pixel_percent,
          scene_count, imbalance_ratio
        - stats['summary'] with global summary statistics

    output_dir : str or Path
        Output directory for tables (created if missing).
        Tables are saved to output_dir/tables/ subdirectory.
        Files: statistics_table.csv, statistics_table.md, statistics_table.tex

    Returns
    -------
    None
        Saves three table files to output_dir/tables/.

    Raises
    ------
    ValueError
        If stats dict is missing required keys: 'train', 'test', 'summary'.

    Notes
    -----
    - Percentages displayed with 4 decimal places (e.g., 98.1700)
    - Imbalance ratio: 2 decimal places, inf shown as "inf" (CSV), "∞" (Markdown), "\infty" (LaTeX)
    - LaTeX table uses standard tabular environment with horizontal lines
    """
    output_dir = Path(output_dir) / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate stats dict structure
    required_keys = {'train', 'test', 'summary'}
    if not all(key in stats for key in required_keys):
        missing = required_keys - set(stats.keys())
        raise ValueError(f"Stats dict missing required keys: {missing}")

    # Prepare table data with extracted row generation logic
    table_rows = []
    formatted_rows = []  # For format-specific output

    for class_id in range(5):
        class_name = CLASS_NAMES[class_id]

        train_stats = stats['train'][class_name]
        test_stats = stats['test'][class_name]

        train_pixels = train_stats['pixel_count']
        train_pct = train_stats['pixel_percent']
        test_pixels = test_stats['pixel_count']
        test_pct = test_stats['pixel_percent']

        # Imbalance ratio (handle inf)
        imbalance = train_stats['imbalance_ratio']  # Use train for primary metric
        imbalance_str = "inf" if imbalance == float('inf') else f"{imbalance:.2f}"

        table_rows.append({
            'Class': class_name,
            'Train Pixels': train_pixels,
            'Train %': f"{train_pct:.4f}",
            'Test Pixels': test_pixels,
            'Test %': f"{test_pct:.4f}",
            'Imbalance Ratio': imbalance_str
        })

        # Pre-extract formatted row data for efficiency
        formatted_rows.append({
            'class': class_name,
            'train_px': f"{int(train_pixels):,}",
            'train_pct': f"{train_pct:.4f}",
            'test_px': f"{int(test_pixels):,}",
            'test_pct': f"{test_pct:.4f}",
            'imbalance': imbalance_str
        })

    # Create DataFrame for CSV output
    df = pd.DataFrame(table_rows)

    # ========== CSV Output ==========
    csv_path = output_dir / "statistics_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ Generated CSV: {csv_path}")

    # ========== Markdown Output ==========
    md_path = output_dir / "statistics_table.md"
    with open(md_path, 'w') as f:
        f.write("# Dataset Statistics Table\n\n")
        f.write("## Class Distribution (Train vs Test)\n\n")

        # Manually create Markdown table
        # Header
        f.write("| Class | Train Pixels | Train % | Test Pixels | Test % | Imbalance Ratio |\n")
        f.write("|-------|-------------:|--------:|------------:|-------:|----------------:|\n")

        # Rows - use pre-extracted formatted data
        for row in formatted_rows:
            imbalance_display = "∞" if row['imbalance'] == "inf" else row['imbalance']
            f.write(f"| {row['class']} | {row['train_px']} | {row['train_pct']} | {row['test_px']} | {row['test_pct']} | {imbalance_display} |\n")

        f.write("\n## Summary\n\n")
        f.write(f"- Train total pixels: {stats['summary']['train_total_pixels']:,}\n")
        f.write(f"- Test total pixels: {stats['summary']['test_total_pixels']:,}\n")
        f.write(f"- Combined total pixels: {stats['summary']['combined_total_pixels']:,}\n")
        f.write(f"- Train scenes: {stats['summary']['train_scenes']}\n")
        f.write(f"- Test scenes: {stats['summary']['test_scenes']}\n")
        f.write(f"- Combined scenes: {stats['summary']['combined_scenes']}\n")

    print(f"✅ Generated Markdown: {md_path}")

    # ========== LaTeX Output ==========
    latex_path = output_dir / "statistics_table.tex"
    with open(latex_path, 'w') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Dataset Statistics: Class Distribution (Train vs Test)}\n")
        f.write("\\label{tab:dataset_statistics}\n")
        f.write("\\begin{tabular}{l|r|r|r|r|c}\n")
        f.write("\\hline\n")
        f.write("Class & Train Pixels & Train \\% & Test Pixels & Test \\% & Imbalance Ratio \\\\\n")
        f.write("\\hline\n")

        # Rows - use pre-extracted formatted data
        for row in formatted_rows:
            imbalance_display = "\\infty" if row['imbalance'] == "inf" else row['imbalance']
            f.write(
                f"{row['class']} & {row['train_px']} & {row['train_pct']} & {row['test_px']} & {row['test_pct']} & {imbalance_display} \\\\\n"
            )

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"✅ Generated LaTeX: {latex_path}")


def convert_inf(obj: Union[Dict, List, float, Any]) -> Union[Dict, List, str, Any]:
    """
    Recursively convert float('inf') to string 'inf' for JSON serialization.

    Parameters
    ----------
    obj : Dict, List, float, or other
        Object to convert

    Returns
    -------
    Converted object with all float('inf') → 'inf'
    """
    if isinstance(obj, dict):
        return {k: convert_inf(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_inf(item) for item in obj]
    elif isinstance(obj, float) and math.isinf(obj):
        return "inf"
    return obj


def generate_json_summary(stats: Dict, output_path: Union[str, Path]) -> None:
    """
    Export full statistics dict to JSON for programmatic access.

    Converts stats dict to JSON with special handling for infinite values.

    Parameters
    ----------
    stats : Dict
        Statistics dict from compute_class_statistics.
    output_path : str or Path
        Output file path (typically results/analysis_summary.json).
        Parent directory created if missing.

    Returns
    -------
    None
        Saves JSON file to output_path.

    Notes
    -----
    - float('inf') values converted to string "inf" for JSON compatibility
    - Output is pretty-printed with indent=2 for readability
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert stats dict, handling infinity values
    stats_converted = convert_inf(stats)

    # Write JSON
    with open(output_path, 'w') as f:
        json.dump(stats_converted, f, indent=2)

    print(f"✅ Generated JSON: {output_path}")


def main(dataset_root: Union[str, Path], output_dir: Union[str, Path]) -> None:
    """
    Execute the complete dataset analysis pipeline.

    Orchestrates all analysis stages:
    1. Load dataset
    2. Compute statistics
    3. Generate visualizations (class examples, distribution, heatmap, intensity, imbalance)
    4. Generate statistics tables (CSV, Markdown, LaTeX)
    5. Export JSON summary

    Parameters
    ----------
    dataset_root : str or Path
        Path to dataset root directory (contains train/ and test/ splits).
    output_dir : str or Path
        Root output directory for all results (figures/, tables/, JSON).
        Created if missing.

    Returns
    -------
    None
        Prints pipeline progress and summary.

    Raises
    ------
    ValueError
        If dataset validation or loading fails.
    """
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"

    # Header
    print("=" * 70)
    print("Dataset Analysis Pipeline v1.0")
    print("=" * 70)

    # Step 1: Load dataset
    print("\n[1/10] Loading dataset...")
    data = load_original_dataset(dataset_root)
    print(f"✅ Loaded {data['metadata']['train_count']} train scenes")
    print(f"✅ Loaded {data['metadata']['test_count']} test scenes")

    # Step 2: Compute statistics
    print("\n[2/10] Computing class statistics...")
    stats = compute_class_statistics(data)
    oil_train_pct = stats['train']['Oil Spill']['pixel_percent']
    oil_test_pct = stats['test']['Oil Spill']['pixel_percent']
    print(f"✅ Oil Spill: {oil_train_pct:.4f}% (train), {oil_test_pct:.4f}% (test)")

    # Step 3: Class examples
    print("\n[3/10] Generating class examples...")
    plot_class_examples(data, figures_dir)

    # Step 4: Class distribution
    print("\n[4/10] Generating class distribution plot...")
    plot_class_distribution(stats, figures_dir)

    # Step 5: Spatial heatmap
    print("\n[5/10] Generating spatial heatmaps...")
    plot_spatial_heatmap(data, figures_dir)

    # Step 6: Pixel intensity distribution
    print("\n[6/10] Generating pixel intensity distribution...")
    plot_pixel_intensity_distribution(data, figures_dir)

    # Step 7: Imbalance violin plot
    print("\n[7/10] Generating imbalance violin plot...")
    plot_imbalance_violin(stats, figures_dir)

    # Step 8: Statistics tables
    print("\n[8/10] Generating statistics tables...")
    generate_statistics_tables(stats, output_dir)

    # Step 9: JSON summary
    print("\n[9/10] Generating JSON summary...")
    generate_json_summary(stats, output_dir / "analysis_summary.json")

    # Step 10: Final summary
    print("\n[10/10] Pipeline complete!")
    print("=" * 70)
    print("✅ Dataset analysis pipeline complete!")
    print("=" * 70)
    print(f"📊 Generated: 8 visualizations, 3 tables, 1 JSON")
    print(f"📁 Output directory: {output_dir.resolve()}")
    print("=" * 70)


# ============================================================================
# MAIN: SMOKE TEST
# ============================================================================

if __name__ == "__main__":
    import sys

    parser = argparse.ArgumentParser(
        description="Dataset Analysis Pipeline - Load, analyze, visualize M4D-SAR-OilSpill dataset"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to dataset root"
    )
    parser.add_argument(
        "--output",
        default="./results",
        help="Output directory for results (default: ./results)"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run smoke tests for all tasks (default if no --input provided)"
    )

    args = parser.parse_args()

    # Run smoke test if --smoke-test flag is set
    if args.smoke_test:
        print("=" * 70)
        print("SMOKE TEST: All Tasks")
        print("=" * 70)

        try:
            # Test full pipeline orchestration via main()
            print("\n[FULL PIPELINE] Running complete analysis pipeline...")
            print("-" * 70)
            main("../../oil-spill-detection-dataset", "./results")

            print("\n" + "=" * 70)
            print("ALL SMOKE TESTS PASSED ✅")
            print("=" * 70)

        except Exception as e:
            print(f"\n❌ Smoke test failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    else:
        # Run full pipeline with argparse
        try:
            main(args.input, args.output)
        except Exception as e:
            print(f"\n❌ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
