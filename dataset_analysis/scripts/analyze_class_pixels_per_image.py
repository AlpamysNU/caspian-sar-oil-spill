"""
Analyze pixel distribution by class for each image in the dataset.

For each image: count pixels of each class (Sea, Oil, Look-alike, Ship, Land)
Then: compute statistics (mean, median, min, max, percentiles, etc.)
Output: detailed CSV, JSON, and console report
"""

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[2]))   # repository root
from segmentation import paths
import numpy as np
import pandas as pd
import json
from pathlib import Path
from PIL import Image

# Paths
TRAIN_DIR = paths.DATASET / "train/labels"
TEST_DIR = paths.DATASET / "test/labels"
OUTPUT_DIR = paths.PROJECT_ROOT / "dataset_analysis_output" / "results_ch3"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Class definitions
CLASS_COLORS = {
    0: (0, 0, 0),           # Sea - BLACK
    1: (0, 255, 255),       # Oil - CYAN
    2: (255, 0, 0),         # Look-alike - RED
    3: (153, 76, 0),        # Ship - BROWN
    4: (0, 153, 0)          # Land - GREEN
}

CLASS_NAMES = {
    0: "Sea (Ocean Surface)",
    1: "Oil Spill",
    2: "Look-alike",
    3: "Ship",
    4: "Land"
}

def count_class_pixels_per_image():
    """Count pixels of each class in every image."""
    print("Counting pixels by class for each image...\n")

    all_masks = sorted(TRAIN_DIR.glob("*.png")) + sorted(TEST_DIR.glob("*.png"))
    print(f"Found {len(all_masks)} images\n")

    # Store results
    image_stats = []

    for i, mask_file in enumerate(all_masks):
        if (i + 1) % 200 == 0:
            print(f"  Processed {i + 1}/{len(all_masks)}")

        mask = np.array(Image.open(mask_file))

        # Count pixels per class
        image_data = {
            'filename': mask_file.name,
            'total_pixels': mask.shape[0] * mask.shape[1]
        }

        for class_id, class_name in CLASS_NAMES.items():
            color = CLASS_COLORS[class_id]

            if len(mask.shape) == 3 and mask.shape[2] == 3:
                class_mask = np.all(mask == color, axis=2)
            else:
                class_mask = (mask == class_id)

            pixel_count = np.sum(class_mask)
            density = 100 * pixel_count / image_data['total_pixels']

            image_data[f'{class_name}_pixels'] = int(pixel_count)
            image_data[f'{class_name}_density_%'] = float(density)

        image_stats.append(image_data)

    return image_stats


def compute_statistics(image_stats):
    """Compute aggregate statistics for each class."""
    print(f"\n{'='*90}")
    print(f"PIXEL-LEVEL STATISTICS BY CLASS (per image)")
    print(f"{'='*90}\n")

    results = {}

    for class_id, class_name in CLASS_NAMES.items():
        pixel_col = f'{class_name}_pixels'
        density_col = f'{class_name}_density_%'

        pixels = np.array([img[pixel_col] for img in image_stats])
        densities = np.array([img[density_col] for img in image_stats])

        print(f"\n{class_id}. {class_name}")
        print(f"   {'-' * 85}")

        # Pixel count statistics
        print(f"   PIXEL COUNT STATISTICS (all {len(image_stats)} images):")
        print(f"     Mean:       {np.mean(pixels):>15,.0f} pixels/image")
        print(f"     Median:     {np.median(pixels):>15,.0f} pixels/image")
        print(f"     Std Dev:    {np.std(pixels):>15,.0f}")
        print(f"     Min:        {np.min(pixels):>15,.0f} pixels")
        print(f"     Max:        {np.max(pixels):>15,.0f} pixels")
        print(f"     P25:        {np.percentile(pixels, 25):>15,.0f} pixels")
        print(f"     P50:        {np.percentile(pixels, 50):>15,.0f} pixels")
        print(f"     P75:        {np.percentile(pixels, 75):>15,.0f} pixels")
        print(f"     P90:        {np.percentile(pixels, 90):>15,.0f} pixels")
        print(f"     P95:        {np.percentile(pixels, 95):>15,.0f} pixels")
        print(f"     P99:        {np.percentile(pixels, 99):>15,.0f} pixels")

        # Density statistics
        print(f"\n   DENSITY STATISTICS (% of image area):")
        print(f"     Mean:       {np.mean(densities):>15.2f}%")
        print(f"     Median:     {np.median(densities):>15.2f}%")
        print(f"     Std Dev:    {np.std(densities):>15.2f}%")
        print(f"     Min:        {np.min(densities):>15.2f}%")
        print(f"     Max:        {np.max(densities):>15.2f}%")
        print(f"     P90:        {np.percentile(densities, 90):>15.2f}%")
        print(f"     P95:        {np.percentile(densities, 95):>15.2f}%")

        # Presence statistics
        images_with_class = np.sum(pixels > 0)
        print(f"\n   PRESENCE:")
        print(f"     Images containing class: {images_with_class}/{len(image_stats)} ({100*images_with_class/len(image_stats):.1f}%)")

        if images_with_class > 0:
            pixels_present = pixels[pixels > 0]
            densities_present = densities[pixels > 0]
            print(f"\n   STATISTICS FOR IMAGES WITH CLASS (only {images_with_class} images):")
            print(f"     Mean (when present):   {np.mean(pixels_present):>15,.0f} pixels")
            print(f"     Median (when present): {np.median(pixels_present):>15,.0f} pixels")
            print(f"     Min:                   {np.min(pixels_present):>15,.0f} pixels")
            print(f"     Max:                   {np.max(pixels_present):>15,.0f} pixels")
            print(f"     Mean density:          {np.mean(densities_present):>15.2f}%")

        # Store results
        results[class_name] = {
            'pixel_count': {
                'mean': float(np.mean(pixels)),
                'median': float(np.median(pixels)),
                'std': float(np.std(pixels)),
                'min': float(np.min(pixels)),
                'max': float(np.max(pixels)),
                'p25': float(np.percentile(pixels, 25)),
                'p50': float(np.percentile(pixels, 50)),
                'p75': float(np.percentile(pixels, 75)),
                'p90': float(np.percentile(pixels, 90)),
                'p95': float(np.percentile(pixels, 95)),
                'p99': float(np.percentile(pixels, 99)),
            },
            'density_percent': {
                'mean': float(np.mean(densities)),
                'median': float(np.median(densities)),
                'std': float(np.std(densities)),
                'min': float(np.min(densities)),
                'max': float(np.max(densities)),
                'p90': float(np.percentile(densities, 90)),
                'p95': float(np.percentile(densities, 95)),
            },
            'presence': {
                'images_with_class': int(images_with_class),
                'total_images': len(image_stats),
                'presence_percent': float(100 * images_with_class / len(image_stats))
            }
        }

        if images_with_class > 0:
            pixels_present = pixels[pixels > 0]
            densities_present = densities[pixels > 0]
            results[class_name]['when_present'] = {
                'mean_pixels': float(np.mean(pixels_present)),
                'median_pixels': float(np.median(pixels_present)),
                'min_pixels': float(np.min(pixels_present)),
                'max_pixels': float(np.max(pixels_present)),
                'mean_density': float(np.mean(densities_present))
            }

    return results


def save_results(image_stats, statistics):
    """Save results to CSV and JSON."""
    print(f"\n{'='*90}")
    print("SAVING RESULTS")
    print(f"{'='*90}\n")

    # Convert to DataFrame and save CSV
    df = pd.DataFrame(image_stats)
    csv_file = OUTPUT_DIR / "class_pixels_per_image.csv"
    df.to_csv(csv_file, index=False)
    print(f"✓ Saved: {csv_file}")
    print(f"  (Contains pixel counts and densities for all {len(df)} images)")

    # Save summary statistics as JSON
    json_file = OUTPUT_DIR / "class_pixel_statistics_summary.json"
    with open(json_file, 'w') as f:
        json.dump(statistics, f, indent=2)
    print(f"✓ Saved: {json_file}")

    # Create summary table
    print(f"\n{'='*90}")
    print("SUMMARY TABLE (Mean values)")
    print(f"{'='*90}\n")

    summary_data = []
    for class_name in CLASS_NAMES.values():
        if class_name in statistics:
            stats = statistics[class_name]
            summary_data.append({
                'Class': class_name,
                'Mean Pixels': f"{stats['pixel_count']['mean']:,.0f}",
                'Median Pixels': f"{stats['pixel_count']['median']:,.0f}",
                'Min': f"{stats['pixel_count']['min']:,.0f}",
                'Max': f"{stats['pixel_count']['max']:,.0f}",
                'P95': f"{stats['pixel_count']['p95']:,.0f}",
                'Mean Density %': f"{stats['density_percent']['mean']:.2f}",
                'Presence %': f"{stats['presence']['presence_percent']:.1f}"
            })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    summary_csv = OUTPUT_DIR / "class_pixels_summary_table.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n✓ Saved: {summary_csv}")


def create_latex_table(statistics):
    """Generate LaTeX table for dissertation."""
    print(f"\n{'='*90}")
    print("LATEX TABLE (for Chapter 3)")
    print(f"{'='*90}\n")

    latex = r"""
\begin{table}[h]
\centering
\caption{M4D-SAR-OilSpill Dataset: Class Pixel Distribution (per image)}
\label{tab:class_pixel_dist}
\small
\begin{tabular}{lrrrrrr}
\toprule
\textbf{Class} & \textbf{Mean} & \textbf{Median} & \textbf{Min} & \textbf{Max} & \textbf{P95} & \textbf{Presence} \\
 & \textbf{(px)} & \textbf{(px)} & \textbf{(px)} & \textbf{(px)} & \textbf{(px)} & \textbf{(\%)} \\
\midrule
"""

    for class_name in CLASS_NAMES.values():
        if class_name in statistics:
            stats = statistics[class_name]
            latex += f"""{class_name:20s} & \
{stats['pixel_count']['mean']:>10,.0f} & \
{stats['pixel_count']['median']:>10,.0f} & \
{stats['pixel_count']['min']:>10,.0f} & \
{stats['pixel_count']['max']:>10,.0f} & \
{stats['pixel_count']['p95']:>10,.0f} & \
{stats['presence']['presence_percent']:>8.1f} \\\\
"""

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    print(latex)

    # Save to file
    latex_file = OUTPUT_DIR / "class_pixel_distribution_latex.txt"
    with open(latex_file, 'w') as f:
        f.write(latex)
    print(f"✓ Saved: {latex_file}")


def main():
    print("="*90)
    print("ANALYZING CLASS PIXEL DISTRIBUTION ACROSS ALL IMAGES")
    print("="*90)

    # Count pixels
    image_stats = count_class_pixels_per_image()
    print(f"\n✓ Analyzed {len(image_stats)} images")

    # Compute statistics
    statistics = compute_statistics(image_stats)

    # Save results
    save_results(image_stats, statistics)

    # LaTeX table
    create_latex_table(statistics)

    print(f"\n{'='*90}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*90}\n")


if __name__ == '__main__':
    main()
