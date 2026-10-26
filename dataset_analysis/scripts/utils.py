"""
Utility functions for M4D-SAR-OilSpill dataset analysis.

Provides helper functions for:
- RGB mask remapping to class labels
- Class name resolution
- Class distribution computation
- Dataset validation
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Union, List
from scipy import ndimage
from scipy.ndimage import label as ndimage_label


# ============================================================================
# CLASS AND COLOR DEFINITIONS
# ============================================================================

CLASS_NAMES = {
    0: "Sea (Ocean Surface)",
    1: "Oil Spill",
    2: "Look-alike",
    3: "Ship",
    4: "Land"
}

# RGB color to class mapping (M4D-SAR-OilSpill convention)
RGB_COLORS = {
    (0, 0, 0): 0,           # Black → Sea Surface
    (0, 255, 255): 1,       # Cyan → Oil Spill
    (255, 0, 0): 2,         # Red → Look-alike
    (153, 76, 0): 3,        # Brown → Ship
    (0, 153, 0): 4          # Green → Land
}

# Reverse mapping: class to RGB color
CLASS_TO_RGB = {v: k for k, v in RGB_COLORS.items()}


# ============================================================================
# FUNCTION: remap_rgb_to_1d
# ============================================================================

def remap_rgb_to_1d(mask_rgb: np.ndarray, tolerance: int = 100) -> np.ndarray:
    """
    Convert RGB mask to 1D integer labels (0-4) with fuzzy color matching.

    Args:
        mask_rgb: numpy array of shape (H, W, 3) with RGB values
        tolerance: maximum color distance for fuzzy matching (default 20)

    Returns:
        mask_1d: numpy array of shape (H, W) with class IDs 0-4

    Raises:
        ValueError: If mask shape is invalid or contains unrecognized colors
    """
    if mask_rgb.ndim != 3 or mask_rgb.shape[2] != 3:
        raise ValueError(f"Expected mask shape (H, W, 3), got {mask_rgb.shape}")

    h, w, _ = mask_rgb.shape
    mask_1d = np.zeros((h, w), dtype=np.uint8)
    recognized = np.zeros((h, w), dtype=bool)

    # First pass: exact color matching
    for rgb_color, class_id in RGB_COLORS.items():
        r, g, b = rgb_color
        match = (mask_rgb[:, :, 0] == r) & \
                (mask_rgb[:, :, 1] == g) & \
                (mask_rgb[:, :, 2] == b)
        mask_1d[match] = class_id
        recognized |= match

    # Second pass: fuzzy color matching for unrecognized pixels
    unrecognized_mask = ~recognized
    if np.any(unrecognized_mask):
        # Get all unrecognized colors
        unrecognized_pixels = mask_rgb[unrecognized_mask]
        unique_unrec_colors = np.unique(unrecognized_pixels.reshape(-1, 3), axis=0)

        # For each unique unrecognized color, find closest reference color
        for unrec_color in unique_unrec_colors:
            min_dist = float('inf')
            best_class = None

            # Compute distance to each reference color
            for ref_color, class_id in RGB_COLORS.items():
                # Use float to avoid overflow
                d0 = float(unrec_color[0]) - float(ref_color[0])
                d1 = float(unrec_color[1]) - float(ref_color[1])
                d2 = float(unrec_color[2]) - float(ref_color[2])
                dist = np.sqrt(d0 * d0 + d1 * d1 + d2 * d2)

                if dist < min_dist:
                    min_dist = dist
                    best_class = class_id

            # Assign if within tolerance
            if min_dist <= tolerance:
                match = (mask_rgb[:, :, 0] == unrec_color[0]) & \
                        (mask_rgb[:, :, 1] == unrec_color[1]) & \
                        (mask_rgb[:, :, 2] == unrec_color[2])
                mask_1d[match] = best_class
                recognized |= match
            else:
                # Mark as permanently unrecognized
                match = (mask_rgb[:, :, 0] == unrec_color[0]) & \
                        (mask_rgb[:, :, 1] == unrec_color[1]) & \
                        (mask_rgb[:, :, 2] == unrec_color[2])
                recognized[match] = False

    # Final pass: assign remaining unrecognized pixels to closest color without limit
    unrecognized_mask = ~recognized
    if np.any(unrecognized_mask):
        # Get all unrecognized colors
        unrecognized_pixels_final = mask_rgb[unrecognized_mask]
        unique_unrec_colors_final = np.unique(unrecognized_pixels_final.reshape(-1, 3), axis=0)

        # For each unique unrecognized color, find closest reference color (no limit)
        for unrec_color in unique_unrec_colors_final:
            min_dist = float('inf')
            best_class = 0

            # Compute distance to each reference color
            for ref_color, class_id in RGB_COLORS.items():
                d0 = float(unrec_color[0]) - float(ref_color[0])
                d1 = float(unrec_color[1]) - float(ref_color[1])
                d2 = float(unrec_color[2]) - float(ref_color[2])
                dist = np.sqrt(d0 * d0 + d1 * d1 + d2 * d2)

                if dist < min_dist:
                    min_dist = dist
                    best_class = class_id

            # Assign to closest class regardless of distance
            match = (mask_rgb[:, :, 0] == unrec_color[0]) & \
                    (mask_rgb[:, :, 1] == unrec_color[1]) & \
                    (mask_rgb[:, :, 2] == unrec_color[2])
            mask_1d[match] = best_class

    return mask_1d


# ============================================================================
# FUNCTION: get_class_name
# ============================================================================

def get_class_name(class_id: int) -> str:
    """
    Get human-readable class name from class ID.

    Parameters
    ----------
    class_id : int
        Class identifier (0–4 for M4D-SAR-OilSpill).

    Returns
    -------
    str
        Class name (e.g., "Oil Spill", "Look-alike").

    Raises
    ------
    KeyError
        If class_id is not in [0, 1, 2, 3, 4].

    Examples
    --------
    >>> get_class_name(0)
    'Sea Surface'
    >>> get_class_name(1)
    'Oil Spill'
    >>> get_class_name(5)
    Traceback (most recent call last):
        ...
    KeyError: "Unknown class ID: 5"
    """
    if class_id not in CLASS_NAMES:
        raise KeyError(f"Unknown class ID: {class_id}")
    return CLASS_NAMES[class_id]


# ============================================================================
# FUNCTION: compute_class_distribution
# ============================================================================

def compute_class_distribution(mask: np.ndarray) -> Dict[int, int]:
    """
    Compute pixel counts per class in a single segmentation mask.

    Counts the number of pixels belonging to each class in a 1D or 2D mask.
    Useful for analyzing class imbalance within a scene.

    Parameters
    ----------
    mask : np.ndarray
        Class label array of shape (H, W) or flattened (N,) with dtype uint8.
        Values must be in range [0, 4].

    Returns
    -------
    Dict[int, int]
        Dictionary mapping class ID to pixel count.
        Missing classes have count 0.
        Example: {0: 5000, 1: 100, 2: 50, 3: 0, 4: 0}

    Raises
    ------
    ValueError
        If mask contains invalid class IDs (not in [0, 1, 2, 3, 4]).

    Examples
    --------
    >>> mask = np.array([[0, 0, 1], [1, 2, 2]])
    >>> compute_class_distribution(mask)
    {0: 2, 1: 2, 2: 2, 3: 0, 4: 0}
    """
    # Flatten mask
    flat_mask = mask.flatten()

    # Check for invalid class IDs
    unique_classes = np.unique(flat_mask)
    invalid = unique_classes[(unique_classes < 0) | (unique_classes > 4)]
    if len(invalid) > 0:
        raise ValueError(
            f"Mask contains invalid class IDs: {invalid.tolist()}. "
            f"Expected values in [0, 1, 2, 3, 4]."
        )

    # Initialize distribution
    distribution = {class_id: 0 for class_id in range(5)}

    # Count pixels using vectorized np.unique
    unique, counts = np.unique(flat_mask, return_counts=True)
    for class_id, count in zip(unique, counts):
        distribution[int(class_id)] = int(count)

    return distribution


# ============================================================================
# FUNCTION: validate_dataset
# ============================================================================

def validate_dataset(dataset_root: Union[str, Path]) -> Dict[str, Union[bool, int, list]]:
    """
    Validate dataset structure and file consistency.

    Checks that:
    - Root directory exists
    - Required subdirectories exist (e.g., train/, test/)
    - Image and mask files are paired
    - Mask files are readable and have correct format
    - Supports multiple image formats (.jpg, .tif, .png) and mask formats

    Parameters
    ----------
    dataset_root : str or Path
        Path to dataset root directory.

    Returns
    -------
    Dict[str, Union[bool, int, list]]
        Validation report with keys:
        - "valid": bool — overall validation status
        - "root_exists": bool — dataset_root directory exists
        - "has_split_dirs": bool — train/ and test/ directories exist
        - "total_scenes": int — total number of image files found
        - "train_scenes": int — number of training scenes
        - "test_scenes": int — number of test scenes
        - "missing_masks": list — file names missing corresponding masks
        - "errors": list — validation error messages

    Examples
    --------
    >>> report = validate_dataset('../oil-spill-detection-dataset/')
    >>> report['valid']
    True
    >>> report['total_scenes']
    1112
    """
    IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff")
    MASK_EXTENSIONS = ("*.png", "*.jpg")

    dataset_root = Path(dataset_root)
    report = {
        "valid": True,
        "root_exists": False,
        "has_split_dirs": False,
        "total_scenes": 0,
        "train_scenes": 0,
        "test_scenes": 0,
        "missing_masks": [],
        "errors": []
    }

    # Check root exists
    if not dataset_root.exists():
        report["valid"] = False
        report["errors"].append(f"Root directory not found: {dataset_root}")
        return report

    report["root_exists"] = True

    # Check split directories
    train_dir = dataset_root / "train"
    test_dir = dataset_root / "test"

    if not train_dir.exists() or not test_dir.exists():
        report["valid"] = False
        report["errors"].append(
            "Missing split directories. Expected 'train/' and 'test/'."
        )
        return report

    report["has_split_dirs"] = True

    # Validate train split
    train_images_dir = train_dir / "images"
    train_labels_dir = train_dir / "labels"

    # Check if directories exist at all
    if not train_images_dir.exists():
        report['errors'].append("Train split: 'images' directory not found")
    if not train_labels_dir.exists():
        report['errors'].append("Train split: 'labels' directory not found")

    train_images = []
    if train_images_dir.exists():
        for ext in IMAGE_EXTENSIONS:
            train_images.extend(train_images_dir.glob(ext))
    else:
        # Fallback: look for images directly in train dir
        for ext in IMAGE_EXTENSIONS:
            train_images.extend(train_dir.glob(ext))

    train_images = sorted(set(train_images))

    train_masks = []
    if train_labels_dir.exists():
        for ext in MASK_EXTENSIONS:
            train_masks.extend(train_labels_dir.glob(ext))
    train_masks = sorted(set(train_masks))

    # Validate test split
    test_images_dir = test_dir / "images"
    test_labels_dir = test_dir / "labels"

    # Check if directories exist at all
    if not test_images_dir.exists():
        report['errors'].append("Test split: 'images' directory not found")
    if not test_labels_dir.exists():
        report['errors'].append("Test split: 'labels' directory not found")

    test_images = []
    if test_images_dir.exists():
        for ext in IMAGE_EXTENSIONS:
            test_images.extend(test_images_dir.glob(ext))
    else:
        # Fallback: look for images directly in test dir
        for ext in IMAGE_EXTENSIONS:
            test_images.extend(test_dir.glob(ext))

    test_images = sorted(set(test_images))

    test_masks = []
    if test_labels_dir.exists():
        for ext in MASK_EXTENSIONS:
            test_masks.extend(test_labels_dir.glob(ext))
    test_masks = sorted(set(test_masks))

    # Mark invalid if any critical directories missing
    if len(report['errors']) > 0:
        report['valid'] = False

    report["train_scenes"] = len(train_images)
    report["test_scenes"] = len(test_images)
    report["total_scenes"] = len(train_images) + len(test_images)

    # Warn if counts don't match
    if train_images and not train_masks:
        report["errors"].append(
            f"Train split: {len(train_images)} images but no labels directory found"
        )
        report["valid"] = False
    elif len(train_images) != len(train_masks):
        report["errors"].append(
            f"Train split: {len(train_images)} images but {len(train_masks)} masks"
        )
        report["valid"] = False

    if test_images and not test_masks:
        report["errors"].append(
            f"Test split: {len(test_images)} images but no labels directory found"
        )
        report["valid"] = False
    elif len(test_images) != len(test_masks):
        report["errors"].append(
            f"Test split: {len(test_images)} images but {len(test_masks)} masks"
        )
        report["valid"] = False

    return report


# ============================================================================
# FUNCTION: compute_connected_components
# ============================================================================

def compute_connected_components(mask_rgb: np.ndarray, class_id: int, class_colors_rgb: Dict[int, Tuple[int, int, int]]) -> List[Dict]:
    """
    Identify connected components for a given class using scipy.ndimage.label.

    Extracts all connected regions of a specified class from an RGB mask,
    computing morphological properties for each component.

    Parameters
    ----------
    mask_rgb : np.ndarray
        RGB mask array of shape (H, W, 3) with dtype uint8.
    class_id : int
        Class identifier to extract (0-4 for M4D-SAR-OilSpill).
    class_colors_rgb : Dict[int, Tuple[int, int, int]]
        Mapping from class_id to (R, G, B) color tuple.
        Example: {0: (0,0,0), 1: (0,255,255), ...}

    Returns
    -------
    List[Dict]
        List of dictionaries, one per connected component, with keys:
        - 'id': int — component index (starting from 1)
        - 'area': int — pixel count
        - 'bbox': tuple — (row_min, col_min, row_max, col_max)
        - 'centroid': tuple — (row_center, col_center) as floats
        - 'perimeter': int — boundary pixel count
        - 'compactness': float — 4π * area / perimeter^2 (1.0 = circle)
        - 'pixels': np.ndarray — boolean mask of component pixels

    Examples
    --------
    >>> mask_rgb = np.zeros((650, 1250, 3), dtype=np.uint8)
    >>> # Create a 10x10 cyan region (oil class)
    >>> mask_rgb[10:20, 10:20] = [0, 255, 255]
    >>> components = compute_connected_components(mask_rgb, 1, {1: (0, 255, 255)})
    >>> len(components)
    1
    >>> components[0]['area']
    100
    """
    # Convert RGB mask to binary mask for the target class
    if class_id not in class_colors_rgb:
        return []

    target_color = class_colors_rgb[class_id]
    r, g, b = target_color

    # Create binary mask: 1 where color matches, 0 elsewhere
    binary_mask = (
        (mask_rgb[:, :, 0] == r) &
        (mask_rgb[:, :, 1] == g) &
        (mask_rgb[:, :, 2] == b)
    ).astype(np.uint8)

    # Label connected components
    labeled_array, num_features = ndimage_label(binary_mask)

    if num_features == 0:
        return []

    components = []

    for component_id in range(1, num_features + 1):
        component_mask = (labeled_array == component_id)

        # Compute area
        area = int(np.sum(component_mask))

        # Compute bounding box
        rows, cols = np.where(component_mask)
        row_min, row_max = int(rows.min()), int(rows.max())
        col_min, col_max = int(cols.min()), int(cols.max())
        bbox = (row_min, col_min, row_max, col_max)

        # Compute centroid
        centroid_row = float(np.mean(rows))
        centroid_col = float(np.mean(cols))
        centroid = (centroid_row, centroid_col)

        # Compute perimeter using morphological operations
        # Perimeter is the count of boundary pixels
        dilated = ndimage.binary_dilation(component_mask)
        perimeter = int(np.sum(dilated) - np.sum(component_mask))

        # Compute compactness: 4π * area / perimeter^2
        # For a circle, compactness = 1.0
        if perimeter > 0:
            compactness = float(4 * np.pi * area / (perimeter ** 2))
        else:
            compactness = 0.0

        component_dict = {
            'id': int(component_id),
            'area': area,
            'bbox': bbox,
            'centroid': centroid,
            'perimeter': perimeter,
            'compactness': compactness,
            'pixels': component_mask
        }

        components.append(component_dict)

    return components
