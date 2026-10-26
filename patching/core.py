"""
Core patching operations: extraction and reconstruction.

Implements reversible patch extraction with symmetric padding guarantee.
Coordinates tracked in both padded and original space for exact reconstruction.
"""

import logging
from pathlib import Path
from typing import NamedTuple, Optional, Tuple, List, Dict
import numpy as np
import pandas as pd
from scipy import signal
import json

from .config import PatchConfig

logger = logging.getLogger(__name__)

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


class PatchMetadata(NamedTuple):
    """Metadata for a single extracted patch with complete coordinate tracking."""

    patch_id: str
    scene_id: str
    split: str  # "train" or "test"

    # Padded space coordinates (sliding window grid)
    y_padded: int
    x_padded: int

    # Original space coordinates (unpadded image)
    y_orig: int
    x_orig: int

    # Padding amounts for this scene
    pad_top: int
    pad_bottom: int
    pad_left: int
    pad_right: int

    # Configuration
    patch_size: int
    stride: int

    # Patch properties
    is_border: bool

    # Class pixel counts (from mask)
    pixels_sea: int
    pixels_oil: int
    pixels_look_alike: int
    pixels_ship: int
    pixels_land: int

    # Class labels (1 if present, 0 otherwise)
    label_sea: int
    label_oil: int
    label_look_alike: int
    label_ship: int
    label_land: int


class PatchExtractor:
    """Extract patches from raw M4D scenes with reversible geometry.

    Features:
    - Symmetric padding ensures exact sliding-window coverage
    - Full coordinate tracking (padded + original space)
    - Separate padding modes for image and mask
    - Reversible reconstruction guaranteed
    """

    # M4D class mappings
    CLASS_NAMES = ["sea", "oil", "look_alike", "ship", "land"]
    CLASS_INDICES = {"sea": 0, "oil": 1, "look_alike": 2, "ship": 3, "land": 4}
    _MASK_PAD_MODES_WITH_CONSTANT_VALUES = {"constant"}
    M4D_RGB_COLORS = {
        "sea": np.array([0, 0, 0], dtype=np.uint8),
        "oil": np.array([0, 255, 255], dtype=np.uint8),
        "look_alike": np.array([255, 0, 0], dtype=np.uint8),
        "land": np.array([0, 153, 0], dtype=np.uint8),
    }

    def __init__(self, config: PatchConfig, output_dir: Path):
        """Initialize patch extractor.

        Args:
            config: PatchConfig specifying patch size, stride, etc.
            output_dir: Directory to save patches and indices.
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.patches_dir = self.output_dir / "patches_npy"
        self.patches_dir.mkdir(exist_ok=True)

        self.metadata_list: List[PatchMetadata] = []

    @classmethod
    def _is_oil_rgb(cls, rgb: np.ndarray) -> np.ndarray:
        return np.all(rgb == cls.M4D_RGB_COLORS["oil"], axis=-1)

    @classmethod
    def _is_look_alike_rgb(cls, rgb: np.ndarray) -> np.ndarray:
        return np.all(rgb == cls.M4D_RGB_COLORS["look_alike"], axis=-1)

    @classmethod
    def _is_land_rgb(cls, rgb: np.ndarray) -> np.ndarray:
        return np.all(rgb == cls.M4D_RGB_COLORS["land"], axis=-1)

    @classmethod
    def _is_sea_rgb(cls, rgb: np.ndarray) -> np.ndarray:
        return np.all(rgb == cls.M4D_RGB_COLORS["sea"], axis=-1)

    @staticmethod
    def _is_ship_rgb(rgb: np.ndarray) -> np.ndarray:
        """Robust brown-color rule matching the reference M4D code."""
        r = rgb[..., 0]
        g = rgb[..., 1]
        b = rgb[..., 2]
        return (
            (r >= 60) & (r <= 170) &
            (g >= 30) & (g <= 100) &
            (b <= 15)
        )

    @classmethod
    def _remap_rgb_mask_to_one_hot(cls, mask_rgb: np.ndarray) -> np.ndarray:
        """Convert an M4D RGB label image to 5-channel one-hot format."""
        h, w = mask_rgb.shape[:2]
        mask = np.zeros((h, w, 5), dtype=np.uint8)

        sea = cls._is_sea_rgb(mask_rgb)
        oil = cls._is_oil_rgb(mask_rgb)
        look_alike = cls._is_look_alike_rgb(mask_rgb)
        ship = cls._is_ship_rgb(mask_rgb)
        land = cls._is_land_rgb(mask_rgb)

        mask[sea, 0] = 1
        mask[oil, 1] = 1
        mask[look_alike, 2] = 1
        mask[ship, 3] = 1
        mask[land, 4] = 1

        unlabeled = ~(sea | oil | look_alike | ship | land)
        if np.any(unlabeled):
            raise ValueError(f"Unknown M4D palette pixels: {int(unlabeled.sum())}")

        return mask

    def _iter_patch_positions(self, height: int, width: int, show_progress: bool, scene_id: str):
        """Yield padded-space patch coordinates, optionally with a progress bar."""
        patch_size = self.config.patch_size
        stride = self.config.stride
        y_positions = list(range(0, height - patch_size + 1, stride))
        x_positions = list(range(0, width - patch_size + 1, stride))

        iterator = y_positions
        if show_progress and tqdm is not None:
            iterator = tqdm(
                y_positions,
                desc=f"Extract {scene_id}",
                unit="row",
                leave=False,
            )

        for y_padded in iterator:
            for x_padded in x_positions:
                yield y_padded, x_padded

    def _compute_symmetric_padding(self, h: int, w: int) -> Dict[str, int]:
        """Compute symmetric padding to guarantee exact sliding-window coverage.

        Algorithm: pad so that (size - patch_size) % stride == 0 after padding.
        This ensures patches placed at stride intervals cover entire image exactly.

        Args:
            h: Original image height
            w: Original image width

        Returns:
            Dictionary with pad_top, pad_bottom, pad_left, pad_right
        """
        patch_size = self.config.patch_size
        stride = self.config.stride

        # How much extra space needed for exact coverage
        pad_h = patch_size - h if h < patch_size else (stride - (h - patch_size) % stride) % stride
        pad_w = patch_size - w if w < patch_size else (stride - (w - patch_size) % stride) % stride

        # Distribute symmetrically
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        return {
            "top": pad_top,
            "bottom": pad_bottom,
            "left": pad_left,
            "right": pad_right,
        }

    def extract_patches_from_scene(
        self,
        scene_path: Path,
        mask_path: Path,
        scene_id: str,
        split: str = "train",
        image_pad_mode: str = "reflect",
        mask_pad_mode: str = "constant",
        show_progress: bool = True,
    ) -> List[PatchMetadata]:
        """Extract patches from a single scene with symmetric padding.

        Args:
            scene_path: Path to raw SAR image (HxW, uint8).
            mask_path: Path to segmentation mask. Supported formats:
                      - HxWx5 uint8 one-hot mask
                      - HxWx3 RGB M4D label PNG using the reference palette
            scene_id: Unique identifier for this scene (e.g., "img_0001").
            split: "train" or "test".
            image_pad_mode: Padding mode for image ("reflect", "edge", "constant").
                           "reflect" recommended for SAR imagery.
            mask_pad_mode: Must be "constant"; zero padding is excluded from class counts.
            show_progress: If True and `tqdm` is installed, display extraction progress.

        Returns:
            List of PatchMetadata for all extracted patches.

        Raises:
            FileNotFoundError: If scene or mask file not found.
            ValueError: If scene and mask dimensions don't match.
        """
        scene_path = Path(scene_path)
        mask_path = Path(mask_path)

        if not scene_path.exists():
            raise FileNotFoundError(f"Scene file not found: {scene_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        # Load scene and mask
        if scene_path.suffix == ".npy":
            scene = np.load(scene_path)
        else:
            from PIL import Image

            scene = np.array(Image.open(scene_path).convert('L'))

        if mask_path.suffix == ".npy":
            mask = np.load(mask_path)
        else:
            from PIL import Image

            mask = np.array(Image.open(mask_path).convert('RGB'))

        # Validate dimensions
        if scene.ndim != 2 or scene.dtype != np.uint8:
            raise ValueError('Scene must be a two-dimensional uint8 array')
        if mask_pad_mode != 'constant':
            raise ValueError('Class counts require zero padding of masks')
        if scene.shape[0] != mask.shape[0] or scene.shape[1] != mask.shape[1]:
            raise ValueError(
                f"Scene shape {scene.shape} doesn't match mask shape {mask.shape[:2]}"
            )

        # Convert RGB label mask to one-hot if needed
        if mask.ndim == 3 and mask.shape[2] == 3:
            mask = self._remap_rgb_mask_to_one_hot(mask)

        # Ensure mask is 5-channel (one-hot per class)
        if mask.ndim == 2:
            raise ValueError(f"Mask must be 3D (HxWx5), got shape {mask.shape}")
        if mask.shape[2] != 5:
            raise ValueError(
                f"Mask must have 5 channels (one per class), got {mask.shape[2]}"
            )

        h_orig, w_orig = scene.shape
        if not np.all((mask == 0) | (mask == 1)) or not np.all(mask.sum(axis=2) == 1):
            raise ValueError('Original mask must have exactly one class per pixel')
        logger.info(
            f"Extracting patches from {scene_id}: scene {scene.shape}, mask {mask.shape}"
        )

        # Compute symmetric padding for exact coverage guarantee
        padding = self._compute_symmetric_padding(h_orig, w_orig)
        pad_top = padding["top"]
        pad_bottom = padding["bottom"]
        pad_left = padding["left"]
        pad_right = padding["right"]

        logger.info(
            f"Symmetric padding: top={pad_top}, bottom={pad_bottom}, "
            f"left={pad_left}, right={pad_right} "
            f"({h_orig}×{w_orig} → {h_orig + pad_top + pad_bottom}×{w_orig + pad_left + pad_right})"
        )

        # Apply padding to image and mask with different modes
        scene_padded = np.pad(
            scene,
            pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
            mode=image_pad_mode
        )
        mask_pad_kwargs = {}
        if mask_pad_mode in self._MASK_PAD_MODES_WITH_CONSTANT_VALUES:
            mask_pad_kwargs["constant_values"] = 0
        mask_padded = np.pad(
            mask,
            pad_width=((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode=mask_pad_mode,
            **mask_pad_kwargs,
        )

        h_padded, w_padded = scene_padded.shape
        patch_size = self.config.patch_size
        stride = self.config.stride
        patches_extracted = 0

        # Extract patches from padded space
        for y_padded, x_padded in self._iter_patch_positions(
            h_padded, w_padded, show_progress=show_progress, scene_id=scene_id
        ):

            # Extract patch from padded scene
            patch = scene_padded[
                y_padded : y_padded + patch_size, x_padded : x_padded + patch_size
            ]
            mask_patch = mask_padded[
                y_padded : y_padded + patch_size, x_padded : x_padded + patch_size
            ]

            # Compute original space coordinates
            y_orig = y_padded - pad_top
            x_orig = x_padded - pad_left

            # Check if patch overlaps with padded region
            is_border = (
                y_padded < pad_top or
                x_padded < pad_left or
                y_padded + patch_size > h_padded - pad_bottom or
                x_padded + patch_size > w_padded - pad_right
            )

            # Create patch ID
            patch_id = f"{scene_id}_y{y_padded}_x{x_padded}"

            # Compute class pixel counts from mask
            class_counts = [
                int(np.sum(mask_patch[:, :, i])) for i in range(5)
            ]  # [sea, oil, look_alike, ship, land]
            class_labels = [1 if count > 0 else 0 for count in class_counts]

            # Save patch NPY file
            patch_output_path = self.patches_dir / f"{patch_id}.npy"
            if patch_output_path.exists():
                raise FileExistsError(patch_output_path)
            np.save(patch_output_path, patch.astype(np.uint8))

            # Create metadata with complete coordinate tracking
            metadata = PatchMetadata(
                patch_id=patch_id,
                scene_id=scene_id,
                split=split,
                y_padded=y_padded,
                x_padded=x_padded,
                y_orig=y_orig,
                x_orig=x_orig,
                pad_top=pad_top,
                pad_bottom=pad_bottom,
                pad_left=pad_left,
                pad_right=pad_right,
                patch_size=patch_size,
                stride=stride,
                is_border=is_border,
                pixels_sea=class_counts[0],
                pixels_oil=class_counts[1],
                pixels_look_alike=class_counts[2],
                pixels_ship=class_counts[3],
                pixels_land=class_counts[4],
                label_sea=class_labels[0],
                label_oil=class_labels[1],
                label_look_alike=class_labels[2],
                label_ship=class_labels[3],
                label_land=class_labels[4],
            )

            self.metadata_list.append(metadata)
            patches_extracted += 1

        logger.info(f"Extracted {patches_extracted} patches from {scene_id}")
        return self.metadata_list[-(patches_extracted):]

    def build_index(self) -> Tuple[pd.DataFrame, dict]:
        """Build parquet index from extracted patch metadata.

        Returns:
            Tuple of (DataFrame, metadata_dict).
                DataFrame: Parquet-compatible dataframe with patch information.
                metadata_dict: Config and statistics dictionary.

        Raises:
            ValueError: If no patches have been extracted.
        """
        if not self.metadata_list:
            raise ValueError("No patches have been extracted yet")

        # Convert to DataFrame
        df = pd.DataFrame([m._asdict() for m in self.metadata_list])
        df['valid_pixels'] = df[[f'pixels_{c}' for c in self.CLASS_NAMES]].sum(axis=1)
        df['padding_pixels'] = df['patch_size'] ** 2 - df['valid_pixels']
        df['pad_ratio'] = df['padding_pixels'] / df['patch_size'] ** 2
        df['patch_path'] = 'patches_npy/' + df['patch_id'] + '.npy'

        # Save parquet
        parquet_path = self.output_dir / f"index_{self.config.name}.parquet"
        df.to_parquet(parquet_path, index=False)
        logger.info(f"Saved parquet index: {parquet_path}")

        # Save CSV for reference
        csv_path = self.output_dir / f"index_{self.config.name}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved CSV index: {csv_path}")

        # Compute statistics
        metadata_dict = {
            "config": self.config.to_dict(),
            "dataset_stats": {
                "total_patches": len(df),
                "train_patches": len(df[df["split"] == "train"]),
                "test_patches": len(df[df["split"] == "test"]),
                "border_patches": int(df["is_border"].sum()),
            },
            "class_distribution": {
                "oil_patches": int(df["label_oil"].sum()),
                "lookalike_patches": int(df["label_look_alike"].sum()),
                "ship_patches": int(df["label_ship"].sum()),
                "land_patches": int(df["label_land"].sum()),
            },
        }

        # Save metadata JSON
        metadata_path = self.output_dir / f"metadata_{self.config.name}.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata_dict, f, indent=2)
        logger.info(f"Saved metadata: {metadata_path}")

        return df, metadata_dict


class SceneReconstructor:
    """Reconstruct scenes from extracted patches."""

    def __init__(self, config: PatchConfig):
        """Initialize reconstructor.

        Args:
            config: PatchConfig specifying window type and patch size.
        """
        self.config = config
        self._window_cache = {}

    def _get_window(self, size: int) -> np.ndarray:
        """Get 2D window function for patch-level blending.

        Args:
            size: Window size (should match patch_size).

        Returns:
            2D window array of shape (size, size).
        """
        if size not in self._window_cache:
            if self.config.window_type == "hann":
                # Create 2D Hann window via outer product
                window_1d = signal.windows.hann(size, sym=False)
                window_2d = np.outer(window_1d, window_1d)
            elif self.config.window_type == "uniform":
                window_2d = np.ones((size, size))
            else:
                raise ValueError(f"Unknown window type: {self.config.window_type}")

            # Avoid zero-weight seams at scene boundaries when a padded border is absent.
            # A tiny floor preserves stable normalization without affecting values materially.
            window_2d = np.clip(window_2d.astype(np.float32), 1e-6, None)
            self._window_cache[size] = window_2d

        return self._window_cache[size]

    @staticmethod
    def _normalize_patch_metadata_entry(metadata_entry):
        """Support PatchMetadata, namedtuples, or metadata-like dicts."""
        if isinstance(metadata_entry, dict):
            return metadata_entry
        if hasattr(metadata_entry, "_asdict"):
            return metadata_entry._asdict()
        raise TypeError(
            "patches_coords values must be dict-like metadata or PatchMetadata instances"
        )

    def reconstruct_scene(
        self,
        patches: dict,
        original_height: int,
        original_width: int,
        patches_coords: Optional[dict] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Reconstruct a scene from patches.

        Args:
            patches: Dictionary mapping (y, x) tuple to patch array (H, W).
            original_height: Height of original scene.
            original_width: Width of original scene.
            patches_coords: Optional dict mapping (y, x) to PatchMetadata for validation.

        Returns:
            Tuple of (reconstructed_scene, weight_map).
                reconstructed_scene: Reconstructed image (original_height, original_width).
                weight_map: Sum of window weights at each pixel (for analysis).

        Raises:
            ValueError: If patches don't fit in original scene.
        """
        patch_size = self.config.patch_size
        window = self._get_window(patch_size)

        pad_top = pad_left = pad_bottom = pad_right = 0
        use_padded_coords = False
        if patches_coords:
            first_meta = self._normalize_patch_metadata_entry(next(iter(patches_coords.values())))
            pad_top = int(first_meta.get("pad_top", 0))
            pad_bottom = int(first_meta.get("pad_bottom", 0))
            pad_left = int(first_meta.get("pad_left", 0))
            pad_right = int(first_meta.get("pad_right", 0))
            use_padded_coords = "y_padded" in first_meta and "x_padded" in first_meta

        canvas_height = original_height + pad_top + pad_bottom
        canvas_width = original_width + pad_left + pad_right

        # Initialize output arrays in padded space when metadata is available
        scene = np.zeros((canvas_height, canvas_width), dtype=np.float32)
        weights = np.zeros((canvas_height, canvas_width), dtype=np.float32)

        # Place patches
        for (y, x), patch in patches.items():
            if patch.shape != (patch_size, patch_size):
                raise ValueError(
                    f"Patch at ({y}, {x}) has shape {patch.shape}, "
                    f"expected {(patch_size, patch_size)}"
                )

            y_canvas = y
            x_canvas = x
            if patches_coords and (y, x) in patches_coords:
                meta = self._normalize_patch_metadata_entry(patches_coords[(y, x)])
                if use_padded_coords:
                    y_canvas = int(meta["y_padded"])
                    x_canvas = int(meta["x_padded"])
                else:
                    y_canvas = int(meta.get("y_orig", y))
                    x_canvas = int(meta.get("x_orig", x))

            if y_canvas < 0 or x_canvas < 0:
                raise ValueError(
                    f"Patch at ({y_canvas}, {x_canvas}) has negative canvas coordinates"
                )
            if y_canvas + patch_size > canvas_height or x_canvas + patch_size > canvas_width:
                raise ValueError(
                    f"Patch at ({y_canvas}, {x_canvas}) extends beyond scene bounds "
                    f"({canvas_height}, {canvas_width})"
                )

            # Add weighted patch
            patch_float = patch.astype(np.float32)
            scene[
                y_canvas : y_canvas + patch_size, x_canvas : x_canvas + patch_size
            ] += patch_float * window
            weights[
                y_canvas : y_canvas + patch_size, x_canvas : x_canvas + patch_size
            ] += window

        # Normalize by weights (avoid division by zero)
        valid = weights > 0
        scene[valid] /= weights[valid]
        scene[~valid] = 0  # Background pixels (no patch coverage)

        if pad_top or pad_bottom or pad_left or pad_right:
            scene = scene[pad_top : pad_top + original_height, pad_left : pad_left + original_width]
            weights = weights[pad_top : pad_top + original_height, pad_left : pad_left + original_width]

        return scene, weights

    def load_patches_from_disk(
        self, patch_paths: List[str], coordinates: List[Tuple[int, int]]
    ) -> dict:
        """Load patches from disk.

        Args:
            patch_paths: List of paths to patch NPY files.
            coordinates: List of (y, x) coordinates corresponding to each patch.

        Returns:
            Dictionary mapping (y, x) to patch array.

        Raises:
            FileNotFoundError: If any patch file not found.
            ValueError: If number of paths and coordinates don't match.
        """
        if len(patch_paths) != len(coordinates):
            raise ValueError(
                f"Number of paths ({len(patch_paths)}) doesn't match "
                f"number of coordinates ({len(coordinates)})"
            )

        patches = {}
        for path, (y, x) in zip(patch_paths, coordinates):
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"Patch file not found: {path}")

            patch = np.load(path)
            patches[(y, x)] = patch

        return patches
