# Patching System: M4D-SAR-OilSpill

Reversible patch extraction and scene reconstruction for multi-size patch-based deep learning.

## Overview

This patching system enables:

1. **Multi-size patch extraction** (256, 384, 512 pixels) with configurable overlap
2. **Reversible reconstruction** — reconstruct original scenes from patches with proper weighted blending
3. **Transparent indexing** — complete traceability of patch↔scene relationships
4. **Validation & metrics** — quantify reconstruction quality and verify reversibility

## Quick Start

### 1. Extract Patches from Raw M4D Dataset

```python
from pathlib import Path
from patching import PatchExtractor, PatchConfig

# Define configuration
config = PatchConfig(
    patch_size=512,
    stride=256,
    overlap=0.5,
    window_type="hann",
    name="ps512_overlap50"
)

# Create extractor
extractor = PatchExtractor(config, output_dir=Path("./patches/ps512"))

# Extract patches from one scene
scene_path = Path("dataset/train/images/img_0001.png")
mask_path = Path("dataset/train/labels/img_0001.png")

metadata_list = extractor.extract_patches_from_scene(
    scene_path=scene_path,
    mask_path=mask_path,
    scene_id="img_0001",
    split="train"
)

# Build index (parquet + metadata JSON)
df, stats = extractor.build_index()
print(f"Extracted {len(df)} patches")
print(f"Oil patches: {stats['class_distribution']['oil_patches']}")
```

### 2. Reconstruct a Scene from Patches

```python
from patching import SceneReconstructor
import numpy as np

# Create reconstructor
reconstructor = SceneReconstructor(config)

# Load patches from disk (or compute in-memory)
patches = {
    (0, 0): np.load("patches/ps512/patches_npy/img_0001_y0_x0.npy"),
    (0, 256): np.load("patches/ps512/patches_npy/img_0001_y0_x256.npy"),
    # ... more patches
}

# Reconstruct original scene
reconstructed, weight_map = reconstructor.reconstruct_scene(
    patches=patches,
    original_height=2048,
    original_width=2048
)

print(f"Reconstructed scene shape: {reconstructed.shape}")
```

### 3. Validate Reversibility

```python
from patching import ReversibilityValidator

# Load original scene
original = np.load("original_scene.npy")

# Validate
validator = ReversibilityValidator(config)
is_reversible, metrics, reconstructed = validator.test_reversibility(
    original_scene=original,
    patches=patches,
    original_height=2048,
    original_width=2048,
    tolerance=2.0  # RMSE tolerance in pixel units
)

print(f"Reversible: {is_reversible}")
print(f"RMSE: {metrics.rmse:.3f}")
print(f"PSNR: {metrics.psnr:.1f} dB")
print(f"Pixels within ±1: {metrics.percent_within_1:.1f}%")
```

## Configuration

### Patch Configurations

| Config | Size | Stride | Overlap | Use Case |
|--------|------|--------|---------|----------|
| `ps256_overlap50` | 256×256 | 128 | 50% | Baseline, many training samples |
| `ps384_overlap50` | 384×384 | 192 | 50% | **NEW**: Medium context balance |
| `ps512_overlap50` | 512×512 | 256 | 50% | **NEW**: Larger context for discrimination |
| `ps512_overlap75` | 512×512 | 128 | 75% | High-quality inference only |

### Load from YAML

```python
from patching.config import load_patching_configs

configs = load_patching_configs("configs/patching.yaml")
config = configs["ps512_overlap50"]
```

### Create Custom Config

```python
from patching import PatchConfig

config = PatchConfig(
    patch_size=512,
    stride=256,
    overlap=0.5,
    window_type="hann",
    name="custom_config"
)
```

## API Reference

### `PatchConfig`

Configuration dataclass for patch parameters.

**Attributes:**
- `patch_size: int` — Patch width/height (256, 384, 512)
- `stride: int` — Stride between patch centers
- `overlap: float` — Overlap ratio (0.0 to 1.0)
- `window_type: str` — "hann" or "uniform"
- `name: str` — Configuration identifier

**Methods:**
- `to_dict()` — Serialize to dictionary
- `from_dict(d)` — Create from dictionary

---

### `PatchExtractor`

Extract patches from raw M4D scenes.

**Constructor:**
```python
PatchExtractor(config: PatchConfig, output_dir: Path)
```

**Methods:**

#### `extract_patches_from_scene(scene_path, mask_path, scene_id, split="train")`
Extract all patches from a single scene.

**Args:**
- `scene_path: Path` — Path to SAR image (HxW)
- `mask_path: Path` — Path to segmentation mask (HxWx5)
- `scene_id: str` — Scene identifier (e.g., "img_0001")
- `split: str` — "train" or "test"

**Returns:**
- `List[PatchMetadata]` — Metadata for all extracted patches

**Raises:**
- `FileNotFoundError` — If paths don't exist
- `ValueError` — If dimensions mismatch

#### `build_index()`
Build parquet index from all extracted patches.

**Returns:**
- `Tuple[pd.DataFrame, dict]` — Index dataframe and metadata dictionary

**Output files:**
- `index_{config_name}.parquet` — Patch index
- `index_{config_name}.csv` — Index in CSV format (for reference)
- `metadata_{config_name}.json` — Config and statistics

---

### `SceneReconstructor`

Reconstruct scenes from patches.

**Constructor:**
```python
SceneReconstructor(config: PatchConfig)
```

**Methods:**

#### `reconstruct_scene(patches, original_height, original_width, patches_coords=None)`
Reconstruct a scene from patch dictionary.

**Args:**
- `patches: dict` — Mapping from (y, x) tuple to patch array (H, W)
- `original_height: int` — Original scene height
- `original_width: int` — Original scene width
- `patches_coords: dict` — Optional metadata dict for validation

**Returns:**
- `Tuple[np.ndarray, np.ndarray]` — (reconstructed_scene, weight_map)
  - `reconstructed_scene`: Blended scene (float32, [0-255])
  - `weight_map`: Sum of window weights at each pixel

**Raises:**
- `ValueError` — If patches exceed scene bounds or have wrong shape

#### `load_patches_from_disk(patch_paths, coordinates)`
Load patches from NPY files on disk.

**Args:**
- `patch_paths: List[str]` — Paths to patch NPY files
- `coordinates: List[Tuple[int, int]]` — Corresponding (y, x) coordinates

**Returns:**
- `dict` — Mapping from (y, x) to loaded patch

---

### `ReversibilityValidator`

Validate reconstruction quality.

**Constructor:**
```python
ReversibilityValidator(config: PatchConfig)
```

**Methods:**

#### `test_reversibility(original_scene, patches, original_height, original_width, tolerance=2.0)`
Test if patches reversibly reconstruct the original scene.

**Args:**
- `original_scene: np.ndarray` — Original image (H, W)
- `patches: dict` — Patches mapping
- `original_height: int` — Original height
- `original_width: int` — Original width
- `tolerance: float` — RMSE tolerance threshold

**Returns:**
- `Tuple[bool, ReversibilityMetrics, np.ndarray]`
  - `bool`: True if RMSE ≤ tolerance
  - `ReversibilityMetrics`: Quality metrics
  - `np.ndarray`: Reconstructed scene

#### `compute_reconstruction_metrics(original, reconstructed)`
Compute per-pixel reconstruction quality metrics.

**Returns:** `ReversibilityMetrics` with:
- `rmse: float` — Root Mean Squared Error
- `mae: float` — Mean Absolute Error
- `psnr: float` — Peak Signal-to-Noise Ratio (dB)
- `percent_within_1: float` — % of pixels within ±1
- `percent_within_5: float` — % of pixels within ±5
- `max_error: float` — Maximum absolute error

#### `compare_edge_artifacts(original, reconstructed, margin=50)`
Analyze reconstruction quality near scene edges.

**Returns:**
- `dict` — Per-edge RMSE, MAE, max_error

#### `save_validation_report(output_dir, metrics, edge_metrics)`
Save validation results to text file.

---

## Data Structure

### Patch NPY Files

**Location:** `{output_dir}/patches_npy/`

**Naming:** `{scene_id}_y{y}_x{x}.npy`
- Example: `img_0001_y128_x256.npy`

**Format:**
- Shape: (patch_size, patch_size)
- dtype: uint8
- Values: [0, 255] (raw SAR intensity)

### Index Files

#### Parquet Index

**Location:** `{output_dir}/index_{config_name}.parquet`

**Columns:**
```
patch_id              string   e.g., "img_0001_y0_x0"
scene_id              string   e.g., "img_0001"
split                 string   "train" or "test"
y, x                  int      Top-left coordinates
patch_size, stride    int      Configuration
is_border             bool     True if touches scene edge
pad_ratio             float    Padding ratio (0.0 if none)
pixels_sea            int      Sea pixel count
pixels_oil            int      Oil pixel count
pixels_look_alike     int      Look-alike pixel count
pixels_ship           int      Ship pixel count
pixels_land           int      Land pixel count
label_sea             int      1 if class present, 0 otherwise
label_oil             int      (and so on...)
label_look_alike      int
label_ship            int
label_land            int
```

#### Metadata JSON

**Location:** `{output_dir}/metadata_{config_name}.json`

**Structure:**
```json
{
  "config": {
    "patch_size": 512,
    "stride": 256,
    "overlap": 0.5,
    "window_type": "hann",
    "name": "ps512_overlap50"
  },
  "dataset_stats": {
    "total_patches": 45000,
    "train_patches": 40500,
    "test_patches": 4500,
    "border_patches": 2500
  },
  "class_distribution": {
    "oil_patches": 2500,
    "lookalike_patches": 1200,
    "ship_patches": 800,
    "land_patches": 600
  }
}
```

## Window Functions

### Hann Window

Smooth tapering at patch edges to reduce artifacts.

```
window(n) = 0.5 * (1 - cos(2π * i / (n-1))) for i in 0..n-1
```

**Advantages:**
- Smooth transitions → less blocking artifacts
- Better for scene reconstruction
- Recommended default

### Uniform Window

Simple mean blending (no windowing).

**Advantages:**
- Faster (no multiplication)
- Better for boundary analysis
- Useful for debugging

## Reversibility Guarantees

### Expected Performance

For the standard Hann window with 50% overlap:

| Metric | Target | Expected |
|--------|--------|----------|
| RMSE | < 2.0 | 0.5–1.5 |
| PSNR | > 40 dB | 45–50 dB |
| Pixels within ±1 | > 99% | 99.5%+ |

### Why Not Perfect?

Floating-point blending introduces small numerical errors:
- Window normalization: weights summed over overlapping regions
- Rounding: uint8 → float → uint8 conversions

### Tolerance Guidelines

- **Training/validation:** tolerance = 2.0 (safe margin)
- **Inference:** tolerance = 5.0 (allows for FP16 precision loss)
- **Validation:** tolerance = 1.0 (strict, for debugging)

## Performance Considerations

### Memory Usage

Assuming 2048×2048 scenes (typical M4D):

| Patch Size | # Patches/Scene | NPY Disk Size | Peak RAM |
|-----------|-----------------|---------------|----------|
| 256×256 | ~256 | ~100 MB | ~150 MB |
| 384×384 | ~81 | ~32 MB | ~50 MB |
| 512×512 | ~36 | ~14 MB | ~25 MB |

### Computation Time

On modern CPU (Intel i7):

| Operation | 256 | 384 | 512 |
|-----------|-----|-----|-----|
| Extraction | ~100 ms | ~50 ms | ~25 ms |
| Reconstruction | ~200 ms | ~100 ms | ~50 ms |

## Troubleshooting

### RMSE Too High (> 5.0)

**Possible causes:**
- Wrong coordinate mapping — verify patch (y, x) values
- Missing patches — check that all expected patches exist on disk
- Window type mismatch — ensure reconstructor uses same window as specification

**Solution:**
```python
# Verify patch coordinates
for (y, x), patch in patches.items():
    assert patch.shape == (config.patch_size, config.patch_size)
    assert y + config.patch_size <= height
    assert x + config.patch_size <= width
```

### Edge Artifacts / Boundary Errors

**Possible cause:** Insufficient overlap or wrong window function.

**Solution:** Try `ps512_overlap75` with dense inference stride, or switch to `uniform` window for debugging.

### Out of Memory

**Solution:** Process scenes in batches instead of loading all patches.

```python
# Process in chunks
for chunk in partition_patches(all_patches, batch_size=64):
    partial_scene, partial_weights = reconstructor.reconstruct_scene(
        chunk, height, width
    )
    # Accumulate
```

## Citation & References

### Hann Window Properties

Taper function from `scipy.signal.hann`:
```python
from scipy import signal
w = signal.hann(N, sym=False)
```

For 2D: Outer product of two 1D windows.

### Patch-Based Segmentation

See reference papers in dissertation Chapter 2 for patch-based CNN architectures and their window design choices.

## License

Part of Ph.D. dissertation project. See parent project for license details.
