# dataset_analysis: statistics and figures of the dataset chapters

Run from the repository root with `--project-root DIR` (or `OILSEG_PROJECT_ROOT`); the unpacked M4D dataset is expected in
`<root>/oil-spill-detection-dataset/`, outputs go to `<root>/dataset_analysis_output/`. M4D itself is not part of this repository.

| Script | Content |
|---|---|
| `analyze_dataset.py` | Class statistics of the 1,112 scenes (pixel shares, scenes per class, metadata) |
| `generate_ch3_figures.py` | Chapter 3 figure set: representative scenes, pixel class distribution, per-scene areas, class co-occurrence, spatial heatmaps, intensity comparison, object sizes |
| `analyze_class_pixels_per_image.py` | Class pixels per scene (table behind the patching figure) |
| `generate_ch4_patching_figures.py` | How class statistics change from scenes to 256 px windows with stride 128 |
| `generate_ch4_scheme_figures.py` | Cascade scheme, window grid on a real scene, two-scale model |
| `generate_ch5_label_review_figures.py` | Label review: before/after examples, possible wind-farm lattices |
| `stage2_geometry_audit.py`, `stage2_component_window_coverage.py`, `plot_stage2_geometry_audit.py` | Patch geometry and how much of each Oil / Look-alike component a patch grid covers; `test_stage2_geometry_audit.py` checks the audit |
| `utils.py` | Mask decoding (RGB to class index), class names and colours, dataset validation |

Reference: Krestenitis et al. (2019), M4D-SAR-OilSpill, <https://m4d.iti.gr/oil-spill-detection-dataset/>.
