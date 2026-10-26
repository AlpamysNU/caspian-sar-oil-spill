# Oil spill detection in Sentinel-1 SAR imagery: code

Code accompanying the PhD dissertation *Context-Aware and Multi-Modal Deep Learning for Oil Spill Detection and Environmental Monitoring Using Sentinel-1 SAR Imagery* and the article Pentayev et al. (2025), *International Journal of Water Resources Development*.

**The M4D-SAR-OilSpill dataset (Krestenitis et al., 2019) is not included and must not be redistributed.** Download it from <https://m4d.iti.gr/oil-spill-detection-dataset/> under the authors' terms. Trained weights, derived patches, predictions and the revised labels of the label review are not included either.

## Layout

| Folder | Content |
|---|---|
| `dataset_analysis/` | Class statistics and the figures of Chapters 3-5 that come from the dataset (8 scripts) |
| `patching/` | Stage 1: 256/384/512 px overlapping patches, scene-grouped five-fold split, reconstruction checks |
| `classification/` | Stage 2: patch classifiers (EfficientNet-B2, ConvNeXt-Tiny, ResNet-50), losses, gate coverage analysis |
| `segmentation/` | Stage 3 as a package: models (SegFormer-B0/B2, HRDA-C/D, DeepLabV3+ ResNet-50), `train.py`, out-of-fold and test inference, scene-bootstrap scoring, ceilings of a gate, figures, label review tools |
| `sentinel1_scenes/` | Four modules: Copernicus Data Space access, preprocessing and running the model on raw scenes, case figures, M4D reference measurements |
| `tests/` | Unit tests |

## Paths

Everything that is not an explicit argument is resolved from one project root, given as `--project-root DIR` or `OILSEG_PROJECT_ROOT`
(default: the current directory): `oil-spill-detection-dataset/` (the unpacked dataset), `stage2_results/` (patches and folds), `results/`
(Stage-3 outputs), `sentinel1/` (raw scenes), `dataset_analysis_output/`. Run from the repository root; see `segmentation/README.md`.

## Notes

- Copernicus Data Space keys are read from `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` in the environment of the run; nothing is stored.
- Cluster job scripts are not part of this repository; the training scripts run from the command line.
- The code was written for research; scripts are provided as they were used for the dissertation figures and tables.

## Licence

MIT (see `LICENSE`). Third-party material and its terms: `THIRD_PARTY.md`. The dataset keeps its own terms.

## Install and test

`pip install -r requirements.txt`, then `python -m unittest discover -s tests -t .`. `transformers` must be a 4.x release.
