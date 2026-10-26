# segmentation: Stage 3 (training, out-of-fold inference, scoring, label review)

Run everything from the repository root as `python -m segmentation.<module>`. Every command takes `--project-root DIR` (or the environment
variable `OILSEG_PROJECT_ROOT`); inside that folder the code expects

```
<project root>/oil-spill-detection-dataset/   unpacked M4D-SAR-OilSpill (train/images, train/labels, train/labels_1D, test/...)
<project root>/stage2_results/                patches and folds from patching/ (training_patches/ps256_overlap50, grouped_folds_v1)
<project root>/results/                       everything this package writes (checkpoints, oof_*, error_analysis_*, thesis_figures, ...)
```

## Modules

| Module | What it does |
|---|---|
| `models.py` | The five models as `SPECS`: `segformer_b0_shift` (control, also the baseline recipe without `--shift-aug`), `segformer_b2`, `segformer_b2_hrda_c` (the model of the thesis), `segformer_b2_hrda_d`, `deeplabv3plus_resnet50` |
| `train.py` | One fold of one model: `python -m segmentation.train --model segformer_b2_hrda_c --fold 1 --shift-aug --epochs 50 --batch-size 64 --output-root results/stage3_segformer_b2_hrda_c_shift_256` (GPU required; `--dry-run` only audits the data) |
| `common.py`, `data.py`, `context.py` | Training loop and checkpoint rules, the 256 px patch dataset, the 512 px scene context of the two-scale models |
| `inference/oof.py` | Five-fold out-of-fold reconstruction of whole scenes (`--arch`, `--checkpoints-root`, `--output-root`) |
| `inference/ensemble.py` | The same for an ensemble of checkpoint sets with optional flip test-time augmentation |
| `inference/test_split.py` | One-time evaluation of the five fold models on the official test scenes (confusion matrices per scene) |
| `inference/scene.py` | Windowed prediction on any grey scene (used by `sentinel1_scenes/`) |
| `evaluation/bootstrap.py` | Pooled metrics and paired scene bootstrap: `revised`, `hrda_c`, `hrda_d`, `ensemble_deeplab` |
| `evaluation/test_split.py` | Scoring of the test evaluation (ensemble of the five folds, bootstrap over scenes) |
| `evaluation/calibration.py` | Calibration of the Oil probability of HRDA-C |
| `cascade/` | The cascade experiment: `b0_baseline.py` and `b0_native512.py` (SegFormer-B0 trainers on 256 px and on native 512 px windows), `infer_b0_oof.py` (ungated OOF, `--patch-size 256|512`), `infer_cascade.py` (segmenter only on windows passed by the Stage-2 gate), `prepare_routing_512.py` (routing tables from the 512 px Stage-2 runs), `native512_width_bins.py` (does native 512 px recover thin slicks, E1(b)); ceilings of a gate (`oracle_gate`, `oracle_window_class`), real Stage-2 classifiers against them (`stage2_window_accuracy`, `stage2_threshold_curve`), whole-object Oil / Look-alike vote (`object_vote`) |
| `figures/` | Figures of Chapter 5 from the saved tables and prediction maps |
| `label_review/` | Label review: `review_napari.py` (editor, writes a revised copy only), `review_logic.py` (validated write path), `tools.py` (ship candidates, suspicion scores, lattice test, ...) |
| `data_prep/patch_masks.py` | Precomputed 256 px mask bundles for training (`--patch-masks-root`) |

## Typical order

```bash
# 1. train the five folds (repeat with --fold 2..5)
python -m segmentation.train --model segformer_b2_hrda_c --fold 1 --shift-aug --epochs 50 --batch-size 64 \
       --output-root results/stage3_segformer_b2_hrda_c_shift_256
# 2. out-of-fold scenes
python -m segmentation.inference.oof --arch segformer_b2_hrda_c --checkpoints-root results/stage3_segformer_b2_hrda_c_shift_256 \
       --output-root results/oof_stage3_b2_hrda_c_shift_256
# 3. paired bootstrap against B2 (needs the B2 and B2 seed 43 OOF folders; the names are listed in evaluation/bootstrap.py)
python -m segmentation.evaluation.bootstrap hrda_c
# 4. official test scenes, once, no tuning
python -m segmentation.inference.test_split --arch segformer_b2_hrda_c --checkpoints-root results/stage3_segformer_b2_hrda_c_shift_256
python -m segmentation.evaluation.test_split
```

`train.py` needs `--baseline-oof-summary` (default `results/oof_gpu_full_v1/summary.json`, the OOF summary of the B0 baseline) only for the
balanced checkpoint rule `best_balsel.pt`; the thesis uses `best.pt` (highest Oil IoU). The B0 baseline is `--model segformer_b0_shift`
without `--shift-aug`.

## Checks

`python -m unittest discover -s tests -t .` runs the unit tests (checkpoint store and selection, reflect crops and two-scale step, review logic,
window weights, ensemble members, scene metrics and patch stitching). The bootstrap, ceiling and test-split scripts were re-run on the original outputs and give identical tables;
`train.py --dry-run` passes on the real patch index; `sentinel1_scenes.scene_pipeline run` reproduces a recorded crop pixel for pixel.

`transformers` must be a 4.x release: 5.x renames the SegFormer layers and the checkpoints no longer load.

## Further material

Earlier exploratory arms (coordinate attention, clDice, H/2 decode fusion, sea-balanced filtering, native 512 px, HRNetV2+OCR, Mask2Former,
FA-MobileUNet, contour-tightened labels) and the external DARTIS check are kept in the working archive of the dissertation.
