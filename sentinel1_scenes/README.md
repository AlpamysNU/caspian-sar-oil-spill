# sentinel1_scenes: raw Sentinel-1 scenes through the model

Working folder: `<project root>/sentinel1/` (scene folders, `raw/`, `out/`, optional `osm/platforms.json`); checkpoints from
`<project root>/results/stage3_segformer_b2_hrda_c_shift_256`. Run from the repository root.

| Module | Commands |
|---|---|
| `cdse.py` | `probe`, `annotation`, `download`: Copernicus Data Space access. S3 keys are read from `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` in the environment of the run; the OData password is typed in. Nothing is stored. |
| `scene_pipeline.py` | `run` (calibrated preprocessing and HRDA-C on a crop; recipes A and B of Section 5.6), `compare-models`, `sensitivity`, `compare-sensitivity`, `grey`, `annotation-info` |
| `case_studies.py` | `dark-area`, `patch-shrink`, `stages-figure`, `raw-cases-figure` (Novorossiysk, Kashagan, Peshnoi) |
| `m4d_reference.py` | `texture`, `scale-cv`, `scale-skew`, `speckle`, `grey-by-class`, `component-sizes`: the M4D measurements behind the preprocessing recipe |

The recipe (2x2 pooling, about 22 grey levels per dB, sea level mapped to a fixed grey) is estimated from M4D statistics, not taken from
documentation. The outputs are evidence levels for dark patches, not oil verdicts.
