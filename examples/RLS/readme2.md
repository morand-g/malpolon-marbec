# Introduction

This document is a reproduction guide for "Transductive learning of the seascape reveals conservation blind spots for rare and threatened reef fishes" available here: XXXXX.

The workflow is not self-contained in this repository. The data must be obtained from Zenodo at this link:
The raw survey data, GeoEnrich products, raster inputs, unlabelled prediction grid, trait tables,
protected-area layers, and trained checkpoints.

The commands below describe the intended order and identify the places that need local configuration.

## 1. Study design

The study uses 27,437 RLS/ATRC surveys from 103 geographic locations and 817
fish species after filtering species with fewer than 50 occurrences. Locations
are assigned to spatial train, validation, and test subsets in a 60/20/20
split; surveys from one location remain in the same subset.

The predictors are four raster modalities generated from 20 socio-environmental
variables:

| Modality | Channels | Nominal spatial extent and size |
| --- | ---: | --- |
| `env` | 15 | 100 km, 32 x 32 |
| `humhd` | 4 | 57 km, 128 x 128 |
| `bathy` | 1 | 57 km, 128 x 128 |
| `dhw` | 1 | 10 years of daily Degree Heating Week, resized to 128 x 128 |

The deep-learning comparison has three families:

1. XGBoost models using summary statistics extracted from each raster.
2. Inductive deep models trained directly on labelled survey data.
3. Transductive deep models initialized with masked-autoencoder pretraining on
   labelled surveys plus 49,390 unlabelled points (4,939 coastal locations at
   10 dates each).

The supervised tasks are presence/absence and abundance for 817 species, plus
three ecological indicators: species richness, threatened-species richness,
and total biomass.

## 2. Repository and working directory

Run the Python scripts from this directory so that the local `deep_utils.py`
module and Hydra configuration directory are found:

```bash
cd /path/to/malpolon-marbec/examples/RLS
```

The package itself is in `../../malpolon` relative to this directory.

## 3. Environment

Use a Linux environment with a CUDA-capable GPU for practical training. Python
3.9-3.11 is recommended even though the package metadata declares Python >=3.7.
Install the package and the RLS dependencies:

```bash
cd /path/to/malpolon-marbec
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ./malpolon
python -m pip install -r examples/RLS/requirements.txt
```

Install R and the packages used by the Boyce scripts:

```r
install.packages(c("readr", "dplyr", "stringr", "purrr", "progress", "modEvA"))
```

## 4. Configure the data root

The checked-in YAML files use:

```text
/marbec-data/RLS-Australia/malpolon/
```

Either mount the study data at that path or override `data.root` on every Hydra
command. The expected layout is:

```text
<data-root>/
  inputs/australia/
    database_common.csv
    database_common_trainval.csv
    database_transductif.csv
    database_common_metadata.csv
    site_metadata.csv
    biomass_lognorm.csv
    biomass_lognorm_trainval.csv
    biomass_lognorm_maxima.csv
    eco_indic_maxima.csv
    env/<survey_id>.npy
    humhd/<survey_id>.npy
    bathy/<survey_id>.npy
    dhw/<survey_id>.npy
  inputs/australiapreds/
    auspredgrid.csv
    auspredgrid_eco.csv
    env/<survey_id>.npy
    humhd/<survey_id>.npy
    bathy/<survey_id>.npy
    dhw/<survey_id>.npy
  outputs/
  logs/
```

`RLSDataModule` loads each `.npy` raster by survey ID, transposes arrays from
height x width x channels to channels x height x width, center-crops `humhd`
and `bathy` to 128 pixels, and resizes `dhw` to 128 pixels. The directory names
must therefore match the modality names in the YAML files.

## 5. Prepare the data with RLS0

Open `RLS0 - Data preparation.ipynb` in Jupyter or VS Code and run its cells
in order. Update the raw-data, GeoEnrich, and output paths before execution.
The notebook is the data-preparation source of truth and performs the following
operations:

1. Read RLS/ATRC survey metadata and fish biomass.
2. Pivot species observations by `survey_id` and remove incomplete records.
3. Assign location-level train, validation, and test subsets.
4. Create survey metadata and distance-to-training diagnostics.
5. Filter to the 817 common species.
6. Create transformed biomass.
7. Build the transductive table containing the labelled and unlabelled points.
8. Prepare GeoEnrich raster modalities.
9. Optionally download satellite data through STAC and Planetary Computer.

The notebook must produce the CSV files listed in Section 4. It also needs to
produce the four raster directories for both survey data and the unlabelled
prediction grid. Do not start model training until a sample survey ID can be
loaded from every modality.

The paper states that the masked-autoencoder data contain 49,390 unlabelled
points. Verify this count and the reported 60/20/20 location split from the
generated tables before proceeding.

## 6. Pretrain the transductive foundation model

`rls_aus_binned.py` is used for masked reconstruction when
`data.mask_inputs > 0`. The foundation configuration is
`config/rls_aus_fm.yaml`:

```bash
cd /path/to/malpolon-marbec/examples/RLS
python rls_aus_binned.py -cn rls_aus_fm \
  data.root=/path/to/RLS-Australia/malpolon \
  run.inference=false
```

The foundation model has four ResNet-18 branches and reconstructs randomly
masked spatial patches with mean squared error. The checked-in configuration
uses:

```text
mask_inputs: 0.5
mae_patch_size: 8
max_epochs: 5
```

Save `last.ckpt` and its Hydra output directory; later configs
must refer to this checkpoint.

## 7. Train and run deep models

All supervised runs should be repeated with seeds 1 through 5. The Hydra
output directory is controlled by `data.root/outputs/${run.run_name}` and, for
multiruns, `Seed-${run.seed}`.

### Presence/absence

Use `config/rls_aus_binned_pa.yaml` for inference over the unlabelled coastal
grid after a checkpoint has been trained:

```bash
python rls_aus_binned.py -cn rls_aus_binned_pa -m \
  data.root=/path/to/RLS-Australia/malpolon \
  run.seed=1,2,3,4,5 \
  run.checkpoint_path=/path/to/trained/checkpoint/last.ckpt
```

This config reads `auspredgrid.csv`, uses two output classes for each of 817
species, and writes prediction CSVs in the run directory. For test evaluation,
use a configuration with `run.testing=true` and a dataset containing the test
subset. For training from scratch, set `run.inference=false` and
`run.checkpoint_path=null`. For a transductive supervised model, initialize
from the foundation checkpoint and use the appropriate finetuning mode or
replace the model head as required by the script.

### Biomass regression

Train the 817-output biomass model from `biomass_lognorm.csv`:

```bash
python rls_aus_reg.py -cn rls_aus_reg \
  data.root=/path/to/RLS-Australia/malpolon \
  run.inference=false \
  run.checkpoint_path=null
```

The script applies `log(x + 1)` to database targets. If a checkpoint is used,
set `run.inference=true` for prediction and set `run.testing=true` when test
metrics and train-plus-validation predictions are required.

### Ecological indicators

Use `config/rls_aus_reg_eco.yaml` with a trained checkpoint to predict the
three indicators in `auspredgrid_eco.csv`:

```bash
python rls_aus_reg.py -cn rls_aus_reg_eco -m \
  data.root=/path/to/RLS-Australia/malpolon \
  run.seed=1,2,3,4,5 \
  run.checkpoint_path=/path/to/trained/checkpoint/last.ckpt
```

The committed config is primarily an inference configuration and points to a
dated checkpoint that is not included in the repository. Replace that path.

### Inductive baseline

Train the same architecture without foundation initialization by setting
`run.inference=false`, `run.checkpoint_path=null`, and leaving the branch
checkpoints unset. Repeat this for five seeds. There is no dedicated inductive
YAML or launch script; create a local copy of the relevant supervised config
or override these fields on the command line.

The provided `launch.sh` and `launchreg.sh` are cluster-specific examples.
They activate `/home/gmorand/venvs/deepsdm3/bin/activate`, request a GPU, and
submit only selected PA and ecological-regression five-seed sweeps. Update the
environment, SLURM dependency IDs, paths, and job names before use.

## 8. Train the XGBoost baselines with RLS2

Run `RLS2 - Baselines.ipynb` after the prepared tables and rasters exist. Set
the notebook's `root`, `inputs_path`, and output paths to the local data root.
The notebook:

1. Extracts mean, standard deviation, and center-value features from raster
   modalities.
2. Extracts mean, standard deviation, and maximum-over-time features for DHW.
3. Trains separate XGBoost models for species PA, biomass, and ecological
   indicators.
4. Sweeps the subset/subsampling value from 0.1 through 0.9.
5. Selects the best value independently per species using validation scores.
6. Writes predictions and validation summaries below the `xgb/` directory.

The notebook contains older Random Forest and binned-model cells marked as
unused; they are not part of the reported baseline unless explicitly enabled.

## 9. Generate evaluation metrics with RLS3

Run `RLS3 - Outputs.ipynb` with the generated prediction directories. Update
all dated checkpoint/output paths in the notebook. The evaluation should use
the held-out location-level test set and include:

- AUC-ROC, F1, TSS/maxTSS, and continuous Boyce index for PA.
- Pearson and Spearman correlation for biomass and ecological indicators.
- Non-zero Pearson correlation for biomass.
- Site-level F1 and embedding/PCA analyses where required by the figures.

For PA, the threshold is selected on validation predictions so that mean
predicted species richness matches observed validation richness. Apply the
selected threshold to test and prediction-grid probabilities; do not tune it
on the test set.

The deep scripts export files such as:

```text
predictions-probs.csv
predictions-probs-trainval.csv
predictions-biomass.csv
predictions-biomass-trainval.csv
presences--TH=<threshold>.csv
```

The helper functions in `results_utils.py` aggregate the five seeds and report
normal-approximation 95% confidence intervals as `mean +/- 1.96 * SEM`.

## 10. Calculate Boyce scores

The R scripts expect the prediction CSVs and dated output directories to
already exist. Edit their hard-coded `root`, `inputs_path`, `output_path`, and
`pred_dir` values first.

For XGBoost predictions:

```bash
Rscript Boyce.R
```

For deep PA predictions:

```bash
Rscript Boyce_deep.R
```

Both scripts read the 817-species ground truth and write `boyce.csv` per seed.

## 11. Compare models with RLS4

Run `RLS4 - Model comparison.ipynb` after RLS2, RLS3, and the Boyce scripts
have completed. Configure the paths to the XGBoost, inductive, and
transductive output trees, then execute the notebook in order. It combines
five-seed metrics and produces the reported comparisons, including:

- species-ranked performance and performance changes;
- rare-species, prevalence, and threatened-species analyses;
- biomass and ecological-indicator comparisons;
- embedding/PCA and Sankey visualizations.

The rarity and threat analyses require external tables including
`/data/data/RLS/traits.csv`,
`/data/data/RLS/especes_menacees_edgar.csv`, and
`/data/data/RLS/Boyce_etal_2022_NATCC.csv`.
Update the notebook paths before interpreting those figures.

## 12. Produce maps with RLS5

Run `RLS5 - Australia maps.ipynb` after prediction-grid inference has been
performed for all five seeds. The notebook joins predictions to latitude,
longitude, depth, and date metadata, then uses the helpers in
`results_utils.py` to calculate seed means and confidence intervals.

The mapping helpers:

- `export_map()` writes two-band GeoTIFFs containing the mean and confidence
  interval;
- `convert_to_png()` renders prediction maps and uncertainty markers;
- `blindspots_map()` overlays threatened-species richness and highly protected
  areas.

Update the hard-coded paths to the ocean raster and protected-area layer:

```text
/home/gaetan/Downloads/oceans50m.tiff
/home/gaetan/Downloads/aus_highly_protected.gpkg
```

The notebook also writes map products under paths such as
`/home/gaetan/maps/log_sr/`, `/home/gaetan/maps/log_uicn_sr/`, and
`/home/gaetan/maps/<species>/`. Replace these with a writable local output
directory. The protected-area overlap result depends on the CRS, geometry,
buffering, and resolution of the replacement layer; record those choices.

## 13. Reproduction checklist

- raw-data and GeoEnrich input versions;
- prepared CSVs and raster-generation settings;
- each Hydra YAML after path and checkpoint overrides;
- foundation and supervised `last.ckpt` files;
- all prediction CSVs, Boyce outputs, metrics, and map rasters;
- Raw surveys and external raster products are absent.
- The unlabelled coastal grid and its rasters are absent.
- Trained checkpoints referenced by the configs are absent and their paths are
  dated and machine-specific.
