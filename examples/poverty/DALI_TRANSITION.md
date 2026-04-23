# Switching to DALI GPU Data Loading

This guide describes how to replace the current `MSDataModule` (per-sample
TIFF reads via rasterio) with an NVIDIA DALI pipeline that runs the entire
decode + transform chain on GPU.

## What changes

| Aspect | Current pipeline | DALI pipeline |
|--------|-----------------|---------------|
| Data format | Individual TIFFs on disk | Pre-packed WebDataset tar shards |
| Band selection | 16 bands from SPECTRUM_ALL | 7 spectral bands (6 reflectance + 1 thermal) |
| Storage dtype | float64 (original Landsat) | uint16 (~2.8 MB/sample vs ~25 MB raw) |
| Decode | Python: 4x rasterio open per sample | GPU: reinterpret uint16 + cast float32 (one kernel) |
| Transforms | CPU: CenterCrop, RandomFlip, Normalize (per trimester) | GPU: Normalize, RandomFlip (on full 28-ch tile) |
| I/O | Per-sample file opens (overwhelms Lustre MDS) | DALI C++ threads read sequentially from tar |
| Transfer | Explicit Host-to-Device per batch | DALI DMA, overlapped with compute |

## New files

Drop these two files next to your existing code (alongside
`cnn_on_ms_poverty.py`):

- **`create_shards.py`** -- one-time data preparation (original TIFFs -> tar shards)
- **`dali_datamodule.py`** -- drop-in DataModule replacement for training

No changes needed to any existing files.

## Prerequisites

```bash
pip install nvidia-dali-cuda120
```

## Step 1: Create shards (one-time)

This reads your original Landsat TIFFs directly (multi-band float64 from
Planetary Computer), selects the 7 spectral bands by name (handling Landsat
sensor naming variations automatically), replaces NaN with 0, scales to
uint16 (reflectance x10000, thermal x100), center-crops to 224x224, stacks
4 trimesters into 28 channels, and writes raw uint16 bytes into tar shards
grouped by fold.

It also computes 7-band normalization statistics (mean/std in physical
units) across the entire dataset and saves them to `normalize.json`.

Edit and run `create_shards.py`:

```python
from create_shards import create_shards

create_shards(
    source_dir="path/to/tiffs",              # country/year/cluster_trimester.tiff
    source_csv="path/to/dataset.csv",        # semicolon-separated CSV
    output_dir="path/to/shards",             # will be created
    n_folds=5,                               # number of CV folds
    num_workers=32,                          # parallel TIFF reads (default: all cores)
    # fold_column="fold",                    # uncomment if your CSV has a fold column
    # seed=42,                               # deterministic fold assignment otherwise
)
```

This produces:

```
path/to/shards/
    fold0-000000.tar
    ...
    fold4-000092.tar
    meta.json          # shard metadata (shape, dtype, fold sizes)
    normalize.json     # 7-band mean/std (physical units)
```

One directory, all 5 folds.  No pickle files.

### Band selection

The 7 target bands are: **red, green, blue, nir08, swir16, swir22, lwir**.
These are the 6 reflectance bands plus 1 thermal band useful for prediction.
The code handles different Landsat sensor naming conventions (e.g.
`SR_B3`/`SR_B4` for red depending on Landsat 7 vs 8) via an alias table.

### Storage format

Tiles are stored as raw uint16 bytes (no numpy headers, no compression).
Reflectance values are multiplied by 10000 and thermal by 100 before
casting to uint16.  This halves storage vs float32 while preserving 4-5
significant digits -- more than sufficient for satellite imagery.

## Step 2: Use in training

Replace your `MSDataModule` with `DALIWebDatasetModule`.  The model code
does not change -- it receives `(B, 28, 224, 224)` float32 tensors,
already normalized.

```python
# BEFORE
datamodule = MSDataModule(
    dataset_path="path/to/tiffs",
    labels_name="/../dataset.csv",
    train_batch_size=32,
    inference_batch_size=16,
    num_workers=2,
    fold="A",
    fold_path="folds.pkl",
    nature="seasonal",
    dict_normalize="mean_std_normalize_all.json",
)
trainer.fit(model, datamodule=datamodule)

# AFTER
from dali_datamodule import DALIWebDatasetModule

datamodule = DALIWebDatasetModule(
    wds_dir="path/to/shards",
    fold=0,                     # 0-4 for 5-fold CV
    n_folds=5,
    train_batch_size=32,
    inference_batch_size=16,
    num_workers=4,              # DALI I/O threads (not PyTorch workers)
)
# DALI delivers tensors already on GPU -- tell Lightning not to re-transfer
datamodule.transfer_batch_to_device = lambda batch, device, idx: batch
trainer.fit(model, datamodule=datamodule)
```

### Key differences from the old pipeline

- **7 bands instead of 16**: only the spectral bands useful for prediction
  are kept.  Model input is `(B, 28, 224, 224)` instead of `(B, 64, 224, 224)`.
  Adjust your model's first layer `in_channels` accordingly.
- **Normalization stats**: computed automatically during shard creation
  and stored in `normalize.json`.  No need to provide `mean_std_normalize_all.json`.
- **Fold indexing**: folds are integers 0-4 (not letters A-E).

## How folds work

At training time, the `fold` parameter controls which shard groups are
used for each split:

| `fold=` | test | val | train |
|---------|------|-----|-------|
| 0 | fold0 | fold1 | fold2, fold3, fold4 |
| 1 | fold1 | fold2 | fold3, fold4, fold0 |
| 2 | fold2 | fold3 | fold4, fold0, fold1 |
| 3 | fold3 | fold4 | fold0, fold1, fold2 |
| 4 | fold4 | fold0 | fold1, fold2, fold3 |

No re-sharding needed to switch folds.

## How normalization works

`create_shards.py` computes 7-band mean/std statistics (in physical units)
across the entire dataset and saves them to `normalize.json`.  The DALI
module reads these stats, tiles them to 28 channels (4 trimesters x 7
bands), and folds the uint16-to-physical rescale into the normalization
constants.  This means the GPU does a single `(tile - mean) / std`
operation that simultaneously rescales and normalizes -- no extra multiply.

## How shuffling works (Lustre-friendly)

DALI does **not** randomly seek across shards to build a batch.  Instead:

1. **Shard-order shuffle**: at the start of each epoch, the *order* of
   shards is permuted.  Each I/O thread then reads its assigned shards
   **sequentially from start to end** -- one contiguous sequential read
   per shard.  This is the ideal Lustre access pattern.

2. **Shuffle buffer**: after reading, samples pass through a finite
   in-memory shuffle buffer.  This provides within-epoch randomization
   without random I/O.

Net effect: at any moment, DALI is reading at most `num_workers` shards
(one per I/O thread), each sequentially.

## Lustre / HPC notes

- Tar shards are Lustre-friendly: ~94 shards per fold for 120k samples
  (vs ~120k individual TIFF files per fold that would overwhelm the MDS)
- Default 256 samples/shard; each sample is ~2.8 MB (28 x 224 x 224 x
  uint16), so ~700 MB per shard
- DALI's `num_workers` controls CPU I/O threads, not PyTorch worker
  processes.  4-8 threads is usually sufficient since the CPU work is
  minimal (just sequential tar reads)
- Sequential reads within shards means Lustre's read-ahead caching works
  well; no random small-file opens
- Shard creation is parallelized with `ProcessPoolExecutor` -- set
  `num_workers` to your core count (40 on typical HPC nodes) for
  maximum throughput when processing the original TIFFs
