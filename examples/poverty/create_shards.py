"""Create WebDataset tar shards from Landsat TIFFs for DALI training.

Reads multi-band float64 Landsat TIFFs (from Planetary Computer), extracts
7 spectral bands, scales to uint16, center-crops to 224x224, stacks
4 trimesters into 28 channels, and writes fold-grouped tar shards ready
for GPU data loading.

The output shards can be consumed by ``DALIWebDatasetModule`` from
``dali_datamodule.py`` for ~20x faster training compared to per-file TIFF
reads.

Example usage::

    from create_shards import create_shards

    create_shards(
        source_dir="path/to/tiffs",          # country/year/cluster_trimester.tiff
        source_csv="path/to/dataset.csv",    # semicolon-separated CSV
        output_dir="path/to/shards",         # will be created
        num_workers=32,                      # parallel TIFF reads (default: all cores)
    )
"""

import io
import json
import os
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

# ── Band selection ──────────────────────────────────────────────────────────
# 6 reflectance + 1 thermal — the spectral bands useful for prediction.
# Aliases handle different Landsat sensor naming conventions.

TARGET_BANDS = ['red', 'green', 'blue', 'nir08', 'swir16', 'swir22']
REFLECTANCE_BANDS = {'red', 'green', 'blue', 'nir08', 'swir16', 'swir22'}
THERMAL_BANDS = set()

BAND_ALIASES = {
    'red': ['red', 'SR_B3', 'SR_B4'],
    'green': ['green', 'SR_B2', 'SR_B3'],
    'blue': ['blue', 'SR_B1', 'SR_B2'],
    'nir08': ['nir08', 'nir', 'SR_B4', 'SR_B5'],
    'swir16': ['swir16', 'swir1', 'SR_B5', 'SR_B6'],
    'swir22': ['swir22', 'swir2', 'SR_B7'],
}


def _find_band_index(descriptions, target_name):
    """Find 1-based band index by name, trying aliases."""
    for alias in BAND_ALIASES[target_name]:
        if alias in descriptions:
            return descriptions.index(alias) + 1
    return None


def _read_tile(tiff_path, crop_size):
    """Read a Landsat TIFF and return a (7, crop_size, crop_size) uint16 array.

    Selects the 7 target bands from the multi-band float64 source,
    replaces NaN with 0, scales to uint16 (reflectance x 10000,
    thermal x 100), and center-crops to *crop_size*.
    """
    with rasterio.open(tiff_path) as src:
        descriptions = src.descriptions
        tile = np.empty((len(TARGET_BANDS), src.height, src.width), dtype=np.uint16)
        for i, name in enumerate(TARGET_BANDS):
            idx = _find_band_index(descriptions, name)
            if idx is None:
                raise ValueError(
                    f"Band '{name}' not found in {tiff_path} "
                    f"(available: {descriptions})"
                )
            data = src.read(idx).astype(np.float64)
            data = np.nan_to_num(data, nan=0.0)
            if name in REFLECTANCE_BANDS:
                data *= 10000.0
            elif name in THERMAL_BANDS:
                data *= 100.0
            np.clip(data, 0, 65535, out=data)
            tile[i] = data.astype(np.uint16)

    # CenterCrop from any input size
    _, h, w = tile.shape
    oh = (h - crop_size) // 2
    ow = (w - crop_size) // 2
    return tile[:, oh:oh + crop_size, ow:ow + crop_size]


def _process_sample(source_dir, country, year, cluster_id, iwi, crop_size):
    """Read and preprocess one sample (4 trimesters). Worker function.

    Returns (key, stacked_bytes, label_bytes, band_sum, band_sq_sum, n_pixels)
    or None if any trimester is missing.

    band_sum/band_sq_sum are (7,) float64 arrays in physical units
    (after uint16 rescale), accumulated over all 4 trimesters and all pixels.
    """
    source_dir = Path(source_dir)
    key = f"{country}_{year}_{cluster_id}"
    tiles = []
    for trimester in range(1, 5):
        tiff_path = source_dir / country / year / f"{cluster_id}_{trimester}.tif"
        if not tiff_path.exists():
            return None
        tiles.append(_read_tile(str(tiff_path), crop_size))

    # Accumulate per-band stats in physical units for normalization
    scale = np.array(
    [1 / 10000.0 if b in REFLECTANCE_BANDS else 1 / 100.0 for b in TARGET_BANDS],
    dtype=np.float64,
)
    band_sum = np.zeros(len(TARGET_BANDS), dtype=np.float64)
    band_sq_sum = np.zeros(len(TARGET_BANDS), dtype=np.float64)
    n_pixels = 0
    for tile in tiles:
        phys = tile.astype(np.float64) * scale[:, None, None]
        band_sum += phys.sum(axis=(1, 2))
        band_sq_sum += (phys ** 2).sum(axis=(1, 2))
        n_pixels += tile.shape[1] * tile.shape[2]

    stacked = np.concatenate(tiles, axis=0)  # (28, crop_size, crop_size) uint16
    return key, stacked.tobytes(), np.float32(iwi).tobytes(), band_sum, band_sq_sum, n_pixels


# ── Shard creation ──────────────────────────────────────────────────────────

def create_shards(
    source_dir: str,
    source_csv: str,
    output_dir: str,
    samples_per_shard: int = 256,
    n_folds: int = 5,
    fold_column: str = None,
    seed: int = 42,
    crop_size: int = 224,
    num_workers: int = None,
):
    """Pack Landsat TIFFs into fold-based WebDataset shards.

    Reads multi-band float64 Landsat TIFFs directly, extracts the 7
    spectral bands (6 reflectance + 1 thermal), scales to uint16,
    center-crops to *crop_size*, stacks 4 trimesters into 28 channels,
    and writes raw uint16 bytes into tar shards grouped by fold.

    TIFF reading is parallelized across *num_workers* processes for
    throughput on multi-core / Lustre systems.

    Per-sample tar entries:
        {key}.input  -- raw uint16 bytes for (28, crop_size, crop_size)
        {key}.target -- raw 4-byte float32 (IWI label)

    Shard naming: fold{k}-{shard_idx:06d}.tar

    Parameters
    ----------
    source_dir : str
        Directory with Landsat TIFFs (country/year/cluster_trimester.tiff).
    source_csv : str
        CSV file with cluster metadata (semicolon-separated).
    output_dir : str
        Output directory for tar shards.
    samples_per_shard : int
        Number of samples per tar shard.
    n_folds : int
        Number of cross-validation folds.
    fold_column : str, optional
        CSV column containing fold assignments.  Values are sorted and
        mapped to 0 .. n_folds-1.  If None, folds are assigned by a
        deterministic shuffle with *seed*.
    seed : int
        Random seed for fold assignment (ignored when *fold_column* is set).
    crop_size : int
        Spatial size after center crop.
    num_workers : int, optional
        Number of parallel worker processes for TIFF reading.
        Defaults to ``os.cpu_count()``.
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if num_workers is None:
        num_workers = os.cpu_count()

    n_bands = len(TARGET_BANDS)
    n_channels = n_bands * 4  # 4 trimesters

    df = pd.read_csv(source_csv, sep=";")
    n = len(df)
    print(f"Packing {n} clusters into {n_folds}-fold shards "
          f"({n_bands} bands x 4 trimesters = {n_channels} channels, "
          f"uint16, {crop_size}x{crop_size}, {num_workers} workers)")

    # --- Assign each sample to a fold ---
    if fold_column and fold_column in df.columns:
        unique_vals = sorted(df[fold_column].unique())
        if len(unique_vals) != n_folds:
            raise ValueError(
                f"fold_column '{fold_column}' has {len(unique_vals)} unique "
                f"values but n_folds={n_folds}"
            )
        val_to_fold = {v: i for i, v in enumerate(unique_vals)}
        fold_ids = df[fold_column].map(val_to_fold).values
        print(f"  Using fold assignments from column '{fold_column}'")
    else:
        rng = np.random.default_rng(seed)
        indices = np.arange(n)
        rng.shuffle(indices)
        fold_ids = np.empty(n, dtype=int)
        for i in range(n):
            fold_ids[indices[i]] = i % n_folds
        print(f"  Generated {n_folds}-fold assignment (seed={seed})")

    # Group rows by fold
    fold_samples = {k: [] for k in range(n_folds)}
    for idx, row in df.iterrows():
        fold_samples[fold_ids[idx]].append((idx, row))

    fold_sizes = {}
    src_dir_str = str(source_dir)

    # Running accumulators for per-band normalization stats (physical units)
    total_band_sum = np.zeros(n_bands, dtype=np.float64)
    total_band_sq_sum = np.zeros(n_bands, dtype=np.float64)
    total_pixels = 0

    for fold_k, samples in fold_samples.items():
        if not samples:
            continue

        n_shards = (len(samples) + samples_per_shard - 1) // samples_per_shard
        print(f"  fold{fold_k}: {len(samples)} samples -> {n_shards} shards")

        # Submit all samples in this fold to the process pool
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            # Submit in order, preserving index for deterministic shard assignment
            futures = []
            for df_idx, row in samples:
                country = str(row["country"]).lower()
                year = str(row["year"])
                cluster_id = str(row["cluster_id"])
                iwi = float(row["iwi"])
                fut = pool.submit(
                    _process_sample,
                    src_dir_str, country, year, cluster_id, iwi, crop_size,
                )
                futures.append(fut)

            # Collect results in submission order and write to tar
            shard_idx = 0
            sample_count = 0
            tar = None
            skipped = 0

            for fut in futures:
                result = fut.result()
                if result is None:
                    skipped += 1
                    continue

                key, input_bytes, label_bytes, band_sum, band_sq_sum, n_pix = result

                # Accumulate normalization stats
                total_band_sum += band_sum
                total_band_sq_sum += band_sq_sum
                total_pixels += n_pix

                if sample_count % samples_per_shard == 0:
                    if tar is not None:
                        tar.close()
                    shard_name = f"fold{fold_k}-{shard_idx:06d}.tar"
                    tar = tarfile.open(output_dir / shard_name, "w")
                    shard_idx += 1

                info = tarfile.TarInfo(name=f"{key}.input")
                info.size = len(input_bytes)
                tar.addfile(info, io.BytesIO(input_bytes))

                info = tarfile.TarInfo(name=f"{key}.target")
                info.size = len(label_bytes)
                tar.addfile(info, io.BytesIO(label_bytes))

                sample_count += 1

            if tar is not None:
                tar.close()

            if skipped:
                print(f"    ({skipped} samples skipped due to missing TIFFs)")

        fold_sizes[fold_k] = sample_count

    # Compute per-band normalization stats (physical units, 7 bands)
    print(total_pixels)
    band_mean = total_band_sum / total_pixels
    band_std = np.sqrt(total_band_sq_sum / total_pixels - band_mean ** 2)

    normalize = {
        "mean": band_mean.tolist(),
        "std": band_std.tolist(),
        "bands": TARGET_BANDS,
    }
    with open(output_dir / "normalize.json", "w") as f:
        json.dump(normalize, f, indent=2)
    print(f"  Normalization stats (7-band) saved to {output_dir / 'normalize.json'}")

    # Write metadata
    meta = {
        "format": "uint16",
        "shape": [n_channels, crop_size, crop_size],
        "dtype": "uint16",
        "n_bands": n_bands,
        "n_trimesters": 4,
        "n_folds": n_folds,
        "fold_sizes": fold_sizes,
        "total_samples": sum(fold_sizes.values()),
        "samples_per_shard": samples_per_shard,
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Done. Shards written to {output_dir}")

if __name__=="__main__":
    
    create_shards(
    source_dir="../../../images/seasonal",              # country/year/cluster_trimester.tiff
    source_csv="madagascar_folds.csv",        # semicolon-separated CSV
    output_dir="../../../webdataset_mada",             # will be created
    n_folds=5,                               # number of CV folds
    num_workers=32,                          # parallel TIFF reads (default: all cores)
    fold_column="fold",                    # uncomment if your CSV has a fold column
    # seed=42,                               # deterministic fold assignment otherwise
)
