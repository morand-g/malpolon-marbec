"""Create canonical WebDataset shards for DALI training.

This version is designed for the poverty benchmark canonical folds. It can
create either:

- composite shards: one 6-band TIFF per sample -> 6 channels
- seasonal shards: four 6-band TIFFs per sample -> 24 channels

Fold membership is NEVER regenerated locally. It is read from the canonical
split CSV (sample_id, fold), so the resulting fold{k}-*.tar files match the
benchmark split exactly.
"""

import argparse
import io
import json
import os
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling


TARGET_BANDS = ["red", "green", "blue", "nir08", "swir16", "swir22"]
NORMALIZE_MEAN = [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0]
NORMALIZE_STD = [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0]
NORMALIZE_MAX_PIXEL_VALUE = 0.0001

BAND_ALIASES = {
    "red": ["red", "SR_B3", "SR_B4"],
    "green": ["green", "SR_B2", "SR_B3"],
    "blue": ["blue", "SR_B1", "SR_B2"],
    "nir08": ["nir08", "nir", "SR_B4", "SR_B5"],
    "swir16": ["swir16", "swir1", "SR_B5", "SR_B6"],
    "swir22": ["swir22", "swir2", "SR_B7"],
}

# Composite TIFFs used in this project may have no band descriptions and use
# the physical order: blue, green, red, nir, swir1, swir2.
# We reorder them to TARGET_BANDS: red, green, blue, nir, swir1, swir2.
UNDESCRIBED_6BAND_INDEX = {
    "red": 3,
    "green": 2,
    "blue": 1,
    "nir08": 4,
    "swir16": 5,
    "swir22": 6,
}


def _read_csv_auto(path):
    """Read comma- or semicolon-separated CSV."""
    return pd.read_csv(path, sep=None, engine="python")


def _normalize_ids(df):
    """Ensure metadata has sample_id, country, year and cluster_id."""
    df = df.copy()

    if "sample_id" not in df.columns:
        required = {"country", "year", "cluster_id"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                "Metadata CSV needs sample_id or country/year/cluster_id. "
                f"Missing: {sorted(missing)}"
            )
        df["sample_id"] = (
            df["country"].astype(str).str.lower()
            + "_"
            + df["year"].astype(int).astype(str)
            + "_"
            + df["cluster_id"].astype(int).astype(str)
        )
    else:
        df["sample_id"] = df["sample_id"].astype(str).str.lower()

    if not {"country", "year", "cluster_id"}.issubset(df.columns):
        parts = df["sample_id"].str.rsplit("_", n=2, expand=True)
        if parts.shape[1] != 3:
            raise ValueError("Cannot reconstruct country/year/cluster_id from sample_id")
        df["country"] = parts[0]
        df["year"] = parts[1].astype(int)
        df["cluster_id"] = parts[2].astype(int)

    df["country"] = df["country"].astype(str).str.lower()
    df["year"] = df["year"].astype(int)
    df["cluster_id"] = df["cluster_id"].astype(int)
    return df


def _find_band_index(src, target_name):
    descriptions = list(src.descriptions or ())
    for alias in BAND_ALIASES[target_name]:
        if alias in descriptions:
            return descriptions.index(alias) + 1

    # Canonical 6-band composite files in this project have no descriptions.
    if src.count == 6 and not any(descriptions):
        return UNDESCRIBED_6BAND_INDEX[target_name]

    return None


def _read_tile(tiff_path, image_size):
    """Read one TIFF and return (6, H, W) uint16 in TARGET_BANDS order."""
    with rasterio.open(tiff_path) as src:
        tile = np.empty((len(TARGET_BANDS), image_size, image_size), dtype=np.uint16)

        for i, name in enumerate(TARGET_BANDS):
            idx = _find_band_index(src, name)
            if idx is None:
                raise ValueError(
                    f"Band '{name}' not found in {tiff_path}; "
                    f"count={src.count}, descriptions={src.descriptions}"
                )

            data = src.read(
                idx,
                out_shape=(image_size, image_size),
                resampling=Resampling.bilinear,
            ).astype(np.float64)

            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

            # Source TIFFs contain reflectance in physical units (~0-1.6).
            # DALI shards store uint16 reflectance * 10000.
            data *= 10000.0
            np.clip(data, 0, 65535, out=data)
            tile[i] = data.astype(np.uint16)

    return tile


def _find_composite_path(source_dir, country, year, cluster_id):
    base = Path(source_dir) / country / str(year)
    candidates = [
        base / f"{cluster_id}_RGB_224.tif",
        base / f"{cluster_id}.tif",
        base / f"{cluster_id}_composite.tif",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _process_sample(source_dir, country, year, cluster_id, iwi, image_size, nature):
    """Worker: return packed sample bytes plus per-band statistics."""
    source_dir = Path(source_dir)
    key = f"{country}_{year}_{cluster_id}"

    if nature == "seasonal":
        paths = [
            source_dir / country / str(year) / f"{cluster_id}_{trimester}.tif"
            for trimester in range(1, 5)
        ]
    elif nature == "composite":
        path = _find_composite_path(source_dir, country, year, cluster_id)
        paths = [] if path is None else [path]
    else:
        raise ValueError(f"Unsupported nature={nature!r}")

    if not paths or any(not p.exists() for p in paths):
        expected = paths if paths else [Path(source_dir) / country / str(year) / f"{cluster_id}_RGB_224.tif"]
        return {
            "error": "missing_tiff",
            "key": key,
            "paths": [str(p) for p in expected],
        }

    tiles = [_read_tile(str(path), image_size) for path in paths]

    band_sum = np.zeros(len(TARGET_BANDS), dtype=np.float64)
    band_sq_sum = np.zeros(len(TARGET_BANDS), dtype=np.float64)
    n_pixels = 0

    for tile in tiles:
        values = tile.astype(np.float64)
        band_sum += values.sum(axis=(1, 2))
        band_sq_sum += (values ** 2).sum(axis=(1, 2))
        n_pixels += tile.shape[1] * tile.shape[2]

    stacked = np.concatenate(tiles, axis=0)

    return {
        "key": key,
        "input_bytes": stacked.tobytes(),
        "label_bytes": np.float32(iwi).tobytes(),
        "band_sum": band_sum,
        "band_sq_sum": band_sq_sum,
        "n_pixels": n_pixels,
    }


def _prepare_dataframe(source_csv, split_csv, label_column, n_folds):
    meta = _normalize_ids(_read_csv_auto(source_csv))
    split = _read_csv_auto(split_csv)

    required_split = {"sample_id", "fold"}
    missing = required_split - set(split.columns)
    if missing:
        raise ValueError(f"Canonical split is missing columns: {sorted(missing)}")

    split = split[["sample_id", "fold"]].copy()
    split["sample_id"] = split["sample_id"].astype(str).str.lower()
    split["fold"] = split["fold"].astype(int)

    if split["sample_id"].duplicated().any():
        dup = split.loc[split["sample_id"].duplicated(), "sample_id"].head().tolist()
        raise ValueError(f"Duplicate sample_id in split CSV, e.g. {dup}")

    expected_folds = set(range(n_folds))
    observed_folds = set(split["fold"].unique())
    if observed_folds != expected_folds:
        raise ValueError(
            f"Expected folds {sorted(expected_folds)}, got {sorted(observed_folds)}"
        )

    if label_column not in meta.columns:
        alternatives = [c for c in ("iwi", "awi", "label", "target") if c in meta.columns]
        if alternatives:
            raise ValueError(
                f"Label column '{label_column}' not found. Available candidate(s): {alternatives}"
            )
        raise ValueError(f"Label column '{label_column}' not found in metadata CSV")

    meta = meta.drop_duplicates("sample_id", keep=False)
    merged = split.merge(meta, on="sample_id", how="left", validate="one_to_one")

    missing_meta = merged[label_column].isna()
    if missing_meta.any():
        ids = merged.loc[missing_meta, "sample_id"].head(20).tolist()
        raise ValueError(
            f"{missing_meta.sum()} canonical split samples are missing metadata/labels. "
            f"Examples: {ids}"
        )

    # Important: preserve canonical split order within each fold.
    return merged


def _clean_output_dir(output_dir):
    for path in output_dir.glob("fold*-*.tar"):
        path.unlink()
    for name in ("meta.json", "normalize.json", "missing_samples.txt"):
        path = output_dir / name
        if path.exists():
            path.unlink()


def create_shards(
    source_dir: str,
    source_csv: str,
    split_csv: str,
    output_dir: str,
    nature: str = "composite",
    label_column: str = "iwi",
    samples_per_shard: int = 256,
    n_folds: int = 5,
    image_size: int = 224,
    num_workers: int | None = None,
    overwrite: bool = False,
):
    """Create WebDataset shards that exactly follow the canonical split."""

    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if num_workers is None:
        num_workers = os.cpu_count() or 1

    existing = list(output_dir.glob("fold*-*.tar"))
    if existing and not overwrite:
        raise FileExistsError(
            f"{output_dir} already contains {len(existing)} fold shard(s). "
            "Use overwrite=True / --overwrite to replace them."
        )
    if overwrite:
        _clean_output_dir(output_dir)

    df = _prepare_dataframe(source_csv, split_csv, label_column, n_folds)

    n_trimesters = 4 if nature == "seasonal" else 1
    n_channels = len(TARGET_BANDS) * n_trimesters

    print(
        f"Packing {len(df)} canonical samples into {n_folds} folds: "
        f"nature={nature}, channels={n_channels}, size={image_size}, workers={num_workers}"
    )
    print("Canonical fold counts:")
    print(df["fold"].value_counts().sort_index().to_string())

    fold_sizes = {}
    total_band_sum = np.zeros(len(TARGET_BANDS), dtype=np.float64)
    total_band_sq_sum = np.zeros(len(TARGET_BANDS), dtype=np.float64)
    total_pixels = 0
    missing_records = []

    src_dir_str = str(source_dir)

    for fold_k in range(n_folds):
        samples = df[df["fold"] == fold_k]
        expected_count = len(samples)
        expected_shards = (expected_count + samples_per_shard - 1) // samples_per_shard
        print(f"\nfold{fold_k}: {expected_count} samples -> {expected_shards} shard(s)")

        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futures = []
            for _, row in samples.iterrows():
                futures.append(
                    pool.submit(
                        _process_sample,
                        src_dir_str,
                        str(row["country"]).lower(),
                        int(row["year"]),
                        int(row["cluster_id"]),
                        float(row[label_column]),
                        image_size,
                        nature,
                    )
                )

            shard_idx = 0
            sample_count = 0
            tar = None

            try:
                for fut in futures:
                    result = fut.result()

                    if "error" in result:
                        missing_records.append(result)
                        continue

                    total_band_sum += result["band_sum"]
                    total_band_sq_sum += result["band_sq_sum"]
                    total_pixels += result["n_pixels"]

                    if sample_count % samples_per_shard == 0:
                        if tar is not None:
                            tar.close()
                        shard_name = f"fold{fold_k}-{shard_idx:06d}.tar"
                        tar = tarfile.open(output_dir / shard_name, "w")
                        shard_idx += 1

                    info = tarfile.TarInfo(name=f"{result['key']}.input")
                    info.size = len(result["input_bytes"])
                    tar.addfile(info, io.BytesIO(result["input_bytes"]))

                    info = tarfile.TarInfo(name=f"{result['key']}.target")
                    info.size = len(result["label_bytes"])
                    tar.addfile(info, io.BytesIO(result["label_bytes"]))

                    sample_count += 1
            finally:
                if tar is not None:
                    tar.close()

        fold_sizes[fold_k] = sample_count
        print(f"  wrote {sample_count}/{expected_count} samples")

    if missing_records:
        missing_path = output_dir / "missing_samples.txt"
        with open(missing_path, "w") as f:
            for rec in missing_records:
                f.write(f"{rec['key']}\t" + "\t".join(rec["paths"]) + "\n")

        # Do not silently create shards that fail canonical validation later.
        raise RuntimeError(
            f"{len(missing_records)} canonical samples are missing TIFFs. "
            f"See {missing_path}. Shards are incomplete and must not be used."
        )

    # Strong final invariant: shard counts must equal canonical split counts.
    expected_sizes = df["fold"].value_counts().sort_index().to_dict()
    for fold_k in range(n_folds):
        if fold_sizes.get(fold_k, 0) != expected_sizes.get(fold_k, 0):
            raise RuntimeError(
                f"fold{fold_k}: wrote {fold_sizes.get(fold_k, 0)} samples, "
                f"expected {expected_sizes.get(fold_k, 0)} from canonical split"
            )

    normalize = {
        "mean": NORMALIZE_MEAN,
        "std": NORMALIZE_STD,
        "max_pixel_value": NORMALIZE_MAX_PIXEL_VALUE,
        "bands": TARGET_BANDS,
    }
    with open(output_dir / "normalize.json", "w") as f:
        json.dump(normalize, f, indent=2)

    meta = {
        "format": "uint16",
        "shape": [n_channels, image_size, image_size],
        "dtype": "uint16",
        "nature": nature,
        "n_bands": len(TARGET_BANDS),
        "n_trimesters": n_trimesters,
        "n_folds": n_folds,
        "fold_sizes": fold_sizes,
        "total_samples": sum(fold_sizes.values()),
        "samples_per_shard": samples_per_shard,
        "source_csv": str(source_csv),
        "split_csv": str(split_csv),
        "label_column": label_column,
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. Canonical shards written to {output_dir}")
    print(f"Total samples: {sum(fold_sizes.values())}")


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", required=True)
    p.add_argument("--source-csv", required=True)
    p.add_argument("--split-csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--nature", choices=("composite", "seasonal"), default="composite")
    p.add_argument("--label-column", default="iwi")
    p.add_argument("--samples-per-shard", type=int, default=256)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    create_shards(
        source_dir=args.source_dir,
        source_csv=args.source_csv,
        split_csv=args.split_csv,
        output_dir=args.output_dir,
        nature=args.nature,
        label_column=args.label_column,
        samples_per_shard=args.samples_per_shard,
        n_folds=args.n_folds,
        image_size=args.image_size,
        num_workers=args.num_workers,
        overwrite=args.overwrite,
    )
