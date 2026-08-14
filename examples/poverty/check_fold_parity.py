from __future__ import annotations

import argparse
import tarfile
from collections import Counter
from pathlib import Path

import pandas as pd


def make_key(country, year, cluster_id) -> str:
    return f"{str(country).lower()}_{int(year)}_{int(cluster_id)}"


def read_fold_csv(csv_path: Path) -> pd.DataFrame:
    """
    Read the full fold CSV.

    Tries semicolon first, then comma if needed.
    """
    df = pd.read_csv(csv_path, sep=";")

    if len(df.columns) == 1:
        df = pd.read_csv(csv_path)

    df["key"] = df["sample_id"]

    return df


def read_dali_fold_keys(
    dali_root: Path,
    fold: int,
) -> set[str]:
    """
    Read all sample keys stored in one DALI fold.
    """
    tar_paths = sorted(
        dali_root.glob(f"fold{fold}-*.tar")
    )

    if not tar_paths:
        raise FileNotFoundError(
            f"No shards found for fold{fold} in {dali_root}"
        )

    keys = set()

    for tar_path in tar_paths:
        with tarfile.open(tar_path, "r") as tar:
            for member in tar.getmembers():
                if member.name.endswith(".input"):
                    key = member.name.removesuffix(".input")
                    keys.add(key)

    return keys


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare DALI fold membership with the spatial_fold "
            "assignments in the full source CSV."
        )
    )

    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Full Madagascar fold CSV.",
    )

    parser.add_argument(
        "--dali-root",
        type=Path,
        required=True,
        help="Directory containing fold0-*.tar ... fold4-*.tar.",
    )

    args = parser.parse_args()

    df = read_fold_csv(args.csv)

    print("=== CSV ===")
    print("rows:", len(df))
    print("unique keys:", df["key"].nunique())
    print(
        "spatial_fold distribution:",
        df["fold"].value_counts().sort_index().to_dict(),
    )

    if df["key"].duplicated().any():
        duplicated = df.loc[
            df["key"].duplicated(keep=False),
            ["key", "fold"],
        ]

        print("\nWARNING: duplicated CSV keys found")
        print(duplicated.head(20).to_string(index=False))

    key_to_spatial_fold = dict(
        zip(
            df["key"],
            df["fold"],
        )
    )

    csv_keys = set(key_to_spatial_fold)

    print("\n=== DALI FOLD MAPPING ===")

    all_dali_keys = set()

    for dali_fold in range(5):
        shard_keys = read_dali_fold_keys(
            args.dali_root,
            dali_fold,
        )

        all_dali_keys |= shard_keys

        matched = shard_keys & csv_keys
        absent_from_csv = shard_keys - csv_keys

        spatial_counts = Counter(
            int(key_to_spatial_fold[key])
            for key in matched
        )

        print(f"\nDALI fold{dali_fold}")
        print("  shard samples:", len(shard_keys))
        print("  matched in CSV:", len(matched))
        print("  absent from CSV:", len(absent_from_csv))
        print(
            "  spatial_fold distribution:",
            dict(sorted(spatial_counts.items())),
        )

        if absent_from_csv:
            print(
                "  first keys absent from CSV:",
                sorted(absent_from_csv)[:10],
            )

        if spatial_counts:
            dominant_fold, dominant_count = spatial_counts.most_common(1)[0]

            purity = dominant_count / len(matched)

            print(
                f"  dominant spatial_fold: {dominant_fold}"
            )
            print(
                f"  mapping purity: {purity:.6f}"
            )

    print("\n=== GLOBAL COVERAGE ===")

    print(
        "unique DALI keys:",
        len(all_dali_keys),
    )

    print(
        "unique CSV keys:",
        len(csv_keys),
    )

    missing_from_dali = csv_keys - all_dali_keys
    extra_in_dali = all_dali_keys - csv_keys

    print(
        "CSV keys absent from all DALI shards:",
        len(missing_from_dali),
    )

    print(
        "DALI keys absent from CSV:",
        len(extra_in_dali),
    )

    if missing_from_dali:
        print(
            "first CSV keys absent from DALI:",
            sorted(missing_from_dali)[:20],
        )

    if extra_in_dali:
        print(
            "first DALI keys absent from CSV:",
            sorted(extra_in_dali)[:20],
        )


if __name__ == "__main__":
    main()
