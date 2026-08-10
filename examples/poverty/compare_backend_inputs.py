"""Validate TIFF labels and existing DALI shards against a canonical split."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from canonical_split import (
    load_canonical_folds,
    sample_id_from_row,
    split_sample_ids,
    validate_shard_folds,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument("--labels-path", type=Path, required=True)
    parser.add_argument("--wds-root", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()

    canonical_fold_ids = load_canonical_folds(args.split_path)
    canonical_ids = {
        sample_id
        for fold_ids in canonical_fold_ids.values()
        for sample_id in fold_ids
    }

    with args.labels_path.open(encoding="utf-8", newline="") as stream:
        dialect = csv.Sniffer().sniff(stream.read(4096), delimiters=",;")
        stream.seek(0)
        tiff_ids = [
            sample_id_from_row(row)
            for row in csv.DictReader(stream, dialect=dialect)
        ]

    selected_tiff_ids = [
        sample_id for sample_id in tiff_ids if sample_id in canonical_ids
    ]
    if (
        set(selected_tiff_ids) != canonical_ids
        or len(selected_tiff_ids) != len(canonical_ids)
    ):
        raise ValueError("TIFF labels do not match canonical sample IDs exactly")

    dali_fold_ids = validate_shard_folds(args.wds_root, canonical_fold_ids)
    tiff_split_ids = split_sample_ids(canonical_fold_ids, args.fold)
    dali_split_ids = split_sample_ids(
        {label: sorted(ids) for label, ids in dali_fold_ids.items()},
        args.fold,
    )
    print(
        "TIFF split counts: "
        + ", ".join(
            f"{split}={len(ids)}" for split, ids in tiff_split_ids.items()
        )
    )
    print(
        "DALI split counts: "
        + ", ".join(
            f"{split}={len(ids)}" for split, ids in dali_split_ids.items()
        )
    )
    print("TIFF sample IDs match canonical split exactly: yes")
    print("DALI fold sample IDs match canonical split exactly: yes")


if __name__ == "__main__":
    main()
