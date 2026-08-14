"""Minimal adapter for an external canonical fold CSV."""

from __future__ import annotations

import csv
import tarfile
from collections.abc import Mapping
from pathlib import Path


FOLD_LABELS = ("A", "B", "C", "D", "E")


def normalize_fold(value: object) -> str:
    value = str(value).strip().upper()
    if value in FOLD_LABELS:
        return value
    if value.isdigit() and 0 <= int(value) < len(FOLD_LABELS):
        return FOLD_LABELS[int(value)]
    raise ValueError(f"Invalid fold label: {value!r}")


def sample_id_from_row(row: Mapping) -> str:
    sample_id = row.get("sample_id")
    if (
        sample_id is not None
        and str(sample_id).strip()
        and str(sample_id).strip().lower() != "nan"
    ):
        return str(sample_id).strip()

    year = row.get("year", row.get("survey_year"))
    required = (row.get("country"), year, row.get("cluster_id"))
    if any(value is None for value in required):
        raise ValueError(
            "Rows must contain sample_id or country/year/cluster_id"
        )
    return f"{str(required[0]).lower()}_{int(year)}_{int(required[2])}"


def load_canonical_folds(split_path: str | Path) -> dict[str, list[str]]:
    fold_ids = {label: [] for label in FOLD_LABELS}
    seen = set()

    with Path(split_path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or "fold" not in reader.fieldnames:
            raise ValueError(f"Canonical split CSV {split_path} needs a fold column")

        for row in reader:
            sample_id = sample_id_from_row(row)
            if sample_id in seen:
                raise ValueError(f"Duplicate canonical sample_id: {sample_id}")
            seen.add(sample_id)
            fold_ids[normalize_fold(row["fold"])].append(sample_id)

    return fold_ids


def split_sample_ids(
    fold_ids: dict[str, list[str]], fold: int
) -> dict[str, list[str]]:
    fold = int(fold) % len(FOLD_LABELS)
    test_fold = FOLD_LABELS[fold]
    val_fold = FOLD_LABELS[(fold + 1) % len(FOLD_LABELS)]
    train_folds = [
        FOLD_LABELS[(fold + offset) % len(FOLD_LABELS)]
        for offset in range(2, len(FOLD_LABELS))
    ]
    return {
        "test": list(fold_ids[test_fold]),
        "val": list(fold_ids[val_fold]),
        "train": [
            sample_id
            for label in train_folds
            for sample_id in fold_ids[label]
        ],
    }


def shard_fold_sample_ids(wds_dir: str | Path, fold: int) -> set[str]:
    shard_paths = sorted(Path(wds_dir).glob(f"fold{fold}-*.tar"))
    if not shard_paths:
        raise FileNotFoundError(f"No fold{fold} shards found in {wds_dir}")

    sample_ids = set()
    for shard_path in shard_paths:
        with tarfile.open(shard_path) as shard:
            for member in shard:
                if member.isfile() and "." in member.name:
                    sample_ids.add(member.name.rsplit(".", 1)[0])
    return sample_ids


def validate_shard_folds(
    wds_dir: str | Path,
    canonical_fold_ids: dict[str, list[str]],
) -> dict[str, set[str]]:
    actual_fold_ids = {}
    for fold, label in enumerate(FOLD_LABELS):
        expected = set(canonical_fold_ids[label])
        actual = shard_fold_sample_ids(wds_dir, fold)
        missing = expected - actual
        extra = actual - expected
        if missing or extra:
            raise ValueError(
                f"DALI fold{fold} does not match canonical fold {label}: "
                f"missing={len(missing)}, extra={len(extra)}"
            )
        actual_fold_ids[label] = actual
    return actual_fold_ids
