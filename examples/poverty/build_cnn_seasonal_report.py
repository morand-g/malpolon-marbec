#!/usr/bin/env python3

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(
    "/lustre/fswork/projects/rech/cvn/udm31lu/"
    "malpolon-marbec/examples/poverty/"
    "outputs/Madagascar_Seasonal_Canonical"
)

OUT_OOF = ROOT / "cnn_seasonal_oof_enriched.csv"
OUT_REPORT = ROOT / "cnn_seasonal_benchmark_report.csv"

MODEL_NAME = "CNN seasonal"
N_FOLDS = 5
EXPECTED_N = 2290


# ============================================================
# METRICS
# ============================================================

def compute_metrics(df):
    y = df["target"].to_numpy(dtype=float)
    pred = df["prediction"].to_numpy(dtype=float)

    r = np.corrcoef(y, pred)[0, 1]

    return {
        "n": len(df),
        "r2": r2_score(y, pred),
        "pearson_r": r,
        "pearson_r2": r ** 2,
        "rmse": np.sqrt(mean_squared_error(y, pred)),
        "mae": mean_absolute_error(y, pred),
    }


# ============================================================
# LOAD OOF
# ============================================================

def load_oof():

    parts = []

    for fold in range(N_FOLDS):

        path = (
            ROOT
            / f"fold_{fold}"
            / f"predictions_test_dataset_{fold}.csv"
        )

        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_csv(path)

        required = {
            "sample_id",
            "fold",
            "target",
            "prediction",
        }

        missing = required - set(df.columns)

        if missing:
            raise RuntimeError(
                f"{path}: missing columns {missing}. "
                f"Available: {df.columns.tolist()}"
            )

        # Make sure this really is the requested fold
        observed_folds = sorted(df["fold"].unique())

        if observed_folds != [fold]:
            raise RuntimeError(
                f"{path}: expected fold {fold}, "
                f"found {observed_folds}"
            )

        print(
            f"fold {fold}: "
            f"{len(df)} predictions"
        )

        parts.append(
            df[
                [
                    "sample_id",
                    "fold",
                    "target",
                    "prediction",
                ]
            ].copy()
        )

    oof = pd.concat(parts, ignore_index=True)

    # --------------------------------------------------------
    # Checks
    # --------------------------------------------------------

    print("\nTotal OOF:", len(oof))

    if len(oof) != EXPECTED_N:
        raise RuntimeError(
            f"Expected {EXPECTED_N} OOF predictions, "
            f"found {len(oof)}"
        )

    duplicates = oof["sample_id"].duplicated().sum()

    if duplicates:
        raise RuntimeError(
            f"Found {duplicates} duplicated sample_id"
        )

    if oof["target"].isna().any():
        raise RuntimeError("NaN in target")

    if oof["prediction"].isna().any():
        raise RuntimeError("NaN in prediction")

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    # sample_id format:
    # madagascar_1997_123

    oof["survey_year"] = (
        oof["sample_id"]
        .astype(str)
        .str.split("_")
        .str[1]
        .astype(int)
    )

    # Keep same convention as Tempo-V report
    oof["sensor"] = "landsat"

    return oof


# ============================================================
# REPORT
# ============================================================

def add_row(
    rows,
    df,
    aggregation,
    group=None,
    group_value=None,
):

    rows.append(
        {
            "model": MODEL_NAME,
            **compute_metrics(df),
            "aggregation": aggregation,
            "group": group,
            "group_value": group_value,
        }
    )


def build_report(oof):

    rows = []

    # ---------------- pooled OOF ----------------

    add_row(
        rows,
        oof,
        aggregation="pooled_oof",
    )

    # ---------------- folds ----------------

    for fold, g in oof.groupby("fold"):

        add_row(
            rows,
            g,
            aggregation="group",
            group="fold",
            group_value=fold,
        )

    # ---------------- survey years ----------------

    for year, g in oof.groupby("survey_year"):

        add_row(
            rows,
            g,
            aggregation="group",
            group="survey_year",
            group_value=year,
        )

    # ---------------- sensor ----------------

    for sensor, g in oof.groupby("sensor"):

        add_row(
            rows,
            g,
            aggregation="group",
            group="sensor",
            group_value=sensor,
        )

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    print("Building CNN Seasonal benchmark report...\n")

    oof = load_oof()

    # Standardized output order
    oof = oof[
        [
            "sample_id",
            "fold",
            "target",
            "prediction",
            "survey_year",
            "sensor",
        ]
    ]

    oof.to_csv(
        OUT_OOF,
        index=False,
    )

    report = build_report(oof)

    report.to_csv(
        OUT_REPORT,
        index=False,
    )

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print("\n========================================")
    print("POOLED OOF")
    print("========================================")

    pooled = compute_metrics(oof)

    for k, v in pooled.items():

        if isinstance(v, float):
            print(f"{k:12s}: {v:.6f}")
        else:
            print(f"{k:12s}: {v}")

    print("\n========================================")
    print("BY FOLD")
    print("========================================")

    for fold, g in oof.groupby("fold"):

        m = compute_metrics(g)

        print(
            f"fold {fold}: "
            f"n={m['n']:3d} "
            f"R2={m['r2']:.4f} "
            f"r={m['pearson_r']:.4f} "
            f"RMSE={m['rmse']:.3f} "
            f"MAE={m['mae']:.3f}"
        )

    print("\n========================================")
    print("BY YEAR")
    print("========================================")

    for year, g in oof.groupby("survey_year"):

        m = compute_metrics(g)

        print(
            f"{year}: "
            f"n={m['n']:3d} "
            f"R2={m['r2']:.4f} "
            f"r={m['pearson_r']:.4f} "
            f"RMSE={m['rmse']:.3f} "
            f"MAE={m['mae']:.3f}"
        )

    print("\nSaved:")
    print(OUT_OOF)
    print(OUT_REPORT)


if __name__ == "__main__":
    main()