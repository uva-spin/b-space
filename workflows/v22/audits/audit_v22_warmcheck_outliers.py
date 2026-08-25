#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def first_existing(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.replace(0.0, np.nan)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--central-run",
        default=(
            "/home/dustin/work/bT-TMD/v21_tail_release_amp0p019_candidate/"
            "production_frozen/PRIMARY_ONE_REPLICA/run"
        ),
    )
    parser.add_argument(
        "--v22-run",
        default="outputs/v22_full_backend_warmcheck_s303",
    )
    parser.add_argument(
        "--warmcheck-audit",
        default="v22/outputs/v22_full_backend_warmcheck_audit",
    )
    parser.add_argument(
        "--out",
        default="v22/outputs/v22_full_backend_warmcheck_outliers",
    )
    parser.add_argument("--top-n", type=int, default=40)
    args = parser.parse_args()

    central_run = Path(args.central_run)
    v22_run = Path(args.v22_run)
    warmcheck = Path(args.warmcheck_audit)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    central_path = central_run / "predictions.csv"
    v22_path = v22_run / "predictions.csv"

    for path in [central_path, v22_path]:
        if not path.exists():
            raise SystemExit(f"Missing {path}")

    central = pd.read_csv(central_path)
    v22 = pd.read_csv(v22_path)

    keys = ["dataset", "row_id"]
    merged = central.merge(
        v22,
        on=keys,
        how="inner",
        suffixes=("_central", "_v22"),
        validate="one_to_one",
    )

    c_pred = "pred_match_CS_central"
    v_pred = "pred_match_CS_v22"
    c_raw = "pred_match_CS_raw_before_dataset_norm_central"
    v_raw = "pred_match_CS_raw_before_dataset_norm_v22"

    sigma = first_existing(
        merged,
        ["sigma_used_central", "sigma_uncorr_central", "error_central"],
    )
    data = first_existing(merged, ["CS_central", "target_used_central"])

    if sigma is None or data is None:
        raise SystemExit(
            "Could not find sigma/data columns in merged predictions."
        )

    merged["prediction_ratio_v22_over_central"] = safe_ratio(
        merged[v_pred],
        merged[c_pred],
    )
    if c_raw in merged.columns and v_raw in merged.columns:
        merged["raw_ratio_v22_over_central"] = safe_ratio(
            merged[v_raw],
            merged[c_raw],
        )

    merged["delta_prediction"] = merged[v_pred] - merged[c_pred]
    merged["delta_pull_units"] = merged["delta_prediction"] / merged[sigma]
    merged["abs_delta_pull_units"] = merged["delta_pull_units"].abs()

    merged["central_pull_to_data"] = (
        merged[c_pred] - merged[data]
    ) / merged[sigma]
    merged["v22_pull_to_data"] = (
        merged[v_pred] - merged[data]
    ) / merged[sigma]

    q_col = first_existing(merged, ["QM_central", "Q_central"])
    qt_col = first_existing(merged, ["qT_central"])
    if q_col and qt_col:
        merged["qT_over_Q"] = merged[qt_col] / merged[q_col]
    else:
        merged["qT_over_Q"] = np.nan

    denom_scale = np.maximum(
        merged[c_pred].abs().to_numpy(float),
        merged[sigma].abs().to_numpy(float),
    )
    merged["central_pred_over_sigma"] = (
        merged[c_pred].abs().to_numpy(float)
        / merged[sigma].abs().replace(0.0, np.nan).to_numpy(float)
    )
    merged["ratio_is_unstable"] = (
        merged[c_pred].abs()
        < 0.1 * merged[sigma].abs()
    )
    merged["v22_negative_prediction"] = merged[v_pred] < 0.0
    merged["central_negative_prediction"] = merged[c_pred] < 0.0
    merged["large_shape_shift"] = merged["abs_delta_pull_units"] > 3.0

    top_delta = merged.sort_values(
        "abs_delta_pull_units",
        ascending=False,
    ).head(args.top_n)

    ratio_extreme = merged[
        np.isfinite(merged["prediction_ratio_v22_over_central"])
    ].copy()
    ratio_extreme["abs_log_ratio"] = np.abs(
        np.log(
            np.abs(
                ratio_extreme["prediction_ratio_v22_over_central"]
            ).replace(0.0, np.nan)
        )
    )
    ratio_extreme = ratio_extreme.sort_values(
        "abs_log_ratio",
        ascending=False,
    ).head(args.top_n)

    negative = merged[
        merged["v22_negative_prediction"]
        | merged["central_negative_prediction"]
        | (merged["prediction_ratio_v22_over_central"] < 0.0)
    ].copy().sort_values("abs_delta_pull_units", ascending=False)

    dataset = (
        merged.groupby("dataset", observed=False)
        .agg(
            n=("row_id", "size"),
            chi2_central=("central_pull_to_data", lambda x: float(np.mean(np.asarray(x) ** 2))),
            chi2_v22=("v22_pull_to_data", lambda x: float(np.mean(np.asarray(x) ** 2))),
            ratio_median=("prediction_ratio_v22_over_central", "median"),
            ratio_p10=("prediction_ratio_v22_over_central", lambda x: float(np.nanquantile(x, 0.10))),
            ratio_p90=("prediction_ratio_v22_over_central", lambda x: float(np.nanquantile(x, 0.90))),
            abs_delta_pull_median=("abs_delta_pull_units", "median"),
            abs_delta_pull_p90=("abs_delta_pull_units", lambda x: float(np.nanquantile(x, 0.90))),
            abs_delta_pull_max=("abs_delta_pull_units", "max"),
            n_ratio_unstable=("ratio_is_unstable", "sum"),
            n_v22_negative=("v22_negative_prediction", "sum"),
            n_large_shape_shift=("large_shape_shift", "sum"),
        )
        .reset_index()
    )

    if q_col and qt_col:
        bins = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
        labels = ["0-0.05", "0.05-0.10", "0.10-0.15", "0.15-0.20", "0.20-0.30", "0.30-0.50"]
        merged["qT_over_Q_bin"] = pd.cut(
            merged["qT_over_Q"],
            bins=bins,
            labels=labels,
            include_lowest=True,
        )
        by_bin = (
            merged.groupby(["dataset", "qT_over_Q_bin"], observed=False)
            .agg(
                n=("row_id", "size"),
                ratio_median=("prediction_ratio_v22_over_central", "median"),
                abs_delta_pull_p90=("abs_delta_pull_units", lambda x: float(np.nanquantile(x, 0.90)) if len(x) else np.nan),
                n_v22_negative=("v22_negative_prediction", "sum"),
                n_ratio_unstable=("ratio_is_unstable", "sum"),
            )
            .reset_index()
        )
    else:
        by_bin = pd.DataFrame()

    output_columns = [
        "dataset",
        "row_id",
    ]
    for col in [q_col, qt_col, "qT_over_Q", data, sigma, c_pred, v_pred,
                "prediction_ratio_v22_over_central", "delta_pull_units",
                "central_pull_to_data", "v22_pull_to_data",
                "central_pred_over_sigma",
                "ratio_is_unstable",
                "central_negative_prediction",
                "v22_negative_prediction"]:
        if col and col in merged.columns and col not in output_columns:
            output_columns.append(col)

    merged.to_csv(out / "warmcheck_outlier_all_rows.csv", index=False)
    top_delta[output_columns].to_csv(out / "top_delta_pull_rows.csv", index=False)
    ratio_extreme[output_columns].to_csv(out / "extreme_ratio_rows.csv", index=False)
    negative[output_columns].to_csv(out / "negative_or_signflip_rows.csv", index=False)
    dataset.to_csv(out / "warmcheck_outlier_by_dataset.csv", index=False)
    if not by_bin.empty:
        by_bin.to_csv(out / "warmcheck_outlier_by_qToQ_bin.csv", index=False)

    pathology_suspected = bool(
        (merged["v22_negative_prediction"].sum() > 0)
        or (merged["abs_delta_pull_units"].max() > 10.0)
        or (
            (merged["ratio_is_unstable"].mean() > 0.10)
            and (merged["large_shape_shift"].mean() > 0.10)
        )
    )

    refit_recommended = bool(
        not pathology_suspected
        and (dataset["chi2_v22"].mean() > 1.2)
    )

    summary = {
        "n_rows": int(len(merged)),
        "n_v22_negative_predictions": int(merged["v22_negative_prediction"].sum()),
        "n_central_negative_predictions": int(merged["central_negative_prediction"].sum()),
        "n_ratio_unstable": int(merged["ratio_is_unstable"].sum()),
        "max_abs_delta_pull": float(merged["abs_delta_pull_units"].max()),
        "median_abs_delta_pull": float(merged["abs_delta_pull_units"].median()),
        "prediction_ratio_median": float(np.nanmedian(merged["prediction_ratio_v22_over_central"])),
        "prediction_ratio_p10": float(np.nanquantile(merged["prediction_ratio_v22_over_central"], 0.10)),
        "prediction_ratio_p90": float(np.nanquantile(merged["prediction_ratio_v22_over_central"], 0.90)),
        "pathology_suspected_before_refit": pathology_suspected,
        "refit_recommended_if_false_pathology": refit_recommended,
        "interpretation": (
            "Sign flips or large ratios are often caused by very small central predictions; "
            "inspect negative_or_signflip_rows.csv before fitting. If these are tail/near-zero artifacts "
            "rather than broad instability, proceed to the central v22 refit."
        ),
    }
    (out / "warmcheck_outlier_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("\n=== Warm-check outlier summary ===")
    print(json.dumps(summary, indent=2))
    print("\n=== By dataset ===")
    print(dataset.to_string(index=False))
    print("\n=== Top absolute prediction shifts ===")
    print(top_delta[output_columns].head(12).to_string(index=False))
    if not negative.empty:
        print("\n=== Negative/sign-flip rows ===")
        print(negative[output_columns].head(20).to_string(index=False))
    else:
        print("\nNo negative/sign-flip rows.")
    print("\nwrote:", out)


if __name__ == "__main__":
    main()
