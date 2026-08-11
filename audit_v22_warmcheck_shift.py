#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_metrics(run: Path) -> dict:
    p = run / "metrics.json"
    if not p.exists():
        return {}
    with p.open() as handle:
        return json.load(handle)


def weighted_total_from_metrics(metrics: dict) -> float:
    per = pd.DataFrame(metrics.get("per_dataset", []))
    if per.empty:
        return float("nan")
    return float((per["chi2_like"] * per["n"]).sum() / per["n"].sum())


def mean_pull2(frame: pd.DataFrame, pred_col: str, target_col: str, sigma_col: str) -> float:
    good = (
        np.isfinite(frame[pred_col])
        & np.isfinite(frame[target_col])
        & np.isfinite(frame[sigma_col])
        & (frame[sigma_col] > 0)
    )
    pull = (frame.loc[good, pred_col] - frame.loc[good, target_col]) / frame.loc[good, sigma_col]
    return float(np.mean(pull.to_numpy(float) ** 2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--central-run", default="production_frozen/PRIMARY_ONE_REPLICA/run")
    parser.add_argument("--v22-run", default="outputs/v22_full_backend_warmcheck_s303")
    parser.add_argument("--out", default="v22/outputs/v22_full_backend_warmcheck_audit")
    args = parser.parse_args()

    central_run = Path(args.central_run)
    v22_run = Path(args.v22_run)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for p in [central_run / "predictions.csv", v22_run / "predictions.csv"]:
        if not p.exists():
            raise SystemExit(f"Missing {p}")

    central = pd.read_csv(central_run / "predictions.csv")
    v22 = pd.read_csv(v22_run / "predictions.csv")

    keys = ["dataset", "row_id"]
    missing = [c for c in keys if c not in central.columns or c not in v22.columns]
    if missing:
        raise SystemExit(f"Missing merge keys: {missing}")

    keep_cols = [
        "pred_match_CS",
        "pred_match_CS_raw_before_dataset_norm",
        "dataset_norm_factor",
        "CS",
        "sigma_used",
        "sigma_uncorr",
        "target_used",
    ]
    keep_cols = [c for c in keep_cols if c in central.columns and c in v22.columns]

    merged = central[keys + keep_cols].merge(
        v22[keys + keep_cols],
        on=keys,
        suffixes=("_central", "_v22"),
        validate="one_to_one",
    )

    for base in ["pred_match_CS", "pred_match_CS_raw_before_dataset_norm"]:
        c0 = f"{base}_central"
        c1 = f"{base}_v22"
        if c0 in merged.columns and c1 in merged.columns:
            merged[f"{base}_ratio_v22_over_central"] = (
                merged[c1] / merged[c0].replace(0, np.nan)
            )
            sigma = "sigma_used_central" if "sigma_used_central" in merged.columns else "sigma_uncorr_central"
            merged[f"{base}_delta_pull_units"] = (
                merged[c1] - merged[c0]
            ) / merged[sigma].replace(0, np.nan)

    if "pred_match_CS_v22" in merged.columns and "CS_v22" in merged.columns:
        sigma = "sigma_used_v22" if "sigma_used_v22" in merged.columns else "sigma_uncorr_v22"
        merged["v22_pull_to_data"] = (
            merged["pred_match_CS_v22"] - merged["CS_v22"]
        ) / merged[sigma].replace(0, np.nan)

    if "pred_match_CS_central" in merged.columns and "CS_central" in merged.columns:
        sigma = "sigma_used_central" if "sigma_used_central" in merged.columns else "sigma_uncorr_central"
        merged["central_pull_to_data"] = (
            merged["pred_match_CS_central"] - merged["CS_central"]
        ) / merged[sigma].replace(0, np.nan)

    central_metrics = load_metrics(central_run)
    v22_metrics = load_metrics(v22_run)

    dataset_rows = []
    for dataset, group in merged.groupby("dataset", observed=False):
        row = {"dataset": dataset, "n": int(len(group))}
        if "pred_match_CS_ratio_v22_over_central" in group:
            r = group["pred_match_CS_ratio_v22_over_central"].to_numpy(float)
            row.update({
                "prediction_ratio_median": float(np.nanmedian(r)),
                "prediction_ratio_min": float(np.nanmin(r)),
                "prediction_ratio_max": float(np.nanmax(r)),
            })
        if "pred_match_CS_delta_pull_units" in group:
            d = group["pred_match_CS_delta_pull_units"].to_numpy(float)
            row.update({
                "delta_pull_median": float(np.nanmedian(d)),
                "delta_pull_abs_p90": float(np.nanquantile(np.abs(d), 0.90)),
                "delta_pull_abs_max": float(np.nanmax(np.abs(d))),
            })
        if "central_pull_to_data" in group:
            row["central_chi2_from_predictions"] = float(np.nanmean(group["central_pull_to_data"] ** 2))
        if "v22_pull_to_data" in group:
            row["v22_chi2_from_predictions"] = float(np.nanmean(group["v22_pull_to_data"] ** 2))
        dataset_rows.append(row)

    by_dataset = pd.DataFrame(dataset_rows)

    summary = {
        "central_run": str(central_run),
        "v22_run": str(v22_run),
        "n_rows_merged": int(len(merged)),
        "central_metrics_chi2_total": weighted_total_from_metrics(central_metrics),
        "v22_metrics_chi2_total": weighted_total_from_metrics(v22_metrics),
    }

    if "pred_match_CS_ratio_v22_over_central" in merged.columns:
        ratio = merged["pred_match_CS_ratio_v22_over_central"].to_numpy(float)
        summary.update({
            "prediction_ratio_median": float(np.nanmedian(ratio)),
            "prediction_ratio_p10": float(np.nanquantile(ratio, 0.10)),
            "prediction_ratio_p90": float(np.nanquantile(ratio, 0.90)),
            "prediction_ratio_min": float(np.nanmin(ratio)),
            "prediction_ratio_max": float(np.nanmax(ratio)),
        })

    if "pred_match_CS_delta_pull_units" in merged.columns:
        delta = merged["pred_match_CS_delta_pull_units"].to_numpy(float)
        summary.update({
            "delta_pull_abs_median": float(np.nanmedian(np.abs(delta))),
            "delta_pull_abs_p90": float(np.nanquantile(np.abs(delta), 0.90)),
            "delta_pull_abs_max": float(np.nanmax(np.abs(delta))),
        })

    summary["next_action_recommendation"] = (
        "Run a v22 central refit if v22 chi2 is materially worse than the frozen central fit; "
        "this warm check is expected to overshoot because the old F_NP was fitted to the legacy perturbative W."
    )

    merged.to_csv(out / "warmcheck_prediction_comparison.csv", index=False)
    by_dataset.to_csv(out / "warmcheck_by_dataset.csv", index=False)
    (out / "warmcheck_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("\n=== v22 full backend warm-check summary ===")
    print(json.dumps(summary, indent=2))

    print("\n=== By dataset ===")
    print(by_dataset.to_string(index=False))

    print("\nwrote:", out)


if __name__ == "__main__":
    main()
