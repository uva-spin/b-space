#!/usr/bin/env python3
"""v23a central refit audit v2.

Changes from the first audit:
  * finite check is restricted to core prediction/data/error columns;
  * all nonfinite numeric columns are still reported separately;
  * norm-source problems are reported but not confused with optional NaN columns.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as handle:
        return json.load(handle)


def first_existing(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def weighted_chi2_from_metrics(metrics: dict[str, Any]) -> float:
    if "chi2_total" in metrics:
        try:
            return float(metrics["chi2_total"])
        except Exception:
            pass
    per = pd.DataFrame(metrics.get("per_dataset", []))
    if per.empty:
        return float("nan")
    return float((per["chi2_like"] * per["n"]).sum() / per["n"].sum())


def per_dataset_metrics(metrics: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(metrics.get("per_dataset", []))


def norm_pulls(metrics: dict[str, Any]) -> dict[str, float]:
    pulls = metrics.get("dataset_norm_pulls", {})
    return {str(k): float(v) for k, v in pulls.items()} if pulls else {}


def prediction_columns(pred: pd.DataFrame):
    pred_col = first_existing(pred, ["pred_match_CS", "prediction", "pred", "matched_CS", "W_plus_Y"])
    data_col = first_existing(pred, ["CS", "target", "data", "target_CS"])
    sigma_col = first_existing(pred, ["sigma_used", "error", "sigma_uncorr", "unc"])
    return pred_col, data_col, sigma_col


def nonfinite_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        n_bad = int((~np.isfinite(vals)).sum())
        if n_bad:
            rows.append({
                "column": col,
                "n_nonfinite": n_bad,
                "fraction_nonfinite": n_bad / max(len(df), 1),
            })
    return pd.DataFrame(rows).sort_values("n_nonfinite", ascending=False) if rows else pd.DataFrame()


def prediction_summary(pred: pd.DataFrame) -> pd.DataFrame:
    pred_col, data_col, sigma_col = prediction_columns(pred)
    if pred_col is None or data_col is None:
        return pd.DataFrame()
    work = pred.copy()
    work["pred_over_data"] = work[pred_col] / work[data_col].replace(0.0, np.nan)
    if sigma_col is not None:
        work["pull"] = (work[pred_col] - work[data_col]) / work[sigma_col].replace(0.0, np.nan)
    else:
        work["pull"] = np.nan
    return (
        work.groupby("dataset", observed=False)
        .agg(
            n=("row_id", "size"),
            pred_over_data_median=("pred_over_data", "median"),
            pred_over_data_p10=("pred_over_data", lambda s: float(np.nanquantile(s, 0.10))),
            pred_over_data_p90=("pred_over_data", lambda s: float(np.nanquantile(s, 0.90))),
            abs_pull_median=("pull", lambda s: float(np.nanmedian(np.abs(s)))),
            abs_pull_p90=("pull", lambda s: float(np.nanquantile(np.abs(s), 0.90))),
            abs_pull_max=("pull", lambda s: float(np.nanmax(np.abs(s)))),
        )
        .reset_index()
    )


def compare_common_predictions(v23: pd.DataFrame, v22: pd.DataFrame) -> pd.DataFrame:
    if v23.empty or v22.empty:
        return pd.DataFrame()
    pred23, _, sigma_col = prediction_columns(v23)
    pred22, _, _ = prediction_columns(v22)
    if pred23 is None or pred22 is None:
        return pd.DataFrame()
    keep = ["dataset", "row_id", pred23]
    if sigma_col:
        keep.append(sigma_col)
    merged = v23[keep].merge(
        v22[["dataset", "row_id", pred22]],
        on=["dataset", "row_id"],
        how="inner",
        suffixes=("_v23", "_v22"),
    )
    if merged.empty:
        return pd.DataFrame()
    p23 = f"{pred23}_v23" if f"{pred23}_v23" in merged.columns else pred23
    p22 = f"{pred22}_v22" if f"{pred22}_v22" in merged.columns else pred22
    merged["v23_over_v22_prediction"] = merged[p23] / merged[p22].replace(0.0, np.nan)
    merged["rel_delta_prediction"] = np.abs(merged[p23] - merged[p22]) / np.maximum(
        np.maximum(np.abs(merged[p23]), np.abs(merged[p22])),
        1.0e-300,
    )
    if sigma_col and sigma_col in merged.columns:
        merged["delta_pull_units_v23_sigma"] = (merged[p23] - merged[p22]) / merged[sigma_col].replace(0.0, np.nan)
    else:
        merged["delta_pull_units_v23_sigma"] = np.nan
    return (
        merged.groupby("dataset", observed=False)
        .agg(
            n_common=("row_id", "size"),
            v23_over_v22_prediction_median=("v23_over_v22_prediction", "median"),
            rel_delta_prediction_p90=("rel_delta_prediction", lambda s: float(np.nanquantile(s, 0.90))),
            abs_delta_pull_p90=("delta_pull_units_v23_sigma", lambda s: float(np.nanquantile(np.abs(s), 0.90))),
            abs_delta_pull_max=("delta_pull_units_v23_sigma", lambda s: float(np.nanmax(np.abs(s)))),
        )
        .reset_index()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--v22-run", default="outputs/v22_full_backend_central_refit_stage1_s303")
    parser.add_argument("--out", default="v23/outputs/v23a_central_refit_audit_v2_s303")
    parser.add_argument("--max-chi2", type=float, default=1.5)
    parser.add_argument("--max-dataset-chi2", type=float, default=2.5)
    parser.add_argument("--max-norm-pull", type=float, default=4.0)
    args = parser.parse_args()

    run = Path(args.run)
    v22_run = Path(args.v22_run)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    metrics = load_json(run / "metrics.json")
    v22_metrics = load_json(v22_run / "metrics.json")
    pred = pd.read_csv(run / "predictions.csv")
    v22_pred = pd.read_csv(v22_run / "predictions.csv") if (v22_run / "predictions.csv").exists() else pd.DataFrame()

    per = per_dataset_metrics(metrics)
    pred_sum = prediction_summary(pred)
    common = compare_common_predictions(pred, v22_pred)
    nonfinite = nonfinite_numeric_columns(pred)

    pred_col, data_col, sigma_col = prediction_columns(pred)
    core_cols = [c for c in [pred_col, data_col, sigma_col] if c is not None]
    core_finite = bool(core_cols and all(np.isfinite(pd.to_numeric(pred[c], errors="coerce")).all() for c in core_cols))

    pulls = norm_pulls(metrics)
    max_norm_pull = max((abs(v) for v in pulls.values()), default=float("nan"))
    chi2_total = weighted_chi2_from_metrics(metrics)
    v22_chi2_total = weighted_chi2_from_metrics(v22_metrics)

    per_dataset_max = float(per["chi2_like"].max()) if not per.empty and "chi2_like" in per else float("nan")
    datasets = sorted(str(x) for x in pred["dataset"].unique()) if "dataset" in pred.columns else []
    required = {"E288_200", "E288_300", "E288_400", "E605", "E772"}

    fit_pass = bool(np.isfinite(chi2_total) and chi2_total < args.max_chi2)
    dataset_pass = bool((not np.isfinite(per_dataset_max)) or per_dataset_max < args.max_dataset_chi2)
    norm_pass = bool((not np.isfinite(max_norm_pull)) or max_norm_pull < args.max_norm_pull)
    has_required = required.issubset(set(datasets))
    pass_status = bool(fit_pass and dataset_pass and norm_pass and core_finite and has_required)

    summary = {
        "run": str(run),
        "v22_reference_run": str(v22_run),
        "n_prediction_rows": int(len(pred)),
        "datasets": datasets,
        "has_required_v23a_datasets": has_required,
        "v23a_chi2_total": chi2_total,
        "v22_chi2_total_reference": v22_chi2_total,
        "v23a_max_dataset_chi2": per_dataset_max if np.isfinite(per_dataset_max) else None,
        "v23a_dataset_norm_pulls": pulls,
        "v23a_max_abs_norm_pull": max_norm_pull if np.isfinite(max_norm_pull) else None,
        "prediction_column": pred_col,
        "data_column": data_col,
        "sigma_column": sigma_col,
        "core_prediction_columns_finite": core_finite,
        "n_numeric_columns_with_nonfinite_values": int(len(nonfinite)),
        "fit_pass": fit_pass,
        "dataset_chi2_pass": dataset_pass,
        "norm_pass": norm_pass,
        "V23A_CENTRAL_REFIT_BASIC_PASS": pass_status,
        "thresholds": {
            "max_chi2": args.max_chi2,
            "max_dataset_chi2": args.max_dataset_chi2,
            "max_norm_pull": args.max_norm_pull,
        },
    }

    (out / "v23a_central_refit_audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    per.to_csv(out / "v23a_per_dataset_metrics.csv", index=False)
    pred_sum.to_csv(out / "v23a_prediction_summary_by_dataset.csv", index=False)
    common.to_csv(out / "v23a_vs_v22_common_prediction_shift.csv", index=False)
    nonfinite.to_csv(out / "v23a_prediction_nonfinite_numeric_columns.csv", index=False)

    print("\n=== v23a central refit audit v2 summary ===")
    print(json.dumps(summary, indent=2))

    print("\n=== Per-dataset metrics ===")
    print(per.to_string(index=False) if not per.empty else "(none)")
    print("\n=== Prediction summary by dataset ===")
    print(pred_sum.to_string(index=False) if not pred_sum.empty else "(none)")
    print("\n=== Nonfinite numeric columns ===")
    print(nonfinite.to_string(index=False) if not nonfinite.empty else "(none)")
    print("\n=== v23a vs v22 common prediction shift ===")
    print(common.to_string(index=False) if not common.empty else "(none)")
    print("\nwrote:", out)

    if not pass_status:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
