#!/usr/bin/env python3
"""Audit v23a replica fit outliers by seed, dataset, and row.

This is designed for the v23a lambda=3 normpriors15 replica pilot after the
20-rep q95 gate fails in fit_distribution_pass while norm and b-space bands
are technically fine.

It does not modify any run.  It writes:
  replica_fit_summary.csv
  per_dataset_by_seed.csv
  per_dataset_distribution_summary.csv
  high_chi2_replica_top_rows.csv
  split_first_second_summary.csv
  v23a_replica_fit_outlier_summary.json

Typical use:
  PYTHONPATH=. python3 v23/tools/audit_v23a_replica_fit_outliers.py \
    --run-glob 'replica_pilot_v23a_lambda3_normpriors15_cached_cuda/outputs/v23a_lambda3_normpriors15_cached_cuda_s*' \
    --central-run outputs/v23a_fixed_target_lowQ_corrected_central_refit_normpriors_15pct_s303 \
    --band-audit replica_pilot_v23a_lambda3_normpriors15_cached_cuda/tmd_bspace_bands_exactx_20rep/audit/bspace_band_audit_summary.json \
    --out v23/outputs/v23a_lambda3_normpriors15_20rep_fit_outliers
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as handle:
        return json.load(handle)


def seed_from_path(path: Path) -> str:
    m = re.search(r"_s(\d+)(?:/)?$", str(path))
    if m:
        return m.group(1)
    m = re.search(r"s(\d+)", path.name)
    return m.group(1) if m else path.name


def first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def chi2_total_from_metrics(metrics: dict[str, Any]) -> float:
    if "chi2_total" in metrics:
        try:
            return float(metrics["chi2_total"])
        except Exception:
            pass
    per = pd.DataFrame(metrics.get("per_dataset", []))
    if not per.empty and {"chi2_like", "n"}.issubset(per.columns):
        return float((per["chi2_like"] * per["n"]).sum() / per["n"].sum())
    return float("nan")


def per_dataset_from_metrics(metrics: dict[str, Any]) -> pd.DataFrame:
    per = pd.DataFrame(metrics.get("per_dataset", []))
    return per


def norm_pulls_from_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    pulls = metrics.get("dataset_norm_pulls", {})
    return {str(k): float(v) for k, v in pulls.items()} if pulls else {}


def prediction_row_summary(run: Path, seed: str, top_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_path = run / "predictions.csv"
    if not pred_path.exists():
        return pd.DataFrame(), pd.DataFrame()

    pred = pd.read_csv(pred_path)
    if "dataset" not in pred.columns or "row_id" not in pred.columns:
        return pd.DataFrame(), pd.DataFrame()

    pred_col = first_existing(pred, ["pred_match_CS", "prediction", "pred", "matched_CS", "W_plus_Y"])
    data_col = first_existing(pred, ["CS", "target", "data", "target_CS"])
    sigma_col = first_existing(pred, ["sigma_used", "error", "sigma_uncorr", "unc"])

    if pred_col is None or data_col is None or sigma_col is None:
        return pd.DataFrame(), pd.DataFrame()

    work = pred.copy()
    for c in [pred_col, data_col, sigma_col]:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    work["pull"] = (work[pred_col] - work[data_col]) / work[sigma_col].replace(0.0, np.nan)
    work["abs_pull"] = np.abs(work["pull"])
    work["pred_over_data"] = work[pred_col] / work[data_col].replace(0.0, np.nan)
    work["seed"] = seed
    work["run"] = str(run)

    # Compact top rows.
    keep_cols = [
        "seed", "run", "dataset", "row_id",
        "qT", "QM", "qT_over_Q", "x1", "x2",
        data_col, pred_col, sigma_col,
        "pull", "abs_pull", "pred_over_data",
    ]
    keep_cols = [c for c in keep_cols if c in work.columns]
    top_rows = work.sort_values("abs_pull", ascending=False).head(top_n)[keep_cols].copy()

    by_dataset = (
        work.groupby("dataset", observed=False)
        .agg(
            n_pred=("row_id", "size"),
            chi2_from_predictions=("pull", lambda s: float(np.nanmean(np.square(s)))),
            abs_pull_median=("abs_pull", "median"),
            abs_pull_p90=("abs_pull", lambda s: float(np.nanquantile(s, 0.90))),
            abs_pull_max=("abs_pull", "max"),
            pred_over_data_median=("pred_over_data", "median"),
            pred_over_data_p10=("pred_over_data", lambda s: float(np.nanquantile(s, 0.10))),
            pred_over_data_p90=("pred_over_data", lambda s: float(np.nanquantile(s, 0.90))),
        )
        .reset_index()
    )
    by_dataset["seed"] = seed
    by_dataset["run"] = str(run)

    return by_dataset, top_rows


def summarize_distribution(per_dataset: pd.DataFrame) -> pd.DataFrame:
    if per_dataset.empty:
        return pd.DataFrame()

    metric_cols = [
        c for c in [
            "chi2_like",
            "chi2_from_predictions",
            "median_abs_pull",
            "abs_pull_median",
            "abs_pull_p90",
            "abs_pull_max",
            "median_pred_over_target",
            "pred_over_data_median",
        ]
        if c in per_dataset.columns
    ]

    rows = []
    for dataset, group in per_dataset.groupby("dataset", observed=False):
        row = {
            "dataset": dataset,
            "n_seeds": int(group["seed"].nunique()),
        }
        for c in metric_cols:
            vals = pd.to_numeric(group[c], errors="coerce")
            vals = vals[np.isfinite(vals)]
            if len(vals):
                row[f"{c}_median"] = float(vals.median())
                row[f"{c}_q90"] = float(vals.quantile(0.90))
                row[f"{c}_q95"] = float(vals.quantile(0.95))
                row[f"{c}_max"] = float(vals.max())
                # Which seed was the max?
                imax = pd.to_numeric(group[c], errors="coerce").idxmax()
                row[f"{c}_max_seed"] = str(group.loc[imax, "seed"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        [c for c in ["chi2_like_q95", "chi2_from_predictions_q95"] if c in pd.DataFrame(rows).columns],
        ascending=False,
    ) if rows else pd.DataFrame()


def split_summary(replica: pd.DataFrame, per_dataset: pd.DataFrame) -> pd.DataFrame:
    if replica.empty:
        return pd.DataFrame()

    ordered = replica.sort_values("seed_numeric").copy()
    n = len(ordered)
    first = set(ordered.head(n // 2)["seed"].astype(str))
    second = set(ordered.tail(n - n // 2)["seed"].astype(str))

    rows = []
    for label, seeds in [("first_half", first), ("second_half", second)]:
        sub = ordered[ordered["seed"].astype(str).isin(seeds)]
        row = {
            "split": label,
            "n": int(len(sub)),
            "seeds": ",".join(sub["seed"].astype(str).tolist()),
            "chi2_median": float(sub["chi2_total"].median()),
            "chi2_q95": float(sub["chi2_total"].quantile(0.95)),
            "chi2_max": float(sub["chi2_total"].max()),
            "norm_pull_q95": float(sub["max_abs_norm_pull"].quantile(0.95)),
            "norm_pull_max": float(sub["max_abs_norm_pull"].max()),
        }
        rows.append(row)

    # Dataset split q95s.
    if not per_dataset.empty and "chi2_like" in per_dataset.columns:
        for dataset, group in per_dataset.groupby("dataset", observed=False):
            for label, seeds in [("first_half", first), ("second_half", second)]:
                sub = group[group["seed"].astype(str).isin(seeds)]
                if sub.empty:
                    continue
                rows.append({
                    "split": f"{label}:{dataset}",
                    "n": int(len(sub)),
                    "seeds": ",".join(sub["seed"].astype(str).tolist()),
                    "chi2_median": float(sub["chi2_like"].median()),
                    "chi2_q95": float(sub["chi2_like"].quantile(0.95)),
                    "chi2_max": float(sub["chi2_like"].max()),
                    "norm_pull_q95": np.nan,
                    "norm_pull_max": np.nan,
                })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-glob", required=True)
    parser.add_argument("--central-run", default="")
    parser.add_argument("--band-audit", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--chi2-q95-max", type=float, default=2.2)
    parser.add_argument("--norm-q95-max", type=float, default=4.0)
    parser.add_argument("--top-rows-per-run", type=int, default=20)
    args = parser.parse_args()

    runs = sorted(Path(p) for p in glob.glob(args.run_glob))
    if not runs:
        raise SystemExit(f"No runs matched: {args.run_glob}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    replica_rows = []
    per_dataset_rows = []
    top_row_pieces = []
    prediction_dataset_rows = []

    for run in runs:
        seed = seed_from_path(run)
        seed_numeric = int(seed) if str(seed).isdigit() else np.nan
        metrics = load_json(run / "metrics.json")

        chi2 = chi2_total_from_metrics(metrics)
        per = per_dataset_from_metrics(metrics)
        pulls = norm_pulls_from_metrics(metrics)
        max_abs_norm = max((abs(v) for v in pulls.values()), default=np.nan)

        row = {
            "seed": seed,
            "seed_numeric": seed_numeric,
            "run": str(run),
            "chi2_total": chi2,
            "max_abs_norm_pull": max_abs_norm,
            "best_epoch": metrics.get("best_epoch", np.nan),
            "epochs_run": metrics.get("epochs_run", np.nan),
            "restored_best": metrics.get("restored_best", np.nan),
        }
        for dataset, pull in pulls.items():
            row[f"norm_pull_{dataset}"] = pull
        replica_rows.append(row)

        if not per.empty:
            per = per.copy()
            per["seed"] = seed
            per["run"] = str(run)
            per_dataset_rows.append(per)

        pred_ds, top = prediction_row_summary(run, seed, args.top_rows_per_run)
        if not pred_ds.empty:
            prediction_dataset_rows.append(pred_ds)
        if not top.empty:
            top_row_pieces.append(top)

    replica = pd.DataFrame(replica_rows).sort_values("seed_numeric")
    per_dataset = pd.concat(per_dataset_rows, ignore_index=True) if per_dataset_rows else pd.DataFrame()
    pred_dataset = pd.concat(prediction_dataset_rows, ignore_index=True) if prediction_dataset_rows else pd.DataFrame()
    top_rows = pd.concat(top_row_pieces, ignore_index=True) if top_row_pieces else pd.DataFrame()

    # Merge prediction-derived dataset summaries with metric-derived summaries.
    if not per_dataset.empty and not pred_dataset.empty:
        merge_cols = ["seed", "run", "dataset"]
        per_dataset_full = per_dataset.merge(pred_dataset, on=merge_cols, how="outer")
    elif not per_dataset.empty:
        per_dataset_full = per_dataset
    else:
        per_dataset_full = pred_dataset

    dist = summarize_distribution(per_dataset_full)
    splits = split_summary(replica, per_dataset_full)

    chi2_q95 = float(replica["chi2_total"].quantile(0.95))
    norm_q95 = float(replica["max_abs_norm_pull"].quantile(0.95))
    high_mask = replica["chi2_total"] >= chi2_q95
    high_replicas = replica[high_mask].copy()

    band_summary = load_json(Path(args.band_audit)) if args.band_audit else {}
    central_metrics = load_json(Path(args.central_run) / "metrics.json") if args.central_run else {}

    recommendation = []
    if norm_q95 > args.norm_q95_max:
        recommendation.append("Normalization q95 fails; inspect norm priors before changing the F_NP anchor.")
    if chi2_q95 > args.chi2_q95_max:
        recommendation.append("Fit q95 fails; identify whether high chi2 is concentrated in one dataset or spread across datasets.")
    if band_summary.get("BSPACE_TMD_BAND_TECHNICAL_PASS") and band_summary.get("BSPACE_TMD_BAND_UNCERTAINTY_USEFUL_PASS"):
        recommendation.append("B-space bands are technically useful; do not discard lambda=3 solely on band topology.")
    if chi2_q95 > args.chi2_q95_max and norm_q95 <= args.norm_q95_max:
        recommendation.append("If high chi2 is not a data-provenance problem, run a small weaker-anchor pilot such as lambda_logF=1 before appending to 50.")

    summary = {
        "run_glob": args.run_glob,
        "n_replicas": int(len(replica)),
        "seeds": replica["seed"].astype(str).tolist(),
        "chi2_median": float(replica["chi2_total"].median()),
        "chi2_q95": chi2_q95,
        "chi2_max": float(replica["chi2_total"].max()),
        "norm_pull_q95": norm_q95,
        "norm_pull_max": float(replica["max_abs_norm_pull"].max()),
        "fit_distribution_pass": bool(chi2_q95 <= args.chi2_q95_max),
        "norm_distribution_pass": bool(norm_q95 <= args.norm_q95_max),
        "high_chi2_replicas_at_or_above_q95": high_replicas[["seed", "chi2_total", "max_abs_norm_pull"]].to_dict(orient="records"),
        "central_run": args.central_run,
        "central_chi2_total": chi2_total_from_metrics(central_metrics) if central_metrics else None,
        "band_audit": args.band_audit,
        "band_technical_pass": band_summary.get("BSPACE_TMD_BAND_TECHNICAL_PASS"),
        "band_uncertainty_useful_pass": band_summary.get("BSPACE_TMD_BAND_UNCERTAINTY_USEFUL_PASS"),
        "recommendation": recommendation,
    }

    replica.to_csv(out / "replica_fit_summary.csv", index=False)
    per_dataset_full.to_csv(out / "per_dataset_by_seed.csv", index=False)
    dist.to_csv(out / "per_dataset_distribution_summary.csv", index=False)
    splits.to_csv(out / "split_first_second_summary.csv", index=False)
    if not top_rows.empty:
        top_rows.to_csv(out / "high_chi2_replica_top_rows.csv", index=False)

    (out / "v23a_replica_fit_outlier_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("\n=== v23a replica fit outlier summary ===")
    print(json.dumps(summary, indent=2))

    print("\n=== Replica fit summary ===")
    print(replica.to_string(index=False))

    print("\n=== Per-dataset distribution summary ===")
    print(dist.to_string(index=False) if not dist.empty else "(none)")

    print("\n=== Split first/second summary ===")
    print(splits.to_string(index=False) if not splits.empty else "(none)")

    if not top_rows.empty:
        print("\n=== Top pull rows among replicas ===")
        print(top_rows.sort_values("abs_pull", ascending=False).head(40).to_string(index=False))

    print("\nwrote:", out)


if __name__ == "__main__":
    main()
