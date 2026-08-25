#!/usr/bin/env python3
"""Distribution-based convergence audit for v22 lambda=3 b-space ensemble.

This is intended for 10+ replicas.  Unlike the earlier hard "max chi2" pilot
audit, this uses q95 fit/norm criteria because the maximum naturally grows as
the ensemble grows.  Max values are still reported as outlier diagnostics.

The kT-space transform remains out of scope.
"""

from __future__ import annotations

import argparse
import glob
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


def weighted_chi2(metrics: dict[str, Any]) -> float:
    per = pd.DataFrame(metrics.get("per_dataset", []))
    if per.empty:
        return float("nan")
    return float((per["chi2_like"] * per["n"]).sum() / per["n"].sum())


def max_abs_norm_pull(metrics: dict[str, Any]) -> float:
    pulls = metrics.get("dataset_norm_pulls", {})
    if pulls:
        return float(max(abs(float(v)) for v in pulls.values()))
    return float("nan")


def infer_seed(run: Path, metrics: dict[str, Any]) -> str:
    cfg = metrics.get("config", {})
    for key in ["replica_seed", "seed"]:
        if key in cfg:
            return str(cfg[key])
    for token in run.name.replace("-", "_").split("_"):
        if token.startswith("s") and token[1:].isdigit():
            return token[1:]
    return run.name


def summarize_runs(runs: list[Path]) -> pd.DataFrame:
    rows = []
    for run in runs:
        metrics = load_json(run / "metrics.json")
        seed = infer_seed(run, metrics)
        row = {
            "seed": seed,
            "run": str(run),
            "chi2_total": weighted_chi2(metrics),
            "max_abs_norm_pull": max_abs_norm_pull(metrics),
        }
        train = metrics.get("train", {})
        for key in ["best_epoch", "epochs_run", "best_objective", "restored_best"]:
            if key in train:
                row[key] = train[key]
        per = pd.DataFrame(metrics.get("per_dataset", []))
        if not per.empty:
            for _, ds in per.iterrows():
                dataset = str(ds["dataset"])
                row[f"chi2_{dataset}"] = float(ds["chi2_like"])
                if "median_pred_over_target" in ds:
                    row[f"ratio_{dataset}"] = float(ds["median_pred_over_target"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("seed")


def quantile_band(frame: pd.DataFrame, quantity: str) -> pd.DataFrame:
    return (
        frame.groupby(["pid", "flavor", "x", "Q", "bT"], observed=False)[quantity]
        .agg(
            median="median",
            q16=lambda v: float(np.nanquantile(v, 0.16)),
            q84=lambda v: float(np.nanquantile(v, 0.84)),
        )
        .reset_index()
    )


def split_half_stability(long: pd.DataFrame, quantities: list[str]) -> pd.DataFrame:
    seeds = sorted(str(seed) for seed in long["seed"].unique())
    if len(seeds) < 4:
        raise ValueError("Need at least four replicas for split-half audit.")

    half = len(seeds) // 2
    first = set(seeds[:half])
    second = set(seeds[half:])

    rows = []
    for quantity in quantities:
        a = quantile_band(long[long["seed"].astype(str).isin(first)], quantity)
        b = quantile_band(long[long["seed"].astype(str).isin(second)], quantity)
        merged = a.merge(
            b,
            on=["pid", "flavor", "x", "Q", "bT"],
            suffixes=("_first", "_second"),
            validate="one_to_one",
        )
        merged["halfwidth_first"] = 0.5 * (merged["q84_first"] - merged["q16_first"])
        merged["halfwidth_second"] = 0.5 * (merged["q84_second"] - merged["q16_second"])
        merged["median_mean_abs"] = 0.5 * (
            np.abs(merged["median_first"]) + np.abs(merged["median_second"])
        )
        merged["center_rel_difference"] = np.abs(
            merged["median_second"] - merged["median_first"]
        ) / np.maximum(merged["median_mean_abs"], 1.0e-300)
        merged["width_rel_difference"] = np.abs(
            merged["halfwidth_second"] - merged["halfwidth_first"]
        ) / np.maximum(
            0.5 * (np.abs(merged["halfwidth_first"]) + np.abs(merged["halfwidth_second"])),
            1.0e-300,
        )

        for (pid, flavor, x, Q), group in merged.groupby(["pid", "flavor", "x", "Q"], observed=False):
            mid = 0.5 * (group["median_first"] + group["median_second"])
            scale = float(np.nanmax(np.abs(mid)))
            active = np.abs(mid) > 0.05 * max(scale, 1.0e-300)
            rows.append({
                "quantity": quantity,
                "pid": int(pid),
                "flavor": str(flavor),
                "x": float(x),
                "Q": float(Q),
                "active_points": int(np.sum(active)),
                "center_split_median_active": float(np.nanmedian(group.loc[active, "center_rel_difference"])) if active.any() else np.nan,
                "center_split_p90_active": float(np.nanquantile(group.loc[active, "center_rel_difference"], 0.90)) if active.any() else np.nan,
                "center_split_max_active": float(np.nanmax(group.loc[active, "center_rel_difference"])) if active.any() else np.nan,
                "width_split_median_active": float(np.nanmedian(group.loc[active, "width_rel_difference"])) if active.any() else np.nan,
                "width_split_p90_active": float(np.nanquantile(group.loc[active, "width_rel_difference"], 0.90)) if active.any() else np.nan,
                "width_split_max_active": float(np.nanmax(group.loc[active, "width_rel_difference"])) if active.any() else np.nan,
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-glob", default="replica_pilot_v22_lambda3_cached_cuda/outputs/v22_lambda3_cached_cuda_s*")
    parser.add_argument("--band-dir", default="replica_pilot_v22_lambda3_cached_cuda/tmd_bspace_bands_exactx")
    parser.add_argument("--band-audit-dir", default="replica_pilot_v22_lambda3_cached_cuda/tmd_bspace_bands_exactx/audit")
    parser.add_argument("--out", default="replica_pilot_v22_lambda3_cached_cuda/audit_convergence_q95")
    parser.add_argument("--min-replicas", type=int, default=10)
    parser.add_argument("--chi2-q95-max", type=float, default=2.2)
    parser.add_argument("--norm-q95-max", type=float, default=4.0)
    parser.add_argument("--center-split-p90-q95-max", type=float, default=0.08)
    parser.add_argument("--width-split-p90-q95-max", type=float, default=0.80)
    parser.add_argument("--width-split-p90-max-info", type=float, default=0.90)
    args = parser.parse_args()

    runs = [Path(p) for p in sorted(glob.glob(args.run_glob))]
    if not runs:
        raise SystemExit(f"No runs matched: {args.run_glob}")

    metrics = summarize_runs(runs)

    band_dir = Path(args.band_dir)
    long_path = band_dir / "v22_tmd_replica_bspace_long.csv"
    band_summary_path = Path(args.band_audit_dir) / "bspace_band_audit_summary.json"
    band_by_quantity_path = Path(args.band_audit_dir) / "bspace_band_audit_by_quantity.csv"

    for path in [long_path, band_summary_path, band_by_quantity_path]:
        if not path.exists():
            raise SystemExit(f"Missing required file: {path}")

    long = pd.read_csv(long_path)
    band_summary = load_json(band_summary_path)
    by_quantity = pd.read_csv(band_by_quantity_path)

    quantities = ["F_NP", "ftilde", "x_ftilde", "b_ftilde", "b_x_ftilde"]
    split = split_half_stability(long, quantities)

    split_by_quantity = (
        split.groupby("quantity", observed=False)
        .agg(
            n_curves=("x", "size"),
            center_split_p90_median=("center_split_p90_active", "median"),
            center_split_p90_q95=("center_split_p90_active", lambda v: float(np.nanquantile(v, 0.95))),
            center_split_p90_max=("center_split_p90_active", "max"),
            width_split_p90_median=("width_split_p90_active", "median"),
            width_split_p90_q95=("width_split_p90_active", lambda v: float(np.nanquantile(v, 0.95))),
            width_split_p90_max=("width_split_p90_active", "max"),
        )
        .reset_index()
    )

    chi2 = metrics["chi2_total"].to_numpy(float)
    norm = metrics["max_abs_norm_pull"].to_numpy(float)

    n_replicas = len(metrics)
    chi2_q95 = float(np.nanquantile(chi2, 0.95))
    chi2_max = float(np.nanmax(chi2))
    norm_q95 = float(np.nanquantile(norm, 0.95)) if np.isfinite(norm).any() else np.nan
    norm_max = float(np.nanmax(norm)) if np.isfinite(norm).any() else np.nan

    center_split_q95 = float(np.nanquantile(split["center_split_p90_active"], 0.95))
    width_split_q95 = float(np.nanquantile(split["width_split_p90_active"], 0.95))
    width_split_max = float(np.nanmax(split["width_split_p90_active"]))

    fit_distribution_pass = bool(n_replicas >= int(args.min_replicas) and chi2_q95 < float(args.chi2_q95_max))
    norm_distribution_pass = bool((not np.isfinite(norm_q95)) or norm_q95 < float(args.norm_q95_max))
    band_pass = bool(
        band_summary.get("BSPACE_TMD_BAND_TECHNICAL_PASS")
        and band_summary.get("BSPACE_TMD_BAND_UNCERTAINTY_USEFUL_PASS")
    )
    split_pass = bool(
        center_split_q95 < float(args.center_split_p90_q95_max)
        and width_split_q95 < float(args.width_split_p90_q95_max)
        and width_split_max < float(args.width_split_p90_max_info)
    )

    pass_status = bool(fit_distribution_pass and norm_distribution_pass and band_pass and split_pass)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    metrics.to_csv(out / "lambda3_fit_summary.csv", index=False)
    by_quantity.to_csv(out / "lambda3_band_by_quantity.csv", index=False)
    split.to_csv(out / "lambda3_split_half_by_curve.csv", index=False)
    split_by_quantity.to_csv(out / "lambda3_split_half_by_quantity.csv", index=False)

    summary = {
        "n_replicas": int(n_replicas),
        "chi2_median": float(np.nanmedian(chi2)),
        "chi2_q95": chi2_q95,
        "chi2_max": chi2_max,
        "norm_pull_q95": norm_q95 if np.isfinite(norm_q95) else None,
        "norm_pull_max": norm_max if np.isfinite(norm_max) else None,
        "fit_distribution_pass": fit_distribution_pass,
        "norm_distribution_pass": norm_distribution_pass,
        "band_technical_pass": bool(band_summary.get("BSPACE_TMD_BAND_TECHNICAL_PASS")),
        "band_uncertainty_useful_pass": bool(band_summary.get("BSPACE_TMD_BAND_UNCERTAINTY_USEFUL_PASS")),
        "max_relative_68_halfwidth_active": band_summary.get("max_relative_68_halfwidth_active"),
        "max_central_vs_replica_median_rel_p90_active": band_summary.get("max_central_vs_replica_median_rel_p90_active"),
        "center_split_p90_q95": center_split_q95,
        "center_split_p90_max": float(np.nanmax(split["center_split_p90_active"])),
        "width_split_p90_q95": width_split_q95,
        "width_split_p90_max": width_split_max,
        "split_half_distribution_pass": split_pass,
        "V22_LAMBDA3_BSPACE_ENSEMBLE_Q95_PASS": pass_status,
        "thresholds": {
            "min_replicas": int(args.min_replicas),
            "chi2_q95_max": float(args.chi2_q95_max),
            "norm_q95_max": float(args.norm_q95_max),
            "center_split_p90_q95_max": float(args.center_split_p90_q95_max),
            "width_split_p90_q95_max": float(args.width_split_p90_q95_max),
            "width_split_p90_max_info": float(args.width_split_p90_max_info),
        },
        "interpretation": (
            "This is a distribution-based ensemble gate. Max chi2 is informational; q95 is the fit criterion. "
            "If this fails only by split-width q95, append more replicas rather than changing physics settings."
        ),
    }

    (out / "lambda3_ensemble_q95_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("\n=== v22 lambda=3 ensemble q95 convergence summary ===")
    print(json.dumps(summary, indent=2))
    print("\n=== Fit summary ===")
    cols = [c for c in ["seed", "chi2_total", "max_abs_norm_pull", "best_epoch", "epochs_run"] if c in metrics.columns]
    print(metrics[cols].to_string(index=False))
    print("\n=== Band by quantity ===")
    print(by_quantity.to_string(index=False))
    print("\n=== Split-half stability by quantity ===")
    print(split_by_quantity.to_string(index=False))
    print("\nwrote:", out)

    if not pass_status:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
