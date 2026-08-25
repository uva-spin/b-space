#!/usr/bin/env python3
"""Convergence audit for the cached-CUDA v22 lambda=3 ten-replica pilot.

Inputs expected:
  replica_pilot_v22_lambda3_cached_cuda/outputs/v22_lambda3_cached_cuda_s*/
  replica_pilot_v22_lambda3_cached_cuda/tmd_bspace_bands_exactx/
  replica_pilot_v22_lambda3_cached_cuda/tmd_bspace_bands_exactx/audit/

The audit combines:
  * replica fit/norm metrics,
  * b-space band technical/usefulness audit,
  * split-half stability of b-space TMD bands.

It does not inspect kT-space transforms.
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
                name = str(ds["dataset"])
                row[f"chi2_{name}"] = float(ds["chi2_like"])
                if "median_pred_over_target" in ds:
                    row[f"ratio_{name}"] = float(ds["median_pred_over_target"])

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
            scale = float(np.nanmax(np.abs(0.5 * (group["median_first"] + group["median_second"]))))
            active = np.abs(0.5 * (group["median_first"] + group["median_second"])) > 0.05 * max(scale, 1.0e-300)

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
    parser.add_argument("--out", default="replica_pilot_v22_lambda3_cached_cuda/audit_convergence")
    parser.add_argument("--max-chi2", type=float, default=2.2)
    parser.add_argument("--max-norm-pull", type=float, default=4.0)
    parser.add_argument("--max-center-split-p90", type=float, default=0.08)
    parser.add_argument("--max-width-split-p90", type=float, default=0.80)
    args = parser.parse_args()

    runs = [Path(p) for p in sorted(glob.glob(args.run_glob))]
    if not runs:
        raise SystemExit(f"No runs matched: {args.run_glob}")

    metrics = summarize_runs(runs)

    band_dir = Path(args.band_dir)
    long_path = band_dir / "v22_tmd_replica_bspace_long.csv"
    bands_path = band_dir / "v22_tmd_replica_bspace_bands.csv"
    band_summary_path = Path(args.band_audit_dir) / "bspace_band_audit_summary.json"
    band_by_quantity_path = Path(args.band_audit_dir) / "bspace_band_audit_by_quantity.csv"

    for path in [long_path, bands_path, band_summary_path, band_by_quantity_path]:
        if not path.exists():
            raise SystemExit(f"Missing required band audit file: {path}")

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
            center_split_p90_max=("center_split_p90_active", "max"),
            width_split_p90_median=("width_split_p90_active", "median"),
            width_split_p90_max=("width_split_p90_active", "max"),
        )
        .reset_index()
    )

    n_replicas = len(metrics)
    chi2_max = float(metrics["chi2_total"].max())
    chi2_median = float(metrics["chi2_total"].median())
    norm_max = float(metrics["max_abs_norm_pull"].max()) if np.isfinite(metrics["max_abs_norm_pull"]).any() else float("nan")

    fit_pass = bool(n_replicas >= 10 and chi2_max < float(args.max_chi2))
    norm_pass = bool((not np.isfinite(norm_max)) or norm_max < float(args.max_norm_pull))
    band_pass = bool(
        band_summary.get("BSPACE_TMD_BAND_TECHNICAL_PASS")
        and band_summary.get("BSPACE_TMD_BAND_UNCERTAINTY_USEFUL_PASS")
    )

    center_split_p90_max = float(split["center_split_p90_active"].max())
    width_split_p90_max = float(split["width_split_p90_active"].max())

    split_pass = bool(
        center_split_p90_max < float(args.max_center_split_p90)
        and width_split_p90_max < float(args.max_width_split_p90)
    )

    pilot_pass = bool(fit_pass and norm_pass and band_pass and split_pass)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    metrics.to_csv(out / "lambda3_10replica_fit_summary.csv", index=False)
    by_quantity.to_csv(out / "lambda3_10replica_band_by_quantity.csv", index=False)
    split.to_csv(out / "lambda3_10replica_split_half_by_curve.csv", index=False)
    split_by_quantity.to_csv(out / "lambda3_10replica_split_half_by_quantity.csv", index=False)

    summary = {
        "n_replicas": int(n_replicas),
        "chi2_median": chi2_median,
        "chi2_max": chi2_max,
        "max_abs_norm_pull": norm_max if np.isfinite(norm_max) else None,
        "fit_pass": fit_pass,
        "norm_pass": norm_pass,
        "band_technical_pass": bool(band_summary.get("BSPACE_TMD_BAND_TECHNICAL_PASS")),
        "band_uncertainty_useful_pass": bool(band_summary.get("BSPACE_TMD_BAND_UNCERTAINTY_USEFUL_PASS")),
        "max_relative_68_halfwidth_active": band_summary.get("max_relative_68_halfwidth_active"),
        "max_central_vs_replica_median_rel_p90_active": band_summary.get("max_central_vs_replica_median_rel_p90_active"),
        "center_split_p90_max": center_split_p90_max,
        "width_split_p90_max": width_split_p90_max,
        "split_half_pass": split_pass,
        "V22_LAMBDA3_10REPLICA_BSPACE_PILOT_PASS": pilot_pass,
        "thresholds": {
            "max_chi2": float(args.max_chi2),
            "max_norm_pull": float(args.max_norm_pull),
            "max_center_split_p90": float(args.max_center_split_p90),
            "max_width_split_p90": float(args.max_width_split_p90),
        },
        "interpretation": (
            "A pass means this is a usable DY-only b-space TMD pilot ensemble. "
            "It is not a final production uncertainty: kT-space remains diagnostic, and larger ensembles are still needed for final band convergence."
        ),
    }

    (out / "lambda3_10replica_convergence_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("\n=== v22 lambda=3 cached-CUDA ten-replica convergence summary ===")
    print(json.dumps(summary, indent=2))

    print("\n=== Fit summary ===")
    display_cols = [c for c in ["seed", "chi2_total", "max_abs_norm_pull", "best_epoch", "epochs_run"] if c in metrics.columns]
    print(metrics[display_cols].to_string(index=False))

    print("\n=== Band by quantity ===")
    print(by_quantity.to_string(index=False))

    print("\n=== Split-half stability by quantity ===")
    print(split_by_quantity.to_string(index=False))

    print("\nwrote:", out)

    if not pilot_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
