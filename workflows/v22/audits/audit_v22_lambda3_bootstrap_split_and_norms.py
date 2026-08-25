#!/usr/bin/env python3
"""Outlier + randomized split-half audit for the v22 lambda=3 b-space ensemble.

Why this exists:
  The deterministic split-half audit compares the first half of seeds to the
  second half of seeds.  That is a useful stress test, but with 20 replicas it
  can be dominated by one unlucky partition.  This script keeps that
  deterministic split, then adds many random balanced split-half tests.

It also reports normalization-pull outliers by seed and dataset.

This does not inspect kT-space transforms.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


KEYS = ["pid", "flavor", "x", "Q", "bT"]
CURVE_KEYS = ["pid", "flavor", "x", "Q"]
QUANTITIES = ["F_NP", "ftilde", "x_ftilde", "b_ftilde", "b_x_ftilde"]


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


def norm_pulls_from_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    pulls = metrics.get("dataset_norm_pulls", {})
    if pulls:
        return {str(k): float(v) for k, v in pulls.items()}
    return {}


def infer_seed(run: Path, metrics: dict[str, Any]) -> str:
    cfg = metrics.get("config", {})
    for key in ["replica_seed", "seed"]:
        if key in cfg:
            return str(cfg[key])
    for token in run.name.replace("-", "_").split("_"):
        if token.startswith("s") and token[1:].isdigit():
            return token[1:]
    return run.name


def summarize_runs(runs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    norm_rows = []

    for run in runs:
        metrics = load_json(run / "metrics.json")
        seed = infer_seed(run, metrics)
        pulls = norm_pulls_from_metrics(metrics)

        row = {
            "seed": seed,
            "run": str(run),
            "chi2_total": weighted_chi2(metrics),
            "max_abs_norm_pull": (
                max(abs(v) for v in pulls.values()) if pulls else float("nan")
            ),
            "max_norm_pull_dataset": (
                max(pulls, key=lambda k: abs(pulls[k])) if pulls else ""
            ),
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

        for dataset, pull in pulls.items():
            norm_rows.append({
                "seed": seed,
                "run": str(run),
                "dataset": dataset,
                "norm_pull": pull,
                "abs_norm_pull": abs(pull),
            })

        rows.append(row)

    return (
        pd.DataFrame(rows).sort_values("seed").reset_index(drop=True),
        pd.DataFrame(norm_rows).sort_values(["abs_norm_pull", "seed"], ascending=[False, True]).reset_index(drop=True),
    )


def prepare_quantity_arrays(long: pd.DataFrame, quantity: str):
    needed = set(KEYS + ["seed", quantity])
    missing = needed.difference(long.columns)
    if missing:
        raise SystemExit(f"Missing columns for {quantity}: {sorted(missing)}")

    work = long[KEYS + ["seed", quantity]].copy()
    work["seed"] = work["seed"].astype(str)

    pivot = (
        work.pivot_table(
            index=KEYS,
            columns="seed",
            values=quantity,
            aggfunc="first",
            observed=False,
        )
        .sort_index()
    )

    seeds = [str(c) for c in pivot.columns]
    values = pivot.to_numpy(float)

    index_frame = pivot.index.to_frame(index=False)
    curve_frame = index_frame[CURVE_KEYS].drop_duplicates().reset_index(drop=True)
    curve_id = (
        index_frame[CURVE_KEYS]
        .merge(
            curve_frame.reset_index().rename(columns={"index": "curve_id"}),
            on=CURVE_KEYS,
            how="left",
            validate="many_to_one",
        )["curve_id"]
        .to_numpy(int)
    )

    # Active mask defined from the full ensemble median, not from each split,
    # so random split comparisons use the same support.
    full_median = np.nanmedian(values, axis=1)
    active = np.zeros(len(full_median), dtype=bool)
    for cid in np.unique(curve_id):
        mask = curve_id == cid
        scale = float(np.nanmax(np.abs(full_median[mask])))
        if np.isfinite(scale) and scale > 0.0:
            active[mask] = np.abs(full_median[mask]) > 0.05 * scale

    return seeds, values, curve_id, active, curve_frame


def split_metrics_for_indices(
    *,
    values: np.ndarray,
    curve_id: np.ndarray,
    active: np.ndarray,
    first_idx: np.ndarray,
    second_idx: np.ndarray,
) -> dict[str, float]:
    a = values[:, first_idx]
    b = values[:, second_idx]

    med_a = np.nanmedian(a, axis=1)
    med_b = np.nanmedian(b, axis=1)

    hw_a = 0.5 * (np.nanquantile(a, 0.84, axis=1) - np.nanquantile(a, 0.16, axis=1))
    hw_b = 0.5 * (np.nanquantile(b, 0.84, axis=1) - np.nanquantile(b, 0.16, axis=1))

    center_rel = np.abs(med_b - med_a) / np.maximum(
        0.5 * (np.abs(med_a) + np.abs(med_b)),
        1.0e-300,
    )

    width_rel = np.abs(hw_b - hw_a) / np.maximum(
        0.5 * (np.abs(hw_a) + np.abs(hw_b)),
        1.0e-300,
    )

    center_curve_p90 = []
    width_curve_p90 = []

    for cid in np.unique(curve_id):
        mask = (curve_id == cid) & active
        if not mask.any():
            continue
        center_curve_p90.append(float(np.nanquantile(center_rel[mask], 0.90)))
        width_curve_p90.append(float(np.nanquantile(width_rel[mask], 0.90)))

    center_curve_p90 = np.asarray(center_curve_p90, dtype=float)
    width_curve_p90 = np.asarray(width_curve_p90, dtype=float)

    return {
        "center_split_p90_median": float(np.nanmedian(center_curve_p90)),
        "center_split_p90_q95": float(np.nanquantile(center_curve_p90, 0.95)),
        "center_split_p90_max": float(np.nanmax(center_curve_p90)),
        "width_split_p90_median": float(np.nanmedian(width_curve_p90)),
        "width_split_p90_q95": float(np.nanquantile(width_curve_p90, 0.95)),
        "width_split_p90_max": float(np.nanmax(width_curve_p90)),
    }


def run_random_splits(long: pd.DataFrame, n_splits: int, rng_seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(int(rng_seed))
    all_rows = []
    deterministic_rows = []

    for quantity in QUANTITIES:
        seeds, values, curve_id, active, _ = prepare_quantity_arrays(long, quantity)
        n = len(seeds)
        if n < 4:
            raise SystemExit("Need at least four replicas for split-half audit.")
        half = n // 2

        det_first = np.arange(half)
        det_second = np.arange(half, n)
        det = split_metrics_for_indices(
            values=values,
            curve_id=curve_id,
            active=active,
            first_idx=det_first,
            second_idx=det_second,
        )
        deterministic_rows.append({
            "quantity": quantity,
            "split_id": "deterministic_first_vs_second",
            "first_seeds": ",".join(seeds[i] for i in det_first),
            "second_seeds": ",".join(seeds[i] for i in det_second),
            **det,
        })

        for split_id in range(int(n_splits)):
            perm = rng.permutation(n)
            first = np.sort(perm[:half])
            second = np.sort(perm[half:])
            metrics = split_metrics_for_indices(
                values=values,
                curve_id=curve_id,
                active=active,
                first_idx=first,
                second_idx=second,
            )
            all_rows.append({
                "quantity": quantity,
                "split_id": int(split_id),
                "first_seeds": ",".join(seeds[i] for i in first),
                "second_seeds": ",".join(seeds[i] for i in second),
                **metrics,
            })

    return pd.DataFrame(all_rows), pd.DataFrame(deterministic_rows)


def summarize_random_splits(random_splits: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for quantity, group in random_splits.groupby("quantity", observed=False):
        for metric in [
            "center_split_p90_q95",
            "center_split_p90_max",
            "width_split_p90_q95",
            "width_split_p90_max",
        ]:
            values = group[metric].to_numpy(float)
            rows.append({
                "quantity": quantity,
                "metric": metric,
                "median_over_random_splits": float(np.nanmedian(values)),
                "q68_over_random_splits": float(np.nanquantile(values, 0.68)),
                "q90_over_random_splits": float(np.nanquantile(values, 0.90)),
                "q95_over_random_splits": float(np.nanquantile(values, 0.95)),
                "max_over_random_splits": float(np.nanmax(values)),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-glob", default="replica_pilot_v22_lambda3_cached_cuda/outputs/v22_lambda3_cached_cuda_s*")
    parser.add_argument("--band-dir", default="replica_pilot_v22_lambda3_cached_cuda/tmd_bspace_bands_exactx_20rep")
    parser.add_argument("--band-audit-dir", default="replica_pilot_v22_lambda3_cached_cuda/tmd_bspace_bands_exactx_20rep/audit")
    parser.add_argument("--out", default="replica_pilot_v22_lambda3_cached_cuda/audit_bootstrap_split_norms_20rep")
    parser.add_argument("--n-random-splits", type=int, default=300)
    parser.add_argument("--rng-seed", type=int, default=20260702)
    parser.add_argument("--chi2-q95-max", type=float, default=2.2)
    parser.add_argument("--norm-q95-max", type=float, default=4.0)
    parser.add_argument("--norm-q95-margin", type=float, default=0.05)
    parser.add_argument("--width-random-q90-max", type=float, default=0.80)
    parser.add_argument("--center-random-q90-max", type=float, default=0.08)
    args = parser.parse_args()

    runs = [Path(p) for p in sorted(glob.glob(args.run_glob))]
    if not runs:
        raise SystemExit(f"No runs matched: {args.run_glob}")

    metrics, norm_long = summarize_runs(runs)

    band_dir = Path(args.band_dir)
    long_path = band_dir / "v22_tmd_replica_bspace_long.csv"
    if not long_path.exists():
        raise SystemExit(f"Missing band long file: {long_path}")

    long = pd.read_csv(long_path)
    long["seed"] = long["seed"].astype(str)

    band_summary = load_json(Path(args.band_audit_dir) / "bspace_band_audit_summary.json")

    random_splits, deterministic = run_random_splits(
        long,
        n_splits=int(args.n_random_splits),
        rng_seed=int(args.rng_seed),
    )
    random_summary = summarize_random_splits(random_splits)

    chi2 = metrics["chi2_total"].to_numpy(float)
    norm = metrics["max_abs_norm_pull"].to_numpy(float)

    chi2_q95 = float(np.nanquantile(chi2, 0.95))
    norm_q95 = float(np.nanquantile(norm, 0.95)) if np.isfinite(norm).any() else np.nan

    width_q90 = float(
        random_summary[
            random_summary["metric"].eq("width_split_p90_q95")
        ]["q90_over_random_splits"].max()
    )
    center_q90 = float(
        random_summary[
            random_summary["metric"].eq("center_split_p90_q95")
        ]["q90_over_random_splits"].max()
    )

    fit_pass = bool(chi2_q95 < float(args.chi2_q95_max))
    norm_pass = bool((not np.isfinite(norm_q95)) or norm_q95 < float(args.norm_q95_max))
    norm_marginal = bool(
        np.isfinite(norm_q95)
        and norm_q95 < float(args.norm_q95_max) + float(args.norm_q95_margin)
    )
    band_pass = bool(
        band_summary.get("BSPACE_TMD_BAND_TECHNICAL_PASS")
        and band_summary.get("BSPACE_TMD_BAND_UNCERTAINTY_USEFUL_PASS")
    )
    random_split_pass = bool(
        width_q90 < float(args.width_random_q90_max)
        and center_q90 < float(args.center_random_q90_max)
    )

    conditional_pass = bool(fit_pass and band_pass and random_split_pass and (norm_pass or norm_marginal))
    strict_pass = bool(fit_pass and band_pass and random_split_pass and norm_pass)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    metrics.to_csv(out / "replica_fit_norm_summary.csv", index=False)
    norm_long.to_csv(out / "norm_pulls_long.csv", index=False)
    deterministic.to_csv(out / "deterministic_split_reference.csv", index=False)
    random_splits.to_csv(out / "random_split_draws.csv", index=False)
    random_summary.to_csv(out / "random_split_summary.csv", index=False)

    top_norm = norm_long.head(12).to_dict(orient="records") if not norm_long.empty else []

    summary = {
        "n_replicas_from_runs": int(len(metrics)),
        "n_replicas_in_band_long": int(long["seed"].nunique()),
        "n_random_splits": int(args.n_random_splits),
        "chi2_median": float(np.nanmedian(chi2)),
        "chi2_q95": chi2_q95,
        "chi2_max": float(np.nanmax(chi2)),
        "norm_pull_q95": norm_q95 if np.isfinite(norm_q95) else None,
        "norm_pull_max": float(np.nanmax(norm)) if np.isfinite(norm).any() else None,
        "fit_pass": fit_pass,
        "norm_pass": norm_pass,
        "norm_marginal_pass": norm_marginal,
        "band_pass": band_pass,
        "random_split_width_q90": width_q90,
        "random_split_center_q90": center_q90,
        "random_split_pass": random_split_pass,
        "STRICT_20REP_FREEZE_PASS": strict_pass,
        "CONDITIONAL_20REP_BSPACE_PILOT_PASS": conditional_pass,
        "top_norm_pull_rows": top_norm,
        "interpretation": (
            "If deterministic split fails but random split pass holds, the deterministic seed ordering is likely an unlucky partition. "
            "If norm fails only marginally and one seed/dataset dominates, inspect the outlier before changing physics settings."
        ),
    }

    (out / "bootstrap_split_norm_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("\n=== v22 lambda=3 20-rep bootstrap split + norm audit ===")
    print(json.dumps(summary, indent=2))
    print("\n=== Top normalization pulls ===")
    if norm_long.empty:
        print("No dataset_norm_pulls found in metrics.")
    else:
        print(norm_long.head(12).to_string(index=False))
    print("\n=== Deterministic split reference ===")
    print(deterministic.to_string(index=False))
    print("\n=== Random split summary ===")
    print(random_summary.to_string(index=False))
    print("\nwrote:", out)

    if not conditional_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
