#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as handle:
        return json.load(handle)


def weighted_chi2(metrics: dict) -> float:
    per = pd.DataFrame(metrics.get("per_dataset", []))
    if per.empty:
        return float("nan")
    return float((per["chi2_like"] * per["n"]).sum() / per["n"].sum())


def max_abs_norm_pull(metrics: dict) -> float:
    pulls = metrics.get("dataset_norm_pulls", {})
    if not pulls:
        # Fall back to per-dataset final norm comparison if explicit pulls absent.
        return float("nan")
    return float(max(abs(float(v)) for v in pulls.values()))


def infer_seed(run: Path, metrics: dict) -> str:
    cfg = metrics.get("config", {})
    for key in ["replica_seed", "seed"]:
        if key in cfg:
            return str(cfg[key])
    name = run.name
    for token in name.replace("-", "_").split("_"):
        if token.startswith("s") and token[1:].isdigit():
            return token[1:]
    return name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", required=True)
    parser.add_argument("--central-run", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-replica-chi2", type=float, default=2.0)
    parser.add_argument("--max-norm-pull", type=float, default=4.0)
    args = parser.parse_args()

    runs = [Path(p) for p in sorted(glob.glob(args.glob))]
    if not runs:
        raise SystemExit(f"No runs matched: {args.glob}")

    central_run = Path(args.central_run)
    central_pred_path = central_run / "predictions.csv"
    if not central_pred_path.exists():
        raise SystemExit(f"Missing central predictions: {central_pred_path}")
    central = pd.read_csv(central_pred_path)

    rows = []
    pred_frames = []
    norm_frames = []

    for run in runs:
        metrics = load_json(run / "metrics.json")
        seed = infer_seed(run, metrics)

        pred_path = run / "predictions.csv"
        norms_path = run / "dataset_norms.csv"
        if not pred_path.exists():
            raise SystemExit(f"Missing predictions: {pred_path}")

        pred = pd.read_csv(pred_path)
        pred["seed"] = seed
        pred["run"] = str(run)
        pred_frames.append(pred)

        if norms_path.exists():
            norms = pd.read_csv(norms_path)
            norms["seed"] = seed
            norms["run"] = str(run)
            norm_frames.append(norms)

        row = {
            "seed": seed,
            "run": str(run),
            "replica_chi2_total": weighted_chi2(metrics),
            "max_abs_norm_pull": max_abs_norm_pull(metrics),
        }

        train = metrics.get("train", {})
        for key in ["best_epoch", "epochs_run", "best_objective", "restored_best"]:
            if key in train:
                row[key] = train[key]

        per = pd.DataFrame(metrics.get("per_dataset", []))
        if not per.empty:
            for _, ds in per.iterrows():
                row[f"chi2_{ds['dataset']}"] = float(ds["chi2_like"])
                if "median_pred_over_target" in ds:
                    row[f"ratio_{ds['dataset']}"] = float(ds["median_pred_over_target"])

        rows.append(row)

    metrics_df = pd.DataFrame(rows).sort_values("seed")
    predictions = pd.concat(pred_frames, ignore_index=True)
    norms = pd.concat(norm_frames, ignore_index=True) if norm_frames else pd.DataFrame()

    keys = ["dataset", "row_id"]
    central_cols = [c for c in ["pred_match_CS", "CS", "sigma_used", "target_used"] if c in central.columns]
    pred_cols = [c for c in ["pred_match_CS"] if c in predictions.columns]

    merged = predictions.merge(
        central[keys + central_cols],
        on=keys,
        how="left",
        suffixes=("_replica", "_central"),
        validate="many_to_one",
    )

    if "pred_match_CS_replica" in merged.columns and "pred_match_CS_central" in merged.columns:
        merged["replica_over_central"] = (
            merged["pred_match_CS_replica"]
            / merged["pred_match_CS_central"].replace(0, np.nan)
        )
        sigma_col = "sigma_used_central" if "sigma_used_central" in merged.columns else None
        if sigma_col:
            merged["delta_to_central_pull_units"] = (
                merged["pred_match_CS_replica"] - merged["pred_match_CS_central"]
            ) / merged[sigma_col].replace(0, np.nan)

    ensemble = (
        merged.groupby(keys, observed=False)
        .agg(
            pred_replica_mean=("pred_match_CS_replica", "mean"),
            pred_replica_median=("pred_match_CS_replica", "median"),
            pred_replica_q16=("pred_match_CS_replica", lambda x: float(np.nanquantile(x, 0.16))),
            pred_replica_q84=("pred_match_CS_replica", lambda x: float(np.nanquantile(x, 0.84))),
            pred_central=("pred_match_CS_central", "first"),
            data=("CS_central", "first") if "CS_central" in merged.columns else ("pred_match_CS_central", "first"),
            sigma=("sigma_used_central", "first") if "sigma_used_central" in merged.columns else ("pred_match_CS_central", "first"),
        )
        .reset_index()
    )
    ensemble["ensemble_shape_distance_to_central"] = (
        ensemble["pred_replica_mean"] - ensemble["pred_central"]
    ) / ensemble["sigma"].replace(0, np.nan)

    if "replica_over_central" in merged.columns:
        band_by_dataset = (
            merged.groupby(["seed", "dataset"], observed=False)
            .agg(
                ratio_median=("replica_over_central", "median"),
                ratio_p10=("replica_over_central", lambda x: float(np.nanquantile(x, 0.10))),
                ratio_p90=("replica_over_central", lambda x: float(np.nanquantile(x, 0.90))),
            )
            .reset_index()
        )
    else:
        band_by_dataset = pd.DataFrame()

    metrics_df["replica_fit_pass"] = metrics_df["replica_chi2_total"] < float(args.max_replica_chi2)
    metrics_df["norm_pull_pass"] = (
        metrics_df["max_abs_norm_pull"].isna()
        | (metrics_df["max_abs_norm_pull"] < float(args.max_norm_pull))
    )

    exact_three = len(metrics_df) == 3
    finite = bool(np.isfinite(metrics_df["replica_chi2_total"].to_numpy(float)).all())
    all_fit = bool(metrics_df["replica_fit_pass"].all())
    all_norm = bool(metrics_df["norm_pull_pass"].all())
    ensemble_distance = float(np.nanmedian(np.abs(ensemble["ensemble_shape_distance_to_central"])))

    pilot_pass = bool(exact_three and finite and all_fit and all_norm)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(out / "v22_replica_pilot_metrics.csv", index=False)
    merged.to_csv(out / "v22_replica_predictions_long.csv", index=False)
    ensemble.to_csv(out / "v22_replica_ensemble_vs_central.csv", index=False)
    if not norms.empty:
        norms.to_csv(out / "v22_replica_dataset_norms.csv", index=False)
    if not band_by_dataset.empty:
        band_by_dataset.to_csv(out / "v22_replica_prediction_ratios_by_dataset.csv", index=False)

    summary = {
        "n_replicas": int(len(metrics_df)),
        "exactly_three": exact_three,
        "all_finite": finite,
        "all_replica_fit_pass": all_fit,
        "all_norm_pull_pass": all_norm,
        "replica_chi2_median": float(metrics_df["replica_chi2_total"].median()),
        "replica_chi2_max": float(metrics_df["replica_chi2_total"].max()),
        "max_abs_norm_pull": (
            float(metrics_df["max_abs_norm_pull"].max())
            if np.isfinite(metrics_df["max_abs_norm_pull"].to_numpy(float)).any()
            else None
        ),
        "ensemble_median_abs_distance_to_central_sigma_units": ensemble_distance,
        "V22_THREE_REPLICA_BASIC_PASS": pilot_pass,
        "notes": [
            "This is a basic fit/norm/prediction audit. It does not yet audit b-space TMD band topology.",
            "Use the b-space band constructor after this passes.",
        ],
    }
    (out / "v22_replica_pilot_basic_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("\n=== v22 three-replica basic summary ===")
    print(metrics_df.to_string(index=False))
    print("\n=== Decision ===")
    print(json.dumps(summary, indent=2))
    print("\nwrote:", out)

    if not pilot_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
