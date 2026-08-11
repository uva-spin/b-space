#!/usr/bin/env python3
"""Summarize cross-start FNP and prediction spread by candidate and phase."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[701, 702, 703])
    return parser.parse_args()


def candidate_metrics(candidate: str, phase: str, seeds: list[int]) -> dict:
    grids = []
    predictions = []
    statuses = []
    for seed in seeds:
        run = BASE / "outputs" / f"{candidate}_s{seed}_{phase}"
        grids.append(pd.read_csv(run / "fnp_grid.csv").assign(seed=seed))
        predictions.append(
            pd.read_csv(run / "accepted_predictions.csv")[
                ["row_id", "control_prediction", "sigma_used"]
            ].assign(seed=seed)
        )
        statuses.append(json.loads((run / "fit_status.json").read_text()))

    long_grid = pd.concat(grids, ignore_index=True)
    wide = long_grid.pivot(index=["x", "bT"], columns="seed", values="F_NP")
    values = wide.to_numpy(float)
    median = np.median(values, axis=1)
    relative_range = (np.max(values, axis=1) - np.min(values, axis=1)) / np.maximum(
        median, 1.0e-12)
    index = wide.index.to_frame(index=False)
    active = median > 0.05

    long_prediction = pd.concat(predictions, ignore_index=True)
    prediction_wide = long_prediction.pivot(
        index="row_id", columns="seed", values="control_prediction")
    sigma = long_prediction.groupby("row_id")["sigma_used"].first().reindex(
        prediction_wide.index).to_numpy(float)
    prediction_range_sigma = (
        prediction_wide.max(axis=1).to_numpy(float)
        - prediction_wide.min(axis=1).to_numpy(float)
    ) / sigma

    regional = {}
    for limit in (1.0, 2.0, 3.0, 8.0):
        mask = active & (index["bT"].to_numpy(float) <= limit)
        regional[f"bT_le_{limit:g}"] = float(np.max(relative_range[mask]))

    return {
        "candidate_id": candidate,
        "phase": phase,
        "seeds": seeds,
        "row_count": statuses[0]["row_count"],
        "all_stopped_on_plateau": all(item["stopped_on_plateau"] for item in statuses),
        "total_chi2_range": float(
            max(item["final"]["total_chi2"] for item in statuses)
            - min(item["final"]["total_chi2"] for item in statuses)),
        "max_prediction_range_over_sigma": float(np.max(prediction_range_sigma)),
        "p95_prediction_range_over_sigma": float(np.quantile(prediction_range_sigma, 0.95)),
        "max_fnp_relative_range_active": float(np.max(relative_range[active])),
        "p95_fnp_relative_range_active": float(np.quantile(relative_range[active], 0.95)),
        "regional_max_fnp_relative_ranges": regional,
    }


def main() -> None:
    args = arguments()
    registry = pd.read_csv(BASE / "manifests/dataset_candidates.csv")
    candidates = registry.loc[registry["central_output_exists"], "candidate_id"].tolist()
    results = [candidate_metrics(candidate, args.phase, args.seeds) for candidate in candidates]
    target = BASE / "summaries" / f"start_spread_{args.phase}"
    target.mkdir(parents=True, exist_ok=True)
    (target / "summary.json").write_text(json.dumps({
        "phase": args.phase,
        "candidate_count": len(results),
        "candidates": results,
    }, indent=2) + "\n")
    pd.DataFrame([{
        **{key: value for key, value in result.items()
           if key not in ("seeds", "regional_max_fnp_relative_ranges")},
        **result["regional_max_fnp_relative_ranges"],
    } for result in results]).to_csv(target / "candidate_metrics.csv", index=False)
    print(pd.DataFrame([{
        "candidate": result["candidate_id"],
        "chi2_range": result["total_chi2_range"],
        "pred_range_sigma": result["max_prediction_range_over_sigma"],
        "fnp_range_b1": result["regional_max_fnp_relative_ranges"]["bT_le_1"],
        "fnp_range_b2": result["regional_max_fnp_relative_ranges"]["bT_le_2"],
    } for result in results]).to_string(index=False))


if __name__ == "__main__":
    main()
