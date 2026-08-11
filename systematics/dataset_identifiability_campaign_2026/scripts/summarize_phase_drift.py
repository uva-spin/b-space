#!/usr/bin/env python3
"""Quantify per-start movement between two optimization phases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-phase", required=True)
    parser.add_argument("--to-phase", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[701, 702, 703])
    return parser.parse_args()


def run_metrics(candidate: str, seed: int, source: str, target: str) -> dict:
    old_dir = BASE / "outputs" / f"{candidate}_s{seed}_{source}"
    new_dir = BASE / "outputs" / f"{candidate}_s{seed}_{target}"
    old_grid = pd.read_csv(old_dir / "fnp_grid.csv")
    new_grid = pd.read_csv(new_dir / "fnp_grid.csv")
    grid = old_grid.merge(
        new_grid, on=["x", "bT"], suffixes=("_from", "_to"),
        validate="one_to_one")
    reference = np.maximum(
        0.5 * (grid["F_NP_from"].to_numpy(float) + grid["F_NP_to"].to_numpy(float)),
        1.0e-12)
    relative = np.abs(
        grid["F_NP_to"].to_numpy(float) - grid["F_NP_from"].to_numpy(float)
    ) / reference
    active = reference > 0.05
    b = grid["bT"].to_numpy(float)

    old_prediction = pd.read_csv(old_dir / "accepted_predictions.csv")
    new_prediction = pd.read_csv(new_dir / "accepted_predictions.csv")
    prediction = old_prediction[["row_id", "control_prediction", "sigma_used"]].merge(
        new_prediction[["row_id", "control_prediction"]],
        on="row_id", suffixes=("_from", "_to"), validate="one_to_one")
    prediction_shift = np.abs(
        prediction["control_prediction_to"].to_numpy(float)
        - prediction["control_prediction_from"].to_numpy(float)
    ) / prediction["sigma_used"].to_numpy(float)

    old_status = json.loads((old_dir / "fit_status.json").read_text())
    new_status = json.loads((new_dir / "fit_status.json").read_text())
    result = {
        "candidate_id": candidate,
        "seed": seed,
        "from_phase": source,
        "to_phase": target,
        "objective_per_row_from": old_status["final"]["objective_per_row"],
        "objective_per_row_to": new_status["final"]["objective_per_row"],
        "objective_per_row_change": (
            new_status["final"]["objective_per_row"]
            - old_status["final"]["objective_per_row"]),
        "max_prediction_shift_over_sigma": float(np.max(prediction_shift)),
        "p95_prediction_shift_over_sigma": float(np.quantile(prediction_shift, 0.95)),
        "max_fnp_relative_drift_active": float(np.max(relative[active])),
    }
    for limit in (1.0, 2.0, 3.0, 8.0):
        mask = active & (b <= limit)
        result[f"max_fnp_relative_drift_bT_le_{limit:g}"] = float(np.max(relative[mask]))
    return result


def main() -> None:
    args = arguments()
    registry = pd.read_csv(BASE / "manifests/dataset_candidates.csv")
    candidates = registry.loc[registry["central_output_exists"], "candidate_id"].tolist()
    results = [
        run_metrics(candidate, seed, args.from_phase, args.to_phase)
        for candidate in candidates for seed in args.seeds
    ]
    frame = pd.DataFrame(results)
    target = BASE / "summaries" / f"phase_drift_{args.from_phase}_to_{args.to_phase}"
    target.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target / "runs.csv", index=False)
    aggregate = frame.groupby("candidate_id", as_index=False).agg(
        max_prediction_shift_over_sigma=("max_prediction_shift_over_sigma", "max"),
        max_fnp_relative_drift_bT_le_1=("max_fnp_relative_drift_bT_le_1", "max"),
        max_fnp_relative_drift_bT_le_2=("max_fnp_relative_drift_bT_le_2", "max"),
        max_fnp_relative_drift_active=("max_fnp_relative_drift_active", "max"),
    )
    aggregate.to_csv(target / "candidate_metrics.csv", index=False)
    (target / "summary.json").write_text(json.dumps({
        "from_phase": args.from_phase,
        "to_phase": args.to_phase,
        "seeds": args.seeds,
        "run_count": len(results),
        "candidates": aggregate.to_dict(orient="records"),
    }, indent=2) + "\n")
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
