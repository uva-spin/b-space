#!/usr/bin/env python3
"""Audit completed three-start log-length fits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
SEEDS = (701, 702, 703)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append")
    parser.add_argument("--lambda-loglength", action="append", type=float)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def lambda_token(value: float) -> str:
    return f"{value:.0e}".replace("-", "m").replace("+", "")


def baseline(candidate: str) -> float:
    values = []
    for seed in SEEDS:
        status = json.loads((
            BASE / "outputs"
            / f"{candidate}_s{seed}_polish64/fit_status.json"
        ).read_text())["final"]
        values.append(float(
            status.get(
                "unpenalized_total_chi2",
                status["data_chi2"] + status["norm_penalty"])))
    return min(values)


def summarize(candidate: str, lam: float) -> dict | None:
    runs = [
        BASE / "outputs" / (
            f"loglength_{candidate}_lam{lambda_token(lam)}_s{seed}")
        for seed in SEEDS
    ]
    if not all((run / "fit_status.json").exists() for run in runs):
        return None
    statuses = [
        json.loads((run / "fit_status.json").read_text())
        for run in runs
    ]
    grids = [
        pd.read_csv(run / "fnp_grid.csv").assign(seed=seed)
        for run, seed in zip(runs, SEEDS)
    ]
    grid = pd.concat(grids).pivot(
        index=["x", "bT"], columns="seed", values="F_NP")
    values = grid.to_numpy(float)
    median = np.median(values, axis=1)
    relative_range = (
        np.max(values, axis=1) - np.min(values, axis=1)
    ) / np.maximum(median, 1.0e-12)
    coordinates = grid.index.to_frame(index=False)
    active = median > 0.05

    predictions = [
        pd.read_csv(run / "accepted_predictions.csv")[
            ["row_id", "control_prediction", "sigma_used"]
        ].assign(seed=seed)
        for run, seed in zip(runs, SEEDS)
    ]
    long = pd.concat(predictions)
    prediction = long.pivot(
        index="row_id", columns="seed",
        values="control_prediction")
    sigma = long.groupby("row_id")["sigma_used"].first().reindex(
        prediction.index).to_numpy(float)
    prediction_range = (
        prediction.max(axis=1).to_numpy(float)
        - prediction.min(axis=1).to_numpy(float)
    ) / sigma

    base = baseline(candidate)
    final = [status["final"] for status in statuses]
    row = {
        "candidate_id": candidate,
        "lambda_loglength": lam,
        "seed_count": len(SEEDS),
        "max_delta_unpenalized_chi2_from_best_unregularized": float(
            max(item["unpenalized_total_chi2"] for item in final)
            - base),
        "unpenalized_chi2_range": float(
            max(item["unpenalized_total_chi2"] for item in final)
            - min(item["unpenalized_total_chi2"] for item in final)),
        "max_fnp_gradient_l2_per_row": float(max(
            item["fnp_gradient_l2_per_row_objective"]
            for item in final)),
        "max_prediction_range_over_sigma": float(
            np.max(prediction_range)),
        "p95_prediction_range_over_sigma": float(
            np.quantile(prediction_range, 0.95)),
    }
    b = coordinates["bT"].to_numpy(float)
    x = coordinates["x"].to_numpy(float)
    for limit in (1.0, 1.5, 2.0, 3.0, 8.0):
        mask = active & (b <= limit)
        row[f"max_fnp_relative_range_bT_le_{limit:g}"] = float(
            np.max(relative_range[mask]))
        x01 = mask & np.isclose(x, 0.1)
        row[
            f"max_fnp_relative_range_x0p1_bT_le_{limit:g}"
        ] = float(np.max(relative_range[x01])) if np.any(x01) else None
    return row


def main() -> None:
    args = arguments()
    config = json.loads(
        (BASE / "config/loglength_strength_ladder.json").read_text())
    candidates = args.candidate or config["candidate_ids"]
    strengths = args.lambda_loglength or config["pilot_strengths"]
    rows, missing = [], []
    for candidate in candidates:
        for lam in strengths:
            row = summarize(candidate, lam)
            if row is None:
                missing.append({
                    "candidate_id": candidate,
                    "lambda_loglength": lam})
            else:
                rows.append(row)
    if args.require_complete and missing:
        raise RuntimeError(f"incomplete groups: {missing}")
    target = BASE / "summaries/loglength_multistart"
    target.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(target / "group_metrics.csv", index=False)
    (target / "summary.json").write_text(json.dumps({
        "status": "isolated_loglength_multistart_audit_not_production",
        "complete_group_count": len(rows),
        "missing_groups": missing,
        "production_sources_modified": False,
        "groups": rows,
    }, indent=2) + "\n")
    print(frame.to_string(index=False) if len(frame)
          else "No complete three-start groups yet.")


if __name__ == "__main__":
    main()
