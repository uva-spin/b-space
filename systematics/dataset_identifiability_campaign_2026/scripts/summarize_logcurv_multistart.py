#!/usr/bin/env python3
"""Audit completed three-start log-curvature fits without assuming selection."""

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
    parser.add_argument("--lambda-logcurv", action="append", type=float)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def lambda_token(value: float) -> str:
    return f"{value:.0e}".replace("-", "m").replace("+", "")


def baseline_reference(candidate: str) -> tuple[float, Path]:
    values = []
    for seed in SEEDS:
        run = BASE / "outputs" / f"{candidate}_s{seed}_polish64"
        status = json.loads((run / "fit_status.json").read_text())
        values.append((float(status["final"]["unpenalized_total_chi2"]), run))
    return min(values, key=lambda item: item[0])


def summarize_group(
        candidate: str, lam: float
) -> tuple[dict, list[dict], list[dict]] | None:
    run_dirs = [
        BASE / "outputs" / (
            f"logcurv_{candidate}_lam{lambda_token(lam)}_s{seed}"
        )
        for seed in SEEDS
    ]
    if not all((run / "fit_status.json").exists() for run in run_dirs):
        return None

    statuses = [
        json.loads((run / "fit_status.json").read_text()) for run in run_dirs
    ]
    grids = [
        pd.read_csv(run / "fnp_grid.csv").assign(seed=seed)
        for run, seed in zip(run_dirs, SEEDS)
    ]
    predictions = [
        pd.read_csv(run / "accepted_predictions.csv")[
            ["row_id", "control_prediction", "sigma_used"]
        ].assign(seed=seed)
        for run, seed in zip(run_dirs, SEEDS)
    ]

    grid = pd.concat(grids, ignore_index=True).pivot(
        index=["x", "bT"], columns="seed", values="F_NP")
    values = grid.to_numpy(float)
    median = np.median(values, axis=1)
    relative_range = (np.max(values, axis=1) - np.min(values, axis=1)) / np.maximum(
        median, 1.0e-12)
    coordinates = grid.index.to_frame(index=False)
    active = median > 0.05

    prediction_long = pd.concat(predictions, ignore_index=True)
    prediction = prediction_long.pivot(
        index="row_id", columns="seed", values="control_prediction")
    sigma = prediction_long.groupby("row_id")["sigma_used"].first().reindex(
        prediction.index).to_numpy(float)
    prediction_range_sigma = (
        prediction.max(axis=1).to_numpy(float)
        - prediction.min(axis=1).to_numpy(float)
    ) / sigma

    regional = {}
    for limit in (1.0, 1.5, 2.0, 3.0, 8.0):
        mask = active & (coordinates["bT"].to_numpy(float) <= limit)
        regional[f"max_fnp_relative_range_bT_le_{limit:g}"] = float(
            np.max(relative_range[mask]))
        x01 = mask & np.isclose(coordinates["x"].to_numpy(float), 0.1)
        regional[f"max_fnp_relative_range_x0p1_bT_le_{limit:g}"] = float(
            np.max(relative_range[x01])) if np.any(x01) else None

    base, base_run = baseline_reference(candidate)
    base_prediction = pd.read_csv(base_run / "accepted_predictions.csv")
    base_dataset_chi2 = (
        base_prediction.assign(pull2=base_prediction["control_pull"].square())
        .groupby("dataset")["pull2"].sum()
    )
    run_rows = []
    dataset_rows = []
    for seed, status, run in zip(SEEDS, statuses, run_dirs):
        final = status["final"]
        run_rows.append({
            "candidate_id": candidate,
            "lambda_logcurv": lam,
            "seed": seed,
            "unpenalized_chi2": float(final["unpenalized_total_chi2"]),
            "delta_unpenalized_chi2_from_best_unregularized": (
                float(final["unpenalized_total_chi2"]) - base),
            "roughness_measure": (
                float(final["logcurv_penalty_per_row_objective"]) / lam),
            "fnp_gradient_l2_per_row": float(
                final["fnp_gradient_l2_per_row_objective"]),
            "norm_gradient_l2_per_row": float(
                final["normalization_gradient_l2_per_row_objective"]),
            "lbfgs_closure_evaluations": int(
                status["lbfgs"]["closure_evaluations"]),
            "adam_stopped_on_plateau": bool(status["stopped_on_plateau"]),
        })
        run_prediction = pd.read_csv(run / "accepted_predictions.csv")
        run_dataset_chi2 = (
            run_prediction.assign(pull2=run_prediction["control_pull"].square())
            .groupby("dataset")["pull2"].sum()
        )
        if set(run_dataset_chi2.index) != set(base_dataset_chi2.index):
            raise RuntimeError(f"dataset mismatch for {run}")
        for dataset in base_dataset_chi2.index:
            dataset_rows.append({
                "candidate_id": candidate,
                "lambda_logcurv": lam,
                "seed": seed,
                "dataset": dataset,
                "baseline_data_chi2": float(base_dataset_chi2[dataset]),
                "regularized_data_chi2": float(run_dataset_chi2[dataset]),
                "delta_data_chi2": float(
                    run_dataset_chi2[dataset] - base_dataset_chi2[dataset]),
            })

    group = {
        "candidate_id": candidate,
        "lambda_logcurv": lam,
        "seed_count": len(SEEDS),
        "unpenalized_chi2_range": float(
            max(row["unpenalized_chi2"] for row in run_rows)
            - min(row["unpenalized_chi2"] for row in run_rows)),
        "max_delta_unpenalized_chi2_from_best_unregularized": float(
            max(row["delta_unpenalized_chi2_from_best_unregularized"]
                for row in run_rows)),
        "max_dataset_delta_data_chi2": float(
            max(row["delta_data_chi2"] for row in dataset_rows)),
        "min_dataset_delta_data_chi2": float(
            min(row["delta_data_chi2"] for row in dataset_rows)),
        "roughness_range": float(
            max(row["roughness_measure"] for row in run_rows)
            - min(row["roughness_measure"] for row in run_rows)),
        "max_fnp_gradient_l2_per_row": float(
            max(row["fnp_gradient_l2_per_row"] for row in run_rows)),
        "max_prediction_range_over_sigma": float(
            np.max(prediction_range_sigma)),
        "p95_prediction_range_over_sigma": float(
            np.quantile(prediction_range_sigma, 0.95)),
        **regional,
    }
    return group, run_rows, dataset_rows


def main() -> None:
    args = arguments()
    config = json.loads((BASE / "config/logcurv_strength_ladder.json").read_text())
    candidates = args.candidate or config["candidate_ids"]
    strengths = args.lambda_logcurv or (
        config["pilot_strengths"] + config.get("extension_strengths", []))

    groups, runs, datasets, missing = [], [], [], []
    for candidate in candidates:
        for lam in strengths:
            result = summarize_group(candidate, lam)
            if result is None:
                missing.append({"candidate_id": candidate, "lambda_logcurv": lam})
                continue
            group, run_rows, dataset_rows = result
            groups.append(group)
            runs.extend(run_rows)
            datasets.extend(dataset_rows)
    if args.require_complete and missing:
        raise RuntimeError(f"incomplete three-start groups: {missing}")

    target = BASE / "summaries/logcurv_multistart"
    target.mkdir(parents=True, exist_ok=True)
    group_frame = pd.DataFrame(groups)
    run_frame = pd.DataFrame(runs)
    dataset_frame = pd.DataFrame(datasets)
    group_frame.to_csv(target / "group_metrics.csv", index=False)
    run_frame.to_csv(target / "run_metrics.csv", index=False)
    dataset_frame.to_csv(target / "dataset_metrics.csv", index=False)
    (target / "summary.json").write_text(json.dumps({
        "status": "isolated_logcurv_multistart_audit_not_production",
        "complete_group_count": len(groups),
        "missing_groups": missing,
        "gates_are_evaluated_only_after_independent_starts_exist": True,
        "production_sources_modified": False,
        "groups": groups,
    }, indent=2) + "\n")
    print(group_frame.to_string(index=False) if len(group_frame)
          else "No complete three-start groups yet.")


if __name__ == "__main__":
    main()
