#!/usr/bin/env python3
"""Measure same-objective drift in log-length continuation fits."""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
TARGET = BASE / "summaries/loglength_continuation"


def main() -> None:
    rows = []
    for filename in glob.glob(
            str(BASE / "outputs/loglength_*_continue/fit_status.json")):
        continued = Path(filename).parent
        original = continued.with_name(
            continued.name.removesuffix("_continue"))
        old_status = json.loads(
            (original / "fit_status.json").read_text())
        new_status = json.loads(Path(filename).read_text())
        old_grid = pd.read_csv(original / "fnp_grid.csv")
        new_grid = pd.read_csv(continued / "fnp_grid.csv")
        grid = old_grid.merge(
            new_grid, on=["x", "bT"],
            suffixes=("_initial", "_continued"),
            validate="one_to_one")
        old = grid["F_NP_initial"].to_numpy(float)
        new = grid["F_NP_continued"].to_numpy(float)
        reference = np.maximum(0.5 * (old + new), 1.0e-12)
        relative = np.abs(new - old) / reference
        active = reference > 0.05
        b = grid["bT"].to_numpy(float)
        x = grid["x"].to_numpy(float)

        old_prediction = pd.read_csv(
            original / "accepted_predictions.csv")
        new_prediction = pd.read_csv(
            continued / "accepted_predictions.csv")
        prediction = old_prediction[
            ["row_id", "control_prediction", "sigma_used"]
        ].merge(
            new_prediction[["row_id", "control_prediction"]],
            on="row_id", suffixes=("_initial", "_continued"),
            validate="one_to_one")
        pred_drift = np.abs(
            prediction["control_prediction_continued"].to_numpy(float)
            - prediction["control_prediction_initial"].to_numpy(float)
        ) / prediction["sigma_used"].to_numpy(float)

        regularizer = new_status[
            "regularization"]["logf_arc_length"]
        row = {
            "tag": continued.name,
            "candidate_id": (
                continued.name.split("_lam")[0]
                .removeprefix("loglength_")),
            "lambda_loglength": float(regularizer["lambda"]),
            "seed": int(new_status["seed"]),
            "unpenalized_chi2_change": float(
                new_status["final"]["unpenalized_total_chi2"]
                - old_status["final"]["unpenalized_total_chi2"]),
            "objective_per_row_change": float(
                new_status["final"]["objective_per_row"]
                - old_status["final"]["objective_per_row"]),
            "max_prediction_drift_over_sigma": float(
                np.max(pred_drift)),
            "p95_prediction_drift_over_sigma": float(
                np.quantile(pred_drift, 0.95)),
            "max_fnp_relative_drift_active": float(
                np.max(relative[active])),
        }
        for limit in (1.0, 1.5, 2.0, 3.0):
            mask = active & (b <= limit)
            row[
                f"max_fnp_relative_drift_bT_le_{limit:g}"
            ] = float(np.max(relative[mask]))
            x01 = mask & np.isclose(x, 0.1)
            row[
                f"max_fnp_relative_drift_x0p1_bT_le_{limit:g}"
            ] = float(np.max(relative[x01])) if np.any(x01) else None
        rows.append(row)

    frame = pd.DataFrame(rows)
    TARGET.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TARGET / "run_metrics.csv", index=False)
    (TARGET / "summary.json").write_text(json.dumps({
        "status": (
            "isolated_same_objective_loglength_"
            "continuation_not_production"),
        "run_count": len(rows),
        "stationarity_drift_gate": (
            "maximum FNP drift <=1% through bT=2 and prediction "
            "drift <=0.05 sigma"),
        "production_sources_modified": False,
        "runs": rows,
    }, indent=2) + "\n")
    print(frame.to_string(index=False) if len(frame)
          else "No continuation fits complete yet.")


if __name__ == "__main__":
    main()
