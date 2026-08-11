#!/usr/bin/env python3
"""Calibrate the selected non-physics constraint on all verified endpoints."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
SOURCE = BASE / "summaries/selected_reference_method_full24/summary.json"
TARGET = BASE / "summaries/selected_constraint_scale"


def main() -> None:
    selected = json.loads(SOURCE.read_text())
    if selected["status"] != "complete":
        raise RuntimeError("selected 24-start verification is not complete")
    strength = float(selected["selected_strength"])
    rows = []
    for tag in selected["endpoint_tags"]:
        status = json.loads((BASE / "outputs" / tag / "fit_status.json").read_text())
        final = status["final"]
        likelihood = float(final["weighted_likelihood_per_row_objective"])
        penalty = float(final["reference_distance_penalty_per_row_objective"])
        objective = float(final["objective_per_row"])
        rows.append({
            "seed": int(status["seed"]),
            "tag": tag,
            "unpenalized_total_chi2": float(final["unpenalized_total_chi2"]),
            "weighted_likelihood_per_row_objective": likelihood,
            "reference_distance_penalty_per_row_objective": penalty,
            "reference_penalty_fraction_of_total_objective": penalty / objective,
            "reference_penalty_fraction_of_weighted_likelihood": penalty / likelihood,
            "rms_normalized_fnp_displacement_implied_by_penalty":
                float(np.sqrt(max(penalty, 0.0) / strength)),
        })
    frame = pd.DataFrame(rows).sort_values("seed")
    TARGET.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TARGET / "endpoint_constraint_scale.csv", index=False)
    metrics = {}
    for column in (
        "reference_distance_penalty_per_row_objective",
        "reference_penalty_fraction_of_total_objective",
        "reference_penalty_fraction_of_weighted_likelihood",
        "rms_normalized_fnp_displacement_implied_by_penalty",
    ):
        values = frame[column].to_numpy(float)
        metrics[column] = {
            "min": float(np.min(values)),
            "median": float(np.median(values)),
            "max": float(np.max(values)),
        }
    summary = {
        "status": "complete",
        "selected_strength": strength,
        "selected_bmax": float(selected["selected_bmax"]),
        "endpoint_count": len(frame),
        "penalty_definition": (
            "lambda times the mean squared normalized FNP displacement from "
            "the empirical reference over the declared x,b support"
        ),
        "metrics": metrics,
        "nominal_lambda_is_not_a_likelihood_weight": True,
        "production_sources_modified": False,
    }
    (TARGET / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
