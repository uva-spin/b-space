#!/usr/bin/env python3
"""Write the gate-by-gate decision after all isolated constraint families."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
TARGET = BASE / "summaries/final_constraint_family_decision"
REFERENCE_CHI2 = 118.802105


def load_status(tag: str) -> dict:
    return json.loads(
        (BASE / "outputs" / tag / "fit_status.json").read_text())


def main() -> None:
    transform = pd.read_csv(
        BASE / "summaries/transform_closure_ladder/metrics.csv")
    v5 = transform[transform["tag"].str.endswith("_c2_endpoint_v5")]
    if len(v5) != 2:
        raise RuntimeError("both authoritative C2 endpoints are required")

    endpoint_rows = []
    for row in v5.to_dict("records"):
        status = load_status(row["tag"])
        endpoint_rows.append({
            "tag": row["tag"],
            "b_start": row["constraint_b_start"],
            "unpenalized_chi2": row["unpenalized_chi2"],
            "delta_chi2": row["unpenalized_chi2"] - REFERENCE_CHI2,
            "fnp_gradient_l2": row["fnp_gradient_l2"],
            "max_endpoint_fnp_b8": row["max_endpoint_fnp_b8"],
            "max_tailmode_change_active": (
                row["max_tailmode_central_change_active"]),
            "minimum_transform_over_peak": (
                row["minimum_tailmode_median_over_peak"]),
            "transform_gate_pass": bool(row["transform_gate_pass"]),
            "lbfgs_closures": status["lbfgs"]["closure_evaluations"],
        })

    summary = {
        "status": "no_constraint_family_passes_all_promotion_gates",
        "reference_unpenalized_chi2": REFERENCE_CHI2,
        "authoritative_c2_endpoint_results": endpoint_rows,
        "dataset_result": (
            "all 11 ordinary dataset candidates fail central "
            "stationarity/independent-start uniqueness"),
        "constraint_family_results": {
            "derivative_only_priors": "rejected: residual nonuniqueness",
            "logF_curvature": "rejected: residual nonuniqueness/stationarity",
            "logF_arc_length": (
                "rejected: independent-start and Fourier robustness"),
            "arc_length_plus_rate_curvature": (
                "rejected: finite-b continuation and stationarity"),
            "matched_or_reduced_tails": (
                "rejected: fit degradation or relocated degeneracy"),
            "reduced_architectures": (
                "rejected: minimum delta chi2 exceeds 10"),
            "C1_remote_closure": (
                "rejected: Fourier ringing and stationarity"),
            "C2_remote_closure": (
                "rejected: Fourier ringing, stationarity, and "
                "onset-location robustness"),
        },
        "probability_statement": (
            "Experimental replicas define a conditional 68% interval. "
            "Dataset choice, local minima, and closure onset have no "
            "validated probability measure and cannot be called a combined "
            "one-sigma band without an additional scientific convention."),
        "scientific_decision_required": [
            (
                "Adopt and disclose a specific prior/model plus a probability "
                "measure over its hyperparameters and optimizer starts, then "
                "run adaptive replicas."),
            (
                "Retain the flexible extraction and publish experimental "
                "68% intervals separately from a non-probabilistic "
                "model/nonuniqueness envelope."),
        ],
        "replica_promotion_authorized": False,
        "production_sources_modified": False,
    }
    TARGET.mkdir(parents=True, exist_ok=True)
    (TARGET / "decision.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
