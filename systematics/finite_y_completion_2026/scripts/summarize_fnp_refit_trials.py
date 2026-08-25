#!/usr/bin/env python3
"""Summarize isolated differentiable FNP promotion trials."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "systematics/high_qt_direct_production_benchmark/experimental_unitary_transition/outputs"
TRIALS = {
    "unanchored_lambda0_50k": "unitary_smootherstep_v1_differentiable_fnp_refit_completion_s303",
    "anchor_lambda1000_50k": "unitary_smootherstep_v1_differentiable_fnp_refit_completion_anchor1000_s303",
    "anchor_lambda1e5_20k": "unitary_smootherstep_v1_differentiable_fnp_refit_completion_anchor1e5_s303",
}


def main() -> None:
    trials = {}
    for name, tag in TRIALS.items():
        status = json.loads((BASE / tag / "fit_status.json").read_text())
        trials[name] = {
            "tag": tag,
            "epochs_run": status["epochs_run"],
            "best_epoch": status["best_epoch"],
            "convergence_gate_pass": status["convergence_gate_pass"],
            "stationarity_gate_pass": status["optimizer_polish"]["stationarity_gate_pass"],
            "total_chi2_per_row": status["refit"]["total_chi2_per_row"],
            "fnp_gradient_l2_per_row": status["optimizer_polish"]["fnp_gradient_l2_per_row_objective"],
            "fnp_parameter_relative_l2_shift": status["refit"]["fnp_parameter_relative_l2_shift"],
            "anchor_strength": status["regularization"]["strength"],
        }
    result = {
        "status": "differentiable_fnp_promotion_blocked_no_stationary_trial",
        "trials": trials,
        "conclusion": "Neither an unanchored 50k-epoch fit nor anchors of 1000 or 1e5 reached the stationarity gate; the finite-Y construction is not promotable with a newly refit flexible FNP under this objective.",
        "fallback": "retain the established FNP and validate unitary Y with frozen-FNP nuisance/replica studies",
        "production_outputs_modified": False,
    }
    out = ROOT / "systematics/finite_y_completion_2026/reports/fnp_refit_promotion_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
