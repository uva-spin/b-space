#!/usr/bin/env python3
"""Summarize the isolated production-only derivative-regularization tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
TARGET = BASE / "summaries/production_fnp_tail_regularization_scan"
CAMPAIGNS = {
    "unregularized": BASE / "summaries/production_fnp_stability_control/campaign_status.json",
    "damping_rate_curvature_lambda_3e-5": BASE / "summaries/production_fnp_ratecurv_lam3em5_stability_control/campaign_status.json",
    "A_slope_lambda_1e-2": BASE / "summaries/production_fnp_aslope_lam1em2_stability_control/campaign_status.json",
}


def main():
    rows = []
    for name, path in CAMPAIGNS.items():
        status = json.loads(path.read_text())
        rows.append({
            "campaign": name,
            "all_runs_converged": status["all_runs_converged"],
            "total_chi2_range": status["total_chi2_range"],
            "max_prediction_range_over_sigma": status["max_prediction_range_over_experimental_sigma"],
            "max_fnp_relative_range": status["max_fnp_relative_range_where_fnp_gt_0p05"],
            "max_fnp_range_bT_le_1": status["regional_max_fnp_relative_ranges"]["bT_le_1"],
            "max_fnp_range_bT_le_2": status["regional_max_fnp_relative_ranges"]["bT_le_2"],
            "fnp_stability_pass": status["fnp_local_stability_gate_pass"],
        })
    comparison = pd.DataFrame(rows)
    summary = {
        "status": "experimental_production_tail_regularization_scan_not_production",
        "tested_constraints": [
            "normalized curvature of h=-d_b log(F_NP)",
            "normalized squared slope of A in F_NP=exp(-b^2 A)",
        ],
        "ratecurv_result": "all starts plateau and bT<=1 stability improves, but smooth broad tail slopes remain non-unique",
        "A_slope_result": "tails become nearly constant-A, but their free amplitudes differ and the instability moves to the unconstrained onset region",
        "selected_constraint": None,
        "decision": "derivative-only tail constraints are insufficient; do not promote either tested regularizer",
        "physical_interpretation": "minimum bending or length fixes tail geometry but cannot determine a boundary amplitude absent from the data",
        "next_gate": "test a reduced low-dimensional A(x) tail parameterization with explicit matching at the data-supported onset",
        "production_state_modified": False,
    }
    TARGET.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(TARGET / "campaign_comparison.csv", index=False)
    (TARGET / "scan_status.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(comparison.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
