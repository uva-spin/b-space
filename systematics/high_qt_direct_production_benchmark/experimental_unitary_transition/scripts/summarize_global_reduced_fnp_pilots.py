#!/usr/bin/env python3
"""Summarize globally reduced production-FNP fit-quality pilots."""

from __future__ import annotations

import json
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
TARGET = BASE / "summaries/production_fnp_global_reduced_pilots"
TAGS = {
    "gaussian_9_parameters": "production_fnp_global_reduced_c1_v2_pilot_s303",
    "two_component_18_parameters": "production_fnp_global_reduced_c2_v2_pilot_s303",
    "spline_54_parameters": "production_fnp_global_spline_pilot_s303",
    "spline_153_parameters": "production_fnp_global_spline_nb17_pilot_s303",
}


def main():
    pilots = {}
    for name, tag in TAGS.items():
        status = json.loads((BASE / "outputs" / tag / "fit_status.json").read_text())
        pilots[name] = {
            "tag": tag, "model_constraint": status["model_constraint"],
            "epochs_run": status["epochs_run"], "best_epoch": status["best_epoch"],
            "learning_rate": status["learning_rate"],
            "data_chi2": status["final"]["data_chi2"],
            "total_chi2": status["final"]["total_chi2"],
            "fnp_gradient_l2": status["final"]["fnp_gradient_l2_per_row_objective"],
        }
    summary = {
        "status": "experimental_global_reduced_fnp_pilots_not_production",
        "pilots": pilots,
        "reference_5000_epoch_unconstrained_data_chi2": 147.85435485839844,
        "selected_candidate": None,
        "decision": "reject the tested globally reduced forms before multi-start stability tests",
        "reason": "all compact positive monotone forms fail the production fit-quality gate by large margins",
        "interpretation": "the production objective requires substantial joint x and b structure; increasing spline resolution begins to restore the original flexibility without guaranteeing identifiability",
        "next_gate": "separate identifiable observable-level combinations from pointwise FNP inference, or adopt an explicit physics prior with acknowledged prior dependence",
        "production_state_modified": False,
    }
    TARGET.mkdir(parents=True, exist_ok=True)
    (TARGET / "pilot_status.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
