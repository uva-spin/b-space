#!/usr/bin/env python3
"""Summarize reduced constant-A tail matching pilots."""

from __future__ import annotations

import json
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
TARGET = BASE / "summaries/production_fnp_reduced_tail_pilots"
TAGS = {
    "match_1_to_2": "production_fnp_reduced_tail_window1p0_2p0_pilot_s303",
    "match_2_to_3": "production_fnp_reduced_tail_window2p0_3p0_pilot_s303",
}


def main():
    pilots = {}
    for name, tag in TAGS.items():
        status = json.loads((BASE / "outputs" / tag / "fit_status.json").read_text())
        pilots[name] = {
            "tag": tag, "model_constraint": status["model_constraint"],
            "epochs_run": status["epochs_run"], "best_epoch": status["best_epoch"],
            "data_chi2": status["final"]["data_chi2"],
            "total_chi2": status["final"]["total_chi2"],
            "max_prediction_shift_over_sigma": status["final"]["max_prediction_shift_over_experimental_sigma"],
        }
    summary = {
        "status": "experimental_reduced_tail_pilots_not_production",
        "model": "FiLM core C2-matched to nine positive log-x knot amplitudes with constant learned A plus the fixed production late-b floor",
        "initialization": "least-squares match to the accumulated accepted log-FNP tail",
        "pilots": pilots,
        "reference_5000_epoch_unregularized_data_chi2": 147.85435485839844,
        "selected_candidate": None,
        "decision": "reject both matching windows before a three-start campaign",
        "reason": "the 1-to-2 window covers the instability but fails fit quality; the 2-to-3 window is tolerable but begins after the unstable region",
        "next_gate": "if pursued, test a globally reduced FNP form rather than a constant-A tail graft",
        "production_state_modified": False,
    }
    TARGET.mkdir(parents=True, exist_ok=True)
    (TARGET / "pilot_status.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
