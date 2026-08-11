#!/usr/bin/env python3
"""Summarize isolated C1-matched log-FNP tail pilots and stability campaign."""

from __future__ import annotations

import json
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
TARGET = BASE / "summaries/production_fnp_c1logtail_decision"
PILOTS = {
    "bmatch_1p5": "production_fnp_c1logtail_b1p5_pilot_s303",
    "bmatch_2p0": "production_fnp_c1logtail_b2p0_pilot_s303",
    "bmatch_2p5": "production_fnp_c1logtail_b2p5_pilot_s303",
}


def main() -> None:
    pilots = {}
    for name, tag in PILOTS.items():
        status = json.loads((BASE / "outputs" / tag / "fit_status.json").read_text())
        pilots[name] = {
            "tag": tag,
            "data_chi2": status["final"]["data_chi2"],
            "total_chi2": status["final"]["total_chi2"],
            "max_prediction_shift_over_sigma": status["final"]["max_prediction_shift_over_experimental_sigma"],
            "converged": status["convergence_gate_pass"],
        }
    campaign_path = BASE / "summaries/production_fnp_c1logtail_b2p5_stability_control/campaign_status.json"
    campaign = json.loads(campaign_path.read_text())
    summary = {
        "status": "experimental_c1_matched_logF_tail_rejected_not_production",
        "model": "production FiLM core for b<=bmatch; C1-matched log-F tail with inherited value and slope and nine positive log-x curvature knots",
        "pilots": pilots,
        "reference_5000_epoch_unconstrained_data_chi2": 147.85435485839844,
        "pilot_decision": "only bmatch=2.5 passed to a three-start diagnostic; earlier matching damaged fit quality",
        "three_start_bmatch_2p5": {
            "all_runs_converged": campaign["all_runs_converged"],
            "total_chi2_range": campaign["total_chi2_range"],
            "max_prediction_range_over_sigma": campaign["max_prediction_range_over_experimental_sigma"],
            "max_fnp_relative_range": campaign["max_fnp_relative_range_where_fnp_gt_0p05"],
            "regional_max_fnp_relative_ranges": campaign["regional_max_fnp_relative_ranges"],
        },
        "selected_candidate": None,
        "decision": "reject the C1-matched low-dimensional tail; it relocates rather than resolves the FiLM-core degeneracy",
        "interpretation": "a tail constraint beginning late enough to preserve fit quality cannot identify the already unstable core below the match; moving the match earlier distorts fitted cross sections",
        "next_step": "retain the frozen production result and quantify model/nonuniqueness uncertainty explicitly rather than manufacturing uniqueness with this tail prior",
        "production_state_modified": False,
    }
    TARGET.mkdir(parents=True, exist_ok=True)
    (TARGET / "decision_status.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
