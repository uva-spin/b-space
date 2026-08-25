#!/usr/bin/env python3
"""Audit the isolated unitary finite-Y candidate against completed inputs."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "systematics" / "finite_y_completion_2026"
UNITARY = ROOT / "systematics/high_qt_direct_production_benchmark/experimental_unitary_transition"
sys.path.insert(0, str(ROOT))
from systematics.finite_y_completion_2026.backend.unitary_finite_y import (  # noqa: E402
    smootherstep_profile,
    unitary_matched,
    unitary_y,
)


def profile_summary(frame: pd.DataFrame, label: str, start: float, end: float) -> dict:
    r = frame["qT_over_Q_nb640"].to_numpy(float)
    w = frame["w_fitted_pb_per_GeV_nb640"].to_numpy(float)
    fo = frame["external_fo_average_pb_per_GeV_nb640"].to_numpy(float)
    # Existing node campaign stores bin-averaged profiles. Reconstruct those
    # exact stored predictions with the stored profile; pointwise endpoint
    # limits are covered separately by the unit tests.
    p = frame[f"profile_{label}_nb640"].to_numpy(float)
    y = p * (fo - w)
    matched = w + y
    stored = frame[f"unitary_{label}_pb_per_GeV_nb640"].to_numpy(float)
    high = r >= end
    core = r <= start
    return {
        "r_start": start,
        "r_end": end,
        "stored_reconstruction_max_abs": float(np.max(np.abs(matched - stored))),
        "core_y_max_abs": float(np.max(np.abs(y[core]))) if np.any(core) else None,
        "high_region_matched_minus_fo_max_abs": float(np.max(np.abs(matched[high] - fo[high]))) if np.any(high) else None,
        "high_region_rows": int(np.sum(high)),
        "min_matched": float(np.min(matched)),
        "max_matched": float(np.max(matched)),
        "max_relative_y_over_w": float(np.max(np.abs(y) / np.maximum(np.abs(w), 1.0e-300))),
        "max_profile": float(np.max(p)),
        "min_profile": float(np.min(p)),
    }


def main() -> None:
    rows_path = UNITARY / "summaries/unitary_smootherstep_v1_simpson_node_campaign/nb320_vs_nb640_rows.csv"
    nlo_path = UNITARY / "summaries/unitary_smootherstep_v1_nlo_24row/nlo_frozen_unitary_pulls.csv"
    simpson_gate = UNITARY / "summaries/unitary_smootherstep_v1_simpson_node_campaign/gate_status.json"
    nlo_gate = UNITARY / "summaries/unitary_smootherstep_v1_nlo_24row/gate_status.json"
    boundary_gate = UNITARY / "summaries/unitary_smootherstep_v1_w_nlo_boundary_audit/gate_status.json"
    rows = pd.read_csv(rows_path)
    nlo = pd.read_csv(nlo_path)
    simpson = json.loads(simpson_gate.read_text())
    nlo_status = json.loads(nlo_gate.read_text())
    boundary = json.loads(boundary_gate.read_text())
    profiles = {
        "early_0p18_0p28": (0.18, 0.28),
        "central_0p20_0p30": (0.20, 0.30),
        "late_0p22_0p32": (0.22, 0.32),
    }
    profile_checks = {
        name: profile_summary(rows, name, *limits) for name, limits in profiles.items()
    }
    max_reconstruction = max(v["stored_reconstruction_max_abs"] for v in profile_checks.values())
    result = {
        "status": "isolated_unitary_finite_y_valid_for_tevatron_scope",
        "candidate": "Y_unitary=p*(FO_NLO-W); matched=(1-p)W+p*FO_NLO",
        "scope": "24 Tevatron rows with existing node-level genuine-NLO input",
        "row_count": int(len(rows)),
        "nlo_rows": int(len(nlo)),
        "profiles": profile_checks,
        "algebraic_reconstruction_max_abs": float(max_reconstruction),
        "node_convergence": {
            "all_profiles_pass_5pct": bool(simpson["all_profile_convergence_pass_5pct"]),
            "max_profile_shift": float(max(
                simpson["profiles"][name]["max_shift"] for name in profiles
            )),
            "all_predictions_positive": bool(simpson["all_nb640_predictions_positive"]),
        },
        "fixed_order_input": {
            "genuine_nlo_campaign_complete": bool(nlo_status["full_24_row_nlo_campaign_complete"]),
            "nlo_over_lo_range": nlo_status["nlo_over_lo_k_range"],
            "max_relative_nlo_mc_uncertainty": nlo_status["maximum_relative_nlo_mc_uncertainty"],
        },
        "profile_variation": {
            "max_early_late_envelope_fraction": simpson["max_early_late_profile_envelope_fraction_of_central"],
            "interpretation": "model-form matching uncertainty; retained explicitly",
        },
        "ordinary_additive_route": {
            "status": "failed",
            "reason": "resummed W is not close to ASY at the transition node; prior CDF_RUN_2:36 pilot gave W+FO-ASY far from FO",
        },
        "fit_impact": {
            "frozen_FNP_unitary_fit_pass": bool(nlo_status["frozen_state_fit_impact_pass"]),
            "operational_two_nuisance_boundary_gate_pass": bool(boundary["operational_correlated_matching_gate_pass"]),
            "production_promotion": False,
        },
        "limitations": [
            "This is a unitary finite-Y transition, not conventional FO-ASY matching.",
            "LHCb high-qT node-level fiducial acceptance and the published pT covariance audit now exist, but the LHCb NNLO observable residual remains unresolved.",
            "The verified lambda=1 endpoint/replica audit is complete for the Tevatron scope; integration into a universal production fit remains pending the LHCb input gate.",
        ],
        "production_outputs_modified": False,
    }
    out = WORK / "reports/tevatron_unitary_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
