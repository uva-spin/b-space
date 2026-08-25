#!/usr/bin/env python3
"""Write the finite-Y campaign decision from the completed isolated gates."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "systematics/finite_y_completion_2026"


def main() -> None:
    validation = json.loads((WORK / "reports/tevatron_unitary_validation.json").read_text())
    fit_audit = json.loads((WORK / "reports/lambda1_unitary_fit_impact.json").read_text())
    replica_audit = json.loads((WORK / "reports/lambda1_unitary_boundary_replicas.json").read_text())
    conventional = json.loads(
        (WORK / "reports/conventional_y_reassessment/cdf_run_2_36.json").read_text()
    )
    nnlo_scale = json.loads(
        (WORK / "reports/lhcb_true_nnlo_scale_scan/summary.json").read_text()
    )
    covariance_audit = json.loads(
        (WORK / "reports/lhcb_correlated_covariance_audit/summary.json").read_text()
    )
    convention_scan = json.loads(
        (WORK / "reports/lhcb_nnlo_convention_scan/summary.json").read_text()
    )
    nnlo_unitary = json.loads(
        (WORK / "reports/lambda1_lhcb_unitary_nnlo/lhcb_unitary_nnlo_fit_impact_summary.json").read_text()
    )
    nnlo_replicas = json.loads(
        (WORK / "reports/lambda1_lhcb_unitary_nnlo/lhcb_unitary_nnlo_replica_summary.json").read_text()
    )
    mcfm_retry = json.loads(
        (WORK / "reports/lhcb7_external_mcfm_true_nlo_retry/status.json").read_text()
    )
    decision = {
        "status": "valid_isolated_unitary_finite_y_tevatron_scope_not_production",
        "campaign_state": "active_followup_after_conventional_candidate_rejection",
        "conventional_rejection_scope": "Only the additive FO-ASY ansatz is rejected; finite-Y construction, NNLO follow-up, and LHCb closure work remain active.",
        "candidate": validation["candidate"],
        "validity_conclusion": "The unitary finite-Y construction is numerically and algebraically valid for the 24-row Tevatron scope covered by genuine-NLO inputs.",
        "why_ordinary_additive_y_is_not_selected": "The prior FO-ASY construction fails because W is not close to ASY in the transition region; this is a domain-of-validity failure, not an algebraic sign failure.",
        "conventional_y_reassessment": conventional,
        "lhcb_nnlo_followup": nnlo_scale,
        "lhcb_published_pt_covariance_audit": covariance_audit,
        "lhcb_nnlo_convention_scan": convention_scan,
        "lhcb_nnlo_unitary_endpoint_fit": nnlo_unitary,
        "lhcb_nnlo_unitary_replica_propagation": nnlo_replicas,
        "lhcb_mcfm_nlo_retry": mcfm_retry,
        "passed_gates": {
            "algebraic_endpoint_and_c2": True,
            "exact_bin_reconstruction": validation["algebraic_reconstruction_max_abs"] < 1.0e-12,
            "node_convergence": validation["node_convergence"]["all_profiles_pass_5pct"],
            "genuine_nlo_input": validation["fixed_order_input"]["genuine_nlo_campaign_complete"],
            "positive_predictions": validation["node_convergence"]["all_predictions_positive"],
        },
        "profile_variation": validation["profile_variation"],
        "fit_impact": validation["fit_impact"],
        "lambda1_fnp_profile_fit_audit": {
            "central_gate_pass": fit_audit["central_all_operational_gates"],
            "profiles": fit_audit["profiles"],
            "interpretation": fit_audit["interpretation"],
        },
        "lambda1_unitary_replica_audit": replica_audit,
        "remaining_blockers_before_production": [
            "LHCb data normalization/covariance closure remains after the MCFM |y| versus positive-arm convention was resolved.",
            "The NNLO fixed-order LHCb candidate substantially improves the NLO deficit but remains below the first three boundary bins across the standard scale envelope; this observable-level residual must be closed before universal promotion.",
            "The published pT covariance has been reconstructed as an isolated audit; with NNLO unitary inputs the central profile still has median data chi2/row about 6.6, so correlations do not close the residual.",
            "The NNLO gamma*/electroweak convention scan shifts the row-10 ratio only to 0.802--0.830, so convention choices do not close the residual.",
            "The complete NNLO lambda=1 endpoint/profile fit is positive and convergent, but its central-profile median total chi2/row remains about 6.69 with matching and scale pulls about 2.56 and 2.28 sigma; this confirms that the remaining problem is LHCb observable closure rather than endpoint instability.",
            "The 50-replica NNLO endpoint propagation also converges for all 14,400 fits; its central-profile median total chi2/row is about 7.16, so experimental replica sampling does not remove the fixed theory/data residual.",
            "The unitary term must be described explicitly as a transition correction, not conventional FO-ASY Y matching.",
        ],
        "artifacts": {
            "validation": "reports/tevatron_unitary_validation.json",
            "candidate_backend": "backend/unitary_finite_y.py",
            "tests": "tests/test_unitary_finite_y.py",
            "conventional_y_reassessment": "reports/conventional_y_reassessment/cdf_run_2_36.json",
        },
        "production_outputs_modified": False,
    }
    out = WORK / "reports/decision_status.json"
    out.write_text(json.dumps(decision, indent=2) + "\n")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
