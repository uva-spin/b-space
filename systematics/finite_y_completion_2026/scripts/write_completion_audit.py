#!/usr/bin/env python3
"""Write the requirement-level finite-Y completion audit.

The report deliberately separates the validated Tevatron production candidate
from the unresolved universal LHCb input gate.  It is an isolated decision
record and never edits frozen production files.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "systematics/finite_y_completion_2026"
REPORTS = WORK / "reports"
OUT = REPORTS / "finite_y_completion_audit.json"


def read(relative: str) -> dict:
    return json.loads((WORK / relative).read_text())


def main() -> None:
    tevatron = read("reports/tevatron_unitary_validation.json")
    decision = read("reports/decision_status.json")
    nnlo_fit = read("reports/lambda1_lhcb_unitary_nnlo/lhcb_unitary_nnlo_fit_impact_summary.json")
    nnlo_replica = read("reports/lambda1_lhcb_unitary_nnlo/lhcb_unitary_nnlo_replica_summary.json")
    grid = read("reports/lhcb_node_acceptance_grid_production_candidate/summary.json")
    covariance = read("reports/lhcb_correlated_covariance_audit/summary.json")
    nnlo_scale = read("reports/lhcb_true_nnlo_scale_scan/summary.json")
    audit = {
        "status": "finite_y_complete_with_exhausted_lhcb_external_input_blocker",
        "objective_scope": "production-level finite-Y term without modifying frozen production files",
        "requirements": {
            "verified_lambda1_fnp_ensemble_used": {
                "status": "passed",
                "evidence": "96 registered lambda=1 endpoints used for Tevatron and LHCb W/Y audits",
            },
            "mathematically_consistent_matching": {
                "status": "passed_for_unitary_candidate",
                "candidate": "Y=p(qT/Q)(FO-W), matched=(1-p)W+pFO",
                "algebraic_and_c2_tests": decision["passed_gates"],
                "additive_fo_minus_asy": "rejected as a domain-of-validity failure only",
            },
            "tevatron_scope": {
                "status": "validated_24row_boundary_candidate_not_full_global_refit",
                "rows": tevatron["row_count"],
                "genuine_nlo_rows": tevatron["nlo_rows"],
                "endpoint_count": 96,
                "replicas": 50,
                "central_profile_total_chi2_per_row_q16_median_q84": read("reports/lambda1_unitary_fit_impact.json")["profiles"]["central_0p20_0p30"]["total_chi2_per_row_q16_median_q84"],
                "replica_central_profile_total_chi2_per_row_q16_median_q84": read("reports/lambda1_unitary_boundary_replicas.json")["profiles"]["central_0p20_0p30"]["total_chi2_per_row_q16_median_q84"],
                "full_fnal_global_w_plus_y_refit": "not performed; the 0.43 historical value remains the 329-row W-only production objective",
                "all_profiles_node_convergence": tevatron["node_convergence"]["all_profiles_pass_5pct"],
                "all_predictions_positive": tevatron["node_convergence"]["all_predictions_positive"],
            },
            "lhcb_fiducial_coverage": {
                "status": "validated_as_isolated_diagnostic",
                "rows": grid["rows"],
                "max_grid_vs_full_bin_relative": grid["max_abs_grid_vs_full_bin_relative"],
                "positive_arm_rapidity_convention": "validated against mirrored DYTurbo/MCFM LO closure",
                "nnlo_endpoint_fits_converged": nnlo_fit["all_optimizer_success"],
                "nnlo_endpoint_predictions_positive": nnlo_fit["all_predictions_positive"],
                "nnlo_replica_fits_converged": nnlo_replica["all_optimizer_success"],
                "nnlo_replica_predictions_positive": nnlo_replica["all_predictions_positive"],
            },
            "systematic_lhcb_closure_tests": {
                "status": "exhausted_with_residual",
                "tests": [
                    "true NLO all-14-bin shape",
                    "NLO six-point scale scan",
                    "NNLO seven-point scale scan",
                    "positive-arm versus inclusive/negative rapidity semantics",
                    "NNLO PDF set scan",
                    "NNLO electroweak/gamma* convention scan",
                    "published correlated pT covariance reconstruction",
                    "MCFM/DYTurbo rapidity-sign closure",
                    "NNLO lambda=1 endpoint/profile fit",
                    "NNLO lambda=1 50-replica propagation",
                ],
                "nnlo_max_scale_ratio_to_data": nnlo_scale["maximum_ratio_to_data"],
                "central_profile_total_chi2_per_row": nnlo_fit["profiles"]["central_0p20_0p30"]["total_chi2_per_row_q16_median_q84"],
                "replica_central_profile_total_chi2_per_row": nnlo_replica["profile_summary"]["central_0p20_0p30"]["total_chi2_per_row_q16_median_q84"],
            },
        },
        "remaining_blocker": {
            "status": "scientifically_explicit_and_exhausted",
            "statement": "Positive-arm LHCb high-qT data remain above the converged NNLO fixed-order input in the first three boundary bins; published covariance and normalization/provenance are not approved for a production fit.",
            "not_a_failure_of": [
                "unitary W/Y algebra",
                "lambda=1 FNP endpoint stability",
                "LHCb node acceptance coverage",
                "rapidity-sign convention",
            ],
            "prohibited_shortcut": "data-driven rescaling of FO or W/Y term",
            "next_external_input_required": "formal LHCb observable/covariance/normalization clarification or an independently validated fixed-order prediction that closes the published spectrum",
        },
        "scope_decision": {
            "tevatron": "unitary finite-Y is a validated production-level candidate",
            "lhcb": "W/Y machinery is complete diagnostically but universal production promotion is withheld",
            "frozen_production_files_modified": False,
            "goal_completion_basis": "A production-level finite-Y candidate is obtained for the verified Tevatron scope, and the universal-LHCb blocker is explicit and exhausted under the available inputs.",
        },
        "authoritative_decision_record": "reports/decision_status.json",
    }
    OUT.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
