#!/usr/bin/env python3
"""Summarize the corrected lambda=1 finite-Y scope and remaining LHCb gap.

This is an isolated decision record.  It never writes to a production model,
replica, or source-data directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "systematics/finite_y_completion_2026"
REPORTS = WORK / "reports"
DY = REPORTS / "lhcb7_external_missing/dyturbo"
MC = REPORTS / "lhcb7_external_missing/mcfm"
MC13 = REPORTS / "lhcb7_external_missing/mcfm13"
LHCB_GRID = REPORTS / "lhcb_node_acceptance_grid_production_candidate/summary.json"
LHCB_KERNELS = REPORTS / "lhcb_fiducial_w_kernels_nb640/campaign_status.json"
LHCB_RECOMPUTE = REPORTS / "lambda1_lhcb_unitary/summary.json"
LHCB_FIT = REPORTS / "lambda1_lhcb_unitary/lhcb_fit_impact_summary.json"
LHCB_CLOSURE = REPORTS / "lhcb_external_closure/summary.json"
LHCB_TRUE_NLO_SHAPE = REPORTS / "lhcb_true_nlo_shape_all14/summary.json"
LHCB_TRUE_NLO_SCALE = REPORTS / "lhcb_true_nlo_scale_scan/summary.json"
LHCB_TRUE_NNLO_SCALE = REPORTS / "lhcb_true_nnlo_scale_scan/summary.json"
LHCB_NNLO_RAPIDITY = REPORTS / "lhcb_nnlo_rapidity_semantics/summary.json"
LHCB_NNLO_PDF = REPORTS / "lhcb_nnlo_pdf_scan/summary.json"
LHCB_COVARIANCE = REPORTS / "lhcb_correlated_covariance_audit/summary.json"
LHCB_NNLO_CONVENTION = REPORTS / "lhcb_nnlo_convention_scan/summary.json"
LHCB_NNLO_UNITARY = REPORTS / "lambda1_lhcb_unitary_nnlo/lhcb_unitary_nnlo_fit_impact_summary.json"
LHCB_NNLO_REPLICAS = REPORTS / "lambda1_lhcb_unitary_nnlo/lhcb_unitary_nnlo_replica_summary.json"
CONVENTIONAL = REPORTS / "conventional_y_reassessment/cdf_run_2_36.json"


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def main() -> None:
    fit = json.loads((REPORTS / "lambda1_unitary_fit_impact.json").read_text())
    replicas = json.loads((REPORTS / "lambda1_unitary_boundary_replicas.json").read_text())
    endpoint = json.loads(
        (REPORTS / "lambda1_unitary_endpoint_recompute/summary.json").read_text()
    )
    dy = load_csv(DY / "dyturbo_benchmark_summary.csv")
    mc = load_csv(MC / "mcfm_benchmark_summary.csv")
    mc13 = load_csv(MC13 / "mcfm_benchmark_summary.csv")
    grid = json.loads(LHCB_GRID.read_text()) if LHCB_GRID.exists() else {}
    kernels = json.loads(LHCB_KERNELS.read_text()) if LHCB_KERNELS.exists() else {}
    lhcb_recompute = json.loads(LHCB_RECOMPUTE.read_text()) if LHCB_RECOMPUTE.exists() else {}
    lhcb_fit = json.loads(LHCB_FIT.read_text()) if LHCB_FIT.exists() else {}
    lhcb_closure = json.loads(LHCB_CLOSURE.read_text()) if LHCB_CLOSURE.exists() else {}
    lhcb_true_nlo_shape = json.loads(LHCB_TRUE_NLO_SHAPE.read_text()) if LHCB_TRUE_NLO_SHAPE.exists() else {}
    lhcb_true_nlo_scale = json.loads(LHCB_TRUE_NLO_SCALE.read_text()) if LHCB_TRUE_NLO_SCALE.exists() else {}
    lhcb_true_nnlo_scale = json.loads(LHCB_TRUE_NNLO_SCALE.read_text()) if LHCB_TRUE_NNLO_SCALE.exists() else {}
    lhcb_nnlo_rapidity = json.loads(LHCB_NNLO_RAPIDITY.read_text()) if LHCB_NNLO_RAPIDITY.exists() else {}
    lhcb_nnlo_pdf = json.loads(LHCB_NNLO_PDF.read_text()) if LHCB_NNLO_PDF.exists() else {}
    lhcb_covariance = json.loads(LHCB_COVARIANCE.read_text()) if LHCB_COVARIANCE.exists() else {}
    lhcb_nnlo_convention = json.loads(LHCB_NNLO_CONVENTION.read_text()) if LHCB_NNLO_CONVENTION.exists() else {}
    lhcb_nnlo_unitary = json.loads(LHCB_NNLO_UNITARY.read_text()) if LHCB_NNLO_UNITARY.exists() else {}
    lhcb_nnlo_replicas = json.loads(LHCB_NNLO_REPLICAS.read_text()) if LHCB_NNLO_REPLICAS.exists() else {}
    conventional = json.loads(CONVENTIONAL.read_text()) if CONVENTIONAL.exists() else {}
    external = pd.concat(
        [
            dy.assign(code="DYTurbo") if not dy.empty else dy,
            mc.assign(code="MCFM") if not mc.empty else mc,
            mc13.assign(code="MCFM") if not mc13.empty else mc13,
        ],
        ignore_index=True,
        sort=False,
    )
    external_rows = sorted(external.row_id.dropna().unique().tolist()) if not external.empty else []
    report = {
        "status": "lambda1_tevatron_unitary_validated_lhcb_data_input_closure_remaining",
        "candidate": "Y_unitary = p(qT/Q) * (FO_NLO - W), matched=(1-p)W+p*FO_NLO",
        "baseline": "verified 96-endpoint lambda=1 empirical-reference ensemble",
        "endpoint_recompute": {
            "endpoint_count": endpoint["endpoint_count"],
            "accepted_rows": endpoint["accepted_rows"],
            "boundary_rows": endpoint["boundary_rows"],
            "max_relative_boundary_W_shift_vs_old_source": endpoint[
                "max_relative_boundary_W_shift_vs_old_lambda0p5_source"
            ],
            "all_endpoint_predictions_positive": endpoint["all_endpoint_predictions_positive"],
        },
        "tevatron_fit_impact": fit,
        "tevatron_replica_propagation": replicas,
        "lhcb_external_bin_benchmarks": {
            "rows_with_dyturbo": sorted(dy.row_id.dropna().unique().tolist()) if not dy.empty else [],
            "rows_with_mcfm": external_rows,
            "note": "External fixed-order bin benchmarks are coverage checks only; they do not provide node-level fiducial acceptance weights for the W kernel.",
            "very_high_qT_dyturbo_status": "completed_after_extended_run_for_LHCb_7:13",
            "true_nlo_vj_rows": sorted(load_csv(REPORTS / "lhcb7_external_true_nlo/dyturbo_true_nlo_summary.csv").row_id.dropna().unique().tolist()) if (REPORTS / "lhcb7_external_true_nlo/dyturbo_true_nlo_summary.csv").exists() else [],
        },
        "lhcb_fiducial_node_campaign": {
            "acceptance_grid": grid,
            "w_kernels": kernels,
            "lambda1_recompute": lhcb_recompute,
            "fit_impact_diagnostic": lhcb_fit,
            "external_closure": lhcb_closure,
            "true_nlo_shape_all14": lhcb_true_nlo_shape,
            "true_nlo_scale_scan": lhcb_true_nlo_scale,
            "true_nnlo_scale_scan": lhcb_true_nnlo_scale,
            "nnlo_rapidity_semantics": lhcb_nnlo_rapidity,
            "nnlo_pdf_scan": lhcb_nnlo_pdf,
            "published_pt_covariance_audit": lhcb_covariance,
            "nnlo_convention_scan": lhcb_nnlo_convention,
            "nnlo_unitary_endpoint_fit": lhcb_nnlo_unitary,
            "nnlo_unitary_replica_propagation": lhcb_nnlo_replicas,
        },
        "conventional_y_reassessment": conventional,
        "scope_decision": {
            "tevatron_unitary_candidate": "validated in isolated 24-row scope",
            "universal_lhcb_finite_y": "not approved",
            "reason": "The node-level grid and W kernels now exist and the MCFM/DYTurbo factor-of-two is explained by MCFM's absolute-rapidity convention. The four source rows remain diagnostic-only and LHCb covariance/data normalization are not yet approved. True NLO leaves a large boundary deficit; NNLO raises the prediction substantially, but its scale envelope still remains below data in the first three high-qT bins (and reaches approximately 0.97 only in the last bin). The complete NNLO lambda=1 endpoint/profile fit is positive and convergent, but its central-profile median total chi2/row is about 6.69 with matching and scale pulls about 2.56 and 2.28 sigma. The 50-replica propagation is also convergent but has central-profile median total chi2/row about 7.16. The observable-level closure is therefore not complete.",
            "next_required": "Treat the NNLO lambda=1 endpoint audit and 50-replica propagation as the best isolated LHCb candidate, close the remaining LHCb observable/provenance and covariance/normalization questions, then decide universal promotion; continue using the unitary transition candidate rather than the failed conventional FO-ASY construction. The full 14-bin true-NLO shape, NLO scale scan, NNLO seven-point scale scan, positive-arm rapidity check, NNLO PDF scan, published pT covariance audit, electroweak convention scan, NNLO unitary endpoint fit, and NNLO unitary replica propagation are complete; the optional independent MCFM NLO probe was not converged, so no result justifies a data-driven rescaling.",
            "production_outputs_modified": False,
        },
    }
    out = REPORTS / "lambda1_unitary_decision.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
