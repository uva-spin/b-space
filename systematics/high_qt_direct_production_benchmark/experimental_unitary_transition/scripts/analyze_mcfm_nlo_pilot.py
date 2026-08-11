#!/usr/bin/env python3
"""Summarize the controlled MCFM Z+jet NLO convergence pilot."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "systematics/high_qt_direct_production_benchmark"
RUNS = BASE / "experimental_unitary_transition/outputs/mcfm_zjet_nlo_order_audit"
OUT = BASE / "experimental_unitary_transition/summaries/unitary_smootherstep_v1_fo_order_audit"
LO = BASE / "summaries/tier1_boundary/central/external_pairs.csv"
SELECTED = {
    "500k_i5": "cdf_run_2_51_controlled_500k_i5",
    "750k_i5": "cdf_run_2_51_controlled_750k_i5",
    "1m_i5": "cdf_run_2_51_controlled_1m_i5",
    "500k_i10": "cdf_run_2_51_controlled_500k_i10",
}


def main() -> None:
    lo = pd.read_csv(LO).set_index("row_id").loc["CDF_RUN_2:51"]
    records = []
    for label, directory in SELECTED.items():
        row = pd.read_csv(RUNS / directory / "mcfm_benchmark_summary.csv").iloc[0]
        records.append({
            "label": label,
            "mcfm_nlo_pb_per_GeV": row.mcfm_pb_per_GeV,
            "mcfm_nlo_mc_unc_pb_per_GeV": row.mcfm_pb_per_GeV_unc,
            "relative_mc_uncertainty": row.mcfm_pb_per_GeV_unc / row.mcfm_pb_per_GeV,
            "k_over_mcfm_lo": row.mcfm_pb_per_GeV / lo.mcfm_pb_per_GeV,
            "data_over_nlo": row.data_pb_per_GeV / row.mcfm_pb_per_GeV,
        })
    table = pd.DataFrame(records)
    a = table.set_index("label").loc["1m_i5"]
    b = table.set_index("label").loc["500k_i10"]
    shift = abs(a.mcfm_nlo_pb_per_GeV - b.mcfm_nlo_pb_per_GeV) / (
        0.5 * (a.mcfm_nlo_pb_per_GeV + b.mcfm_nlo_pb_per_GeV)
    )
    combined_mc = (a.mcfm_nlo_mc_unc_pb_per_GeV**2 + b.mcfm_nlo_mc_unc_pb_per_GeV**2) ** 0.5
    data_error = 0.078102497
    data_gap_sigma = (lo.data_pb_per_GeV - b.mcfm_nlo_pb_per_GeV) / (
        data_error**2 + b.mcfm_nlo_mc_unc_pb_per_GeV**2
    ) ** 0.5
    status = {
        "status": "experimental_unitary_transition_not_production",
        "test": "controlled_genuine_mcfm_zjet_nlo_pilot",
        "dataset": "CDF_RUN_2",
        "row_id": "CDF_RUN_2:51",
        "nlo_order_verified": True,
        "reference_result": {
            "configuration": "500k calls/component, 2 warmup iterations, 10 production iterations",
            "pb_per_GeV": float(b.mcfm_nlo_pb_per_GeV),
            "mc_unc_pb_per_GeV": float(b.mcfm_nlo_mc_unc_pb_per_GeV),
            "relative_mc_uncertainty": float(b.relative_mc_uncertainty),
        },
        "independent_convergence_result": {
            "configuration": "1m calls/component, 2 warmup iterations, 5 production iterations",
            "pb_per_GeV": float(a.mcfm_nlo_pb_per_GeV),
            "mc_unc_pb_per_GeV": float(a.mcfm_nlo_mc_unc_pb_per_GeV),
        },
        "convergence_relative_shift": float(shift),
        "convergence_difference_over_combined_mc_uncertainty": float(
            abs(a.mcfm_nlo_pb_per_GeV - b.mcfm_nlo_pb_per_GeV) / combined_mc
        ),
        "convergence_pass": bool(shift < 0.05),
        "mcfm_lo_pb_per_GeV": float(lo.mcfm_pb_per_GeV),
        "nlo_over_lo_k": float(b.k_over_mcfm_lo),
        "data_pb_per_GeV": float(lo.data_pb_per_GeV),
        "data_over_nlo": float(b.data_over_nlo),
        "data_minus_nlo_sigma_including_data_and_mc_uncertainty": float(data_gap_sigma),
        "interpretation": (
            "Genuine Z+jet NLO explains most of the LO deficit but does not close the selected "
            "fixed-order-dominated row; representative-row and NLO scale studies are required."
        ),
        "central_refit_authorized": False,
        "replica_stability_authorized": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "mcfm_nlo_pilot_convergence.csv", index=False)
    (OUT / "nlo_pilot_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
