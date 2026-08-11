#!/usr/bin/env python3
"""Analyze the 24-row central NLO campaign and frozen unitary fit impact."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "systematics/high_qt_direct_production_benchmark/experimental_unitary_transition"
NLO = BASE / "outputs/mcfm_zjet_nlo_24row_central_500k_i10/mcfm_benchmark_summary.csv"
NODES = BASE / "outputs/unitary_smootherstep_v1_nodes_nb640_nqt2_ny2_simpson/tevatron_rows.csv"
LO = ROOT / "systematics/high_qt_direct_production_benchmark/summaries/tier1_boundary/central/external_pairs.csv"
PRODUCTION = ROOT / "systematics/collins_factorization_validity/outputs/rowidfix_stageFT_all_qmax0p20_lam0p50_central_s303"
TARGET = BASE / "summaries/unitary_smootherstep_v1_nlo_24row"
PROFILES = ("early_0p18_0p28", "central_0p20_0p30", "late_0p22_0p32")


def main() -> None:
    nlo = pd.read_csv(NLO)
    lo = pd.read_csv(LO)[["row_id", "mcfm_pb_per_GeV"]].rename(
        columns={"mcfm_pb_per_GeV": "mcfm_lo_pb_per_GeV"}
    )
    frame = nlo.merge(lo, on="row_id", validate="one_to_one")
    frame["nlo_over_lo_k"] = frame.mcfm_pb_per_GeV / frame.mcfm_lo_pb_per_GeV
    frame["data_over_nlo"] = frame.data_pb_per_GeV / frame.mcfm_pb_per_GeV
    frame["relative_nlo_mc_uncertainty"] = frame.mcfm_pb_per_GeV_unc / frame.mcfm_pb_per_GeV

    nodes = pd.read_csv(NODES).drop(columns=["mcfm_pb_per_GeV"])
    impact = nodes.merge(
        frame[["row_id", "mcfm_pb_per_GeV", "mcfm_pb_per_GeV_unc"]],
        on="row_id", validate="one_to_one",
    ).rename(columns={
        "mcfm_pb_per_GeV": "mcfm_nlo_pb_per_GeV",
        "mcfm_pb_per_GeV_unc": "mcfm_nlo_mc_unc_pb_per_GeV",
    })
    observed = []
    for dataset in impact.dataset.unique():
        observed.append(pd.read_csv(
            ROOT / "Data/v23a_tevatron_plus_lhcb7_fiducial_candidate" / f"{dataset}.csv"
        )[["row_id", "CS", "error"]])
    impact = impact.merge(pd.concat(observed), on="row_id", validate="one_to_one")
    norms = pd.read_csv(PRODUCTION / "dataset_norms.csv")[["dataset", "norm_scale"]]
    impact = impact.merge(norms, on="dataset", validate="many_to_one")

    summaries = {}
    for profile in PROFILES:
        weight = impact[f"profile_{profile}"]
        unitary = (1.0 - weight) * impact.w_fitted_pb_per_GeV + weight * impact.mcfm_nlo_pb_per_GeV
        prediction = unitary * impact.norm_scale
        pull = (prediction - impact.CS) / impact.error
        impact[f"nlo_unitary_{profile}_pb_per_GeV"] = unitary
        impact[f"{profile}_accepted_norm_prediction"] = prediction
        impact[f"{profile}_pull"] = pull
        summaries[profile] = {
            "chi2_added_rows": float(np.sum(pull**2)),
            "chi2_per_added_row": float(np.mean(pull**2)),
            "median_absolute_pull": float(np.median(np.abs(pull))),
            "max_absolute_pull": float(np.max(np.abs(pull))),
        }

    locked = impact.profile_central_0p20_0p30 >= 0.95
    central_pull = impact.central_0p20_0p30_pull
    status = {
        "status": "experimental_unitary_transition_not_production",
        "test": "full_24row_central_genuine_nlo_and_frozen_fit_impact",
        "row_count": int(len(frame)),
        "all_rows_complete": bool(len(frame) == 24),
        "nlo_over_lo_k_range": [float(frame.nlo_over_lo_k.min()), float(frame.nlo_over_lo_k.max())],
        "data_over_central_nlo_range": [float(frame.data_over_nlo.min()), float(frame.data_over_nlo.max())],
        "median_data_over_central_nlo": float(frame.data_over_nlo.median()),
        "maximum_relative_nlo_mc_uncertainty": float(frame.relative_nlo_mc_uncertainty.max()),
        "accepted_dataset_normalizations_applied": True,
        "frozen_impact_summaries": summaries,
        "central_rows_with_profile_ge_0p95": int(locked.sum()),
        "central_locked_subset_chi2": float(np.sum(central_pull[locked] ** 2)),
        "central_locked_subset_chi2_per_row": float(np.mean(central_pull[locked] ** 2)),
        "full_24_row_nlo_campaign_complete": True,
        "frozen_state_fit_impact_pass": False,
        "central_refit_authorized": False,
        "replica_stability_authorized": False,
        "decision": "retain_fit_block_after_central_nlo_campaign",
        "reason": "Central NLO improves normalization but leaves excessive frozen transition-region and fixed-order-locked pulls.",
        "required_next_experiment": "construct a correlated NLO scale-uncertainty treatment across the 24 rows before reconsidering a central refit",
    }
    TARGET.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TARGET / "mcfm_nlo_24row_audit.csv", index=False)
    impact.to_csv(TARGET / "nlo_frozen_unitary_pulls.csv", index=False)
    (TARGET / "gate_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
