#!/usr/bin/env python3
"""Evaluate added-row pulls at the accepted state before authorizing a refit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "systematics/high_qt_direct_production_benchmark/experimental_unitary_transition"
ROWS = BASE / "outputs/unitary_smootherstep_v1_nodes_nb640_nqt2_ny2_simpson/tevatron_rows.csv"
PRODUCTION = ROOT / "systematics/collins_factorization_validity/outputs/rowidfix_stageFT_all_qmax0p20_lam0p50_central_s303"
PROFILES = ("early_0p18_0p28", "central_0p20_0p30", "late_0p22_0p32")


def main() -> None:
    frame = pd.read_csv(ROWS)
    observed = []
    for dataset in frame.dataset.unique():
        observed.append(pd.read_csv(
            ROOT / "Data/v23a_tevatron_plus_lhcb7_fiducial_candidate" / f"{dataset}.csv"
        )[["row_id", "CS", "error"]])
    frame = frame.merge(pd.concat(observed), on="row_id", validate="one_to_one")
    norms = pd.read_csv(PRODUCTION / "dataset_norms.csv")[["dataset", "norm_scale"]]
    frame = frame.merge(norms, on="dataset", validate="many_to_one")
    summaries = {}
    for profile in PROFILES:
        prediction = frame[f"unitary_{profile}_pb_per_GeV"] * frame.norm_scale
        pull = (prediction - frame.CS) / frame.error
        frame[f"{profile}_accepted_norm_prediction"] = prediction
        frame[f"{profile}_pull"] = pull
        summaries[profile] = {
            "chi2_added_rows": float(np.sum(pull**2)),
            "chi2_per_added_row": float(np.mean(pull**2)),
            "median_absolute_pull": float(np.median(np.abs(pull))),
            "max_absolute_pull": float(np.max(np.abs(pull))),
        }
    profile = frame.profile_central_0p20_0p30
    locked = profile >= 0.95
    central_pull = frame.central_0p20_0p30_pull
    status = {
        "status": "experimental_unitary_transition_not_production",
        "test": "accepted_state_added_row_prefit_impact",
        "row_count": len(frame),
        "accepted_dataset_normalizations_applied": True,
        "summaries": summaries,
        "central_rows_with_profile_ge_0p95": int(locked.sum()),
        "central_locked_subset_chi2": float(np.sum(central_pull[locked] ** 2)),
        "central_locked_subset_chi2_per_row": float(np.mean(central_pull[locked] ** 2)),
        "frozen_state_fit_impact_pass": False,
        "central_refit_authorized": False,
        "replica_stability_authorized": False,
        "direct_production_approval_pass": False,
        "decision": "reject_current_fo_normalization_before_refit",
        "reason": "Rows with profile near one are fixed-order dominated and cannot be repaired by refitting the nonperturbative W factor; their pulls are already excessive.",
        "required_next_experiment": "establish data-level fixed-order normalization/order/electroweak closure or a justified external K-factor with scale uncertainty before revisiting fits",
    }
    target = BASE / "summaries/unitary_smootherstep_v1_frozen_fit_impact"
    target.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target / "added_row_pulls.csv", index=False)
    (target / "gate_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
