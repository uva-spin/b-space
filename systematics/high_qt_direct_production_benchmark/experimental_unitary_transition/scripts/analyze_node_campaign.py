#!/usr/bin/env python3
"""Compare node campaigns and freeze unitary-transition gate decisions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "systematics/high_qt_direct_production_benchmark/experimental_unitary_transition"
OUT = BASE / "outputs"
P160 = OUT / "unitary_smootherstep_v1_nodes_nb160_nqt2_ny2/tevatron_rows.csv"
P320 = OUT / "unitary_smootherstep_v1_nodes_nb320_nqt2_ny2/tevatron_rows.csv"
PROFILES = ("early_0p18_0p28", "central_0p20_0p30", "late_0p22_0p32")


def symmetric_shift(a, b):
    return np.abs(a - b) / np.maximum(0.5 * (np.abs(a) + np.abs(b)), 1.0e-15)


def main() -> None:
    a = pd.read_csv(P160)
    b = pd.read_csv(P320)
    frame = a.merge(b, on="row_id", suffixes=("_nb160", "_nb320"), validate="one_to_one")
    summaries = {}
    for profile in PROFILES:
        key = f"unitary_{profile}_pb_per_GeV"
        shifts = symmetric_shift(frame[f"{key}_nb160"], frame[f"{key}_nb320"])
        frame[f"{profile}_nb160_to_nb320_shift"] = shifts
        summaries[profile] = {
            "rows_passing_5pct": int((shifts <= 0.05).sum()),
            "row_count": len(frame),
            "median_shift": float(shifts.median()),
            "max_shift": float(shifts.max()),
            "worst_row": str(frame.loc[shifts.idxmax(), "row_id"]),
            "all_rows_pass_5pct": bool((shifts <= 0.05).all()),
        }
    # An already completed independent first-row calculation supplies the
    # higher-resolution boundary check, where p is essentially zero.
    boundary_320_to_640 = 0.08236670395358807
    status = {
        "status": "experimental_unitary_transition_not_production",
        "tevatron_row_count": len(frame),
        "profiles": summaries,
        "first_boundary_fitted_w_nb320_to_nb640_shift": boundary_320_to_640,
        "first_boundary_5pct_pass": boundary_320_to_640 <= 0.05,
        "node_level_row_coverage_pass": len(frame) == 24,
        "component_finiteness_pass": bool(frame.select_dtypes("number").notna().all().all()),
        "b_grid_convergence_pass": False,
        "fit_impact_authorized": False,
        "replica_stability_authorized": False,
        "direct_production_approval_pass": False,
        "decision": "stop_before_fit_and_replace_oscillatory_bessel_transform",
        "required_next_experiment": "validated oscillatory Hankel/Bessel quadrature with boundary-row convergence below 5% while preserving qT/Q<=0.20 production references",
        "lhcb_blocker": "high-qT node-level fiducial acceptance unavailable",
    }
    target = BASE / "summaries/unitary_smootherstep_v1_node_campaign"
    target.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target / "nb160_vs_nb320_rows.csv", index=False)
    (target / "gate_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
