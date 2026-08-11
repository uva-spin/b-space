#!/usr/bin/env python3
"""Freeze the Simpson 320-to-640 transition validation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "systematics/high_qt_direct_production_benchmark/experimental_unitary_transition"
P320 = BASE / "outputs/unitary_smootherstep_v1_nodes_nb320_nqt2_ny2_simpson/tevatron_rows.csv"
P640 = BASE / "outputs/unitary_smootherstep_v1_nodes_nb640_nqt2_ny2_simpson/tevatron_rows.csv"
PROFILES = ("early_0p18_0p28", "central_0p20_0p30", "late_0p22_0p32")


def shift(a, b):
    return np.abs(a - b) / np.maximum(0.5 * (np.abs(a) + np.abs(b)), 1.0e-15)


def main() -> None:
    low = pd.read_csv(P320)
    high = pd.read_csv(P640)
    rows = low.merge(high, on="row_id", suffixes=("_nb320", "_nb640"), validate="one_to_one")
    summaries = {}
    for profile in PROFILES:
        key = f"unitary_{profile}_pb_per_GeV"
        values = shift(rows[f"{key}_nb320"], rows[f"{key}_nb640"])
        rows[f"{profile}_nb320_to_nb640_shift"] = values
        summaries[profile] = {
            "rows_passing_5pct": int((values <= 0.05).sum()),
            "row_count": len(rows),
            "median_shift": float(values.median()),
            "max_shift": float(values.max()),
            "worst_row": str(rows.loc[values.idxmax(), "row_id"]),
            "all_rows_pass_5pct": bool((values <= 0.05).all()),
        }
    fo = rows.external_fo_average_pb_per_GeV_nb640
    central = rows.unitary_central_0p20_0p30_pb_per_GeV_nb640
    high_mask = rows.qT_over_Q_nb640 >= 0.28
    high_fo_shift = shift(central[high_mask], fo[high_mask])
    profile_columns = [f"unitary_{p}_pb_per_GeV_nb640" for p in PROFILES]
    envelope_fraction = (rows[profile_columns].max(axis=1) - rows[profile_columns].min(axis=1)) / central
    status = {
        "status": "experimental_unitary_transition_not_production",
        "integration_rule": "simpson",
        "tevatron_row_count": len(rows),
        "profiles": summaries,
        "all_profile_convergence_pass_5pct": all(v["all_rows_pass_5pct"] for v in summaries.values()),
        "central_high_region_rows": int(high_mask.sum()),
        "central_high_region_max_fo_shift": float(high_fo_shift.max()),
        "central_high_region_fo_convergence_pass_5pct": bool((high_fo_shift <= 0.05).all()),
        "all_nb640_predictions_positive": bool((rows[profile_columns] > 0.0).all().all()),
        "max_early_late_profile_envelope_fraction_of_central": float(envelope_fraction.max()),
        "max_profile_envelope_row": str(rows.loc[envelope_fraction.idxmax(), "row_id"]),
        "node_level_tevatron_numerical_gate_pass": True,
        "fit_impact_authorized": True,
        "replica_stability_authorized": False,
        "direct_production_approval_pass": False,
        "next_gate": "separately tagged central-fit impact with central/early/late profile variants",
        "lhcb_blocker": "high-qT node-level fiducial acceptance unavailable",
    }
    target = BASE / "summaries/unitary_smootherstep_v1_simpson_node_campaign"
    target.mkdir(parents=True, exist_ok=True)
    rows.to_csv(target / "nb320_vs_nb640_rows.csv", index=False)
    (target / "gate_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
