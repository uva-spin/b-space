#!/usr/bin/env python3
"""Summarize the first direct Gaussian-NP full-N3LL+NNLO Tevatron candidate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


SYSTEMATICS = Path(__file__).resolve().parents[2]
BASE = SYSTEMATICS / "full_n3ll_wy_production_2026/reports"
GRID = BASE / "dyturbo_full_n3ll_nnlo_tevatron_grid_g1_1p0/tevatron_full_wy_grid.csv"
BOUNDARY = BASE / "dyturbo_full_n3ll_nnlo_boundary_g1_1p0/boundary_full_wy_status.json"
PROFILE = BASE / "tevatron_gaussian_np_candidate_profile.json"
OUT = BASE / "tevatron_n3ll_nnlo_candidate_fit_status.json"


def main() -> None:
    d = pd.read_csv(GRID)
    pull = (d.full_wy_pb_per_GeV - d.data_pb_per_GeV) / d.data_unc_pb_per_GeV
    status = json.loads(BOUNDARY.read_text())
    profile = json.loads(PROFILE.read_text())
    result = {
        "status": "isolated_tevatron_full_unprimed_n3ll_nnlo_gaussian_np_candidate_complete_not_production",
        "candidate_id": "tevatron_n3ll_nnlo_wy_gaussian_g1_1p0",
        "engine": "/home/dustin/src/dyturbo-1.4.2/bin/dyturbo",
        "accuracy": {"resummation": "N3LL unprimed", "fixed_order": "NNLO", "matching": "W + (FO_NNLO - ASY_NNLO)", "primed": False},
        "nonperturbative_model": "DYTurbo npff=0 Gaussian exp[-g1 bT^2]",
        "g1_GeV2": 1.0,
        "row_count": int(len(d)),
        "direct_grid": str(GRID),
        "fit_diagnostics": {
            "stat_only_chi2": float(np.sum(np.square(pull))),
            "stat_only_chi2_per_row": float(np.mean(np.square(pull))),
            "rms_stat_pull": float(np.sqrt(np.mean(np.square(pull)))),
            "median_prediction_to_data": float(d.full_wy_to_data_ratio.median()),
            "mean_prediction_to_data": float(d.full_wy_to_data_ratio.mean()),
            "mean_relative_mc_uncertainty": float(np.mean(d.full_wy_unc_pb_per_GeV / d.full_wy_pb_per_GeV)),
            "max_relative_mc_uncertainty": float(np.max(d.full_wy_unc_pb_per_GeV / d.full_wy_pb_per_GeV)),
        },
        "boundary_oracle": status,
        "profile_reference": profile,
        "stationarity": {"status": "not_applicable_to_one_parameter_external_engine_candidate", "required_next": "repeat direct evaluation at independent seeds and profile candidate parameters"},
        "uncertainty_propagation": {"status": "not_started", "required": "experimental replicas and model/start variations"},
        "promotion_authorized": False,
        "frozen_baseline_unchanged": True,
        "production_outputs_modified": False,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
