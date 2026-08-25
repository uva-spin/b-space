#!/usr/bin/env python3
"""Compare independent DYTurbo integration seeds for the g1=1 candidate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


SYSTEMATICS = Path(__file__).resolve().parents[2]
BASE = SYSTEMATICS / "full_n3ll_wy_production_2026/reports/dyturbo_full_n3ll_nnlo_tevatron_grid_g1_1p0/tevatron_full_wy_grid.csv"
SEED = SYSTEMATICS / "full_n3ll_wy_production_2026/reports/dyturbo_full_n3ll_nnlo_tevatron_grid_g1_1p0_seed1357911/tevatron_full_wy_grid.csv"
OUT = SYSTEMATICS / "full_n3ll_wy_production_2026/reports/tevatron_external_seed_stability.json"


def main() -> None:
    a = pd.read_csv(BASE).set_index("row_id")
    b = pd.read_csv(SEED).set_index("row_id").loc[a.index]
    diff = b.full_wy_pb_per_GeV.to_numpy(float) - a.full_wy_pb_per_GeV.to_numpy(float)
    denom = np.sqrt(np.square(a.full_wy_unc_pb_per_GeV) + np.square(b.full_wy_unc_pb_per_GeV))
    result = {
        "status": "isolated_external_seed_stability_complete",
        "base_grid": str(BASE), "independent_grid": str(SEED), "row_count": int(len(a)),
        "checks": {
            "median_absolute_relative_difference": float(np.median(np.abs(diff / a.full_wy_pb_per_GeV.to_numpy(float)))),
            "max_absolute_relative_difference": float(np.max(np.abs(diff / a.full_wy_pb_per_GeV.to_numpy(float)))),
            "rms_difference_over_combined_mc_sigma": float(np.sqrt(np.mean(np.square(diff / denom)))),
            "max_difference_over_combined_mc_sigma": float(np.max(np.abs(diff / denom))),
        },
        "interpretation": "independent numerical-integration check only; not a model-start or experimental-replica variation",
        "frozen_baseline_unchanged": True, "production_outputs_modified": False,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
