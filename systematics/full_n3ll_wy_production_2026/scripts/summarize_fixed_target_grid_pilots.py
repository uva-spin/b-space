#!/usr/bin/env python3
"""Summarize isolated fixed-target W+Y convention pilots."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
REPORT = BASE / "reports/fixed_target_n3ll_nnlo_grid_pilot_summary.json"
PILOTS = {
    "E288_200": BASE / "reports/fixed_target_grid_pilot_e288200_30M/fixed_target_full_wy_grid.csv",
    "E288_300": BASE / "reports/fixed_target_grid_pilot_e288300_30M/fixed_target_full_wy_grid.csv",
    "E288_400": BASE / "reports/fixed_target_grid_pilot_e288400_30M/fixed_target_full_wy_grid.csv",
    "E605": BASE / "reports/fixed_target_grid_pilot_e605_30M/fixed_target_full_wy_grid.csv",
    "E772": BASE / "reports/fixed_target_grid_pilot_e772_30M/fixed_target_full_wy_grid.csv",
}


def main() -> None:
    rows = {}
    for dataset, path in PILOTS.items():
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        row = frame.iloc[0]
        rows[dataset] = {
            "row_id": str(row.row_id),
            "calls": 30_000_000,
            "per_nucleon_value_fb_per_bin": float(row.per_nucleon_full_wy_fb_per_bin),
            "per_nucleon_uncertainty_fb_per_bin": float(row.per_nucleon_unc_fb_per_bin),
            "data_CS_A_over_PreFactor": float(row.data_CS_A_over_PreFactor),
            "ratio_to_data": float(row.per_nucleon_to_data_ratio),
            "target_Z": float(row.target_Z),
            "target_A": float(row.target_A),
        }
    payload = {
        "status": "isolated_fixed_target_wy_pilot_convention_not_closed_not_production",
        "formula": "W_N3LL + (FO_NNLO - ASY_NNLO)",
        "runtime": "unprimed order=3; W=RES; ASY=-CT; FO=VJ; Y=VJ+CT",
        "observable_comparison": "DYTurbo raw fb/bin per nucleus divided by target A versus fit-ready CS=A/PreFactor",
        "pilots": rows,
        "finding": "E605 and the diagnostic E772 point are near 10-13% high, while the three Be E288 points span 0.42-4.75 times data; a single target normalization factor is therefore not established",
        "decision": "do not run or promote the 243-row fixed-target grid until the E288 observable/bin/target convention is resolved",
        "production_authorized": False,
        "frozen_baseline_unchanged": True,
        "production_outputs_modified": False,
    }
    REPORT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
