#!/usr/bin/env python3
"""Summarize a directly evaluated profiled-g1 Tevatron W+Y grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=Path, required=True)
    ap.add_argument("--g1", type=float, required=True)
    ap.add_argument("--reference-grid", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    d = pd.read_csv(args.grid)
    ref = pd.read_csv(args.reference_grid)
    required = {"row_id", "full_wy_pb_per_GeV", "full_wy_unc_pb_per_GeV", "data_pb_per_GeV", "data_unc_pb_per_GeV"}
    if not required.issubset(d.columns) or len(d) != 122:
        raise RuntimeError("profiled grid is not the expected 122-row Tevatron grid")
    pull = (d.full_wy_pb_per_GeV - d.data_pb_per_GeV) / d.data_unc_pb_per_GeV
    merged = d[["row_id", "full_wy_pb_per_GeV"]].merge(ref[["row_id", "full_wy_pb_per_GeV"]], on="row_id", suffixes=("_profile", "_g1"))
    rel_shift = merged.full_wy_pb_per_GeV_profile / merged.full_wy_pb_per_GeV_g1 - 1.0
    rel_mc = d.full_wy_unc_pb_per_GeV / d.full_wy_pb_per_GeV
    result = {
        "status": "isolated_direct_profiled_gaussian_wy_grid_complete_not_production",
        "grid": str(args.grid), "reference_grid": str(args.reference_grid),
        "g1_GeV2": float(args.g1), "row_count": int(len(d)),
        "checks": {
            "all_finite": bool(d[["full_wy_pb_per_GeV", "full_wy_unc_pb_per_GeV"]].notna().all().all()),
            "all_positive": bool((d.full_wy_pb_per_GeV > 0).all()),
            "median_prediction_to_data": float(d.full_wy_to_data_ratio.median()),
            "stat_only_chi2": float(np.sum(pull * pull)),
            "stat_only_chi2_per_row": float(np.mean(pull * pull)),
            "rms_stat_pull": float(np.sqrt(np.mean(pull * pull))),
            "mean_relative_mc_uncertainty": float(rel_mc.mean()),
            "max_relative_mc_uncertainty": float(rel_mc.max()),
            "median_relative_shift_vs_g1_one": float(np.median(rel_shift)),
            "max_absolute_relative_shift_vs_g1_one": float(np.max(np.abs(rel_shift))),
        },
        "interpretation": "directly evaluated controlled variation around the g1=1 candidate; not a new production fit",
        "promotion_authorized": False,
        "frozen_baseline_unchanged": True,
        "production_outputs_modified": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
