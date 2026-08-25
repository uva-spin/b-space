#!/usr/bin/env python3
"""Compare the isolated high-statistics and independent-seed W+Y grids."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", required=True)
    ap.add_argument("--seed", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    a = pd.read_csv(args.primary).set_index("row_id")
    b = pd.read_csv(args.seed).set_index("row_id").loc[a.index]
    if len(a) != len(b) or not a.index.equals(b.index):
        raise SystemExit("stationarity grids have different row identities")
    pa = a.full_wy_pb_per_GeV.to_numpy(float)
    pb = b.full_wy_pb_per_GeV.to_numpy(float)
    ea = a.full_wy_unc_pb_per_GeV.to_numpy(float)
    eb = b.full_wy_unc_pb_per_GeV.to_numpy(float)
    diff = pb - pa
    denom = np.sqrt(np.square(ea) + np.square(eb))
    result = {
        "status": "isolated_tevatron_n3ll_nnlo_seed_stationarity_complete_not_promoted",
        "primary_grid": str(Path(args.primary).resolve()),
        "independent_seed_grid": str(Path(args.seed).resolve()),
        "row_count": int(len(a)),
        "checks": {
            "all_finite": bool(np.isfinite(np.r_[pa, pb, ea, eb]).all()),
            "all_positive": bool((pa > 0).all() and (pb > 0).all()),
            "median_absolute_relative_difference": float(np.median(np.abs(diff / pa))),
            "mean_absolute_relative_difference": float(np.mean(np.abs(diff / pa))),
            "max_absolute_relative_difference": float(np.max(np.abs(diff / pa))),
            "rms_difference_over_combined_mc_sigma": float(np.sqrt(np.mean(np.square(diff / denom)))),
            "max_difference_over_combined_mc_sigma": float(np.max(np.abs(diff / denom))),
            "primary_mean_relative_mc_uncertainty": float(np.mean(ea / pa)),
            "independent_mean_relative_mc_uncertainty": float(np.mean(eb / pb)),
        },
        "interpretation": "independent numerical-integration stationarity only; not F_NP/start, PDF, or experimental propagation",
        "frozen_baseline_unchanged": True,
        "production_outputs_modified": False,
        "promotion_authorized": False,
    }
    out = Path(args.out)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
