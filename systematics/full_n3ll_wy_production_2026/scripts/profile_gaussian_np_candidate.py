#!/usr/bin/env python3
"""Profile the isolated Gaussian-NP Tevatron candidate between g1=0 and 1.

The external DYTurbo grids already provide direct W+Y evaluations at both
endpoints.  This reports the quadratic profile implied by linear interpolation
between those two grids, including a dataset-normalization diagnostic.  It is
an initialization/profile report, not permission to overwrite the frozen
lambda=1 DNN production result; a final promoted fit must evaluate its chosen
g1 directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


SYSTEMATICS = Path(__file__).resolve().parents[2]
BASE = SYSTEMATICS / "full_n3ll_wy_production_2026/reports"
ZERO = BASE / "dyturbo_full_n3ll_nnlo_tevatron_grid/tevatron_full_wy_grid.csv"
ONE = BASE / "dyturbo_full_n3ll_nnlo_tevatron_grid_g1_1p0/tevatron_full_wy_grid.csv"
OUT = BASE / "tevatron_gaussian_np_candidate_profile.json"


def main() -> None:
    z = pd.read_csv(ZERO).set_index("row_id")
    o = pd.read_csv(ONE).set_index("row_id")
    if set(z.index) != set(o.index):
        raise RuntimeError("g1 endpoint grids have different row sets")
    o = o.loc[z.index]
    a = z.full_wy_pb_per_GeV.to_numpy(float)
    b = o.full_wy_pb_per_GeV.to_numpy(float) - a
    y = z.data_pb_per_GeV.to_numpy(float)
    e = z.data_unc_pb_per_GeV.to_numpy(float)
    w = 1.0 / np.square(e)
    q0 = float(np.sum(w * np.square(a - y)))
    q1 = float(np.sum(w * b * (a - y)))
    q2 = float(np.sum(w * np.square(b)))
    g1_hat = float(np.clip(-q1 / q2, 0.0, 2.0))
    chi2_hat = float(q0 + 2.0 * g1_hat * q1 + g1_hat * g1_hat * q2)
    pred = a + g1_hat * b
    groups = {}
    for ds in z.dataset.astype(str).unique():
        mask = z.dataset.astype(str).eq(ds).to_numpy()
        xx, yy, ww = pred[mask], y[mask], w[mask]
        norm = float(np.sum(ww * xx * yy) / np.sum(ww * xx * xx))
        chi2 = float(np.sum(ww * np.square(norm * xx - yy)))
        groups[ds] = {"row_count": int(mask.sum()), "best_norm_unpenalized": norm,
                      "stat_only_chi2": chi2, "stat_only_chi2_per_row": chi2 / int(mask.sum())}
    result = {
        "status": "isolated_gaussian_np_candidate_profile_complete_not_production",
        "endpoint_models": {"g1_zero": str(ZERO), "g1_one": str(ONE)},
        "row_count": int(len(z)),
        "profile_interpolation": "linear in the directly evaluated endpoint predictions; use for initialization only",
        "g1_profile_hat_GeV2": g1_hat,
        "g1_profile_domain_GeV2": [0.0, 1.0],
        "g1_profile_requires_extrapolation_beyond_direct_endpoint": bool(g1_hat > 1.0),
        "g1_one_direct_candidate": {"stat_only_chi2": float(q0 + 2*q1 + q2), "stat_only_chi2_per_row": float((q0 + 2*q1 + q2) / len(z))},
        "g1_profile_hat_stat_only_chi2": chi2_hat,
        "g1_profile_hat_stat_only_chi2_per_row": chi2_hat / len(z),
        "g1_zero_stat_only_chi2_per_row": q0 / len(z),
        "dataset_normalization_diagnostic": groups,
        "interpretation": "g1=1 is the directly evaluated candidate grid; the profiled g1 is not promoted until directly re-evaluated",
        "frozen_baseline_unchanged": True,
        "production_outputs_modified": False,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
