#!/usr/bin/env python3
"""Propagate Tevatron point-to-point errors through the Gaussian-NP profile.

This is the first uncertainty layer for the isolated external candidate. It
uses the two directly evaluated endpoint grids and profiles one common g1 for
each pseudo-data replica. It is deliberately separate from the frozen DNN
replica package and does not claim to replace correlated experimental
covariances or model-form uncertainty.
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
OUT = BASE / "tevatron_gaussian_np_replica_profile"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    z = pd.read_csv(ZERO).set_index("row_id")
    o = pd.read_csv(ONE).set_index("row_id").loc[z.index]
    p0 = z.full_wy_pb_per_GeV.to_numpy(float)
    p1 = o.full_wy_pb_per_GeV.to_numpy(float)
    delta = p1 - p0
    y0 = z.data_pb_per_GeV.to_numpy(float)
    e = z.data_unc_pb_per_GeV.to_numpy(float)
    w = 1.0 / np.square(e)
    rng = np.random.default_rng(20260818)
    replicas = []
    curves = []
    for member in range(50):
        y = y0 + rng.normal(size=len(y0)) * e
        q0 = np.sum(w * np.square(p0 - y))
        q1 = np.sum(w * delta * (p0 - y))
        q2 = np.sum(w * np.square(delta))
        g1 = float(np.clip(-q1 / q2, 0.0, 2.0))
        pred = p0 + g1 * delta
        replicas.append({"replica": member, "g1_GeV2": g1, "stat_only_chi2_per_row": float(np.mean(np.square((pred - y) / e)))})
        curves.append(pred)
    curves = np.asarray(curves)
    band = pd.DataFrame({"row_id": z.index.astype(str), "qT_low": z.qT_low.to_numpy(float), "qT_high": z.qT_high.to_numpy(float), "central": p0 + delta, "q16": np.quantile(curves, 0.16, axis=0), "median": np.quantile(curves, 0.50, axis=0), "q84": np.quantile(curves, 0.84, axis=0)})
    band.to_csv(OUT / "tevatron_gaussian_np_replica_band.csv", index=False)
    result = {
        "status": "isolated_tevatron_gaussian_np_experimental_replica_profile_complete_not_production",
        "replica_count": 50,
        "rng_seed": 20260818,
        "source_models": {"g1_zero": str(ZERO), "g1_one": str(ONE)},
        "profile": "one common g1 profiled by quadratic interpolation between direct endpoint grids",
        "g1_distribution_GeV2": {"median": float(np.median([r["g1_GeV2"] for r in replicas])), "q16": float(np.quantile([r["g1_GeV2"] for r in replicas], 0.16)), "q84": float(np.quantile([r["g1_GeV2"] for r in replicas], 0.84))},
        "chi2_distribution_per_row": {"median": float(np.median([r["stat_only_chi2_per_row"] for r in replicas])), "q16": float(np.quantile([r["stat_only_chi2_per_row"] for r in replicas], 0.16)), "q84": float(np.quantile([r["stat_only_chi2_per_row"] for r in replicas], 0.84))},
        "band_csv": str(OUT / "tevatron_gaussian_np_replica_band.csv"),
        "interpretation": "point-to-point experimental pseudo-replica propagation only; no correlated covariance, PDF replicas, or model-form envelope",
        "frozen_baseline_unchanged": True, "production_outputs_modified": False,
    }
    (OUT / "replica_profile_status.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / "replica_members.json").write_text(json.dumps(replicas, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
