#!/usr/bin/env python3
"""Profile a 150-member diagonal pseudo-replica layer around direct g1=1.017.

This is an isolated numerical diagnostic for the external unprimed N3LL+NNLO
candidate.  It uses direct grids at g1=0, 1, 1.017, and 1.024191 and
piecewise-linear interpolation only between those grids.  It is not a
replacement for the frozen DNN/F_NP replica ensemble or a formal confidence
band.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


SYSTEMATICS = Path(__file__).resolve().parents[2]
BASE = SYSTEMATICS / "full_n3ll_wy_production_2026/reports"
GRID_PATHS = {
    0.0: BASE / "dyturbo_full_n3ll_nnlo_tevatron_grid/tevatron_full_wy_grid.csv",
    1.0: BASE / "dyturbo_full_n3ll_nnlo_tevatron_grid_g1_1p0/tevatron_full_wy_grid.csv",
    1.017: BASE / "dyturbo_full_n3ll_nnlo_tevatron_grid_g1_1p017_30m/tevatron_full_wy_grid.csv",
    1.0241911542738864: BASE / "dyturbo_full_n3ll_nnlo_tevatron_grid_g1_1p024191_30m/tevatron_full_wy_grid.csv",
}
OUT = BASE / "tevatron_gaussian_np_replica_profile_150_g1_1p017"


def interp(g: float, points: dict[float, np.ndarray]) -> np.ndarray:
    keys = sorted(points)
    if g <= keys[0]:
        return points[keys[0]]
    if g >= keys[-1]:
        return points[keys[-1]]
    hi = next(i for i, key in enumerate(keys) if key >= g)
    lo = hi - 1
    x0, x1 = keys[lo], keys[hi]
    return points[x0] + (g - x0) / (x1 - x0) * (points[x1] - points[x0])


def main() -> None:
    frames = {g: pd.read_csv(path).set_index("row_id") for g, path in GRID_PATHS.items()}
    ids = list(frames[0.0].index)
    for g in frames:
        frames[g] = frames[g].loc[ids]
    points = {g: frame.full_wy_pb_per_GeV.to_numpy(float) for g, frame in frames.items()}
    y = frames[0.0].data_pb_per_GeV.to_numpy(float)
    e = frames[0.0].data_unc_pb_per_GeV.to_numpy(float)
    w = 1.0 / np.square(e)
    g_grid = np.linspace(0.0, 1.2, 1201)
    predictions = np.asarray([interp(float(g), points) for g in g_grid])
    rng = np.random.default_rng(20260819)
    records, curves = [], []
    for member in range(150):
        replica = y + rng.normal(size=len(y)) * e
        chi2 = np.sum(np.square((predictions - replica[None, :]) / e[None, :]), axis=1)
        index = int(np.argmin(chi2))
        g = float(g_grid[index])
        pred = predictions[index]
        records.append({
            "replica": member,
            "g1_GeV2": g,
            "stat_only_chi2_per_row": float(chi2[index] / len(y)),
        })
        curves.append(pred)
    curves = np.asarray(curves)
    central = points[1.017]
    band = pd.DataFrame({
        "row_id": ids,
        "qT_low": frames[0.0].qT_low.to_numpy(float),
        "qT_high": frames[0.0].qT_high.to_numpy(float),
        "central": central,
        "q16": np.quantile(curves, 0.16, axis=0),
        "median": np.quantile(curves, 0.50, axis=0),
        "q84": np.quantile(curves, 0.84, axis=0),
    })
    OUT.mkdir(parents=True, exist_ok=True)
    band.to_csv(OUT / "tevatron_gaussian_np_replica_band.csv", index=False)
    gs = np.asarray([r["g1_GeV2"] for r in records])
    chis = np.asarray([r["stat_only_chi2_per_row"] for r in records])
    result = {
        "status": "isolated_tevatron_gaussian_np_experimental_replica_profile_150_g1_1p017_complete_not_production",
        "replica_count": 150,
        "rng_seed": 20260819,
        "central_g1_GeV2": 1.017,
        "source_models": {str(g): str(path) for g, path in GRID_PATHS.items()},
        "profile": "dense piecewise-linear interpolation across four directly evaluated external grids",
        "g1_distribution_GeV2": {"median": float(np.median(gs)), "q16": float(np.quantile(gs, .16)), "q84": float(np.quantile(gs, .84))},
        "chi2_distribution_per_row": {"median": float(np.median(chis)), "q16": float(np.quantile(chis, .16)), "q84": float(np.quantile(chis, .84))},
        "band_csv": str(OUT / "tevatron_gaussian_np_replica_band.csv"),
        "interpretation": "150 diagonal point-to-point pseudo-replicas centered on direct g1=1.017; no correlated covariance, PDF replicas, or model-form/start envelope",
        "frozen_baseline_unchanged": True,
        "production_outputs_modified": False,
    }
    (OUT / "replica_profile_status.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / "replica_members.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
