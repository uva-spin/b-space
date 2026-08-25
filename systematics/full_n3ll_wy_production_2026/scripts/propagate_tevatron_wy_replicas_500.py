#!/usr/bin/env python3
"""Propagate Tevatron point-to-point and normalization replicas through the
isolated external Gaussian W+Y candidate.

This is the next uncertainty layer after the direct high-statistics grid.  It
profiles the single Gaussian ``g1`` parameter over directly evaluated grids;
the output is deliberately marked diagnostic until PDF, F_NP/start, and
model-form layers are added.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SYSTEMATICS = Path(__file__).resolve().parents[2]
DATA_ROOT = SYSTEMATICS.parent / "Data/v23a_tevatron_plus_lhcb7_fiducial_candidate"
BASE = SYSTEMATICS / "full_n3ll_wy_production_2026/reports"


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary-grid", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--replicas", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260819)
    args = ap.parse_args()
    source = {
        0.0: BASE / "dyturbo_full_n3ll_nnlo_tevatron_grid/tevatron_full_wy_grid.csv",
        1.0: BASE / "dyturbo_full_n3ll_nnlo_tevatron_grid_g1_1p0/tevatron_full_wy_grid.csv",
        1.017: Path(args.primary_grid),
        1.0241911542738864: BASE / "dyturbo_full_n3ll_nnlo_tevatron_grid_g1_1p024191_30m/tevatron_full_wy_grid.csv",
    }
    frames = {g: pd.read_csv(path).set_index("row_id") for g, path in source.items()}
    ids = list(frames[1.017].index)
    for g in frames:
        frames[g] = frames[g].loc[ids]
    raw = pd.concat([pd.read_csv(DATA_ROOT / f"{ds}.csv") for ds in ("CDF_RUN_1", "CDF_RUN_2", "D0_RUN_1")], ignore_index=True).set_index("row_id").loc[ids]
    points = {g: frame.full_wy_pb_per_GeV.to_numpy(float) for g, frame in frames.items()}
    y = raw.CS.to_numpy(float)
    e = raw.error.to_numpy(float)
    groups = raw.dataset.astype(str).to_numpy()
    norm_sigma = np.array([float(raw.loc[row_id, "sysNorm_rel"]) for row_id in ids])
    g_grid = np.linspace(0.0, 1.20, 1201)
    predictions = np.asarray([interp(float(g), points) for g in g_grid])
    rng = np.random.default_rng(args.seed)
    records, curves = [], []
    for member in range(args.replicas):
        norm_draws = {ds: float(rng.normal() * norm_sigma[groups == ds][0]) for ds in np.unique(groups)}
        replica = y * np.asarray([1.0 + norm_draws[ds] for ds in groups]) + rng.normal(size=len(y)) * e
        chi2 = np.empty(len(g_grid), dtype=float)
        for j, pred in enumerate(predictions):
            total = 0.0
            for ds in np.unique(groups):
                mask = groups == ds
                err = e[mask]
                s = norm_sigma[mask][0]
                p = pred[mask]
                yy = replica[mask]
                w = 1.0 / np.square(err)
                a = float(np.sum(w * p * p) + 1.0 / (s * s))
                b = float(np.sum(w * p * (p - yy)))
                nuisance = -b / a
                total += float(np.sum(w * (p * (1.0 + nuisance) - yy) ** 2) + (nuisance / s) ** 2)
            chi2[j] = total
        index = int(np.argmin(chi2))
        records.append({"replica": member, "g1_GeV2": float(g_grid[index]), "chi2_per_row": float(chi2[index] / len(y)), "normalization_draws": norm_draws})
        curves.append(predictions[index])
    curves = np.asarray(curves)
    central = points[1.017]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    band = pd.DataFrame({
        "row_id": ids,
        "qT_low": frames[1.017].qT_low.to_numpy(float),
        "qT_high": frames[1.017].qT_high.to_numpy(float),
        "central": central,
        "q16": np.quantile(curves, .16, axis=0),
        "median": np.quantile(curves, .50, axis=0),
        "q84": np.quantile(curves, .84, axis=0),
    })
    band.to_csv(out / "tevatron_wy_replica_band.csv", index=False)
    gs = np.asarray([r["g1_GeV2"] for r in records])
    chis = np.asarray([r["chi2_per_row"] for r in records])
    result = {
        "status": "isolated_tevatron_n3ll_nnlo_wy_replica_propagation_complete_not_production",
        "replica_count": int(args.replicas),
        "rng_seed": int(args.seed),
        "central_g1_GeV2": 1.017,
        "source_models": {str(g): str(path) for g, path in source.items()},
        "profile": "dense piecewise-linear interpolation across four direct Gaussian grids; normalization nuisance profiled per dataset",
        "g1_distribution_GeV2": {"median": float(np.median(gs)), "q16": float(np.quantile(gs, .16)), "q84": float(np.quantile(gs, .84))},
        "chi2_distribution_per_row": {"median": float(np.median(chis)), "q16": float(np.quantile(chis, .16)), "q84": float(np.quantile(chis, .84))},
        "band_csv": str(out / "tevatron_wy_replica_band.csv"),
        "included": ["diagonal point-to-point errors", "one Gaussian normalization nuisance per Tevatron dataset"],
        "missing": ["PDF replicas", "F_NP/start ensemble", "model-form envelope", "scale envelope"],
        "formal_confidence_level_assigned": False,
        "frozen_baseline_unchanged": True,
        "production_outputs_modified": False,
        "promotion_authorized": False,
    }
    (out / "replica_profile_status.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "replica_members.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
