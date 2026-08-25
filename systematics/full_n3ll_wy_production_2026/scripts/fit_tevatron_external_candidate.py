#!/usr/bin/env python3
"""Profile the isolated external W+Y Gaussian candidate with released covariance.

The Tevatron tables encode diagonal point-to-point errors plus one correlated
normalization nuisance per dataset.  This script profiles those nuisances and
the Gaussian g1 over the directly evaluated g1=0, 1, and 1.024191 grids.  It
is an observable-level fit diagnostic; it is not a replacement for the frozen
DNN/F_NP production extraction.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


SYSTEMATICS = Path(__file__).resolve().parents[2]
BASE = SYSTEMATICS / "full_n3ll_wy_production_2026/reports"
DATA_ROOT = SYSTEMATICS.parent / "Data/v23a_tevatron_plus_lhcb7_fiducial_candidate"
GRID0 = BASE / "dyturbo_full_n3ll_nnlo_tevatron_grid/tevatron_full_wy_grid.csv"
GRID1 = BASE / "dyturbo_full_n3ll_nnlo_tevatron_grid_g1_1p0/tevatron_full_wy_grid.csv"
GRID2 = BASE / "dyturbo_full_n3ll_nnlo_tevatron_grid_g1_1p024191_30m/tevatron_full_wy_grid.csv"
GRID17 = BASE / "dyturbo_full_n3ll_nnlo_tevatron_grid_g1_1p017_30m/tevatron_full_wy_grid.csv"
OUT = BASE / "tevatron_external_candidate_correlated_fit.json"


def load(path: Path, ids: list[str] | None = None) -> pd.DataFrame:
    d = pd.read_csv(path).set_index("row_id")
    return d if ids is None else d.loc[ids]


def prediction(g: float, p0: np.ndarray, p1: np.ndarray, p17: np.ndarray, p2: np.ndarray) -> np.ndarray:
    # Piecewise linear interpolation is used only between directly evaluated
    # external grids; no fitted DNN output is synthesized or promoted.
    if g <= 1.0:
        return p0 + g * (p1 - p0)
    if g <= 1.017:
        return p1 + (g - 1.0) / 0.017 * (p17 - p1)
    return p17 + (g - 1.017) / (1.0241911542738864 - 1.017) * (p2 - p17)


def profile_norm(pred: np.ndarray, data: pd.DataFrame) -> tuple[float, float, float]:
    err = data.error.to_numpy(float)
    y = data.CS.to_numpy(float)
    s = float(data.sysNorm_rel.iloc[0])
    w = 1.0 / np.square(err)
    a = float(np.sum(w * pred * pred) + 1.0 / (s * s))
    b = float(np.sum(w * pred * (pred - y)))
    nuisance = -b / a
    chi2 = float(np.sum(w * (pred * (1.0 + nuisance) - y) ** 2) + (nuisance / s) ** 2)
    return nuisance, chi2, float(np.sum(w * (pred * (1.0 + nuisance) - y) ** 2))


def main() -> None:
    g0 = load(GRID0)
    g1 = load(GRID1, list(g0.index))
    g17 = load(GRID17, list(g0.index))
    g2 = load(GRID2, list(g0.index))
    raw = pd.concat([pd.read_csv(DATA_ROOT / f"{ds}.csv") for ds in ("CDF_RUN_1", "CDF_RUN_2", "D0_RUN_1")], ignore_index=True).set_index("row_id")
    data = raw.loc[g0.index]
    p0 = g0.full_wy_pb_per_GeV.to_numpy(float)
    p1 = g1.full_wy_pb_per_GeV.to_numpy(float)
    p17 = g17.full_wy_pb_per_GeV.to_numpy(float)
    p2 = g2.full_wy_pb_per_GeV.to_numpy(float)
    grid = np.linspace(0.0, 1.20, 1201)
    rows = []
    for g in grid:
        p = prediction(float(g), p0, p1, p17, p2)
        nuis = {}
        chi2 = 0.0
        diag = 0.0
        for ds in data.dataset.astype(str).unique():
            mask = data.dataset.astype(str).eq(ds).to_numpy()
            n, c, d = profile_norm(p[mask], data.iloc[np.flatnonzero(mask)])
            nuis[ds] = n
            chi2 += c
            diag += d
        rows.append({"g1_GeV2": float(g), "chi2": chi2, "chi2_per_row": chi2 / len(data), "diag_chi2": diag, "normalization_nuisances": nuis})
    best = min(rows, key=lambda r: r["chi2"])
    pbest = prediction(best["g1_GeV2"], p0, p1, p17, p2)
    dataset_rows = {}
    for ds in data.dataset.astype(str).unique():
        mask = data.dataset.astype(str).eq(ds).to_numpy()
        n, c, d = profile_norm(pbest[mask], data.iloc[np.flatnonzero(mask)])
        dataset_rows[ds] = {"row_count": int(mask.sum()), "normalization_nuisance": n, "chi2": c, "chi2_per_row": c / int(mask.sum()), "diagonal_point_to_point_chi2": d}
    result = {
        "status": "isolated_tevatron_external_n3ll_nnlo_correlated_candidate_fit_complete_not_production",
        "engine": "/home/dustin/src/dyturbo-1.4.2/bin/dyturbo",
        "convention": "unprimed order=3; W=RES, ASY=-CT, FO=VJ, Y=FO-ASY=VJ+CT",
        "scope": {"datasets": ["CDF_RUN_1", "CDF_RUN_2", "D0_RUN_1"], "row_count": int(len(data))},
        "profiled_g1_GeV2": best["g1_GeV2"],
        "profiled_chi2": best["chi2"],
        "profiled_chi2_per_row": best["chi2_per_row"],
        "profile_grid": [0.0, 1.0, 1.017, 1.0241911542738864],
        "profile_method": "piecewise-linear interpolation between directly evaluated external grids (including direct g1=1.017 refinement); one released normalization nuisance per dataset with Gaussian penalty",
        "dataset_diagnostics": dataset_rows,
        "input_grids": {"g1_zero": str(GRID0), "g1_one": str(GRID1), "g1_1p017_direct": str(GRID17), "g1_profiled_direct": str(GRID2)},
        "uncertainty_status": "not yet full experimental/PDF/model-form propagation; this fit only profiles released point-to-point errors plus dataset normalization nuisances",
        "promotion_authorized": False,
        "frozen_baseline_unchanged": True,
        "production_outputs_modified": False,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
