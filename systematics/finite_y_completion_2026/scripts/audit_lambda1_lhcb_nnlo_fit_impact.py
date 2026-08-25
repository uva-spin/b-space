#!/usr/bin/env python3
"""Fit-impact audit for the lambda=1 unitary candidate with NNLO input.

This is intentionally diagnostic.  It uses the completed positive-arm NNLO
DYTurbo boundary rows, the 96 lambda=1 endpoint W ensemble, and the published
LHCb pT covariance reconstruction.  The NNLO scale uncertainty is represented
symmetrically by the largest absolute six-point excursion at each row.  No
LHCb source or frozen production output is changed.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]
DATA_RAW = ROOT / "Data/global_dy_raw/LHCb_7.csv"
ENDPOINTS = BASE / "reports/lambda1_lhcb_unitary_nnlo/lhcb_unitary_lambda1_endpoints.csv"
NNLO = BASE / "reports/lhcb7_external_true_nnlo_positive_y_10m_combined/dyturbo_true_nnlo_summary.csv"
NNLO_SCALE = BASE / "reports/lhcb_true_nnlo_scale_scan/summary.json"
OUT = BASE / "reports/lambda1_lhcb_unitary_nnlo"
ROW_IDS = [f"LHCb_7:{i}" for i in (10, 11, 12, 13)]
PROFILES = ("early_0p18_0p28", "central_0p20_0p30", "late_0p22_0p32")


def load_covariance_module():
    path = BASE / "scripts/audit_lhcb_correlated_covariance.py"
    spec = importlib.util.spec_from_file_location("lhcb_covariance_source", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def covariance() -> np.ndarray:
    module = load_covariance_module()
    raw = pd.read_csv(DATA_RAW).iloc[:14].copy()
    nonbeam = np.sqrt(raw["stat"].to_numpy(float) ** 2 + raw["sys"].to_numpy(float) ** 2)
    beam = raw["sys,beam"].to_numpy(float)
    lumi = raw["sys,lumi"].to_numpy(float)
    corr = module.correlation_matrix()
    full = corr * np.outer(nonbeam, nonbeam) + np.outer(beam, beam) + np.outer(lumi, lumi)
    idx = [10, 11, 12, 13]
    return full[np.ix_(idx, idx)]


def nnlo_maps() -> tuple[dict[str, float], dict[str, float]]:
    central = pd.read_csv(NNLO).set_index("row_id")
    fo = {row_id: float(central.loc[row_id, "dyturbo_pb_per_GeV"]) for row_id in ROW_IDS}
    scale_summary = json.loads(NNLO_SCALE.read_text())
    scale = {}
    for row in scale_summary["rows"]:
        row_id = row["row_id"]
        scale[row_id] = max(
            abs(float(row["theory_min_pb_per_GeV"]) - float(row["theory_central_pb_per_GeV"])),
            abs(float(row["theory_max_pb_per_GeV"]) - float(row["theory_central_pb_per_GeV"])),
        )
    missing = [row_id for row_id in ROW_IDS if row_id not in fo or row_id not in scale]
    if missing:
        raise RuntimeError(f"missing NNLO input rows: {missing}")
    return fo, scale


def whitening_matrix(cov: np.ndarray) -> np.ndarray:
    eigval, eigvec = np.linalg.eigh(cov)
    floor = max(float(eigval.max()) * 1.0e-12, 1.0e-18)
    return (eigvec / np.sqrt(np.clip(eigval, floor, None))) @ eigvec.T


def fit_one(values: pd.DataFrame, white: np.ndarray, fo_map: dict[str, float], scale_map: dict[str, float], endpoint: str, profile: str) -> dict:
    sub = values[(values.endpoint == endpoint) & values.profile.eq(profile)].set_index("row_id").loc[ROW_IDS]
    p = sub.profile_value.to_numpy(float)
    w = sub.W_lambda1_fiducial.to_numpy(float)
    data = sub.CS.to_numpy(float)
    fo = np.array([fo_map[row_id] for row_id in ROW_IDS])
    scale_abs = np.array([scale_map[row_id] for row_id in ROW_IDS])
    base = (1.0 - p) * w + p * fo
    matching = (1.0 - p) * (fo - w)
    scale = p * scale_abs

    def residual(theta: np.ndarray) -> np.ndarray:
        eta_m, eta_s = theta
        prediction = base + eta_m * matching + eta_s * scale
        return np.concatenate([white @ (prediction - data), [eta_m, eta_s]])

    result = least_squares(
        residual, np.array([1.0, 0.0]), bounds=([-5.0, -5.0], [5.0, 5.0]),
        xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=10000,
    )
    eta_m, eta_s = result.x
    prediction = base + eta_m * matching + eta_s * scale
    pull = white @ (prediction - data)
    data_chi2 = float(np.dot(pull, pull))
    total = data_chi2 + float(eta_m**2 + eta_s**2)
    return {
        "endpoint": endpoint,
        "profile": profile,
        "optimizer_success": bool(result.success),
        "data_chi2": data_chi2,
        "data_chi2_per_row": data_chi2 / len(ROW_IDS),
        "total_chi2": total,
        "total_chi2_per_row": total / len(ROW_IDS),
        "matching_nuisance_sigma": float(eta_m),
        "nnlo_scale_nuisance_sigma": float(eta_s),
        "max_whitened_pull": float(np.max(np.abs(pull))),
        "min_prediction": float(prediction.min()),
        "max_prediction": float(prediction.max()),
        "all_positive": bool((prediction > 0).all()),
    }


def main() -> None:
    values = pd.read_csv(ENDPOINTS)
    if set(values.row_id.unique()) != set(ROW_IDS):
        raise RuntimeError("NNLO endpoint table does not contain exactly the four LHCb boundary rows")
    fo_map, scale_map = nnlo_maps()
    cov = covariance()
    white = whitening_matrix(cov)
    results = pd.DataFrame([
        fit_one(values, white, fo_map, scale_map, endpoint, profile)
        for endpoint in values.endpoint.unique() for profile in PROFILES
    ])
    OUT.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT / "lhcb_unitary_lambda1_nnlo_fit_impact.csv", index=False)
    summary = {
        "status": "lambda1_lhcb_unitary_nnlo_fit_impact_complete_diagnostic_not_production",
        "endpoint_count": int(results.endpoint.nunique()),
        "row_count": len(ROW_IDS),
        "profiles": {},
        "covariance": "published pT Table 10 correlations plus fully correlated beam/luminosity components",
        "fixed_order": "DYTurbo order=2 (NNLO), doVJREAL/doVJVIRT=true, positive y_Z arm",
        "scale_treatment": "symmetric largest absolute six-point NNLO scale excursion per row",
        "all_optimizer_success": bool(results.optimizer_success.all()),
        "all_predictions_positive": bool(results.all_positive.all()),
        "production_outputs_modified": False,
    }
    for profile in PROFILES:
        sub = results[results.profile.eq(profile)]
        summary["profiles"][profile] = {
            "data_chi2_per_row_q16_median_q84": [float(x) for x in np.quantile(sub.data_chi2_per_row, [0.16, 0.50, 0.84])],
            "total_chi2_per_row_q16_median_q84": [float(x) for x in np.quantile(sub.total_chi2_per_row, [0.16, 0.50, 0.84])],
            "matching_nuisance_sigma_q16_median_q84": [float(x) for x in np.quantile(sub.matching_nuisance_sigma, [0.16, 0.50, 0.84])],
            "nnlo_scale_nuisance_sigma_q16_median_q84": [float(x) for x in np.quantile(sub.nnlo_scale_nuisance_sigma, [0.16, 0.50, 0.84])],
            "max_whitened_pull_q84": float(np.quantile(sub.max_whitened_pull, 0.84)),
            "max_whitened_pull": float(sub.max_whitened_pull.max()),
        }
    summary["interpretation"] = (
        "The NNLO unitary endpoint ensemble is numerically stable and positive, but the "
        "published LHCb covariance fit remains a diagnostic rather than a production gate. "
        "The result tests the finite-Y construction with the best available fixed-order input; "
        "it does not silently rescale the LHCb data or authorize universal promotion."
    )
    (OUT / "lhcb_unitary_nnlo_fit_impact_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
