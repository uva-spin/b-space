#!/usr/bin/env python3
"""Propagate LHCb covariance replicas through the NNLO unitary ensemble.

This is a finite-Y diagnostic, not a production fit.  It uses all 96 verified
lambda=1 endpoint W curves, all three transition profiles, the published LHCb
pT covariance reconstruction, and 50 reproducible Gaussian pseudo-replicas.
Frozen data and production outputs are read-only inputs.
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
N_REPLICAS = 50
SEED = 20260817


def covariance_matrix() -> np.ndarray:
    path = BASE / "scripts/audit_lhcb_correlated_covariance.py"
    spec = importlib.util.spec_from_file_location("lhcb_covariance_source_replica", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw = pd.read_csv(DATA_RAW).iloc[:14].copy()
    nonbeam = np.sqrt(raw["stat"].to_numpy(float) ** 2 + raw["sys"].to_numpy(float) ** 2)
    beam = raw["sys,beam"].to_numpy(float)
    lumi = raw["sys,lumi"].to_numpy(float)
    full = module.correlation_matrix() * np.outer(nonbeam, nonbeam)
    full += np.outer(beam, beam) + np.outer(lumi, lumi)
    idx = [10, 11, 12, 13]
    return full[np.ix_(idx, idx)]


def load_maps() -> tuple[dict[str, float], dict[str, float]]:
    nnlo = pd.read_csv(NNLO).set_index("row_id")
    fo = {rid: float(nnlo.loc[rid, "dyturbo_pb_per_GeV"]) for rid in ROW_IDS}
    scan = json.loads(NNLO_SCALE.read_text())
    scale = {}
    for row in scan["rows"]:
        scale[row["row_id"]] = max(
            abs(float(row["theory_min_pb_per_GeV"]) - float(row["theory_central_pb_per_GeV"])),
            abs(float(row["theory_max_pb_per_GeV"]) - float(row["theory_central_pb_per_GeV"])),
        )
    return fo, scale


def fit_one(sub: pd.DataFrame, white: np.ndarray, data: np.ndarray, fo_map: dict[str, float], scale_map: dict[str, float]) -> dict:
    sub = sub.set_index("row_id").loc[ROW_IDS]
    p = sub.profile_value.to_numpy(float)
    w = sub.W_lambda1_fiducial.to_numpy(float)
    fo = np.array([fo_map[rid] for rid in ROW_IDS])
    scale = p * np.array([scale_map[rid] for rid in ROW_IDS])
    base = (1.0 - p) * w + p * fo
    matching = (1.0 - p) * (fo - w)

    def residual(theta: np.ndarray) -> np.ndarray:
        eta_m, eta_s = theta
        prediction = base + eta_m * matching + eta_s * scale
        return np.concatenate([white @ (prediction - data), [eta_m, eta_s]])

    result = least_squares(
        residual, np.array([1.0, 0.0]), bounds=([-5.0, -5.0], [5.0, 5.0]),
        xtol=1e-11, ftol=1e-11, gtol=1e-11, max_nfev=3000,
    )
    eta_m, eta_s = result.x
    prediction = base + eta_m * matching + eta_s * scale
    pull = white @ (prediction - data)
    data_chi2 = float(np.dot(pull, pull))
    total = data_chi2 + float(eta_m**2 + eta_s**2)
    return {
        "optimizer_success": bool(result.success),
        "data_chi2_per_row": data_chi2 / len(ROW_IDS),
        "total_chi2_per_row": total / len(ROW_IDS),
        "matching_nuisance_sigma": float(eta_m),
        "nnlo_scale_nuisance_sigma": float(eta_s),
        "max_whitened_pull": float(np.max(np.abs(pull))),
        "all_positive": bool((prediction > 0).all()),
    }


def main() -> None:
    endpoints = pd.read_csv(ENDPOINTS)
    if set(endpoints.row_id.unique()) != set(ROW_IDS):
        raise RuntimeError("endpoint table does not contain exactly the LHCb boundary rows")
    cov = covariance_matrix()
    eigval, eigvec = np.linalg.eigh(cov)
    floor = max(float(eigval.max()) * 1e-12, 1e-18)
    cov_sqrt = eigvec @ np.diag(np.sqrt(np.clip(eigval, floor, None))) @ eigvec.T
    white = (eigvec / np.sqrt(np.clip(eigval, floor, None))) @ eigvec.T
    # The raw released table has no row_id column; its ordered pT rows map
    # directly to LHCb_7:10--:13 for the four boundary bins.
    central = pd.read_csv(DATA_RAW).iloc[[10, 11, 12, 13]]["A"].to_numpy(float)
    fo_map, scale_map = load_maps()
    rng = np.random.default_rng(SEED)
    replicas = central.reshape(1, -1) + rng.normal(size=(N_REPLICAS, len(ROW_IDS))) @ cov_sqrt.T

    records = []
    for replica_id, pseudo_data in enumerate(replicas):
        for endpoint in endpoints.endpoint.unique():
            for profile in PROFILES:
                sub = endpoints[(endpoints.endpoint == endpoint) & endpoints.profile.eq(profile)]
                fit = fit_one(sub, white, pseudo_data, fo_map, scale_map)
                records.append({"replica": replica_id, "endpoint": endpoint, "profile": profile, **fit})
    results = pd.DataFrame(records)
    OUT.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT / "lhcb_unitary_lambda1_nnlo_replica_fits.csv", index=False)
    summary = {
        "status": "lambda1_lhcb_unitary_nnlo_replica_propagation_complete_diagnostic_not_production",
        "replicas": N_REPLICAS,
        "seed": SEED,
        "endpoint_count": int(endpoints.endpoint.nunique()),
        "profiles": list(PROFILES),
        "rows": ROW_IDS,
        "covariance": "published pT correlations plus fully correlated beam/luminosity components",
        "fixed_order": "DYTurbo order=2 (NNLO), positive y_Z arm",
        "scale_treatment": "symmetric largest-absolute six-point NNLO excursion per row",
        "all_optimizer_success": bool(results.optimizer_success.all()),
        "all_predictions_positive": bool(results.all_positive.all()),
        "profile_summary": {},
        "production_outputs_modified": False,
    }
    for profile in PROFILES:
        sub = results[results.profile.eq(profile)]
        summary["profile_summary"][profile] = {
            "data_chi2_per_row_q16_median_q84": [float(x) for x in np.quantile(sub.data_chi2_per_row, [0.16, 0.50, 0.84])],
            "total_chi2_per_row_q16_median_q84": [float(x) for x in np.quantile(sub.total_chi2_per_row, [0.16, 0.50, 0.84])],
            "matching_nuisance_sigma_q16_median_q84": [float(x) for x in np.quantile(sub.matching_nuisance_sigma, [0.16, 0.50, 0.84])],
            "nnlo_scale_nuisance_sigma_q16_median_q84": [float(x) for x in np.quantile(sub.nnlo_scale_nuisance_sigma, [0.16, 0.50, 0.84])],
            "max_whitened_pull_q16_median_q84": [float(x) for x in np.quantile(sub.max_whitened_pull, [0.16, 0.50, 0.84])],
        }
    summary["interpretation"] = (
        "Experimental replica sampling is propagated through the complete endpoint ensemble. "
        "The resulting spread quantifies the LHCb statistical/systematic input contribution; "
        "the much larger fixed central residual remains an observable-level closure issue."
    )
    (OUT / "lhcb_unitary_nnlo_replica_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
