#!/usr/bin/env python3
"""Audit the LHCb pT covariance and its impact on the unitary finite-Y fit.

The source manifest currently carries diagonal diagnostic errors.  The LHCb
publication supplies a pT correlation matrix (with beam-energy and luminosity
uncertainties excluded).  This isolated audit reconstructs the four high-qT
sub-covariance from the published correlation matrix, adds the fully
correlated beam/luminosity components from the released error decomposition,
and profiles the same matching/scale nuisances used by the earlier diagnostic.
No source or production files are modified.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]
DATA_RAW = ROOT / "Data/global_dy_raw/LHCb_7.csv"
ENDPOINTS = BASE / "reports/lambda1_lhcb_unitary/lhcb_unitary_lambda1_endpoints.csv"
NNLO_DIR = BASE / "reports/lhcb7_external_true_nnlo_positive_y_10m_"
NNLO_SCALE = BASE / "reports/lhcb_true_nnlo_scale_scan/summary.json"
OUT = BASE / "reports/lhcb_correlated_covariance_audit"
ROW_IDS = [f"LHCb_7:{i}" for i in (10, 11, 12, 13)]
PROFILES = ("early_0p18_0p28", "central_0p20_0p30", "late_0p22_0p32")

# Lower triangle of the pT correlation matrix in the publication's Table 10.
# It excludes beam-energy and luminosity uncertainties, which are added below.
CORR_LOWER = [
    [1.00],
    [-0.01, 1.00],
    [0.00, 0.03, 1.00],
    [0.04, 0.00, 0.02, 1.00],
    [0.05, 0.05, 0.00, 0.03, 1.00],
    [0.07, 0.07, 0.05, 0.00, 0.03, 1.00],
    [0.08, 0.08, 0.06, 0.06, 0.02, 0.02, 1.00],
    [0.07, 0.06, 0.05, 0.05, 0.07, 0.04, 0.00, 1.00],
    [0.07, 0.07, 0.05, 0.05, 0.06, 0.09, 0.07, -0.01, 1.00],
    [0.08, 0.08, 0.05, 0.06, 0.07, 0.10, 0.12, 0.08, -0.01, 1.00],
    [0.08, 0.08, 0.06, 0.06, 0.07, 0.10, 0.12, 0.10, 0.10, 0.02, 1.00],
    [0.08, 0.07, 0.05, 0.06, 0.07, 0.10, 0.11, 0.09, 0.10, 0.11, 0.05, 1.00],
    [0.07, 0.06, 0.05, 0.05, 0.06, 0.08, 0.09, 0.08, 0.08, 0.09, 0.10, 0.06, 1.00],
    [0.20, 0.20, 0.15, 0.15, 0.19, 0.26, 0.30, 0.26, 0.27, 0.30, 0.32, 0.31, 0.30, 1.00],
]


def correlation_matrix() -> np.ndarray:
    matrix = np.zeros((14, 14), dtype=float)
    for i, row in enumerate(CORR_LOWER):
        matrix[i, : i + 1] = row
        matrix[: i + 1, i] = row
    return matrix


def load_nnlo() -> dict[str, float]:
    result = {}
    for index in (10, 11, 12, 13):
        frame = pd.read_csv(NNLO_DIR.with_name(NNLO_DIR.name + str(index)) / "dyturbo_true_nlo_summary.csv")
        row = frame.iloc[0]
        result[str(row.row_id)] = float(row.dyturbo_pb_per_GeV)
    return result


def fit_one(values: pd.DataFrame, covariance: np.ndarray, endpoint: str, profile: str, fo_map: dict[str, float], scale_up: dict[str, float]) -> dict:
    sub = values[(values.endpoint == endpoint) & values.profile.eq(profile)].set_index("row_id").loc[ROW_IDS]
    p = sub.profile_value.to_numpy(float)
    w = sub.W_lambda1_fiducial.to_numpy(float)
    data = sub.CS.to_numpy(float)
    fo = np.array([fo_map[row_id] for row_id in ROW_IDS], dtype=float)
    base = (1.0 - p) * w + p * fo
    matching = (1.0 - p) * (fo - w)
    scale = p * np.array([scale_up[row_id] for row_id in ROW_IDS], dtype=float)
    eigval, eigvec = np.linalg.eigh(covariance)
    floor = max(float(eigval.max()) * 1.0e-12, 1.0e-18)
    whitening = (eigvec / np.sqrt(np.clip(eigval, floor, None))) @ eigvec.T

    def residual(theta: np.ndarray) -> np.ndarray:
        eta_m, eta_s = theta
        prediction = base + eta_m * matching + eta_s * scale
        return np.concatenate([whitening @ (prediction - data), [eta_m, eta_s]])

    result = least_squares(residual, np.array([1.0, 0.0]), bounds=([-5.0, -5.0], [5.0, 5.0]),
                           xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=10000)
    eta_m, eta_s = result.x
    prediction = base + eta_m * matching + eta_s * scale
    pull = whitening @ (prediction - data)
    data_chi2 = float(np.dot(pull, pull))
    return {
        "endpoint": endpoint, "profile": profile, "optimizer_success": bool(result.success),
        "data_chi2": data_chi2, "data_chi2_per_row": data_chi2 / len(ROW_IDS),
        "total_chi2": data_chi2 + float(eta_m**2 + eta_s**2),
        "total_chi2_per_row": (data_chi2 + float(eta_m**2 + eta_s**2)) / len(ROW_IDS),
        "matching_nuisance_sigma": float(eta_m), "nnlo_scale_nuisance_sigma": float(eta_s),
        "max_whitened_pull": float(np.max(np.abs(pull))),
        "min_prediction": float(prediction.min()), "max_prediction": float(prediction.max()),
    }


def main() -> None:
    raw_all = pd.read_csv(DATA_RAW).iloc[:14].copy()
    # The imported A/stat/sys columns are already per-GeV values (the source
    # table's pb/bin values were divided by dPT when the manifest was built).
    nonbeam_sigma = np.sqrt(raw_all["stat"].to_numpy(float) ** 2 + raw_all["sys"].to_numpy(float) ** 2)
    beam_all = raw_all["sys,beam"].to_numpy(float)
    lumi_all = raw_all["sys,lumi"].to_numpy(float)
    corr = correlation_matrix()
    covariance_all = corr * np.outer(nonbeam_sigma, nonbeam_sigma)
    covariance_all += np.outer(beam_all, beam_all) + np.outer(lumi_all, lumi_all)
    high_indices = [10, 11, 12, 13]
    covariance = covariance_all[np.ix_(high_indices, high_indices)]
    endpoints = pd.read_csv(ENDPOINTS)
    fo_nnlo = load_nnlo()
    nnlo_summary = json.loads(NNLO_SCALE.read_text())
    scale_up = {
        row["row_id"]: float(row["theory_max_pb_per_GeV"] - row["theory_central_pb_per_GeV"])
        for row in nnlo_summary["rows"]
    }
    values = endpoints["endpoint profile row_id W_lambda1_fiducial profile_value CS".split()].copy()
    results = pd.DataFrame([fit_one(values, covariance, endpoint, profile, fo_nnlo, scale_up)
                            for endpoint in values.endpoint.unique() for profile in PROFILES])
    OUT.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT / "lhcb_correlated_covariance_nnlo_fit.csv", index=False)
    pd.DataFrame(covariance, index=ROW_IDS, columns=ROW_IDS).to_csv(OUT / "lhcb_pT_covariance_pb2_per_GeV2.csv")
    pd.DataFrame({
        "row_id": ROW_IDS,
        "nonbeam_sigma_pb_per_GeV": nonbeam_sigma[high_indices],
        "beam_sigma_pb_per_GeV": beam_all[high_indices],
        "luminosity_sigma_pb_per_GeV": lumi_all[high_indices],
    }).to_csv(OUT / "lhcb_pT_covariance_components.csv", index=False)
    summary = {
        "status": "lhcb_correlated_covariance_audit_complete_diagnostic_not_production",
        "row_ids": ROW_IDS,
        "covariance_source": "LHCb 1505.07024 Table 10 pT correlations, with released beam/luminosity components added as fully correlated",
        "covariance_positive_eigenvalues": [float(x) for x in np.linalg.eigvalsh(covariance)],
        "nnlo_input": str(NNLO_DIR),
        "profile_summary": {
            profile: {
                "data_chi2_per_row_q16_median_q84": [float(x) for x in np.quantile(results.loc[results.profile.eq(profile), "data_chi2_per_row"], [0.16, 0.5, 0.84])],
                "total_chi2_per_row_q16_median_q84": [float(x) for x in np.quantile(results.loc[results.profile.eq(profile), "total_chi2_per_row"], [0.16, 0.5, 0.84])],
                "nnlo_scale_nuisance_q16_median_q84": [float(x) for x in np.quantile(results.loc[results.profile.eq(profile), "nnlo_scale_nuisance_sigma"], [0.16, 0.5, 0.84])],
            } for profile in PROFILES
        },
        "interpretation": "The published pT correlations and beam/luminosity components are now explicitly represented. This is an input-covariance audit only; it does not authorize production because the source manifest and covariance provenance still require formal promotion.",
        "production_outputs_modified": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
