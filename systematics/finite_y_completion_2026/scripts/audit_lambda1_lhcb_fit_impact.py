#!/usr/bin/env python3
"""Audit the lambda=1 unitary finite-Y candidate on the four LHCb tail bins.

This is deliberately an isolated diagnostic.  The LHCb rows have no approved
production normalization nuisance (``sysNorm_rel`` is zero in the data
manifest), so only the declared correlated matching and NLO-scale nuisances
are profiled here.  The result is a fit-impact check, not a promotion gate:
the four high-qT rows are still a diagnostic fiducial candidate in the source
dataset and need their own publication/scale audit.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "Data/v23a_tevatron_plus_lhcb7_fiducial_candidate/LHCb_7.csv"
ENDPOINTS = ROOT / "systematics/finite_y_completion_2026/reports/lambda1_lhcb_unitary/lhcb_unitary_lambda1_endpoints.csv"
OUT = ROOT / "systematics/finite_y_completion_2026/reports/lambda1_lhcb_unitary"
PROFILES = ("early_0p18_0p28", "central_0p20_0p30", "late_0p22_0p32")
UPWARD_NLO_SCALE = 0.19217428727157315


def fit_one(frame: pd.DataFrame, endpoint: str, profile: str) -> dict:
    values = frame[(frame.endpoint == endpoint) & frame.profile.eq(profile)].sort_values("row_id")
    if len(values) != 4:
        raise RuntimeError(f"expected four LHCb rows for {endpoint} {profile}, got {len(values)}")
    p = values.profile_value.to_numpy(float)
    w = values.W_lambda1_fiducial.to_numpy(float)
    nlo = values.FO_DYTurbo.to_numpy(float)
    data = values.CS.to_numpy(float)
    error = values.error.to_numpy(float)
    base = (1.0 - p) * w + p * nlo
    matching = (1.0 - p) * (nlo - w)
    scale = p * nlo * UPWARD_NLO_SCALE

    def residual(theta: np.ndarray) -> np.ndarray:
        eta_m, eta_s = theta
        prediction = base + eta_m * matching + eta_s * scale
        return np.concatenate([(prediction - data) / error, [eta_m, eta_s]])

    result = least_squares(
        residual, np.array([1.35, 0.94]), bounds=([-5.0, -5.0], [5.0, 5.0]),
        xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=10000,
    )
    eta_m, eta_s = result.x
    prediction = base + eta_m * matching + eta_s * scale
    pulls = (prediction - data) / error
    data_chi2 = float(np.dot(pulls, pulls))
    total = data_chi2 + float(eta_m**2 + eta_s**2)
    return {
        "endpoint": endpoint,
        "profile": profile,
        "optimizer_success": bool(result.success),
        "data_chi2": data_chi2,
        "data_chi2_per_row": float(np.mean(pulls**2)),
        "total_chi2": total,
        "total_chi2_per_row": float(total / len(values)),
        "matching_nuisance_sigma": float(eta_m),
        "nlo_scale_nuisance_sigma": float(eta_s),
        "max_absolute_pull": float(np.max(np.abs(pulls))),
        "min_prediction": float(np.min(prediction)),
        "max_prediction": float(np.max(prediction)),
        # This is intentionally diagnostic only.  It is not a universal
        # production promotion criterion for the LHCb candidate rows.
        "pass_diagnostic_gate": bool(
            result.success and np.isfinite(prediction).all() and (prediction > 0).all()
            and np.mean(pulls**2) < 2.0 and abs(eta_m) < 2.0 and abs(eta_s) < 2.0
            and np.max(np.abs(pulls)) < 3.5
        ),
    }


def main() -> None:
    data = pd.read_csv(DATA)
    rows = data[data.row_id.isin(["LHCb_7:10", "LHCb_7:11", "LHCb_7:12", "LHCb_7:13"])].copy()
    values = pd.read_csv(ENDPOINTS)
    # The recompute table carries the immutable data CS/error columns.  Keep
    # the source normalization metadata separately to avoid duplicate CS_x/
    # CS_y columns when joining the data manifest.
    values = values.merge(rows[["row_id", "sysNorm_rel"]], on="row_id", how="left", validate="many_to_one")
    if values.CS.isna().any() or values.FO_DYTurbo.isna().any():
        raise RuntimeError("LHCb endpoint table is missing data or DYTurbo inputs")
    endpoints = values.endpoint.drop_duplicates().tolist()
    results = pd.DataFrame([fit_one(values, endpoint, profile) for endpoint in endpoints for profile in PROFILES])
    results.to_csv(OUT / "lhcb_unitary_lambda1_fit_impact.csv", index=False)
    profile_summary = {}
    for profile in PROFILES:
        sub = results[results.profile.eq(profile)]
        profile_summary[profile] = {
            "all_optimizer_success": bool(sub.optimizer_success.all()),
            "all_diagnostic_gates": bool(sub.pass_diagnostic_gate.all()),
            "data_chi2_per_row_q16_median_q84": [float(x) for x in np.quantile(sub.data_chi2_per_row, [0.16, 0.5, 0.84])],
            "total_chi2_per_row_q16_median_q84": [float(x) for x in np.quantile(sub.total_chi2_per_row, [0.16, 0.5, 0.84])],
            "matching_nuisance_q16_median_q84": [float(x) for x in np.quantile(sub.matching_nuisance_sigma, [0.16, 0.5, 0.84])],
            "nlo_scale_nuisance_q16_median_q84": [float(x) for x in np.quantile(sub.nlo_scale_nuisance_sigma, [0.16, 0.5, 0.84])],
            "max_absolute_pull_q84": float(np.quantile(sub.max_absolute_pull, 0.84)),
            "max_absolute_pull": float(sub.max_absolute_pull.max()),
        }
    report = {
        "status": "lambda1_lhcb_unitary_fit_impact_diagnostic_complete",
        "endpoint_count": int(len(endpoints)),
        "row_count": int(len(rows)),
        "profiles": profile_summary,
        "all_diagnostic_gates": bool(results.pass_diagnostic_gate.all()),
        "normalization_treatment": "fixed; LHCb sysNorm_rel is zero in the diagnostic data manifest",
        "interpretation": "This checks the corrected lambda=1 unitary finite-Y candidate on the four high-qT LHCb fiducial rows after explicit DYTurbo node acceptance. The independent MCFM/DYTurbo factor-of-two is now understood as MCFM's absolute-rapidity convention; the remaining approximately 4.25-sigma scale pull reflects the positive-arm candidate data versus true-NLO normalization. It is not yet a universal production approval because LHCb covariance/publication and observable-normalization closure remain separate requirements.",
        "production_outputs_modified": False,
    }
    (OUT / "lhcb_fit_impact_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
