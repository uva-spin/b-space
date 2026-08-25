#!/usr/bin/env python3
"""Fit-impact audit for unitary finite-Y using all registered lambda=1 endpoints."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[3]
UNITARY = ROOT / "systematics/high_qt_direct_production_benchmark/experimental_unitary_transition"
LAMBDA1 = ROOT / "systematics/dataset_identifiability_campaign_2026"
INPUT = UNITARY / "summaries/unitary_smootherstep_v1_nlo_24row/nlo_frozen_unitary_pulls.csv"
ENDPOINTS = ROOT / "systematics/finite_y_completion_2026/reports/lambda1_unitary_endpoint_recompute/boundary_unitary_lambda1_endpoints.csv"
LAMBDA1_ENDPOINT = ROOT / "systematics/dataset_identifiability_campaign_2026/outputs/lambda1_start_expansion96_s353_cont120000"
PROFILES = ("early_0p18_0p28", "central_0p20_0p30", "late_0p22_0p32")
UPWARD_NLO_SCALE = 0.19217428727157315


def endpoint_norms(endpoint: str, datasets: list[str]) -> np.ndarray:
    path = LAMBDA1 / "outputs" / endpoint / "dataset_norms.csv"
    frame = pd.read_csv(path).set_index("dataset")
    return frame.reindex(datasets).production_norm.to_numpy(float)


def fit(frame: pd.DataFrame, endpoint: str, profile: str, norm_prior: dict[str, float]) -> dict:
    names = list(dict.fromkeys(frame.dataset.astype(str)))
    index = frame.dataset.map({name: i for i, name in enumerate(names)}).to_numpy()
    endpoint_values = pd.read_csv(ENDPOINTS)
    endpoint_values = endpoint_values[
        (endpoint_values.endpoint == endpoint) & endpoint_values.profile.eq(profile)
    ].set_index("row_id").reindex(frame.row_id.astype(str))
    if endpoint_values.W_lambda1.isna().any():
        raise RuntimeError(f"missing endpoint rows for {endpoint} {profile}")
    p = frame[f"profile_{profile}"].to_numpy(float)
    w = endpoint_values.W_lambda1.to_numpy(float)
    nlo = frame.mcfm_nlo_pb_per_GeV.to_numpy(float)
    data = frame.CS.to_numpy(float)
    error = frame.error.to_numpy(float)
    base = (1.0 - p) * w + p * nlo
    matching = (1.0 - p) * (nlo - w)
    scale = p * nlo * UPWARD_NLO_SCALE
    sigma = np.asarray([norm_prior[name] for name in names])
    start_norms = endpoint_norms(endpoint, names)

    def residual(theta):
        norms = theta[:len(names)]
        eta_m, eta_s = theta[-2:]
        prediction = norms[index] * (base + eta_m * matching + eta_s * scale)
        return np.concatenate([(prediction - data) / error, (norms - 1.0) / sigma, [eta_m, eta_s]])

    start = np.concatenate([np.clip(start_norms, 0.5, 1.5), [1.35, 0.94]])
    result = least_squares(
        residual, start,
        bounds=(np.concatenate([np.full(len(names), 0.5), [-5.0, -5.0]]),
                np.concatenate([np.full(len(names), 1.5), [5.0, 5.0]])),
        xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=10000,
    )
    norms = result.x[:len(names)]
    eta_m, eta_s = result.x[-2:]
    prediction = norms[index] * (base + eta_m * matching + eta_s * scale)
    pulls = (prediction - data) / error
    data_chi2 = float(np.dot(pulls, pulls))
    norm_penalty = float(np.sum(((norms - 1.0) / sigma) ** 2))
    total = data_chi2 + norm_penalty + float(eta_m**2 + eta_s**2)
    return {
        "endpoint": endpoint, "profile": profile, "optimizer_success": bool(result.success),
        "data_chi2_per_row": float(np.mean(pulls**2)), "data_chi2": data_chi2,
        "total_chi2": total, "total_chi2_per_row": float(total / len(frame)),
        "matching_nuisance_sigma": float(eta_m), "nlo_scale_nuisance_sigma": float(eta_s),
        "max_absolute_pull": float(np.max(np.abs(pulls))),
        "min_prediction": float(np.min(prediction)), "max_prediction": float(np.max(prediction)),
        "pass_operational_gate": bool(
            result.success and np.mean(pulls**2) < 1.5 and abs(eta_m) < 2.0
            and abs(eta_s) < 2.0 and np.max(np.abs(pulls)) < 3.0
        ),
    }


def main() -> None:
    frame = pd.read_csv(INPUT)
    endpoints = pd.read_csv(ENDPOINTS).endpoint.drop_duplicates().tolist()
    production = pd.read_csv(LAMBDA1_ENDPOINT / "dataset_norms.csv")
    norm_prior = production.set_index(production.dataset.astype(str)).norm_width.to_dict()
    rows = [fit(frame, endpoint, profile, norm_prior) for endpoint in endpoints for profile in PROFILES]
    results = pd.DataFrame(rows)
    output = ROOT / "systematics/finite_y_completion_2026/reports"
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "lambda1_unitary_fit_impact.csv", index=False)
    central = results[results.profile.eq("central_0p20_0p30")]
    profile_summary = {}
    for profile in PROFILES:
        sub = results[results.profile.eq(profile)]
        profile_summary[profile] = {
            "all_optimizer_success": bool(sub.optimizer_success.all()),
            "all_operational_gates": bool(sub.pass_operational_gate.all()),
            "total_chi2_per_row_q16_median_q84": [float(x) for x in np.quantile(sub.total_chi2_per_row, [0.16, 0.5, 0.84])],
            "data_chi2_per_row_q16_median_q84": [float(x) for x in np.quantile(sub.data_chi2_per_row, [0.16, 0.5, 0.84])],
            "matching_nuisance_q16_median_q84": [float(x) for x in np.quantile(sub.matching_nuisance_sigma, [0.16, 0.5, 0.84])],
            "nlo_scale_nuisance_q16_median_q84": [float(x) for x in np.quantile(sub.nlo_scale_nuisance_sigma, [0.16, 0.5, 0.84])],
            "max_absolute_pull_q84": float(np.quantile(sub.max_absolute_pull, 0.84)),
            "max_absolute_pull": float(sub.max_absolute_pull.max()),
        }
    report = {
        "status": "lambda1_unitary_fit_impact_audit_complete",
        "endpoint_count": int(len(endpoints)), "row_count": int(len(frame)),
        "profiles": profile_summary,
        "central_all_operational_gates": bool(central.pass_operational_gate.all()),
        "interpretation": "The unitary finite-Y boundary fit has been evaluated using every registered lambda=1 endpoint. This is a Tevatron fit-impact audit; it is not yet the full experimental-replica propagation or LHCb production result.",
        "production_promotion": False,
    }
    (output / "lambda1_unitary_fit_impact.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
