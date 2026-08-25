#!/usr/bin/env python3
"""Propagate boundary experimental replicas through the lambda=1 unitary fit."""

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
N_REPLICAS = 50
SEED = 20260817


def fit(frame: pd.DataFrame, endpoint_rows: pd.DataFrame, endpoint: str, profile: str,
        norm_prior: dict[str, float], rng: np.random.Generator) -> dict:
    names = list(dict.fromkeys(frame.dataset.astype(str)))
    index = frame.dataset.map({name: i for i, name in enumerate(names)}).to_numpy()
    row_values = endpoint_rows[
        (endpoint_rows.endpoint == endpoint) & endpoint_rows.profile.eq(profile)
    ].set_index("row_id").reindex(frame.row_id.astype(str))
    p = frame[f"profile_{profile}"].to_numpy(float)
    w = row_values.W_lambda1.to_numpy(float)
    nlo = frame.mcfm_nlo_pb_per_GeV.to_numpy(float)
    data = frame.CS.to_numpy(float) + rng.normal(0.0, frame.error.to_numpy(float))
    error = frame.error.to_numpy(float)
    base = (1.0 - p) * w + p * nlo
    matching = (1.0 - p) * (nlo - w)
    scale = p * nlo * UPWARD_NLO_SCALE
    sigma = np.asarray([norm_prior[name] for name in names])
    endpoint_norm_frame = pd.read_csv(LAMBDA1 / "outputs" / endpoint / "dataset_norms.csv").set_index("dataset")
    start_norms = endpoint_norm_frame.reindex(names).production_norm.to_numpy(float)

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
    pulls = residual(result.x)[:len(frame)]
    return {
        "endpoint": endpoint, "profile": profile,
        "optimizer_success": bool(result.success),
        "total_chi2_per_row": float(np.dot(result.fun, result.fun) / len(frame)),
        "data_chi2_per_row": float(np.mean(pulls**2)),
        "matching_nuisance_sigma": float(result.x[-2]),
        "nlo_scale_nuisance_sigma": float(result.x[-1]),
        "max_absolute_pull": float(np.max(np.abs(pulls))),
    }


def main() -> None:
    frame = pd.read_csv(INPUT)
    endpoint_rows = pd.read_csv(ENDPOINTS)
    endpoints = endpoint_rows.endpoint.drop_duplicates().tolist()
    production = pd.read_csv(LAMBDA1_ENDPOINT / "dataset_norms.csv")
    norm_prior = production.set_index(production.dataset.astype(str)).norm_width.to_dict()
    rng = np.random.default_rng(SEED)
    rows = []
    for replica in range(N_REPLICAS):
        for endpoint in endpoints:
            for profile in PROFILES:
                result = fit(frame, endpoint_rows, endpoint, profile, norm_prior, rng)
                result["replica"] = replica
                rows.append(result)
    results = pd.DataFrame(rows)
    output = ROOT / "systematics/finite_y_completion_2026/reports"
    results.to_csv(output / "lambda1_unitary_boundary_replicas.csv", index=False)
    profile_summary = {}
    for profile in PROFILES:
        sub = results[results.profile.eq(profile)]
        profile_summary[profile] = {
            "all_optimizer_success": bool(sub.optimizer_success.all()),
            "total_chi2_per_row_q16_median_q84": [float(x) for x in np.quantile(sub.total_chi2_per_row, [0.16, 0.5, 0.84])],
            "data_chi2_per_row_q16_median_q84": [float(x) for x in np.quantile(sub.data_chi2_per_row, [0.16, 0.5, 0.84])],
            "matching_nuisance_q16_median_q84": [float(x) for x in np.quantile(sub.matching_nuisance_sigma, [0.16, 0.5, 0.84])],
            "nlo_scale_nuisance_q16_median_q84": [float(x) for x in np.quantile(sub.nlo_scale_nuisance_sigma, [0.16, 0.5, 0.84])],
            "max_absolute_pull_q95": float(np.quantile(sub.max_absolute_pull, 0.95)),
            "max_absolute_pull": float(sub.max_absolute_pull.max()),
        }
    report = {
        "status": "lambda1_unitary_boundary_replica_propagation_complete",
        "endpoint_count": int(len(endpoints)), "replicas": N_REPLICAS,
        "row_count": int(len(frame)), "seed": SEED,
        "profiles": profile_summary,
        "interpretation": "This propagates Gaussian experimental errors on the 24 Tevatron boundary rows through all 96 registered lambda=1 endpoints and correlated matching/scale nuisance fits. It is not an LHCb result and does not by itself authorize production promotion.",
        "production_promotion": False,
    }
    (output / "lambda1_unitary_boundary_replicas.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
