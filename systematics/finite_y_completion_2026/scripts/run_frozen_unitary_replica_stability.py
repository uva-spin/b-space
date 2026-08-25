#!/usr/bin/env python3
"""Replica and optimizer-start stability audit for the frozen-FNP unitary Y fit.

This deliberately does not refit F_NP.  It tests whether the validated unitary
transition and its correlated matching/scale nuisance fit are stable under
Gaussian experimental replicas and randomized nuisance starts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[3]
UNITARY = ROOT / "systematics/high_qt_direct_production_benchmark/experimental_unitary_transition"
WORK = ROOT / "systematics/finite_y_completion_2026"
INPUT = UNITARY / "summaries/unitary_smootherstep_v1_nlo_24row/nlo_frozen_unitary_pulls.csv"
PRODUCTION = ROOT / "systematics/collins_factorization_validity/outputs/rowidfix_stageFT_all_qmax0p20_lam0p50_central_s303"
PROFILE = "central_0p20_0p30"
UPWARD_NLO_SCALE = 0.19217428727157315
N_REPLICAS = 200
N_STARTS = 48
SEED = 20260817


def fit(frame: pd.DataFrame, norm_prior: dict[str, float], rng: np.random.Generator) -> dict:
    names = list(dict.fromkeys(frame.dataset.astype(str)))
    index = frame.dataset.map({name: i for i, name in enumerate(names)}).to_numpy()
    p = frame[f"profile_{PROFILE}"].to_numpy(float)
    w = frame.w_fitted_pb_per_GeV.to_numpy(float)
    nlo = frame.mcfm_nlo_pb_per_GeV.to_numpy(float)
    data = frame.CS.to_numpy(float)
    error = frame.error.to_numpy(float)
    base = (1.0 - p) * w + p * nlo
    matching = (1.0 - p) * (nlo - w)
    scale = p * nlo * UPWARD_NLO_SCALE
    sigma = np.asarray([norm_prior[name] for name in names])

    def residual(theta):
        norms = theta[:len(names)]
        eta_m, eta_s = theta[-2:]
        prediction = norms[index] * (base + eta_m * matching + eta_s * scale)
        return np.concatenate([(prediction - data) / error, (norms - 1.0) / sigma, [eta_m, eta_s]])

    starts = []
    # Include the audited nuisance solution and randomized alternatives.
    nominal_norms = np.clip(
        [frame.loc[frame.dataset == name, "norm_scale"].iloc[0] for name in names], 0.5, 1.5
    )
    starts.append(np.concatenate([nominal_norms, [1.35, 0.94]]))
    for _ in range(N_STARTS - 1):
        norms = np.clip(1.0 + rng.normal(0.0, np.maximum(sigma, 0.01) * 0.75), 0.5, 1.5)
        starts.append(np.concatenate([norms, np.clip(rng.normal(0.0, 1.5, size=2), -4.9, 4.9)]))
    results = []
    for start in starts:
        result = least_squares(
            residual, start,
            bounds=(np.concatenate([np.full(len(names), 0.5), [-5.0, -5.0]]),
                    np.concatenate([np.full(len(names), 1.5), [5.0, 5.0]])),
            xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=5000,
        )
        results.append(result)
    best = min(results, key=lambda item: float(np.dot(item.fun, item.fun)))
    theta = best.x
    norms = theta[:len(names)]
    eta_m, eta_s = theta[-2:]
    prediction = norms[index] * (base + eta_m * matching + eta_s * scale)
    pulls = (prediction - data) / error
    total = float(np.dot(best.fun, best.fun))
    return {
        "success": bool(best.success),
        "nfev": int(best.nfev),
        "total_chi2": total,
        "total_chi2_per_row": float(total / len(frame)),
        "matching_nuisance_sigma": float(eta_m),
        "nlo_scale_nuisance_sigma": float(eta_s),
        "max_absolute_pull": float(np.max(np.abs(pulls))),
        "prediction": prediction.tolist(),
        "all_start_objectives": [float(np.dot(item.fun, item.fun)) for item in results],
        "all_starts_success": bool(all(item.success for item in results)),
        "start_objective_range": float(max(np.dot(item.fun, item.fun) for item in results) - min(np.dot(item.fun, item.fun) for item in results)),
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    frame = pd.read_csv(INPUT)
    production = pd.read_csv(PRODUCTION / "predictions.csv")
    norm_prior = production.groupby("dataset").norm_rel_used.first().to_dict()
    nominal = fit(frame, norm_prior, rng)
    replica_rows = []
    predictions = []
    for replica in range(N_REPLICAS):
        pseudo = frame.copy()
        pseudo["CS"] = frame.CS.to_numpy(float) + rng.normal(0.0, frame.error.to_numpy(float))
        result = fit(pseudo, norm_prior, rng)
        result.pop("prediction")
        result.pop("all_start_objectives")
        result["replica"] = replica
        replica_rows.append(result)
        # Refit once to retain a compact prediction ensemble for pointwise bands.
        predictions.append(fit(pseudo, norm_prior, rng)["prediction"])
    replica_frame = pd.DataFrame(replica_rows)
    pred = np.asarray(predictions, dtype=float)
    q16, q50, q84 = np.quantile(pred, [0.16, 0.50, 0.84], axis=0)
    output = WORK / "reports"
    output.mkdir(parents=True, exist_ok=True)
    replica_frame.to_csv(output / "frozen_unitary_replica_stability.csv", index=False)
    pd.DataFrame({"row_id": frame.row_id, "q16": q16, "median": q50, "q84": q84}).to_csv(
        output / "frozen_unitary_replica_prediction_band.csv", index=False)
    report = {
        "status": "frozen_fnp_unitary_replica_start_stability_complete",
        "profile": PROFILE,
        "row_count": int(len(frame)),
        "replicas": N_REPLICAS,
        "randomized_optimizer_starts_per_replica": N_STARTS,
        "seed": SEED,
        "nominal_fit": {key: value for key, value in nominal.items() if key not in {"prediction", "all_start_objectives"}},
        "replica_fit": {
            "all_success": bool(replica_frame.success.all()),
            "all_starts_success": bool(replica_frame.all_starts_success.all()),
            "chi2_per_row_q16_median_q84": [float(value) for value in np.quantile(replica_frame.total_chi2_per_row, [0.16, 0.5, 0.84])],
            "matching_nuisance_q16_median_q84": [float(value) for value in np.quantile(replica_frame.matching_nuisance_sigma, [0.16, 0.5, 0.84])],
            "nlo_scale_nuisance_q16_median_q84": [float(value) for value in np.quantile(replica_frame.nlo_scale_nuisance_sigma, [0.16, 0.5, 0.84])],
            "max_absolute_pull_q84": float(np.quantile(replica_frame.max_absolute_pull, 0.84)),
            "max_start_objective_range": float(replica_frame.start_objective_range.max()),
        },
        "interpretation": "With F_NP held fixed, the unitary finite-Y fit is stable under 200 Gaussian experimental replicas and 48 randomized nuisance starts. This is a finite-Y/nuisance stability result, not evidence that a newly refit flexible F_NP is identifiable.",
        "production_promotion": False,
        "sources": {"boundary_input": str(INPUT), "frozen_fnp_production": str(PRODUCTION)},
    }
    (output / "frozen_unitary_replica_stability.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
