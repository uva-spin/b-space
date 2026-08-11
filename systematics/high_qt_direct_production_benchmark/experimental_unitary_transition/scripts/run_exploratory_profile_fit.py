#!/usr/bin/env python3
"""Fit Tevatron normalizations and two correlated theory nuisances at frozen FNP."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "systematics/high_qt_direct_production_benchmark/experimental_unitary_transition"
INPUT = BASE / "summaries/unitary_smootherstep_v1_nlo_24row/nlo_frozen_unitary_pulls.csv"
PRODUCTION = ROOT / "systematics/collins_factorization_validity/outputs/rowidfix_stageFT_all_qmax0p20_lam0p50_central_s303"
TARGET = BASE / "outputs/unitary_smootherstep_v1_exploratory_profile_fit"
PROFILES = ("early_0p18_0p28", "central_0p20_0p30", "late_0p22_0p32")
UPWARD_NLO_SCALE = 0.19217428727157315


def fit_profile(frame: pd.DataFrame, profile_name: str, norm_prior: dict[str, float]) -> tuple[dict, pd.DataFrame]:
    datasets = list(dict.fromkeys(frame.dataset.astype(str)))
    dataset_index = frame.dataset.map({name: i for i, name in enumerate(datasets)}).to_numpy()
    p = frame[f"profile_{profile_name}"].to_numpy()
    w = frame.w_fitted_pb_per_GeV.to_numpy()
    nlo = frame.mcfm_nlo_pb_per_GeV.to_numpy()
    data = frame.CS.to_numpy()
    error = frame.error.to_numpy()
    base = (1.0 - p) * w + p * nlo
    matching = (1.0 - p) * (nlo - w)
    scale = p * nlo * UPWARD_NLO_SCALE
    norm_sigma = np.asarray([norm_prior[name] for name in datasets])

    def residual(theta: np.ndarray) -> np.ndarray:
        norms = theta[:len(datasets)]
        eta_match, eta_scale = theta[-2:]
        prediction = norms[dataset_index] * (base + eta_match * matching + eta_scale * scale)
        return np.concatenate([
            (prediction - data) / error,
            (norms - 1.0) / norm_sigma,
            np.asarray([eta_match, eta_scale]),
        ])

    start_norms = np.asarray([
        frame.loc[frame.dataset == name, "norm_scale"].iloc[0] for name in datasets
    ])
    start = np.concatenate([start_norms, [1.45, 1.20]])
    result = least_squares(
        residual, start,
        bounds=(np.concatenate([np.full(len(datasets), 0.5), [-5.0, -5.0]]),
                np.concatenate([np.full(len(datasets), 1.5), [5.0, 5.0]])),
        xtol=1.0e-13, ftol=1.0e-13, gtol=1.0e-13, max_nfev=10000,
    )
    norms = result.x[:len(datasets)]
    eta_match, eta_scale = result.x[-2:]
    prediction = norms[dataset_index] * (base + eta_match * matching + eta_scale * scale)
    pulls = (prediction - data) / error
    norm_penalty = float(np.sum(((norms - 1.0) / norm_sigma) ** 2))
    theory_penalty = float(eta_match**2 + eta_scale**2)
    fitted = frame[["dataset", "row_id", "qT_over_Q", "CS", "error"]].copy()
    fitted["profile"] = profile_name
    fitted["fitted_prediction_pb_per_GeV"] = prediction
    fitted["pull"] = pulls
    fitted["fitted_dataset_norm"] = norms[dataset_index]
    fitted["matching_direction_pb_per_GeV_before_norm"] = matching
    fitted["nlo_scale_direction_pb_per_GeV_before_norm"] = scale
    summary = {
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "n_function_evaluations": int(result.nfev),
        "fitted_dataset_norms": {name: float(norms[i]) for i, name in enumerate(datasets)},
        "dataset_norm_prior_widths": {name: float(norm_sigma[i]) for i, name in enumerate(datasets)},
        "matching_nuisance_sigma": float(eta_match),
        "nlo_scale_nuisance_sigma": float(eta_scale),
        "data_chi2": float(np.dot(pulls, pulls)),
        "data_chi2_per_row": float(np.mean(pulls**2)),
        "dataset_norm_penalty_chi2": norm_penalty,
        "theory_nuisance_penalty_chi2": theory_penalty,
        "total_chi2": float(np.dot(pulls, pulls) + norm_penalty + theory_penalty),
        "max_absolute_pull": float(np.max(np.abs(pulls))),
    }
    return summary, fitted


def main() -> None:
    frame = pd.read_csv(INPUT)
    production = pd.read_csv(PRODUCTION / "predictions.csv")
    norm_prior = production.groupby("dataset").norm_rel_used.first().to_dict()
    summaries = {}
    fitted_rows = []
    for profile in PROFILES:
        summaries[profile], fitted = fit_profile(frame, profile, norm_prior)
        fitted_rows.append(fitted)
    central = summaries["central_0p20_0p30"]
    status = {
        "status": "experimental_unitary_transition_not_production",
        "test": "frozen_fnp_exploratory_fit_with_profiled_norm_and_theory_nuisances",
        "row_count": int(len(frame)),
        "fnp_state_frozen": True,
        "dataset_normalizations_profiled": True,
        "matching_and_nlo_scale_nuisances_profiled": True,
        "summaries": summaries,
        "exploratory_profile_fit_pass": bool(
            central["optimizer_success"]
            and central["data_chi2_per_row"] < 1.5
            and abs(central["matching_nuisance_sigma"]) < 2.0
            and abs(central["nlo_scale_nuisance_sigma"]) < 2.0
            and central["max_absolute_pull"] < 3.0
        ),
        "full_differentiable_fnp_refit_authorized": False,
        "replica_stability_authorized": False,
        "production_promotion_authorized": False,
        "next_gate": "if this profile fit passes, build differentiable added-row kernels before an FNP refit",
    }
    TARGET.mkdir(parents=True, exist_ok=True)
    pd.concat(fitted_rows, ignore_index=True).to_csv(TARGET / "fitted_rows.csv", index=False)
    (TARGET / "fit_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
