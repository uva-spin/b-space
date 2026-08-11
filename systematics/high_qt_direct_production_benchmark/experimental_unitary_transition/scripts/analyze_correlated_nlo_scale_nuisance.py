#!/usr/bin/env python3
"""Profile a fully correlated NLO scale nuisance through the unitary prediction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "systematics/high_qt_direct_production_benchmark/experimental_unitary_transition"
INPUT = BASE / "summaries/unitary_smootherstep_v1_nlo_24row/nlo_frozen_unitary_pulls.csv"
SCALE_STATUS = BASE / "summaries/unitary_smootherstep_v1_fo_order_audit/nlo_representative_status.json"
TARGET = BASE / "summaries/unitary_smootherstep_v1_nlo_correlated_scale"
PROFILES = ("early_0p18_0p28", "central_0p20_0p30", "late_0p22_0p32")


def averaged_profile(frame: pd.DataFrame, start: float, end: float) -> np.ndarray:
    nodes, weights = np.polynomial.legendre.leggauss(16)
    values = []
    for row in frame.itertuples():
        half = 0.5 * (row.qT_high - row.qT_low)
        qt = 0.5 * (row.qT_high + row.qT_low) + half * nodes
        t = np.clip((qt / row.QM - start) / (end - start), 0.0, 1.0)
        profile = t**3 * (t * (6.0 * t - 15.0) + 10.0)
        values.append(float(0.5 * np.dot(weights, profile)))
    return np.asarray(values)


def profile_scale(frame: pd.DataFrame, profile: np.ndarray, upward: float) -> dict[str, float]:
    prediction = ((1.0 - profile) * frame.w_fitted_pb_per_GeV
                  + profile * frame.mcfm_nlo_pb_per_GeV) * frame.norm_scale
    residual = (prediction - frame.CS) / frame.error
    response = profile * frame.mcfm_nlo_pb_per_GeV * frame.norm_scale * upward / frame.error
    nuisance = max(0.0, -float(np.dot(response, residual)) / (1.0 + float(np.dot(response, response))))
    pull = residual + nuisance * response
    return {
        "best_fit_upward_scale_nuisance_sigma": nuisance,
        "data_chi2": float(np.dot(pull, pull)),
        "total_profiled_chi2": float(np.dot(pull, pull) + nuisance**2),
        "data_chi2_per_row": float(np.mean(pull**2)),
        "max_absolute_pull": float(np.max(np.abs(pull))),
    }


def main() -> None:
    frame = pd.read_csv(INPUT)
    scale = json.loads(SCALE_STATUS.read_text())
    upward = float(np.mean([
        scale["anchor_scale_envelope_relative_to_central"][1],
        scale["midpoint_scale_envelope_relative_to_central"][1],
    ]))
    downward = float(-np.mean([
        scale["anchor_scale_envelope_relative_to_central"][0],
        scale["midpoint_scale_envelope_relative_to_central"][0],
    ]))
    summaries = {}
    for profile in PROFILES:
        central = frame[f"{profile}_accepted_norm_prediction"].to_numpy()
        residual = (central - frame.CS.to_numpy()) / frame.error.to_numpy()
        response = (
            frame[f"profile_{profile}"].to_numpy()
            * frame.mcfm_nlo_pb_per_GeV.to_numpy()
            * frame.norm_scale.to_numpy()
            * upward
        )
        standardized_response = response / frame.error.to_numpy()
        nuisance = max(0.0, -float(np.dot(standardized_response, residual)) /
                       (1.0 + float(np.dot(standardized_response, standardized_response))))
        pull = residual + nuisance * standardized_response
        frame[f"{profile}_scale_response_pb_per_GeV"] = response
        frame[f"{profile}_profiled_scale_nuisance"] = nuisance
        frame[f"{profile}_profiled_prediction"] = central + nuisance * response
        frame[f"{profile}_profiled_pull"] = pull
        summaries[profile] = {
            "best_fit_upward_scale_nuisance_sigma": nuisance,
            "data_chi2": float(np.dot(pull, pull)),
            "nuisance_penalty_chi2": nuisance**2,
            "total_profiled_chi2": float(np.dot(pull, pull) + nuisance**2),
            "data_chi2_per_row": float(np.mean(pull**2)),
            "median_absolute_pull": float(np.median(np.abs(pull))),
            "max_absolute_pull": float(np.max(np.abs(pull))),
        }

    central_profile = "central_0p20_0p30"
    locked = frame.profile_central_0p20_0p30 >= 0.95
    locked_pull = frame.loc[locked, f"{central_profile}_profiled_pull"]
    scan_rows = []
    for start in np.arange(0.12, 0.221, 0.005):
        for width in np.arange(0.04, 0.141, 0.005):
            result = profile_scale(frame, averaged_profile(frame, start, start + width), upward)
            scan_rows.append({"r_start": start, "r_end": start + width, "width": width, **result})
    scan = pd.DataFrame(scan_rows)
    best = scan.loc[scan.total_profiled_chi2.idxmin()]
    preserving = scan.loc[scan.r_start >= 0.20]
    best_preserving = preserving.loc[preserving.total_profiled_chi2.idxmin()]
    conventional = scan.loc[(scan.r_start >= 0.18) & (scan.width >= 0.08)]
    best_conventional = conventional.loc[conventional.total_profiled_chi2.idxmin()]
    status = {
        "status": "experimental_unitary_transition_not_production",
        "test": "fully_correlated_asymmetric_nlo_scale_nuisance",
        "row_count": int(len(frame)),
        "scale_response_source": "mean of independently converged CDF_RUN_2:45 and :51 seven-point envelopes",
        "one_sigma_relative_scale_response": {"downward": -downward, "upward": upward},
        "internal_mcfm_scale_reweighting_accepted": False,
        "internal_reweighting_rejection_reason": (
            "The production-statistics anchor gave +9.3% at correlated low scales, "
            "inconsistent with the independent +19.6% anchor and +18.8% midpoint envelopes."
        ),
        "correlation_model": "one fully correlated Gaussian nuisance acting on the NLO component of every row",
        "profiled_summaries": summaries,
        "central_locked_subset_row_count": int(locked.sum()),
        "central_locked_subset_profiled_chi2": float(np.sum(locked_pull**2)),
        "central_locked_subset_profiled_chi2_per_row": float(np.mean(locked_pull**2)),
        "profile_scan_best": {
            "r_start": float(best.r_start), "r_end": float(best.r_end),
            "data_chi2_per_row": float(best.data_chi2_per_row),
            "best_fit_upward_scale_nuisance_sigma": float(best.best_fit_upward_scale_nuisance_sigma),
        },
        "profile_scan_best_boundary_preserving": {
            "r_start": float(best_preserving.r_start), "r_end": float(best_preserving.r_end),
            "data_chi2_per_row": float(best_preserving.data_chi2_per_row),
        },
        "profile_scan_best_start_ge_0p18_width_ge_0p08": {
            "r_start": float(best_conventional.r_start), "r_end": float(best_conventional.r_end),
            "data_chi2_per_row": float(best_conventional.data_chi2_per_row),
        },
        "profile_scan_diagnosis": (
            "Acceptable frozen agreement requires an NLO-dominated transition by r about 0.21, "
            "which intrudes below the accepted r<=0.20 production boundary."
        ),
        "correlated_scale_treatment_pass": False,
        "central_refit_authorized": False,
        "replica_stability_authorized": False,
        "decision": "scale_uncertainty_does_not_resolve_transition_candidate",
        "reason": (
            "The central profile requires a 2.59-sigma upward scale excursion and still has excessive "
            "transition-region and fixed-order-locked pulls."
        ),
        "required_next_experiment": (
            "do not promote this transition candidate; investigate the W/NLO incompatibility at the accepted r=0.20 boundary"
        ),
    }
    TARGET.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TARGET / "profiled_correlated_scale_pulls.csv", index=False)
    scan.to_csv(TARGET / "profile_window_scan.csv", index=False)
    (TARGET / "gate_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
