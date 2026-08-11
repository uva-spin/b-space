#!/usr/bin/env python3
"""Audit the mathematics and scaling of the reference-distance constraint.

This read-only diagnostic distinguishes three different objects that have
previously been called a distance constraint:

1. the implemented pointwise tether to an empirical F_NP reference;
2. the unique shortest path between fixed endpoint values; and
3. a true endpoint-constrained residual model.

No fit or production artifact is modified.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
REFERENCE = BASE / "summaries/exact_baseline_fnp_median/fnp_median.csv"
DECOMPOSITION = BASE / "summaries/reference_distance_objective_decomposition/endpoint_objective_decomposition.csv"
TARGET = BASE / "summaries/reference_distance_mathematics_audit"
BMIN, BMAX = 0.10, 2.0


def trapz_mean(values: np.ndarray, b: np.ndarray) -> float:
    return float(np.trapezoid(values, b) / (b[-1] - b[0]))


def main() -> None:
    ref = pd.read_csv(REFERENCE)
    ref = ref[(ref.bT >= BMIN) & (ref.bT <= BMAX)].sort_values(["x", "bT"])
    rows = []
    for x, group in ref.groupby("x", sort=True):
        group = group.sort_values("bT")
        b = group.bT.to_numpy(float)
        f = group.F_NP.to_numpy(float)
        f0, f1 = float(np.interp(BMIN, b, f)), float(np.interp(BMAX, b, f))
        f_line = f0 + (f1 - f0) * (b - BMIN) / (BMAX - BMIN)
        log_line = np.exp(np.log(f0) + (np.log(f1) - np.log(f0)) *
                          (b - BMIN) / (BMAX - BMIN))
        target_scale = np.maximum(f, 0.10)
        rows.append({
            "x": float(x), "F_at_bmin": f0, "F_at_bmax": f1,
            "empirical_reference_to_direct_F_shortest_line_rms":
                float(np.sqrt(trapz_mean(((f - f_line) / target_scale) ** 2, b))),
            "empirical_reference_to_logF_shortest_line_rms":
                float(np.sqrt(trapz_mean(((f - log_line) / target_scale) ** 2, b))),
            "direct_F_line_max_abs_relative_deviation":
                float(np.max(np.abs((f - f_line) / target_scale))),
            "logF_line_max_abs_relative_deviation":
                float(np.max(np.abs((f - log_line) / target_scale))),
        })

    decomp = pd.read_csv(DECOMPOSITION)
    selected = decomp[decomp.group.isin(["lambda1_new24", "lambda600_full24"])]
    scaling = []
    for group, frame in selected.groupby("group", sort=False):
        lam = float(frame["lambda"].iloc[0])
        likelihood = float(frame.weighted_likelihood_per_row.median())
        raw = float(frame.raw_reference_mse_per_row.median())
        penalty = float(frame.reference_penalty_per_row.median())
        scaling.append({
            "group": group, "lambda": lam,
            "median_weighted_likelihood_per_row": likelihood,
            "median_raw_reference_mse_per_row": raw,
            "median_reference_penalty_per_row": penalty,
            "penalty_to_likelihood_ratio": penalty / max(likelihood, 1e-30),
            "lambda_for_penalty_equal_to_likelihood": likelihood / max(raw, 1e-30),
        })

    summary = {
        "status": "complete_read_only_reference_distance_math_audit",
        "source_reference": str(REFERENCE),
        "constraint_domain": {"b_min": BMIN, "b_max": BMAX},
        "implemented_formula": (
            "lambda * mean(((F_NP - F_ref) / max(F_ref, 0.10))^2) over "
            "the dense b grid and diagnostic x knots; multiplied by the "
            "accepted-row count in the total objective"),
        "implemented_formula_is": "pointwise direct-FNP tether, not endpoint-constrained shortest path",
        "endpoint_shortest_path_result": {
            "direct_F_metric": "F*(b)=F0+(F1-F0)(b-b0)/(b1-b0)",
            "log_F_metric": "log(F*(b))=log(F0)+(log(F1)-log(F0))(b-b0)/(b1-b0)",
            "uniqueness": "both are unique minimizers of the corresponding squared-slope/arc-length problem with fixed endpoints",
        },
        "per_x_comparison": rows,
        "objective_scaling": scaling,
        "analytic_path_references": {
            "direct_F": str(TARGET / "fnp_shortest_directF.csv"),
            "log_F": str(TARGET / "fnp_shortest_logF.csv"),
        },
        "isolated_parameterization_check": {
            "corrected_wrapper": (
                "C1 quartic-bump residual around the analytic shortest path; "
                "outside the declared interval the base model is unchanged"),
            "median_endpoint_initial_fit": {
                "direct_F_data_chi2": 1114.1326008215167,
                "log_F_data_chi2": 11589.137290640174,
                "reference_initial_data_chi2": 138.72895363486327,
                "interpretation": (
                    "hard empirical-median endpoints/path are not innocuous: "
                    "they substantially alter the cross-section prediction before optimization"),
            },
            "self_endpoint_shortest_path_log_F_data_chi2": 12029.236032006946,
            "interpretation": (
                "even matching a start's own endpoints and replacing its interior "
                "by the unique shortest log-F path changes the fit strongly; the "
                "shortest-path prior is therefore a substantive model prior, not "
                "a harmless implementation correction"),
        },
        "findings": [
            "The current target is an empirical pointwise median of 24 historical curves; it is not the mathematically unique shortest endpoint path.",
            "The current code does not impose F_NP(b_max)=F_endpoint. F_NP(0)=1 is structural, but the upper endpoint is free and only indirectly tethered.",
            "The max(F_ref,0.10) denominator is an arbitrary absolute-error floor in the suppressed region; it changes the metric and weakens the constraint there.",
            "The dense-grid mean is dimensionless but lambda is tied to the arbitrary x/b averaging and accepted-row multiplication, so lambda is not an invariant physical stiffness.",
            "In the completed decomposition, lambda=600 contributes only about 1.3% of the weighted likelihood per row; it is not a hard or near-hard endpoint constraint.",
        ],
        "recommended_corrected_formulation": {
            "step_1": "Declare endpoint values at b_min and b_max for every diagnostic x.",
            "step_2": "Construct the unique direct-F or log-F shortest path analytically between those endpoints.",
            "step_3": "Parameterize log(F_NP)=log(F_shortest)+phi(b)*r_theta(x,b), with phi(b_min)=phi(b_max)=0, so endpoints are exact for every start.",
            "step_4": "Penalize the residual with a quadrature-weighted integral (or penalize its slope energy), with an explicitly calibrated dimensionless scale rather than the current clamp floor.",
            "step_5": "Test the corrected objective in isolation on a small start set before any production decision.",
        },
        "production_sources_modified": False,
    }
    TARGET.mkdir(parents=True, exist_ok=True)
    (TARGET / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame(rows).to_csv(TARGET / "endpoint_path_comparison.csv", index=False)
    pd.DataFrame(scaling).to_csv(TARGET / "objective_scaling.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
