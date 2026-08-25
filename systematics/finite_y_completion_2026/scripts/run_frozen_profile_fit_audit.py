#!/usr/bin/env python3
"""Independent frozen-FNP profile/nuisance fit audit for the unitary candidate."""

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
PROFILES = ("early_0p18_0p28", "central_0p20_0p30", "late_0p22_0p32")
UPWARD_NLO_SCALE = 0.19217428727157315


def fit_profile(frame: pd.DataFrame, profile: str, norm_prior: dict[str, float]) -> dict:
    names = list(dict.fromkeys(frame.dataset.astype(str)))
    index = frame.dataset.map({name: i for i, name in enumerate(names)}).to_numpy()
    p = frame[f"profile_{profile}"].to_numpy(float)
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

    start = np.concatenate([[frame.loc[frame.dataset == name, "norm_scale"].iloc[0] for name in names], [1.45, 1.20]])
    result = least_squares(
        residual, start,
        bounds=(np.concatenate([np.full(len(names), 0.5), [-5.0, -5.0]]),
                np.concatenate([np.full(len(names), 1.5), [5.0, 5.0]])),
        xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=10000,
    )
    norms = result.x[:len(names)]
    eta_m, eta_s = result.x[-2:]
    prediction = norms[index] * (base + eta_m * matching + eta_s * scale)
    pulls = (prediction - data) / error
    data_chi2 = float(np.dot(pulls, pulls))
    norm_penalty = float(np.sum(((norms - 1.0) / sigma) ** 2))
    total = data_chi2 + norm_penalty + float(eta_m**2 + eta_s**2)
    return {
        "optimizer_success": bool(result.success),
        "data_chi2_per_row": float(np.mean(pulls**2)),
        "data_chi2": data_chi2,
        "total_chi2": total,
        "matching_nuisance_sigma": float(eta_m),
        "nlo_scale_nuisance_sigma": float(eta_s),
        "max_absolute_pull": float(np.max(np.abs(pulls))),
        "dataset_norms": {name: float(norms[i]) for i, name in enumerate(names)},
        "pass_operational_gate": bool(
            result.success and np.mean(pulls**2) < 1.5 and abs(eta_m) < 2.0 and abs(eta_s) < 2.0 and np.max(np.abs(pulls)) < 3.0
        ),
    }


def main() -> None:
    frame = pd.read_csv(INPUT)
    production = pd.read_csv(PRODUCTION / "predictions.csv")
    norm_prior = production.groupby("dataset").norm_rel_used.first().to_dict()
    summaries = {profile: fit_profile(frame, profile, norm_prior) for profile in PROFILES}
    report = {
        "status": "frozen_fnp_profile_fit_audit_complete",
        "row_count": int(len(frame)),
        "profiles": summaries,
        "central_operational_gate_pass": summaries["central_0p20_0p30"]["pass_operational_gate"],
        "interpretation": "The unitary construction can be fit with endpoint-separated correlated matching/scale nuisances while holding FNP fixed; this is an impact audit, not an FNP or replica production fit.",
        "production_promotion": False,
    }
    out = WORK / "reports/frozen_profile_fit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
