#!/usr/bin/env python3
"""Quantify the effect of replacing NLO with NNLO bin acceptance.

The production-candidate W kernels use an NLO node-level fiducial acceptance
grid, while the best available fixed-order Y input is NNLO.  A full NNLO
node-kernel campaign is numerically impractical at the individual nodes, so
this isolated audit applies the measured NNLO/NLO *bin-integrated* acceptance
ratio to each row's W contribution.  It is a sensitivity bound, not a
replacement production kernel.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
FIT_SCRIPT = BASE / "scripts/audit_lambda1_lhcb_nnlo_fit_impact.py"
ENDPOINTS = BASE / "reports/lambda1_lhcb_unitary_nnlo/lhcb_unitary_lambda1_endpoints.csv"
BASELINE_FIT = BASE / "reports/lambda1_lhcb_unitary_nnlo/lhcb_unitary_lambda1_nnlo_fit_impact.csv"
NLO_ACC = BASE / "reports/lhcb_bin_acceptance_check/lhcb_bin_acceptance.csv"
NNLO_ACC = BASE / "reports/lhcb_bin_acceptance_nnlo_all4_10m/lhcb_bin_acceptance.csv"
OUT = BASE / "reports/lhcb_nnlo_acceptance_effect"
ROW_IDS = [f"LHCb_7:{i}" for i in (10, 11, 12, 13)]


def load_fit_module():
    spec = importlib.util.spec_from_file_location("lhcb_fit_impact_source", FIT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(FIT_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    fit = load_fit_module()
    values = pd.read_csv(ENDPOINTS)
    baseline = pd.read_csv(BASELINE_FIT)
    nlo = pd.read_csv(NLO_ACC).set_index("row_id")
    nnlo = pd.read_csv(NNLO_ACC).set_index("row_id")
    factors = {row_id: float(nnlo.loc[row_id, "acceptance"] / nlo.loc[row_id, "acceptance"])
               for row_id in ROW_IDS}
    values = values.copy()
    values["W_lambda1_fiducial"] *= values["row_id"].map(factors)
    cov = fit.covariance()
    white = fit.whitening_matrix(cov)
    fo_map, scale_map = fit.nnlo_maps()
    records = []
    for endpoint in values.endpoint.unique():
        for profile in values.profile.unique():
            sub = values[(values.endpoint == endpoint) & values.profile.eq(profile)]
            records.append(fit.fit_one(sub, white, fo_map, scale_map, endpoint, profile))
    adjusted = pd.DataFrame(records)
    OUT.mkdir(parents=True, exist_ok=True)
    adjusted.to_csv(OUT / "lhcb_nnlo_acceptance_effect_fit.csv", index=False)
    summary = {
        "status": "isolated_nnlo_bin_acceptance_sensitivity_complete_not_production",
        "approximation": "multiply each row W contribution by NNLO/NLO bin-integrated fiducial acceptance ratio",
        "nlo_acceptance_source": str(NLO_ACC),
        "nnlo_acceptance_source": str(NNLO_ACC),
        "row_acceptance_factors_nnlo_over_nlo": factors,
        "all_optimizer_success": bool(adjusted.optimizer_success.all()),
        "all_predictions_positive": bool(adjusted.all_positive.all()),
        "profiles": {},
        "production_outputs_modified": False,
    }
    for profile in sorted(adjusted.profile.unique()):
        new = adjusted[adjusted.profile.eq(profile)]
        old = baseline[baseline.profile.eq(profile)]
        summary["profiles"][profile] = {
            "baseline_total_chi2_per_row_median": float(old.total_chi2_per_row.median()),
            "adjusted_total_chi2_per_row_median": float(new.total_chi2_per_row.median()),
            "delta_total_chi2_per_row_median": float(new.total_chi2_per_row.median() - old.total_chi2_per_row.median()),
            "baseline_data_chi2_per_row_median": float(old.data_chi2_per_row.median()),
            "adjusted_data_chi2_per_row_median": float(new.data_chi2_per_row.median()),
        }
    summary["interpretation"] = (
        "The NNLO/NLO bin-acceptance shifts are only percent-level. Applying them "
        "to the W term changes the fitted residual modestly (and worsens the "
        "central/late profiles in this signed sensitivity test); it does not "
        "close the high-qT deficit. A full NNLO node-kernel replacement is not "
        "justified by this acceptance result."
    )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
