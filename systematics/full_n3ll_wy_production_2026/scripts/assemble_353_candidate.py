#!/usr/bin/env python3
"""Assemble the isolated 353-row W+Y candidate without promotion.

Tevatron rows come from the existing high-stat g1=1.017 grid, fixed-target
rows from the complete expcreg=2.0 Eq.(3.3) diagnostic, and LHCb rows from the
1M-call fiducial diagnostic.  This is an auditable union, not a production
cache or a fit replacement.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
SYSTEMATICS = BASE.parent
PROJECT = SYSTEMATICS.parent
OUT = BASE / "reports/tevatron_353_candidate_g1_1p017_expcreg2p0"
SCOPE = BASE / "reports/tevatron_353_scope_audit.json"
TEV = BASE / "reports/tevatron_n3ll_nnlo_wy_final_g1_1p017/tevatron_full_wy_grid.csv"
FIXED = BASE / "reports/fixed_target_expcreg2p0_full_100k/fixed_target_full_wy_grid.csv"
LHC = BASE / "reports/lhcb7_full_n3ll_nnlo_fiducial_expcreg2p0_1m/lhcb7_full_wy_grid.csv"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scope = json.loads(SCOPE.read_text())
    core = pd.read_csv(PROJECT / "systematics/collins_factorization_validity/outputs/rowidfix_stageFT_all_qmax0p20_lam0p50_central_s303/predictions.csv")
    boundary = pd.read_csv(BASE / "reports/dyturbo_full_n3ll_nnlo_boundary_g1_1p017/tevatron_boundary_input.csv")
    tev_ids = set(core.loc[core.dataset.astype(str).isin(["CDF_RUN_1", "CDF_RUN_2", "D0_RUN_1"]), "row_id"].astype(str)) | set(boundary.row_id.astype(str))
    tev = pd.read_csv(TEV)
    tev = tev[tev.row_id.astype(str).isin(tev_ids)].copy()
    if len(tev) != 104:
        raise RuntimeError(f"Tevatron scope resolved to {len(tev)}, expected 104")
    tev_out = tev[["dataset", "row_id", "qT_low", "qT_high", "data_pb_per_GeV", "data_unc_pb_per_GeV",
                   "full_wy_pb_per_GeV", "full_wy_unc_pb_per_GeV", "full_wy_to_data_ratio"]].copy()
    tev_out["source_status"] = "validated_tevatron_external_grid"

    fixed = pd.read_csv(FIXED)
    if len(fixed) != 243:
        raise RuntimeError(f"fixed-target scope resolved to {len(fixed)}, expected 243")
    fixed_out = fixed[["dataset", "row_id", "qT_low", "qT_high", "data_CS_A_over_PreFactor", "data_error",
                       "predicted_CS", "raw_unc_fb_per_bin_nucleus", "predicted_CS_to_data"]].copy()
    fixed_out = fixed_out.rename(columns={"data_CS_A_over_PreFactor": "data_pb_per_GeV",
                                          "data_error": "data_unc_pb_per_GeV",
                                          "predicted_CS": "full_wy_pb_per_GeV",
                                          "raw_unc_fb_per_bin_nucleus": "raw_full_wy_unc_fb_per_bin",
                                          "predicted_CS_to_data": "full_wy_to_data_ratio"})
    fixed_out["full_wy_unc_pb_per_GeV"] = np.nan
    fixed_out["source_status"] = "fixed_target_eq33_expcreg2p0_diagnostic"

    lh = pd.read_csv(LHC)
    if len(lh) != 6:
        raise RuntimeError(f"LHCb scope resolved to {len(lh)}, expected 6")
    lh_out = lh[["dataset", "row_id", "qT_low", "qT_high", "CS", "error", "full_wy_pb_per_GeV",
                 "full_wy_unc_pb_per_GeV", "full_wy_to_data_ratio"]].copy()
    lh_out = lh_out.rename(columns={"CS": "data_pb_per_GeV", "error": "data_unc_pb_per_GeV"})
    lh_out["source_status"] = "lhcb_fiducial_expcreg2p0_1m_diagnostic"

    combined = pd.concat([tev_out, fixed_out, lh_out], ignore_index=True, sort=False)
    if len(combined) != 353 or combined.row_id.astype(str).nunique() != 353:
        raise RuntimeError("combined candidate is not unique 353 rows")
    combined.to_csv(OUT / "candidate_353_full_wy.csv", index=False)
    ratio = combined.full_wy_to_data_ratio.to_numpy(float)
    status = {
        "status": "isolated_353_row_unprimed_n3ll_nnlo_wy_candidate_assembled_not_production",
        "formula": "W_N3LL + (FO_NNLO - ASY_NNLO)", "order": 3, "primed": False,
        "g1_GeV2": 1.017, "expcreg_fixed_target_and_lhcb": 2.0,
        "row_count": int(len(combined)),
        "dataset_row_counts": combined.dataset.value_counts().to_dict(),
        "checks": {
            "all_finite": bool(np.isfinite(ratio).all()),
            "all_positive": bool((ratio > 0).all()),
            "median_prediction_to_data": float(np.median(ratio)),
            "min_prediction_to_data": float(np.min(ratio)),
            "max_prediction_to_data": float(np.max(ratio)),
        },
        "component_sources": {"tevatron": str(TEV), "fixed_target": str(FIXED), "lhcb": str(LHC)},
        "readiness": {
            "tevatron": "validated external W+Y grid",
            "fixed_target": "finite/positive but target and Eq.(3.3) convention diagnostic only",
            "LHCb": "finite/positive at 1M but cancellation-driven MC uncertainty remains large",
            "F_NP_replica_start_propagation": "existing 96-start x 50-replica baseline is separate and not refit to this assembled grid",
        },
        "promotion_authorized": False, "frozen_baseline_unchanged": True,
        "production_outputs_modified": False,
    }
    (OUT / "candidate_353_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
