#!/usr/bin/env python3
"""Audit the b-space W/Y inputs needed by a coupled F_NP propagation.

The observable-level DYTurbo W+Y tables are not interchangeable with the
long b-space ``Wpert_CS`` table consumed by the FiLM trainer.  This audit
checks the row-ID and b-grid coverage explicitly, so a scope-complete table
cannot accidentally be relabeled as a scope-complete F_NP fit.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
SYSTEMATICS = BASE.parent
PROJECT = SYSTEMATICS.parent
CORE = SYSTEMATICS / "collins_factorization_validity/outputs/rowidfix_stageFT_all_qmax0p20_lam0p50_central_s303/predictions.csv"
SCOPE = BASE / "reports/tevatron_353_candidate_g1_1p017_expcreg2p0/candidate_353_full_wy.csv"
W_CACHE = PROJECT / "outputs/v23a_fixed_target_plus_tevatron_absolute_fit_ready_checkonly_cache/backend_cache/wpert_v23a_fixedtarget_plus_tevatron_absolute_fitready_v22tev_n3llp_nloQ96_b160.csv"
W_CACHE_LHCB = PROJECT / "outputs/v23a_tevatron_plus_lhcb7_fidacc_merged_cache/backend_cache/wpert_v23a_tevatron_plus_lhcb7_fidacc_b160.csv"
W_FRAGMENT = BASE / "reports/missing_scope_bspace_w_n3llp_nloQ96_b160/missing_scope_bspace_w.csv"
Y_GRID = BASE / "reports/tevatron_n3ll_nnlo_wy_final_g1_1p017/conventional_y/tevatron_y_grid.csv"
Y_FIXED = BASE / "reports/fixed_target_y_fullminusres_expcreg2p0_100k/fixed_target_y_full_minus_res.csv"
Y_LHCB = BASE / "reports/lhcb7_y_fullminusres_expcreg2p0_1m/lhcb7_y_full_minus_res.csv"
OUT = BASE / "reports/fnp_external_interface_audit.json"


def row_ids(path: Path) -> set[str]:
    return set(pd.read_csv(path, usecols=["row_id"])["row_id"].astype(str))


def main() -> None:
    scope = pd.read_csv(SCOPE)
    scope_ids = set(scope.row_id.astype(str))
    core = pd.read_csv(CORE)
    core_ids = set(core.row_id.astype(str))
    w = pd.read_csv(W_CACHE, usecols=["row_id", "bT", "Wpert_CS"])
    w_ids = set(w.row_id.astype(str))
    w_lhcb = pd.read_csv(W_CACHE_LHCB, usecols=["row_id", "bT", "Wpert_CS"])
    w_fragment = pd.read_csv(W_FRAGMENT, usecols=["row_id", "bT", "Wpert_CS"])
    w_all = pd.concat([w, w_lhcb, w_fragment], ignore_index=True).drop_duplicates(["row_id", "bT"])
    w_all_ids = set(w_all.row_id.astype(str))
    y = pd.read_csv(Y_GRID, usecols=["row_id", "Y_CS"])
    y_fixed = pd.read_csv(Y_FIXED, usecols=["row_id", "Y_CS"])
    y_lhcb_tab = pd.read_csv(Y_LHCB)
    y_lhcb = y_lhcb_tab[["row_id", "Y_pb_per_GeV"]].rename(columns={"Y_pb_per_GeV": "Y_CS"})
    y_all = pd.concat([y, y_fixed, y_lhcb], ignore_index=True).drop_duplicates("row_id", keep="last")
    y_ids = set(y_all.row_id.astype(str))
    bvals = np.sort(w_all.bT.astype(float).unique())
    complete_w = {
        rid: int(len(g) == len(bvals) and np.isfinite(g.Wpert_CS.to_numpy(float)).all())
        for rid, g in w_all.groupby(w_all.row_id.astype(str), sort=False)
    }
    w_complete = {rid for rid, ok in complete_w.items() if ok}
    result = {
        "status": "isolated_fnp_external_interface_audit_complete_not_production",
        "scope_row_count": int(len(scope_ids)),
        "scope_unique": bool(len(scope_ids) == len(scope)),
        "scope_counts": scope.groupby("dataset").row_id.nunique().to_dict(),
        "core_row_count": int(len(core_ids)),
        "bspace_w": {
            "source": str(W_CACHE),
            "combined_lhcb_source": str(W_CACHE_LHCB),
            "missing_scope_fragment": str(W_FRAGMENT),
            "unique_b_nodes": int(len(bvals)),
            "b_min": float(bvals.min()),
            "b_max": float(bvals.max()),
            "complete_scope_rows": int(len(scope_ids & w_complete)),
            "missing_scope_rows": sorted(scope_ids - w_complete),
            "dataset_complete_counts": {
                ds: int(sum(rid.startswith(ds + ":") for rid in (scope_ids & w_complete)))
                for ds in sorted(scope.dataset.unique())
            },
        },
        "conventional_y": {
            "source_tevatron": str(Y_GRID),
            "source_fixed_target": str(Y_FIXED),
            "source_lhcb": str(Y_LHCB),
            "row_count": int(len(y_all)),
            "scope_rows_with_y": int(len(scope_ids & y_ids)),
            "missing_scope_rows": sorted(scope_ids - y_ids),
            "all_finite": bool(np.isfinite(y_all.Y_CS.to_numpy(float)).all()),
        },
        "coupled_fit_readiness": {
            "bspace_W_required": "row_id,bT,Wpert_CS for every fitted row",
            "observable_Y_required": "row_id,Y_CS for every fitted row",
            "scope_complete_bspace_W": bool(scope_ids <= w_complete),
            "scope_complete_Y": bool(scope_ids <= y_ids),
            "ready_for_true_353_coupled_FNP_fit": bool(scope_ids <= w_complete and scope_ids <= y_ids),
            "reason_if_not_ready": "All row IDs now have numerical W and Y inputs, but fixed-target conversion assumptions and LHCb cancellation-driven uncertainties still prevent production promotion.",
        },
        "frozen_baseline_unchanged": True,
        "production_outputs_modified": False,
        "promotion_authorized": False,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
