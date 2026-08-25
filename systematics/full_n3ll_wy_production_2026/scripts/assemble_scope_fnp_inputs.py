#!/usr/bin/env python3
"""Assemble row-complete candidate b-space W and observable Y inputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
PROJECT = BASE.parent.parent
SCOPE = BASE / "reports/tevatron_353_candidate_g1_1p017_expcreg2p0/candidate_353_full_wy.csv"
W_CACHE = PROJECT / "outputs/v23a_tevatron_plus_lhcb7_fidacc_merged_cache/backend_cache/wpert_v23a_tevatron_plus_lhcb7_fidacc_b160.csv"
W_FRAGMENT = BASE / "reports/missing_scope_bspace_w_n3llp_nloQ96_b160/missing_scope_bspace_w.csv"
Y_TEVA = BASE / "reports/tevatron_n3ll_nnlo_wy_final_g1_1p017/conventional_y/tevatron_y_grid.csv"
Y_FIXED = BASE / "reports/fixed_target_y_fullminusres_expcreg2p0_100k/fixed_target_y_full_minus_res.csv"
Y_LHCB = BASE / "reports/lhcb7_y_fullminusres_expcreg2p0_1m/lhcb7_y_full_minus_res.csv"
DATA_LHCB = PROJECT / "Data/v23a_tevatron_plus_lhcb7_fiducial_candidate/LHCb_7.csv"
OUT = BASE / "reports/scope_353_fnp_inputs"
ROW_MAP = OUT / "scope_row_id_map.csv"


def main() -> None:
    scope = pd.read_csv(SCOPE)
    ids = set(scope.row_id.astype(str))
    mapping = pd.read_csv(ROW_MAP)
    old_to_new = dict(zip(mapping.scope_row_id.astype(str), mapping.trainer_row_id.astype(str)))
    w = pd.concat([
        pd.read_csv(W_CACHE, usecols=["row_id", "bT", "Wpert_CS"]),
        pd.read_csv(W_FRAGMENT, usecols=["row_id", "bT", "Wpert_CS"]),
    ], ignore_index=True).drop_duplicates(["row_id", "bT"], keep="last")
    # The LHCb data rows are fiducial-acceptance candidates.  The reusable
    # internal W cache is boson-level, so apply the documented row factor to
    # those six W rows before pairing them with the direct fiducial Y table.
    lhcb = pd.read_csv(DATA_LHCB, usecols=["row_id", "theory_fiducial_factor"])
    factors = dict(zip(lhcb.row_id.astype(str), lhcb.theory_fiducial_factor.astype(float)))
    mask_lhcb = w.row_id.astype(str).str.startswith("LHCb_7:")
    w.loc[mask_lhcb, "Wpert_CS"] *= w.loc[mask_lhcb, "row_id"].astype(str).map(factors).to_numpy(float)
    y_parts = [pd.read_csv(Y_TEVA), pd.read_csv(Y_FIXED)]
    if Y_LHCB.exists():
        y_parts.append(pd.read_csv(Y_LHCB))
    y_tables = []
    for tab in y_parts:
        if "Y_CS" in tab.columns:
            y_tables.append(tab[["row_id", "Y_CS"]].copy())
        elif "Y_pb_per_GeV" in tab.columns:
            y_tables.append(tab[["row_id", "Y_pb_per_GeV"]].rename(columns={"Y_pb_per_GeV": "Y_CS"}))
    y = pd.concat(y_tables, ignore_index=True).drop_duplicates("row_id", keep="last")
    b = np.sort(w.bT.astype(float).unique())
    w_complete = {rid for rid, g in w.groupby(w.row_id.astype(str), sort=False)
                  if len(g) == len(b) and np.isfinite(g.Wpert_CS.to_numpy(float)).all()}
    y_ids = set(y.row_id.astype(str))
    missing_w = sorted(ids - w_complete)
    missing_y = sorted(ids - y_ids)
    OUT.mkdir(parents=True, exist_ok=True)
    w_scope = w[w.row_id.astype(str).isin(ids)].copy()
    y_scope = y[y.row_id.astype(str).isin(ids)].copy()
    w_scope["row_id"] = w_scope.row_id.astype(str).map(old_to_new)
    y_scope["row_id"] = y_scope.row_id.astype(str).map(old_to_new)
    if w_scope.row_id.isna().any() or y_scope.row_id.isna().any():
        raise RuntimeError("scope row-ID remapping is incomplete")
    w_scope.to_csv(OUT / "scope_353_bspace_w.csv", index=False)
    y_scope.to_csv(OUT / "scope_353_y.csv", index=False)
    status = {
        "status": "isolated_scope_353_fnp_inputs_ready" if not missing_w and not missing_y else "isolated_scope_353_fnp_inputs_incomplete_not_production",
        "scope_rows": int(len(ids)), "b_nodes": int(len(b)),
        "w_scope_rows_complete": int(len(ids & w_complete)), "y_scope_rows": int(len(ids & y_ids)),
        "missing_w_rows": missing_w, "missing_y_rows": missing_y,
        "ready_for_coupled_fnp_fit": bool(not missing_w and not missing_y),
        "sources": {"w_cache": str(W_CACHE), "w_fragment": str(W_FRAGMENT), "y_tevatron": str(Y_TEVA), "y_fixed": str(Y_FIXED), "y_lhcb": str(Y_LHCB), "lhcb_fiducial_factors": str(DATA_LHCB), "row_id_map": str(ROW_MAP)},
        "frozen_baseline_unchanged": True, "production_outputs_modified": False, "promotion_authorized": False,
    }
    (OUT / "scope_353_fnp_inputs_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__": main()
