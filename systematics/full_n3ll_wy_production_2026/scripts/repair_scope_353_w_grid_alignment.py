#!/usr/bin/env python3
"""Repair the isolated scope-353 W-grid row alignment by invariant kinematics.

The external W cache uses source row labels from an older filtered data
directory.  The new scope directory renumbers rows densely, so selecting the
cache by row_id silently pairs some data with another qT/Q bin.  This utility
matches the fixed W cache to the scope data using dataset, kinematics, and the
central cross section, then assigns the trainer row IDs from the explicit scope
map.  It writes only a new diagnostic grid.
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "reports/scope_353_fnp_inputs/data_with_csv_uncertainties"
ROW_MAP = BASE / "reports/scope_353_fnp_inputs/scope_row_id_map.csv"
W_CACHE = (BASE.parent.parent / "outputs/v23a_tevatron_plus_lhcb7_fidacc_merged_cache/"
           "backend_cache/wpert_v23a_tevatron_plus_lhcb7_fidacc_b160.csv")
OUT_BASE = BASE / "reports/scope_353_fnp_inputs"


def key(frame: pd.DataFrame) -> pd.Series:
    return (frame["dataset"].astype(str) + "|" +
            frame["qT"].map(lambda v: f"{float(v):.10f}") + "|" +
            frame["QM"].map(lambda v: f"{float(v):.10f}") + "|" +
            frame["x1"].map(lambda v: f"{float(v):.10f}") + "|" +
            frame["x2"].map(lambda v: f"{float(v):.10f}") + "|" +
            frame["CS"].map(lambda v: f"{float(v):.10f}"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply-lhcb-factor", action="store_true",
                    help="Apply the fiducial factor.  This is retained only for comparison; the external W cache used by the frozen baseline already follows the fiducial convention.")
    ap.add_argument("--tag", default="kinematic_corrected_nofidfactor")
    args = ap.parse_args()
    out_path = OUT_BASE / f"scope_353_bspace_w_{args.tag}.csv"
    manifest_path = OUT_BASE / f"scope_353_bspace_w_{args.tag}.json"
    row_map = pd.read_csv(ROW_MAP)
    scope_to_trainer = dict(zip(row_map.scope_row_id.astype(str),
                                row_map.trainer_row_id.astype(str)))
    scope = pd.concat([pd.read_csv(p) for p in sorted(DATA.glob("*.csv"))], ignore_index=True)
    scope = scope[scope.row_id.astype(str).isin(scope_to_trainer)].copy()
    scope["_key"] = key(scope)
    if scope["_key"].duplicated().any():
        raise RuntimeError("scope data has duplicate invariant row keys")
    w = pd.read_csv(W_CACHE)
    w["_key"] = key(w)
    w_meta = w.drop_duplicates("_key")[["_key", "row_id"]].set_index("_key")
    missing = sorted(set(scope._key) - set(w_meta.index))
    if missing:
        raise RuntimeError(f"W cache is missing {len(missing)} scope kinematic rows")
    pieces = []
    max_rel = []
    for _, row in scope.sort_values(["dataset", "row_id"]).iterrows():
        source_row_id = str(w_meta.loc[row["_key"], "row_id"])
        curve = w[w.row_id.astype(str).eq(source_row_id)].copy()
        curve = curve.sort_values("bT")
        if len(curve) != 160 or not np.isfinite(curve.Wpert_CS.to_numpy(float)).all():
            raise RuntimeError(f"invalid W curve for {row['row_id']}")
        # The reusable external W cache is already in the convention used by
        # the frozen baseline.  Applying the LHCb acceptance factor here would
        # double count it.  Keep an explicit opt-in only for an audit control.
        factor = (float(row.get("theory_fiducial_factor", 1.0))
                  if args.apply_lhcb_factor and str(row["dataset"]) == "LHCb_7" else 1.0)
        out = curve[["bT", "Wpert_CS"]].copy()
        out["Wpert_CS"] *= factor
        out["row_id"] = scope_to_trainer[str(row["row_id"])]
        pieces.append(out[["row_id", "bT", "Wpert_CS"]])
        max_rel.append(abs(factor - 1.0))
    result = pd.concat(pieces, ignore_index=True)
    result.to_csv(out_path, index=False)
    payload = {
        "status": "isolated_scope_353_w_grid_kinematic_alignment_repaired",
        "scope_rows": int(len(scope)), "b_nodes": int(result.bT.nunique()),
        "output": str(out_path), "source_w_cache": str(W_CACHE),
        "scope_data": str(DATA), "row_map": str(ROW_MAP),
        "apply_lhcb_factor": bool(args.apply_lhcb_factor),
        "lhcb_fiducial_factor_max_abs_shift": float(max(max_rel) if max_rel else 0.0),
        "frozen_production_modified": False, "promotion_authorized": False,
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
