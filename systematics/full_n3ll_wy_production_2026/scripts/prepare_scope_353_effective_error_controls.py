#!/usr/bin/env python3
"""Build isolated candidate data with the frozen low-qT effective errors.

The new 353-row data files contain the raw table errors.  The established
329-row production baseline instead uses the already-audited Collins
factorization effective-error inflation.  This utility transfers only the
per-row uncertainty fields for overlapping row IDs into a new diagnostic data
directory; central values and kinematics are untouched.  It never edits the
source data or frozen production outputs.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
SRC = BASE / "reports/scope_353_fnp_inputs/data_with_csv_uncertainties"
ROW_MAP = BASE / "reports/scope_353_fnp_inputs/scope_row_id_map.csv"
OLD = (BASE.parent / "collins_factorization_validity/outputs/"
       "rowidfix_stageFT_all_qmax0p20_lam0p50_central_s303/predictions.csv")
OUT = BASE / "reports/scope_353_fnp_inputs/data_with_old_effective_errors"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(OLD)
    row_map = pd.read_csv(ROW_MAP)
    scope_to_trainer = dict(zip(row_map.scope_row_id.astype(str),
                                row_map.trainer_row_id.astype(str)))
    def make_key(frame: pd.DataFrame) -> pd.Series:
        # Source row labels are not stable after the qT cut; central kinematics
        # and the published central value are the invariant row identity.
        return (frame["dataset"].astype(str) + "|" +
                frame["qT"].map(lambda v: f"{float(v):.10f}") + "|" +
                frame["QM"].map(lambda v: f"{float(v):.10f}") + "|" +
                frame["CS"].map(lambda v: f"{float(v):.10f}"))
    old["_key"] = make_key(old)
    if old["_key"].duplicated().any():
        raise RuntimeError("effective-error reference has duplicate invariant row keys")
    old = old.set_index("_key")
    summary = {"source": str(SRC), "effective_error_reference": str(OLD),
               "row_id_map": str(ROW_MAP), "files": {},
               "central_values_modified": False, "frozen_production_modified": False}
    for src_path in sorted(SRC.glob("*.csv")):
        df = pd.read_csv(src_path)
        out = df.copy()
        keys = make_key(out)
        overlap = keys.isin(old.index)
        for col in ("error", "sysP2P_rel", "sysNorm_rel"):
            if col in out and col in old.columns:
                vals = out[col].to_numpy(copy=True)
                for i, key in enumerate(keys):
                    if key in old.index:
                        vals[i] = float(old.loc[key, col])
                out[col] = vals
        source_ids = out["row_id"].astype(str)
        out["scope_row_id_original"] = source_ids
        out["row_id"] = source_ids.map(scope_to_trainer)
        if out["row_id"].isna().any():
            raise RuntimeError(f"row-ID remapping is incomplete for {src_path.name}")
        # Keep all rows, including the 24 high-qT rows with no frozen reference.
        target = OUT / src_path.name
        out.to_csv(target, index=False)
        summary["files"][src_path.name] = {
            "rows": int(len(out)), "overlap_rows": int(overlap.sum()),
            "raw_only_rows": int((~overlap).sum()),
            "error_ratio_min": float(np.nanmin(out.loc[overlap, "error"].to_numpy(float) /
                                                df.loc[overlap, "error"].to_numpy(float))) if overlap.any() else None,
            "error_ratio_max": float(np.nanmax(out.loc[overlap, "error"].to_numpy(float) /
                                                df.loc[overlap, "error"].to_numpy(float))) if overlap.any() else None,
        }
    (OUT / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
