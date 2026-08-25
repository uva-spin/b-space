#!/usr/bin/env python3
"""Write a filtered candidate data directory for the 353-row interface."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
PROJECT = BASE.parent.parent
SCOPE = BASE / "reports/tevatron_353_candidate_g1_1p017_expcreg2p0/candidate_353_full_wy.csv"
FIXED_ROOT = PROJECT / "Data/v23a_fixed_target_plus_tevatron_absolute_fit_ready"
LHCB_ROOT = PROJECT / "Data/v23a_tevatron_plus_lhcb7_fiducial_candidate"
OUT = BASE / "reports/scope_353_fnp_inputs/data"


def main() -> None:
    scope = pd.read_csv(SCOPE)
    ids = set(scope.row_id.astype(str))
    datasets = sorted(scope.dataset.astype(str).unique())
    records = {}
    mapping_rows = []
    OUT.mkdir(parents=True, exist_ok=True)
    for ds in datasets:
        root = LHCB_ROOT if ds == "LHCb_7" else FIXED_ROOT
        frame = pd.read_csv(root / f"{ds}.csv")
        frame = frame[frame.row_id.astype(str).isin(ids)].copy()
        if frame.empty:
            raise RuntimeError(f"no scoped rows found for {ds}")
        frame["scope_row_id"] = frame.row_id.astype(str)
        frame["_scope_index"] = frame["scope_row_id"].str.rsplit(":", n=1).str[-1].astype(int)
        frame = frame.sort_values("_scope_index").reset_index(drop=True)
        for new_index, old_id in enumerate(frame.scope_row_id.astype(str)):
            mapping_rows.append({"dataset": ds, "scope_row_id": old_id, "trainer_row_id": f"{ds}:{new_index}", "trainer_local_index": new_index})
        frame = frame.drop(columns=["_scope_index"])
        frame.to_csv(OUT / f"{ds}.csv", index=False)
        records[ds] = int(len(frame))
    status = {
        "status": "isolated_scope_353_data_directory_complete_not_production",
        "row_count": int(sum(records.values())), "dataset_counts": records,
        "source_fixed_target_tevatron": str(FIXED_ROOT), "source_lhcb": str(LHCB_ROOT),
        "output": str(OUT), "frozen_baseline_unchanged": True,
        "production_outputs_modified": False, "promotion_authorized": False,
    }
    pd.DataFrame(mapping_rows).to_csv(OUT.parent / "scope_row_id_map.csv", index=False)
    (OUT.parent / "scope_353_data_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__": main()
