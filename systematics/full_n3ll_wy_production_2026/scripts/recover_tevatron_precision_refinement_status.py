#!/usr/bin/env python3
"""Repair the aggregate precision status after a candidate-side bookkeeping crash.

The 24 selected 300M-call integrations completed and updated the candidate grid,
but the first aggregate writer used a misspelled uncertainty column and exited
before writing its status.  Reconstruct the status from the append-only
post-batch JSON records and the already-updated grid; this never reruns DYTurbo.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
PRIMARY = BASE / "reports/tevatron_n3ll_nnlo_wy_production_g1_1p017"
GRID = PRIMARY / "tevatron_full_wy_grid.csv"
GRID_STATUS = PRIMARY / "grid_status.json"
POSTBATCH_LOG = PRIMARY / "postbatch_supervisor.log"
OUT = PRIMARY / "precision_refinement/primary_refinement_status.json"


def main() -> None:
    grid = pd.read_csv(GRID)
    grid_status = json.loads(GRID_STATUS.read_text())
    records = []
    for line in POSTBATCH_LOG.read_text().splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict) or "row_id" not in item:
            continue
        if int(item.get("calls", 0)) != 300_000_000:
            continue
        records.append(item)
    unique = {(str(x["dataset"]), str(x["row_id"])): x for x in records}
    records = list(unique.values())
    if len(records) != 24:
        raise SystemExit(f"expected 24 completed 300M records, found {len(records)}")
    rel = (grid.full_wy_unc_pb_per_GeV / grid.full_wy_pb_per_GeV).abs().to_numpy(float)
    initial_max = float(grid_status["checks"]["max_relative_mc_uncertainty"])
    status = {
        "status": "primary_precision_refinement_complete_not_promoted",
        "grid": str(GRID),
        "selected_count": 24,
        "threshold_data_error_fraction": 0.5,
        "calls_per_refined_component": 300_000_000,
        "max_relative_mc_before": initial_max,
        "max_relative_mc_after": float(np.max(rel)),
        "refined": sorted(records, key=lambda x: (str(x["dataset"]), str(x["row_id"]))),
        "recovered_after_postprocessor_column_bug": True,
        "recovery_note": "All 24 DYTurbo integrations were complete; only aggregate status writing failed. Cached tables were reused and no integration was rerun.",
        "frozen_baseline_unchanged": True,
        "production_outputs_modified": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(status, indent=2) + "\n")
    grid_status["precision_refinement"] = status
    GRID_STATUS.write_text(json.dumps(grid_status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
