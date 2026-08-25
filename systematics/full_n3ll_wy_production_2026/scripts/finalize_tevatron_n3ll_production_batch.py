#!/usr/bin/env python3
"""Validate and record the isolated high-statistics Tevatron W+Y batch.

This script is intentionally fail-closed: it can mark the batch numerically
complete, but it never promotes or overwrites the frozen production package.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "reports/tevatron_n3ll_nnlo_wy_production_g1_1p017"
MANIFEST = BASE / "manifests/tevatron_n3ll_nnlo_wy_production_g1_1p017.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    grid_path = OUT / "tevatron_full_wy_grid.csv"
    status_path = OUT / "grid_status.json"
    if not grid_path.exists() or not status_path.exists():
        raise SystemExit("production batch is not complete: grid/status are missing")
    grid = pd.read_csv(grid_path)
    status = json.loads(status_path.read_text())
    refinement_path = OUT / "precision_refinement/primary_refinement_status.json"
    refinement = json.loads(refinement_path.read_text()) if refinement_path.exists() else None
    expected = 122
    finite = bool(grid[["full_wy_pb_per_GeV", "full_wy_unc_pb_per_GeV"]].notna().all().all())
    positive = bool((grid["full_wy_pb_per_GeV"] > 0).all())
    result = {
        "status": "isolated_tevatron_n3ll_nnlo_wy_batch_complete_not_promoted",
        "candidate_id": "tevatron_n3ll_nnlo_wy_g1_1p017_highstat",
        "formula": "W_N3LL + (FO_NNLO - ASY_NNLO)",
        "engine": "/home/dustin/src/dyturbo-1.4.2/bin/dyturbo",
        "runtime_card": {
            "fixedorder_only": False,
            "order": 3,
            "primed": False,
            "doBORN": True,
            "doCT": True,
            "doVJREAL": True,
            "doVJVIRT": True,
            "VJquad": False,
            "matching_identity": "RES + CT + VJ = W + (FO_NNLO - ASY_NNLO)",
        },
        "scope": {"datasets": ["CDF_RUN_1", "CDF_RUN_2", "D0_RUN_1"], "row_count": int(len(grid)), "expected_row_count": expected},
        "checks": {
            "row_count": bool(len(grid) == expected),
            "all_finite": finite,
            "all_positive": positive,
            "mean_relative_mc_uncertainty": float((grid.raw_full_wy_unc_fb_per_bin / grid.raw_full_wy_fb_per_bin).mean()),
            "max_relative_mc_uncertainty": float((grid.raw_full_wy_unc_fb_per_bin / grid.raw_full_wy_fb_per_bin).max()),
            "median_prediction_to_data": float(grid.full_wy_to_data_ratio.median()),
            "min_prediction_to_data": float(grid.full_wy_to_data_ratio.min()),
            "max_prediction_to_data": float(grid.full_wy_to_data_ratio.max()),
        },
        "runner_status": status,
        "precision_refinement": refinement,
        "artifacts": {
            "grid": {"path": str(grid_path), "sha256": sha256(grid_path)},
            "status": {"path": str(status_path), "sha256": sha256(status_path)},
        },
        "promotion_gates_remaining": [
            "independent-seed stationarity",
            "scale variation with an integration precision floor",
            "correlated experimental plus replica/PDF propagation",
            "F_NP model-form and start propagation",
            "same-scheme fixed-target closure before any 353-row claim",
        ],
        "frozen_baseline_unchanged": True,
        "production_outputs_modified": False,
        "promotion_authorized": False,
    }
    out = OUT / "final_production_status.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    manifest = json.loads(MANIFEST.read_text())
    manifest["status"] = result["status"]
    manifest["completed_batch"] = result["artifacts"]
    manifest["completed_checks"] = result["checks"]
    manifest["promotion_authorized"] = False
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
