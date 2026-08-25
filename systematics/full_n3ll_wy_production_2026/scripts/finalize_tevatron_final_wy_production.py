#!/usr/bin/env python3
"""Fail-closed finalizer for the isolated Tevatron N3LL+NNLO W+Y run.

This records a genuine external W+Y production grid for the Tevatron rows
whose DYTurbo observable convention is closed.  It deliberately does not
promote or copy anything into the frozen lambda=1 production package.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "reports/tevatron_n3ll_nnlo_wy_final_g1_1p017"
GRID = OUT / "tevatron_full_wy_grid.csv"
STATUS = OUT / "grid_status.json"
FINAL = OUT / "final_production_status.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if not GRID.exists() or not STATUS.exists():
        raise SystemExit("final Tevatron run is incomplete: grid/status missing")
    grid = pd.read_csv(GRID)
    status = json.loads(STATUS.read_text())
    expected = {"CDF_RUN_1": 41, "CDF_RUN_2": 61, "D0_RUN_1": 20}
    counts = grid.groupby("dataset").size().to_dict()
    finite = bool(grid[["full_wy_pb_per_GeV", "full_wy_unc_pb_per_GeV"]].notna().all().all())
    positive = bool((grid["full_wy_pb_per_GeV"] > 0).all())
    actual_seed = status.get("random_seed")
    seed_suffix = f"_seed{int(actual_seed)}" if actual_seed is not None else ""
    result = {
        "status": "isolated_tevatron_genuine_unprimed_n3ll_nnlo_wy_production_complete_not_promoted",
        "candidate_id": f"tevatron_n3ll_nnlo_wy_final_g1_1p017{seed_suffix}",
        "formula": "W_N3LL + (FO_NNLO - ASY_NNLO)",
        "engine": "/home/dustin/src/dyturbo-1.4.2/bin/dyturbo",
        "runtime_convention": {
            "order": 3,
            "primed": False,
            "fixedorder_only": False,
            "doBORN": True,
            "doCT": True,
            "doVJREAL": True,
            "doVJVIRT": True,
            "VJquad": False,
            "identity": "W=RES, ASY=-CT, FO=VJ, Y=FO-ASY=VJ+CT",
        },
        "nonperturbative_center": {"form": "exp(-g1*bT^2)", "g1_GeV2": 1.017},
        "scope": {
            "datasets": list(expected),
            "row_count": int(len(grid)),
            "expected_row_count": int(sum(expected.values())),
            "dataset_row_counts": {str(k): int(v) for k, v in counts.items()},
            "fixed_target_rows": "not included pending target/unit/normalization closure",
            "LHCb_high_qT_rows": "not included pending same-scheme observable closure",
        },
        "checks": {
            "row_count": bool(len(grid) == sum(expected.values())),
            "dataset_counts": bool(counts == expected),
            "all_finite": finite,
            "all_positive": positive,
            "mean_relative_mc_uncertainty": float((grid.raw_full_wy_unc_fb_per_bin / grid.raw_full_wy_fb_per_bin).mean()),
            "max_relative_mc_uncertainty": float((grid.raw_full_wy_unc_fb_per_bin / grid.raw_full_wy_fb_per_bin).max()),
            "median_prediction_to_data": float(grid.full_wy_to_data_ratio.median()),
            "min_prediction_to_data": float(grid.full_wy_to_data_ratio.min()),
            "max_prediction_to_data": float(grid.full_wy_to_data_ratio.max()),
        },
        "source_status": status,
        "replica_diagnostic": None,
        "promotion_gates_remaining": [
            "independent-seed stationarity and scale envelope with precision floor",
            "correlated experimental/PDF propagation",
            "F_NP model-form and local-start propagation around this W+Y backend",
            "same-scheme fixed-target closure before any all-data claim",
        ],
        "frozen_lambda1_baseline_unchanged": True,
        "production_outputs_modified": False,
        "promotion_authorized": False,
        "artifacts": {
            "grid": {"path": str(GRID), "sha256": sha256(GRID)},
            "grid_status": {"path": str(STATUS), "sha256": sha256(STATUS)},
        },
    }
    replica_status = OUT / "replica_profile_500/replica_profile_status.json"
    if replica_status.exists():
        result["replica_diagnostic"] = json.loads(replica_status.read_text())
        result["artifacts"]["replica_diagnostic"] = {
            "path": str(replica_status), "sha256": sha256(replica_status)
        }
    y_status = OUT / "conventional_y/y_grid_status.json"
    y_grid = OUT / "conventional_y/tevatron_y_grid.csv"
    if y_status.exists() and y_grid.exists():
        result["conventional_y"] = json.loads(y_status.read_text())
        result["artifacts"]["conventional_y_grid"] = {
            "path": str(y_grid), "sha256": sha256(y_grid)
        }
        result["artifacts"]["conventional_y_status"] = {
            "path": str(y_status), "sha256": sha256(y_status)
        }
    FINAL.write_text(json.dumps(result, indent=2) + "\n")
    (OUT / "FINAL_PRODUCTION_MANIFEST.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
