#!/usr/bin/env python3
"""Write the candidate-center conventional-Y accuracy-closure record.

The older accuracy_closure_v2.json records a historical g1=1.0 probe.  The
retained candidate is g1=1.017, so its freeze must carry a closure record tied
to the same center and the complete 122-row grid.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PRIMARY = REPORTS / "tevatron_n3ll_nnlo_wy_production_g1_1p017"
BOUNDARY = REPORTS / "dyturbo_full_n3ll_nnlo_boundary_g1_1p017/boundary_full_wy_status.json"
TERMS = REPORTS / "dyturbo_term_decomposition_g1_1p017/term_decomposition_status.json"
OUT = REPORTS / "accuracy_closure_g1_1p017.json"


def main() -> None:
    grid = pd.read_csv(PRIMARY / "tevatron_full_wy_grid.csv")
    boundary = json.loads(BOUNDARY.read_text())
    terms = json.loads(TERMS.read_text())
    if len(grid) != 122:
        raise SystemExit("candidate accuracy closure requires 122 primary rows")
    if abs(float(terms.get("g1_GeV2", float("nan"))) - 1.017) > 1.0e-12:
        raise SystemExit("term closure is not at candidate g1=1.017")
    if int(boundary.get("row_count", -1)) != 24 or abs(float(boundary.get("g1_GeV2", float("nan"))) - 1.017) > 1.0e-12:
        raise SystemExit("boundary closure is not the 24-row g1=1.017 oracle")
    result = {
        "status": "isolated_external_unprimed_n3ll_nnlo_candidate_accuracy_closure_passed_not_promoted",
        "engine": "/home/dustin/src/dyturbo-1.4.2/bin/dyturbo",
        "g1_GeV2": 1.017,
        "convention": {
            "resummation": "unprimed N3LL, order=3, primed=false",
            "fixed_order": "NNLO V+jet (order_vjet=2)",
            "terms": "W=RES; ASY=-CT; FO=VJ; Y=FO-ASY=VJ+CT",
            "full": "RES+CT+VJ",
        },
        "checks": {
            "tevatron_primary_grid_122_rows": str(PRIMARY / "tevatron_full_wy_grid.csv"),
            "tevatron_boundary_24_rows": str(BOUNDARY),
            "term_level_candidate_center": str(TERMS),
            "primary_all_finite": bool(grid[["full_wy_pb_per_GeV", "full_wy_unc_pb_per_GeV"]].notna().all().all()),
            "primary_all_positive": bool((grid["full_wy_pb_per_GeV"] > 0).all()),
        },
        "scope_limits": [
            "external-engine candidate only",
            "fixed-target 329-row core still requires same-scheme closure",
            "LHCb high-qT rows remain outside this Tevatron candidate",
            "no production promotion or uncertainty replacement authorized",
        ],
        "frozen_baseline_unchanged": True,
        "production_outputs_modified": False,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
