#!/usr/bin/env python3
"""Hash the completed isolated Tevatron candidate without promoting it."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PRIMARY = REPORTS / "tevatron_n3ll_nnlo_wy_production_g1_1p017"
STATION = REPORTS / "tevatron_n3ll_nnlo_wy_stationarity_g1_1p017_seed_20260820"
OUT = ROOT / "manifests/tevatron_n3ll_nnlo_wy_candidate_freeze.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    required = {
        "primary_grid": PRIMARY / "tevatron_full_wy_grid.csv",
        "primary_status": PRIMARY / "grid_status.json",
        "primary_finalizer": PRIMARY / "final_production_status.json",
        "stationarity_grid": STATION / "tevatron_full_wy_grid.csv",
        "stationarity_status": STATION / "grid_status.json",
        "stationarity_compare": STATION / "stationarity_status.json",
        "accuracy_closure": REPORTS / "accuracy_closure_g1_1p017.json",
        "term_decomposition": REPORTS / "dyturbo_term_decomposition_g1_1p017/term_decomposition_status.json",
        "boundary_oracle": REPORTS / "dyturbo_full_n3ll_nnlo_boundary_g1_1p017/boundary_full_wy_status.json",
        "campaign_summary": REPORTS / "tevatron_n3ll_nnlo_wy_campaign_summary.json",
        "primary_precision_refinement": PRIMARY / "precision_refinement/primary_refinement_status.json",
        "grid_figure_png": PRIMARY / "figures/tevatron_n3ll_nnlo_wy_grid_comparison.png",
        "grid_figure_pdf": PRIMARY / "figures/tevatron_n3ll_nnlo_wy_grid_comparison.pdf",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise SystemExit("candidate freeze incomplete; missing: " + ", ".join(missing))
    status = json.loads((PRIMARY / "grid_status.json").read_text())
    station_status = json.loads((STATION / "grid_status.json").read_text())
    if int(status.get("row_count", -1)) != 122 or int(station_status.get("row_count", -1)) != 122:
        raise SystemExit("candidate freeze requires two complete 122-row grids")
    term = json.loads(required["term_decomposition"].read_text())
    if abs(float(term.get("g1_GeV2", float("nan"))) - 1.017) > 1.0e-12:
        raise SystemExit("candidate freeze requires term decomposition at g1=1.017")
    files = {name: {"path": str(path), "sha256": digest(path)} for name, path in required.items()}
    optional = REPORTS / "tevatron_n3ll_nnlo_wy_replica_profile_500_g1_1p017/replica_profile_status.json"
    if optional.exists():
        files["replica_diagnostic"] = {"path": str(optional), "sha256": digest(optional)}
    result = {
        "status": "isolated_tevatron_n3ll_nnlo_wy_candidate_frozen_not_promoted",
        "candidate_id": "tevatron_n3ll_nnlo_wy_g1_1p017_highstat",
        "formula": "W_N3LL + (FO_NNLO - ASY_NNLO)",
        "scope": {"tevatron_rows": 122, "boundary_oracle_rows": 24},
        "files": files,
        "frozen_lambda1_baseline_unchanged": True,
        "production_outputs_modified": False,
        "promotion_authorized": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
