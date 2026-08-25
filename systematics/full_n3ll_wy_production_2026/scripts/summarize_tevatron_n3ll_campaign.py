#!/usr/bin/env python3
"""Build a fail-closed summary of the isolated Tevatron N3LL+NNLO campaign."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PRIMARY = REPORTS / "tevatron_n3ll_nnlo_wy_production_g1_1p017"
STATION = REPORTS / "tevatron_n3ll_nnlo_wy_stationarity_g1_1p017_seed_20260820"
OUT = REPORTS / "tevatron_n3ll_nnlo_wy_campaign_summary.json"


def read_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def main() -> None:
    grid_path = PRIMARY / "tevatron_full_wy_grid.csv"
    station_path = STATION / "tevatron_full_wy_grid.csv"
    grid = pd.read_csv(grid_path) if grid_path.exists() else None
    station = pd.read_csv(station_path) if station_path.exists() else None
    final = read_json(PRIMARY / "final_production_status.json")
    stationarity = read_json(STATION / "stationarity_status.json")
    replica = read_json(REPORTS / "tevatron_n3ll_nnlo_wy_replica_profile_500_g1_1p017" / "replica_profile_status.json")
    refinement = read_json(PRIMARY / "precision_refinement/primary_refinement_status.json")
    scale = read_json(REPORTS / "tevatron_scale_variations_g1_1p017" / "scale_variation_status.json")
    accuracy = read_json(REPORTS / "accuracy_closure_g1_1p017.json")
    boundary = read_json(REPORTS / "dyturbo_full_n3ll_nnlo_boundary_g1_1p017" / "boundary_full_wy_status.json")

    complete_grid = bool(grid is not None and len(grid) == 122 and grid[["full_wy_pb_per_GeV", "full_wy_unc_pb_per_GeV"]].notna().all().all() and (grid.full_wy_pb_per_GeV > 0).all())
    complete_station = bool(station is not None and len(station) == 122 and station[["full_wy_pb_per_GeV", "full_wy_unc_pb_per_GeV"]].notna().all().all() and (station.full_wy_pb_per_GeV > 0).all())
    result: dict = {
        "status": "isolated_tevatron_full_n3ll_nnlo_wy_campaign_summary_not_promoted",
        "formula": "W_N3LL + (FO_NNLO - ASY_NNLO)",
        "convention": {"resummation": "unprimed N3LL", "fixed_order": "NNLO V+jet", "primed": False, "term_identity": "W=RES; ASY=-CT; FO=VJ; Y=VJ+CT"},
        "scope": {"published_tevatron_qT_rows": 122, "boundary_oracle_rows": 24, "fixed_target_329_row_core": "pending same-scheme external closure", "LHCb_high_qT_rows": "outside scope"},
        "primary": {"row_count": int(len(grid)) if grid is not None else 0, "complete_122_rows": complete_grid},
        "stationarity": {"row_count": int(len(station)) if station is not None else 0, "complete_122_rows": complete_station, "status": stationarity},
        "finalizer": final,
        "precision_refinement": refinement,
        "replica_diagnostic": replica,
        "scale_diagnostic": scale,
        "accuracy_closure": accuracy,
        "boundary_oracle": boundary,
        "uncertainty_layers": {
            "integration_mc": complete_grid and complete_station,
            "experimental_point_to_point_and_norm": bool(replica is not None),
            "pdf_replicas": False,
            "F_NP_model_form_and_start": False,
            "scale_envelope": bool(scale is not None),
            "formal_1sigma_assigned": False,
        },
        "promotion_gates_remaining": [
            "same-scheme fixed-target closure and exact 329-row selection",
            "PDF-replica propagation",
            "F_NP/start/model-form propagation into TMD observables",
            "scale envelope with a precision floor on every retained variation",
            "final figures only after all retained uncertainty layers are defined",
        ],
        "frozen_lambda1_baseline_unchanged": True,
        "production_outputs_modified": False,
        "promotion_authorized": False,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "primary_complete": complete_grid, "stationarity_complete": complete_station}, indent=2))


if __name__ == "__main__":
    main()
