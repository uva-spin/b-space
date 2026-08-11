#!/usr/bin/env python3
"""Analyze b-grid/bin-quadrature convergence for the isolated ASY pilot."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
HERE = ROOT / "systematics/high_qt_direct_production_benchmark/experimental_matched_y"
PILOT = HERE / "outputs/asymptotic_pilot/cdf_run_2_36"


def load(name: str) -> dict:
    return json.loads((PILOT / name).read_text())


def symmetric(a: float, b: float) -> float:
    return abs(a - b) / max(0.5 * (abs(a) + abs(b)), 1.0e-300)


def main() -> None:
    nb160 = load("result.json")
    nb320 = load("result_nb320_nqt2_ny2.json")
    quad3 = load("result_nb320_nqt3_ny3.json")
    nb640 = load("result_nb640_nqt2_ny2.json")
    v160, v320 = float(nb160["value_pb_per_GeV"]), float(nb320["value_pb_per_GeV"])
    v640, vquad3 = float(nb640["value_pb_per_GeV"]), float(quad3["value_pb_per_GeV"])
    richardson = v640 + (v640 - v320) / 3.0
    external = pd.read_csv(
        ROOT / "systematics/high_qt_direct_production_benchmark/summaries/tier1_boundary/central/external_pairs.csv"
    ).set_index("row_id").loc["CDF_RUN_2:36"]
    fo_dy = float(external.dyturbo_pb_per_GeV)
    fo_mc = float(external.mcfm_pb_per_GeV)
    fo_average = 0.5 * (fo_dy + fo_mc)
    table = pd.DataFrame([
        {"study": "b_grid", "n_b": 160, "n_qT": 2, "n_y": 2, "value_pb_per_GeV": v160},
        {"study": "b_grid", "n_b": 320, "n_qT": 2, "n_y": 2, "value_pb_per_GeV": v320},
        {"study": "b_grid", "n_b": 640, "n_qT": 2, "n_y": 2, "value_pb_per_GeV": v640},
        {"study": "bin_quadrature", "n_b": 320, "n_qT": 3, "n_y": 3, "value_pb_per_GeV": vquad3},
    ])
    table.to_csv(PILOT / "convergence.csv", index=False)
    status = {
        "status": "experimental_not_production", "row_id": "CDF_RUN_2:36",
        "b_grid_160_to_320_symmetric_shift": symmetric(v160, v320),
        "b_grid_320_to_640_symmetric_shift": symmetric(v320, v640),
        "bin_quadrature_2x2_to_3x3_shift_at_nb320": symmetric(v320, vquad3),
        "richardson_O_h2_asymptotic_pb_per_GeV": richardson,
        "richardson_vs_nb640_shift": symmetric(richardson, v640),
        "external_fo_dyturbo_pb_per_GeV": fo_dy,
        "external_fo_mcfm_pb_per_GeV": fo_mc,
        "external_fo_average_pb_per_GeV": fo_average,
        "experimental_y_fo_minus_asym_richardson_pb_per_GeV": fo_average - richardson,
        "bin_quadrature_numerical_pass": bool(symmetric(v320, vquad3) <= 0.005),
        "b_grid_numerical_pass": bool(symmetric(v320, v640) <= 0.02),
        "asymptotic_component_numerical_pilot_pass": bool(
            symmetric(v320, vquad3) <= 0.005 and symmetric(v320, v640) <= 0.02
        ),
        "w_asymptotic_cancellation_pass": False,
        "matched_prediction_pass": False,
        "lhcb_fiducial_asymptotic_available": False,
        "note": "Large negative ASY and positive FO-ASY are not interpreted physically until exact-bin resummed-W cancellation is demonstrated.",
    }
    (PILOT / "convergence_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
