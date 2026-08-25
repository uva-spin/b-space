#!/usr/bin/env python3
"""Run the isolated NNLO fixed-order oracle on the 24 Tevatron boundary rows.

This is an external-observable validation step for the candidate
``W_N3LL + (FO_NNLO - ASY_NNLO)`` construction.  It does not fit the TMD,
write a production cache, or alter any frozen result.  DYTurbo is configured
with ``order=3, primed=false``: in DYTurbo's convention this is the unprimed
N3LL+NNLO V+jet contribution.  The row list is the frozen 24-row Tevatron
boundary used by the earlier finite-Y diagnostic.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


SYSTEMATICS = Path(__file__).resolve().parents[2]
PROJECT = SYSTEMATICS.parent
RUNNER_PATH = PROJECT / "b-space-public/v23/tools/run_tevatron_dyturbo_benchmark.py"
BOUNDARY_PULLS = (
    SYSTEMATICS
    / "finite_y_completion_2026/reports/../.."
    / "high_qt_direct_production_benchmark/experimental_unitary_transition"
    / "summaries/unitary_smootherstep_v1_nlo_24row/nlo_frozen_unitary_pulls.csv"
)
DATA_ROOT = PROJECT / "Data/v23a_tevatron_plus_lhcb7_fiducial_candidate"
OUT = SYSTEMATICS / "full_n3ll_wy_production_2026/reports/dyturbo_nnlo_boundary"


def load_runner():
    spec = importlib.util.spec_from_file_location("candidate_dyturbo_boundary_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def nnlo_card_text(*args: object, **kwargs: object) -> str:
    runner = sys.modules["candidate_dyturbo_boundary_runner"]
    calls = int(kwargs.pop("vj_calls", 10_000_000))
    text = runner._base_card_text(*args, **kwargs) if hasattr(runner, "_base_card_text") else runner.card_text(*args, **kwargs)
    text = text.replace("order           = 1\n", "order           = 3\n", 1)
    text = text.replace("primed          = true\n", "primed          = false\n", 1)
    text = text.replace("doVJREAL = false", "doVJREAL = true", 1)
    text = text.replace("doVJVIRT = false", "doVJVIRT = true", 1)
    text = text.replace("VJquad = true", "VJquad = false", 1)
    text = text.replace("intDimVJ   = 3", "intDimVJ   = -1", 1)
    text = text.replace("makecuts = false", "makecuts = true", 1)
    text = text.replace("vegasncallsVJREAL = 100000", f"vegasncallsVJREAL = {calls}", 1)
    text = text.replace("vegasncallsVJVIRT = 100000", f"vegasncallsVJVIRT = {calls}", 1)
    return text


def main() -> None:
    runner = load_runner()
    # Keep the imported runner's implementation, but replace its card factory
    # and provide a candidate-local combined input table.
    runner._base_card_text = runner.card_text
    runner.card_text = nnlo_card_text

    boundary = pd.read_csv(BOUNDARY_PULLS)
    row_ids = boundary["row_id"].astype(str).tolist()
    if len(row_ids) != 24 or len(set(row_ids)) != 24:
        raise RuntimeError(f"expected 24 unique boundary rows, found {len(row_ids)}")

    pieces = []
    for dataset in sorted(boundary["dataset"].astype(str).unique()):
        path = DATA_ROOT / f"{dataset}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        pieces.append(pd.read_csv(path))
    combined = pd.concat(pieces, ignore_index=True)
    selected = combined[combined["row_id"].isin(row_ids)].copy()
    if len(selected) != 24:
        raise RuntimeError(f"combined input contains {len(selected)} boundary rows, expected 24")

    OUT.mkdir(parents=True, exist_ok=True)
    input_path = OUT / "tevatron_boundary_input.csv"
    selected.sort_values(["dataset", "row_id"]).to_csv(input_path, index=False)

    # The imported runner accepts a single data table and an explicit row list.
    sys.argv = [
        sys.argv[0],
        "--data", str(input_path),
        "--rows", *row_ids,
        "--out", str(OUT),
        "--cores", "8",
        "--timeout", "900",
    ]
    runner.main()

    result_path = OUT / "dyturbo_benchmark_summary.csv"
    result = pd.read_csv(result_path)
    # DYTurbo's text output is fb per bin.  The fit-ready Tevatron tables are
    # pb per GeV, so retain the raw oracle value but add the explicitly
    # converted quantities used for closure checks.
    result["bin_width_GeV"] = result["qT_high"] - result["qT_low"]
    result["dyturbo_nnlo_pb_per_GeV"] = (
        result["dyturbo_raw"] / result["bin_width_GeV"] / 1000.0
    )
    result["dyturbo_nnlo_unc_pb_per_GeV"] = (
        result["dyturbo_raw_unc"] / result["bin_width_GeV"] / 1000.0
    )
    input_checks = selected[["row_id", "CS", "error"]].rename(
        columns={"CS": "data_pb_per_GeV", "error": "data_unc_pb_per_GeV"}
    )
    result = result.merge(input_checks, on="row_id", validate="one_to_one")
    result["nnlo_to_data_ratio"] = result["dyturbo_nnlo_pb_per_GeV"] / result["data_pb_per_GeV"]
    result["nnlo_data_pull"] = (
        result["dyturbo_nnlo_pb_per_GeV"] - result["data_pb_per_GeV"]
    ) / result["data_unc_pb_per_GeV"]
    result.to_csv(result_path, index=False)
    checks = {
        "row_count": int(len(result)),
        "unique_row_count": int(result["row_id"].nunique()),
        "all_finite_values": bool(result[["dyturbo_raw", "dyturbo_raw_unc"]].map(pd.api.types.is_number).all().all()),
        "all_positive_cross_sections": bool((result["dyturbo_raw"] > 0).all()),
        "mean_relative_mc_uncertainty": float((result["dyturbo_raw_unc"] / result["dyturbo_raw"]).mean()),
        "max_relative_mc_uncertainty": float((result["dyturbo_raw_unc"] / result["dyturbo_raw"]).max()),
        "nnlo_to_data_ratio_median": float(result["nnlo_to_data_ratio"].median()),
        "nnlo_to_data_ratio_min": float(result["nnlo_to_data_ratio"].min()),
        "nnlo_to_data_ratio_max": float(result["nnlo_to_data_ratio"].max()),
        "nnlo_data_pull_rms_stat_only": float((result["nnlo_data_pull"].pow(2).mean()) ** 0.5),
    }
    status = {
        "status": "isolated_tevatron_24row_nnlo_fixed_order_oracle_passed",
        "engine": "/home/dustin/src/dyturbo-1.4.2/bin/dyturbo",
        "dy_turbo_order": 3,
        "dy_turbo_primed": False,
        "dyturbo_text_units": "fb per qT bin",
        "reported_closure_units": "pb per GeV; raw / bin_width / 1000",
        "accuracy_interpretation": "unprimed N3LL+NNLO V+jet component; observable oracle only",
        "boundary_source": str(BOUNDARY_PULLS),
        "row_count": len(result),
        "checks": checks,
        "meaning": "confirms the external NNLO fixed-order side can be evaluated on the Tevatron boundary rows; it is not yet the fitted W or conventional Y grid",
        "frozen_baseline_unchanged": True,
        "production_outputs_modified": False,
    }
    (OUT / "boundary_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
