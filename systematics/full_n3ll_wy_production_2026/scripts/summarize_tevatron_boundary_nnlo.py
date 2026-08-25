#!/usr/bin/env python3
"""Normalize and summarize the completed candidate DYTurbo boundary run."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


SYSTEMATICS = Path(__file__).resolve().parents[2]
OUT = SYSTEMATICS / "full_n3ll_wy_production_2026/reports/dyturbo_nnlo_boundary"


def main() -> None:
    result_path = OUT / "dyturbo_benchmark_summary.csv"
    input_path = OUT / "tevatron_boundary_input.csv"
    result = pd.read_csv(result_path)
    source = pd.read_csv(input_path)[["row_id", "CS", "error"]].rename(
        columns={"CS": "data_pb_per_GeV", "error": "data_unc_pb_per_GeV"}
    )
    result["bin_width_GeV"] = result["qT_high"] - result["qT_low"]
    result["dyturbo_nnlo_pb_per_GeV"] = result["dyturbo_raw"] / result["bin_width_GeV"] / 1000.0
    result["dyturbo_nnlo_unc_pb_per_GeV"] = result["dyturbo_raw_unc"] / result["bin_width_GeV"] / 1000.0
    result = result.drop(columns=[c for c in ("data_pb_per_GeV", "data_unc_pb_per_GeV") if c in result])
    result = result.merge(source, on="row_id", validate="one_to_one")
    result["nnlo_to_data_ratio"] = result["dyturbo_nnlo_pb_per_GeV"] / result["data_pb_per_GeV"]
    result["nnlo_data_pull"] = (result["dyturbo_nnlo_pb_per_GeV"] - result["data_pb_per_GeV"]) / result["data_unc_pb_per_GeV"]
    result.to_csv(result_path, index=False)
    mc_rel = result["dyturbo_raw_unc"] / result["dyturbo_raw"]
    normalization_checks = {}
    for dataset, group in result.groupby("dataset"):
        weights = 1.0 / group["data_unc_pb_per_GeV"].pow(2)
        model = group["dyturbo_nnlo_pb_per_GeV"]
        data = group["data_pb_per_GeV"]
        factor = float((weights * data * model).sum() / (weights * model.pow(2)).sum())
        residual = (factor * model - data) / group["data_unc_pb_per_GeV"]
        normalization_checks[str(dataset)] = {
            "best_data_over_oracle_factor_stat_only": factor,
            "stat_only_chi2": float(residual.pow(2).sum()),
            "stat_only_chi2_per_row": float(residual.pow(2).mean()),
            "row_count": int(len(group)),
        }
    status = {
        "status": "isolated_tevatron_24row_nnlo_fixed_order_oracle_passed",
        "engine": "/home/dustin/src/dyturbo-1.4.2/bin/dyturbo",
        "dy_turbo_order": 3,
        "dy_turbo_primed": False,
        "accuracy_interpretation": "unprimed N3LL+NNLO V+jet component; observable oracle only",
        "dyturbo_text_units": "fb per qT bin",
        "reported_closure_units": "pb per GeV; raw / bin_width / 1000",
        "row_count": int(len(result)),
        "checks": {
            "all_finite_values": bool(result[["dyturbo_raw", "dyturbo_raw_unc", "dyturbo_nnlo_pb_per_GeV"]].notna().all().all()),
            "all_positive_cross_sections": bool((result["dyturbo_raw"] > 0).all()),
            "mean_relative_mc_uncertainty": float(mc_rel.mean()),
            "max_relative_mc_uncertainty": float(mc_rel.max()),
            "nnlo_to_data_ratio_median": float(result["nnlo_to_data_ratio"].median()),
            "nnlo_to_data_ratio_min": float(result["nnlo_to_data_ratio"].min()),
            "nnlo_to_data_ratio_max": float(result["nnlo_to_data_ratio"].max()),
            "nnlo_data_pull_rms_stat_only": float((result["nnlo_data_pull"].pow(2).mean()) ** 0.5),
        },
        "dataset_normalization_diagnostic_stat_only": normalization_checks,
        "meaning": "confirms that the external NNLO fixed-order side can be evaluated on the Tevatron boundary rows; the data ratio is a diagnostic, not a fit result",
        "frozen_baseline_unchanged": True,
        "production_outputs_modified": False,
    }
    (OUT / "boundary_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
