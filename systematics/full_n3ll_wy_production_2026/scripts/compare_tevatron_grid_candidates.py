#!/usr/bin/env python3
"""Compare isolated g1=0 and g1=1 external W+Y Tevatron grids."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


SYSTEMATICS = Path(__file__).resolve().parents[2]
BASE = SYSTEMATICS / "full_n3ll_wy_production_2026/reports/dyturbo_full_n3ll_nnlo_tevatron_grid/tevatron_full_wy_grid.csv"
G1 = SYSTEMATICS / "full_n3ll_wy_production_2026/reports/dyturbo_full_n3ll_nnlo_tevatron_grid_g1_1p0/tevatron_full_wy_grid.csv"
OUT = SYSTEMATICS / "full_n3ll_wy_production_2026/reports/tevatron_external_grid_candidate_comparison.json"


def summarize(frame: pd.DataFrame) -> dict:
    rel = frame["raw_full_wy_unc_fb_per_bin"] / frame["raw_full_wy_fb_per_bin"]
    residual = (frame["full_wy_pb_per_GeV"] - frame["data_pb_per_GeV"]) / frame["data_unc_pb_per_GeV"]
    return {
        "row_count": int(len(frame)),
        "median_ratio_to_data": float(frame["full_wy_to_data_ratio"].median()),
        "mean_ratio_to_data": float(frame["full_wy_to_data_ratio"].mean()),
        "mean_relative_mc_uncertainty": float(rel.mean()),
        "max_relative_mc_uncertainty": float(rel.max()),
        "rms_stat_only_data_pull": float((residual.pow(2).mean()) ** 0.5),
        "stat_only_chi2_per_row": float(residual.pow(2).mean()),
    }


def main() -> None:
    base = pd.read_csv(BASE)
    g1 = pd.read_csv(G1)
    merged = base[["row_id", "full_wy_pb_per_GeV"]].rename(columns={"full_wy_pb_per_GeV": "g1_zero_pb_per_GeV"}).merge(
        g1[["row_id", "full_wy_pb_per_GeV", "data_pb_per_GeV"]].rename(columns={"full_wy_pb_per_GeV": "g1_one_pb_per_GeV"}),
        on="row_id", validate="one_to_one",
    )
    merged["g1_one_over_g1_zero"] = merged.g1_one_pb_per_GeV / merged.g1_zero_pb_per_GeV
    result = {
        "status": "isolated_external_n3ll_nnlo_np_candidate_comparison_complete",
        "g1_zero": summarize(base),
        "g1_one": summarize(g1),
        "ratio_change": {
            "qT_zero_median_g1_one_over_zero": float(merged[merged.row_id.str.endswith(":0")].g1_one_over_g1_zero.median()),
            "transition_median_g1_one_over_zero": float(merged[merged.row_id.str.match(r".*:(17|18|22|28)$")].g1_one_over_g1_zero.median()),
            "all_bin_median_g1_one_over_zero": float(merged.g1_one_over_g1_zero.median()),
        },
        "interpretation": "candidate NP-form comparison only; no production promotion or replacement of the frozen DNN baseline",
        "production_outputs_modified": False,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
