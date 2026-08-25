#!/usr/bin/env python3
"""Merge independently completed dataset tables into one isolated Tevatron grid."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd

from run_tevatron_full_n3ll_nnlo_grid import DATA_ROOT, DYROOT, parse_grid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs=3, metavar=("DATASET", "TABLE_DIR", "TABLE_DIR2"), action="append")
    ap.add_argument("--out", required=True)
    ap.add_argument("--g1", type=float, default=1.017)
    ap.add_argument("--calls", type=int, default=100000000)
    args = ap.parse_args()
    # A dataset is resolved from the first directory containing its unique
    # DYTurbo text table; the second directory is an optional fallback.
    records = []
    for dataset, d1, d2 in args.parts:
        data = pd.read_csv(DATA_ROOT / f"{dataset}.csv").sort_values("qT_low").copy()
        edges = [float(data.iloc[0].qT_low)] + [float(x) for x in data.qT_high]
        tag = f"{args.g1:g}".replace(".", "p").replace("-", "m")
        table_name = f"{dataset}_full_n3ll_nnlo_grid_g1_{tag}_seed_20260818.txt"
        table = None
        for base in (Path(d1), Path(d2)):
            candidate = base / table_name
            if candidate.exists():
                table = candidate
                break
        if table is None:
            candidate = DYROOT / table_name
            if candidate.exists():
                table = candidate
        if table is None:
            raise SystemExit(f"missing table for {dataset}: {table_name}")
        grid = parse_grid(table, first_edge=edges[0], last_edge=edges[-1])
        if len(grid) != len(data):
            raise SystemExit(f"{dataset}: parsed {len(grid)} rows, expected {len(data)}")
        merged = data[["dataset", "row_id", "qT_low", "qT_high", "CS", "error"]].reset_index(drop=True).join(grid.reset_index(drop=True), rsuffix="_grid")
        width = merged.qT_high - merged.qT_low
        merged["data_pb_per_GeV"] = merged.CS
        merged["data_unc_pb_per_GeV"] = merged.error
        merged["full_wy_pb_per_GeV"] = merged.raw_full_wy_fb_per_bin / width / 1000.0
        merged["full_wy_unc_pb_per_GeV"] = merged.raw_full_wy_unc_fb_per_bin / width / 1000.0
        merged["full_wy_to_data_ratio"] = merged.full_wy_pb_per_GeV / merged.data_pb_per_GeV
        records.append(merged.drop(columns=["qT_low_grid", "qT_high_grid"]))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result = pd.concat(records, ignore_index=True)
    result.to_csv(out / "tevatron_full_wy_grid.csv", index=False)
    rel = result.raw_full_wy_unc_fb_per_bin / result.raw_full_wy_fb_per_bin
    status = {
        "status": "isolated_tevatron_full_n3ll_nnlo_wy_grid_merged_not_promoted",
        "engine": "/home/dustin/src/dyturbo-1.4.2/bin/dyturbo",
        "order": 3,
        "primed": False,
        "enabled_terms": ["resummation_W_N3LL", "counterterm_ASY_NNLO", "VJ_NNLO"],
        "datasets": [x[0] for x in args.parts],
        "row_count": int(len(result)),
        "g1_GeV2": float(args.g1),
        "checks": {"all_finite": bool(result[["full_wy_pb_per_GeV", "full_wy_unc_pb_per_GeV"]].notna().all().all()), "all_positive": bool((result.full_wy_pb_per_GeV > 0).all()), "mean_relative_mc_uncertainty": float(rel.mean()), "max_relative_mc_uncertainty": float(rel.max()), "full_wy_to_data_ratio_median": float(result.full_wy_to_data_ratio.median()), "full_wy_to_data_ratio_min": float(result.full_wy_to_data_ratio.min()), "full_wy_to_data_ratio_max": float(result.full_wy_to_data_ratio.max())},
        "meaning": "merged external full W+Y perturbative grid; no production authorization",
        "frozen_baseline_unchanged": True,
        "production_outputs_modified": False,
        "promotion_authorized": False,
    }
    (out / "grid_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
