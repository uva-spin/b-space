#!/usr/bin/env python3
"""Refine the three high-MC-uncertainty full W+Y boundary bins."""

from __future__ import annotations

import json
import os
import re
import subprocess
import argparse
from pathlib import Path

import pandas as pd

from run_dyturbo_full_n3ll_nnlo_probe import DYTURBO, DYROOT, full_card_text, load_runner


SYSTEMATICS = Path(__file__).resolve().parents[2]
PROJECT = SYSTEMATICS.parent
INPUT = SYSTEMATICS / "full_n3ll_wy_production_2026/reports/dyturbo_full_n3ll_nnlo_boundary/tevatron_boundary_input.csv"
OUT = SYSTEMATICS / "full_n3ll_wy_production_2026/reports/dyturbo_full_n3ll_nnlo_boundary"
ROWS = ("CDF_RUN_2:44", "CDF_RUN_2:46", "CDF_RUN_2:47")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(INPUT / "tevatron_boundary_input.csv"))
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--g1", type=float, default=0.0)
    parser.add_argument("--calls", type=int, default=15_000_000)
    parser.add_argument("--rows", nargs="+", default=list(ROWS))
    args = parser.parse_args()
    runner = load_runner()
    out = Path(args.out).resolve()
    rows = tuple(args.rows)
    data = pd.read_csv(args.input)
    selected = data[data["row_id"].isin(rows)].copy()
    if len(selected) != len(rows):
        raise RuntimeError("refinement input rows are missing")
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(p for p in ("/home/dustin/miniforge3/envs/pdf-fit/lib", str(DYROOT / "lib"), env.get("LD_LIBRARY_PATH", "")) if p)
    calls = int(args.calls)
    refined = {}
    for _, row in selected.iterrows():
        tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(row.row_id))
        g1_tag = f"{args.g1:g}".replace(".", "p").replace("-", "m")
        name = f"{tag}_full_n3ll_nnlo_refined_g1_{g1_tag}"
        card = out / "cards" / f"{name}.in"
        log = out / "logs" / f"{name}.log"
        table = DYROOT / f"{name}.txt"
        card_text = full_card_text(row, output_name=name, pdf_set="NNPDF40_nnlo_as_01180", pdf_member=0, cores=8, calls=calls)
        card_text = card_text.replace("g1 = 0.0", f"g1 = {args.g1:.12g}", 1)
        card.write_text(card_text)
        if table.exists():
            table.unlink()
        with log.open("w") as handle:
            subprocess.run([str(DYTURBO), str(card)], cwd=DYROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True, timeout=1800)
        value, uncertainty = runner.parse_first_value(table)
        width = float(row.qT_high - row.qT_low)
        refined[str(row.row_id)] = {
            "raw_full_wy_fb_per_bin": value,
            "raw_full_wy_unc_fb_per_bin": uncertainty,
            "full_wy_pb_per_GeV": value / width / 1000.0,
            "full_wy_unc_pb_per_GeV": uncertainty / width / 1000.0,
            "card": str(card),
            "log": str(log),
        }
    summary_path = out / "full_wy_boundary_summary.csv"
    summary = pd.read_csv(summary_path)
    for row_id, values in refined.items():
        mask = summary["row_id"].eq(row_id)
        for key, value in values.items():
            summary.loc[mask, key] = value
        summary.loc[mask, "full_wy_to_data_ratio"] = summary.loc[mask, "full_wy_pb_per_GeV"] / summary.loc[mask, "data_pb_per_GeV"]
    summary.to_csv(summary_path, index=False)
    rel = summary["raw_full_wy_unc_fb_per_bin"] / summary["raw_full_wy_fb_per_bin"]
    status = json.loads((out / "boundary_full_wy_status.json").read_text())
    status["checks"]["mean_relative_mc_uncertainty"] = float(rel.mean())
    status["checks"]["max_relative_mc_uncertainty"] = float(rel.max())
    status["checks"]["full_wy_to_data_ratio_median"] = float(summary["full_wy_to_data_ratio"].median())
    status["checks"]["full_wy_to_data_ratio_min"] = float(summary["full_wy_to_data_ratio"].min())
    status["checks"]["full_wy_to_data_ratio_max"] = float(summary["full_wy_to_data_ratio"].max())
    status["refined_rows"] = list(rows)
    status["g1_GeV2"] = float(args.g1)
    status["refinement_calls_per_vegas_component"] = calls
    (out / "boundary_full_wy_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps({"refined_rows": rows, "max_relative_mc_uncertainty": float(rel.max()), "status": status["status"]}, indent=2))


if __name__ == "__main__":
    main()
