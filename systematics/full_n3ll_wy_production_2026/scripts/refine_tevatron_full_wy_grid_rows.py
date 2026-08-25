#!/usr/bin/env python3
"""Refine representative high-variance bins in the external Tevatron grid."""

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
DATA_ROOT = PROJECT / "Data/v23a_tevatron_plus_lhcb7_fiducial_candidate"
GRID_DIR = SYSTEMATICS / "full_n3ll_wy_production_2026/reports/dyturbo_full_n3ll_nnlo_tevatron_grid"
ROWS = ("CDF_RUN_2:2", "CDF_RUN_1:1", "D0_RUN_1:0")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", nargs="+", default=list(ROWS))
    parser.add_argument("--calls", type=int, default=30_000_000)
    parser.add_argument("--grid-dir", default=str(GRID_DIR))
    parser.add_argument("--g1", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=246810)
    args = parser.parse_args()
    rows_to_refine = tuple(args.rows)
    grid_dir = Path(args.grid_dir).resolve()
    runner = load_runner()
    source = pd.concat([pd.read_csv(DATA_ROOT / f"{ds}.csv") for ds in ("CDF_RUN_1", "CDF_RUN_2", "D0_RUN_1")], ignore_index=True)
    selected = source[source.row_id.isin(rows_to_refine)].copy()
    if len(selected) != len(rows_to_refine):
        raise RuntimeError("refinement rows missing")
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(p for p in ("/home/dustin/miniforge3/envs/pdf-fit/lib", str(DYROOT / "lib"), env.get("LD_LIBRARY_PATH", "")) if p)
    calls = int(args.calls)
    refined = {}
    for _, row in selected.iterrows():
        tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(row.row_id))
        seed_tag = str(int(args.seed))
        name = f"{tag}_full_n3ll_nnlo_refined_{calls // 1000000}m_seed_{seed_tag}"
        card = grid_dir / "cards" / f"{name}_g1_{str(args.g1).replace('.', 'p')}.in"
        log = grid_dir / "logs" / f"{name}_g1_{str(args.g1).replace('.', 'p')}.log"
        table = DYROOT / f"{name}.txt"
        card_text = full_card_text(row, output_name=name, pdf_set="NNPDF40_nnlo_as_01180", pdf_member=0, cores=8, calls=calls, seed=int(args.seed))
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
            "full_wy_to_data_ratio": value / width / 1000.0 / float(row.CS),
            "card": str(card),
            "log": str(log),
        }
    path = grid_dir / "tevatron_full_wy_grid.csv"
    result = pd.read_csv(path)
    for row_id, values in refined.items():
        mask = result.row_id.eq(row_id)
        for key, value in values.items():
            result.loc[mask, key] = value
    result.to_csv(path, index=False)
    rel = result.raw_full_wy_unc_fb_per_bin / result.raw_full_wy_fb_per_bin
    status = json.loads((grid_dir / "grid_status.json").read_text())
    status["checks"]["mean_relative_mc_uncertainty"] = float(rel.mean())
    status["checks"]["max_relative_mc_uncertainty"] = float(rel.max())
    status["checks"]["all_finite"] = bool(result[["full_wy_pb_per_GeV", "full_wy_unc_pb_per_GeV"]].notna().all().all())
    status["checks"]["all_positive"] = bool((result.full_wy_pb_per_GeV > 0).all())
    status["checks"]["full_wy_to_data_ratio_median"] = float(result.full_wy_to_data_ratio.median())
    status["checks"]["full_wy_to_data_ratio_min"] = float(result.full_wy_to_data_ratio.min())
    status["checks"]["full_wy_to_data_ratio_max"] = float(result.full_wy_to_data_ratio.max())
    status["refined_rows"] = list(rows_to_refine)
    status["refinement_calls_per_vegas_component"] = calls
    (grid_dir / "grid_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps({"status": status["status"], "refined_rows": rows_to_refine, "max_relative_mc_uncertainty": float(rel.max())}, indent=2))


if __name__ == "__main__":
    main()
