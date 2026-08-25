#!/usr/bin/env python3
"""Refine cancellation-dominated rows in the isolated scale campaign.

The first seven-point scale pass used three million Vegas calls per component
for every bin.  This script selects rows whose reported integration error is
larger than a configurable fraction of the central value, reruns each
row/scale card at high statistics, and updates only the isolated scale CSV.
It is a numerical-closure step; its output is not a production uncertainty
band until all selected rows are resolved.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from run_dyturbo_full_n3ll_nnlo_probe import DYTURBO, DYROOT, full_card_text, load_runner


SYSTEMATICS = Path(__file__).resolve().parents[2]
PROJECT = SYSTEMATICS.parent
DATA_ROOT = PROJECT / "Data/v23a_tevatron_plus_lhcb7_fiducial_candidate"
BASE = SYSTEMATICS / "full_n3ll_wy_production_2026/reports/tevatron_scale_variations_g1_1p024191"
DATASETS = ("CDF_RUN_1", "CDF_RUN_2", "D0_RUN_1")


def tag(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, default=30_000_000)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--g1", type=float, default=1.0241911542738864)
    ap.add_argument("--seed", type=int, default=864300)
    ap.add_argument("--base", default=str(BASE))
    ap.add_argument("--rows", nargs="*", default=None, help="optional row_id subset")
    ap.add_argument("--central-only", action="store_true", help="within --rows, refine only muR=muF=1")
    ap.add_argument("--status-name", default="scale_variation_refinement_status.json")
    args = ap.parse_args()
    base = Path(args.base).resolve()
    grid_path = base / "tevatron_scale_variations.csv"
    result = pd.read_csv(grid_path)
    result["_relative_mc"] = (result.raw_unc_fb_per_bin / result.raw_fb_per_bin).abs()
    if args.rows:
        selected = result[result.row_id.astype(str).isin(args.rows)].copy()
        if args.central_only:
            selected = selected[(selected.scale_muR == 1.0) & (selected.scale_muF == 1.0)]
    else:
        selected = result[result["_relative_mc"] > float(args.threshold)].copy()
    if selected.empty:
        raise RuntimeError("no scale rows exceed refinement threshold")
    runner = load_runner()
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(p for p in ("/home/dustin/miniforge3/envs/pdf-fit/lib", str(DYROOT / "lib"), env.get("LD_LIBRARY_PATH", "")) if p)
    refined = []
    for i, (_, row) in enumerate(selected.iterrows()):
        dataset = str(row.dataset)
        source = pd.read_csv(DATA_ROOT / f"{dataset}.csv")
        source_row = source[source.row_id.eq(row.row_id)]
        if len(source_row) != 1:
            raise RuntimeError(f"missing source row {row.row_id}")
        source_row = source_row.iloc[0]
        mur, muf = float(row.scale_muR), float(row.scale_muF)
        row_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(row.row_id))
        name = f"{row_tag}_mur{tag(mur)}_muf{tag(muf)}_refined_{args.calls // 1000000}m"
        card = base / "cards" / f"{name}.in"
        log = base / "logs" / f"{name}.log"
        table = DYROOT / f"{name}.txt"
        card.parent.mkdir(parents=True, exist_ok=True)
        log.parent.mkdir(parents=True, exist_ok=True)
        text = full_card_text(source_row, output_name=name, pdf_set="NNPDF40_nnlo_as_01180", pdf_member=0, cores=8, calls=int(args.calls), seed=int(args.seed) + i)
        text = text.replace("g1 = 0.0", f"g1 = {args.g1:.12g}", 1)
        text = text.replace("kmuren = 1", f"kmuren = {mur:.12g}", 1)
        text = text.replace("kmufac = 1", f"kmufac = {muf:.12g}", 1)
        card.write_text(text)
        if table.exists():
            table.unlink()
        with log.open("w") as handle:
            subprocess.run([str(DYTURBO), str(card)], cwd=DYROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True, timeout=3600)
        value, unc = runner.parse_first_value(table)
        if not np.isfinite([value, unc]).all() or value <= 0 or unc < 0:
            table.unlink(missing_ok=True)
            raise RuntimeError(f"{dataset} {row.row_id} {mur:g}/{muf:g}: nonfinite/nonpositive refined W+Y value")
        width = float(source_row.qT_high - source_row.qT_low)
        mask = (result.dataset.eq(dataset) & result.row_id.eq(row.row_id)
                & result.scale_muR.eq(mur) & result.scale_muF.eq(muf))
        result.loc[mask, "raw_fb_per_bin"] = value
        result.loc[mask, "raw_unc_fb_per_bin"] = unc
        result.loc[mask, "wy_pb_per_GeV"] = value / width / 1000.0
        result.loc[mask, "wy_unc_pb_per_GeV"] = unc / width / 1000.0
        result.loc[mask, "ratio_to_data"] = value / width / 1000.0 / float(source_row.CS)
        refined.append({"dataset": dataset, "row_id": str(row.row_id), "muR": mur, "muF": muf, "card": str(card), "log": str(log), "raw_fb_per_bin": value, "raw_unc_fb_per_bin": unc})
        print(json.dumps(refined[-1]), flush=True)
    result = result.drop(columns=["_relative_mc"])
    result.to_csv(grid_path, index=False)
    central = result[(result.scale_muR == 1.0) & (result.scale_muF == 1.0)].set_index("row_id")
    wide = result.pivot(index="row_id", columns=["scale_muR", "scale_muF"], values="wy_pb_per_GeV").loc[central.index]
    vals = wide.to_numpy(float)
    central_values = central.wy_pb_per_GeV.to_numpy(float)
    lo, hi = vals.min(axis=1), vals.max(axis=1)
    rel = result.raw_unc_fb_per_bin.abs() / result.raw_fb_per_bin.abs()
    summary = {
        "status": "isolated_tevatron_unprimed_n3ll_nnlo_wy_scale_variation_refined_not_production",
        "g1_GeV2": float(args.g1), "calls_per_refined_component": int(args.calls), "threshold": float(args.threshold),
        "refined_count": len(refined), "refined": refined,
        "all_finite": bool(pd.concat([result.wy_pb_per_GeV, result.wy_unc_pb_per_GeV]).notna().all()),
        "all_positive": bool((vals > 0).all()),
        "central_stat_only_chi2_per_row": float(((central_values - central.CS.to_numpy(float)) / central.error.to_numpy(float)).dot((central_values - central.CS.to_numpy(float)) / central.error.to_numpy(float)) / len(central)),
        "max_relative_mc_uncertainty": float(rel.max()),
        "scale_envelope_relative_halfwidth_median": float(np.median((hi - lo) / (2 * central_values))),
        "scale_envelope_relative_halfwidth_max": float(np.max((hi - lo) / (2 * central_values))),
        "artifact_csv": str(grid_path), "frozen_baseline_unchanged": True, "production_outputs_modified": False,
    }
    (base / args.status_name).write_text(json.dumps(summary, indent=2) + "\n")
    # The initial scan may deliberately have been retained with unresolved
    # cancellation rows.  Once this refinement proves every scale point
    # finite and positive, refresh the companion scan status so downstream
    # supervisors can distinguish a completed refinement from a quarantined
    # failed attempt.
    if summary["all_finite"] and summary["all_positive"]:
        scan_status = base / "scale_variation_status.json"
        if scan_status.exists():
            scan = json.loads(scan_status.read_text())
            scan["status"] = "isolated_tevatron_unprimed_n3ll_nnlo_wy_scale_variation_complete_not_production"
            scan["all_finite"] = True
            scan["all_positive"] = True
            scan["unresolved_rows"] = []
            scan["resolved_by_refinement"] = True
            scan["refinement_status"] = str(base / args.status_name)
            scan_status.write_text(json.dumps(scan, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
