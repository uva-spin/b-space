#!/usr/bin/env python3
"""Refine cancellation-dominated rows of the isolated Tevatron grid.

The primary 100M-call grid is retained as the starting point.  Only rows whose
reported integration uncertainty exceeds a documented fraction of the data
relative error are reevaluated, and the refined values remain candidate-side.
"""

from __future__ import annotations

import argparse
import concurrent.futures
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
RUNNER = None


def tag(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def one_row(task: tuple[dict, int, int, Path, float]) -> dict:
    row, index, calls, out, g1 = task
    dataset = str(row["dataset"])
    source = pd.read_csv(DATA_ROOT / f"{dataset}.csv")
    source_row = source[source.row_id.eq(row["row_id"])]
    if len(source_row) != 1:
        raise RuntimeError(f"missing source row {row['row_id']}")
    source_row = source_row.iloc[0]
    name = f"{str(row['row_id']).replace(':', '_')}_g1_{tag(g1)}_refined_{calls // 1000000}m"
    card = out / "cards" / f"{name}.in"
    log = out / "logs" / f"{name}.log"
    table = DYROOT / f"{name}.txt"
    card.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    text = full_card_text(source_row, output_name=name, pdf_set="NNPDF40_nnlo_as_01180", pdf_member=0, cores=8, calls=calls, seed=20263000 + index)
    text = text.replace("g1 = 0.0", f"g1 = {g1:.12g}", 1)
    lo, hi = float(source_row.qT_low), float(source_row.qT_high)
    width = hi - lo
    text = re.sub(r"qt_bins = \[ [^\]]+ \]", f"qt_bins = [ {lo:.12g} {hi:.12g} ]", text, count=1)
    card.write_text(text)
    # A prior run may have completed the DYTurbo integration but failed while
    # writing its aggregate status.  Reuse the matching card/table/log rather
    # than spending another 300M calls; the table is keyed by this exact row,
    # center, and call count.
    if table.exists() and log.exists():
        value, unc = RUNNER.parse_first_value(table)
        if not np.isfinite([value, unc]).all() or value <= 0 or unc < 0:
            raise RuntimeError(f"{dataset} {row['row_id']}: cached refined W+Y value is invalid")
        return {"dataset": dataset, "row_id": str(row["row_id"]), "raw_full_wy_fb_per_bin": float(value), "raw_full_wy_unc_fb_per_bin": float(unc), "card": str(card), "log": str(log), "calls": calls, "width": width, "cached": True}
    if table.exists():
        table.unlink()
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(p for p in ("/home/dustin/miniforge3/envs/pdf-fit/lib", str(DYROOT / "lib"), env.get("LD_LIBRARY_PATH", "")) if p)
    with log.open("w") as handle:
        subprocess.run([str(DYTURBO), str(card)], cwd=DYROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True, timeout=14400)
    if RUNNER is None:
        raise RuntimeError("DYTurbo parser was not initialized")
    value, unc = RUNNER.parse_first_value(table)
    if not np.isfinite([value, unc]).all() or value <= 0 or unc < 0:
        table.unlink(missing_ok=True)
        raise RuntimeError(f"{dataset} {row['row_id']}: refined W+Y value is invalid")
    return {"dataset": dataset, "row_id": str(row["row_id"]), "raw_full_wy_fb_per_bin": float(value), "raw_full_wy_unc_fb_per_bin": float(unc), "card": str(card), "log": str(log), "calls": calls, "width": width}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--g1", type=float, default=1.017)
    ap.add_argument("--calls", type=int, default=300_000_000)
    ap.add_argument("--data-error-fraction", type=float, default=0.5, help="refine when MC relative error exceeds this fraction of data relative error")
    ap.add_argument("--max-workers", type=int, default=3)
    args = ap.parse_args()
    global RUNNER
    RUNNER = load_runner()
    grid_path = Path(args.grid).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    grid = pd.read_csv(grid_path)
    grid["_mc_rel"] = (grid.raw_full_wy_unc_fb_per_bin / grid.raw_full_wy_fb_per_bin).abs()
    grid["_data_rel"] = (grid.error / grid.CS).abs()
    selected = grid[grid._mc_rel > float(args.data_error_fraction) * grid._data_rel].copy()
    before = grid._mc_rel.to_numpy(float)
    if selected.empty:
        status = {"status": "no_primary_rows_need_precision_refinement", "selected_count": 0, "threshold": float(args.data_error_fraction), "grid": str(grid_path), "frozen_baseline_unchanged": True, "production_outputs_modified": False}
        (out / "primary_refinement_status.json").write_text(json.dumps(status, indent=2) + "\n")
        print(json.dumps(status, indent=2))
        return
    tasks = [(row.to_dict(), i, int(args.calls), out, float(args.g1)) for i, (_, row) in enumerate(selected.iterrows())]
    refined = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.max_workers)) as pool:
        futures = [pool.submit(one_row, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            refined.append(future.result())
            print(json.dumps(refined[-1]), flush=True)
    for item in refined:
        mask = grid.row_id.astype(str).eq(item["row_id"]) & grid.dataset.astype(str).eq(item["dataset"])
        grid.loc[mask, "raw_full_wy_fb_per_bin"] = item["raw_full_wy_fb_per_bin"]
        grid.loc[mask, "raw_full_wy_unc_fb_per_bin"] = item["raw_full_wy_unc_fb_per_bin"]
        grid.loc[mask, "full_wy_pb_per_GeV"] = item["raw_full_wy_fb_per_bin"] / item["width"] / 1000.0
        grid.loc[mask, "full_wy_unc_pb_per_GeV"] = item["raw_full_wy_unc_fb_per_bin"] / item["width"] / 1000.0
        grid.loc[mask, "full_wy_to_data_ratio"] = grid.loc[mask, "full_wy_pb_per_GeV"] / grid.loc[mask, "data_pb_per_GeV"]
    grid = grid.drop(columns=["_mc_rel", "_data_rel"])
    grid.to_csv(grid_path, index=False)
    after = (grid.full_wy_unc_pb_per_GeV / grid.full_wy_pb_per_GeV).abs().to_numpy(float)
    status = {
        "status": "primary_precision_refinement_complete_not_promoted",
        "grid": str(grid_path), "selected_count": len(refined), "threshold_data_error_fraction": float(args.data_error_fraction),
        "calls_per_refined_component": int(args.calls), "max_relative_mc_before": float(np.max(before)), "max_relative_mc_after": float(np.max(after)),
        "refined": refined, "frozen_baseline_unchanged": True, "production_outputs_modified": False,
    }
    (out / "primary_refinement_status.json").write_text(json.dumps(status, indent=2) + "\n")
    grid_status_path = grid_path.parent / "grid_status.json"
    if grid_status_path.exists():
        grid_status = json.loads(grid_status_path.read_text())
        grid_status["precision_refinement"] = status
        grid_status_path.write_text(json.dumps(grid_status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
