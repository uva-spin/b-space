#!/usr/bin/env python3
"""Generate the isolated external full N3LL+NNLO W+Y Tevatron grid.

One DYTurbo card is used per Tevatron dataset with all of that dataset's qT
bins.  This is an observable-level perturbative grid, not yet a fitted F_NP
production extraction.  All outputs stay under the candidate directory.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from run_dyturbo_full_n3ll_nnlo_probe import DYTURBO, DYROOT, full_card_text, load_runner


SYSTEMATICS = Path(__file__).resolve().parents[2]
PROJECT = SYSTEMATICS.parent
DATA_ROOT = PROJECT / "Data/v23a_tevatron_plus_lhcb7_fiducial_candidate"
DATASETS = ("CDF_RUN_1", "CDF_RUN_2", "D0_RUN_1")
OUT = SYSTEMATICS / "full_n3ll_wy_production_2026/reports/dyturbo_full_n3ll_nnlo_tevatron_grid"


def parse_grid(path: Path, *, first_edge: float, last_edge: float) -> pd.DataFrame:
    records = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 4 or parts[0].startswith("#"):
            continue
        try:
            lo, hi, value, unc = map(float, parts[:4])
        except ValueError:
            continue
        records.append({"qT_low": lo, "qT_high": hi, "raw_full_wy_fb_per_bin": value, "raw_full_wy_unc_fb_per_bin": unc})
    # DYTurbo appends an inclusive total line for multi-bin text tables.  It
    # has the full edge range and must not be mistaken for a qT bin.
    records = [r for r in records if not (abs(r["qT_low"] - first_edge) < 1e-12 and abs(r["qT_high"] - last_edge) < 1e-12)]
    if not records:
        raise RuntimeError(f"no grid rows parsed from {path}")
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g1", type=float, default=0.0)
    parser.add_argument("--calls", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=246810)
    parser.add_argument("--timeout", type=int, default=7_200,
                        help="per-dataset DYTurbo subprocess timeout in seconds")
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS),
                        help="dataset subset; existing tables can be consolidated by rerunning all three")
    args = parser.parse_args()
    out_dir = Path(args.out).resolve()
    runner = load_runner()
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(p for p in ("/home/dustin/miniforge3/envs/pdf-fit/lib", str(DYROOT / "lib"), env.get("LD_LIBRARY_PATH", "")) if p)
    all_records = []
    calls = int(args.calls)
    selected_datasets = tuple(args.datasets)
    for dataset in selected_datasets:
        data = pd.read_csv(DATA_ROOT / f"{dataset}.csv").sort_values("qT_low").copy()
        if data.empty:
            raise RuntimeError(dataset)
        edges = [float(data.iloc[0].qT_low)] + [float(x) for x in data.qT_high]
        if len(data) > 1 and not (abs(data.qT_low.iloc[1:].to_numpy(float) - data.qT_high.iloc[:-1].to_numpy(float)) < 1e-9).all():
            raise RuntimeError(f"non-contiguous qT bins for {dataset}")
        first = data.iloc[0]
        g1_tag = f"{args.g1:g}".replace(".", "p").replace("-", "m")
        seed_tag = str(int(args.seed))
        name = f"{dataset}_full_n3ll_nnlo_grid_g1_{g1_tag}_seed_{seed_tag}"
        card = out_dir / "cards" / f"{name}.in"
        log = out_dir / "logs" / f"{name}.log"
        table = DYROOT / f"{name}.txt"
        card.parent.mkdir(parents=True, exist_ok=True)
        log.parent.mkdir(parents=True, exist_ok=True)
        text = full_card_text(first, output_name=name, pdf_set="NNPDF40_nnlo_as_01180", pdf_member=0, cores=8, calls=calls, seed=int(args.seed))
        text = text.replace("g1 = 0.0", f"g1 = {args.g1:.12g}", 1)
        edges_text = " ".join(f"{x:.12g}" for x in edges)
        text = re.sub(r"qt_bins = \[ [^\]]+ \]", f"qt_bins = [ {edges_text} ]", text, count=1)
        card.write_text(text)
        # A supervisor may have interrupted a previous card after DYTurbo
        # opened its table.  Treat an unparsable/partial table as incomplete
        # and rerun it rather than silently accepting truncated output.
        if table.exists():
            try:
                existing = parse_grid(table, first_edge=edges[0], last_edge=edges[-1])
                if len(existing) != len(data):
                    table.unlink()
            except (RuntimeError, OSError, ValueError):
                table.unlink()
        if not table.exists():
            with log.open("w") as handle:
                subprocess.run([str(DYTURBO), str(card)], cwd=DYROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True, timeout=int(args.timeout))
        grid = parse_grid(table, first_edge=edges[0], last_edge=edges[-1])
        if len(grid) != len(data):
            raise RuntimeError(f"{dataset}: parsed {len(grid)} bins, expected {len(data)}")
        if not np.isfinite(grid[["raw_full_wy_fb_per_bin", "raw_full_wy_unc_fb_per_bin"]].to_numpy(float)).all():
            # Do not allow a cancellation-induced NaN/inf table to be reused
            # on a later restart. A seed-rotated supervisor can retry it.
            table.unlink(missing_ok=True)
            raise RuntimeError(f"{dataset}: DYTurbo table contains nonfinite W+Y values")
        if not (grid.raw_full_wy_fb_per_bin.to_numpy(float) > 0).all():
            table.unlink(missing_ok=True)
            raise RuntimeError(f"{dataset}: DYTurbo table contains nonpositive W+Y values")
        merged = data[["dataset", "row_id", "qT_low", "qT_high", "CS", "error"]].reset_index(drop=True).join(grid.reset_index(drop=True), rsuffix="_grid")
        if not (abs(merged.qT_low - merged.qT_low_grid) < 1e-9).all() or not (abs(merged.qT_high - merged.qT_high_grid) < 1e-9).all():
            raise RuntimeError(f"{dataset}: grid bin edges do not match input")
        width = merged.qT_high - merged.qT_low
        merged["data_pb_per_GeV"] = merged.CS
        merged["data_unc_pb_per_GeV"] = merged.error
        merged["full_wy_pb_per_GeV"] = merged.raw_full_wy_fb_per_bin / width / 1000.0
        merged["full_wy_unc_pb_per_GeV"] = merged.raw_full_wy_unc_fb_per_bin / width / 1000.0
        merged["full_wy_to_data_ratio"] = merged.full_wy_pb_per_GeV / merged.data_pb_per_GeV
        all_records.append(merged.drop(columns=["qT_low_grid", "qT_high_grid"]))
    result = pd.concat(all_records, ignore_index=True)
    result.to_csv(out_dir / "tevatron_full_wy_grid.csv", index=False)
    rel_unc = result.raw_full_wy_unc_fb_per_bin / result.raw_full_wy_fb_per_bin
    status = {
        "status": "isolated_tevatron_full_n3ll_nnlo_wy_grid_passed",
        "engine": str(DYTURBO),
        "order": 3,
        "primed": False,
        "enabled_terms": ["resummation_W_N3LL", "counterterm_ASY_NNLO", "VJ_NNLO"],
        "runtime_card": {
            "fixedorder_only": False,
            "order": 3,
            "primed": False,
            "doBORN": True,
            "doCT": True,
            "doVJREAL": True,
            "doVJVIRT": True,
            "VJquad": False,
            "matching_identity": "RES + CT + VJ = W + (FO_NNLO - ASY_NNLO)",
        },
        "datasets": list(selected_datasets),
        "row_count": len(result),
        "calls_per_vegas_component": calls,
        "npff": 0,
        "g1_GeV2": float(args.g1),
        "random_seed": int(args.seed),
        "checks": {
            "all_finite": bool(result[["full_wy_pb_per_GeV", "full_wy_unc_pb_per_GeV"]].notna().all().all()),
            "all_positive": bool((result.full_wy_pb_per_GeV > 0).all()),
            "mean_relative_mc_uncertainty": float(rel_unc.mean()),
            "max_relative_mc_uncertainty": float(rel_unc.max()),
            "full_wy_to_data_ratio_median": float(result.full_wy_to_data_ratio.median()),
            "full_wy_to_data_ratio_min": float(result.full_wy_to_data_ratio.min()),
            "full_wy_to_data_ratio_max": float(result.full_wy_to_data_ratio.max()),
        },
        "meaning": "external full W+Y perturbative grid; no custom fitted F_NP and no production authorization",
        "frozen_baseline_unchanged": True,
        "production_outputs_modified": False,
    }
    (out_dir / "grid_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
