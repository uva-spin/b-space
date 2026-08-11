#!/usr/bin/env python3
"""Run/resume the authorized 24-row central genuine-Z+jet-NLO campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "systematics/high_qt_direct_production_benchmark"
EXP = BASE / "experimental_unitary_transition"
ROW_LIST = EXP / "outputs/unitary_smootherstep_v1_nodes_nb640_nqt2_ny2_simpson/tevatron_rows.csv"
OUT = EXP / "outputs/mcfm_zjet_nlo_24row_central_500k_i10"
RUNNER = EXP / "scripts/run_tevatron_mcfm_nlo_audit.py"
DATA = ROOT / "Data/v23a_fixed_target_plus_tevatron_absolute_fit_ready"
REP = EXP / "outputs/mcfm_zjet_nlo_representative/central_500k_i10"
ANCHOR = EXP / "outputs/mcfm_zjet_nlo_order_audit/cdf_run_2_51_controlled_500k_i10"


def tag(row_id: str) -> str:
    return row_id.lower().replace(":", "_")


def seed_completed(row_id: str, destination: Path) -> bool:
    sources = {
        "CDF_RUN_2:36": REP,
        "CDF_RUN_2:45": REP,
        "CDF_RUN_2:51": ANCHOR,
    }
    source = sources.get(row_id)
    if source is None:
        return False
    frame = pd.read_csv(source / "mcfm_benchmark_summary.csv")
    selected = frame.loc[frame.row_id == row_id]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one seeded result for {row_id} in {source}")
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("cards", "logs", "tables"):
        source_dir = source / name
        destination_dir = destination / name
        destination_dir.mkdir(exist_ok=True)
        row_number = row_id.split(":", 1)[1]
        for path in source_dir.glob(f"*tev{row_number}*"):
            shutil.copy2(path, destination_dir / path.name)
    selected.to_csv(destination / "mcfm_benchmark_summary.csv", index=False)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--no-aggregate", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("shard-index must be in [0, shard-count)")
    requested = pd.read_csv(ROW_LIST)[["dataset", "row_id"]].drop_duplicates()
    assigned = requested.iloc[args.shard_index::args.shard_count]
    OUT.mkdir(parents=True, exist_ok=True)
    for position, item in zip(assigned.index + 1, assigned.itertuples(index=False)):
        row_out = OUT / "rows" / tag(item.row_id)
        summary = row_out / "mcfm_benchmark_summary.csv"
        if summary.exists():
            frame = pd.read_csv(summary)
            if len(frame) == 1 and frame.row_id.iloc[0] == item.row_id:
                print(f"[{position:02d}/24] resume {item.row_id}", flush=True)
                continue
        if seed_completed(item.row_id, row_out):
            print(f"[{position:02d}/24] seed {item.row_id}", flush=True)
            continue
        print(f"[{position:02d}/24] run {item.row_id}", flush=True)
        subprocess.run([
            sys.executable, str(RUNNER),
            "--nlo-real-calls", "500000",
            "--nlo-virtual-calls", "500000",
            "--iterbatch-warmup", "2",
            "--iterbatch-first", "10",
            "--iterbatch-later", "1",
            "--calls", "500000",
            "--data", str(DATA / f"{item.dataset}.csv"),
            "--rows", item.row_id,
            "--mu-r-factor", "1",
            "--mu-f-factor", "1",
            "--out", str(row_out),
            "--timeout", "900",
        ], check=True)

    if args.no_aggregate:
        return
    frames = [pd.read_csv(OUT / "rows" / tag(row_id) / "mcfm_benchmark_summary.csv")
              for row_id in requested.row_id]
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(OUT / "mcfm_benchmark_summary.csv", index=False)
    status = {
        "status": "experimental_unitary_transition_not_production",
        "tag": "mcfm_zjet_nlo_24row_central_500k_i10",
        "row_count": int(len(result)),
        "requested_row_count": int(len(requested)),
        "all_rows_complete": bool(len(result) == len(requested)),
        "nlo_real_calls": 500000,
        "nlo_virtual_calls": 500000,
        "result_iterations": 10,
        "central_refit_authorized": False,
        "replica_stability_authorized": False,
        "next_gate": "analyze 24-row NLO normalization, uncertainty, and frozen-fit impact",
    }
    (OUT / "campaign_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
