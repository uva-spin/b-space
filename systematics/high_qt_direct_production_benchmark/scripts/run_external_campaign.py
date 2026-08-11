#!/usr/bin/env python3
"""Resumable row-isolated DYTurbo/MCFM campaign orchestration."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "systematics" / "high_qt_direct_production_benchmark"
PLAN = STUDY / "summaries" / "benchmark_batch_plan.csv"
PYTHON = Path("/home/dustin/miniforge3/envs/pdf-fit/bin/python")


def runner(dataset: str, code: str) -> Path:
    if dataset == "LHCb_7":
        name = f"run_lhcb7_{code}_benchmark.py"
        return ROOT / "systematics" / "finite_y_tail_benchmark" / "scripts" / name
    return ROOT / "v23" / "tools" / f"run_tevatron_{code}_benchmark.py"


def summary_name(code: str) -> str:
    return f"{code}_benchmark_summary.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True, choices=["tier1_boundary", "tier2_high_qt", "tier3_exceptional"])
    ap.add_argument("--codes", nargs="+", choices=["dyturbo", "mcfm"], default=["dyturbo", "mcfm"])
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None, help="Run only the first N selected rows (smoke/pilot use).")
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--keep-going", action="store_true")
    ap.add_argument("--dyturbo-cores", type=int, default=4)
    ap.add_argument("--dyturbo-timeout", type=int, default=900)
    ap.add_argument("--mcfm-calls", type=int, default=1000000)
    ap.add_argument("--mcfm-timeout", type=int, default=1200)
    args = ap.parse_args()

    plan = pd.read_csv(PLAN)
    plan = plan.loc[plan["benchmark_tier"].eq(args.tier)].copy()
    if args.datasets:
        plan = plan.loc[plan["dataset"].isin(args.datasets)]
    plan = plan.sort_values(["dataset", "qT_over_Q", "row_id"])
    if args.limit is not None:
        plan = plan.head(args.limit)
    if plan.empty:
        raise SystemExit("No pending rows selected")

    events = STUDY / "logs" / f"{args.tier}_campaign.jsonl"
    failures = 0
    for _, row in plan.iterrows():
        dataset = str(row.dataset)
        row_id = str(row.row_id)
        row_tag = row_id.replace(":", "_").lower()
        data = ROOT / "Data" / "v23a_tevatron_plus_lhcb7_fiducial_candidate" / f"{dataset}.csv"
        for code in args.codes:
            out = STUDY / "outputs" / "central" / args.tier / dataset.lower() / row_tag / code
            summary = out / summary_name(code)
            if summary.exists() and not args.rerun:
                print(f"SKIP complete {row_id} {code}", flush=True)
                continue
            command = [str(PYTHON), str(runner(dataset, code)), "--data", str(data), "--rows", row_id, "--out", str(out)]
            if code == "dyturbo":
                command += ["--cores", str(args.dyturbo_cores), "--timeout", str(args.dyturbo_timeout)]
            else:
                command += ["--calls", str(args.mcfm_calls), "--timeout", str(args.mcfm_timeout)]
            started = time.monotonic()
            event = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(), "tier": args.tier,
                "dataset": dataset, "row_id": row_id, "code": code, "command": command,
            }
            print(f"RUN {row_id} {code}", flush=True)
            try:
                subprocess.run(command, cwd=ROOT, check=True)
                event["status"] = "pass"
            except subprocess.CalledProcessError as exc:
                event["status"] = "fail"
                event["returncode"] = exc.returncode
                failures += 1
            event["elapsed_seconds"] = time.monotonic() - started
            with events.open("a") as stream:
                stream.write(json.dumps(event) + "\n")
            if event["status"] == "fail" and not args.keep_going:
                raise SystemExit(f"Failed {row_id} {code}; see {events}")
    if failures:
        raise SystemExit(f"Campaign completed with {failures} failures; see {events}")


if __name__ == "__main__":
    main()
