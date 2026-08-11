#!/usr/bin/env python3
"""Resume the full campaign if the original watcher stops on a lower survivor."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from alternate_lambda_authorization import (
    require_alternate_lambda_authorization,
)


BASE = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/dustin/miniforge3/envs/pdf-fit/bin/python")
SCRIPTS = BASE / "scripts"
REQUIRED_STRENGTHS = [
    637.5, 600.0, 562.5, 525.0, 487.5, 450.0, 412.5, 375.0,
    337.5, 300.0, 262.5, 225.0, 187.5, 150.0, 112.5, 75.0, 37.5, 0.0,
]


def running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    stat = Path(f"/proc/{pid}/stat")
    return not stat.exists() or stat.read_text().split()[2] != "Z"


def run(name: str, *args: str) -> None:
    subprocess.run([str(PYTHON), str(SCRIPTS / name), *args], check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, required=True)
    args = parser.parse_args()
    while running(args.wait_pid):
        time.sleep(30)
    final = BASE / "summaries/final_study_report/summary.json"
    minimum = BASE / "summaries/minimum_fitbar_constraint_search/summary.json"
    if final.exists() and json.loads(final.read_text()).get("status") == "complete":
        registered = (json.loads(minimum.read_text()).get(
            "candidate_order_strong_to_weak", []) if minimum.exists() else [])
        if [float(value) for value in registered] == REQUIRED_STRENGTHS:
            return
    lower = json.loads((BASE / "summaries/lambda637p5_fitbar_minimum_control/summary.json").read_text())
    if lower.get("status") != "lower_candidate_survives_discriminator":
        raise RuntimeError("original pipeline stopped without a lower survivor")
    require_alternate_lambda_authorization("recover_after_lower_survival")
    run("continue_minimum_fitbar_search.py")
    selection = json.loads(minimum.read_text())
    if selection["status"] != "complete":
        raise RuntimeError("minimum fit-barrier search is not bracketed")
    strength = float(selection["selected_weakest_surviving_strength"])
    rejected_value = selection["strongest_rejected_strength_below_selection"]
    run("continue_minimum_barrier_search.py")
    barrier_selection = json.loads((
        BASE / "summaries/minimum_barrier_constraint_search/summary.json"
    ).read_text())
    if barrier_selection["status"] != "complete":
        raise RuntimeError("barrier-strength search is not complete")
    barrier_strength = float(
        barrier_selection["selected_weakest_surviving_barrier_strength"])
    verify_args = [
        "--strength", str(strength),
        "--fit-quality-barrier-strength", str(barrier_strength),
        "--fit-quality-barrier-power", "2",
    ]
    if rejected_value is not None:
        verify_args.extend([
            "--strongest-failing-strength", str(float(rejected_value))])
    run("verify_replica_robust_reference_full24.py", *verify_args)
    run("summarize_replica_robust_constraint_scale.py")
    run("supervise_selected_reference_central_replicas.py")
    run("finish_campaign_automatically.py", "--wait-pid", "2147483647")


if __name__ == "__main__":
    main()
