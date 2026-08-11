#!/usr/bin/env python3
"""Continue the campaign if the already-loaded coordinator stops at lambda=0.

The active coordinator predates the exact-lower-boundary handling in
``continue_minimum_fitbar_search.py``.  This durable successor waits for it,
repairs only that machine-readable classification when all registered trials
are terminal and lambda=0 survived, then launches the patched continuation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


BASE = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/dustin/miniforge3/envs/pdf-fit/bin/python")
SCRIPTS = BASE / "scripts"
SEARCH = BASE / "summaries/minimum_fitbar_constraint_search/summary.json"
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, required=True)
    args = parser.parse_args()
    while running(args.wait_pid):
        time.sleep(30)
    final = BASE / "summaries/final_study_report/summary.json"
    if final.exists() and json.loads(final.read_text()).get("status") == "complete":
        existing = json.loads(SEARCH.read_text()) if SEARCH.exists() else {}
        registered = existing.get("candidate_order_strong_to_weak", [])
        if [float(value) for value in registered] == REQUIRED_STRENGTHS:
            return
    search = json.loads(SEARCH.read_text())
    tested = search.get("tested_candidates", [])
    if (search.get("status") == "unbracketed_below_525"
            and float(search.get("selected_weakest_surviving_strength")) == 0.0
            and any(float(row["strength"]) == 0.0 and row["outcome"] == "survives"
                    for row in tested)):
        search["status"] = "complete"
        search["strongest_rejected_strength_below_selection"] = None
        SEARCH.write_text(json.dumps(search, indent=2) + "\n")
    elif search.get("status") not in {"complete", "in_progress"}:
        raise RuntimeError(f"predecessor stopped in unexpected state: {search.get('status')}")
    subprocess.run([
        str(PYTHON), str(SCRIPTS / "recover_after_lower_survival.py"),
        "--wait-pid", "2147483647",
    ], check=True)


if __name__ == "__main__":
    main()
