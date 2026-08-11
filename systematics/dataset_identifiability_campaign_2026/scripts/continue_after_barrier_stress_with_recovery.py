#!/usr/bin/env python3
"""Keep the selected-reference campaign moving after the active stress audit.

The already-running watcher owns the ordinary success path.  This companion
owns only the terminal-failure path: it runs the controlled stronger-strength
bracket and then resumes the same full24/replica/finalization chain.
"""

from __future__ import annotations

import argparse
import fcntl
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
STAGED = BASE / "summaries/lambda675_fit_quality_barrier_stress/summary.json"
LOCK = BASE / "summaries/barrier_stress_recovery_supervisor.lock"


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    stat = Path(f"/proc/{pid}/stat")
    return not stat.exists() or stat.read_text().split()[2] != "Z"


def run(name: str) -> None:
    subprocess.run([str(PYTHON), "-u", str(SCRIPTS / name)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, required=True)
    args = parser.parse_args()
    require_alternate_lambda_authorization(
        "continue_after_barrier_stress_with_recovery")
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock.write(f"pid={os.getpid()} wait_pid={args.wait_pid}\n")
        lock.flush()
        while alive(args.wait_pid):
            time.sleep(30)
        status = json.loads(STAGED.read_text())["status"]
        if status == "complete":
            # The pre-existing success watcher owns this path.
            return
        if status != "stress_failed":
            raise RuntimeError(f"unexpected stress status: {status}")
        for script in (
            "recover_barrier_reference_strength_after_stress_failure.py",
            "verify_replica_robust_reference_full24.py",
            "summarize_replica_robust_constraint_scale.py",
            "supervise_selected_reference_central_replicas.py",
        ):
            run(script)
        subprocess.run([
            str(PYTHON), "-u", str(SCRIPTS / "finish_campaign_automatically.py"),
            "--wait-pid", "2147483647",
        ], check=True)


if __name__ == "__main__":
    main()
