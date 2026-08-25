#!/usr/bin/env python3
"""After replicas, validate outlier starts and build the crossed F_NP record."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
REPORTS = BASE / "reports"
REPLICA_STATUS = REPORTS / "scope_353_coupled_replica_batch_status.json"
LONG_STATUS = REPORTS / "scope_353_long_outlier_start_batch_status.json"


def wait_for(path: Path, complete: str, stream) -> dict:
    while True:
        if path.exists():
            try:
                status = json.loads(path.read_text())
            except json.JSONDecodeError:
                status = {}
            if status.get("status") == complete:
                return status
            if status.get("status", "").endswith("incomplete"):
                raise RuntimeError(f"incomplete batch: {path}")
        time.sleep(30)


def run(cmd: list[str], stream) -> None:
    proc = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT)
    if proc.returncode:
        raise RuntimeError(f"command failed with {proc.returncode}: {cmd[0]}")


def main() -> None:
    log = REPORTS / "scope_353_postreplica_supervisor.log"
    with log.open("a") as stream:
        stream.write("postreplica supervisor started\n")
        stream.flush()
        wait_for(REPLICA_STATUS, "isolated_scope_353_replica_batch_complete", stream)
        stream.write("replica batch complete; launching long outlier starts\n")
        stream.flush()
        if not LONG_STATUS.exists():
            run([
                "/home/dustin/miniforge3/envs/pdf-fit/bin/python",
                str(BASE / "scripts/run_scope_353_long_outlier_starts.py"),
                "--seeds", "310", "311", "319", "322", "--epochs", "15000", "--patience", "2500", "--workers", "4",
            ], stream)
        wait_for(LONG_STATUS, "isolated_scope_353_long_outlier_start_batch_complete", stream)
        stream.write("long outlier starts complete; building start and crossed summaries\n")
        stream.flush()
        run([
            "/home/dustin/miniforge3/envs/pdf-fit/bin/python",
            str(BASE / "scripts/summarize_scope_353_start_batch.py"),
            "--seeds", *[str(x) for x in range(303, 327)],
            "--long-seeds", "310", "311", "319", "322",
            "--tag", "scope_353_coupled_start_summary_final",
        ], stream)
        run([
            "/home/dustin/miniforge3/envs/pdf-fit/bin/python",
            str(BASE / "scripts/summarize_scope_353_replica_batch.py"),
            "--start-seeds", *[str(x) for x in range(303, 327)],
            "--long-start-seeds", "310", "311", "319", "322",
            "--replica-seeds", *[str(x) for x in range(1001, 1051)],
            "--tag", "scope_353_start_replica_propagation_final",
        ], stream)
        stream.write("postreplica summary complete\n")


if __name__ == "__main__":
    main()
