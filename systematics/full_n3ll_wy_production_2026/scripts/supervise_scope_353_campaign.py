#!/usr/bin/env python3
"""Continue the isolated scope campaign without an interactive prompt."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
REPORTS = BASE / "reports"
START_STATUS = REPORTS / "scope_353_coupled_start_batch_status.json"
REPLICA_STATUS = REPORTS / "scope_353_coupled_replica_batch_status.json"


def main() -> None:
    log = REPORTS / "scope_353_campaign_supervisor.log"
    with log.open("a") as stream:
        stream.write("supervisor started\n")
        stream.flush()
        while True:
            if START_STATUS.exists():
                try:
                    status = json.loads(START_STATUS.read_text())
                except json.JSONDecodeError:
                    status = {}
                if status.get("status") == "isolated_scope_353_start_batch_complete":
                    break
                if status.get("status") == "isolated_scope_353_start_batch_incomplete":
                    stream.write("start batch incomplete; replica stage not started\n")
                    return
            time.sleep(30)
        if REPLICA_STATUS.exists():
            stream.write("replica status already present; no duplicate launch\n")
            return
        stream.write("start batch complete; launching 50-replica batch\n")
        stream.flush()
        cmd = [
            "/home/dustin/miniforge3/envs/pdf-fit/bin/python",
            str(BASE / "scripts/run_scope_353_coupled_replica_batch.py"),
            "--seed-start", "1001", "--count", "50", "--workers", "4",
            "--epochs", "5000", "--patience", "1000",
        ]
        proc = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT)
        stream.write(f"replica stage exited returncode={proc.returncode}\n")


if __name__ == "__main__":
    main()
