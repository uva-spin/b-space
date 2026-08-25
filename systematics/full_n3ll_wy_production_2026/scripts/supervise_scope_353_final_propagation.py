#!/usr/bin/env python3
"""Build the final isolated start x replica propagation after all validation.

This supervisor deliberately waits for both long-horizon validation batches,
then replaces the selected outlier curves in the crossed ensemble before
rendering Fig. 2 and Fig. 6.  It never writes to frozen production paths.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
REPORTS = BASE / "reports"
PYTHON = "/home/dustin/miniforge3/envs/pdf-fit/bin/python"
STARTS = list(range(303, 327))
REPLICAS = list(range(1001, 1051))
LONG_STARTS = [310, 311, 317, 319, 322]
LONG_REPLICAS = [1008, 1033, 1040, 1010, 1038, 1026]


def wait_for(path: Path, expected: str, stream) -> dict:
    while True:
        if path.exists():
            try:
                data = json.loads(path.read_text())
            except json.JSONDecodeError:
                data = {}
            status = data.get("status", "")
            if status == expected:
                return data
            if status.endswith("incomplete"):
                raise RuntimeError(f"incomplete batch: {path}: {status}")
        stream.write(f"waiting for {path.name}\n")
        stream.flush()
        time.sleep(30)


def wait_for_file(path: Path, stream) -> None:
    while not path.exists():
        stream.write(f"waiting for {path.name}\n")
        stream.flush()
        time.sleep(30)


def run(cmd: list[str], stream) -> None:
    stream.write("running: " + " ".join(cmd) + "\n")
    stream.flush()
    proc = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT)
    if proc.returncode:
        raise RuntimeError(f"command failed ({proc.returncode}): {cmd[0]}")


def main() -> None:
    log_path = REPORTS / "scope_353_final_propagation_supervisor.log"
    with log_path.open("a") as stream:
        stream.write("final propagation supervisor started\n")
        stream.flush()
        # Seed 317 was identified by the completed 24-start scan as a poor
        # basin and is revalidated separately from the original four starts.
        wait_for_file(
            REPORTS / "scope_353_coupled_fnp_fit_lambda1_candidate_s317_csvnorm_long/metrics.json",
            stream,
        )
        wait_for(REPORTS / "scope_353_long_outlier_start_batch_status.json",
                 "isolated_scope_353_long_outlier_start_batch_complete", stream)
        wait_for(REPORTS / "scope_353_long_outlier_replica_batch_status.json",
                 "isolated_scope_353_long_outlier_replica_batch_complete", stream)
        run([
            PYTHON, str(BASE / "scripts/summarize_scope_353_start_batch.py"),
            "--seeds", *map(str, STARTS),
            "--long-seeds", *map(str, LONG_STARTS),
            "--tag", "scope_353_coupled_start_summary_final",
        ], stream)
        run([
            PYTHON, str(BASE / "scripts/summarize_scope_353_replica_batch.py"),
            "--start-seeds", *map(str, STARTS),
            "--long-start-seeds", *map(str, LONG_STARTS),
            "--replica-seeds", *map(str, REPLICAS),
            "--long-replica-seeds", *map(str, LONG_REPLICAS),
            "--tag", "scope_353_start_replica_propagation_final",
        ], stream)
        run([
            PYTHON, str(BASE / "scripts/plot_scope_353_fnp_start_replica_fig2_fig6.py"),
        ], stream)
        final = {
            "status": "isolated_scope_353_final_propagation_complete",
            "start_count": len(STARTS),
            "replica_count": len(REPLICAS),
            "long_start_replacements": LONG_STARTS,
            "long_replica_replacements": LONG_REPLICAS,
            "crossed_member_count": len(STARTS) * len(REPLICAS),
            "cross_rule": "log(F_NP) start curves plus centered log(F_NP) replica residuals",
            "figures": {
                "fig2": str(REPORTS / "scope_353_final_fig2_fig6/fig2.png"),
                "fig6": str(REPORTS / "scope_353_final_fig2_fig6/fig6.png"),
            },
            "frozen_production_modified": False,
            "promotion_authorized": False,
        }
        (REPORTS / "scope_353_final_propagation_status.json").write_text(
            json.dumps(final, indent=2) + "\n"
        )
        stream.write(json.dumps(final, indent=2) + "\n")


if __name__ == "__main__":
    main()
