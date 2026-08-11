#!/usr/bin/env python3
"""Finish the selected-reference campaign after an already-running bracket.

This supervisor deliberately fails closed.  It waits for the external bracket
process, confirms difficult replica endpoints, and, if that stricter audit
fails, feeds the newly exposed identities back into the strength escalation.
Only a passing boundary audit can trigger construction of the final ensemble.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from alternate_lambda_authorization import (
    require_alternate_lambda_authorization,
)


BASE = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/dustin/miniforge3/envs/pdf-fit/bin/python")
SCRIPTS = BASE / "scripts"
CAMPAIGN = BASE / "summaries/selected_reference_central_replicas"
PRIMARY = CAMPAIGN / "summary.json"
FAILED = CAMPAIGN / "failed_80k_summary.json"
BOUNDARY = BASE / "summaries/selected_reference_boundary_confirmation/summary.json"
SUPERVISOR_LOCK = BASE / "summaries/automatic_finisher.lock"


def running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    stat = Path(f"/proc/{pid}/stat")
    if stat.exists():
        # A completed unified-terminal process can remain briefly visible as a
        # zombie until its session is polled.  It must not hold this supervisor.
        fields = stat.read_text().split()
        if len(fields) > 2 and fields[2] == "Z":
            return False
    return True


def run(script: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run([str(PYTHON), str(SCRIPTS / script)], check=check)


def archive_failed_primary() -> None:
    result = json.loads(PRIMARY.read_text())
    strength = float(result["selected_strength"])
    archive = CAMPAIGN / f"failed_lam{strength:g}_boundary_summary.json"
    shutil.copy2(PRIMARY, archive)
    if FAILED.exists():
        prior = json.loads(FAILED.read_text())
        prior_strength = float(prior["selected_strength"])
        shutil.copy2(FAILED, CAMPAIGN / f"failed_lam{prior_strength:g}_80k_summary.json")
    shutil.copy2(PRIMARY, FAILED)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, required=True)
    args = parser.parse_args()
    SUPERVISOR_LOCK.parent.mkdir(parents=True, exist_ok=True)
    lock = SUPERVISOR_LOCK.open("w")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        # Another supervisor owns the complete downstream transition.
        return
    lock.write(f"pid={os.getpid()} wait_pid={args.wait_pid}\n")
    lock.flush()
    while running(args.wait_pid):
        time.sleep(30)

    while True:
        primary = json.loads(PRIMARY.read_text())
        if primary["status"] != "complete":
            raise RuntimeError("strength escalation ended without a complete 50-replica campaign")
        confirmation = run("confirm_boundary_replicas.py", check=False)
        boundary = json.loads(BOUNDARY.read_text())
        if confirmation.returncode == 0 and boundary["status"] == "complete":
            break
        if not boundary.get("failed_replica_seeds"):
            raise RuntimeError("boundary confirmation failed without identified replicas")
        if primary.get("staged_prescription") or float(
                primary.get("fit_quality_barrier_strength", 0.0)) > 0:
            raise RuntimeError(
                "staged lambda675+fit-barrier boundary audit failed; blind "
                "reference-strength escalation is scientifically invalid. "
                "Classify the failure as FNP stationarity, fit quality, or "
                "cross-optimizer disagreement before changing one constraint.")
        require_alternate_lambda_authorization("finish_campaign_automatically")
        archive_failed_primary()
        run("escalate_reference_strength_after_full50_failure.py")

    for script in (
        "build_final_combined_tmd_ensemble.py",
        "audit_final_combined_ensemble.py",
        "plot_validated_final_fig2_fig6.py",
        "audit_campaign_completion.py",
        "write_final_study_report.py",
        "promote_validated_final_champion.py",
    ):
        run(script)


if __name__ == "__main__":
    main()
