#!/usr/bin/env python3
"""Keep GPU slots filled with selected-method conditional replicas.

This supervisor is intentionally confined to the 2026 identifiability
campaign.  It counts only active production-objective training children,
never modifies frozen inputs, and launches each requested replica at most
once unless its output lacks a completed fit status.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time


BASE = Path(__file__).resolve().parents[1]
LAUNCHER = BASE / "scripts" / "run_selected_fslope_closure_replica.py"
OUTPUTS = BASE / "outputs"
LOG = BASE / "logs" / "supervise_selected_closure_replicas.log"
PYTHON = Path("/home/dustin/miniforge3/envs/pdf-fit/bin/python")
TRAINING_MARKER = "run_production_fnp_stability_control.py"


def completed(tag: str) -> bool:
    path = OUTPUTS / tag / "fit_status.json"
    if not path.exists():
        return False
    try:
        status = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return bool(status.get("final"))


def active_trainings() -> int:
    result = subprocess.run(
        ["ps", "-eo", "args="], check=True, text=True, capture_output=True)
    return sum(
        line.strip().startswith(str(PYTHON))
        and TRAINING_MARKER in line
        for line in result.stdout.splitlines()
    )


def record(message: str) -> None:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with LOG.open("a") as stream:
        stream.write(f"{stamp} {message}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replica-first", type=int, default=2005)
    parser.add_argument("--replica-last", type=int, default=2024)
    parser.add_argument("--fit-seed-first", type=int, default=1605)
    parser.add_argument("--max-active", type=int, default=6)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument(
        "--source-fit",
        default="fslope_dense_lam0p01_c2closure_b5p5_pilot_s1511")
    args = parser.parse_args()
    pairs = [
        (replica, args.fit_seed_first + replica - args.replica_first)
        for replica in range(args.replica_first, args.replica_last + 1)
    ]
    children: dict[int, tuple[subprocess.Popen, object]] = {}
    launched: set[int] = set()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    record(
        f"start replicas={args.replica_first}:{args.replica_last} "
        f"max_active={args.max_active}")

    while True:
        for replica, (process, stream) in list(children.items()):
            code = process.poll()
            if code is not None:
                stream.close()
                record(f"finish replica={replica} returncode={code}")
                del children[replica]

        pending = []
        for replica, fit_seed in pairs:
            tag = (
                "selectedrep_fslope_lam0p01_c2closure_b5p5_"
                f"r{replica}_s{fit_seed}")
            if completed(tag):
                continue
            if replica not in launched:
                pending.append((replica, fit_seed, tag))
        if not pending and not children:
            record("complete")
            return

        capacity = max(args.max_active - active_trainings(), 0)
        for replica, fit_seed, tag in pending[:capacity]:
            log_path = BASE / "logs" / f"{tag}.log"
            stream = log_path.open("a")
            command = [
                str(PYTHON), str(LAUNCHER),
                "--source-fit", str(OUTPUTS / args.source_fit),
                "--replica-seed", str(replica),
                "--fit-seed", str(fit_seed),
            ]
            process = subprocess.Popen(
                command, cwd=BASE.parent, stdout=stream,
                stderr=subprocess.STDOUT)
            children[replica] = (process, stream)
            launched.add(replica)
            record(f"launch replica={replica} fit_seed={fit_seed} pid={process.pid}")
            time.sleep(1.0)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
