#!/usr/bin/env python3
"""Keep the isolated 48-start controller alive across transient exits."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time


BASE = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/dustin/miniforge3/envs/pdf-fit/bin/python")
CONTROLLER = BASE / "scripts/run_lambda1_start_expansion48.py"
TARGET = BASE / "summaries/lambda1_start_expansion48"
LOG = BASE / "logs/lambda1_start_expansion48.watchdog.log"


def controller_active() -> bool:
    result = subprocess.run(["pgrep", "-af", "run_lambda1_start_expansion48.py --launch"],
                            capture_output=True, text=True, check=False)
    return bool(result.stdout.strip())


def terminal() -> bool:
    path = TARGET / "summary.json"
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text()).get("status") in {
            "complete", "horizon_reached_without_plateau"
        }
    except (OSError, json.JSONDecodeError):
        return False


def main() -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as log:
        log.write(f"watchdog start {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
        while not terminal():
            if not controller_active():
                log.write(f"restart {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
                log.flush()
                subprocess.Popen([str(PYTHON), str(CONTROLLER), "--launch"],
                                 cwd=BASE.parent, stdout=log, stderr=subprocess.STDOUT)
            time.sleep(30)
        log.write(f"watchdog terminal {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")


if __name__ == "__main__":
    main()
