#!/usr/bin/env python3
"""Wait for the 12-basin batch, then run its deterministic ensemble audits."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time


BASE = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/dustin/miniforge3/envs/pdf-fit/bin/python")
SCRIPTS = BASE / "scripts"
LOG_DIR = BASE / "logs/local_candidate_multistart"
SEEDS = tuple(range(303, 315))
EXISTING_307 = (
    "logcurv5em5_fslope4em3_xslope3em4_"
    "c2closure_b5p5_s1971_init307"
)


def tag(seed: int) -> str:
    return EXISTING_307 if seed == 307 else f"selected_local_xslope3em4_init{seed}"


def run(command: list[str], stream) -> None:
    stream.write("$ " + " ".join(command) + "\n")
    stream.flush()
    subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                   text=True, check=True)


def main() -> None:
    marker = LOG_DIR / "COMPLETE"
    while not marker.exists():
        time.sleep(60)
    missing = [
        tag(seed) for seed in SEEDS
        if not (BASE / "outputs" / tag(seed) / "fit_status.json").exists()
    ]
    if missing:
        raise RuntimeError(f"completion marker exists but runs are missing: {missing}")

    b_name = "selected_local_xslope3em4_multistart_bspace"
    k_name = "selected_local_xslope3em4_multistart_kspace"
    commands = [
        [str(PYTHON), str(SCRIPTS / "audit_selected_local_multistart.py")],
        [
            str(PYTHON), str(SCRIPTS / "build_bspace_tmd_ensemble.py"),
            "--target-name", b_name,
            *sum((["--run-tag", tag(seed)] for seed in SEEDS), []),
        ],
        [
            str(PYTHON), str(SCRIPTS / "summarize_tmd_ensemble_stability.py"),
            "--ensemble-long",
            str(BASE / "summaries" / b_name / "bspace_tmd_ensemble_long.csv"),
            "--coordinate", "bT", "--target-name", f"{b_name}_stability",
            "--minimum-member-count", "12",
        ],
        [
            str(PYTHON), str(SCRIPTS / "transform_bspace_ensemble_to_kspace.py"),
            "--bspace-ensemble",
            str(BASE / "summaries" / b_name / "bspace_tmd_ensemble_long.csv"),
            "--target-name", k_name,
        ],
        [
            str(PYTHON), str(SCRIPTS / "summarize_tmd_ensemble_stability.py"),
            "--ensemble-long",
            str(BASE / "summaries" / k_name / "kspace_tmd_ensemble_long.csv"),
            "--coordinate", "kT", "--target-name", f"{k_name}_stability",
            "--minimum-member-count", "12",
        ],
        [
            str(PYTHON), str(SCRIPTS / "audit_kspace_transform_robustness.py"),
            "--bspace-ensemble",
            str(BASE / "summaries" / b_name / "bspace_tmd_ensemble_long.csv"),
            "--target-name", f"{k_name}_transform_robustness",
        ],
    ]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with (LOG_DIR / "postprocess.log").open("w") as stream:
        for command in commands:
            run(command, stream)
    record = {
        "status": "complete",
        "commands_run": len(commands),
        "production_sources_modified": False,
    }
    (LOG_DIR / "POSTPROCESS_COMPLETE").write_text(
        json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
