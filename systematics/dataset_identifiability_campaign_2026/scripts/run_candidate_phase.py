#!/usr/bin/env python3
"""Run one resumable central-start phase for a registered dataset candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
SYSTEMATICS = BASE.parent
RUNNER = (
    SYSTEMATICS
    / "high_qt_direct_production_benchmark/experimental_unitary_transition/"
    "scripts/run_production_fnp_stability_control.py"
)
PYTHON = Path("/home/dustin/miniforge3/envs/pdf-fit/bin/python")
W_GRID = (
    SYSTEMATICS.parent
    / "outputs/v23a_tevatron_plus_lhcb7_fidacc_merged_cache/backend_cache/"
    "wpert_v23a_tevatron_plus_lhcb7_fidacc_b160.csv"
)
PHASES = {
    "explore": {
        "max_epochs": 20000,
        "min_epochs": 5000,
        "plateau_patience": 3000,
        "learning_rate": "2e-5",
        "initial_perturbation": "0.02",
    },
    "settle": {
        "max_epochs": 40000,
        "min_epochs": 10000,
        "plateau_patience": 5000,
        "learning_rate": "2e-6",
    },
    "polish64": {
        "max_epochs": 0,
        "min_epochs": 0,
        "plateau_patience": 0,
        "learning_rate": "2e-6",
        "lbfgs_max_iter": 20000,
        "float64": True,
    },
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--phase", required=True, choices=tuple(PHASES))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def candidate_row(candidate_id: str) -> pd.Series:
    table = pd.read_csv(BASE / "manifests/dataset_candidates.csv")
    selected = table[table["candidate_id"].eq(candidate_id)]
    if len(selected) != 1:
        raise ValueError(f"unknown or duplicate candidate {candidate_id}")
    row = selected.iloc[0]
    if not bool(row["central_output_exists"]):
        raise ValueError(
            f"{candidate_id} is a constructed theory-nuisance candidate and "
            "requires its dedicated runner")
    return row


def main() -> None:
    args = arguments()
    row = candidate_row(args.candidate)
    phase = PHASES[args.phase]
    tag = f"{args.candidate}_s{args.seed}_{args.phase}"
    target = BASE / "outputs" / tag
    status = target / "fit_status.json"
    if status.exists() and not args.force:
        print(json.dumps({"status": "already_complete", "tag": tag}))
        return

    source = Path(row["central_output"])
    if args.phase == "explore":
        initial_state = source / "model_state.pt"
        initial_norms = source / "dataset_norms.csv"
    else:
        prior = "explore" if args.phase == "settle" else "settle"
        prior_target = BASE / "outputs" / f"{args.candidate}_s{args.seed}_{prior}"
        initial_state = prior_target / "model_state.pt"
        initial_norms = prior_target / "dataset_norms.csv"
    for path in (initial_state, initial_norms):
        if not path.exists():
            raise FileNotFoundError(path)

    command = [
        str(PYTHON), str(RUNNER),
        "--seed", str(args.seed),
        "--source-production", str(source),
        "--w-grid", str(W_GRID),
        "--output-root", str(BASE / "outputs"),
        "--tag", tag,
        "--initial-state", str(initial_state),
        "--initial-norms", str(initial_norms),
        "--max-epochs", str(phase["max_epochs"]),
        "--min-epochs", str(phase["min_epochs"]),
        "--plateau-patience", str(phase["plateau_patience"]),
        "--learning-rate", phase["learning_rate"],
    ]
    if args.phase == "explore":
        command.extend([
            "--initial-perturbation", phase["initial_perturbation"],
            "--allow-initial-state-perturbation",
        ])
    if phase.get("lbfgs_max_iter"):
        command.extend(["--lbfgs-max-iter", str(phase["lbfgs_max_iter"])])
    if phase.get("float64"):
        command.append("--float64")

    log = BASE / "logs" / f"{tag}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        result = subprocess.run(
            command, stdout=stream, stderr=subprocess.STDOUT, text=True,
            check=False,
        )
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)
    if not status.exists():
        raise RuntimeError(f"runner completed without {status}")
    print(json.dumps({
        "status": "complete",
        "candidate": args.candidate,
        "seed": args.seed,
        "phase": args.phase,
        "tag": tag,
        "log": str(log),
    }))


if __name__ == "__main__":
    main()
