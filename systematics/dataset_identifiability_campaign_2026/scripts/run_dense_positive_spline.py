#!/usr/bin/env python3
"""Run an isolated dense positive-A spline preflight or full fit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
SYSTEMATICS = BASE.parent
RUNNER = (
    SYSTEMATICS / "high_qt_direct_production_benchmark/experimental_unitary_transition/"
    "scripts/run_production_fnp_stability_control.py")
PYTHON = Path("/home/dustin/miniforge3/envs/pdf-fit/bin/python")
W_GRID = (
    SYSTEMATICS.parent / "outputs/v23a_tevatron_plus_lhcb7_fidacc_merged_cache/"
    "backend_cache/wpert_v23a_tevatron_plus_lhcb7_fidacc_b160.csv")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, required=True)
    parser.add_argument("--nb", type=int, required=True)
    parser.add_argument("--mode", choices=("preflight", "full"), required=True)
    parser.add_argument(
        "--architecture", choices=("positive_a", "monotone_logf"),
        default="monotone_logf")
    parser.add_argument("--seed", type=int, default=1101)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    config = json.loads((BASE / "config/dense_positive_spline_ladder.json").read_text())
    allowed = {(row["nx"], row["nb"]) for row in config["preflight_grid"]}
    if (args.nx, args.nb) not in allowed:
        raise ValueError("resolution is not preregistered")
    table = pd.read_csv(BASE / "manifests/dataset_candidates.csv")
    row = table[table.candidate_id.eq(config["candidate_id"])]
    if len(row) != 1:
        raise RuntimeError("dataset candidate is unresolved")
    source = Path(row.iloc[0].central_output)
    tag = (
        f"dense_{args.architecture}_spline_D020_E772_"
        f"nx{args.nx}_nb{args.nb}_{args.mode}_s{args.seed}")
    target = BASE / "outputs" / tag
    if (target / "fit_status.json").exists():
        print(json.dumps({"status": "already_complete", "tag": tag}))
        return
    lock = BASE / "outputs/.locks" / f"{tag}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock.mkdir()
    except FileExistsError:
        print(json.dumps({"status": "already_running", "tag": tag}))
        return
    full = args.mode == "full"
    command = [
        str(PYTHON), str(RUNNER), "--seed", str(args.seed),
        "--source-production", str(source), "--w-grid", str(W_GRID),
        "--output-root", str(BASE / "outputs"), "--tag", tag,
        ("--global-spline" if args.architecture == "positive_a"
         else "--global-monotone-logf-spline"),
        "--global-spline-nx", str(args.nx),
        "--global-spline-nb", str(args.nb), "--float64",
        "--max-epochs", str(config["full_max_epochs"] if full else 0),
        "--min-epochs", str(config["full_min_epochs"] if full else 0),
        "--plateau-patience", str(config["full_plateau_patience"] if full else 0),
        "--learning-rate", "1e-4",
        "--lbfgs-max-iter", str(config["full_lbfgs_max_iter"] if full else 0),
    ]
    log = BASE / "logs" / f"{tag}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log.open("w") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise subprocess.CalledProcessError(result.returncode, command)
    finally:
        lock.rmdir()
    print(json.dumps({"status": "complete", "tag": tag}))


if __name__ == "__main__":
    main()
