#!/usr/bin/env python3
"""Fit the selected F-length objective from an independent old-basin state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
SYSTEMATICS = BASE.parent
OLD = SYSTEMATICS / "high_qt_direct_production_benchmark/experimental_unitary_transition"
RUNNER = OLD / "scripts/run_production_fnp_stability_control.py"
PYTHON = Path("/home/dustin/miniforge3/envs/pdf-fit/bin/python")
W_GRID = (
    SYSTEMATICS.parent / "outputs/v23a_tevatron_plus_lhcb7_fidacc_merged_cache/"
    "backend_cache/wpert_v23a_tevatron_plus_lhcb7_fidacc_b160.csv")
SEEDS = tuple(range(303, 327))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    args = parser.parse_args()

    registry = pd.read_csv(BASE / "manifests/dataset_candidates.csv")
    row = registry[registry.candidate_id.eq("D020_E772")]
    if len(row) != 1:
        raise RuntimeError("D020_E772 is unresolved")
    production = Path(row.iloc[0].central_output)
    initial = OLD / "outputs" / f"fig6_lbfgs_stationary_s{args.seed}"
    norms = BASE / "outputs/D020_E772_s701_polish64/dataset_norms.csv"
    tag = f"flength_independent_D020_E772_lam3em01_init{args.seed}"
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
    command = [
        str(PYTHON), str(RUNNER), "--seed", str(args.seed),
        "--source-production", str(production), "--w-grid", str(W_GRID),
        "--output-root", str(BASE / "outputs"), "--tag", tag,
        "--initial-state", str(initial / "model_state.pt"),
        "--initial-norms", str(norms),
        "--max-epochs", "20000", "--min-epochs", "5000",
        "--plateau-patience", "3000", "--learning-rate", "1e-5",
        "--lbfgs-max-iter", "30000", "--float64",
        "--lambda-fnp-loglength", "0.3", "--fnp-length-space", "F",
        "--fnp-loglength-bmin", "0.10", "--fnp-loglength-bmax", "3.0",
    ]
    log = BASE / "logs" / f"{tag}.log"
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
