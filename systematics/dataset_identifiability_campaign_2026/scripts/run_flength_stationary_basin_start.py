#!/usr/bin/env python3
"""Sample a perturbed start around the stationary F-length basin."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    config = json.loads(
        (BASE / "config/flength_stationary_basin_sampling.json").read_text())
    rows = [row for row in config["pilot_perturbations"] if row["seed"] == args.seed]
    if len(rows) != 1:
        raise ValueError("seed was not preregistered")
    perturbation = float(rows[0]["relative_scale"])
    initial = BASE / "outputs" / config["stationary_source_tag"]
    registry = pd.read_csv(BASE / "manifests/dataset_candidates.csv")
    row = registry[registry.candidate_id.eq(config["candidate_id"])]
    production = Path(row.iloc[0].central_output)
    ptoken = str(perturbation).replace(".", "p")
    tag = f"flength_stationary_lam3em01_p{ptoken}_s{args.seed}"
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
        "--initial-norms", str(initial / "dataset_norms.csv"),
        "--initial-perturbation", str(perturbation),
        "--allow-initial-state-perturbation",
        "--max-epochs", str(config["max_epochs"]),
        "--min-epochs", str(config["min_epochs"]),
        "--plateau-patience", str(config["plateau_patience"]),
        "--learning-rate", "3e-6",
        "--lbfgs-max-iter", str(config["lbfgs_max_iter"]),
        "--float64", "--lambda-fnp-loglength", str(config["lambda_F_length"]),
        "--fnp-length-space", "F",
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
