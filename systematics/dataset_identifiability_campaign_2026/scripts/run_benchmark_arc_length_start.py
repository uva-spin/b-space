#!/usr/bin/env python3
"""Run one isolated arc-length fit from an independent old production basin."""

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
    "backend_cache/wpert_v23a_tevatron_plus_lhcb7_fidacc_b160.csv"
)


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda-loglength", type=float, required=True)
    parser.add_argument("--initialization-seed", type=int, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def token(value: float) -> str:
    return f"{value:.0e}".replace("-", "m").replace("+", "")


def main() -> None:
    options = args()
    config = json.loads(
        (BASE / "config/benchmark_arc_length_multistart.json").read_text())
    if options.lambda_loglength not in config["pilot_strengths"]:
        raise ValueError("strength was not preregistered")
    if options.initialization_seed not in config["expansion_initialization_seeds"]:
        raise ValueError("initialization seed was not preregistered")

    manifest = pd.read_csv(BASE / "manifests/dataset_candidates.csv")
    row = manifest[manifest.candidate_id.eq(config["candidate_id"])]
    if len(row) != 1:
        raise RuntimeError("selected dataset candidate is unresolved")
    source = Path(row.iloc[0].central_output)

    initial_model = (
        OLD / "outputs" / f"fig6_lbfgs_stationary_s{options.initialization_seed}"
        / "model_state.pt")
    if not initial_model.exists():
        raise FileNotFoundError(initial_model)
    norm_run = BASE / "outputs/D020_E772_s701_polish64"
    tag = (
        f"benchmark_loglength_D020_E772_lam{token(options.lambda_loglength)}"
        f"_init{options.initialization_seed}")
    target = BASE / "outputs" / tag
    if (target / "fit_status.json").exists() and not options.force:
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
        str(PYTHON), str(RUNNER),
        "--seed", str(options.initialization_seed),
        "--source-production", str(source),
        "--w-grid", str(W_GRID),
        "--output-root", str(BASE / "outputs"),
        "--tag", tag,
        "--initial-state", str(initial_model),
        "--initial-norms", str(norm_run / "dataset_norms.csv"),
        "--max-epochs", str(config["max_epochs"]),
        "--min-epochs", str(config["min_epochs"]),
        "--plateau-patience", str(config["plateau_patience"]),
        "--learning-rate", "1e-5",
        "--lbfgs-max-iter", str(config["lbfgs_max_iter"]),
        "--float64",
        "--lambda-fnp-loglength", str(options.lambda_loglength),
        "--fnp-loglength-bmin", str(config["b_min_GeV_inverse"]),
        "--fnp-loglength-bmax", str(config["b_max_GeV_inverse"]),
    ]
    log = BASE / "logs" / f"{tag}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log.open("w") as stream:
            result = subprocess.run(
                command, stdout=stream, stderr=subprocess.STDOUT,
                text=True, check=False)
        if result.returncode:
            raise subprocess.CalledProcessError(result.returncode, command)
    finally:
        lock.rmdir()
    print(json.dumps({
        "status": "complete", "tag": tag,
        "lambda_loglength": options.lambda_loglength,
        "initialization_seed": options.initialization_seed,
    }))


if __name__ == "__main__":
    main()
