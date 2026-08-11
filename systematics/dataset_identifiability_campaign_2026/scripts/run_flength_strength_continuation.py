#!/usr/bin/env python3
"""Continue a fitted F-length solution at a stronger registered weight."""

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


def token(value: float) -> str:
    return f"{value:.0e}".replace("-", "m").replace("+", "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source-lambda", type=float, required=True)
    parser.add_argument("--target-lambda", type=float, required=True)
    args = parser.parse_args()
    if not (args.target_lambda > args.source_lambda > 0):
        raise ValueError("this launcher only strengthens a positive F-length prior")
    if args.target_lambda not in (0.5, 0.7):
        raise ValueError("target strength was not registered for the boundary bracket")
    source_tag = (
        f"flength_D020_E772_lam{token(args.source_lambda)}_s{args.seed}")
    initial = BASE / "outputs" / source_tag
    source_status = json.loads((initial / "fit_status.json").read_text())
    if source_status["regularization"]["logf_arc_length"].get("space") != "F":
        raise ValueError("source is not an F-length fit")
    registry = pd.read_csv(BASE / "manifests/dataset_candidates.csv")
    row = registry[registry.candidate_id.eq("D020_E772")]
    production = Path(row.iloc[0].central_output)
    tag = (
        f"flength_D020_E772_lam{token(args.target_lambda)}_s{args.seed}"
        f"_from{token(args.source_lambda)}")
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
        "--max-epochs", "20000", "--min-epochs", "5000",
        "--plateau-patience", "3000", "--learning-rate", "3e-6",
        "--lbfgs-max-iter", "20000", "--float64",
        "--lambda-fnp-loglength", str(args.target_lambda),
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
