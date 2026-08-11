#!/usr/bin/env python3
"""Apply the registered fit-quality barrier to an existing F-length state."""

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
    parser.add_argument("--source-tag", required=True)
    parser.add_argument("--target-tag", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--barrier-lambda", type=float, choices=(10.0, 100.0), required=True)
    args = parser.parse_args()
    initial = BASE / "outputs" / args.source_tag
    status = json.loads((initial / "fit_status.json").read_text())
    regularizer = status["regularization"]["logf_arc_length"]
    if regularizer.get("space") != "F" or abs(regularizer["lambda"] - 0.3) > 1e-12:
        raise ValueError("source is not a lambda_F=0.3 F-length fit")
    registry = pd.read_csv(BASE / "manifests/dataset_candidates.csv")
    production = Path(
        registry[registry.candidate_id.eq("D020_E772")].iloc[0].central_output)
    target = BASE / "outputs" / args.target_tag
    if (target / "fit_status.json").exists():
        print(json.dumps({"status": "already_complete", "tag": args.target_tag}))
        return
    lock = BASE / "outputs/.locks" / f"{args.target_tag}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock.mkdir()
    except FileExistsError:
        print(json.dumps({"status": "already_running", "tag": args.target_tag}))
        return
    command = [
        str(PYTHON), str(RUNNER), "--seed", str(args.seed),
        "--source-production", str(production), "--w-grid", str(W_GRID),
        "--output-root", str(BASE / "outputs"), "--tag", args.target_tag,
        "--initial-state", str(initial / "model_state.pt"),
        "--initial-norms", str(initial / "dataset_norms.csv"),
        "--max-epochs", "20000", "--min-epochs", "5000",
        "--plateau-patience", "3000", "--learning-rate", "1e-6",
        "--lbfgs-max-iter", "30000", "--float64",
        "--lambda-fnp-loglength", "0.3", "--fnp-length-space", "F",
        "--fnp-loglength-bmin", "0.10", "--fnp-loglength-bmax", "3.0",
        "--fit-quality-ceiling-total-chi2", "119.8021",
        "--lambda-fit-quality-barrier", str(args.barrier_lambda),
    ]
    log = BASE / "logs" / f"{args.target_tag}.log"
    try:
        with log.open("w") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise subprocess.CalledProcessError(result.returncode, command)
    finally:
        lock.rmdir()
    print(json.dumps({"status": "complete", "tag": args.target_tag}))


if __name__ == "__main__":
    main()
