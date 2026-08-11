#!/usr/bin/env python3
"""Continue one frozen-baseline endpoint with the true localized F-length."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


BASE = Path(__file__).resolve().parents[1]
SYSTEMATICS = BASE.parent
ROOT = SYSTEMATICS.parent
UNITARY = (
    SYSTEMATICS
    / "high_qt_direct_production_benchmark/experimental_unitary_transition")
RUNNER = UNITARY / "scripts/run_production_fnp_stability_control.py"
SOURCE = (
    SYSTEMATICS / "collins_factorization_validity/outputs/"
    "rowidfix_stageFT_all_qmax0p20_lam0p50_central_s303")
W_GRID = (
    ROOT / "outputs/v23a_tevatron_plus_lhcb7_fidacc_merged_cache/"
    "backend_cache/wpert_v23a_tevatron_plus_lhcb7_fidacc_b160.csv")
PYTHON = Path("/home/dustin/miniforge3/envs/pdf-fit/bin/python")


def token(value: float) -> str:
    return f"{value:.0e}".replace("-", "m").replace("+", "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--strength", type=float, required=True)
    parser.add_argument("--max-epochs", type=int, default=15000)
    parser.add_argument("--min-epochs", type=int, default=5000)
    parser.add_argument("--plateau-patience", type=int, default=2000)
    parser.add_argument("--lbfgs-max-iter", type=int, default=15000)
    parser.add_argument("--learning-rate", type=float, default=3.0e-6)
    args = parser.parse_args()
    if args.seed not in range(303, 327):
        raise ValueError("seed is not one of the 24 frozen baseline endpoints")
    if args.strength < 0.0:
        raise ValueError("strength cannot be negative")

    initial = UNITARY / "outputs" / f"fig6_lbfgs_stationary_s{args.seed}"
    if not (initial / "fit_status.json").exists():
        raise FileNotFoundError(initial)
    tag = (
        f"exactbaseline_zero_control_s{args.seed}"
        if args.strength == 0.0 else
        f"exactbaseline_trueFlength_b0p1_2p0_"
        f"lam{token(args.strength)}_s{args.seed}")
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
        str(PYTHON), str(RUNNER),
        "--seed", str(args.seed),
        "--source-production", str(SOURCE),
        "--w-grid", str(W_GRID),
        "--output-root", str(BASE / "outputs"),
        "--tag", tag,
        "--initial-state", str(initial / "model_state.pt"),
        "--initial-norms", str(initial / "dataset_norms.csv"),
        "--max-epochs", str(args.max_epochs),
        "--min-epochs", str(args.min_epochs),
        "--plateau-patience", str(args.plateau_patience),
        "--learning-rate", str(args.learning_rate),
        "--lbfgs-max-iter", str(args.lbfgs_max_iter),
        "--float64",
        "--lambda-fnp-loglength", str(args.strength),
        "--fnp-length-space", "F",
        "--fnp-loglength-bmin", "0.10",
        "--fnp-loglength-bmax", "2.0",
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
        "status": "complete",
        "tag": tag,
        "source": str(initial),
        "frozen_baseline_modified": False,
    }))


if __name__ == "__main__":
    main()
