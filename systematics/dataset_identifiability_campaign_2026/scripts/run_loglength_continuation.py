#!/usr/bin/env python3
"""Continue a completed log-length fit under the identical objective."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

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


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--lambda-loglength", required=True, type=float)
    parser.add_argument("--length-space", choices=("logF", "F"), default="logF")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--max-epochs", type=int, default=10000)
    parser.add_argument("--min-epochs", type=int, default=3000)
    parser.add_argument("--plateau-patience", type=int, default=2000)
    parser.add_argument("--lbfgs-max-iter", type=int, default=10000)
    parser.add_argument("--learning-rate", type=float, default=3.0e-6)
    parser.add_argument("--source-suffix", default="")
    parser.add_argument("--tag-suffix", default="_continue")
    parser.add_argument("--fit-quality-ceiling-total-chi2", type=float)
    parser.add_argument("--lambda-fit-quality-barrier", type=float, default=0.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def lambda_token(value: float) -> str:
    return f"{value:.0e}".replace("-", "m").replace("+", "")


def main() -> None:
    args = arguments()
    registry = pd.read_csv(BASE / "manifests/dataset_candidates.csv")
    selected = registry[registry["candidate_id"].eq(args.candidate)]
    if len(selected) != 1 or not bool(selected.iloc[0]["central_output_exists"]):
        raise ValueError(args.candidate)
    source = Path(selected.iloc[0]["central_output"])

    prefix = "flength" if args.length_space == "F" else "loglength"
    base_stem = (
        f"{prefix}_{args.candidate}_"
        f"lam{lambda_token(args.lambda_loglength)}_s{args.seed}")
    stem = base_stem + args.source_suffix
    initial = BASE / "outputs" / stem
    status_path = initial / "fit_status.json"
    if not status_path.exists():
        raise FileNotFoundError(status_path)
    status = json.loads(status_path.read_text())
    actual = float(
        status["regularization"]["logf_arc_length"]["lambda"])
    if not abs(actual - args.lambda_loglength) <= (
            1.0e-12 * max(abs(args.lambda_loglength), 1.0)):
        raise ValueError("source fit uses a different regularizer")
    actual_space = status["regularization"]["logf_arc_length"].get(
        "space", "logF")
    if actual_space != args.length_space:
        raise ValueError("source fit uses a different length space")

    tag = base_stem + args.tag_suffix
    target = BASE / "outputs" / tag
    if (target / "fit_status.json").exists() and not args.force:
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
        "--source-production", str(source),
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
        "--lambda-fnp-loglength", str(args.lambda_loglength),
        "--fnp-length-space", args.length_space,
        "--fnp-loglength-bmin", "0.10",
        "--fnp-loglength-bmax", "3.0",
    ]
    if args.lambda_fit_quality_barrier > 0.0:
        if args.fit_quality_ceiling_total_chi2 is None:
            raise ValueError("barrier requires a chi2 ceiling")
        command.extend([
            "--fit-quality-ceiling-total-chi2",
            str(args.fit_quality_ceiling_total_chi2),
            "--lambda-fit-quality-barrier",
            str(args.lambda_fit_quality_barrier),
        ])
    log = BASE / "logs" / f"{tag}.log"
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
        "source_fit": stem,
        "objective_unchanged": True,
    }))


if __name__ == "__main__":
    main()
