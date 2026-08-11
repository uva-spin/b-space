#!/usr/bin/env python3
"""Run one isolated dimensionless log-FNP arc-length pilot."""

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
    parser.add_argument("--lambda-loglength", type=float, required=True)
    parser.add_argument("--length-space", choices=("logF", "F"), default="logF")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=10000)
    parser.add_argument("--min-epochs", type=int, default=3000)
    parser.add_argument("--plateau-patience", type=int, default=2000)
    parser.add_argument("--lbfgs-max-iter", type=int, default=5000)
    parser.add_argument("--tag-suffix", default="")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def lambda_token(value: float) -> str:
    return f"{value:.0e}".replace("-", "m").replace("+", "")


def main() -> None:
    args = arguments()
    table = pd.read_csv(BASE / "manifests/dataset_candidates.csv")
    rows = table[table["candidate_id"].eq(args.candidate)]
    if len(rows) != 1 or not bool(rows.iloc[0]["central_output_exists"]):
        raise ValueError(args.candidate)
    source = Path(rows.iloc[0]["central_output"])

    choices = []
    for seed in (701, 702, 703):
        run = BASE / "outputs" / f"{args.candidate}_s{seed}_polish64"
        status = json.loads((run / "fit_status.json").read_text())
        choices.append((
            status["final"]["data_chi2"] + status["final"]["norm_penalty"],
            seed, run))
    _, best_seed, best_run = min(choices)
    seed = args.seed if args.seed is not None else best_seed
    initial = (
        BASE / "outputs" / f"{args.candidate}_s{seed}_polish64"
        if args.seed is not None else best_run)
    prefix = "flength" if args.length_space == "F" else "loglength"
    tag = (
        f"{prefix}_{args.candidate}_lam"
        f"{lambda_token(args.lambda_loglength)}_s{seed}{args.tag_suffix}"
    )
    target = BASE / "outputs" / tag
    if (target / "fit_status.json").exists() and not args.force:
        print(json.dumps({"status": "already_complete", "tag": tag}))
        return
    lock = BASE / "outputs" / ".locks" / f"{tag}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock.mkdir()
    except FileExistsError:
        print(json.dumps({"status": "already_running", "tag": tag}))
        return

    command = [
        str(PYTHON), str(RUNNER),
        "--seed", str(seed),
        "--source-production", str(source),
        "--w-grid", str(W_GRID),
        "--output-root", str(BASE / "outputs"),
        "--tag", tag,
        "--initial-state", str(initial / "model_state.pt"),
        "--initial-norms", str(initial / "dataset_norms.csv"),
        "--max-epochs", str(args.max_epochs),
        "--min-epochs", str(args.min_epochs),
        "--plateau-patience", str(args.plateau_patience),
        "--learning-rate", "1e-5",
        "--lbfgs-max-iter", str(args.lbfgs_max_iter),
        "--float64",
        "--lambda-fnp-loglength", str(args.lambda_loglength),
        "--fnp-length-space", args.length_space,
        "--fnp-loglength-bmin", "0.10",
        "--fnp-loglength-bmax", "3.0",
    ]
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
        "seed": seed,
        "lambda_loglength": args.lambda_loglength,
        "length_space": args.length_space,
    }))


if __name__ == "__main__":
    main()
