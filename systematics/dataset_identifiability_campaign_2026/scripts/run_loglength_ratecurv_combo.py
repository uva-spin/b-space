#!/usr/bin/env python3
"""Run one isolated log-length plus damping-rate-curvature pilot."""

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
    parser.add_argument("--lambda-ratecurv", required=True, type=float)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=15000)
    parser.add_argument("--min-epochs", type=int, default=5000)
    parser.add_argument("--plateau-patience", type=int, default=2500)
    parser.add_argument("--lbfgs-max-iter", type=int, default=10000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def token(value: float) -> str:
    return f"{value:.0e}".replace("-", "m").replace("+", "")


def main() -> None:
    args = arguments()
    config = json.loads((
        BASE / "config/loglength_ratecurv_combo_ladder.json").read_text())
    if args.lambda_ratecurv not in config["lambda_ratecurv_ladder"]:
        raise ValueError("rate-curvature strength is outside declared ladder")
    registry = pd.read_csv(BASE / "manifests/dataset_candidates.csv")
    selected = registry[registry["candidate_id"].eq(args.candidate)]
    if len(selected) != 1 or not bool(selected.iloc[0]["central_output_exists"]):
        raise ValueError(args.candidate)
    source = Path(selected.iloc[0]["central_output"])

    choices = []
    for seed in (701, 702, 703):
        run = BASE / "outputs" / f"{args.candidate}_s{seed}_polish64"
        status = json.loads((run / "fit_status.json").read_text())
        final = status["final"]
        choices.append((
            float(final["data_chi2"] + final["norm_penalty"]),
            seed, run))
    _, best_seed, best_run = min(choices)
    seed = args.seed if args.seed is not None else best_seed
    initial = (
        BASE / "outputs" / f"{args.candidate}_s{seed}_polish64"
        if args.seed is not None else best_run)
    lambda_length = float(config["fixed_lambda_loglength"])
    tag = (
        f"lengthrate_{args.candidate}_llen{token(lambda_length)}_"
        f"lrate{token(args.lambda_ratecurv)}_s{seed}")
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
        "--lambda-fnp-loglength", str(lambda_length),
        "--fnp-loglength-bmin", str(config["b_min_GeV_inverse"]),
        "--fnp-loglength-bmax", str(config["b_max_GeV_inverse"]),
        "--lambda-fnp-ratecurv", str(args.lambda_ratecurv),
        "--fnp-ratecurv-bmin", str(config["b_min_GeV_inverse"]),
        "--fnp-ratecurv-bmax", str(config["b_max_GeV_inverse"]),
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
        "candidate_id": args.candidate,
        "seed": seed,
        "lambda_loglength": lambda_length,
        "lambda_ratecurv": args.lambda_ratecurv,
    }))


if __name__ == "__main__":
    main()
