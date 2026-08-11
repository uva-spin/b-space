#!/usr/bin/env python3
"""Continue a completed log-curvature fit under the identical objective."""

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
    parser.add_argument("--lambda-logcurv", required=True, type=float)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--max-epochs", type=int, default=10000)
    parser.add_argument("--min-epochs", type=int, default=3000)
    parser.add_argument("--plateau-patience", type=int, default=2000)
    parser.add_argument("--lbfgs-max-iter", type=int, default=5000)
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
    source_production = Path(selected.iloc[0]["central_output"])

    stem = (
        f"logcurv_{args.candidate}_lam{lambda_token(args.lambda_logcurv)}_"
        f"s{args.seed}"
    )
    initial = BASE / "outputs" / stem
    initial_status_path = initial / "fit_status.json"
    if not initial_status_path.exists():
        raise FileNotFoundError(initial_status_path)
    initial_status = json.loads(initial_status_path.read_text())
    actual_lambda = float(
        initial_status["regularization"]["logf_curvature"]["lambda"])
    if not abs(actual_lambda - args.lambda_logcurv) <= 1.0e-12 * max(
            abs(args.lambda_logcurv), 1.0):
        raise ValueError("source fit uses a different regularization strength")

    tag = f"{stem}_continue"
    target = BASE / "outputs" / tag
    if (target / "fit_status.json").exists() and not args.force:
        print(json.dumps({"status": "already_complete", "tag": tag}))
        return
    command = [
        str(PYTHON), str(RUNNER),
        "--seed", str(args.seed),
        "--source-production", str(source_production),
        "--w-grid", str(W_GRID),
        "--output-root", str(BASE / "outputs"),
        "--tag", tag,
        "--initial-state", str(initial / "model_state.pt"),
        "--initial-norms", str(initial / "dataset_norms.csv"),
        "--max-epochs", str(args.max_epochs),
        "--min-epochs", str(args.min_epochs),
        "--plateau-patience", str(args.plateau_patience),
        "--learning-rate", "3e-6",
        "--lbfgs-max-iter", str(args.lbfgs_max_iter),
        "--float64",
        "--lambda-fnp-logcurv", str(args.lambda_logcurv),
        "--fnp-logcurv-bmin", "0.10",
        "--fnp-logcurv-bmax", "3.0",
    ]
    log = BASE / "logs" / f"{tag}.log"
    with log.open("w") as stream:
        result = subprocess.run(
            command, stdout=stream, stderr=subprocess.STDOUT, text=True,
            check=False)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)
    print(json.dumps({
        "status": "complete",
        "tag": tag,
        "source_fit": stem,
        "objective_unchanged": True,
    }))


if __name__ == "__main__":
    main()
