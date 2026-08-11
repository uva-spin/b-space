#!/usr/bin/env python3
"""Fit one experimental replica under an explicitly selected log-curvature model."""

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
    parser.add_argument("--source-fit", type=Path, required=True)
    parser.add_argument("--replica-seed", type=int, required=True)
    parser.add_argument("--fit-seed", type=int, required=True)
    parser.add_argument("--initial-perturbation", type=float, default=0.01)
    parser.add_argument("--max-epochs", type=int, default=20000)
    parser.add_argument("--min-epochs", type=int, default=5000)
    parser.add_argument("--plateau-patience", type=int, default=3000)
    parser.add_argument("--lbfgs-max-iter", type=int, default=10000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    source_fit = args.source_fit.resolve()
    source_status = json.loads((source_fit / "fit_status.json").read_text())
    regularizer = source_status["regularization"]["logf_curvature"]
    lam = float(regularizer["lambda"])
    if lam <= 0.0:
        raise ValueError("source fit is not a positive log-curvature candidate")

    registry = pd.read_csv(BASE / "manifests/dataset_candidates.csv")
    selected = registry[registry["candidate_id"].eq(args.candidate)]
    if len(selected) != 1 or not bool(selected.iloc[0]["central_output_exists"]):
        raise ValueError(args.candidate)
    source_production = Path(selected.iloc[0]["central_output"]).resolve()
    if source_production != Path(source_status["source_production"]).resolve():
        raise ValueError("candidate and selected source fit use different datasets")

    lambda_token = f"{lam:.0e}".replace("-", "m").replace("+", "")
    tag = (
        f"selectedrep_{args.candidate}_lam{lambda_token}"
        f"_r{args.replica_seed}_s{args.fit_seed}")
    target = BASE / "outputs" / tag
    if (target / "fit_status.json").exists() and not args.force:
        print(json.dumps({"status": "already_complete", "tag": tag}))
        return

    command = [
        str(PYTHON), str(RUNNER),
        "--seed", str(args.fit_seed),
        "--replica-seed", str(args.replica_seed),
        "--source-production", str(source_production),
        "--w-grid", str(W_GRID),
        "--output-root", str(BASE / "outputs"),
        "--tag", tag,
        "--initial-state", str(source_fit / "model_state.pt"),
        "--initial-norms", str(source_fit / "dataset_norms.csv"),
        "--initial-perturbation", str(args.initial_perturbation),
        "--allow-initial-state-perturbation",
        "--max-epochs", str(args.max_epochs),
        "--min-epochs", str(args.min_epochs),
        "--plateau-patience", str(args.plateau_patience),
        "--learning-rate", "1e-5",
        "--lbfgs-max-iter", str(args.lbfgs_max_iter),
        "--float64",
        "--lambda-fnp-logcurv", str(lam),
        "--fnp-logcurv-bmin", str(regularizer["b_min"]),
        "--fnp-logcurv-bmax", str(regularizer["b_max"]),
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
        "candidate_id": args.candidate,
        "source_fit": str(source_fit),
        "replica_seed": args.replica_seed,
        "fit_seed": args.fit_seed,
        "lambda_logcurv": lam,
    }))


if __name__ == "__main__":
    main()
