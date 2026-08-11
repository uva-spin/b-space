#!/usr/bin/env python3
"""Run one isolated smoothness-plus-transform-closure pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


BASE = Path(__file__).resolve().parents[1]
SYSTEMATICS = BASE.parent
RUNNER = (
    SYSTEMATICS
    / "high_qt_direct_production_benchmark/experimental_unitary_transition"
    / "scripts/run_production_fnp_stability_control.py"
)
PYTHON = Path("/home/dustin/miniforge3/envs/pdf-fit/bin/python")
SOURCE = (
    SYSTEMATICS
    / "collins_factorization_validity/outputs"
    / "rowidfix_stageFT_E772_qmax0p20_lam0p50_central_s303"
)
W_GRID = (
    SYSTEMATICS.parent
    / "outputs/v23a_tevatron_plus_lhcb7_fidacc_merged_cache/backend_cache"
    / "wpert_v23a_tevatron_plus_lhcb7_fidacc_b160.csv"
)


def token(value: float) -> str:
    return f"{value:.0e}".replace("e-0", "em").replace("e-", "em")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda-closure", type=float, required=True)
    parser.add_argument("--closure-max", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=901)
    parser.add_argument("--initial-state", type=Path)
    parser.add_argument("--initial-norms", type=Path)
    args = parser.parse_args()

    config = json.loads(
        (BASE / "config/transform_closure_ladder.json").read_text())
    allowed_lambdas = config["lambda_ladder"]
    allowed_thresholds = config["threshold_robustness_values"]
    if args.lambda_closure not in allowed_lambdas:
        raise ValueError("closure strength is outside the preregistered ladder")
    if args.closure_max not in allowed_thresholds:
        raise ValueError("closure threshold is outside the preregistered ladder")

    base = config["base_regularization"]
    tag = (
        f"closure_D020_E772_lam{token(args.lambda_closure)}_"
        f"max{token(args.closure_max)}_s{args.seed}")
    command = [
        str(PYTHON), str(RUNNER),
        "--seed", str(args.seed),
        "--source-production", str(SOURCE),
        "--w-grid", str(W_GRID),
        "--output-root", str(BASE / "outputs"),
        "--tag", tag,
        "--lambda-fnp-loglength", str(base["lambda_logf_arc_length"]),
        "--fnp-loglength-bmin", str(base["b_min_GeV_inverse"]),
        "--fnp-loglength-bmax", str(base["b_max_GeV_inverse"]),
        "--lambda-fnp-ratecurv", str(base["lambda_damping_rate_curvature"]),
        "--fnp-ratecurv-bmin", str(base["ratecurv_b_min_GeV_inverse"]),
        "--fnp-ratecurv-bmax", str(base["ratecurv_b_max_GeV_inverse"]),
        "--lambda-fnp-transform-closure", str(args.lambda_closure),
        "--fnp-transform-closure-bmin",
        str(config["closure_definition"]["b_min_GeV_inverse"]),
        "--fnp-transform-closure-max", str(args.closure_max),
        "--max-epochs", "20000",
        "--min-epochs", "5000",
        "--plateau-patience", "2500",
        "--learning-rate", "0.0001",
        "--lbfgs-max-iter", "10000",
        "--float64",
    ]
    if args.initial_state is not None:
        if args.initial_norms is None:
            raise ValueError("--initial-state requires --initial-norms")
        command.extend([
            "--initial-state", str(args.initial_state.resolve()),
            "--initial-norms", str(args.initial_norms.resolve()),
        ])
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
