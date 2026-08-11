#!/usr/bin/env python3
"""Run one isolated explicit remote-tail closure-coordinate pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


BASE = Path(__file__).resolve().parents[1]
SYSTEMATICS = BASE.parent
RUNNER = (
    SYSTEMATICS
    / "high_qt_direct_production_benchmark/experimental_unitary_transition"
    / "scripts/run_production_fnp_stability_control.py"
)
SOURCE = (
    SYSTEMATICS / "collins_factorization_validity/outputs"
    / "rowidfix_stageFT_E772_qmax0p20_lam0p50_central_s303"
)
W_GRID = (
    SYSTEMATICS.parent
    / "outputs/v23a_tevatron_plus_lhcb7_fidacc_merged_cache/backend_cache"
    / "wpert_v23a_tevatron_plus_lhcb7_fidacc_b160.csv"
)
INITIAL = (
    BASE / "outputs/closure_D020_E772_lam1em3_max1em4_s901")


def number_token(value: float) -> str:
    if value >= 1.0:
        return f"{value:.1f}".replace(".", "p")
    return f"{value:.0e}".replace("e-0", "em").replace("e-", "em")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b-start", type=float, required=True)
    parser.add_argument("--b-end", type=float, default=8.0)
    parser.add_argument("--lambda-closure", type=float, default=1.0e-3)
    parser.add_argument("--closure-max", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=902)
    parser.add_argument("--tag-suffix", default="")
    args = parser.parse_args()

    config = json.loads(
        (BASE / "config/transform_closure_ladder.json").read_text())
    windows = config["explicit_coordinate_robustness"][
        "b_windows_GeV_inverse"]
    if [args.b_start, args.b_end] not in windows:
        raise ValueError("tail window is outside the preregistered ladder")
    if args.lambda_closure not in config["lambda_ladder"]:
        raise ValueError("closure strength is outside the preregistered ladder")
    if args.closure_max not in config["threshold_robustness_values"]:
        raise ValueError("closure threshold is outside the preregistered ladder")

    base = config["base_regularization"]
    tag = (
        f"closurecoord_D020_E772_b{number_token(args.b_start)}_"
        f"{number_token(args.b_end)}_lam{number_token(args.lambda_closure)}_"
        f"max{number_token(args.closure_max)}_s{args.seed}"
        f"{args.tag_suffix}")
    command = [
        sys.executable, str(RUNNER),
        "--seed", str(args.seed),
        "--source-production", str(SOURCE),
        "--w-grid", str(W_GRID),
        "--output-root", str(BASE / "outputs"),
        "--tag", tag,
        "--lambda-fnp-loglength", str(base["lambda_logf_arc_length"]),
        "--fnp-loglength-bmin", str(base["b_min_GeV_inverse"]),
        "--fnp-loglength-bmax", str(base["b_max_GeV_inverse"]),
        "--lambda-fnp-ratecurv",
        str(base["lambda_damping_rate_curvature"]),
        "--fnp-ratecurv-bmin",
        str(base["ratecurv_b_min_GeV_inverse"]),
        "--fnp-ratecurv-bmax",
        str(base["ratecurv_b_max_GeV_inverse"]),
        "--closure-tail-coordinate",
        "--closure-tail-b-start", str(args.b_start),
        "--closure-tail-b-end", str(args.b_end),
        "--lambda-fnp-transform-closure", str(args.lambda_closure),
        # The coordinate transitions between b_start and b_end. Numerical
        # closure is an endpoint condition; penalizing the transition itself
        # would contradict the declared C2 window.
        "--fnp-transform-closure-bmin", str(args.b_end),
        "--fnp-transform-closure-max", str(args.closure_max),
        "--initial-state", str(INITIAL / "model_state.pt"),
        "--initial-norms", str(INITIAL / "dataset_norms.csv"),
        "--max-epochs", "20000",
        "--min-epochs", "5000",
        "--plateau-patience", "2500",
        "--learning-rate", "0.0001",
        "--lbfgs-max-iter", "10000",
        "--float64",
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
