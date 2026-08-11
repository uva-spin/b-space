#!/usr/bin/env python3
"""Run one isolated pointwise log-FNP profile refit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
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
CENTRAL_SEEDS = (701, 702, 703)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--x", type=float, default=0.1)
    parser.add_argument("--bT", type=float, required=True)
    parser.add_argument("--ratio", type=float, required=True)
    parser.add_argument("--profile-lambda", type=float, default=1.0e5)
    parser.add_argument("--lbfgs-max-iter", type=int, default=5000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def token(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def candidate_row(candidate: str) -> pd.Series:
    table = pd.read_csv(BASE / "manifests/dataset_candidates.csv")
    rows = table[table["candidate_id"].eq(candidate)]
    if len(rows) != 1 or not bool(rows.iloc[0]["central_output_exists"]):
        raise ValueError(f"ordinary candidate not found: {candidate}")
    return rows.iloc[0]


def reference(candidate: str, x: float, bT: float) -> tuple[float, int, float]:
    values = []
    objectives = []
    for seed in CENTRAL_SEEDS:
        run = BASE / "outputs" / f"{candidate}_s{seed}_polish64"
        grid = pd.read_csv(run / "fnp_grid.csv")
        curve = grid[np.isclose(grid["x"], x)].sort_values("bT")
        if len(curve) < 2:
            raise ValueError(f"x={x} is absent from {run / 'fnp_grid.csv'}")
        values.append(float(np.interp(bT, curve["bT"], curve["F_NP"])))
        status = json.loads((run / "fit_status.json").read_text())
        objectives.append((
            float(status["final"]["data_chi2"] + status["final"]["norm_penalty"]),
            seed,
        ))
    best_objective, best_seed = min(objectives)
    return float(np.median(values)), best_seed, best_objective


def main() -> None:
    args = arguments()
    if args.ratio <= 0.0:
        raise ValueError("--ratio must be positive")
    row = candidate_row(args.candidate)
    reference_fnp, seed, baseline_objective = reference(
        args.candidate, args.x, args.bT)
    target_fnp = args.ratio * reference_fnp
    tag = (
        f"profile_{args.candidate}_x{token(args.x)}_b{token(args.bT)}_"
        f"r{token(args.ratio)}"
    )
    target = BASE / "outputs" / tag
    status_path = target / "fit_status.json"
    if status_path.exists() and not args.force:
        print(json.dumps({"status": "already_complete", "tag": tag}))
        return

    initial = BASE / "outputs" / f"{args.candidate}_s{seed}_polish64"
    command = [
        str(PYTHON), str(RUNNER),
        "--seed", str(seed),
        "--source-production", str(Path(row["central_output"])),
        "--w-grid", str(W_GRID),
        "--output-root", str(BASE / "outputs"),
        "--tag", tag,
        "--initial-state", str(initial / "model_state.pt"),
        "--initial-norms", str(initial / "dataset_norms.csv"),
        "--max-epochs", "0",
        "--min-epochs", "0",
        "--plateau-patience", "0",
        "--lbfgs-max-iter", str(args.lbfgs_max_iter),
        "--float64",
        "--profile-x", str(args.x),
        "--profile-b", str(args.bT),
        f"--profile-logf-target={np.log(target_fnp)}",
        "--profile-lambda", str(args.profile_lambda),
        "--profile-local-shift",
    ]
    log = BASE / "logs" / f"{tag}.log"
    with log.open("w") as stream:
        result = subprocess.run(
            command, stdout=stream, stderr=subprocess.STDOUT, text=True,
            check=False)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)
    status = json.loads(status_path.read_text())
    status["point_profile"].update({
        "reference_definition": "pointwise median of seeds 701--703 after polish64",
        "reference_fnp": reference_fnp,
        "target_ratio": args.ratio,
        "initial_seed": seed,
        "best_polished_unpenalized_chi2": baseline_objective,
        "delta_unpenalized_chi2_from_best_polished": (
            status["final"]["unpenalized_total_chi2"] - baseline_objective),
    })
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps({
        "status": "complete", "tag": tag,
        "achieved_ratio": status["point_profile"]["achieved_fnp"] / reference_fnp,
        "delta_chi2": status["point_profile"]["delta_unpenalized_chi2_from_best_polished"],
        "closures": status["lbfgs"]["closure_evaluations"],
    }))


if __name__ == "__main__":
    main()
