#!/usr/bin/env python3
"""Fit one experimental replica under the selected F-slope+closure method."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


BASE = Path(__file__).resolve().parents[1]
SYSTEMATICS = BASE.parent
PYTHON = Path("/home/dustin/miniforge3/envs/pdf-fit/bin/python")
RUNNER = (
    SYSTEMATICS
    / "high_qt_direct_production_benchmark/experimental_unitary_transition/"
    "scripts/run_production_fnp_stability_control.py"
)
SOURCE = (
    SYSTEMATICS
    / "collins_factorization_validity/outputs/"
    "rowidfix_stageFT_E772_qmax0p20_lam0p50_central_s303"
)
W_GRID = (
    SYSTEMATICS.parent
    / "outputs/v23a_tevatron_plus_lhcb7_fidacc_merged_cache/backend_cache/"
    "wpert_v23a_tevatron_plus_lhcb7_fidacc_b160.csv"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-fit", type=Path, required=True)
    parser.add_argument("--replica-seed", type=int, required=True)
    parser.add_argument("--fit-seed", type=int, required=True)
    parser.add_argument("--initial-perturbation", type=float, default=0.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source = args.source_fit.resolve()
    status = json.loads((source / "fit_status.json").read_text())
    regularization = status["regularization"]
    slope = regularization["fnp_slope_energy"]
    closure = regularization["transform_closure"]
    constraint = status["model_constraint"]
    if (
        float(slope["lambda"]) <= 0
        or constraint["kind"] != "c2_remote_tail_closure_coordinate"
    ):
        raise ValueError("source is not the selected F-slope+closure method")

    tag = (
        f"selectedrep_fslope_lam0p01_c2closure_b5p5"
        f"_r{args.replica_seed}_s{args.fit_seed}"
    )
    target = BASE / "outputs" / tag
    if (target / "fit_status.json").exists() and not args.force:
        print(json.dumps({"status": "already_complete", "tag": tag}))
        return

    command = [
        str(PYTHON), str(RUNNER),
        "--seed", str(args.fit_seed),
        "--replica-seed", str(args.replica_seed),
        "--source-production", str(SOURCE),
        "--w-grid", str(W_GRID),
        "--output-root", str(BASE / "outputs"),
        "--tag", tag,
        "--initial-state", str(source / "model_state.pt"),
        "--initial-norms", str(source / "dataset_norms.csv"),
        "--initial-perturbation", str(args.initial_perturbation),
        "--max-epochs", "20000",
        "--min-epochs", "5000",
        "--plateau-patience", "3000",
        "--learning-rate", "1e-5",
        "--lbfgs-max-iter", "30000",
        "--float64",
        "--lambda-fnp-f-slope", str(slope["lambda"]),
        "--fnp-f-slope-bmin", str(slope["b_min"]),
        "--fnp-f-slope-bmax", str(slope["b_max"]),
        "--closure-tail-coordinate",
        "--closure-tail-b-start", str(constraint["b_start"]),
        "--closure-tail-b-end", str(constraint["b_end"]),
        "--lambda-fnp-transform-closure", str(closure["lambda"]),
        "--fnp-transform-closure-bmin", str(closure["b_min"]),
        "--fnp-transform-closure-max", str(closure["maximum_fnp"]),
    ]
    if args.initial_perturbation > 0.0:
        command.append("--allow-initial-state-perturbation")
    log = BASE / "logs" / f"{tag}.log"
    with log.open("w") as stream:
        result = subprocess.run(
            command, stdout=stream, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)
    print(json.dumps({
        "status": "complete",
        "tag": tag,
        "source_fit": str(source),
        "replica_seed": args.replica_seed,
        "fit_seed": args.fit_seed,
    }))


if __name__ == "__main__":
    main()
