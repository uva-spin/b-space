#!/usr/bin/env python3
"""Run one isolated reduced-capacity FiLM pilot."""

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
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--seed", type=int, default=801)
    parser.add_argument("--max-epochs", type=int, default=20000)
    parser.add_argument("--min-epochs", type=int, default=5000)
    parser.add_argument("--plateau-patience", type=int, default=2500)
    parser.add_argument("--lbfgs-max-iter", type=int, default=10000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    registry = pd.read_csv(BASE / "manifests/dataset_candidates.csv")
    selected = registry[registry["candidate_id"].eq(args.candidate)]
    if len(selected) != 1 or not bool(selected.iloc[0]["central_output_exists"]):
        raise ValueError(args.candidate)
    source = Path(selected.iloc[0]["central_output"])

    config = json.loads((BASE / "config/film_capacity_ladder.json").read_text())
    matches = [
        item for item in config["architectures_ordered_small_to_large"]
        if item["id"] == args.architecture
    ]
    if len(matches) != 1:
        raise ValueError(args.architecture)
    architecture = matches[0]
    tag = f"filmcap_{args.candidate}_{args.architecture}_s{args.seed}"
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
        "--np-width", str(architecture["np_width"]),
        "--np-cond-width", str(architecture["np_cond_width"]),
        "--np-blocks", str(architecture["np_blocks"]),
        "--distill-accepted-steps",
        str(config["initialization"]["distill_steps"]),
        "--distill-b-scale",
        str(config["initialization"]["b_weight_scale_GeV_inverse"]),
        "--distill-b-power",
        str(config["initialization"]["b_weight_power"]),
        "--distill-logx-nodes",
        str(config["initialization"]["dense_logx_nodes"]),
        "--distill-prediction-steps",
        str(config["initialization"]["prediction_distill_steps"]),
        "--max-epochs", str(args.max_epochs),
        "--min-epochs", str(args.min_epochs),
        "--plateau-patience", str(args.plateau_patience),
        "--learning-rate", "1e-4",
        "--lbfgs-max-iter", str(args.lbfgs_max_iter),
        "--float64",
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
        "architecture": architecture,
        "seed": args.seed,
    }))


if __name__ == "__main__":
    main()
