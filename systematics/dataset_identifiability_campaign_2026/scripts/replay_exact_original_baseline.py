#!/usr/bin/env python3
"""Replay the exact optimizer sequence that produced all 24 Fig. 6 basins."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
SYSTEMATICS = BASE.parent
ROOT = SYSTEMATICS.parent
UNITARY = (
    SYSTEMATICS
    / "high_qt_direct_production_benchmark/experimental_unitary_transition")
RUNNER = UNITARY / "scripts/run_production_fnp_stability_control.py"
SOURCE = (
    SYSTEMATICS / "collins_factorization_validity/outputs/"
    "rowidfix_stageFT_all_qmax0p20_lam0p50_central_s303")
W_GRID = (
    ROOT / "outputs/v23a_tevatron_plus_lhcb7_fidacc_merged_cache/"
    "backend_cache/wpert_v23a_tevatron_plus_lhcb7_fidacc_b160.csv")
PYTHON = Path("/home/dustin/miniforge3/envs/pdf-fit/bin/python")
SEEDS = tuple(range(303, 327))
TARGET = BASE / "summaries/exact_original_baseline_replay"


def main() -> None:
    for seed in SEEDS:
        original_dir = (
            UNITARY / "outputs" / f"fig6_lbfgs_stationary_s{seed}")
        original = json.loads((original_dir / "fit_status.json").read_text())
        initial = Path(original["initial_state"])
        if not initial.is_absolute():
            initial = UNITARY / initial
        tag = f"exact_original_baseline_replay_with_norms_s{seed}"
        target = BASE / "outputs" / tag
        if (target / "fit_status.json").exists():
            continue
        log = BASE / "logs" / f"{tag}.log"
        command = [
            str(PYTHON), str(RUNNER),
            "--seed", str(seed),
            "--source-production", str(SOURCE),
            "--w-grid", str(W_GRID),
            "--output-root", str(BASE / "outputs"),
            "--tag", tag,
            "--initial-state", str(initial),
            "--initial-norms", str(initial.parent / "dataset_norms.csv"),
            "--max-epochs", "0",
            "--min-epochs", "0",
            "--plateau-patience", "0",
            "--lbfgs-max-iter", "500",
        ]
        with log.open("w") as stream:
            subprocess.run(
                command, stdout=stream, stderr=subprocess.STDOUT,
                check=True)

    rows = []
    max_grid_difference = 0.0
    for seed in SEEDS:
        old = UNITARY / "outputs" / f"fig6_lbfgs_stationary_s{seed}"
        new = (
            BASE / "outputs"
            / f"exact_original_baseline_replay_with_norms_s{seed}")
        old_status = json.loads((old / "fit_status.json").read_text())
        new_status = json.loads((new / "fit_status.json").read_text())
        old_grid = pd.read_csv(old / "fnp_grid.csv")
        new_grid = pd.read_csv(new / "fnp_grid.csv")
        difference = float(np.max(np.abs(
            old_grid["F_NP"].to_numpy(float)
            - new_grid["F_NP"].to_numpy(float))))
        max_grid_difference = max(max_grid_difference, difference)
        rows.append({
            "seed": seed,
            "old_total_chi2": old_status["final"]["total_chi2"],
            "replay_total_chi2": (
                new_status["final"]["unpenalized_total_chi2"]),
            "absolute_total_chi2_difference": abs(
                new_status["final"]["unpenalized_total_chi2"]
                - old_status["final"]["total_chi2"]),
            "max_absolute_fnp_grid_difference": difference,
            "old_lbfgs_closures": (
                old_status["lbfgs"]["closure_evaluations"]),
            "replay_lbfgs_closures": (
                new_status["lbfgs"]["closure_evaluations"]),
        })
    table = pd.DataFrame(rows)
    TARGET.mkdir(parents=True, exist_ok=True)
    table.to_csv(TARGET / "runs.csv", index=False)
    summary = {
        "status": "isolated_exact_original_baseline_replay_complete",
        "endpoint_count": len(SEEDS),
        "protocol": (
            "same precursor model and nuisance normalizations, float32, "
            "zero Adam epochs, "
            "L-BFGS max_iter=500, zero added regularization"),
        "maximum_absolute_total_chi2_difference": float(
            table["absolute_total_chi2_difference"].max()),
        "maximum_absolute_fnp_grid_difference": max_grid_difference,
        "reproduction_gate": {
            "chi2_absolute_tolerance": 1.0e-3,
            "fnp_absolute_tolerance": 1.0e-5,
            "passed": bool(
                table["absolute_total_chi2_difference"].max() <= 1.0e-3
                and max_grid_difference <= 1.0e-5),
        },
        "production_sources_modified": False,
    }
    (TARGET / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
