#!/usr/bin/env python3
"""Run matched-protocol localized F-length pilots and summarize them."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
SYSTEMATICS = BASE.parent
UNITARY = (
    SYSTEMATICS
    / "high_qt_direct_production_benchmark/experimental_unitary_transition")
PYTHON = Path("/home/dustin/miniforge3/envs/pdf-fit/bin/python")
LAUNCHER = BASE / "scripts/run_matched_baseline_flength_polish.py"
SEEDS = (316, 303, 307)
STRENGTHS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
TARGET = BASE / "summaries/matched_baseline_flength_pilot_ladder"


def token(value: float) -> str:
    return f"{value:.0e}".replace("-", "m").replace("+", "")


def tag(strength: float, seed: int) -> str:
    return (
        f"exactbaseline_matched_trueFlength_b0p1_2p0_"
        f"lam{token(strength)}_s{seed}")


def max_range(curves, b):
    active = b <= 2.0
    values = np.asarray(curves)[:, active]
    median = np.median(values, axis=0)
    return float(np.max(
        (values.max(axis=0) - values.min(axis=0))
        / np.maximum(median, 0.05)))


def main() -> None:
    for strength in STRENGTHS:
        for seed in SEEDS:
            subprocess.run([
                str(PYTHON), str(LAUNCHER), "--seed", str(seed),
                "--strength", str(strength)], check=True)
    baseline_curves, rows = [], []
    for seed in SEEDS:
        old = UNITARY / "outputs" / f"fig6_lbfgs_stationary_s{seed}"
        grid = pd.read_csv(old / "fnp_grid.csv")
        grid = grid[np.isclose(grid["x"], .1)].sort_values("bT")
        baseline_curves.append(grid["F_NP"].to_numpy(float))
    b = grid["bT"].to_numpy(float)
    baseline_range = max_range(baseline_curves, b)
    for strength in STRENGTHS:
        curves = []
        for seed in SEEDS:
            run = BASE / "outputs" / tag(strength, seed)
            status = json.loads((run / "fit_status.json").read_text())
            old_status = json.loads((
                UNITARY / "outputs" / f"fig6_lbfgs_stationary_s{seed}"
                / "fit_status.json").read_text())
            grid = pd.read_csv(run / "fnp_grid.csv")
            grid = grid[np.isclose(grid["x"], .1)].sort_values("bT")
            curves.append(grid["F_NP"].to_numpy(float))
            rows.append({
                "strength": strength, "seed": seed,
                "delta_unpenalized_total_chi2": (
                    status["final"]["unpenalized_total_chi2"]
                    - old_status["final"]["total_chi2"]),
                "chi2_per_point": (
                    status["final"]["unpenalized_total_chi2"]
                    / status["row_count"]),
                "fnp_gradient_l2": (
                    status["final"]["fnp_gradient_l2_per_row_objective"]),
                "lbfgs_closures": status["lbfgs"]["closure_evaluations"],
            })
        trial_range = max_range(curves, b)
        for row in rows[-len(SEEDS):]:
            row["max_relative_full_range_bT_le_2"] = trial_range
            row["range_reduction_fraction"] = (
                1.0 - trial_range / baseline_range)
    table = pd.DataFrame(rows)
    summary_table = table.groupby("strength", as_index=False).agg(
        max_delta_total_chi2=("delta_unpenalized_total_chi2", "max"),
        max_chi2_per_point=("chi2_per_point", "max"),
        max_fnp_gradient_l2=("fnp_gradient_l2", "max"),
        max_relative_full_range_bT_le_2=(
            "max_relative_full_range_bT_le_2", "first"),
        range_reduction_fraction=("range_reduction_fraction", "first"),
    )
    TARGET.mkdir(parents=True, exist_ok=True)
    table.to_csv(TARGET / "runs.csv", index=False)
    summary_table.to_csv(TARGET / "strength_summary.csv", index=False)
    summary = {
        "status": "isolated_matched_baseline_Flength_pilots_complete",
        "baseline_pilot_max_relative_full_range_bT_le_2": baseline_range,
        "protocol": (
            "exact historical precursor states, nuisance normalizations, "
            "float32, zero Adam, L-BFGS max_iter=500; only F-length differs"),
        "pilot_seeds": list(SEEDS),
        "strengths": list(STRENGTHS),
        "production_sources_modified": False,
    }
    (TARGET / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(summary_table.to_string(index=False))


if __name__ == "__main__":
    main()
