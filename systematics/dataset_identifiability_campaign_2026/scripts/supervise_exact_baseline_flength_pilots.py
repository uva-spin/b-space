#!/usr/bin/env python3
"""Run and summarize the exact-baseline localized F-length pilot ladder."""

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
LAUNCHER = BASE / "scripts/run_exact_baseline_flength.py"
SEEDS = (316, 303, 307)  # low, central, and high FNP(x=.1,b=2) basins
STRENGTHS = (1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3, 1.0e-2)
TARGET = BASE / "summaries/exact_baseline_flength_pilot_ladder"


def token(value: float) -> str:
    return f"{value:.0e}".replace("-", "m").replace("+", "")


def tag(strength: float, seed: int) -> str:
    return (
        f"exactbaseline_trueFlength_b0p1_2p0_"
        f"lam{token(strength)}_s{seed}")


def source_status(seed: int) -> dict:
    return json.loads((
        UNITARY / "outputs" / f"fig6_lbfgs_stationary_s{seed}"
        / "fit_status.json").read_text())


def summarize() -> dict:
    rows = []
    baseline_grids = {}
    for seed in SEEDS:
        baseline_grids[seed] = pd.read_csv(
            UNITARY / "outputs" / f"fig6_lbfgs_stationary_s{seed}"
            / "fnp_grid.csv")
    for strength in STRENGTHS:
        grids = []
        for seed in SEEDS:
            run = BASE / "outputs" / tag(strength, seed)
            status = json.loads((run / "fit_status.json").read_text())
            original = source_status(seed)
            grid = pd.read_csv(run / "fnp_grid.csv")
            grid = grid[np.isclose(grid["x"], 0.1)].sort_values("bT")
            grids.append(grid["F_NP"].to_numpy(float))
            rows.append({
                "strength": strength,
                "seed": seed,
                "unpenalized_total_chi2": (
                    status["final"]["unpenalized_total_chi2"]),
                "baseline_total_chi2": original["final"]["total_chi2"],
                "delta_total_chi2": (
                    status["final"]["unpenalized_total_chi2"]
                    - original["final"]["total_chi2"]),
                "chi2_per_point": (
                    status["final"]["unpenalized_total_chi2"]
                    / status["row_count"]),
                "fnp_gradient_l2": (
                    status["final"]["fnp_gradient_l2_per_row_objective"]),
                "stopped_on_plateau": status["stopped_on_plateau"],
                "epochs_run": status["epochs_run"],
            })
        matrix = np.asarray(grids)
        b = grid["bT"].to_numpy(float)
        active = b <= 2.0
        median = np.median(matrix[:, active], axis=0)
        relative_range = (
            np.max(matrix[:, active], axis=0)
            - np.min(matrix[:, active], axis=0)
        ) / np.maximum(median, 0.05)
        for row in rows[-len(SEEDS):]:
            row["pilot_max_relative_full_range_bT_le_2"] = float(
                np.max(relative_range))

    table = pd.DataFrame(rows)
    baseline_matrix = []
    for seed in SEEDS:
        grid = baseline_grids[seed]
        grid = grid[np.isclose(grid["x"], 0.1)].sort_values("bT")
        baseline_matrix.append(grid["F_NP"].to_numpy(float))
    baseline_matrix = np.asarray(baseline_matrix)
    b = grid["bT"].to_numpy(float)
    active = b <= 2.0
    baseline_median = np.median(baseline_matrix[:, active], axis=0)
    baseline_range = np.max(
        (np.max(baseline_matrix[:, active], axis=0)
         - np.min(baseline_matrix[:, active], axis=0))
        / np.maximum(baseline_median, 0.05))

    grouped = table.groupby("strength", as_index=False).agg(
        max_delta_total_chi2=("delta_total_chi2", "max"),
        max_chi2_per_point=("chi2_per_point", "max"),
        max_fnp_gradient_l2=("fnp_gradient_l2", "max"),
        all_plateaued=("stopped_on_plateau", "all"),
        max_relative_full_range_bT_le_2=(
            "pilot_max_relative_full_range_bT_le_2", "first"),
    )
    grouped["range_reduction_fraction"] = (
        1.0 - grouped["max_relative_full_range_bT_le_2"] / baseline_range)
    TARGET.mkdir(parents=True, exist_ok=True)
    table.to_csv(TARGET / "runs.csv", index=False)
    grouped.to_csv(TARGET / "strength_summary.csv", index=False)
    result = {
        "status": "isolated_exact_baseline_Flength_pilots_complete",
        "baseline_pilot_max_relative_full_range_bT_le_2": float(
            baseline_range),
        "pilot_seeds": list(SEEDS),
        "strengths": list(STRENGTHS),
        "selection_gate": {
            "fit": (
                "maximum source-relative raw chi2 increase <= 3.29 "
                "(0.01 per point)"),
            "stationarity": (
                "all plateaued and maximum FNP objective-gradient <= 1e-3"),
            "improvement": (
                "positive reduction in x=.1, bT<=2 pilot full range; "
                "full 24-start validation required"),
        },
        "production_sources_modified": False,
    }
    (TARGET / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    for strength in STRENGTHS:
        for seed in SEEDS:
            subprocess.run([
                str(PYTHON), str(LAUNCHER),
                "--seed", str(seed), "--strength", str(strength),
            ], check=True)
    result = summarize()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
