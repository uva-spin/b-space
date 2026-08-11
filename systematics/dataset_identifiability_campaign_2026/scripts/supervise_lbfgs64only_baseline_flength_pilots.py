#!/usr/bin/env python3
"""Supervise stationary zero/F-length comparisons on separated basins."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/dustin/miniforge3/envs/pdf-fit/bin/python")
LAUNCHER = BASE / "scripts/run_lbfgs64only_baseline_flength.py"
SEEDS = (316, 303, 307)
STRENGTHS = (0.0, 0.03, 0.10)
TARGET = BASE / "summaries/lbfgs64only_baseline_flength_pilots"


def token(value: float) -> str:
    return "zero" if value == 0 else f"{value:.0e}".replace("-", "m").replace("+", "")


def tag(strength: float, seed: int) -> str:
    return (
        "exactbaseline_lbfgs64only_trueFlength_b0p1_2p0_"
        f"lam{token(strength)}_s{seed}")


def main() -> None:
    for strength in STRENGTHS:
        for seed in SEEDS:
            subprocess.run([
                str(PYTHON), str(LAUNCHER), "--seed", str(seed),
                "--strength", str(strength)], check=True)
    rows = []
    for strength in STRENGTHS:
        curves = []
        for seed in SEEDS:
            run = BASE / "outputs" / tag(strength, seed)
            status = json.loads((run / "fit_status.json").read_text())
            grid = pd.read_csv(run / "fnp_grid.csv")
            grid = grid[np.isclose(grid["x"], .1)].sort_values("bT")
            curves.append(grid["F_NP"].to_numpy(float))
            rows.append({
                "strength": strength, "seed": seed,
                "unpenalized_total_chi2": (
                    status["final"]["unpenalized_total_chi2"]),
                "chi2_per_point": (
                    status["final"]["unpenalized_total_chi2"]
                    / status["row_count"]),
                "fnp_gradient_l2": (
                    status["final"]["fnp_gradient_l2_per_row_objective"]),
                "norm_gradient_l2": (
                    status["final"][
                        "normalization_gradient_l2_per_row_objective"]),
                "lbfgs_closures": status["lbfgs"]["closure_evaluations"],
            })
        matrix = np.asarray(curves)
        b = grid["bT"].to_numpy(float)
        active = b <= 2.0
        median = np.median(matrix[:, active], axis=0)
        relative_range = float(np.max(
            (matrix[:, active].max(axis=0) - matrix[:, active].min(axis=0))
            / np.maximum(median, 0.05)))
        for row in rows[-len(SEEDS):]:
            row["max_relative_full_range_bT_le_2"] = relative_range
    table = pd.DataFrame(rows)
    zero = table[table.strength.eq(0)].set_index("seed")
    for index, row in table.iterrows():
        table.loc[index, "delta_chi2_vs_matched_zero"] = (
            row.unpenalized_total_chi2
            - zero.loc[row.seed, "unpenalized_total_chi2"])
    grouped = table.groupby("strength", as_index=False).agg(
        max_delta_chi2_vs_zero=("delta_chi2_vs_matched_zero", "max"),
        max_chi2_per_point=("chi2_per_point", "max"),
        max_fnp_gradient_l2=("fnp_gradient_l2", "max"),
        max_norm_gradient_l2=("norm_gradient_l2", "max"),
        max_relative_full_range_bT_le_2=(
            "max_relative_full_range_bT_le_2", "first"),
    )
    TARGET.mkdir(parents=True, exist_ok=True)
    table.to_csv(TARGET / "runs.csv", index=False)
    grouped.to_csv(TARGET / "strength_summary.csv", index=False)
    summary = {
        "status": "isolated_LBFGS64only_baseline_Flength_pilots_complete",
        "protocol": (
            "same three stored endpoints and nuisance normalizations; zero "
            "Adam; float64 L-BFGS max_iter=20000 for zero and both priors"),
        "pilot_seeds": list(SEEDS),
        "strengths": list(STRENGTHS),
        "production_sources_modified": False,
    }
    (TARGET / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(grouped.to_string(index=False))


if __name__ == "__main__":
    main()
