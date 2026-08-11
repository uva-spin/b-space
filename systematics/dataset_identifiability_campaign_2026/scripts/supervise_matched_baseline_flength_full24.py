#!/usr/bin/env python3
"""Run matched-protocol F-length candidates on all 24 baseline basins."""

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
BUILD_B = BASE / "scripts/build_bspace_tmd_ensemble.py"
TRANSFORM_K = BASE / "scripts/transform_bspace_ensemble_to_kspace.py"
SEEDS = tuple(range(303, 327))
STRENGTHS = (0.03, 0.05, 0.07, 0.10)
TARGET = BASE / "summaries/matched_baseline_flength_full24"


def token(value: float) -> str:
    return f"{value:.0e}".replace("-", "m").replace("+", "")


def tag(strength: float, seed: int) -> str:
    return (
        f"exactbaseline_matched_trueFlength_b0p1_2p0_"
        f"lam{token(strength)}_s{seed}")


def main() -> None:
    for strength in STRENGTHS:
        for seed in SEEDS:
            subprocess.run([
                str(PYTHON), str(LAUNCHER), "--seed", str(seed),
                "--strength", str(strength)], check=True)
        name = f"matched_baseline_flength_lam{token(strength)}_full24"
        build_command = [
            str(PYTHON), str(BUILD_B), "--target-name", f"{name}_bspace",
            "--Q", "10", "--flavor", "u", "--flavor", "d"]
        for seed in SEEDS:
            build_command.extend(["--run-tag", tag(strength, seed)])
        subprocess.run(build_command, check=True)
        subprocess.run([
            str(PYTHON), str(TRANSFORM_K),
            "--bspace-ensemble",
            str(BASE / "summaries" / f"{name}_bspace"
                / "bspace_tmd_ensemble_long.csv"),
            "--target-name", f"{name}_kspace"], check=True)

    rows = []
    for strength in STRENGTHS:
        curves = []
        for seed in SEEDS:
            run = BASE / "outputs" / tag(strength, seed)
            status = json.loads((run / "fit_status.json").read_text())
            original = json.loads((
                UNITARY / "outputs" / f"fig6_lbfgs_stationary_s{seed}"
                / "fit_status.json").read_text())
            grid = pd.read_csv(run / "fnp_grid.csv")
            grid = grid[np.isclose(grid["x"], .1)].sort_values("bT")
            curves.append(grid["F_NP"].to_numpy(float))
            rows.append({
                "strength": strength, "seed": seed,
                "delta_total_chi2": (
                    status["final"]["unpenalized_total_chi2"]
                    - original["final"]["total_chi2"]),
                "chi2_per_point": (
                    status["final"]["unpenalized_total_chi2"]
                    / status["row_count"]),
                "fnp_gradient_l2": (
                    status["final"]["fnp_gradient_l2_per_row_objective"]),
            })
        matrix = np.asarray(curves)
        b = grid["bT"].to_numpy(float)
        active = b <= 2.0
        median = np.median(matrix[:, active], axis=0)
        q16, q84 = np.quantile(matrix[:, active], [.16, .84], axis=0)
        full_range = float(np.max(
            (matrix[:, active].max(axis=0) - matrix[:, active].min(axis=0))
            / np.maximum(median, .05)))
        central68 = float(np.max(
            (q84 - q16) / np.maximum(median, .05)))
        for row in rows[-len(SEEDS):]:
            row["max_relative_full_range_bT_le_2"] = full_range
            row["max_relative_central68_width_bT_le_2"] = central68
    table = pd.DataFrame(rows)
    grouped = table.groupby("strength", as_index=False).agg(
        max_delta_total_chi2=("delta_total_chi2", "max"),
        median_delta_total_chi2=("delta_total_chi2", "median"),
        max_chi2_per_point=("chi2_per_point", "max"),
        max_relative_full_range_bT_le_2=(
            "max_relative_full_range_bT_le_2", "first"),
        max_relative_central68_width_bT_le_2=(
            "max_relative_central68_width_bT_le_2", "first"),
    )
    TARGET.mkdir(parents=True, exist_ok=True)
    table.to_csv(TARGET / "runs.csv", index=False)
    grouped.to_csv(TARGET / "candidate_summary.csv", index=False)
    summary = {
        "status": "isolated_matched_baseline_Flength_full24_complete",
        "protocol": (
            "exact historical final-polish protocol with only localized "
            "direct-FNP path length added"),
        "endpoint_count": len(SEEDS),
        "strengths": list(STRENGTHS),
        "downstream": (
            "b-space ensembles and regularized k-space transforms built for "
            "both candidates; experimental residual crossing follows"),
        "production_sources_modified": False,
    }
    (TARGET / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(grouped.to_string(index=False))


if __name__ == "__main__":
    main()
