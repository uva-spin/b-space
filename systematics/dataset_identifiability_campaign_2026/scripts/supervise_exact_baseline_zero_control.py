#!/usr/bin/env python3
"""Reproduce and fully polish all 24 baseline endpoints with zero new prior."""

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
SEEDS = tuple(range(303, 327))
TARGET = BASE / "summaries/exact_baseline_zero_control"


def main() -> None:
    for seed in SEEDS:
        subprocess.run([
            str(PYTHON), str(LAUNCHER),
            "--seed", str(seed), "--strength", "0",
        ], check=True)

    rows, curves, original_curves = [], [], []
    for seed in SEEDS:
        original_dir = (
            UNITARY / "outputs" / f"fig6_lbfgs_stationary_s{seed}")
        control_dir = (
            BASE / "outputs" / f"exactbaseline_zero_control_s{seed}")
        original = json.loads((original_dir / "fit_status.json").read_text())
        control = json.loads((control_dir / "fit_status.json").read_text())
        original_grid = pd.read_csv(original_dir / "fnp_grid.csv")
        control_grid = pd.read_csv(control_dir / "fnp_grid.csv")
        original_grid = original_grid[
            np.isclose(original_grid["x"], 0.1)].sort_values("bT")
        control_grid = control_grid[
            np.isclose(control_grid["x"], 0.1)].sort_values("bT")
        original_curves.append(original_grid["F_NP"].to_numpy(float))
        curves.append(control_grid["F_NP"].to_numpy(float))
        rows.append({
            "seed": seed,
            "original_total_chi2": original["final"]["total_chi2"],
            "control_unpenalized_total_chi2": (
                control["final"]["unpenalized_total_chi2"]),
            "delta_total_chi2": (
                control["final"]["unpenalized_total_chi2"]
                - original["final"]["total_chi2"]),
            "control_chi2_per_point": (
                control["final"]["unpenalized_total_chi2"]
                / control["row_count"]),
            "control_fnp_gradient_l2": (
                control["final"]["fnp_gradient_l2_per_row_objective"]),
            "control_norm_gradient_l2": (
                control["final"][
                    "normalization_gradient_l2_per_row_objective"]),
            "stopped_on_adam_plateau": control["stopped_on_plateau"],
            "lbfgs_closure_evaluations": (
                control["lbfgs"]["closure_evaluations"]),
        })
    curves = np.asarray(curves)
    original_curves = np.asarray(original_curves)
    b = control_grid["bT"].to_numpy(float)
    active = b <= 2.0

    def max_relative_range(values):
        median = np.median(values[:, active], axis=0)
        return float(np.max(
            (np.max(values[:, active], axis=0)
             - np.min(values[:, active], axis=0))
            / np.maximum(median, 0.05)))

    table = pd.DataFrame(rows)
    TARGET.mkdir(parents=True, exist_ok=True)
    table.to_csv(TARGET / "runs.csv", index=False)
    summary = {
        "status": "isolated_exact_baseline_zero_prior_control_complete",
        "endpoint_count": len(SEEDS),
        "objective": (
            "exact original likelihood and architecture, zero added "
            "regularization, float64 Adam plus L-BFGS continuation"),
        "original_max_relative_full_range_x0p1_bT_le_2": (
            max_relative_range(original_curves)),
        "repolished_max_relative_full_range_x0p1_bT_le_2": (
            max_relative_range(curves)),
        "fit": {
            "maximum_chi2_per_point": float(
                table["control_chi2_per_point"].max()),
            "maximum_source_relative_raw_chi2_change": float(
                table["delta_total_chi2"].max()),
            "minimum_source_relative_raw_chi2_change": float(
                table["delta_total_chi2"].min()),
        },
        "stationarity": {
            "maximum_fnp_gradient_l2": float(
                table["control_fnp_gradient_l2"].max()),
            "maximum_norm_gradient_l2": float(
                table["control_norm_gradient_l2"].max()),
        },
        "next_gate": (
            "rebuild b/k-space baseline from these controls and require "
            "agreement before comparing any positive length strength"),
        "production_sources_modified": False,
    }
    (TARGET / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
