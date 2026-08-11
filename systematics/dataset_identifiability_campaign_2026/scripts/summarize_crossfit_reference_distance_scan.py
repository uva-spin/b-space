#!/usr/bin/env python3
"""Summarize reciprocal cross-fitted FNP-reference fits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
SYSTEMATICS = BASE.parent
UNITARY = SYSTEMATICS / "high_qt_direct_production_benchmark/experimental_unitary_transition"
SEEDS = tuple(range(303, 327))
STRENGTHS = (0.5, 0.75, 1.0, 1.5, 2.0)
TARGET = BASE / "summaries/crossfit_reference_distance_scan"


def fit_token(value: float) -> str:
    if value == 2.0:
        return "2p0"
    return f"{value:.0e}".replace("-", "m").replace("+", "")


def label_for(seed: int) -> str:
    return "evenref" if seed % 2 else "oddref"


def tag(strength: float, seed: int) -> str:
    return (f"exactbaseline_matched_reference_distance_{label_for(seed)}_"
            f"b0p1_2p0_lam{fit_token(strength)}_s{seed}")


def main() -> None:
    rows = []
    summaries = []
    for strength in STRENGTHS:
        curves = []
        template = None
        for seed in SEEDS:
            run = BASE / "outputs" / tag(strength, seed)
            if not (run / "fit_status.json").exists():
                continue
            status = json.loads((run / "fit_status.json").read_text())
            old = json.loads((UNITARY / "outputs" / f"fig6_lbfgs_stationary_s{seed}"
                              / "fit_status.json").read_text())
            grid = pd.read_csv(run / "fnp_grid.csv")
            grid = grid[np.isclose(grid.x, .1)].sort_values("bT")
            template = grid
            curves.append(grid.F_NP.to_numpy(float))
            rows.append({
                "strength": strength,
                "seed": seed,
                "reference_fold": label_for(seed),
                "delta_unpenalized_chi2": status["final"]["unpenalized_total_chi2"] - old["final"]["total_chi2"],
                "unpenalized_chi2": status["final"]["unpenalized_total_chi2"],
                "fnp_gradient_l2_per_row_objective": status["final"]["fnp_gradient_l2_per_row_objective"],
            })
        if not curves:
            continue
        matrix = np.asarray(curves)
        b = template.bT.to_numpy(float)
        active = b <= 2.0
        med = np.median(matrix[:, active], axis=0)
        q16, q84 = np.quantile(matrix[:, active], [.16, .84], axis=0)
        scale = np.maximum(med, .05)
        summaries.append({
            "strength": strength,
            "completed_members": len(curves),
            "max_source_relative_chi2_increase": max(r["delta_unpenalized_chi2"] for r in rows if r["strength"] == strength),
            "median_source_relative_chi2_change": float(np.median([r["delta_unpenalized_chi2"] for r in rows if r["strength"] == strength])),
            "max_fnp_full_range_bT_le_2": float(np.max((matrix[:, active].max(0) - matrix[:, active].min(0)) / scale)),
            "max_fnp_central68_width_bT_le_2": float(np.max((q84 - q16) / scale)),
        })
    TARGET.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(TARGET / "runs.csv", index=False)
    table = pd.DataFrame(summaries)
    table.to_csv(TARGET / "strength_summary.csv", index=False)
    even_path = BASE / "summaries/crossfit_reference_evenref/fnp_median.csv"
    odd_path = BASE / "summaries/crossfit_reference_oddref/fnp_median.csv"
    reference_difference = None
    if even_path.exists() and odd_path.exists():
        even, odd = pd.read_csv(even_path), pd.read_csv(odd_path)
        mask = np.isclose(even.x, .1) & (even.bT <= 2.0)
        a = even.loc[mask, "F_NP"].to_numpy()
        b = odd.loc[mask, "F_NP"].to_numpy()
        reference_difference = float(np.max(
            np.abs(a - b) / np.maximum(.5 * (a + b), .05)))
    summary = {
        "status": "complete" if len(rows) == len(SEEDS) * len(STRENGTHS) else "partial",
        "design": "reciprocal two-fold cross-fit; evaluated members excluded from reference construction",
        "expected_members_per_strength": len(SEEDS),
        "reference_max_relative_difference_x0p1_bT_le_2": reference_difference,
        "production_sources_modified": False,
    }
    (TARGET / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
