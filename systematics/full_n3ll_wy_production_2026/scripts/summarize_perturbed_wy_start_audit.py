#!/usr/bin/env python3
"""Summarize the corrected W+Y perturbed-start diagnostics.

This report is deliberately candidate-side.  It records the spread generated
by controlled 1% parameter perturbations and keeps the LHCb-Y, non-LHCb-Y, and
all-Y controls separate; they are not interchangeable uncertainty ensembles.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
REPORTS = BASE / "reports"
SEEDS = tuple(range(303, 311))
B_PROBES = (1.0, 2.0, 4.0, 6.0, 8.0)

CASES = {
    "all_y_no_tail": "scope_329_perturbed1pct_all_y",
    "non_lhcb_y_no_tail": "scope_329_perturbed1pct_non_lhcb_y",
    "lhcb_only_y_no_tail": "scope_329_perturbed1pct_lhcb_only_y",
    "all_y_tail_lambda1": "scope_329_perturbed1pct_all_y_tail1",
    "non_lhcb_y_tail_lambda1": "scope_329_perturbed1pct_non_lhcb_y_tail1",
    "non_lhcb_y_tail_lambda1_refdist1": "scope_329_perturbed1pct_non_lhcb_y_tail1_refdist1",
    "lhcb_only_y_tail_lambda1": "scope_329_perturbed1pct_lhcb_only_y_tail1",
}


def probe_fnp(label: str) -> tuple[np.ndarray, dict]:
    values = []
    objectives = []
    lhcb_chi2 = []
    for seed in SEEDS:
        directory = REPORTS / f"{label}_wy_s{seed}"
        frame = pd.read_csv(directory / "fnp_debug_grid.csv")
        frame = frame[np.isclose(frame["x"].to_numpy(float), 0.1)].sort_values("bT")
        values.append([
            float(frame.iloc[np.argmin(np.abs(frame["bT"].to_numpy(float) - b))]["F_NP"])
            for b in B_PROBES
        ])
        metrics = json.loads((directory / "metrics.json").read_text())
        objectives.append(float(metrics["train"]["final_chi2_like"]))
        by_dataset = {row["dataset"]: row for row in metrics["per_dataset"]}
        lhcb_chi2.append(float(by_dataset["LHCb_7"]["chi2_like"]))
    array = np.asarray(values, dtype=float)
    stats = {}
    for index, b in enumerate(B_PROBES):
        q16, median, q84 = np.quantile(array[:, index], (0.16, 0.50, 0.84))
        stats[str(b)] = {
            "min": float(np.min(array[:, index])),
            "q16": float(q16), "median": float(median), "q84": float(q84),
            "max": float(np.max(array[:, index])),
            "full_range_over_median": float(
                (np.max(array[:, index]) - np.min(array[:, index])) / max(median, 1e-12)
            ),
        }
    return array, {
        "label": label,
        "seeds": list(SEEDS),
        "bT_probes_GeV_inv": list(B_PROBES),
        "fnp_x0p1": stats,
        "objective_per_point": {
            "min": float(np.min(objectives)), "median": float(np.median(objectives)),
            "max": float(np.max(objectives)),
        },
        "LHCb_7_chi2_like": {
            "min": float(np.min(lhcb_chi2)), "median": float(np.median(lhcb_chi2)),
            "max": float(np.max(lhcb_chi2)),
        },
    }


def main() -> None:
    result = {
        "status": "isolated_perturbed_start_wy_diagnostic_not_production",
        "start_perturbation": "1 percent of tensor RMS, independent Gaussian parameter perturbations around the frozen candidate state",
        "ensemble_interpretation": "controlled sensitivity envelope, not a calibrated probability distribution or confidence interval",
        "fnp_definition": "pointwise positive monotone F_NP from each fitted start at x=0.1",
        "cases": {},
        "frozen_production_modified": False,
        "promotion_authorized": False,
    }
    for name, label in CASES.items():
        _, summary = probe_fnp(label)
        result["cases"][name] = summary
    target = REPORTS / "wy_perturbed_start_audit.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
