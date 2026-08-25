#!/usr/bin/env python3
"""Summarize isolated baseline-reference W+Y trials in one machine-readable record."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
REPORTS = BASE / "reports"
SEEDS = tuple(range(303, 311))
TRIALS = {
    "refdist1_b2_3k": {
        "label": "scope_329_perturbed1pct_non_lhcb_y_tail1_refdist1",
        "seeds": SEEDS,
        "lambda": 1.0, "bmax": 2.0, "epochs": 3000,
        "reference": "historical exact 24-start all-x table",
    },
    "refdist3_b2_3k": {
        "label": "scope_329_perturbed1pct_non_lhcb_y_tail1_refdist3_b2",
        "seeds": tuple(range(303, 307)),
        "lambda": 3.0, "bmax": 2.0, "epochs": 3000,
        "reference": "historical exact 24-start all-x table",
    },
    "refdist3_b8_3k": {
        "label": "scope_329_perturbed1pct_non_lhcb_y_tail1_refdist3_b8",
        "seeds": SEEDS,
        "lambda": 3.0, "bmax": 8.0, "epochs": 3000,
        "reference": "historical exact 24-start all-x table",
    },
    "refdist3_b8_long10k": {
        "label": "scope_329_perturbed1pct_non_lhcb_y_tail1_refdist3_b8_long10k",
        "seeds": SEEDS,
        "lambda": 3.0, "bmax": 8.0, "epochs": 10000,
        "reference": "historical exact 24-start all-x table",
    },
    "refdist3_b8_promoted96_long10k": {
        "label": "scope_329_perturbed1pct_non_lhcb_y_tail1_refdist3_b8_promoted96_long10k",
        "seeds": SEEDS,
        "lambda": 3.0, "bmax": 8.0, "epochs": 10000,
        "reference": "promoted 96-start x=0.1 slice plus historical other x knots",
    },
    "refdist1_b8_promoted96_long10k": {
        "label": "scope_329_perturbed1pct_non_lhcb_y_tail1_refdist1_b8_promoted96_long10k",
        "seeds": SEEDS,
        "lambda": 1.0, "bmax": 8.0, "epochs": 10000,
        "reference": "promoted 96-start x=0.1 slice plus historical other x knots",
    },
    "refdist2_b8_promoted96_long10k": {
        "label": "scope_329_perturbed1pct_non_lhcb_y_tail1_refdist2_b8_promoted96_long10k",
        "seeds": SEEDS,
        "lambda": 2.0, "bmax": 8.0, "epochs": 10000,
        "reference": "promoted 96-start x=0.1 slice plus historical other x knots",
    },
    "refdist4_b8_promoted96_long10k": {
        "label": "scope_329_perturbed1pct_non_lhcb_y_tail1_refdist4_b8_promoted96_long10k",
        "seeds": SEEDS,
        "lambda": 4.0, "bmax": 8.0, "epochs": 10000,
        "reference": "promoted 96-start x=0.1 slice plus historical other x knots",
    },
}
B_PROBES = (1.0, 2.0, 4.0, 8.0)


def summarize_trial(spec: dict) -> dict:
    rows = []
    for seed in spec["seeds"]:
        directory = REPORTS / f'{spec["label"]}_wy_s{seed}'
        metrics = json.loads((directory / "metrics.json").read_text())
        by_dataset = {row["dataset"]: row for row in metrics["per_dataset"]}
        grid = pd.read_csv(directory / "fnp_debug_grid.csv")
        grid = grid[np.isclose(grid["x"].to_numpy(float), 0.1)].sort_values("bT")
        values = [float(grid.iloc[np.argmin(np.abs(grid["bT"].to_numpy(float) - b))]["F_NP"])
                  for b in B_PROBES]
        history = pd.read_csv(directory / "loss_history.csv")
        rows.append({
            "seed": int(seed),
            "objective_per_point": float(metrics["train"]["final_chi2_like"]),
            "best_epoch": int(metrics["train"]["best_epoch"]),
            "epochs_run": int(metrics["train"]["epochs_run"]),
            "last_objective_per_point": float(history.iloc[-1]["chi2_like"]),
            "LHCb_7_objective_per_point": float(by_dataset["LHCb_7"]["chi2_like"]),
            "F_NP_x0p1": {str(b): values[i] for i, b in enumerate(B_PROBES)},
        })
    objectives = np.asarray([row["objective_per_point"] for row in rows], dtype=float)
    lhcb = np.asarray([row["LHCb_7_objective_per_point"] for row in rows], dtype=float)
    fnp = np.asarray([[row["F_NP_x0p1"][str(b)] for b in B_PROBES] for row in rows], dtype=float)
    probes = {}
    for i, b in enumerate(B_PROBES):
        med = float(np.median(fnp[:, i]))
        probes[str(b)] = {
            "min": float(np.min(fnp[:, i])), "q16": float(np.quantile(fnp[:, i], .16)),
            "median": med, "q84": float(np.quantile(fnp[:, i], .84)),
            "max": float(np.max(fnp[:, i])),
            "full_range_over_median": float((np.max(fnp[:, i]) - np.min(fnp[:, i])) / max(med, 1e-12)),
        }
    return {
        **{k: v for k, v in spec.items() if k != "seeds"},
        "seeds": list(spec["seeds"]),
        "objective_per_point": {"min": float(np.min(objectives)), "median": float(np.median(objectives)), "max": float(np.max(objectives))},
        "LHCb_7_objective_per_point": {"min": float(np.min(lhcb)), "median": float(np.median(lhcb)), "max": float(np.max(lhcb))},
        "fnp_x0p1": probes,
        "runs": rows,
    }


def main() -> None:
    result = {
        "status": "isolated_baseline_reference_distance_trial_summary_not_production",
        "scope": "329-row corrected non-LHCb-Y W+Y case, 1% perturbed starts",
        "metric": "F_NP relative distance to empirical reference plus existing lambda_tail=1 scaffold",
        "trials": {name: summarize_trial(spec) for name, spec in TRIALS.items()},
        "frozen_production_modified": False,
        "promotion_authorized": False,
    }
    target = REPORTS / "wy_reference_distance_trial_summary.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
