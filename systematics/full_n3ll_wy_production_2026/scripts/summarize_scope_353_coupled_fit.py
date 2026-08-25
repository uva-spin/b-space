#!/usr/bin/env python3
"""Summarize the first isolated 353-row coupled F_NP fit.

This is deliberately a candidate-side record.  It does not promote or copy
anything into the frozen production campaign.
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
FIT = BASE / "reports/scope_353_coupled_fnp_fit_lambda1_candidate_s303"
OUT = BASE / "reports/scope_353_coupled_fnp_fit_lambda1_candidate_summary.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default=str(FIT), help="candidate fit directory")
    ap.add_argument("--output", default=None, help="summary JSON path")
    args = ap.parse_args()
    fit = Path(args.fit)
    output = Path(args.output) if args.output else fit.parent / (fit.name + "_summary.json")
    metrics = json.loads((fit / "metrics.json").read_text())
    history = pd.read_csv(fit / "loss_history.csv")
    mono = pd.read_csv(fit / "fnp_monotonicity.csv")
    train = metrics["train"]
    per_dataset = {str(row["dataset"]): row for row in metrics["per_dataset"]}
    summary = {
        "status": "isolated_scope_353_coupled_fnp_fit_complete_not_production",
        "scope_rows": int(train["n_points"]),
        "architecture": "v19 local-bcurv trainer, monotone positive F_NP, no learned CS kernel",
        "w_y_inputs": "candidate full-scope external W plus conventional full-minus-RES Y",
        "seed": int(metrics["config"]["seed"]),
        "epochs_run": int(train["epochs_run"]),
        "best_epoch": int(train["best_epoch"]),
        "best_chi2_like": float(train["best_chi2_like"]),
        "best_objective_per_row": float(train["best_chi2_like"] / train["n_points"]),
        "last_chi2_like": float(train["last_epoch_chi2_like"]),
        "plateau_indicator": {
            "best_in_final_10_percent": bool(train["best_epoch"] >= 0.9 * train["epochs_run"]),
            "last_minus_best": float(train["last_epoch_chi2_like"] - train["best_chi2_like"]),
            "interpretation": "optimization reached a late best epoch, but start stability has not yet been tested",
        },
        "fnp_monotonicity": {
            "all_probe_curves_pass": bool(mono["is_monotone_tol_1e_minus_4"].all()),
            "max_increase": float(mono["max_increase"].max()),
            "min_fnp": float(mono["min_F_NP"].min()),
        },
        "per_dataset": per_dataset,
        "known_warnings": [
            "This is a coupled candidate fit, not the frozen lambda=1 production objective.",
            "The W fragment is internal n3llp/nloQ96 while Tevatron Y is DYTurbo-derived; source-level consistency remains open.",
            "LHCb full-minus-RES Y has cancellation-driven relative MC uncertainty above 100% and remains diagnostic.",
            "The candidate trainer reports a pathological large normalization pull for CDF_RUN_2 because its imported norm metadata has zero norm_rel; this must be repaired before any production comparison.",
        ],
        "promotion_authorized": False,
        "frozen_production_modified": False,
    }
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
