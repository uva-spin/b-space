#!/usr/bin/env python3
"""Summarize isolated shortest-path penalty pilots and multi-start checks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
OUTPUTS = BASE / "outputs"
TARGET = BASE / "summaries/shortest_path_soft_campaign"


def ensemble_metrics(tags: list[str]) -> dict:
    curves = []
    status_rows = []
    for tag in tags:
        status = json.loads((OUTPUTS / tag / "fit_status.json").read_text())
        frame = pd.read_csv(OUTPUTS / tag / "fnp_grid.csv")
        curves.append(frame.F_NP.to_numpy(float))
        status_rows.append({
            "tag": tag,
            "converged": status["convergence_gate_pass"],
            "epochs": status["epochs_run"],
            "best_epoch": status["best_epoch"],
            "data_chi2": status["final"]["data_chi2"],
            "unpenalized_total_chi2": status["final"]["unpenalized_total_chi2"],
            "shortest_path_penalty": status["final"]["shortest_path_penalty_per_row_objective"],
            "max_prediction_shift_over_sigma": status["final"]["max_prediction_shift_over_experimental_sigma"],
        })
    arr = np.asarray(curves)
    ref = pd.read_csv(OUTPUTS / tags[0] / "fnp_grid.csv")
    b = ref.bT.to_numpy(float)
    active = arr.mean(axis=0) > 0.05
    result = {"runs": status_rows}
    for bmax in (1.0, 2.0):
        mask = active & (b <= bmax)
        relative = (arr[:, mask].max(axis=0) - arr[:, mask].min(axis=0)) / np.maximum(arr[:, mask].mean(axis=0), 1.0e-30)
        result[f"b_le_{bmax:g}"] = {
            "max_relative_range": float(np.max(relative)),
            "median_relative_range": float(np.median(relative)),
            "point_count": int(mask.sum()),
        }
    return result


def main() -> None:
    groups = {}
    for metric, strengths in (("directF", (0.3, 1.0, 3.0, 10.0)),
                              ("logF", (1.0, 3.0, 10.0))):
        for strength in strengths:
            label = f"{strength:g}".replace(".", "p")
            pilot_tags = [f"shortest_{metric}_lambda{label}_pilot_s303"]
            long_tags = [f"shortest_{metric}_lambda{label}_long_s{seed}" for seed in (303, 304, 305)]
            if all((OUTPUTS / tag / "fit_status.json").exists() for tag in long_tags):
                groups[f"{metric}_lambda{strength:g}_long3"] = ensemble_metrics(long_tags)
            elif all((OUTPUTS / tag / "fit_status.json").exists() for tag in pilot_tags):
                groups[f"{metric}_lambda{strength:g}_pilot"] = ensemble_metrics(pilot_tags)
    cont_tags = [f"shortest_directF_lambda3_cont_s{seed}" for seed in (303, 304, 305)]
    if all((OUTPUTS / tag / "fit_status.json").exists() for tag in cont_tags):
        groups["directF_lambda3_continuation"] = ensemble_metrics(cont_tags)
    horizon_tags = [f"shortest_directF_lambda3_horizon10k_s{seed}" for seed in (303, 304, 305)]
    if all((OUTPUTS / tag / "fit_status.json").exists() for tag in horizon_tags):
        groups["directF_lambda3_horizon10k"] = ensemble_metrics(horizon_tags)

    summary = {
        "status": "isolated_shortest_path_soft_campaign_complete",
        "penalty": "quadrature-weighted mean squared residual from analytic fixed-endpoint path, normalized by endpoint span",
        "metrics": groups,
        "decision": (
            "No tested soft shortest-path strength is promoted: the best 5,000-epoch continuation "
            "(direct-F lambda=3) preserved fit quality but left roughly 99% worst-case small-x "
            "non-uniqueness below bT=2; the 10,000-epoch continuation remained non-plateaued and "
            "the spread increased to roughly 124%. Larger strengths do not improve this monotonically, "
            "and log-F lambda=10 approaches the fit-quality boundary."
        ),
        "production_state_modified": False,
    }
    TARGET.mkdir(parents=True, exist_ok=True)
    (TARGET / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
