#!/usr/bin/env python3
"""Summarize the global log-FNP-curvature strength pilots."""

from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
TARGET = BASE / "summaries" / "logcurv_strength_pilots"


def main() -> None:
    rows = []
    for filename in glob.glob(str(BASE / "outputs/logcurv_*/fit_status.json")):
        status = json.loads(Path(filename).read_text())
        tag = Path(filename).parent.name
        if tag.endswith("_smoke"):
            continue
        candidate = tag.split("_lam")[0].removeprefix("logcurv_")
        regularizer = status["regularization"]["logf_curvature"]
        lam = float(regularizer["lambda"])
        final = status["final"]
        baseline = min(
            json.loads((
                BASE / "outputs" / f"{candidate}_s{seed}_polish64/fit_status.json"
            ).read_text())["final"]["data_chi2"]
            + json.loads((
                BASE / "outputs" / f"{candidate}_s{seed}_polish64/fit_status.json"
            ).read_text())["final"]["norm_penalty"]
            for seed in (701, 702, 703)
        )
        grid = pd.read_csv(Path(filename).parent / "fnp_grid.csv")
        monotonic_violations = 0
        for _, group in grid.groupby("x"):
            monotonic_violations += int(np.count_nonzero(
                np.diff(group.sort_values("bT")["F_NP"].to_numpy(float)) > 1.0e-8))
        rows.append({
            "tag": tag,
            "candidate_id": candidate,
            "lambda_logcurv": lam,
            "seed": int(status["seed"]),
            "unpenalized_chi2": float(final["unpenalized_total_chi2"]),
            "delta_unpenalized_chi2_from_best_polished": (
                float(final["unpenalized_total_chi2"]) - baseline),
            "roughness_measure": (
                float(final["logcurv_penalty_per_row_objective"]) / lam),
            "penalty_per_row": float(final["logcurv_penalty_per_row_objective"]),
            "fnp_gradient_l2": float(final["fnp_gradient_l2_per_row_objective"]),
            "norm_gradient_l2": float(
                final["normalization_gradient_l2_per_row_objective"]),
            "closure_evaluations": int(status["lbfgs"]["closure_evaluations"]),
            "stopped_on_adam_plateau": bool(status["stopped_on_plateau"]),
            "max_prediction_shift_over_sigma": float(
                final["max_prediction_shift_over_experimental_sigma"]),
            "fnp_monotonicity_violations_on_output_grid": monotonic_violations,
        })
    frame = pd.DataFrame(rows).sort_values(["candidate_id", "lambda_logcurv"])
    TARGET.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TARGET / "pilot_metrics.csv", index=False)
    (TARGET / "summary.json").write_text(json.dumps({
        "status": "global_logF_curvature_strength_pilots_not_production",
        "pilot_count": len(frame),
        "selection_rule": "weakest lambda preserving unregularized fit quality and passing subsequent three-start stationarity/stability gates",
        "pilots": frame.to_dict(orient="records"),
        "production_sources_modified": False,
    }, indent=2) + "\n")

    if len(frame):
        candidates = sorted(frame["candidate_id"].unique())
        fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.2), sharex=True,
                                 constrained_layout=True)
        for candidate in candidates:
            selected = frame[frame["candidate_id"].eq(candidate)]
            axes[0].plot(selected["lambda_logcurv"],
                         selected["delta_unpenalized_chi2_from_best_polished"],
                         marker="o", label=candidate)
            axes[1].plot(selected["lambda_logcurv"],
                         selected["roughness_measure"], marker="o", label=candidate)
        axes[0].set_ylabel(r"unpenalized $\Delta\chi^2$")
        axes[1].set_ylabel("log-FNP curvature measure")
        axes[1].set_xlabel(r"$\lambda_{\rm logcurv}$")
        for ax in axes:
            ax.set_xscale("log")
            ax.grid(alpha=0.2)
            ax.legend(frameon=False)
        axes[1].set_yscale("log")
        fig.savefig(TARGET / "strength_tradeoff.png", dpi=220)
        fig.savefig(TARGET / "strength_tradeoff.pdf")
        plt.close(fig)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
