#!/usr/bin/env python3
"""Summarize the dimensionless log-FNP arc-length strength pilots."""

from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
TARGET = BASE / "summaries" / "loglength_strength_pilots"


def main() -> None:
    rows = []
    for filename in glob.glob(
            str(BASE / "outputs/loglength_*/fit_status.json")):
        path = Path(filename)
        status = json.loads(path.read_text())
        tag = path.parent.name
        candidate = tag.split("_lam")[0].removeprefix("loglength_")
        lam = float(
            status["regularization"]["logf_arc_length"]["lambda"])
        final = status["final"]
        baseline_values = []
        for seed in (701, 702, 703):
            baseline_status = json.loads((
                BASE / "outputs"
                / f"{candidate}_s{seed}_polish64/fit_status.json"
            ).read_text())["final"]
            baseline_values.append(float(
                baseline_status.get(
                    "unpenalized_total_chi2",
                    baseline_status["data_chi2"]
                    + baseline_status["norm_penalty"])))
        baseline = min(baseline_values)
        rows.append({
            "tag": tag,
            "candidate_id": candidate,
            "lambda_loglength": lam,
            "seed": int(status["seed"]),
            "unpenalized_chi2": float(
                final["unpenalized_total_chi2"]),
            "delta_unpenalized_chi2_from_best_polished": (
                float(final["unpenalized_total_chi2"]) - baseline),
            "arc_length_measure": (
                float(final["loglength_penalty_per_row_objective"])
                / lam),
            "penalty_per_row": float(
                final["loglength_penalty_per_row_objective"]),
            "fnp_gradient_l2": float(
                final["fnp_gradient_l2_per_row_objective"]),
            "norm_gradient_l2": float(
                final["normalization_gradient_l2_per_row_objective"]),
            "closure_evaluations": int(
                status["lbfgs"]["closure_evaluations"]),
            "stopped_on_adam_plateau": bool(
                status["stopped_on_plateau"]),
            "max_prediction_shift_over_sigma": float(
                final["max_prediction_shift_over_experimental_sigma"]),
        })
    frame = pd.DataFrame(rows)
    if len(frame):
        frame = frame.sort_values(
            ["candidate_id", "lambda_loglength"])
    TARGET.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TARGET / "pilot_metrics.csv", index=False)
    (TARGET / "summary.json").write_text(json.dumps({
        "status": "global_logF_arc_length_strength_pilots_not_production",
        "pilot_count": len(frame),
        "selection_rule": (
            "weakest lambda preserving unpenalized fit quality and "
            "passing subsequent multi-start and robustness gates"),
        "pilots": frame.to_dict(orient="records"),
        "production_sources_modified": False,
    }, indent=2) + "\n")

    if len(frame):
        fig, axes = plt.subplots(
            2, 1, figsize=(7.2, 6.2), sharex=True,
            constrained_layout=True)
        for candidate in sorted(frame["candidate_id"].unique()):
            selected = frame[
                frame["candidate_id"].eq(candidate)]
            axes[0].plot(
                selected["lambda_loglength"],
                selected[
                    "delta_unpenalized_chi2_from_best_polished"],
                marker="o", label=candidate)
            axes[1].plot(
                selected["lambda_loglength"],
                selected["arc_length_measure"],
                marker="o", label=candidate)
        axes[0].set_ylabel(r"unpenalized $\Delta\chi^2$")
        axes[1].set_ylabel("log-FNP arc-length measure")
        axes[1].set_xlabel(r"$\lambda_{\rm loglength}$")
        for axis in axes:
            axis.set_xscale("log")
            axis.grid(alpha=0.2)
            axis.legend(frameon=False)
        axes[1].set_yscale("log")
        fig.savefig(TARGET / "strength_tradeoff.png", dpi=220)
        fig.savefig(TARGET / "strength_tradeoff.pdf")
        plt.close(fig)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
