#!/usr/bin/env python3
"""Summarize log-length plus damping-rate-curvature pilots."""

from __future__ import annotations

import glob
import json
from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
TARGET = BASE / "summaries/loglength_ratecurv_combo"


def baseline(candidate: str) -> float:
    values = []
    for seed in (701, 702, 703):
        final = json.loads((
            BASE / "outputs"
            / f"{candidate}_s{seed}_polish64/fit_status.json"
        ).read_text())["final"]
        values.append(float(final.get(
            "unpenalized_total_chi2",
            final["data_chi2"] + final["norm_penalty"])))
    return min(values)


def main() -> None:
    rows = []
    for filename in glob.glob(
            str(BASE / "outputs/lengthrate_*/fit_status.json")):
        path = Path(filename)
        status = json.loads(path.read_text())
        tag = path.parent.name
        candidate = tag.removeprefix("lengthrate_").split("_llen")[0]
        regularization = status["regularization"]
        lambda_length = float(
            regularization["logf_arc_length"]["lambda"])
        lambda_rate = float(
            regularization["ratecurv"]["lambda"])
        final = status["final"]
        rows.append({
            "tag": tag,
            "candidate_id": candidate,
            "seed": int(status["seed"]),
            "lambda_loglength": lambda_length,
            "lambda_ratecurv": lambda_rate,
            "unpenalized_chi2": float(
                final["unpenalized_total_chi2"]),
            "delta_unpenalized_chi2_from_best_polished": (
                float(final["unpenalized_total_chi2"])
                - baseline(candidate)),
            "arc_length_measure": float(
                final["loglength_penalty_per_row_objective"])
                / lambda_length,
            "ratecurv_measure": float(
                final["ratecurv_penalty_per_row_objective"])
                / lambda_rate,
            "fnp_gradient_l2": float(
                final["fnp_gradient_l2_per_row_objective"]),
            "norm_gradient_l2": float(
                final["normalization_gradient_l2_per_row_objective"]),
            "closure_evaluations": int(
                status["lbfgs"]["closure_evaluations"]),
            "stopped_on_adam_plateau": bool(
                status["stopped_on_plateau"]),
        })
    frame = pd.DataFrame(rows)
    if len(frame):
        frame = frame.sort_values(
            ["candidate_id", "lambda_ratecurv", "seed"])
    TARGET.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TARGET / "pilot_metrics.csv", index=False)
    (TARGET / "summary.json").write_text(json.dumps({
        "status": (
            "isolated_loglength_plus_ratecurv_pilots_not_production"),
        "pilot_count": len(rows),
        "selection_rule": (
            "weakest rate-curvature strength passing fit, stationarity, "
            "multi-start, adjacent-strength, and transform gates"),
        "production_sources_modified": False,
        "pilots": rows,
    }, indent=2) + "\n")
    print(frame.to_string(index=False) if len(frame)
          else "No combined pilots complete yet.")


if __name__ == "__main__":
    main()
