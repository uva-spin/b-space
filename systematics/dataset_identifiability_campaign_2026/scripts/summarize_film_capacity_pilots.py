#!/usr/bin/env python3
"""Summarize isolated reduced-capacity FiLM pilots."""

from __future__ import annotations

import glob
import json
from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
TARGET = BASE / "summaries/film_capacity_pilots"


def main() -> None:
    rows = []
    initialization_diagnostics = []
    for filename in glob.glob(
            str(BASE / "outputs/filmcap_*/fit_status.json")):
        path = Path(filename)
        status = json.loads(path.read_text())
        tag = path.parent.name
        if int(status["epochs_run"]) <= 1:
            initialization_diagnostics.append({
                "tag": tag,
                "seed": int(status["seed"]),
                "unpenalized_chi2": float(
                    status["final"]["unpenalized_total_chi2"]),
                "max_prediction_shift_over_experimental_sigma": float(
                    status["final"][
                        "max_prediction_shift_over_experimental_sigma"]),
                "model_complexity": status["model_complexity"],
            })
            continue
        candidate = tag.removeprefix("filmcap_").split("_w")[0]
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
        complexity = status["model_complexity"]
        rows.append({
            "tag": tag,
            "candidate_id": candidate,
            "seed": int(status["seed"]),
            "np_width": int(complexity["np_width"]),
            "np_cond_width": int(complexity["np_cond_width"]),
            "np_blocks": int(complexity["np_blocks"]),
            "distill_steps": int(complexity["distill_accepted_steps"]),
            "unpenalized_chi2": float(
                final["unpenalized_total_chi2"]),
            "delta_unpenalized_chi2_from_best_polished": (
                float(final["unpenalized_total_chi2"]) - baseline),
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
        frame = frame.sort_values([
            "candidate_id", "np_width", "np_cond_width", "np_blocks"])
    TARGET.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TARGET / "pilot_metrics.csv", index=False)
    (TARGET / "summary.json").write_text(json.dumps({
        "status": "isolated_reduced_capacity_film_pilots_not_production",
        "pilot_count": len(frame),
        "selection_rule": (
            "smallest architecture preserving fit quality, followed by "
            "independent-start and continuation gates"),
        "pilots": frame.to_dict(orient="records"),
        "initialization_diagnostics_excluded_from_pilots": (
            initialization_diagnostics),
        "production_sources_modified": False,
    }, indent=2) + "\n")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
