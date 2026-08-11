#!/usr/bin/env python3
"""Audit FNP identifiability for the selected 12-basin smoothness candidate.

This script only reads isolated campaign outputs.  It records fit/stationarity
as eligibility/completion metadata, while the actual DNN stability result is
the cross-basin FNP distribution.  TMD projections are handled downstream as
consequences of that FNP distribution, not as DNN-identifiability gates.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
SEEDS = tuple(range(303, 315))
FIT_CEILING = 119.8021
EXISTING_307 = (
    "logcurv5em5_fslope4em3_xslope3em4_"
    "c2closure_b5p5_s1971_init307"
)


def tag(seed: int) -> str:
    return EXISTING_307 if seed == 307 else f"selected_local_xslope3em4_init{seed}"


def main() -> None:
    target = BASE / "summaries/selected_local_xslope3em4_multistart"
    target.mkdir(parents=True, exist_ok=True)
    run_rows: list[dict] = []
    grids: list[pd.DataFrame] = []
    predictions: list[pd.DataFrame] = []
    for source_seed in SEEDS:
        run = BASE / "outputs" / tag(source_seed)
        status_path = run / "fit_status.json"
        if not status_path.exists():
            raise RuntimeError(f"incomplete member: {status_path}")
        status = json.loads(status_path.read_text())
        final = status["final"]
        run_rows.append({
            "source_seed": source_seed,
            "run_tag": run.name,
            "fit_seed": status["seed"],
            "epochs_run": status["epochs_run"],
            "stopped_on_plateau": status["stopped_on_plateau"],
            "convergence_gate_pass": status["convergence_gate_pass"],
            "unpenalized_total_chi2": final["unpenalized_total_chi2"],
            "fit_gate_pass":
                final["unpenalized_total_chi2"] <= FIT_CEILING,
            "objective_per_row": final["objective_per_row"],
            "fnp_gradient_l2_per_row_objective":
                final["fnp_gradient_l2_per_row_objective"],
            "max_prediction_shift_over_experimental_sigma":
                final["max_prediction_shift_over_experimental_sigma"],
        })
        grids.append(
            pd.read_csv(run / "fnp_grid.csv").assign(
                source_seed=source_seed, run_tag=run.name))
        predictions.append(
            pd.read_csv(run / "accepted_predictions.csv")[
                ["row_id", "control_prediction", "sigma_used"]
            ].assign(source_seed=source_seed))

    runs = pd.DataFrame(run_rows)
    grid = pd.concat(grids, ignore_index=True)
    prediction = pd.concat(predictions, ignore_index=True)
    runs.to_csv(target / "runs.csv", index=False)

    wide = grid.pivot(
        index=["x", "bT"], columns="source_seed", values="F_NP")
    if wide.isna().any().any():
        raise RuntimeError("FNP grids are incomplete or inconsistent")
    values = wide.to_numpy(float)
    index = wide.index.to_frame(index=False)
    quantiles = np.quantile(values, [0.16, 0.50, 0.84], axis=1)
    bands = index.copy()
    bands["q16"], bands["median"], bands["q84"] = quantiles
    bands["minimum"] = np.min(values, axis=1)
    bands["maximum"] = np.max(values, axis=1)
    bands.to_csv(target / "fnp_bands.csv", index=False)

    def fnp_metrics(member_seeds: list[int], label: str) -> dict:
        selected = wide[member_seeds].to_numpy(float)
        selected_quantiles = np.quantile(
            selected, [0.16, 0.50, 0.84], axis=1)
        selected_median = selected_quantiles[1]
        active = selected_median > 0.05
        relative_range = (
            np.max(selected, axis=1) - np.min(selected, axis=1)
        ) / np.maximum(np.abs(selected_median), 1.0e-12)
        relative_68_width = (
            selected_quantiles[2] - selected_quantiles[0]
        ) / np.maximum(np.abs(selected_median), 1.0e-12)
        target_x = np.isclose(index["x"].to_numpy(float), 0.1)
        regional: dict[str, dict] = {}
        for limit in (1.0, 2.0, 3.0, 8.0):
            mask = active & target_x & (index["bT"].to_numpy(float) <= limit)
            regional[f"x0p1_bT_le_{limit:g}"] = {
                "maximum_relative_full_range": float(
                    np.max(relative_range[mask])),
                "maximum_relative_central68_width": float(
                    np.max(relative_68_width[mask])),
                "median_relative_central68_width": float(
                    np.median(relative_68_width[mask])),
            }
        selected_bands = index.copy()
        selected_bands["q16"], selected_bands["median"], selected_bands["q84"] = (
            selected_quantiles)
        selected_bands["minimum"] = np.min(selected, axis=1)
        selected_bands["maximum"] = np.max(selected, axis=1)
        selected_bands.to_csv(target / f"fnp_bands_{label}.csv", index=False)
        return {
            "member_count": len(member_seeds),
            "source_seeds": member_seeds,
            "max_relative_full_range_global_active": float(
                np.max(relative_range[active])),
            "p95_relative_full_range_global_active": float(
                np.quantile(relative_range[active], 0.95)),
            "target_slice": regional,
        }

    admissible_seeds = runs.loc[
        runs["fit_gate_pass"], "source_seed"].astype(int).tolist()
    fnp_all = fnp_metrics(list(SEEDS), "all_completed_starts")
    fnp_admissible = (
        fnp_metrics(admissible_seeds, "fit_admissible_starts")
        if len(admissible_seeds) >= 3 else {
            "member_count": len(admissible_seeds),
            "source_seeds": admissible_seeds,
            "status": "fewer_than_three_fit_admissible_starts",
        })

    pred_wide = prediction.pivot(
        index="row_id", columns="source_seed", values="control_prediction")
    sigma = (
        prediction.groupby("row_id")["sigma_used"].first()
        .reindex(pred_wide.index).to_numpy(float)
    )
    pred_range_sigma = (
        pred_wide.max(axis=1).to_numpy(float)
        - pred_wide.min(axis=1).to_numpy(float)
    ) / sigma

    summary = {
        "status": "isolated_selected_local_candidate_multistart_audit",
        "member_count": len(SEEDS),
        "source_seeds": list(SEEDS),
        "all_stopped_on_plateau": bool(runs["stopped_on_plateau"].all()),
        "all_runner_convergence_gates_pass": bool(
            runs["convergence_gate_pass"].all()),
        "fit_ceiling_unpenalized_total_chi2": FIT_CEILING,
        "fit_ceiling_pass_count": int(
            runs["fit_gate_pass"].sum()),
        "unpenalized_total_chi2": {
            "minimum": float(runs["unpenalized_total_chi2"].min()),
            "median": float(runs["unpenalized_total_chi2"].median()),
            "maximum": float(runs["unpenalized_total_chi2"].max()),
        },
        "fnp_gradient_l2_per_row_objective": {
            "minimum": float(
                runs["fnp_gradient_l2_per_row_objective"].min()),
            "median": float(
                runs["fnp_gradient_l2_per_row_objective"].median()),
            "maximum": float(
                runs["fnp_gradient_l2_per_row_objective"].max()),
        },
        "prediction_cross_basin_range_over_sigma": {
            "maximum": float(np.max(pred_range_sigma)),
            "p95": float(np.quantile(pred_range_sigma, 0.95)),
        },
        "primary_dnn_identifiability_measure": "cross-start FNP distribution",
        "fnp_distribution_all_completed_starts": fnp_all,
        "fnp_distribution_fit_admissible_starts": fnp_admissible,
        "interpretation": (
            "Fit quality is an eligibility gate and optimizer stationarity is "
            "completion metadata. DNN stability is assessed only from the "
            "cross-start FNP distribution. Downstream bT/kT TMD bands are "
            "projections of this distribution, not independent DNN gates. "
            "Cross-start quantiles are descriptive, not confidence intervals."),
        "production_sources_modified": False,
    }
    (target / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
