#!/usr/bin/env python3
"""Compare isolated local starts of the accepted production-only objective."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "systematics/high_qt_direct_production_benchmark/experimental_unitary_transition"
PRODUCTION = ROOT / "systematics/collins_factorization_validity/outputs/rowidfix_stageFT_all_qmax0p20_lam0p50_central_s303"
TRAINER = ROOT / "v21_smoothedA_tail_candidate/train_bt_dnn_v21_smoothedA_tail.py"
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag-pattern", default="production_fnp_stability_control_s{seed}")
    parser.add_argument("--target-name", default="production_fnp_stability_control")
    args = parser.parse_args()
    tags = [args.tag_pattern.format(seed=seed) for seed in (303, 304, 305)]
    target = BASE / "summaries" / args.target_name
    trainer = load_module("production_control_analysis_trainer", TRAINER)
    refit = load_module("production_control_analysis_refit", BASE / "scripts/run_differentiable_fnp_refit.py")
    production_metrics = json.loads((PRODUCTION / "metrics.json").read_text())
    config = production_metrics["config"]
    x = torch.tensor(refit.FNP_ANCHOR_X, dtype=torch.float32)
    b = torch.linspace(0.0001, 8.0, 321, dtype=torch.float32)
    statuses, fnps, predictions, norms = [], [], [], []
    for tag in tags:
        path = BASE / "outputs" / tag
        statuses.append(json.loads((path / "fit_status.json").read_text()))
        if (path / "fnp_grid.csv").exists():
            fnps.append(pd.read_csv(path / "fnp_grid.csv").F_NP.to_numpy().reshape(len(x), len(b)))
        else:
            model = refit.make_model(trainer, config, torch.device("cpu"))
            state = torch.load(path / "model_state.pt", map_location="cpu", weights_only=True)
            model.load_state_dict({k.removeprefix("np_factor."): v for k, v in state.items()})
            with torch.no_grad():
                fnps.append(model(x, b).numpy())
        predictions.append(pd.read_csv(path / "accepted_predictions.csv").control_prediction.to_numpy())
        norms.append(pd.read_csv(path / "dataset_norms.csv").control_norm.to_numpy())

    fnp_stack, prediction_stack, norm_stack = np.stack(fnps), np.stack(predictions), np.stack(norms)
    fnp_mean = fnp_stack.mean(axis=0)
    relative_range = np.ptp(fnp_stack, axis=0) / np.maximum(fnp_mean, 1.0e-12)
    grid = pd.DataFrame({"x": np.repeat(x.numpy(), len(b)), "bT": np.tile(b.numpy(), len(x)),
                         "fnp_mean": fnp_mean.ravel(), "fnp_relative_range": relative_range.ravel()})
    for seed, values in zip((303, 304, 305), fnp_stack):
        grid[f"fnp_s{seed}"] = values.ravel()

    reference = pd.read_csv(BASE / "outputs" / tags[0] / "accepted_predictions.csv")
    comparison = reference[["dataset", "row_id", "sigma_used", "production_prediction"]].copy()
    comparison["control_prediction_mean"] = prediction_stack.mean(axis=0)
    comparison["control_prediction_range"] = np.ptp(prediction_stack, axis=0)
    comparison["range_over_experimental_sigma"] = comparison.control_prediction_range / comparison.sigma_used
    norm_reference = pd.read_csv(BASE / "outputs" / tags[0] / "dataset_norms.csv")
    norm_comparison = norm_reference[["dataset", "norm_width", "production_norm"]].copy()
    norm_comparison["control_norm_mean"] = norm_stack.mean(axis=0)
    norm_comparison["control_norm_range"] = np.ptp(norm_stack, axis=0)

    runs = pd.DataFrame([{
        "tag": tag, "seed": status["seed"], "perturbation": status["initial_relative_parameter_perturbation"],
        "converged": status["convergence_gate_pass"], "best_epoch": status["best_epoch"],
        "total_chi2": status["final"]["total_chi2"], "data_chi2": status["final"]["data_chi2"],
        "norm_penalty": status["final"]["norm_penalty"],
        "fnp_gradient_l2": status["final"]["fnp_gradient_l2_per_row_objective"],
    } for tag, status in zip(tags, statuses)])
    relevant = grid.fnp_mean > 0.05
    max_fnp = float(grid.loc[relevant, "fnp_relative_range"].max())
    summary = {
        "status": "experimental_production_objective_stability_control_not_production",
        "regularization": statuses[0].get("regularization"),
        "model_constraint": statuses[0].get("model_constraint", {"kind": "none"}),
        "source_production_modified": False, "run_count": 3,
        "all_runs_converged": bool(runs.converged.all()),
        "total_chi2_range": float(runs.total_chi2.max() - runs.total_chi2.min()),
        "max_prediction_range_over_experimental_sigma": float(comparison.range_over_experimental_sigma.max()),
        "p95_prediction_range_over_experimental_sigma": float(comparison.range_over_experimental_sigma.quantile(0.95)),
        "max_single_run_prediction_shift_from_accepted_production_over_experimental_sigma": float(max(
            status["final"]["max_prediction_shift_over_experimental_sigma"] for status in statuses)),
        "max_dataset_normalization_range": float(norm_comparison.control_norm_range.max()),
        "max_fnp_relative_range_where_fnp_gt_0p05": max_fnp,
        "p95_fnp_relative_range_where_fnp_gt_0p05": float(grid.loc[relevant, "fnp_relative_range"].quantile(0.95)),
        "regional_max_fnp_relative_ranges": {
            f"bT_le_{limit:g}": float(grid.loc[relevant & (grid.bT <= limit), "fnp_relative_range"].max())
            for limit in (1.0, 2.0, 3.0, 8.0)},
        "fnp_local_stability_threshold": 0.02,
        "fnp_local_stability_gate_pass": max_fnp < 0.02,
        "accepted_production_stopping_record": {
            "epochs_run": production_metrics["train"]["epochs_run"],
            "best_epoch": production_metrics["train"]["best_epoch"],
            "stopped_before_horizon": production_metrics["train"]["epochs_run"] < config["epochs"],
        },
        "prior_replica_studies": {
            "central_uniqueness_test": False,
            "reason": "the qmax0p20 50-replica campaign used fluctuated data, head-only training, 500 epochs, and a log-FNP anchor of strength 3",
        },
        "decision": "the production-only objective reproduces the prediction-stable but functionally non-identifiable behavior; the unitary transition did not create the core FNP uniqueness problem",
        "next_gate": "define a reduced identifiable FNP parameterization or an explicit prior before interpreting FNP replicas",
        "production_promotion_or_modification_authorized": False,
    }
    target.mkdir(parents=True, exist_ok=True)
    runs.to_csv(target / "runs.csv", index=False)
    grid.to_csv(target / "fnp_grid_comparison.csv", index=False)
    comparison.to_csv(target / "prediction_comparison.csv", index=False)
    norm_comparison.to_csv(target / "normalization_comparison.csv", index=False)
    (target / "campaign_status.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
