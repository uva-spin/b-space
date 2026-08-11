#!/usr/bin/env python3
"""Compare fixed-horizon local starts for the differentiable FNP refit."""

from __future__ import annotations

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
TAGS = [
    "unitary_smootherstep_v1_differentiable_fnp_refit_central_converged_s303",
    "unitary_smootherstep_v1_differentiable_fnp_refit_central_converged_s304",
    "unitary_smootherstep_v1_differentiable_fnp_refit_central_converged_s305",
]
TARGET = BASE / "summaries/unitary_smootherstep_v1_differentiable_fnp_convergence_campaign"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    trainer = load_module("campaign_trainer", TRAINER)
    refit = load_module("campaign_refit", BASE / "scripts/run_differentiable_fnp_refit.py")
    config = json.loads((PRODUCTION / "metrics.json").read_text())["config"]
    x = torch.tensor([0.001, 0.003, 0.01, 0.03, 0.1, 0.2, 0.4, 0.7], dtype=torch.float32)
    b = torch.linspace(0.0001, 8.0, 321, dtype=torch.float32)
    statuses, fnps, accepted_predictions, boundary_predictions = [], [], [], []
    for tag in TAGS:
        path = BASE / "outputs" / tag
        status = json.loads((path / "fit_status.json").read_text())
        statuses.append(status)
        model = refit.make_model(trainer, config, torch.device("cpu"))
        state = torch.load(path / "model_state.pt", map_location="cpu", weights_only=True)
        model.load_state_dict({k.removeprefix("np_factor."): v for k, v in state.items()})
        with torch.no_grad():
            fnps.append(model(x, b).numpy())
        accepted_predictions.append(pd.read_csv(path / "accepted_predictions.csv").refit_prediction.to_numpy())
        boundary_predictions.append(pd.read_csv(path / "boundary_predictions.csv").refit_prediction.to_numpy())

    fnp_stack = np.stack(fnps)
    accepted_stack = np.stack(accepted_predictions)
    boundary_stack = np.stack(boundary_predictions)
    fnp_mean = np.mean(fnp_stack, axis=0)
    fnp_rel_spread = (np.max(fnp_stack, axis=0) - np.min(fnp_stack, axis=0)) / np.maximum(fnp_mean, 1e-12)
    grid = pd.DataFrame({
        "x": np.repeat(x.numpy(), len(b)), "bT": np.tile(b.numpy(), len(x)),
        "fnp_mean": fnp_mean.ravel(), "fnp_relative_range": fnp_rel_spread.ravel(),
    })
    for i, tag in enumerate(TAGS):
        grid[f"fnp_{tag[-4:]}"] = fnp_stack[i].ravel()

    accepted_ref = pd.read_csv(BASE / "outputs" / TAGS[0] / "accepted_predictions.csv")
    boundary_ref = pd.read_csv(BASE / "outputs" / TAGS[0] / "boundary_predictions.csv")
    accepted_abs_range = np.ptp(accepted_stack, axis=0)
    boundary_abs_range = np.ptp(boundary_stack, axis=0)
    accepted_comparison = accepted_ref[["dataset", "row_id", "sigma_used"]].copy()
    accepted_comparison["prediction_mean"] = np.mean(accepted_stack, axis=0)
    accepted_comparison["prediction_range"] = accepted_abs_range
    accepted_comparison["range_over_experimental_sigma"] = accepted_abs_range / accepted_ref.sigma_used.to_numpy()
    boundary_comparison = boundary_ref[["dataset", "row_id", "error"]].copy()
    boundary_comparison["prediction_mean"] = np.mean(boundary_stack, axis=0)
    boundary_comparison["prediction_range"] = boundary_abs_range
    boundary_comparison["range_over_experimental_sigma"] = boundary_abs_range / boundary_ref.error.to_numpy()

    rows = []
    for tag, status in zip(TAGS, statuses):
        r = status["refit"]
        rows.append({"tag": tag, "seed": status["seed"], "perturbation": status["initial_relative_parameter_perturbation"],
                     "converged": status["convergence_gate_pass"], "total_chi2": r["total_chi2"],
                     "accepted_chi2": r["accepted_chi2"], "boundary_chi2": r["boundary_chi2"],
                     "eta_match": r["matching_nuisance_sigma"], "eta_scale": r["nlo_scale_nuisance_sigma"]})
    runs = pd.DataFrame(rows)
    relevant = grid.fnp_mean > 0.05
    summary = {
        "status": "experimental_unitary_transition_not_production",
        "run_count": len(TAGS), "common_epoch_horizon": 20000,
        "all_runs_converged": bool(runs.converged.all()),
        "total_chi2_range": float(runs.total_chi2.max() - runs.total_chi2.min()),
        "boundary_chi2_range": float(runs.boundary_chi2.max() - runs.boundary_chi2.min()),
        "max_accepted_prediction_range_over_experimental_sigma": float(accepted_comparison.range_over_experimental_sigma.max()),
        "p95_accepted_prediction_range_over_experimental_sigma": float(accepted_comparison.range_over_experimental_sigma.quantile(0.95)),
        "max_boundary_prediction_range_over_experimental_sigma": float(boundary_comparison.range_over_experimental_sigma.max()),
        "max_fnp_relative_range_where_fnp_gt_0p05": float(grid.loc[relevant, "fnp_relative_range"].max()),
        "p95_fnp_relative_range_where_fnp_gt_0p05": float(grid.loc[relevant, "fnp_relative_range"].quantile(0.95)),
        "replica_stability_authorized": False,
        "decision": "prediction-level local stability is measured, but the convergence gate must pass before replicas",
        "next_gate": "add a controlled regularization or freeze criterion; do not extend unconstrained training indefinitely",
    }
    TARGET.mkdir(parents=True, exist_ok=True)
    runs.to_csv(TARGET / "runs.csv", index=False)
    grid.to_csv(TARGET / "fnp_grid_comparison.csv", index=False)
    accepted_comparison.to_csv(TARGET / "accepted_prediction_comparison.csv", index=False)
    boundary_comparison.to_csv(TARGET / "boundary_prediction_comparison.csv", index=False)
    (TARGET / "campaign_status.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
