#!/usr/bin/env python3
"""Analyze the isolated three-start log-FNP-anchored refit campaign."""

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
def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strength", type=int, default=100)
    parser.add_argument("--tag-suffix", default="")
    args = parser.parse_args()
    tags = [f"unitary_smootherstep_v1_regularized_fnp_anchor{args.strength}_s{seed}{args.tag_suffix}" for seed in (303, 304, 305)]
    target = BASE / "summaries" / f"unitary_smootherstep_v1_regularized_fnp_anchor{args.strength}{args.tag_suffix}_convergence_campaign"
    trainer = load_module("regularized_campaign_trainer", TRAINER)
    refit = load_module("regularized_campaign_refit", BASE / "scripts/run_differentiable_fnp_refit.py")
    config = json.loads((PRODUCTION / "metrics.json").read_text())["config"]
    x = torch.tensor(refit.FNP_ANCHOR_X, dtype=torch.float32)
    b = torch.linspace(0.0001, 8.0, 321, dtype=torch.float32)
    statuses, fnps, accepted_predictions, boundary_predictions = [], [], [], []
    for tag in tags:
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
    fnp_rel_spread = np.ptp(fnp_stack, axis=0) / np.maximum(fnp_mean, 1.0e-12)
    grid = pd.DataFrame({
        "x": np.repeat(x.numpy(), len(b)), "bT": np.tile(b.numpy(), len(x)),
        "fnp_mean": fnp_mean.ravel(), "fnp_relative_range": fnp_rel_spread.ravel(),
    })
    for seed, values in zip((303, 304, 305), fnp_stack):
        grid[f"fnp_s{seed}"] = values.ravel()

    accepted_ref = pd.read_csv(BASE / "outputs" / tags[0] / "accepted_predictions.csv")
    boundary_ref = pd.read_csv(BASE / "outputs" / tags[0] / "boundary_predictions.csv")
    accepted_comparison = accepted_ref[["dataset", "row_id", "sigma_used"]].copy()
    accepted_comparison["prediction_mean"] = np.mean(accepted_stack, axis=0)
    accepted_comparison["prediction_range"] = np.ptp(accepted_stack, axis=0)
    accepted_comparison["range_over_experimental_sigma"] = accepted_comparison.prediction_range / accepted_ref.sigma_used
    boundary_comparison = boundary_ref[["dataset", "row_id", "error"]].copy()
    boundary_comparison["prediction_mean"] = np.mean(boundary_stack, axis=0)
    boundary_comparison["prediction_range"] = np.ptp(boundary_stack, axis=0)
    boundary_comparison["range_over_experimental_sigma"] = boundary_comparison.prediction_range / boundary_ref.error

    rows = []
    for tag, status in zip(tags, statuses):
        r = status["refit"]
        rows.append({
            "tag": tag, "seed": status["seed"], "perturbation": status["initial_relative_parameter_perturbation"],
            "converged": status["convergence_gate_pass"], "best_epoch": status["best_epoch"],
            "total_chi2": r["total_chi2"], "accepted_chi2": r["accepted_chi2"],
            "boundary_chi2": r["boundary_chi2"], "fnp_anchor_penalty": r["fnp_anchor_penalty"],
            "eta_match": r["matching_nuisance_sigma"], "eta_scale": r["nlo_scale_nuisance_sigma"],
        })
    runs = pd.DataFrame(rows)
    fnp_frozen = all(bool(status.get("fnp_frozen_during_polish", False)) for status in statuses)
    optimizer_converged = bool(runs.converged.all())
    relevant = grid.fnp_mean > 0.05
    masks = {f"bT_le_{limit:g}": grid.bT <= limit for limit in (1.0, 2.0, 3.0, 8.0)}
    regional_spreads = {
        name: float(grid.loc[relevant & mask, "fnp_relative_range"].max()) for name, mask in masks.items()
    }
    max_fnp_range = float(grid.loc[relevant, "fnp_relative_range"].max())
    fnp_stable = max_fnp_range < 0.02
    summary = {
        "status": "experimental_unitary_transition_not_production",
        "regularization": statuses[0]["regularization"],
        "run_count": len(tags), "common_epoch_horizon": statuses[0]["max_epochs"],
        "fit_scope": "frozen_fnp_nuisance_polish" if fnp_frozen else "joint_fnp_and_nuisance",
        "all_runs_converged": bool(optimizer_converged and not fnp_frozen),
        "all_runs_conditionally_converged": bool(optimizer_converged and fnp_frozen),
        "total_chi2_range": float(runs.total_chi2.max() - runs.total_chi2.min()),
        "boundary_chi2_range": float(runs.boundary_chi2.max() - runs.boundary_chi2.min()),
        "max_accepted_prediction_range_over_experimental_sigma": float(accepted_comparison.range_over_experimental_sigma.max()),
        "p95_accepted_prediction_range_over_experimental_sigma": float(accepted_comparison.range_over_experimental_sigma.quantile(0.95)),
        "max_boundary_prediction_range_over_experimental_sigma": float(boundary_comparison.range_over_experimental_sigma.max()),
        "max_fnp_relative_range_where_fnp_gt_0p05": max_fnp_range,
        "p95_fnp_relative_range_where_fnp_gt_0p05": float(grid.loc[relevant, "fnp_relative_range"].quantile(0.95)),
        "regional_max_fnp_relative_ranges": regional_spreads,
        "fnp_local_stability_threshold": 0.02,
        "fnp_local_stability_gate_pass": fnp_stable,
        "replica_stability_authorized": False,
        "decision": (
            "conditional frozen-FNP nuisance convergence and functional stability pass; joint FNP convergence remains unestablished and replicas remain unauthorized"
            if fnp_stable and optimizer_converged and fnp_frozen else
            "functional local stability passes, but joint convergence does not; replicas remain unauthorized"
            if fnp_stable and not optimizer_converged else
            "functional local stability and/or convergence does not pass; replicas remain unauthorized"
        ),
        "next_gate": (
            "decide whether replicas are scientifically meaningful with FNP fixed or require a reduced identifiable FNP parameterization"
            if fnp_stable and optimizer_converged and fnp_frozen else
            "resolve or justify the persistent fixed-horizon convergence behavior before replicas"
        ),
    }
    target.mkdir(parents=True, exist_ok=True)
    runs.to_csv(target / "runs.csv", index=False)
    grid.to_csv(target / "fnp_grid_comparison.csv", index=False)
    accepted_comparison.to_csv(target / "accepted_prediction_comparison.csv", index=False)
    boundary_comparison.to_csv(target / "boundary_prediction_comparison.csv", index=False)
    (target / "campaign_status.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
