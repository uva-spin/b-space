#!/usr/bin/env python3
"""Summarize seed-count and training-horizon completeness tests for Fig. 6."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "systematics/high_qt_direct_production_benchmark/experimental_unitary_transition"
CENTRAL_BSPACE = (
    ROOT / "systematics/collins_factorization_validity/plots/"
    "rowidfix_stageFT_all_qmax0p20_lam0p50_central_exactx/"
    "v22_scheme_tmd_bspace_long.csv"
)
TRANSFORMER = ROOT / "workflows/v23a/construction/construct_v23a_regularized_kspace_tmd_v2.py"
FIG6_SUMMARY = BASE / "summaries/fig6_updated_ud_band/summary.json"
TARGET = BASE / "summaries/fig6_completeness_validation"
CONTINUED_SEEDS = (304, 307, 313)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def transform_output_pattern(flavor: str, tag_pattern: str) -> tuple[np.ndarray, np.ndarray]:
    transform = load_module(f"fig6_continuation_transform_{flavor}", TRANSFORMER)
    all_b = pd.read_csv(CENTRAL_BSPACE)
    central = all_b[
        np.isclose(all_b["x"], 0.1)
        & np.isclose(all_b["Q"], 10.0)
        & all_b["flavor"].astype(str).eq(flavor)
    ].sort_values("bT").copy()
    b = central["bT"].to_numpy(float)
    perturbative = central["ftilde_no_np"].to_numpy(float)
    curves = []
    for seed in CONTINUED_SEEDS:
        grid = pd.read_csv(BASE / "outputs" / tag_pattern.format(seed=seed) / "fnp_grid.csv")
        grid = grid[np.isclose(grid["x"], 0.1)].sort_values("bT")
        trial = central.copy()
        trial["F_NP"] = np.interp(b, grid["bT"], grid["F_NP"])
        trial["ftilde"] = perturbative * trial["F_NP"].to_numpy(float)
        trial["seed"] = seed
        trial["pdf_member"] = 0
        trial["_replica_key"] = f"continued_s{seed}|pdf0"
        curves.append(trial)
    args = argparse.Namespace(
        quantities=["ftilde"], tail_mode="expb2", tail_fit_bmin=None,
        eps=1.0e-300, b_transform_max=24.0, n_b_transform=6001,
        k_max=4.0, n_k=401, end_taper_start_fraction=0.92,
    )
    long, _ = transform.transform_curves(pd.concat(curves, ignore_index=True), args)
    wide = long.pivot(index="kT", columns="seed", values="value").sort_index()
    return wide.index.to_numpy(float), wide.to_numpy(float).T


def main() -> None:
    plot_module = load_module(
        "fig6_updated_band_for_validation", BASE / "scripts/plot_fig6_updated_ud_band.py"
    )
    fig6_summary = json.loads(FIG6_SUMMARY.read_text())
    horizon = {}
    for flavor in ("u", "d"):
        k, all_20k, _ = plot_module.transformed_starts(flavor)
        indices = [plot_module.SEEDS.index(seed) for seed in CONTINUED_SEEDS]
        selected_20k = all_20k[indices]
        k_cont, selected_40k = transform_output_pattern(flavor, "fig6_newcenter_cont40k_s{seed}")
        if not np.allclose(k, k_cont):
            raise RuntimeError("continuation kT grid mismatch")
        reference = np.median(selected_20k, axis=0)
        active = (k <= 2.25) & (reference > 0.05 * np.max(reference[k <= 2.25]))
        per_seed = {}
        for seed, before, after in zip(CONTINUED_SEEDS, selected_20k, selected_40k):
            per_seed[str(seed)] = float(np.max(np.abs(after[active] - before[active]) / reference[active]))
        horizon[flavor] = {
            "max_individual_curve_drift_relative_active": per_seed,
            "max_three_seed_median_drift_relative_active": float(np.max(
                np.abs(np.median(selected_40k, axis=0)[active] - reference[active]) / reference[active]
            )),
        }

    polishing = {}
    for flavor in ("u", "d"):
        k_120k, curves_120k = transform_output_pattern(
            flavor, "fig6_longplateau_unanchored_s{seed}"
        )
        k_polish, curves_polish = transform_output_pattern(
            flavor, "fig6_longplateau_polish_s{seed}"
        )
        if not np.allclose(k_120k, k_polish):
            raise RuntimeError("polishing kT grid mismatch")
        reference = np.median(curves_120k, axis=0)
        active = (k_120k <= 2.25) & (
            reference > 0.05 * np.max(reference[k_120k <= 2.25])
        )
        polishing[flavor] = {
            "max_individual_curve_drift_relative_active": {
                str(seed): float(np.max(np.abs(after[active] - before[active]) / reference[active]))
                for seed, before, after in zip(CONTINUED_SEEDS, curves_120k, curves_polish)
            },
            "max_three_seed_median_drift_relative_active": float(np.max(
                np.abs(np.median(curves_polish, axis=0)[active] - reference[active])
                / reference[active]
            )),
        }

    statuses = {}
    for seed in CONTINUED_SEEDS:
        status = json.loads(
            (BASE / f"outputs/fig6_newcenter_cont40k_s{seed}/fit_status.json").read_text()
        )
        statuses[str(seed)] = {
            "stopped_on_plateau": status["stopped_on_plateau"],
            "best_epoch_in_continuation": status["best_epoch"],
            "objective_per_row": status["final"]["objective_per_row"],
            "fnp_gradient_l2_per_row_objective": status["final"]["fnp_gradient_l2_per_row_objective"],
        }

    summary = {
        "status": "fig6_combined_band_completeness_gate_failed_not_production",
        "central_start_count": 12,
        "all_20k_runs_plateaued": False,
        "seed_count_validation": {
            flavor: {
                "split_six_start_stability": fig6_summary["flavors"][flavor]["split_six_start_stability"],
                "seed_bootstrap_68pct_numerical_uncertainty":
                    fig6_summary["flavors"][flavor]["seed_bootstrap_68pct_numerical_uncertainty"],
            }
            for flavor in ("u", "d")
        },
        "training_horizon_validation_20k_to_40k": horizon,
        "low_lr_polishing_validation_120k_plus_40k": polishing,
        "continuation_fit_status": statuses,
        "multi_start_replica_validation_launched": False,
        "multi_start_replica_reason": (
            "predeclared central-ensemble gate failed: seed-split band edges and continued "
            "central curves are unstable, so replica-by-start fits would not validate a fixed target distribution"
        ),
        "decision": (
            "the current 12-start crossed band is a larger diagnostic band, not a complete 1-sigma interval"
        ),
        "production_sources_modified": False,
    }
    TARGET.mkdir(parents=True, exist_ok=True)
    (TARGET / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
