#!/usr/bin/env python3
"""Adaptively find the weakest fit barrier for the selected FNP prior."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
SYSTEMATICS = BASE.parent
PYTHON = Path("/home/dustin/miniforge3/envs/pdf-fit/bin/python")
RUNNER = (
    SYSTEMATICS
    / "high_qt_direct_production_benchmark/experimental_unitary_transition/"
    "scripts/run_production_fnp_stability_control.py"
)
SOURCE = (
    SYSTEMATICS
    / "collins_factorization_validity/outputs/"
    "rowidfix_stageFT_E772_qmax0p20_lam0p50_central_s303"
)
W_GRID = (
    SYSTEMATICS.parent
    / "outputs/v23a_tevatron_plus_lhcb7_fidacc_merged_cache/backend_cache/"
    "wpert_v23a_tevatron_plus_lhcb7_fidacc_b160.csv"
)
SOURCE_SEEDS = (304, 312, 313)
STRENGTHS = (1.0, 10.0, 100.0)
CEILING = 119.8021


def strength_tag(value: float) -> str:
    return {1.0: "mu1", 10.0: "mu10", 100.0: "mu100"}[value]


def tag(strength: float, source_seed: int) -> str:
    return (
        f"selected_local_xslope3em4_barrier_{strength_tag(strength)}_"
        f"init{source_seed}"
    )


def command(strength: float, source_seed: int) -> list[str]:
    initial = (
        BASE / "outputs"
        / f"independent_datafit_D020_E772_init{source_seed}"
    )
    return [
        str(PYTHON), str(RUNNER),
        "--source-production", str(SOURCE),
        "--w-grid", str(W_GRID),
        "--output-root", str(BASE / "outputs"),
        "--initial-state", str(initial / "model_state.pt"),
        "--initial-norms", str(initial / "dataset_norms.csv"),
        "--seed", str(3000 + source_seed + int(strength)),
        "--initial-perturbation", "0",
        "--max-epochs", "20000",
        "--min-epochs", "5000",
        "--plateau-patience", "3000",
        "--learning-rate", "1e-5",
        "--lbfgs-max-iter", "15000",
        "--float64",
        "--lambda-fnp-logcurv", "5e-5",
        "--fnp-logcurv-bmin", "0.10",
        "--fnp-logcurv-bmax", "3",
        "--lambda-fnp-f-slope", "0.004",
        "--fnp-f-slope-bmin", "0.10",
        "--fnp-f-slope-bmax", "3",
        "--lambda-fnp-x-slope", "3e-4",
        "--fnp-x-slope-bmin", "0.10",
        "--fnp-x-slope-bmax", "3",
        "--closure-tail-coordinate",
        "--closure-tail-b-start", "5.5",
        "--closure-tail-b-end", "8",
        "--lambda-fnp-transform-closure", "0.001",
        "--fnp-transform-closure-bmin", "8",
        "--fnp-transform-closure-max", "0.0001",
        "--fit-quality-ceiling-total-chi2", str(CEILING),
        "--lambda-fit-quality-barrier", str(strength),
        "--fit-quality-barrier-power", "2",
        "--tag", tag(strength, source_seed),
    ]


def audit_strength(strength: float) -> dict:
    statuses = []
    grids = []
    for source_seed in SOURCE_SEEDS:
        run = BASE / "outputs" / tag(strength, source_seed)
        status = json.loads((run / "fit_status.json").read_text())
        statuses.append((source_seed, status))
        grids.append(
            pd.read_csv(run / "fnp_grid.csv").assign(
                source_seed=source_seed))
    frame = pd.concat(grids, ignore_index=True)
    frame = frame[
        np.isclose(frame["x"], 0.1) & (frame["bT"] <= 2.0)
    ]
    wide = frame.pivot(
        index="bT", columns="source_seed", values="F_NP")
    values = wide.to_numpy(float)
    median = np.median(values, axis=1)
    active = median > 0.05
    relative_range = (
        np.max(values, axis=1) - np.min(values, axis=1)
    ) / np.maximum(np.abs(median), 1.0e-12)
    rows = [{
        "source_seed": source_seed,
        "run_tag": tag(strength, source_seed),
        "unpenalized_total_chi2":
            status["final"]["unpenalized_total_chi2"],
        "fit_gate_pass":
            status["final"]["unpenalized_total_chi2"] <= CEILING,
    } for source_seed, status in statuses]
    return {
        "strength": strength,
        "runs": rows,
        "all_fit_gate_pass": all(row["fit_gate_pass"] for row in rows),
        "x0p1_bT_le_2_max_relative_fnp_full_range": float(
            np.max(relative_range[active])),
    }


def main() -> None:
    log_dir = BASE / "logs/selected_local_fit_barrier"
    target = BASE / "summaries/selected_local_fit_barrier_ladder"
    log_dir.mkdir(parents=True, exist_ok=True)
    target.mkdir(parents=True, exist_ok=True)
    audits = []
    selected = None
    for strength in STRENGTHS:
        for source_seed in SOURCE_SEEDS:
            run = BASE / "outputs" / tag(strength, source_seed)
            if (run / "fit_status.json").exists():
                continue
            with (log_dir / f"{tag(strength, source_seed)}.log").open("w") as stream:
                subprocess.run(
                    command(strength, source_seed), stdout=stream,
                    stderr=subprocess.STDOUT, text=True, check=True)
        audit = audit_strength(strength)
        audits.append(audit)
        (target / "progress.json").write_text(
            json.dumps(audits, indent=2) + "\n")
        if audit["all_fit_gate_pass"]:
            selected = strength
            break
    summary = {
        "status": "isolated_selected_local_fit_barrier_ladder_complete",
        "source_seeds": list(SOURCE_SEEDS),
        "fit_quality_ceiling_total_chi2": CEILING,
        "strength_audits": audits,
        "selected_weakest_barrier_strength": selected,
        "primary_measure": "cross-start FNP distribution",
        "production_sources_modified": False,
    }
    (target / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (log_dir / "COMPLETE").write_text("complete\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
