#!/usr/bin/env python3
"""Test whether lambda=637.5 plus the selected fit barrier is still viable.

The unbarriered lambda=637.5 full ensemble failed on the primary optimizer
trajectory for pseudo-data replica 1033.  Its initial pseudo-data fit was far
outside the later-selected one-sided fit barrier, so that failure alone cannot
exclude the combined lambda=637.5+mu100 prescription.  Rebuild the same
primary trajectory with the barrier from its stationary lambda=637.5 central.

The old trajectory reversed through 200k iterations.  Consequently 200k is
mandatory exposure, followed by two wholly subsequent <=2% FNP-drift blocks.
A terminal drift >=1% triggers two additional wholly fresh confirmations.  A
failure rejects the lower reference weight.  A pass deliberately prevents
lambda=675 finalization: the lower prescription would then require full24 and
full-replica coverage rather than being silently skipped.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
SYSTEMATICS = BASE.parent
ROOT = SYSTEMATICS.parent
RUNNER = (SYSTEMATICS /
          "high_qt_direct_production_benchmark/experimental_unitary_transition/"
          "scripts/run_production_fnp_stability_control.py")
SOURCE = (SYSTEMATICS /
          "collins_factorization_validity/outputs/"
          "rowidfix_stageFT_all_qmax0p20_lam0p50_central_s303")
W_GRID = (ROOT / "outputs/v23a_tevatron_plus_lhcb7_fidacc_merged_cache/"
          "backend_cache/wpert_v23a_tevatron_plus_lhcb7_fidacc_b160.csv")
REFERENCE = BASE / "summaries/exact_baseline_fnp_median/fnp_median.csv"
INITIAL = BASE / "outputs/fullref_lam637p5_b4_central_polish64_50000"
TARGET = BASE / "summaries/lambda637p5_fitbar_minimum_control"
PYTHON = Path("/home/dustin/miniforge3/envs/pdf-fit/bin/python")
STRENGTH = 637.5
BARRIER_STRENGTH = 100.0
BARRIER_POWER = 2
REPLICA_SEED = 1033
FIT_SEED = 2033
MANDATORY_ITERATIONS = 200_000
MAXIMUM_ITERATIONS = 250_000
BLOCK = 5_000
GATE = 0.02
SENSITIVITY_TRIGGER = 0.01


def vector(run: Path) -> np.ndarray:
    frame = pd.read_csv(run / "fnp_grid.csv")
    mask = (np.isclose(frame.x, 0.1)
            & frame.bT.between(0.1, 4.0, inclusive="both"))
    values = frame.loc[mask, "F_NP"].to_numpy(float)
    if len(values) < 3 or not np.all(np.isfinite(values)):
        raise RuntimeError(f"invalid selected-domain FNP grid: {run}")
    return values


def write(status: str, records: list[dict], endpoint: Path | None) -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    (TARGET / "runs.csv").write_text(pd.DataFrame(records).to_csv(index=False))
    summary = {
        "status": status,
        "tested_reference_strength": STRENGTH,
        "fit_quality_barrier_strength": BARRIER_STRENGTH,
        "fit_quality_barrier_power": BARRIER_POWER,
        "replica_seed": REPLICA_SEED,
        "fit_seed": FIT_SEED,
        "initial_central_tag": INITIAL.name,
        "mandatory_same_objective_iterations": MANDATORY_ITERATIONS,
        "maximum_same_objective_iterations": MAXIMUM_ITERATIONS,
        "stationarity_gate": GATE,
        "sensitivity_trigger": SENSITIVITY_TRIGGER,
        "endpoint_tag": endpoint.name if endpoint is not None else None,
        "selection_semantics": (
            "failure rejects lambda637.5+power2-mu100; survival requires a "
            "new full24/full-replica campaign before lambda675 can be called "
            "the minimum passing reference strength"
        ),
        "production_sources_modified": False,
    }
    (TARGET / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    if not (INITIAL / "fit_status.json").exists():
        raise RuntimeError("stationary lambda637.5 central is missing")
    initial_status = json.loads((INITIAL / "fit_status.json").read_text())
    row_count = int(initial_status["row_count"])
    ceiling = row_count + 5.0 * math.sqrt(2.0 * row_count)
    previous = INITIAL
    records: list[dict] = []
    consecutive = 0
    sensitivity_active = False
    fresh_after_sensitivity = 0
    mandatory_anchor: np.ndarray | None = None
    passed = False
    write("running", records, None)
    for cumulative in range(BLOCK, MAXIMUM_ITERATIONS + BLOCK, BLOCK):
        tag = (f"lowerctl_lam637p5_fitbar_p2_mu100_r1033_"
               f"polish64_{cumulative}")
        target = BASE / "outputs" / tag
        if not (target / "fit_status.json").exists():
            command = [
                str(PYTHON), str(RUNNER), "--seed", str(FIT_SEED),
                "--source-production", str(SOURCE), "--w-grid", str(W_GRID),
                "--output-root", str(BASE / "outputs"), "--tag", tag,
                "--initial-state", str(previous / "model_state.pt"),
                "--initial-norms", str(previous / "dataset_norms.csv"),
                "--max-epochs", "0", "--min-epochs", "0",
                "--plateau-patience", "0", "--lbfgs-max-iter", str(BLOCK),
                "--float64", "--replica-seed", str(REPLICA_SEED),
                "--lambda-fnp-reference-distance", str(STRENGTH),
                "--fnp-reference-distance-csv", str(REFERENCE),
                "--fnp-reference-distance-bmin", "0.10",
                "--fnp-reference-distance-bmax", "4.0",
                "--fit-quality-ceiling-total-chi2", str(ceiling),
                "--lambda-fit-quality-barrier", str(BARRIER_STRENGTH),
                "--fit-quality-barrier-power", str(BARRIER_POWER),
            ]
            log = BASE / "logs" / f"{tag}.log"
            with log.open("w") as stream:
                subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                               check=True)
        status = json.loads((target / "fit_status.json").read_text())
        old, new = vector(previous), vector(target)
        drift = float(np.max(np.abs(new - old) / np.maximum(old, 0.05)))
        fit_value = float(status["final"]["unpenalized_total_chi2"])
        fit_pass = fit_value <= ceiling
        eligible = cumulative > MANDATORY_ITERATIONS
        if cumulative == MANDATORY_ITERATIONS:
            mandatory_anchor = vector(target)
        window_drift = (float(np.max(np.abs(vector(target) - mandatory_anchor) /
                                    np.maximum(mandatory_anchor, 0.05)))
                        if eligible and mandatory_anchor is not None else None)
        window_quiet = window_drift is not None and window_drift <= GATE
        quiet = drift <= GATE
        consecutive = consecutive + 1 if eligible and quiet else 0
        if sensitivity_active:
            fresh_after_sensitivity = (
                fresh_after_sensitivity + 1 if eligible and quiet else 0
            )
            passed = fresh_after_sensitivity >= 2 and fit_pass and window_quiet
        elif eligible and consecutive >= 2 and fit_pass:
            if drift >= SENSITIVITY_TRIGGER:
                sensitivity_active = True
                fresh_after_sensitivity = 0
            else:
                passed = window_quiet
        records.append({
            "cumulative_lbfgs_iterations": cumulative,
            "fnp_drift_from_previous_chunk": drift,
            "eligible_post_200k_confirmation": eligible,
            "passes_drift_2pct": quiet,
            "post_mandatory_window_fnp_drift": window_drift,
            "passes_post_mandatory_window_drift_2pct": window_quiet,
            "consecutive_post_200k_quiet_blocks": consecutive,
            "sensitivity_confirmation_triggered": sensitivity_active,
            "fresh_quiet_blocks_after_sensitivity_trigger":
                fresh_after_sensitivity,
            "unpenalized_total_chi2": fit_value,
            "fit_quality_ceiling_total_chi2": ceiling,
            "fit_gate_pass": fit_pass,
            "stationarity_and_fit_pass": passed,
            "tag": tag,
        })
        write("running", records, target)
        previous = target
        if passed:
            write("lower_candidate_survives_discriminator", records, target)
            raise RuntimeError(
                "lambda637.5+mu100 survived its decisive control; run full "
                "coverage before finalizing lambda675"
            )
        remaining = (MAXIMUM_ITERATIONS - cumulative) // BLOCK
        confirmations_needed = (2 - fresh_after_sensitivity
                                if sensitivity_active else 2 - consecutive)
        if cumulative >= MANDATORY_ITERATIONS and remaining < confirmations_needed:
            break
    write("lower_candidate_rejected", records, previous)
    print((TARGET / "summary.json").read_text(), end="")


if __name__ == "__main__":
    main()
