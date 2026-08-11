#!/usr/bin/env python3
"""Long-horizon discriminator for one reference-strength+fit-barrier candidate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
SYSTEMATICS = BASE.parent
ROOT = SYSTEMATICS.parent
UNITARY = SYSTEMATICS / "high_qt_direct_production_benchmark/experimental_unitary_transition"
RUNNER = UNITARY / "scripts/run_production_fnp_stability_control.py"
SOURCE = SYSTEMATICS / "collins_factorization_validity/outputs/rowidfix_stageFT_all_qmax0p20_lam0p50_central_s303"
W_GRID = ROOT / "outputs/v23a_tevatron_plus_lhcb7_fidacc_merged_cache/backend_cache/wpert_v23a_tevatron_plus_lhcb7_fidacc_b160.csv"
REFERENCE = BASE / "summaries/exact_baseline_fnp_median/fnp_median.csv"
PYTHON = Path("/home/dustin/miniforge3/envs/pdf-fit/bin/python")
BLOCK = 5_000
GATE = 0.02
SENSITIVITY_TRIGGER = 0.01


def token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def vector(run: Path) -> np.ndarray:
    frame = pd.read_csv(run / "fnp_grid.csv")
    mask = np.isclose(frame.x, 0.1) & frame.bT.between(0.1, 4.0, inclusive="both")
    values = frame.loc[mask, "F_NP"].to_numpy(float)
    if len(values) < 3 or not np.all(np.isfinite(values)):
        raise RuntimeError(f"invalid selected-domain FNP grid: {run}")
    return values


def write(target: Path, status: str, args: argparse.Namespace,
          records: list[dict], endpoint: Path | None, ceiling: float) -> None:
    target.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(target / "runs.csv", index=False)
    payload = {
        "status": status,
        "tested_reference_strength": args.strength,
        "fit_quality_barrier_strength": args.barrier_strength,
        "fit_quality_barrier_power": args.barrier_power,
        "replica_seed": args.replica_seed,
        "fit_seed": args.fit_seed,
        "initial_central_tag": args.initial_tag,
        "mandatory_same_objective_iterations": args.mandatory_iterations,
        "maximum_same_objective_iterations": args.maximum_iterations,
        "iteration_accounting": (
            "cumulative labels are requested LBFGS max-iteration capacity; "
            "executed closure evaluations are recorded per block and may be "
            "smaller when LBFGS reaches its own convergence tolerance"
        ),
        "mandatory_continuation_checkpoint_count": (
            args.mandatory_iterations // BLOCK),
        "completed_continuation_checkpoint_count": len(records),
        "executed_lbfgs_closure_evaluations_total": sum(
            int(item.get("executed_lbfgs_closure_evaluations_this_block", 0))
            for item in records),
        "stationarity_gate": GATE,
        "sensitivity_trigger": SENSITIVITY_TRIGGER,
        "fit_quality_ceiling_total_chi2": ceiling,
        "endpoint_tag": endpoint.name if endpoint is not None else None,
        "production_sources_modified": False,
    }
    (target / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strength", type=float, required=True)
    parser.add_argument("--replica-seed", type=int, required=True)
    parser.add_argument("--fit-seed", type=int, required=True)
    parser.add_argument("--initial-tag", required=True)
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--barrier-strength", type=float, default=100.0)
    parser.add_argument("--barrier-power", type=int, default=2)
    parser.add_argument("--mandatory-iterations", type=int, required=True)
    parser.add_argument("--maximum-iterations", type=int, default=250_000)
    args = parser.parse_args()
    if args.mandatory_iterations % BLOCK or args.maximum_iterations % BLOCK:
        raise ValueError("iteration horizons must be multiples of 5000")
    if args.maximum_iterations < args.mandatory_iterations + 2 * BLOCK:
        raise ValueError("maximum horizon must allow two post-mandatory blocks")
    initial = BASE / "outputs" / args.initial_tag
    if not (initial / "fit_status.json").exists():
        raise RuntimeError(f"missing initial endpoint: {initial}")
    initial_status = json.loads((initial / "fit_status.json").read_text())
    row_count = int(initial_status["row_count"])
    ceiling = row_count + 5.0 * math.sqrt(2.0 * row_count)
    target_summary = BASE / "summaries" / args.target_name
    previous = initial
    records: list[dict] = []
    cumulative_closures = 0
    consecutive = 0
    sensitivity_active = False
    fresh_after_sensitivity = 0
    mandatory_anchor: np.ndarray | None = None
    write(target_summary, "running", args, records, None, ceiling)
    for cumulative in range(BLOCK, args.maximum_iterations + BLOCK, BLOCK):
        tag = (f"lowerctl_lam{token(args.strength)}_fitbar_p{args.barrier_power}_"
               f"mu{token(args.barrier_strength)}_"
               f"r{args.replica_seed}_polish64_{cumulative}")
        run = BASE / "outputs" / tag
        if not (run / "fit_status.json").exists():
            command = [
                str(PYTHON), str(RUNNER), "--seed", str(args.fit_seed),
                "--source-production", str(SOURCE), "--w-grid", str(W_GRID),
                "--output-root", str(BASE / "outputs"), "--tag", tag,
                "--initial-state", str(previous / "model_state.pt"),
                "--initial-norms", str(previous / "dataset_norms.csv"),
                "--max-epochs", "0", "--min-epochs", "0",
                "--plateau-patience", "0", "--lbfgs-max-iter", str(BLOCK),
                "--float64", "--replica-seed", str(args.replica_seed),
                "--lambda-fnp-reference-distance", str(args.strength),
                "--fnp-reference-distance-csv", str(REFERENCE),
                "--fnp-reference-distance-bmin", "0.10",
                "--fnp-reference-distance-bmax", "4.0",
                "--fit-quality-ceiling-total-chi2", str(ceiling),
                "--lambda-fit-quality-barrier", str(args.barrier_strength),
                "--fit-quality-barrier-power", str(args.barrier_power),
            ]
            with (BASE / "logs" / f"{tag}.log").open("w") as stream:
                subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                               check=True)
        status = json.loads((run / "fit_status.json").read_text())
        block_closures = int(status["lbfgs"]["closure_evaluations"])
        cumulative_closures += block_closures
        drift = float(np.max(np.abs(vector(run) - vector(previous)) /
                             np.maximum(vector(previous), 0.05)))
        fit_value = float(status["final"]["unpenalized_total_chi2"])
        fit_pass = fit_value <= ceiling
        eligible = cumulative > args.mandatory_iterations
        if cumulative == args.mandatory_iterations:
            mandatory_anchor = vector(run)
        window_drift = (float(np.max(np.abs(vector(run) - mandatory_anchor) /
                                    np.maximum(mandatory_anchor, 0.05)))
                        if eligible and mandatory_anchor is not None else None)
        window_quiet = window_drift is not None and window_drift <= GATE
        quiet = drift <= GATE
        consecutive = consecutive + 1 if eligible and quiet else 0
        passed = False
        # Any eligible 1--2% move is a sensitivity event.  It must not count
        # as one of the two wholly subsequent confirmation blocks.  The old
        # implementation activated this only when the *second* quiet block
        # happened to exceed 1%, allowing a first-block event to slip through.
        triggered_now = (eligible and quiet and drift >= SENSITIVITY_TRIGGER
                         and not sensitivity_active)
        if triggered_now:
            sensitivity_active = True
            fresh_after_sensitivity = 0
        elif sensitivity_active:
            fresh_after_sensitivity = (fresh_after_sensitivity + 1
                                       if eligible and quiet else 0)
            passed = fresh_after_sensitivity >= 2 and fit_pass and window_quiet
        elif eligible and consecutive >= 2 and fit_pass:
            passed = window_quiet
        records.append({
            "cumulative_lbfgs_iterations": cumulative,
            "requested_lbfgs_max_iterations_this_block": BLOCK,
            "executed_lbfgs_closure_evaluations_this_block": block_closures,
            "cumulative_executed_lbfgs_closure_evaluations": cumulative_closures,
            "fnp_drift_from_previous_chunk": drift,
            "eligible_post_mandatory_confirmation": eligible,
            "passes_drift_2pct": quiet,
            "post_mandatory_window_fnp_drift": window_drift,
            "passes_post_mandatory_window_drift_2pct": window_quiet,
            "consecutive_post_mandatory_quiet_blocks": consecutive,
            "sensitivity_confirmation_triggered": sensitivity_active,
            "fresh_quiet_blocks_after_sensitivity_trigger": fresh_after_sensitivity,
            "unpenalized_total_chi2": fit_value,
            "fit_quality_ceiling_total_chi2": ceiling,
            "fit_gate_pass": fit_pass,
            "stationarity_and_fit_pass": passed,
            "tag": tag,
        })
        write(target_summary, "running", args, records, run, ceiling)
        previous = run
        if passed:
            write(target_summary, "candidate_survives_discriminator", args,
                  records, run, ceiling)
            print((target_summary / "summary.json").read_text(), end="")
            return
        remaining = (args.maximum_iterations - cumulative) // BLOCK
        needed = (2 - fresh_after_sensitivity if sensitivity_active
                  else 2 - consecutive)
        if cumulative >= args.mandatory_iterations and remaining < needed:
            break
    write(target_summary, "candidate_rejected", args, records, previous, ceiling)
    print((target_summary / "summary.json").read_text(), end="")


if __name__ == "__main__":
    main()
