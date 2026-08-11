#!/usr/bin/env python3
"""Bracket a stronger reference weight if the lambda=675 barrier audit fails.

This recovery changes only the reference-distance coefficient.  The empirical
reference, selected b range, quadratic fit-quality barrier, stress identities,
fit gate, stationarity gate, and 200k horizon are held fixed.  Every candidate
starts from the same lambda=300 stationary central so a stronger candidate
cannot inherit an apparent plateau from lambda=675.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from alternate_lambda_authorization import (
    require_alternate_lambda_authorization,
)


BASE = Path(__file__).resolve().parents[1]
SYSTEMATICS = BASE.parent
ROOT = SYSTEMATICS.parent
PYTHON = Path("/home/dustin/miniforge3/envs/pdf-fit/bin/python")
RUNNER = SYSTEMATICS / "high_qt_direct_production_benchmark/experimental_unitary_transition/scripts/run_production_fnp_stability_control.py"
SOURCE = SYSTEMATICS / "collins_factorization_validity/outputs/rowidfix_stageFT_all_qmax0p20_lam0p50_central_s303"
W_GRID = ROOT / "outputs/v23a_tevatron_plus_lhcb7_fidacc_merged_cache/backend_cache/wpert_v23a_tevatron_plus_lhcb7_fidacc_b160.csv"
REFERENCE = BASE / "summaries/exact_baseline_fnp_median/fnp_median.csv"
CENTRAL = BASE / "outputs/fullref_lam300_b4_central_polish64_15000"
STAGED = BASE / "summaries/lambda675_fit_quality_barrier_stress"
TARGET = BASE / "summaries/fitbar_reference_strength_recovery"
LOGS = BASE / "logs/fitbar_reference_strength_recovery"
STRENGTHS = (712.5, 750.0, 825.0, 900.0, 1000.0, 1100.0, 1200.0,
             1350.0, 1500.0, 1650.0, 1800.0, 2000.0)
FNP_GATE = 0.02
SENSITIVITY_TRIGGER = 0.01
MINIMUM_ITERATIONS = 50_000
MAX_BLOCKS = 40


def token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def vector(run: Path) -> np.ndarray:
    frame = pd.read_csv(run / "fnp_grid.csv")
    mask = np.isclose(frame.x, 0.1) & (frame.bT >= 0.1) & (frame.bT <= 4.0)
    return frame.loc[mask, "F_NP"].to_numpy(float)


def run_candidate(strength: float, seed: int, mu: float, power: int,
                  records: list[dict]) -> tuple[bool, Path]:
    previous = CENTRAL
    consecutive = 0
    sensitivity_confirmation = False
    fresh_after_sensitivity = 0
    passed = False
    for block in range(1, MAX_BLOCKS + 1):
        cumulative = 5000 * block
        tag = (f"fullref_lam{token(strength)}_fitbar_p{power}_mu{token(mu)}_"
               f"recovery_r{seed}_polish64_{cumulative}")
        target = BASE / "outputs" / tag
        prior_status = json.loads((previous / "fit_status.json").read_text())
        row_count = int(prior_status["row_count"])
        ceiling = row_count + 5.0 * math.sqrt(2.0 * row_count)
        if not (target / "fit_status.json").exists():
            command = [
                str(PYTHON), str(RUNNER), "--seed", str(4000 + seed),
                "--source-production", str(SOURCE), "--w-grid", str(W_GRID),
                "--output-root", str(BASE / "outputs"), "--tag", tag,
                "--initial-state", str(previous / "model_state.pt"),
                "--initial-norms", str(previous / "dataset_norms.csv"),
                "--max-epochs", "0", "--min-epochs", "0",
                "--plateau-patience", "0", "--lbfgs-max-iter", "5000",
                "--float64", "--lambda-fnp-reference-distance", str(strength),
                "--fnp-reference-distance-csv", str(REFERENCE),
                "--fnp-reference-distance-bmin", "0.10",
                "--fnp-reference-distance-bmax", "4.0",
                "--fit-quality-ceiling-total-chi2", str(ceiling),
                "--lambda-fit-quality-barrier", str(mu),
                "--fit-quality-barrier-power", str(power),
                "--replica-seed", str(seed),
            ]
            with (LOGS / f"{tag}.log").open("w") as stream:
                subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                               check=True)
        status = json.loads((target / "fit_status.json").read_text())
        drift = float(np.max(np.abs(vector(target) - vector(previous)) /
                             np.maximum(np.abs(vector(previous)), 0.05)))
        quiet = drift <= FNP_GATE
        consecutive = consecutive + 1 if quiet else 0
        chi2 = float(status["final"]["unpenalized_total_chi2"])
        fit_pass = chi2 <= ceiling
        candidate_pass = (cumulative >= MINIMUM_ITERATIONS
                          and consecutive >= 2 and fit_pass)
        if sensitivity_confirmation:
            fresh_after_sensitivity = fresh_after_sensitivity + 1 if quiet else 0
            passed = fresh_after_sensitivity >= 2 and fit_pass
        elif candidate_pass and drift >= SENSITIVITY_TRIGGER:
            sensitivity_confirmation = True
            fresh_after_sensitivity = 0
        else:
            passed = candidate_pass
        records.append({
            "reference_strength": strength, "replica_seed": seed,
            "barrier_block": block, "historical_reference_iterations": 0,
            "total_reference_iterations": cumulative, "tag": tag,
            "fnp_drift": drift, "unpenalized_total_chi2": chi2,
            "fit_gate_pass": fit_pass,
            "sensitivity_confirmation_triggered": sensitivity_confirmation,
            "fresh_quiet_blocks_after_sensitivity_trigger": fresh_after_sensitivity,
            "stationarity_pass": passed,
        })
        pd.DataFrame(records).to_csv(TARGET / "runs.csv", index=False)
        previous = target
        if passed:
            return True, target
        remaining = MAX_BLOCKS - block
        needed = (2 - fresh_after_sensitivity if sensitivity_confirmation
                  else 2 - consecutive)
        if remaining < needed:
            break
    return False, previous


def main() -> None:
    require_alternate_lambda_authorization(
        "recover_barrier_reference_strength_after_stress_failure")
    failed = json.loads((STAGED / "summary.json").read_text())
    if failed.get("status") != "stress_failed":
        raise RuntimeError("recovery is authorized only after a terminal stress failure")
    failed_seed = int(failed["failed_replica_seed"])
    mu = float(failed["fit_quality_barrier_strength"])
    power = int(failed["fit_quality_barrier_power"])
    seeds = list(dict.fromkeys(
        [failed_seed] + [int(x) for x in failed["stress_replica_seeds_hardest_first"]]
    ))
    TARGET.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    outcomes: dict[str, dict] = {}
    selected = None
    selected_endpoints: list[str] = []
    for strength in STRENGTHS:
        endpoints: list[str] = []
        candidate_failure = None
        for seed in seeds:
            passed, endpoint = run_candidate(strength, seed, mu, power, records)
            endpoints.append(endpoint.name)
            if not passed:
                candidate_failure = seed
                break
        outcomes[f"{strength:g}"] = {
            "all_stress_replicas_pass": candidate_failure is None,
            "failed_replica": candidate_failure, "endpoint_tags": endpoints,
        }
        if candidate_failure is None:
            selected = strength
            selected_endpoints = endpoints
            break
    summary = {
        "status": "complete" if selected is not None else "no_candidate_passed",
        "failed_reference_strength": float(failed["reference_strength"]),
        "candidate_strengths_weakest_first": list(STRENGTHS),
        "weakest_tested_passing_strength": selected,
        "strongest_tested_failing_strength": max(
            [675.0] + [float(k) for k, v in outcomes.items()
                       if not v["all_stress_replicas_pass"]]),
        "candidate_outcomes": outcomes,
        "fit_quality_barrier_strength": mu,
        "fit_quality_barrier_power": power,
        "stress_replica_seeds_hardest_first": seeds,
        "endpoint_tags": selected_endpoints,
        "minimum_total_reference_iterations": MINIMUM_ITERATIONS,
        "selection_rule": "weakest strength above failed lambda675 passing every hard replica with fixed barrier and >=50k exposure",
        "production_sources_modified": False,
    }
    (TARGET / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if selected is None:
        print(json.dumps(summary, indent=2))
        raise RuntimeError("no controlled recovery strength passed")
    # Preserve the original failed audit separately, then expose the selected
    # generation through the established staged interface consumed downstream.
    original = STAGED / "lambda675_failed_summary.json"
    if not original.exists():
        original.write_text(json.dumps(failed, indent=2) + "\n")
    selected_rows = pd.DataFrame(records)
    selected_rows = selected_rows[np.isclose(
        selected_rows["reference_strength"], selected)]
    selected_rows.to_csv(STAGED / "runs.csv", index=False)
    staged = {
        "status": "complete", "reference_strength": selected,
        "strongest_tested_failing_strength": summary[
            "strongest_tested_failing_strength"],
        "fit_quality_barrier_strength": mu,
        "fit_quality_barrier_power": power,
        "stress_replica_seeds_hardest_first": seeds,
        "failed_replica_seed": None, "endpoint_tags": selected_endpoints,
        "stationarity_rule": "at least 50k same-strength exposure and two consecutive 5k blocks with <=2% FNP drift; terminal drift >=1% requires two wholly fresh confirmations",
        "minimum_total_reference_iterations": MINIMUM_ITERATIONS,
        "sensitivity_confirmation_trigger": SENSITIVITY_TRIGGER,
        "fit_quality_rule": "unpenalized total chi2 <= N+5sqrt(2N)",
        "strength_recovery_summary": str(TARGET / "summary.json"),
        "next_step": "reverify full24 starts and rebuild central plus 50 replicas",
        "production_sources_modified": False,
    }
    (STAGED / "summary.json").write_text(json.dumps(staged, indent=2) + "\n")
    print(json.dumps(staged, indent=2))


if __name__ == "__main__":
    main()
