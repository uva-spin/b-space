#!/usr/bin/env python3
"""Verify the selected staged lambda675 + fit-barrier prescription on stress replicas."""

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
PYTHON = Path("/home/dustin/miniforge3/envs/pdf-fit/bin/python")
RUNNER = SYSTEMATICS / "high_qt_direct_production_benchmark/experimental_unitary_transition/scripts/run_production_fnp_stability_control.py"
SOURCE = SYSTEMATICS / "collins_factorization_validity/outputs/rowidfix_stageFT_all_qmax0p20_lam0p50_central_s303"
W_GRID = ROOT / "outputs/v23a_tevatron_plus_lhcb7_fidacc_merged_cache/backend_cache/wpert_v23a_tevatron_plus_lhcb7_fidacc_b160.csv"
REFERENCE = BASE / "summaries/exact_baseline_fnp_median/fnp_median.csv"
CENTRAL = BASE / "outputs/fullref_lam300_b4_central_polish64_15000"
QUADRATIC_PILOT = BASE / "summaries/lambda675_fit_quality_barrier_rescue/summary.json"
LINEAR_PILOT = BASE / "summaries/lambda675_fit_quality_barrier_rescue_linear/summary.json"
BRACKET = BASE / "summaries/full_reference_replica_strength_bracket/summary.json"
TARGET = BASE / "summaries/lambda675_fit_quality_barrier_stress"
LOGS = BASE / "logs/lambda675_fit_quality_barrier_stress"
FNP_GATE = 0.02
SENSITIVITY_TRIGGER = 0.01
MAX_BLOCKS = 40
MINIMUM_TOTAL_REFERENCE_ITERATIONS = 50_000


def token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def vector(run: Path) -> np.ndarray:
    frame = pd.read_csv(run / "fnp_grid.csv")
    mask = np.isclose(frame.x, 0.1) & (frame.bT >= 0.1) & (frame.bT <= 4.0)
    return frame.loc[mask, "F_NP"].to_numpy(float)


def latest_base(seed: int) -> Path:
    candidates = []
    prefix = f"fullref_replica_stress_lam675_r{seed}_polish64_"
    for path in (BASE / "outputs").glob(prefix + "*"):
        if (path / "fit_status.json").exists():
            try:
                iterations = int(path.name.rsplit("_", 1)[-1])
            except ValueError:
                continue
            candidates.append((iterations, path))
    return max(candidates)[1] if candidates else CENTRAL


def run_block(seed: int, mu: float, power: int, block: int, previous: Path,
              ceiling: float) -> Path:
    tag = (f"fullref_lam675_fitbar_p{power}_mu{token(mu)}_stress_r{seed}_"
           f"polish64_{5000 * block}")
    target = BASE / "outputs" / tag
    if not (target / "fit_status.json").exists():
        command = [
            str(PYTHON), str(RUNNER), "--seed", str(4000 + seed),
            "--source-production", str(SOURCE), "--w-grid", str(W_GRID),
            "--output-root", str(BASE / "outputs"), "--tag", tag,
            "--initial-state", str(previous / "model_state.pt"),
            "--initial-norms", str(previous / "dataset_norms.csv"),
            "--max-epochs", "0", "--min-epochs", "0", "--plateau-patience", "0",
            "--lbfgs-max-iter", "5000", "--float64",
            "--lambda-fnp-reference-distance", "675",
            "--fnp-reference-distance-csv", str(REFERENCE),
            "--fnp-reference-distance-bmin", "0.10",
            "--fnp-reference-distance-bmax", "4.0",
            "--fit-quality-ceiling-total-chi2", str(ceiling),
            "--lambda-fit-quality-barrier", str(mu),
            "--fit-quality-barrier-power", str(power),
            "--replica-seed", str(seed),
        ]
        with (LOGS / f"{tag}.log").open("w") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
    return target


def main() -> None:
    pilot_path = LINEAR_PILOT if LINEAR_PILOT.exists() else QUADRATIC_PILOT
    pilot = json.loads(pilot_path.read_text())
    if pilot["status"] != "complete":
        raise RuntimeError("no fit-quality barrier passed the replica1041 pilot")
    mu = float(pilot["selected_weakest_barrier_strength"])
    power = int(pilot.get("fit_quality_barrier_power", 2))
    bracket = json.loads(BRACKET.read_text())
    seeds = list(dict.fromkeys([1041] + [int(x) for x in bracket["stress_replica_seeds"]]))
    TARGET.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    records = []
    endpoints = []
    failed_seed = None
    for seed in seeds:
        if seed == 1041:
            endpoint = BASE / "outputs" / str(pilot["selected_endpoint"])
            status = json.loads((endpoint / "fit_status.json").read_text())
            endpoints.append(endpoint.name)
            records.append({
                "replica_seed": seed, "barrier_block": 2,
                "historical_reference_iterations": 200_000,
                "total_reference_iterations": 210_000,
                "tag": endpoint.name,
                "fnp_drift": pilot["trials"][-1]["fnp_drift"],
                "unpenalized_total_chi2": status["final"]["unpenalized_total_chi2"],
                "fit_gate_pass": pilot["trials"][-1]["fit_gate_pass"],
                "stationarity_pass": pilot["trials"][-1]["fnp_stationarity_pass"],
            })
            continue
        previous = latest_base(seed)
        starts_from_central = previous == CENTRAL
        historical_iterations = 0
        if not starts_from_central:
            historical_iterations = int(previous.name.rsplit("_", 1)[-1])
        minimum_blocks = max(
            2,
            math.ceil((MINIMUM_TOTAL_REFERENCE_ITERATIONS
                       - historical_iterations) / 5000),
        )
        consecutive = 0
        sensitivity_confirmation = False
        fresh_after_sensitivity = 0
        passed = False
        for block in range(1, MAX_BLOCKS + 1):
            status0 = json.loads((previous / "fit_status.json").read_text())
            n = int(status0["row_count"])
            ceiling = n + 5.0 * math.sqrt(2.0 * n)
            current = run_block(seed, mu, power, block, previous, ceiling)
            status = json.loads((current / "fit_status.json").read_text())
            drift = float(np.max(np.abs(vector(current) - vector(previous)) /
                                 np.maximum(np.abs(vector(previous)), 0.05)))
            quiet = drift <= FNP_GATE
            consecutive = consecutive + 1 if quiet else 0
            chi2 = float(status["final"]["unpenalized_total_chi2"])
            fit_pass = chi2 <= ceiling
            candidate_pass = block >= minimum_blocks and consecutive >= 2 and fit_pass
            if sensitivity_confirmation:
                fresh_after_sensitivity = fresh_after_sensitivity + 1 if quiet else 0
                passed = fresh_after_sensitivity >= 2 and fit_pass
            elif candidate_pass and drift >= SENSITIVITY_TRIGGER:
                # A terminal value near the 2% gate receives two wholly new
                # blocks.  The 1% threshold only triggers more evidence; it
                # does not replace the unchanged 2% acceptance gate.
                sensitivity_confirmation = True
                fresh_after_sensitivity = 0
                passed = False
            else:
                passed = candidate_pass
            records.append({
                "replica_seed": seed, "barrier_block": block,
                "historical_reference_iterations": historical_iterations,
                "total_reference_iterations": historical_iterations + 5000 * block,
                "tag": current.name, "fnp_drift": drift,
                "unpenalized_total_chi2": chi2,
                "fit_gate_pass": fit_pass,
                "sensitivity_confirmation_triggered": sensitivity_confirmation,
                "fresh_quiet_blocks_after_sensitivity_trigger": fresh_after_sensitivity,
                "stationarity_pass": passed,
            })
            pd.DataFrame(records).to_csv(TARGET / "runs.csv", index=False)
            previous = current
            if passed:
                endpoints.append(current.name)
                break
        if not passed:
            failed_seed = seed
            endpoints.append(previous.name)
            break
    summary = {
        "status": "complete" if failed_seed is None else "stress_failed",
        "reference_strength": 675.0,
        "fit_quality_barrier_strength": mu,
        "fit_quality_barrier_power": power,
        "stress_replica_seeds_hardest_first": seeds,
        "failed_replica_seed": failed_seed,
        "endpoint_tags": endpoints,
        "stationarity_rule": "at least 50k total historical-plus-staged lambda675 exposure and two consecutive 5k blocks with <=2% selected-domain FNP drift; a terminal drift >=1% triggers two wholly subsequent <=2% blocks",
        "minimum_total_reference_iterations": MINIMUM_TOTAL_REFERENCE_ITERATIONS,
        "sensitivity_confirmation_trigger": SENSITIVITY_TRIGGER,
        "fit_quality_rule": "unpenalized total chi2 <= N+5sqrt(2N)",
        "next_step": "reverify full24 starts and rebuild central plus 50 replicas with the staged barrier" if failed_seed is None else "revise the constrained optimizer; do not increase reference strength blindly",
        "production_sources_modified": False,
    }
    (TARGET / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
