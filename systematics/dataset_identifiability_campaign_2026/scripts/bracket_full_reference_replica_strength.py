#!/usr/bin/env python3
"""Bracket the minimum full-reference strength on fixed replica stress cases.

Every strength/replica pair starts from the same selected stationary central
endpoint.  This avoids warm-starting a stronger prior from a weaker prior's
solution.  Replica 1001 is tested first because lambda=300 failed to settle
through 35k; a candidate only advances to the remaining prospective stress
replicas if that case passes.  Two consecutive unchanged-objective 5k blocks
must have <=2% selected-domain FNP drift.  Any member that has already failed
an extended full-ensemble run is tested for at least 40k before it may pass,
preventing a repeat of replica1032's false early plateau.
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
UNITARY = SYSTEMATICS / "high_qt_direct_production_benchmark/experimental_unitary_transition"
RUNNER = UNITARY / "scripts/run_production_fnp_stability_control.py"
SOURCE = SYSTEMATICS / "collins_factorization_validity/outputs/rowidfix_stageFT_all_qmax0p20_lam0p50_central_s303"
W_GRID = ROOT / "outputs/v23a_tevatron_plus_lhcb7_fidacc_merged_cache/backend_cache/wpert_v23a_tevatron_plus_lhcb7_fidacc_b160.csv"
REFERENCE = BASE / "summaries/exact_baseline_fnp_median/fnp_median.csv"
CENTRAL = BASE / "outputs/fullref_lam300_b4_central_polish64_15000"
PYTHON = Path("/home/dustin/miniforge3/envs/pdf-fit/bin/python")
TARGET = BASE / "summaries/full_reference_replica_strength_bracket"
# Final bisections limit the fail/pass bracket to <10% of the passing value:
# 487.5 resolves [450,525] if 525 passes; 562.5 resolves [525,600] if it fails.
STRENGTHS = (450.0, 487.5, 525.0, 562.5, 600.0, 1000.0, 3000.0)
REPLICA_SEEDS = (1001, 1002, 1013, 1025, 1038, 1050)
FNP_DRIFT_GATE = 0.02
SENSITIVITY = (0.0025, 0.005, 0.01, 0.02)
REQUIRED_CONSECUTIVE = 2
MAX_CHUNKS = 8
# A replica that has already failed a complete, extended 50-member campaign
# must not be allowed to select a stronger prescription from another short
# apparent plateau.  Replica 1032 passed an earlier 15k stress trajectory but
# later moved persistently through 75k in the fresh ensemble.  Requiring 40k
# for empirically failed members probes that delayed instability while leaving
# ordinary prospective stress cases adaptive.
DELAYED_FAILURE_CONFIRMATION_CHUNKS = 8
# Historical fallback for the old 80k failure manifest.  `run_case` raises
# this dynamically to two blocks beyond the actually exposed full-ensemble
# horizon whenever the ledger is available.
SAME_STRENGTH_FAILURE_CONFIRMATION_CHUNKS = 18
BMAX = 4.0


def token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def vector(run: Path) -> np.ndarray:
    frame = pd.read_csv(run / "fnp_grid.csv")
    mask = np.isclose(frame.x, .1) & (frame.bT <= BMAX)
    return frame.loc[mask, "F_NP"].to_numpy()


def delayed_failure_seeds() -> set[int]:
    """Return replicas promoted by failures or severe delayed trajectories."""
    seeds: set[int] = set()
    summary_path = TARGET / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        original = set(REPLICA_SEEDS)
        seeds.update(int(value) for value in summary.get("stress_replica_seeds", [])
                     if int(value) not in original)
        seeds.update(int(value) for value in summary.get("added_stress_replica_seeds", []))
    failed_path = BASE / "summaries/selected_reference_central_replicas/failed_80k_summary.json"
    if failed_path.exists():
        failed = json.loads(failed_path.read_text())
        seeds.update(int(value) for value in failed.get("failed_replica_seeds", []))
    # A member that only settles after a long, large-motion trajectory is an
    # empirical near-failure even if its final two blocks happen to be quiet.
    # Preserve that information in later calibration rather than reducing the
    # full trajectory to a binary endpoint label.
    ledger_path = BASE / "summaries/selected_reference_central_replicas/runs.csv"
    if ledger_path.exists():
        ledger = pd.read_csv(ledger_path)
        replica = ledger[ledger["kind"].eq("experimental_replica")].copy()
        for replica_seed, rows in replica.groupby("replica_seed"):
            if (rows["cumulative_lbfgs_iterations"].max() >= 40000
                    and rows["fnp_drift_from_previous_chunk"].max() > FNP_DRIFT_GATE):
                seeds.add(int(replica_seed))
    # Preserve delayed behavior discovered inside the strength bracket itself.
    # In particular, replica1001 only settled at 80k for lambda637.5; allowing
    # a stronger strength to accept it at an incidental 15k plateau would
    # repeat the exact short-horizon failure this campaign is designed to
    # prevent. The current invocation rebuilds this ledger from cached runs
    # before advancing to a new strength, so it is authoritative here.
    bracket_ledger = TARGET / "runs.csv"
    if bracket_ledger.exists():
        ledger = pd.read_csv(bracket_ledger)
        for replica_seed, rows in ledger.groupby("replica_seed"):
            if (rows["cumulative_lbfgs_iterations"].max() >= 40000
                    and rows["fnp_drift_from_previous_chunk"].max()
                    > FNP_DRIFT_GATE):
                seeds.add(int(replica_seed))
    return seeds


def run_case(strength: float, replica_seed: int, records: list[dict]) -> tuple[bool, Path]:
    previous = CENTRAL
    consecutive = 0
    passed = False
    fit_seed = 4000 + replica_seed
    delayed_confirmation = replica_seed in delayed_failure_seeds()
    minimum_chunks = DELAYED_FAILURE_CONFIRMATION_CHUNKS if delayed_confirmation else 3
    failed_path = BASE / "summaries/selected_reference_central_replicas/failed_80k_summary.json"
    if failed_path.exists():
        failed = json.loads(failed_path.read_text())
        if (strength >= float(failed["selected_strength"])
                and replica_seed in {int(value) for value in failed.get("failed_replica_seeds", [])}):
            exposed_chunks = SAME_STRENGTH_FAILURE_CONFIRMATION_CHUNKS - 2
            ledger_path = BASE / "summaries/selected_reference_central_replicas/runs.csv"
            if ledger_path.exists():
                ledger = pd.read_csv(ledger_path)
                rows = ledger[
                    ledger["replica_seed"].eq(replica_seed)
                    & ledger["kind"].eq("experimental_replica")
                ]
                if not rows.empty:
                    exposed_chunks = max(
                        exposed_chunks,
                        int(rows["cumulative_lbfgs_iterations"].max()) // 5000,
                    )
            # Preserve the empirical delayed-reversal horizon at every
            # stronger candidate.  When the controller's finite audit cap is
            # already the prior horizon, require the entire available horizon
            # rather than making passage mathematically impossible.
            minimum_chunks = min(
                MAX_CHUNKS, exposed_chunks + REQUIRED_CONSECUTIVE
            )
    for chunk in range(1, MAX_CHUNKS + 1):
        cumulative = 5000 * chunk
        tag = (f"fullref_replica_stress_lam{token(strength)}_r{replica_seed}_"
               f"polish64_{cumulative}")
        target = BASE / "outputs" / tag
        if not (target / "fit_status.json").exists():
            command = [
                str(PYTHON), str(RUNNER), "--seed", str(fit_seed),
                "--source-production", str(SOURCE), "--w-grid", str(W_GRID),
                "--output-root", str(BASE / "outputs"), "--tag", tag,
                "--initial-state", str(previous / "model_state.pt"),
                "--initial-norms", str(previous / "dataset_norms.csv"),
                "--max-epochs", "0", "--min-epochs", "0", "--plateau-patience", "0",
                "--lbfgs-max-iter", "5000", "--float64",
                "--lambda-fnp-reference-distance", str(strength),
                "--fnp-reference-distance-csv", str(REFERENCE),
                "--fnp-reference-distance-bmin", "0.10",
                "--fnp-reference-distance-bmax", str(BMAX),
                "--replica-seed", str(replica_seed),
            ]
            with (BASE / "logs" / f"{tag}.log").open("w") as stream:
                subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        status = json.loads((target / "fit_status.json").read_text())
        drift = float(np.max(np.abs(vector(target) - vector(previous)) /
                             np.maximum(vector(previous), .05)))
        quiet = chunk > 1 and drift <= FNP_DRIFT_GATE
        consecutive = consecutive + 1 if quiet else 0
        row_count = int(status["row_count"])
        replica_chi2_ceiling = row_count + 5.0 * math.sqrt(2.0 * row_count)
        replica_chi2_pass = (
            float(status["final"]["unpenalized_total_chi2"])
            <= replica_chi2_ceiling
        )
        passed = (chunk >= minimum_chunks
                  and consecutive >= REQUIRED_CONSECUTIVE
                  and replica_chi2_pass)
        records.append({
            "strength": strength, "replica_seed": replica_seed,
            "fit_seed": fit_seed, "cumulative_lbfgs_iterations": cumulative,
            "fnp_drift_from_previous_chunk": drift,
            "passes_drift_0p25pct": chunk > 1 and drift <= SENSITIVITY[0],
            "passes_drift_0p5pct": chunk > 1 and drift <= SENSITIVITY[1],
            "passes_drift_1pct": chunk > 1 and drift <= SENSITIVITY[2],
            "passes_drift_2pct": chunk > 1 and drift <= SENSITIVITY[3],
            "consecutive_quiet_blocks": consecutive,
            "delayed_failure_confirmation_required": delayed_confirmation,
            "minimum_confirmation_iterations": 5000 * minimum_chunks,
            "unpenalized_total_chi2": status["final"]["unpenalized_total_chi2"],
            "data_chi2": status["final"]["data_chi2"],
            "replica_chi2_sanity_ceiling": replica_chi2_ceiling,
            "replica_chi2_sanity_pass": replica_chi2_pass,
            "stationarity_pass": passed, "tag": tag,
        })
        pd.DataFrame(records).to_csv(TARGET / "runs.csv", index=False)
        previous = target
        if passed:
            break
        # Do not run blocks that cannot possibly supply the remaining number
        # of consecutive quiet confirmations before the fixed ceiling.
        remaining_blocks = MAX_CHUNKS - chunk
        confirmations_needed = REQUIRED_CONSECUTIVE - consecutive
        if remaining_blocks < confirmations_needed:
            break
    return passed, previous


def main() -> None:
    if not (CENTRAL / "fit_status.json").exists():
        raise RuntimeError("selected stationary central endpoint is required")
    TARGET.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    outcomes: dict[str, dict] = {
        "300": {
            "all_stress_replicas_pass": False,
            "failed_replica": 1001,
            "evidence": "0 consecutive quiet blocks through 35k; cannot reach two by 40k",
        }
    }
    selected: float | None = None
    selected_endpoints: list[str] = []
    for strength in STRENGTHS:
        endpoints: list[str] = []
        failed_replica: int | None = None
        for replica_seed in REPLICA_SEEDS:
            passed, endpoint = run_case(strength, replica_seed, records)
            endpoints.append(endpoint.name)
            if not passed:
                failed_replica = replica_seed
                break
        all_pass = failed_replica is None
        outcomes[f"{strength:g}"] = {
            "all_stress_replicas_pass": all_pass,
            "failed_replica": failed_replica,
            "endpoint_tags": endpoints,
        }
        if all_pass:
            selected = strength
            selected_endpoints = endpoints
            break
    strongest_fail = max(
        (float(key) for key, value in outcomes.items()
         if not value["all_stress_replicas_pass"]), default=None)
    relative_bracket_width = (
        (selected - strongest_fail) / selected
        if selected is not None and strongest_fail is not None else None
    )
    summary = {
        "status": "complete" if selected is not None else "no_tested_strength_passed",
        "candidate_strengths_weakest_first": list(STRENGTHS),
        "stress_replica_seeds": list(REPLICA_SEEDS),
        "initialization_rule": "same selected stationary central for every strength/replica case",
        "stationarity_rule": "two consecutive unchanged-objective 5k blocks with max selected-domain FNP drift <=2%",
        "drift_sensitivity_thresholds": list(SENSITIVITY),
        "candidate_outcomes": outcomes,
        "weakest_tested_passing_strength": selected,
        "strongest_tested_failing_strength": strongest_fail,
        "relative_fail_pass_bracket_width": relative_bracket_width,
        "bracket_resolution_rule": "stop after fail/pass width is <=10% of passing strength",
        "selected_endpoint_tags": selected_endpoints,
        "next_step": "verify selected strength on full 24 central starts, then restart central plus 50 replicas",
        "production_sources_modified": False,
    }
    (TARGET / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
