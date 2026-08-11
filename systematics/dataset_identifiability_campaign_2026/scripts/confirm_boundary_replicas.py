#!/usr/bin/env python3
"""Confirm endpoints near the FNP drift boundary or above the chi2 ceiling.

The primary supervisor requires two consecutive 5k blocks below 2% FNP
movement. A final endpoint whose drift is at or above the recorded 1%
sensitivity threshold receives at least one additional block. If the extra
block is not quiet, the original two-block rule is reaccumulated for up to
eight additional blocks beyond that member's primary endpoint. An endpoint above the declared N+5sqrt(2N) pseudo-data
chi-square sanity ceiling is likewise continued until both fit quality and
FNP stationarity pass. The prospective hard-stress replicas used to bracket
the selected strength are always confirmed, even if their primary endpoint
lies below both triggers.
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
RUNNER = SYSTEMATICS / "high_qt_direct_production_benchmark/experimental_unitary_transition/scripts/run_production_fnp_stability_control.py"
SOURCE = SYSTEMATICS / "collins_factorization_validity/outputs/rowidfix_stageFT_all_qmax0p20_lam0p50_central_s303"
W_GRID = ROOT / "outputs/v23a_tevatron_plus_lhcb7_fidacc_merged_cache/backend_cache/wpert_v23a_tevatron_plus_lhcb7_fidacc_b160.csv"
REFERENCE = BASE / "summaries/exact_baseline_fnp_median/fnp_median.csv"
CAMPAIGN = BASE / "summaries/selected_reference_central_replicas"
SUMMARY = CAMPAIGN / "summary.json"
BRACKET = BASE / "summaries/full_reference_replica_strength_bracket/summary.json"
STAGED_STRESS = BASE / "summaries/lambda675_fit_quality_barrier_stress/summary.json"
TARGET = BASE / "summaries/selected_reference_boundary_confirmation"
PYTHON = Path("/home/dustin/miniforge3/envs/pdf-fit/bin/python")
GATE = 0.02
BOUNDARY_FRACTION = 0.50
LATE_HORIZON_START = 30000
# This is deliberately relative to each primary endpoint. An absolute ceiling
# would make an independent two-block confirmation impossible for a member
# that first settles near the 160k primary horizon.
MAX_ADDITIONAL_BLOCKS = 8
def token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def vector(run: Path, bmax: float) -> np.ndarray:
    frame = pd.read_csv(run / "fnp_grid.csv")
    frame = frame[np.isclose(frame.x, .1) & (frame.bT >= .1)
                  & (frame.bT <= bmax)].sort_values("bT")
    return frame.F_NP.to_numpy(float)


def main() -> None:
    summary = json.loads(SUMMARY.read_text())
    bracket = json.loads(BRACKET.read_text())
    staged_stress = (json.loads(STAGED_STRESS.read_text())
                     if STAGED_STRESS.exists() else None)
    stress_values = (staged_stress["stress_replica_seeds_hardest_first"]
                     if staged_stress is not None and staged_stress.get("status") == "complete"
                     else bracket["stress_replica_seeds"])
    prospective_stress_replicas = {int(value) for value in stress_values}
    if not prospective_stress_replicas:
        raise RuntimeError("replica-strength bracket has no stress identities")
    if summary["status"] != "complete" or not summary["all_replicas_fnp_plateaued"]:
        raise RuntimeError("primary 50-replica campaign is not complete")
    records = pd.read_csv(CAMPAIGN / "runs.csv").to_dict("records")
    frame = pd.DataFrame(records)
    strength = float(summary["selected_strength"])
    bmax = float(summary["selected_bmax"])
    barrier_strength = float(summary.get("fit_quality_barrier_strength", 0.0))
    barrier_power = int(summary.get("fit_quality_barrier_power", 2))
    endpoint_tags = list(summary["replica_endpoint_tags"])
    selected = []
    selection_reasons = {}
    empirical_delayed_replicas = set()
    for replica_seed in range(1001, 1051):
        rows = frame[(frame.kind == "experimental_replica")
                     & np.isclose(frame.replica_seed, replica_seed)].sort_values(
                         "cumulative_lbfgs_iterations")
        if rows.empty:
            raise RuntimeError(f"missing replica {replica_seed}")
        last = rows.iloc[-1]
        endpoint = BASE / "outputs" / str(last.tag)
        endpoint_status = json.loads((endpoint / "fit_status.json").read_text())
        row_count = int(endpoint_status["row_count"])
        chi2_ceiling = row_count + 5.0 * math.sqrt(2.0 * row_count)
        reasons = []
        # The full campaign exposed delayed motion as early as 30k that could
        # quiet again by 45k/50k.  Such a member still needs an independent
        # fresh confirmation; restricting this trigger to >40k would miss it.
        late_horizon = rows[
            rows.cumulative_lbfgs_iterations >= LATE_HORIZON_START
        ]
        if (not late_horizon.empty
                and float(late_horizon.fnp_drift_from_previous_chunk.max()) > GATE):
            reasons.append("late_horizon_fnp_drift_above_2pct_from_30k")
            empirical_delayed_replicas.add(replica_seed)
        if float(last.fnp_drift_from_previous_chunk) >= BOUNDARY_FRACTION * GATE:
            reasons.append("terminal_fnp_drift_at_or_above_1pct")
        if float(last.unpenalized_total_chi2) > chi2_ceiling:
            reasons.append("unpenalized_chi2_above_declared_sanity_ceiling")
        if replica_seed in prospective_stress_replicas:
            reasons.append("prospective_strength_bracket_stress_member")
        if reasons:
            selected.append(replica_seed)
            selection_reasons[str(replica_seed)] = reasons

    TARGET.mkdir(parents=True, exist_ok=True)
    (TARGET / "summary.json").write_text(json.dumps({
        "status": "running", "boundary_fraction": BOUNDARY_FRACTION,
        "fnp_drift_domain": {"x": 0.1, "b_min": 0.1, "b_max": bmax},
        "selected_replica_seeds": selected, "selection_reasons": selection_reasons,
        "production_sources_modified": False,
    }, indent=2) + "\n")

    failures = []
    confirmations = []
    for replica_seed in selected:
        index = replica_seed - 1001
        rows = pd.DataFrame(records)
        rows = rows[(rows["kind"] == "experimental_replica")
                    & np.isclose(rows["replica_seed"], replica_seed)].sort_values(
                        "cumulative_lbfgs_iterations")
        last = rows.iloc[-1]
        cumulative = int(last.cumulative_lbfgs_iterations)
        consecutive = int(last.consecutive_quiet_blocks)
        if (replica_seed in prospective_stress_replicas
                or replica_seed in empirical_delayed_replicas):
            # A quiet block was followed by a later reversal either in the
            # prospective bracket or in this primary campaign. Require two
            # entirely new quiet blocks beyond the primary endpoint.
            consecutive = 0
        previous = BASE / "outputs" / str(last.tag)
        confirmed = False
        additional_blocks = 0
        while additional_blocks < MAX_ADDITIONAL_BLOCKS:
            additional_blocks += 1
            cumulative += 5000
            barrier_tag = (f"_fitbar_p{barrier_power}_mu{token(barrier_strength)}"
                           if barrier_strength > 0 else "")
            tag = (f"fullref_lam{token(strength)}{barrier_tag}_b{token(bmax)}_replica_r"
                   f"{replica_seed}_polish64_{cumulative}")
            target = BASE / "outputs" / tag
            if not (target / "fit_status.json").exists():
                command = [
                    str(PYTHON), str(RUNNER), "--seed", str(int(last.fit_seed)),
                    "--source-production", str(SOURCE), "--w-grid", str(W_GRID),
                    "--output-root", str(BASE / "outputs"), "--tag", tag,
                    "--initial-state", str(previous / "model_state.pt"),
                    "--initial-norms", str(previous / "dataset_norms.csv"),
                    "--max-epochs", "0", "--min-epochs", "0",
                    "--plateau-patience", "0", "--lbfgs-max-iter", "5000",
                    "--float64", "--lambda-fnp-reference-distance", str(strength),
                    "--fnp-reference-distance-csv", str(REFERENCE),
                    "--fnp-reference-distance-bmin", "0.10",
                    "--fnp-reference-distance-bmax", str(bmax),
                    "--replica-seed", str(replica_seed),
                ]
                if barrier_strength > 0:
                    endpoint_status = json.loads((previous / "fit_status.json").read_text())
                    endpoint_rows = int(endpoint_status["row_count"])
                    barrier_ceiling = endpoint_rows + 5.0 * math.sqrt(2.0 * endpoint_rows)
                    command.extend([
                        "--fit-quality-ceiling-total-chi2", str(barrier_ceiling),
                        "--lambda-fit-quality-barrier", str(barrier_strength),
                        "--fit-quality-barrier-power", str(barrier_power),
                    ])
                with (BASE / "logs" / f"{tag}.log").open("w") as stream:
                    subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                   check=True)
            status = json.loads((target / "fit_status.json").read_text())
            drift = float(np.max(np.abs(vector(target, bmax) - vector(previous, bmax))
                                 / np.maximum(vector(previous, bmax), .05)))
            quiet = drift <= GATE
            consecutive = consecutive + 1 if quiet else 0
            row_count = int(status["row_count"])
            chi2_ceiling = row_count + 5.0 * math.sqrt(2.0 * row_count)
            chi2_pass = float(status["final"]["unpenalized_total_chi2"]) <= chi2_ceiling
            confirmed = quiet and consecutive >= 2 and chi2_pass
            records.append({
                "kind": "experimental_replica", "replica_seed": replica_seed,
                "fit_seed": int(last.fit_seed),
                "cumulative_lbfgs_iterations": cumulative,
                "requested_lbfgs_max_iterations_this_block": 5000,
                "executed_lbfgs_closure_evaluations_this_block": int(
                    status["lbfgs"]["closure_evaluations"]),
                "fnp_drift_from_previous_chunk": drift,
                "passes_drift_0p25pct": drift <= .0025,
                "passes_drift_0p5pct": drift <= .005,
                "passes_drift_1pct": drift <= .01,
                "passes_drift_2pct": quiet,
                "consecutive_quiet_blocks": consecutive,
                "unpenalized_total_chi2": status["final"]["unpenalized_total_chi2"],
                "data_chi2": status["final"]["data_chi2"],
                "replica_chi2_sanity_ceiling": chi2_ceiling,
                "replica_chi2_sanity_pass": chi2_pass,
                "fnp_plateau_gate_pass": confirmed, "tag": tag,
            })
            pd.DataFrame(records).to_csv(CAMPAIGN / "runs.csv", index=False)
            previous = target
            if confirmed:
                endpoint_tags[index] = tag
                confirmations.append({"replica_seed": replica_seed,
                                      "endpoint_tag": tag, "final_drift": drift,
                                      "cumulative_iterations": cumulative,
                                      "unpenalized_total_chi2": float(status["final"]["unpenalized_total_chi2"]),
                                      "replica_chi2_sanity_ceiling": chi2_ceiling})
                break
        if not confirmed:
            failures.append(replica_seed)

    summary["replica_endpoint_tags"] = endpoint_tags
    summary["boundary_confirmation"] = {
        "rule": "endpoint with final drift at or above the recorded 1% confirmation trigger, any drift above 2% from 30k onward, unpenalized chi2 above the declared N+5sqrt(2N) sanity ceiling, or any authoritative replica-strength-bracket stress identity receives extra 5k blocks; the original two-consecutive-block FNP rule and the chi2 ceiling are both enforced, and delayed/stress identities must establish two fresh quiet blocks beyond their primary endpoints",
        "selected_replica_seeds": selected, "selection_reasons": selection_reasons,
        "confirmations": confirmations,
        "failed_replica_seeds": failures,
    }
    if failures:
        summary["status"] = "replica_stationarity_failed"
        summary["all_replicas_fnp_plateaued"] = False
        summary["failed_replica_seeds"] = sorted(set(
            summary.get("failed_replica_seeds", []) + failures))
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")
    result = {
        "status": "complete" if not failures else "failed",
        "boundary_fraction": BOUNDARY_FRACTION, "drift_gate": GATE,
        "late_horizon_start_iterations": LATE_HORIZON_START,
        "iteration_accounting": "cumulative labels are requested LBFGS max-iteration capacity; each continuation row records actual closure evaluations separately",
        "fnp_drift_domain": {"x": 0.1, "b_min": 0.1, "b_max": bmax},
        "selected_replica_seeds": selected, "selection_reasons": selection_reasons,
        "confirmations": confirmations,
        "failed_replica_seeds": failures, "production_sources_modified": False,
        "max_additional_confirmation_blocks": MAX_ADDITIONAL_BLOCKS,
    }
    (TARGET / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if failures:
        raise RuntimeError(f"boundary confirmations failed: {failures}")


if __name__ == "__main__":
    main()
