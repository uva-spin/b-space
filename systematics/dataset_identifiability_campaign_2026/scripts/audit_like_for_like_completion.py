#!/usr/bin/env python3
"""Fail-closed completion audit for the strengthened lambda=600 comparison."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from postfit_tail_transform_validation import (
    validated_postfit_tail_audit,
)
from nested_interaction_and_final_envelope_validation import (
    final_promotion_gate,
    validated_final_directional_envelope,
    validated_nested_interaction,
)


BASE = Path(__file__).resolve().parents[1]
SYSTEMATICS = BASE.parent
ROOT = SYSTEMATICS.parent
SUM = BASE / "summaries"
OUT = BASE / "outputs"
SOURCE_PRODUCTION = (SYSTEMATICS /
    "collins_factorization_validity/outputs/rowidfix_stageFT_all_qmax0p20_lam0p50_central_s303")
W_GRID = (ROOT /
    "outputs/v23a_tevatron_plus_lhcb7_fidacc_merged_cache/backend_cache/wpert_v23a_tevatron_plus_lhcb7_fidacc_b160.csv")
REFERENCE = SUM / "exact_baseline_fnp_median/fnp_median.csv"
START_SOURCE_ENSEMBLE = SUM / "selected_reference_method_full24/summary.json"
TARGET = SUM / "campaign_completion_audit"
STATE_CHAIN_SCRIPT = BASE / "scripts/audit_lambda600_state_chains.py"
STATE_CHAIN_AUDIT = SUM / "lambda600_state_chain_audit/summary.json"
START_CHAIN_AUDIT = SUM / "lambda600_start_chain_audit/summary.json"
TERMINAL_EVIDENCE = SUM / "lambda600_terminal_evidence/summary.json"
COMPARISON = SUM / "lambda600_vs_lambda1_diagnostic/summary.json"
POSTFIT_TAIL_AUDIT = (
    SUM / "lambda600_postfit_tail_transform_audit/summary.json"
)
NESTED_INTERACTION = (
    SUM / "lambda600_nested_start_replica_interaction/summary.json"
)
FINAL_DIRECTIONAL_ENVELOPE = (
    SUM / "lambda600_final_directional_envelope/summary.json"
)
ANCHOR = 200_000
REQUIRED = 10
DRIFT_GATE = 0.02
LOCKED_INCUMBENT_WIDTHS = {
    "u": 0.11772613918747582,
    "d": 0.12490071924111977,
}


def load(relative: str) -> dict:
    path = SUM / relative
    if not path.is_file():
        raise RuntimeError(f"missing summary: {path}")
    return json.loads(path.read_text())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def exact_numeric(observed: object, expected: float) -> bool:
    try:
        value = float(observed)
    except (TypeError, ValueError):
        return False
    return (np.isfinite(value)
            and np.isclose(value, float(expected),
                           rtol=0.0, atol=1.0e-12))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def status(tag: str) -> dict:
    path = OUT / tag / "fit_status.json"
    require(path.is_file(), f"missing fit status: {tag}")
    return json.loads(path.read_text())


def fnp_vector(tag: str) -> np.ndarray:
    frame = pd.read_csv(OUT / tag / "fnp_grid.csv")
    mask = np.isclose(frame.x, 0.1) & frame.bT.between(0.1, 4.0)
    return frame.loc[mask].sort_values("bT").F_NP.to_numpy(float)


def verify_objective(tag: str, *, strength: float, bmax: float,
                     barrier_strength: float, barrier_power: int,
                     replica_seed: int | None,
                     fit_seed: int | None = None,
                     expected_barrier_ceiling: float | None = None) -> None:
    fit = status(tag)
    reference = fit["regularization"]["fnp_reference_distance"]
    barrier = fit["regularization"]["fit_quality_barrier"]
    nonzero_regularizers = {
        name for name, spec in fit["regularization"].items()
        if isinstance(spec, dict) and "lambda" in spec
        and not np.isclose(float(spec["lambda"]), 0.0)
    }
    profile = fit.get("point_profile", {})
    complexity = fit.get("model_complexity", {})
    require(bool_value(fit["convergence_gate_pass"]),
            f"fit convergence failed: {tag}")
    require(not bool_value(fit["production_state_modified"]),
            f"endpoint reports production mutation: {tag}")
    require(exact_numeric(reference["lambda"], strength)
            and exact_numeric(reference["b_min"], 0.1)
            and exact_numeric(reference["b_max"], bmax),
            f"reference-prior provenance mismatch: {tag}")
    require(exact_numeric(barrier["lambda"], barrier_strength)
            and int(barrier["power"]) == barrier_power,
            f"fit-barrier provenance mismatch: {tag}")
    require(Path(fit["source_production"]).resolve() ==
            SOURCE_PRODUCTION.resolve()
            and Path(fit["w_grid"]).resolve() == W_GRID.resolve()
            and Path(reference["target_csv"]).resolve() == REFERENCE.resolve(),
            f"source/reference provenance mismatch: {tag}")
    require(nonzero_regularizers == {
                "fnp_reference_distance", "fit_quality_barrier"}
            and fit.get("model_constraint", {}).get("kind") == "none",
            f"unexpected extra model constraint/regularizer: {tag}")
    require(exact_numeric(fit["regularization"]
                ["likelihood_weight"]["value"], 1.0)
            and not bool_value(profile.get("enabled"))
            and exact_numeric(profile.get("lambda_per_row"), 0.0)
            and int(complexity.get("np_width", -1)) == 48
            and int(complexity.get("np_cond_width", -1)) == 32
            and int(complexity.get("np_blocks", -1)) == 3
            and complexity.get("global_spline_nx") is None
            and complexity.get("global_spline_nb") is None
            and int(complexity.get("distill_accepted_steps", -1)) == 0
            and int(complexity.get("distill_prediction_steps", -1)) == 0,
            f"likelihood/profile/model fingerprint mismatch: {tag}")
    observed = fit.get("replica_seed")
    require((observed is None and replica_seed is None)
            or (observed is not None and replica_seed is not None
                and int(observed) == int(replica_seed)),
            f"replica identity mismatch: {tag}")
    if fit_seed is not None:
        require(int(fit["seed"]) == fit_seed,
                f"fit-seed identity mismatch: {tag}")
    if replica_seed is not None and barrier_strength > 0:
        row_count = int(fit["row_count"])
        expected_ceiling = row_count + 5.0 * math.sqrt(2.0 * row_count)
        require(exact_numeric(barrier["ceiling_total_chi2"],
                              expected_ceiling),
                f"replica barrier ceiling mismatch: {tag}")
    if expected_barrier_ceiling is not None:
        require(exact_numeric(barrier["ceiling_total_chi2"],
                              expected_barrier_ceiling),
                f"declared barrier ceiling mismatch: {tag}")


def bool_value(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise RuntimeError(f"value is not an explicit boolean: {value!r}")


def verify_terminal_window(group: pd.DataFrame, endpoint: str,
                           pass_column: str) -> dict:
    group = group.sort_values("cumulative_lbfgs_iterations")
    require(not group.duplicated("cumulative_lbfgs_iterations").any(),
            f"duplicate continuation checkpoints for {endpoint}")
    anchors = group[group.cumulative_lbfgs_iterations.eq(ANCHOR)]
    require(len(anchors) == 1,
            f"missing unique 200k requested-capacity anchor for {endpoint}")
    terminal = group[group.tag.astype(str).eq(endpoint)]
    require(len(terminal) == 1, f"missing unique terminal ledger row: {endpoint}")
    row = terminal.iloc[0]
    require(bool_value(row[pass_column]),
            f"terminal stationarity flag failed: {endpoint}")
    cumulative = int(row.cumulative_lbfgs_iterations)
    require(cumulative >= ANCHOR + REQUIRED * 5000,
            f"terminal lacks 50k post-anchor requested capacity: {endpoint}")
    post = group[(group.cumulative_lbfgs_iterations > ANCHOR)
                 & (group.cumulative_lbfgs_iterations <= cumulative)].tail(REQUIRED)
    require(len(post) == REQUIRED
            and np.array_equal(post.cumulative_lbfgs_iterations.to_numpy(int),
                               np.arange(cumulative - 45_000,
                                         cumulative + 1, 5000)),
            f"terminal quiet window is not ten contiguous blocks: {endpoint}")
    require(post.eligible_post_mandatory_confirmation.map(bool_value).all(),
            f"terminal window includes ineligible blocks: {endpoint}")
    require("stationarity_window_anchor_iterations" in post.columns,
            f"terminal ledger lacks restartable-window provenance: {endpoint}")
    active_anchors = post.stationarity_window_anchor_iterations.to_numpy(float)
    require(np.all(np.isfinite(active_anchors))
            and len(np.unique(active_anchors)) == 1,
            f"terminal blocks do not share one stationarity anchor: {endpoint}")
    active_anchor = int(active_anchors[0])
    require(active_anchor >= ANCHOR and cumulative - active_anchor >= 50_000,
            f"terminal active window is shorter than 50k: {endpoint}")
    anchor_rows = group[group.cumulative_lbfgs_iterations.eq(active_anchor)]
    require(len(anchor_rows) == 1,
            f"active stationarity anchor row is missing: {endpoint}")
    adjacent = post.fnp_drift_from_previous_chunk.to_numpy(float)
    fixed = post.post_mandatory_window_fnp_drift.to_numpy(float)
    require(np.all(np.isfinite(adjacent)) and np.all(adjacent <= DRIFT_GATE),
            f"terminal adjacent drift exceeds 2%: {endpoint}")
    require(np.all(np.isfinite(fixed)) and np.all(fixed <= DRIFT_GATE),
            f"terminal active-anchor drift exceeds 2%: {endpoint}")
    anchor_vector = fnp_vector(str(anchor_rows.iloc[0].tag))
    endpoint_vector = fnp_vector(endpoint)
    recomputed = float(np.max(np.abs(endpoint_vector - anchor_vector) /
                              np.maximum(anchor_vector, 0.05)))
    require(recomputed <= DRIFT_GATE,
            f"recomputed terminal active-anchor drift exceeds 2%: {endpoint}")
    require(int(row.consecutive_quiet_blocks) >= REQUIRED,
            f"terminal quiet-block counter is incomplete: {endpoint}")
    return {
        "endpoint": endpoint,
        "cumulative_iterations": cumulative,
        "active_window_anchor_iterations": active_anchor,
        "max_adjacent_drift_last_50k": float(np.max(adjacent)),
        "max_active_anchor_drift_last_50k": float(np.max(fixed)),
        "recomputed_terminal_active_anchor_drift": recomputed,
    }


def main() -> None:
    subprocess.run([sys.executable, str(BASE / "scripts/audit_frozen_inputs.py")],
                   check=True, stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable, str(STATE_CHAIN_SCRIPT)],
                   check=True, stdout=subprocess.DEVNULL)
    frozen = load("frozen_input_audit/summary.json")
    state_chain = load("lambda600_state_chain_audit/summary.json")
    terminal_evidence = load("lambda600_terminal_evidence/summary.json")
    comparison = load("lambda600_vs_lambda1_diagnostic/summary.json")
    starts = load("replica_robust_reference_full24/summary.json")
    replicas = load("selected_reference_central_replicas/summary.json")
    ensemble = load("final_combined_tmd_ensemble/summary.json")
    stability = load("final_combined_ensemble_stability/summary.json")
    postfit_tail, postfit_tail_hash = validated_postfit_tail_audit(
        POSTFIT_TAIL_AUDIT,
        SUM / "final_combined_ensemble_stability/summary.json",
    )
    nested_interaction, nested_interaction_hash = validated_nested_interaction(
        NESTED_INTERACTION,
        POSTFIT_TAIL_AUDIT,
        SUM / "final_combined_ensemble_stability/summary.json",
    )
    final_envelope, final_envelope_hash = validated_final_directional_envelope(
        FINAL_DIRECTIONAL_ENVELOPE,
        NESTED_INTERACTION,
        POSTFIT_TAIL_AUDIT,
        SUM / "final_combined_ensemble_stability/summary.json",
    )
    figures = load("final_fig2_fig6/summary.json")
    incumbent = load(
        "champion_registry/empirical_reference_lambda1_b0p1_2p0_full24.json")
    current = load("champion_registry/current.json")
    inventory = load("campaign_runs_status.json")
    family = load("final_constraint_family_decision/decision.json")

    require(frozen["status"] == "pass"
            and frozen["registered_input_count"] == frozen["unchanged_input_count"],
            "frozen-input audit failed")
    state_chain_hash = sha256(STATE_CHAIN_AUDIT)
    start_chain_hash = sha256(START_CHAIN_AUDIT)
    chain_prescription = state_chain.get("selected_prescription", {})
    central_chain_evidence = state_chain.get("central_chain", {})
    require(state_chain.get("status") == "pass"
            and int(state_chain.get("lambda300_source_count", 0)) == 24
            and int(state_chain.get("start_chain_count", 0)) == 24
            and int(state_chain.get("central_chain_count", 0)) == 1
            and int(state_chain.get(
                "experimental_replica_chain_count", 0)) == 50
            and exact_numeric(chain_prescription.get(
                "reference_strength"), 600.0)
            and exact_numeric(chain_prescription.get(
                "fit_quality_barrier_strength"), 100.0)
            and int(chain_prescription.get(
                "fit_quality_barrier_power", -1)) == 2
            and Path(state_chain.get(
                "terminal_start_ancestry_audit", "")).resolve()
                == START_CHAIN_AUDIT.resolve()
            and state_chain.get(
                "terminal_start_ancestry_audit_sha256") == start_chain_hash
            and int(central_chain_evidence.get(
                "terminal_cumulative_requested_capacity", -1)) == 300_000
            and bool_value(central_chain_evidence.get(
                "restart_state_and_norm_content_ancestry_recorded"))
            and not bool_value(state_chain.get("production_sources_modified")),
            "lambda=600 state-chain audit failed")
    require(terminal_evidence.get("status") == "pass"
            and int(terminal_evidence.get("start_count", 0)) == 24
            and int(terminal_evidence.get("central_count", 0)) == 1
            and int(terminal_evidence.get("replica_count", 0)) == 50
            and terminal_evidence.get("comparison_champion_id") ==
                "empirical_reference_lambda1_b0p1_2p0_full24"
            and bool_value(terminal_evidence.get(
                "candidate_endpoint_gate_pass"))
            and comparison.get("status") ==
                "complete_validated_candidate_comparison"
            and comparison.get("comparison_champion_id") ==
                "empirical_reference_lambda1_b0p1_2p0_full24"
            and bool_value(comparison.get(
                "immutable_incumbent_hashes_validated"))
            and not bool_value(terminal_evidence.get(
                "production_sources_modified"))
            and not bool_value(terminal_evidence.get("registry_modified")),
            "terminal Fig2/Fig6/comparison evidence audit failed")
    terminal_bound_artifacts = terminal_evidence.get("artifact_sha256")
    require(isinstance(terminal_bound_artifacts, dict)
            and terminal_bound_artifacts,
            "terminal evidence manifest lacks artifact hashes")
    for path_text, expected_hash in terminal_bound_artifacts.items():
        path = Path(path_text)
        require(path.is_file() and sha256(path) == expected_hash,
                f"terminal evidence artifact changed: {path}")
    require(incumbent.get("champion_id") ==
            "empirical_reference_lambda1_b0p1_2p0_full24",
            "immutable lambda1 incumbent record is invalid")
    require(current.get("champion_id") == incumbent.get("champion_id"),
            "current champion changed during the lambda600 comparison")
    incumbent_band = Path(incumbent["artifacts"]["kspace_combined_bands"])
    require(incumbent_band.is_file()
            and sha256(incumbent_band) == incumbent["artifact_sha256"][
                "kspace_combined_bands"],
            "immutable lambda1 incumbent band hash mismatch")
    require(inventory.get("all_report_production_unmodified", False),
            "trial inventory reports a production mutation")
    require(int(inventory.get("completed_run_count", 0)) >= 347,
            "systematic trial inventory is unexpectedly incomplete")
    require(family.get("status") ==
            "no_constraint_family_passes_all_promotion_gates",
            "constraint-family study evidence is incomplete")

    require(starts.get("status") == "complete"
            and starts.get("all_starts_fnp_plateaued_and_fit_preserved"),
            "24-start candidate is not complete and valid")
    strength = float(starts["selected_strength"])
    bmax = float(starts["selected_bmax"])
    barrier_strength = float(starts.get("fit_quality_barrier_strength", 0.0))
    barrier_power = int(starts.get("fit_quality_barrier_power", 2))
    require(exact_numeric(strength, 600.0) and exact_numeric(bmax, 4.0)
            and exact_numeric(barrier_strength, 100.0) and barrier_power == 2,
            "completion audit received a different prescription")
    start_tags = list(starts["endpoint_tags"])
    require(len(start_tags) == 24 and len(set(start_tags)) == 24,
            "24-start endpoint coverage is not exact")
    start_ledger = pd.read_csv(SUM / "replica_robust_reference_full24/runs.csv")
    source_ensemble = json.loads(START_SOURCE_ENSEMBLE.read_text())
    source_tags = [str(value) for value in source_ensemble.get(
        "endpoint_tags", [])]
    require(source_ensemble.get("status") == "complete"
            and len(source_tags) == 24 and len(set(source_tags)) == 24,
            "lambda300 start-source ensemble is incomplete")
    start_windows = []
    for seed, tag, source_tag in zip(
            range(303, 327), start_tags, source_tags, strict=True):
        group = start_ledger[start_ledger.seed.eq(seed)]
        start_windows.append(verify_terminal_window(
            group, tag, "stationarity_and_fit_pass"))
        terminal = group[group.tag.astype(str).eq(tag)].iloc[0]
        require(bool_value(terminal.passes_natural_chi2_scale),
                f"start endpoint fails natural fit scale: {tag}")
        source_fit = status(source_tag)
        expected_ceiling = (
            float(source_fit["final"]["unpenalized_total_chi2"])
            + math.sqrt(2.0 * int(source_fit["row_count"])))
        verify_objective(tag, strength=strength, bmax=bmax,
                         barrier_strength=barrier_strength,
                         barrier_power=barrier_power, replica_seed=None,
                         fit_seed=seed,
                         expected_barrier_ceiling=expected_ceiling)

    require(replicas.get("status") == "complete"
            and replicas.get("central_fnp_plateau_pass")
            and replicas.get("all_replicas_fnp_plateaued")
            and bool_value(replicas.get("central_full_horizon_complete"))
            and int(replicas.get(
                "central_full_horizon_requested_capacity", -1)) == 300_000
            and int(replicas.get(
                "central_terminal_requested_capacity", -1)) == 300_000
            and bool_value(replicas.get(
                "restart_state_and_norm_content_ancestry_recorded"))
            and bool_value(replicas.get(
                "fit_target_realizations_content_validated")),
            "central/50-replica campaign is incomplete")
    require(exact_numeric(replicas["selected_strength"], strength)
            and exact_numeric(replicas["selected_bmax"], bmax)
            and exact_numeric(replicas["fit_quality_barrier_strength"],
                              barrier_strength)
            and int(replicas["fit_quality_barrier_power"]) == barrier_power,
            "replica prescription differs from start prescription")
    replica_tags = list(replicas["replica_endpoint_tags"])
    require(len(replica_tags) == 50 and len(set(replica_tags)) == 50,
            "50-replica endpoint coverage is not exact")
    ledger = pd.read_csv(SUM / "selected_reference_central_replicas/runs.csv")
    central_tag = str(replicas["central_endpoint_tag"])
    central_group = ledger[ledger.kind.astype(str).eq("central")]
    central_window = verify_terminal_window(
        central_group, central_tag, "fnp_plateau_gate_pass")
    require(int(central_window["cumulative_iterations"]) == 300_000,
            "central endpoint did not exhaust the exact 300k horizon")
    central_terminal = central_group[
        central_group.tag.astype(str).eq(central_tag)].iloc[0]
    require("fit_quality_gate_pass" in central_group.columns
            and bool_value(central_terminal.fit_quality_gate_pass)
            and bool_value(replicas.get("central_fit_quality_gate_pass")),
            "central endpoint fails the explicit fit-preservation gate")
    initializer = status(str(replicas["central_initializer_tag"]))
    central_ceiling = (float(initializer["final"]["unpenalized_total_chi2"])
                       + math.sqrt(2.0 * int(initializer["row_count"])))
    require(exact_numeric(replicas[
                "central_fit_quality_ceiling_total_chi2"], central_ceiling)
            and float(replicas["central_unpenalized_total_chi2"])
                <= central_ceiling,
            "central fit-preservation summary is invalid")
    verify_objective(central_tag, strength=strength, bmax=bmax,
                     barrier_strength=barrier_strength,
                     barrier_power=barrier_power, replica_seed=None,
                     expected_barrier_ceiling=central_ceiling)
    replica_windows = []
    for replica_seed, tag in zip(range(1001, 1051), replica_tags, strict=True):
        group = ledger[(ledger.kind.astype(str).eq("experimental_replica"))
                       & ledger.replica_seed.eq(replica_seed)]
        replica_windows.append(verify_terminal_window(
            group, tag, "fnp_plateau_gate_pass"))
        terminal = group[group.tag.astype(str).eq(tag)].iloc[0]
        require("fit_quality_gate_pass" in group.columns
                and bool_value(terminal.fit_quality_gate_pass),
                f"replica endpoint fails explicit fit-quality gate: {tag}")
        verify_objective(tag, strength=strength, bmax=bmax,
                         barrier_strength=barrier_strength,
                         barrier_power=barrier_power,
                         replica_seed=replica_seed,
                         fit_seed=2001 + replica_seed - 1001)
    require(replicas.get("all_stress_replicas_cross_optimizer_consistent")
            and replicas.get("cross_optimizer_checks")
            and all(item["agreement_pass"]
                    for item in replicas["cross_optimizer_checks"]),
            "cross-optimizer replica agreement is incomplete")
    stress_tags = []
    for item in replicas["cross_optimizer_checks"]:
        replica_seed = int(item["replica_seed"])
        primary = str(item["primary_endpoint_tag"])
        independent = str(item["independent_stress_endpoint_tag"])
        require(primary in replica_tags,
                f"cross-optimizer primary is outside the 50 replicas: {primary}")
        verify_objective(independent, strength=strength, bmax=bmax,
                         barrier_strength=barrier_strength,
                         barrier_power=barrier_power,
                         replica_seed=replica_seed)
        stress_tags.append(independent)
    require(len(stress_tags) == len(set(stress_tags)),
            "cross-optimizer stress endpoints are duplicated")

    require(ensemble.get("status") == "complete"
            and int(ensemble["start_count"]) == 24
            and int(ensemble["experimental_replica_count"]) == 50
            and int(ensemble["combined_member_count"]) == 1200
            and "empirical q16--q84" in ensemble.get("interval", "")
            and "neither a formal 68%" in ensemble.get(
                "interval_probability_semantics", "")
            and "operational separability assumption" in ensemble.get(
                "hierarchical_transfer_assumption", "")
            and ensemble.get("joint_nested_start_by_replica_refits_performed") is False
            and ensemble.get("independent_sampling_axis_counts") == {
                "optimizer_starts": 24, "experimental_replicas": 50}
            and int(ensemble.get(
                "available_same_replica_cross_optimizer_check_count", -1))
                == len(replicas.get("cross_optimizer_checks", []))
            and bool_value(ensemble.get("state_chain_gate_pass"))
            and Path(ensemble.get("state_chain_audit", "")).resolve()
                == STATE_CHAIN_AUDIT.resolve()
            and ensemble.get("state_chain_audit_sha256") == state_chain_hash,
            "hierarchical 24x50 propagation is incomplete")
    require(stability.get("status") == "complete"
            and stability.get("coverage_gate_pass")
            and stability.get("band_integrity_gate_pass")
            and bool_value(stability.get("state_chain_gate_pass"))
            and Path(stability.get("state_chain_audit", "")).resolve()
                == STATE_CHAIN_AUDIT.resolve()
            and stability.get("state_chain_audit_sha256") == state_chain_hash
            and final_promotion_gate(final_envelope),
            "final ensemble stability/improvement audit failed")
    widths = {key: float(final_envelope[
        "width_metrics_by_flavor"][key][
            "joint_convergence_interaction_raw_full_width"])
        for key in ("u", "d")}
    allowance = {key: float(final_envelope[
        "width_metrics_by_flavor"][key][
            "corrected_finite_sampling_full_width_margin"])
        for key in ("u", "d")}
    previous = {key: float(stability[
        "comparison_champion_max_active_relative_full_width"][key])
        for key in ("u", "d")}
    union_diagnostic = {key: float(stability[
        "comparison_champion_union_mask_relative_full_width"][key])
        for key in ("u", "d")}
    require(stability.get("comparison_champion_id") == incumbent.get("champion_id"),
            "stability audit compared against a different incumbent")
    require(all(exact_numeric(previous[key], LOCKED_INCUMBENT_WIDTHS[key])
                and exact_numeric(previous[key], incumbent[
                    "combined_fig6_max_active_relative_full_width"][key])
                for key in ("u", "d")),
            "promotion threshold is not the immutable registered lambda1 width")
    require(all(np.isfinite(union_diagnostic[key])
                and union_diagnostic[key] > 0.0 for key in ("u", "d")),
            "secondary union-mask incumbent diagnostic is incomplete")
    require(stability.get("comparison_active_mask_definition")
            and stability.get("comparison_semantics"),
            "common-mask incumbent comparison metadata is missing")
    require(all(widths[key] + allowance[key] < previous[key]
                for key in ("u", "d")),
            "candidate does not robustly improve both incumbent flavors")
    require(figures.get("status") == "final_validated_figures"
            and figures.get("updated_only")
            and not figures.get("contains_individual_seed_curves")
            and not figures.get("contains_legacy_conditional_result")
            and figures.get("formal_confidence_level_assigned") is False
            and figures.get("one_sigma_claimed") is False
            and "empirical product band plus residual convergence/interaction envelope"
                in figures.get("uncertainty", ""),
            "final Fig. 2/Fig. 6 metadata is invalid")
    figure_paths = [
        SUM / "final_fig2_fig6/updated_fnp_bspace_product_plus_directional_envelope.png",
        SUM / "final_fig2_fig6/updated_fig2_bspace_product_plus_directional_envelope.png",
        SUM / "final_fig2_fig6/updated_fig6_kspace_ud_product_plus_directional_envelope.png",
    ]
    require(all(path.is_file() and path.stat().st_size > 0
                for path in figure_paths), "validated figure artifacts are missing")
    evidence_paths = [
        SUM / "replica_robust_reference_full24/runs.csv",
        SUM / "replica_robust_reference_full24/summary.json",
        STATE_CHAIN_AUDIT,
        TERMINAL_EVIDENCE,
        COMPARISON,
        SUM / "selected_reference_central_replicas/runs.csv",
        SUM / "selected_reference_central_replicas/summary.json",
        SUM / "final_combined_tmd_ensemble/summary.json",
        SUM / "final_combined_tmd_ensemble/fnp_bands.csv",
        SUM / "final_combined_tmd_ensemble/bT_tmd_bands.csv",
        SUM / "final_combined_tmd_ensemble/kT_tmd_bands.csv",
        SUM / "final_combined_tmd_ensemble/kT_tmd_ensemble_long.csv",
        SUM / "final_combined_ensemble_stability/summary.json",
        POSTFIT_TAIL_AUDIT,
        *[Path(path) for path in postfit_tail.get("artifacts", {}).values()],
        NESTED_INTERACTION,
        *[Path(path) for path in nested_interaction.get("artifacts", {}).values()],
        FINAL_DIRECTIONAL_ENVELOPE,
        *[Path(path) for path in final_envelope.get("artifacts", {}).values()],
        SUM / "final_fig2_fig6/summary.json",
        SUM / "harmonized_lambda1_logfnp_24x50_comparator/summary.json",
        SUM / "harmonized_lambda1_logfnp_24x50_comparator/input_provenance.json",
        SUM / "harmonized_lambda1_logfnp_24x50_comparator/kspace_combined_bands.csv",
        BASE / "manifests/harmonized_lambda1_inputs.json",
        incumbent_band,
        *figure_paths,
        *[Path(path) for path in terminal_bound_artifacts],
    ]
    audited_endpoint_tags = (
        start_tags + [central_tag] + replica_tags + stress_tags)
    for tag in audited_endpoint_tags:
        evidence_paths.extend([
            OUT / tag / "fit_status.json",
            OUT / tag / "fnp_grid.csv",
        ])
    require(all(path.is_file() and path.stat().st_size > 0
                for path in evidence_paths), "audited evidence artifact is missing")

    payload = {
        "status": "complete",
        "audit_protocol": "lambda600 strengthened forty sequential 5k-capped continuation blocks plus restartable contiguous ten-block active-anchor stationarity window",
        "iteration_accounting": "200k/50k labels denote requested LBFGS max-iteration capacity; executed closure evaluations are recorded but are not equivalent to iterations",
        "selected_strength": strength,
        "selected_bmax": bmax,
        "fit_quality_barrier_strength": barrier_strength,
        "fit_quality_barrier_power": barrier_power,
        "start_count": 24,
        "experimental_replica_count": 50,
        "combined_member_count": 1200,
        "start_terminal_windows": start_windows,
        "central_terminal_window": central_window,
        "replica_terminal_windows": replica_windows,
        "final_fig6_max_active_relative_full_width": widths,
        "resampling_full_width_allowance_by_flavor": allowance,
        "incumbent_fig6_max_active_relative_full_width": previous,
        "figure_sha256": {path.name: sha256(path) for path in figure_paths},
        "state_chain_gate_pass": True,
        "state_chain_audit": str(STATE_CHAIN_AUDIT),
        "state_chain_audit_sha256": state_chain_hash,
        "terminal_evidence_gate_pass": True,
        "terminal_evidence_audit": str(TERMINAL_EVIDENCE),
        "terminal_evidence_audit_sha256": sha256(TERMINAL_EVIDENCE),
        "postfit_tail_transform_gate_pass": True,
        "postfit_tail_transform_audit": str(POSTFIT_TAIL_AUDIT),
        "postfit_tail_transform_audit_sha256": postfit_tail_hash,
        "nested_interaction_validation_gate_pass": True,
        "nested_interaction_validation": str(NESTED_INTERACTION),
        "nested_interaction_validation_sha256": nested_interaction_hash,
        "final_directional_envelope_gate_pass": True,
        "final_directional_envelope": str(FINAL_DIRECTIONAL_ENVELOPE),
        "final_directional_envelope_sha256": final_envelope_hash,
        "evidence_sha256": {str(path): sha256(path) for path in evidence_paths},
        "frozen_input_audit": frozen,
        "production_sources_modified": False,
    }
    TARGET.mkdir(parents=True, exist_ok=True)
    (TARGET / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
