#!/usr/bin/env python3
"""Write the terminal fixed-challenger lambda600-versus-lambda1 report.

The writer has two fail-closed entry points:

* With no arguments it requires an already published, validated terminal
  ``lambda600_like_for_like_decision``.
* With ``--decision-status`` it validates the complete standalone upstream
  graph and the proposed terminal status before that status is published.

The second form lets the controller create and hash the report before its
atomic terminal decision write.  Neither form runs fits, changes frozen
production inputs, or writes the champion registry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Iterable

import numpy as np

from alternate_lambda_authorization import (
    validate_complete_lambda600_comparison,
)
from fixed_challenger_protocol import (
    EXPECTED_FNP_REFERENCE_SHA256,
    EXPECTED_IMPLEMENTATION_SHA256,
    FNP_REFERENCE,
    INCUMBENT_ID,
    LOCKED_INCUMBENT_WIDTHS,
    PROTOCOL,
    require_fixed_implementation_binding,
    validate_fixed_challenger_protocol,
)
from nested_interaction_and_final_envelope_validation import (
    final_promotion_gate,
    validated_final_directional_envelope,
    validated_nested_interaction,
)
from postfit_tail_transform_validation import validated_postfit_tail_audit


BASE = Path(__file__).resolve().parents[1]
SUM = BASE / "summaries"
TARGET = SUM / "final_study_report"

DECISION = SUM / "lambda600_like_for_like_decision/summary.json"
STARTS = SUM / "replica_robust_reference_full24/summary.json"
START_CHAIN_AUDIT = SUM / "lambda600_start_chain_audit/summary.json"
START_SEAL_ROOT = SUM / "lambda600_start_chain_audit/current_byte_seals"
REPLICAS = SUM / "selected_reference_central_replicas/summary.json"
PRODUCT = SUM / "final_combined_tmd_ensemble/summary.json"
STABILITY = SUM / "final_combined_ensemble_stability/summary.json"
POSTFIT = SUM / "lambda600_postfit_tail_transform_audit/summary.json"
NESTED = SUM / "lambda600_nested_start_replica_interaction/summary.json"
FINAL_ENVELOPE = SUM / "lambda600_final_directional_envelope/summary.json"
FIGURES = SUM / "final_fig2_fig6/summary.json"
FIGURE_DIR = FIGURES.parent
COMPARISON = SUM / "lambda600_vs_lambda1_diagnostic/summary.json"
TERMINAL_EVIDENCE = SUM / "lambda600_terminal_evidence/summary.json"
COMPLETION = SUM / "campaign_completion_audit/summary.json"
FROZEN_AUDIT = SUM / "frozen_input_audit/summary.json"
INCUMBENT = (
    SUM / "champion_registry/empirical_reference_lambda1_b0p1_2p0_full24.json"
)
REGISTRY = SUM / "champion_registry"
FROZEN_MANIFEST = BASE / "manifests/input_files.json"

PROMOTED = "candidate_promoted_as_new_study_champion"
REJECTED = "candidate_rejected"
TERMINAL_STATUSES = (PROMOTED, REJECTED)
FLAVORS = ("u", "d")
EXPECTED_PRESCRIPTION = {
    "reference_strength": 600.0,
    "reference_bmax": 4.0,
    "fit_quality_barrier_strength": 100.0,
    "fit_quality_barrier_power": 2,
}
VALIDATED_STEMS = {
    "fnp": "updated_fnp_bspace_product_plus_directional_envelope",
    "fig2": "updated_fig2_bspace_product_plus_directional_envelope",
    "fig6": "updated_fig6_kspace_ud_product_plus_directional_envelope",
}
DIAGNOSTIC_STEMS = {
    "fnp": "diagnostic_failed_fnp_bspace_product_plus_directional_envelope",
    "fig2": "diagnostic_failed_fig2_bspace_product_plus_directional_envelope",
    "fig6": "diagnostic_failed_fig6_kspace_ud_product_plus_directional_envelope",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, label: str) -> dict:
    require(path.is_file() and path.stat().st_size > 0,
            f"missing {label}: {path}")
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid JSON for {label}: {path}") from error
    require(isinstance(payload, dict), f"{label} is not a JSON object")
    return payload


def explicit_bool(value: object, label: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise RuntimeError(f"{label} is not an explicit boolean")


def exact_number(value: object, expected: float, label: str) -> float:
    try:
        observed = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{label} is not numeric") from error
    require(math.isfinite(observed) and math.isclose(
        observed, float(expected), rel_tol=0.0, abs_tol=1.0e-12),
        f"{label}={observed!r}, expected {expected!r}")
    return observed


def pct(value: float) -> str:
    return f"{100.0 * float(value):.3f}%"


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=f".{path.name}.", suffix=".tmp",
            dir=path.parent, delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: dict) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def validate_hash_map(paths: dict, hashes: dict, label: str,
                      parent: Path | None = None) -> dict[str, Path]:
    require(isinstance(paths, dict) and paths,
            f"{label} artifact mapping is missing")
    require(isinstance(hashes, dict) and set(hashes) == set(paths),
            f"{label} artifact/hash coverage differs")
    resolved: dict[str, Path] = {}
    for role, path_text in paths.items():
        path = Path(path_text)
        if not path.is_absolute() and parent is not None:
            path = parent / path
        require(path.is_file() and path.stat().st_size > 0,
                f"{label} artifact is missing ({role}): {path}")
        require(sha256(path) == str(hashes[role]),
                f"{label} artifact changed ({role}): {path}")
        resolved[str(role)] = path
    return resolved


def validate_path_hash_map(mapping: dict, label: str) -> None:
    require(isinstance(mapping, dict) and mapping,
            f"{label} path/hash mapping is missing")
    for path_text, expected in mapping.items():
        path = Path(path_text)
        require(path.is_file() and path.stat().st_size > 0,
                f"{label} artifact is missing: {path}")
        require(str(expected) == sha256(path),
                f"{label} artifact changed: {path}")


def expected_figure_paths(outcome: str) -> dict[str, Path]:
    require(outcome in TERMINAL_STATUSES, f"unknown outcome: {outcome}")
    stems = VALIDATED_STEMS if outcome == PROMOTED else DIAGNOSTIC_STEMS
    return {
        role: FIGURE_DIR / f"{stem}.pdf"
        for role, stem in stems.items()
    }


def validate_prescription(payload: dict, label: str) -> None:
    exact_number(payload.get("selected_strength"), 600.0,
                 f"{label} selected_strength")
    exact_number(payload.get("selected_bmax"), 4.0,
                 f"{label} selected_bmax")
    exact_number(payload.get("fit_quality_barrier_strength"), 100.0,
                 f"{label} fit_quality_barrier_strength")
    exact_number(payload.get("fit_quality_barrier_power"), 2.0,
                 f"{label} fit_quality_barrier_power")


def validate_start_ancestry_limitation(start_audit: dict,
                                       starts: dict) -> dict:
    """Validate and preserve the pre-receipt start-chain assurance boundary."""
    require(start_audit.get("status") == "pass"
            and int(start_audit.get("start_chain_count", -1)) == 24,
            "terminal start-chain ancestry audit is incomplete")
    selection = start_audit.get("central_initializer_selection", {})
    require(selection.get("ordered_start_tags") == starts.get("endpoint_tags")
            and int(selection.get("member_count", -1)) == 24,
            "terminal start-chain audit covers a different start generation")
    seal = start_audit.get("terminal_start_current_byte_seal")
    require(isinstance(seal, dict),
            "terminal start-chain audit lacks its current-byte seal")
    path = Path(str(seal.get("path", "")))
    launch_count = int(seal.get("launch_time_receipt_checkpoint_count", -1))
    legacy_count = int(seal.get("legacy_pre_receipt_checkpoint_count", -1))
    checkpoint_count = int(seal.get("checkpoint_count", -1))
    limitation = str(seal.get("historical_ancestry_limitation", "")).strip()
    require(seal.get("status") in {
                "sealed_current_bytes_with_disclosed_legacy_limit",
                "sealed_launch_time_receipt_graph"}
            and path.parent.resolve() == START_SEAL_ROOT.resolve()
            and path.is_file() and path.stat().st_size > 0
            and str(seal.get("sha256")) == sha256(path)
            and checkpoint_count > 0
            and launch_count >= 0 and legacy_count >= 0
            and launch_count + legacy_count == checkpoint_count
            and bool(limitation),
            "terminal start current-byte seal is invalid")
    exact_all = explicit_bool(start_audit.get(
        "exact_launch_time_ancestry_proven_for_all_start_checkpoints"),
        "all-start launch-time ancestry flag")
    require(exact_all == (legacy_count == 0)
            and start_audit.get("legacy_start_ancestry_limitation")
                == limitation,
            "start ancestry assurance/limitation semantics disagree")
    return {
        "status": seal["status"],
        "audit": str(START_CHAIN_AUDIT),
        "audit_sha256": sha256(START_CHAIN_AUDIT),
        "seal": str(path),
        "seal_sha256": seal["sha256"],
        "checkpoint_count": checkpoint_count,
        "launch_time_receipt_checkpoint_count": launch_count,
        "legacy_pre_receipt_checkpoint_count": legacy_count,
        "exact_launch_time_ancestry_proven_for_all_start_checkpoints": exact_all,
        "historical_ancestry_limitation": limitation,
    }


def validate_width_evidence(
    stability: dict, final_envelope: dict,
) -> tuple[dict[str, dict], bool]:
    """Validate same-statistic resampling and the final replacement gate."""
    require(stability.get("status") == "complete",
            "final product stability audit is incomplete")
    require(int(stability.get("start_count", -1)) == 24
            and int(stability.get("replica_count", -1)) == 50,
            "final product audit lacks exact 24x50 coverage")
    require(explicit_bool(stability.get("coverage_gate_pass"),
                          "stability coverage gate")
            and explicit_bool(stability.get("band_integrity_gate_pass"),
                              "stability band-integrity gate")
            and explicit_bool(stability.get("state_chain_gate_pass"),
                              "stability state-chain gate"),
            "final product audit lacks valid coverage/band/state provenance")
    require(explicit_bool(final_envelope.get(
                "final_joint_sampling_gate_authoritative"),
                "final joint sampling authority")
            and not explicit_bool(final_envelope.get(
                "prior_product_median_sampling_gate_authoritative"),
                "prior product-median sampling authority"),
            "the exact final-statistic sampling gate is not authoritative")

    widths = final_envelope.get("width_metrics_by_flavor")
    require(isinstance(widths, dict) and set(widths) == set(FLAVORS),
            "final envelope lacks exact u/d width metrics")
    locked = stability.get(
        "comparison_champion_max_active_relative_full_width", {})
    require(stability.get("comparison_champion_id") == INCUMBENT_ID
            and isinstance(locked, dict) and set(locked) == set(FLAVORS),
            "final audit does not use the immutable lambda1 incumbent")

    resampling = final_envelope.get("final_statistic_resampling")
    require(isinstance(resampling, dict)
            and int(resampling.get("bootstrap_replicates", -1)) == 300
            and int(resampling.get("split_half_replicates", -1)) == 200
            and isinstance(resampling.get("rng_seed"), int),
            "exact final-statistic resampling coverage is incomplete")
    require("absolute fixed trained-300k central" in str(
                resampling.get("statistic", ""))
            and "identical resampled" in str(
                resampling.get("correlation_preserved", ""))
            and not explicit_bool(resampling.get("interaction_resampled"),
                                  "interaction_resampled"),
            "final-statistic resampling semantics are incomplete")
    direct_inputs = {
        "bootstrap": resampling.get(
            "bootstrap_p95_absolute_deviation_by_flavor", {}),
        "start_split": resampling.get(
            "start_split_p95_absolute_difference_by_flavor", {}),
        "replica_split": resampling.get(
            "replica_split_p95_absolute_difference_by_flavor", {}),
        "joint_split": resampling.get(
            "joint_split_p95_absolute_difference_by_flavor", {}),
    }
    declared_allowance = resampling.get("allowance_by_flavor", {})
    exact_full = resampling.get("full_exact_final_statistic_by_flavor", {})
    prior_allowance = final_envelope.get(
        "prior_product_median_sampling_allowance_diagnostic_by_flavor", {})
    for mapping in (*direct_inputs.values(), declared_allowance, exact_full,
                    prior_allowance):
        require(isinstance(mapping, dict) and set(mapping) == set(FLAVORS),
                "final-statistic resampling evidence lacks exact u/d coverage")

    result: dict[str, dict] = {}
    all_widths_pass = True
    for flavor in FLAVORS:
        item = widths[flavor]
        require(isinstance(item, dict), f"invalid {flavor} width metrics")
        product = float(item["terminal_product_raw_full_width"])
        convergence = float(item[
            "terminal_anchor_convergence_raw_full_width"])
        final = float(item[
            "joint_convergence_interaction_raw_full_width"])
        allowance = float(item[
            "final_statistic_finite_sampling_full_width_margin"])
        compatibility_allowance = float(item[
            "corrected_finite_sampling_full_width_margin"])
        adjusted = float(item[
            "joint_raw_width_plus_final_statistic_sampling_margin"])
        compatibility_adjusted = float(item[
            "joint_raw_width_plus_corrected_sampling_margin"])
        incumbent = float(item["immutable_lambda1_width"])
        direct_values = [float(mapping[flavor])
                         for mapping in direct_inputs.values()]
        values = np.asarray([
            product, convergence, final, allowance, compatibility_allowance,
            adjusted, compatibility_adjusted, incumbent,
            float(exact_full[flavor]), float(prior_allowance[flavor]),
            *direct_values,
        ])
        require(np.all(np.isfinite(values)) and np.all(values >= 0.0),
                f"non-finite or negative {flavor} width evidence")
        tolerance = 1.0e-12
        require(convergence + tolerance >= product
                and final + tolerance >= convergence,
                f"{flavor} directional-envelope widths are not nested")
        require(math.isclose(allowance, max(direct_values),
                             rel_tol=1.0e-12, abs_tol=1.0e-14),
                f"{flavor} allowance is not the maximum exact-final-statistic p95")
        require(math.isclose(final, float(exact_full[flavor]),
                             rel_tol=5.0e-13, abs_tol=5.0e-15),
                f"{flavor} resampled full statistic differs from final width")
        require(math.isclose(allowance, float(declared_allowance[flavor]),
                             rel_tol=1.0e-12, abs_tol=1.0e-14),
                f"{flavor} final-statistic allowance differs")
        require(math.isclose(allowance, compatibility_allowance,
                             rel_tol=0.0, abs_tol=1.0e-15),
                f"{flavor} compatibility allowance differs")
        require(math.isclose(adjusted, final + allowance,
                             rel_tol=1.0e-12, abs_tol=1.0e-14),
                f"{flavor} robust width is not raw final plus allowance")
        require(math.isclose(adjusted, compatibility_adjusted,
                             rel_tol=0.0, abs_tol=1.0e-15),
                f"{flavor} compatibility adjusted width differs")
        require(explicit_bool(item.get(
                    "sampling_allowance_statistic_matches_width_statistic"),
                    f"{flavor} same-statistic sampling flag"),
                f"{flavor} sampling allowance targets a different statistic")
        exact_number(incumbent, LOCKED_INCUMBENT_WIDTHS[flavor],
                     f"{flavor} immutable lambda1 width")
        exact_number(locked[flavor], LOCKED_INCUMBENT_WIDTHS[flavor],
                     f"{flavor} stability lambda1 width")
        passed = adjusted < incumbent
        require(explicit_bool(item.get("replacement_gate_pass"),
                              f"{flavor} final replacement gate") == passed,
                f"{flavor} final replacement-gate semantics disagree")
        result[flavor] = {
            "terminal_product_raw_full_width": product,
            "terminal_anchor_convergence_raw_full_width": convergence,
            "joint_convergence_interaction_raw_full_width": final,
            "direct_finite_sampling_full_width_margin": allowance,
            "joint_raw_plus_direct_sampling_margin": adjusted,
            "immutable_lambda1_full_width": incumbent,
            "prior_product_median_sampling_margin_diagnostic": float(
                prior_allowance[flavor]),
            "replacement_gate_pass": passed,
            "direct_resampling_p95_components": dict(zip(
                direct_inputs, direct_values, strict=True)),
        }
        all_widths_pass = all_widths_pass and passed

    require(explicit_bool(final_envelope.get(
                "joint_width_replacement_gate_pass"),
                "final joint-width gate") == all_widths_pass,
            "joint u/d replacement-gate semantics disagree")
    return result, bool(all_widths_pass)


def validate_completion_audit(
    completion: dict, final_envelope: dict, final_hash: str,
    width_metrics: dict[str, dict], postfit_hash: str, nested_hash: str,
) -> None:
    """Revalidate the promotion-only completion manifest and all bound bytes."""
    require(completion.get("status") == "complete",
            "promoted outcome lacks a complete campaign audit")
    validate_prescription(completion, "completion audit")
    require(int(completion.get("start_count", -1)) == 24
            and int(completion.get("experimental_replica_count", -1)) == 50
            and int(completion.get("combined_member_count", -1)) == 1200,
            "completion audit lacks exact 24+50/1200 coverage")
    require(explicit_bool(completion.get("terminal_evidence_gate_pass"),
                          "completion terminal-evidence gate")
            and explicit_bool(completion.get(
                "postfit_tail_transform_gate_pass"),
                "completion postfit-tail gate")
            and explicit_bool(completion.get(
                "nested_interaction_validation_gate_pass"),
                "completion nested-interaction gate")
            and explicit_bool(completion.get(
                "final_directional_envelope_gate_pass"),
                "completion final-envelope gate"),
            "promotion completion gates are not all true")
    require(completion.get("postfit_tail_transform_audit_sha256") == postfit_hash
            and completion.get(
                "nested_interaction_validation_sha256") == nested_hash
            and completion.get(
                "final_directional_envelope_sha256") == final_hash,
            "completion audit is not bound to the validated final graph")
    for flavor in FLAVORS:
        exact_number(completion[
            "final_fig6_max_active_relative_full_width"][flavor],
            width_metrics[flavor][
                "joint_convergence_interaction_raw_full_width"],
            f"completion {flavor} final width")
        exact_number(completion[
            "resampling_full_width_allowance_by_flavor"][flavor],
            width_metrics[flavor][
                "direct_finite_sampling_full_width_margin"],
            f"completion {flavor} sampling allowance")
        exact_number(completion[
            "incumbent_fig6_max_active_relative_full_width"][flavor],
            LOCKED_INCUMBENT_WIDTHS[flavor],
            f"completion {flavor} incumbent width")
    validate_path_hash_map(completion.get("evidence_sha256", {}),
                           "completion evidence")
    require(not explicit_bool(completion.get("production_sources_modified"),
                              "completion production_sources_modified"),
            "completion reports production-source modification")


def validate_terminal_evidence_manifest(
    terminal: dict, protocol_hash: str, postfit_hash: str,
    nested_hash: str, final_hash: str, endpoint_gate: bool,
) -> None:
    require(terminal.get("status") == "pass"
            and terminal.get("comparison_champion_id") == INCUMBENT_ID
            and int(terminal.get("start_count", -1)) == 24
            and int(terminal.get("central_count", -1)) == 1
            and int(terminal.get("replica_count", -1)) == 50,
            "terminal evidence manifest lacks exact 24+1+50 coverage")
    require(explicit_bool(terminal.get("candidate_endpoint_gate_pass"),
                          "terminal candidate endpoint gate") == endpoint_gate
            and explicit_bool(terminal.get(
                "candidate_final_directional_envelope_gate_pass"),
                "terminal final-envelope gate") == endpoint_gate,
            "terminal evidence outcome gate differs from final envelope")
    require(terminal.get("fixed_challenger_protocol_sha256") == protocol_hash
            and terminal.get("postfit_tail_transform_audit_sha256") == postfit_hash
            and terminal.get("nested_interaction_validation_sha256") == nested_hash
            and terminal.get("final_directional_envelope_sha256") == final_hash,
            "terminal evidence hashes differ from validated inputs")
    require_fixed_implementation_binding(terminal, "terminal evidence")
    validate_path_hash_map(terminal.get("artifact_sha256", {}),
                           "terminal evidence")
    roles = terminal.get("artifact_roles")
    require(isinstance(roles, dict)
            and {"figure_2", "figure_6", "lambda600_vs_lambda1_comparison",
                 "supporting_terminal_evidence"}.issubset(roles),
            "terminal evidence lacks required artifact roles")
    require(not explicit_bool(terminal.get("production_sources_modified"),
                              "terminal production_sources_modified")
            and not explicit_bool(terminal.get("registry_modified"),
                                  "terminal registry_modified"),
            "terminal audit reports a source or registry modification")


def validate_outcome_consistency(
    outcome: str, starts: dict, replicas: dict, stability: dict,
    postfit: dict, nested: dict, final_envelope: dict,
    figures: dict, comparison: dict, width_gate: bool,
) -> dict[str, bool]:
    """Require proposed promotion/rejection to equal the complete gate result."""
    require(outcome in TERMINAL_STATUSES, f"unknown terminal outcome: {outcome}")
    start_gate = explicit_bool(
        starts.get("all_starts_fnp_plateaued_and_fit_preserved"),
        "24-start scientific gate")
    require((starts.get("status") == "complete") == start_gate,
            "24-start status and scientific gate disagree")
    central_gate = explicit_bool(replicas.get("central_fnp_plateau_pass"),
                                 "central scientific gate")
    replica_gate = explicit_bool(replicas.get("all_replicas_fnp_plateaued"),
                                 "replica scientific gate")
    base_gate = bool(
        explicit_bool(stability.get("diagnostic_figure_gate_pass"),
                      "base diagnostic evidence gate")
        and explicit_bool(stability.get("candidate_stationarity_gate_pass"),
                          "base candidate stationarity gate")
    )
    require(explicit_bool(final_envelope.get(
                "base_product_stability_gate_pass"),
                "final base-product stability gate") == base_gate,
            "final/base evidence gate semantics disagree")
    postfit_gate = explicit_bool(postfit.get("promotion_validation_gate_pass"),
                                 "postfit tail/transform gate")
    nested_gate = explicit_bool(nested.get("interaction_validation_gate_pass"),
                                "nested interaction gate")
    containment_gate = explicit_bool(final_envelope.get(
        "trained_central_containment_gate_pass"), "trained-central containment")
    final_gate = final_promotion_gate(final_envelope)
    recomputed_final_gate = bool(
        base_gate and postfit_gate and nested_gate and width_gate
        and containment_gate)
    require(final_gate == recomputed_final_gate,
            "final directional-envelope gate does not equal its conjuncts")
    promotion_gate = bool(
        final_gate and start_gate and central_gate and replica_gate)
    require((outcome == PROMOTED) == promotion_gate,
            "proposed terminal status disagrees with the complete promotion gate")

    expected_figure_status = (
        "final_validated_figures" if outcome == PROMOTED
        else "diagnostic_figures_not_promotable")
    expected_comparison_status = (
        "complete_validated_candidate_comparison" if outcome == PROMOTED
        else "complete_diagnostic_scientific_failure_comparison")
    require(figures.get("status") == expected_figure_status
            and explicit_bool(figures.get("endpoint_gate_pass"),
                              "figure endpoint gate") == final_gate
            and explicit_bool(figures.get("diagnostic_only"),
                              "figure diagnostic_only") == (not final_gate)
            and not explicit_bool(figures.get("formal_confidence_level_assigned"),
                                  "figure formal confidence flag")
            and not explicit_bool(figures.get("one_sigma_claimed"),
                                  "figure one-sigma flag"),
            "figure summary disagrees with the final outcome")
    require(comparison.get("status") == expected_comparison_status
            and comparison.get("comparison_champion_id") == INCUMBENT_ID
            and explicit_bool(comparison.get("candidate_endpoint_gate_pass"),
                              "comparison endpoint gate") == final_gate
            and explicit_bool(comparison.get("diagnostic_only"),
                              "comparison diagnostic_only") == (not final_gate)
            and explicit_bool(comparison.get(
                "legacy_lambda1_fig6_widths_remain_gating"),
                "comparison immutable-lambda1 gate"),
            "explicit lambda600/lambda1 comparison disagrees with outcome")
    return {
        "start_stationarity_and_fit": start_gate,
        "trained_central_stationarity": central_gate,
        "experimental_replica_stationarity_and_agreement": replica_gate,
        "base_product_stability": base_gate,
        "postfit_tail_transform": postfit_gate,
        "nested_interaction": nested_gate,
        "trained_central_containment": containment_gate,
        "joint_width_replacement": width_gate,
        "complete_promotion_gate": promotion_gate,
    }


def validate_figure_names(figures: dict, outcome: str) -> dict[str, Path]:
    expected = expected_figure_paths(outcome)
    keys = {"fnp": "fnp_diagnostic", "fig2": "figure_2", "fig6": "figure_6"}
    for role, key in keys.items():
        declared = Path(str(figures.get(key, "")))
        path = declared if declared.is_absolute() else FIGURE_DIR / declared
        require(path.resolve() == expected[role].resolve(),
                f"{role} figure name does not match final probability semantics")
        require(path.is_file() and path.stat().st_size > 0
                and path.with_suffix(".png").is_file()
                and path.with_suffix(".png").stat().st_size > 0,
                f"{role} figure PDF/PNG pair is missing")
    return expected


def snapshot_protected_state() -> dict:
    """Snapshot all registered frozen inputs and campaign-local registry bytes."""
    manifest = read_json(FROZEN_MANIFEST, "frozen-input manifest")
    files = manifest.get("files")
    require(isinstance(files, dict) and files,
            "frozen-input manifest has no files")
    frozen: dict[str, str] = {}
    for path_text, metadata in files.items():
        path = Path(path_text)
        require(path.is_file() and int(metadata["bytes"]) == path.stat().st_size,
                f"registered frozen input is missing or resized: {path}")
        digest = sha256(path)
        require(digest == str(metadata["sha256"]),
                f"registered frozen input changed: {path}")
        frozen[str(path.resolve())] = digest
    registry = {
        str(path.resolve()): sha256(path)
        for path in sorted(REGISTRY.rglob("*")) if path.is_file()
    }
    return {"frozen": frozen, "registry": registry}


def collect_and_validate(proposed_status: str | None = None) -> dict:
    protocol, protocol_hash = validate_fixed_challenger_protocol()
    starts = read_json(STARTS, "24-start summary")
    start_chain_audit = read_json(
        START_CHAIN_AUDIT, "terminal start-chain audit")
    replicas = read_json(REPLICAS, "central/replica summary")
    product = read_json(PRODUCT, "24x50 product summary")
    stability = read_json(STABILITY, "final product stability audit")
    require_fixed_implementation_binding(starts, "24-start summary")
    require_fixed_implementation_binding(replicas, "central/replica summary")
    require_fixed_implementation_binding(product, "24x50 product summary")
    require_fixed_implementation_binding(stability, "final stability summary")
    validate_prescription(starts, "24-start summary")
    validate_prescription(replicas, "central/replica summary")
    validate_prescription(product, "24x50 product summary")
    require(starts.get("status") in {"complete", "verification_failed"}
            and int(starts.get("member_count", -1)) == 24
            and len(starts.get("endpoint_tags", [])) == 24
            and len(set(starts.get("endpoint_tags", []))) == 24,
            "24-start terminal coverage is incomplete")
    start_ancestry = validate_start_ancestry_limitation(
        start_chain_audit, starts)
    require(replicas.get("status") in {
                "complete", "complete_with_scientific_failures",
                "central_stationarity_failed", "replica_stationarity_failed"}
            and int(replicas.get("completed_replica_count", -1)) == 50
            and len(replicas.get("replica_endpoint_tags", [])) == 50
            and len(set(replicas.get("replica_endpoint_tags", []))) == 50
            and bool(str(replicas.get("central_endpoint_tag", "")).strip()),
            "trained-central/50-replica terminal coverage is incomplete")
    require(explicit_bool(replicas.get("central_full_horizon_complete"),
                          "central full-horizon completion"),
            "trained central did not complete its full horizon")
    exact_number(replicas.get("central_full_horizon_requested_capacity"),
                 300_000.0, "central full-horizon capacity")
    exact_number(replicas.get("central_terminal_requested_capacity"),
                 300_000.0, "central terminal capacity")
    require(product.get("status") == "complete"
            and int(product.get("start_count", -1)) == 24
            and int(product.get("experimental_replica_count", -1)) == 50
            and int(product.get("combined_member_count", -1)) == 1200,
            "centered-log-FNP product hierarchy lacks exact 24x50 coverage")

    postfit, postfit_hash = validated_postfit_tail_audit(POSTFIT, STABILITY)
    nested, nested_hash = validated_nested_interaction(
        NESTED, POSTFIT, STABILITY)
    final_envelope, final_hash = validated_final_directional_envelope(
        FINAL_ENVELOPE, NESTED, POSTFIT, STABILITY)
    require(not explicit_bool(final_envelope.get("formal_confidence_level_assigned"),
                              "final formal confidence flag")
            and not explicit_bool(final_envelope.get("one_sigma_claimed"),
                                  "final one-sigma flag"),
            "final directional envelope overstates probability semantics")
    width_metrics, width_gate = validate_width_evidence(stability, final_envelope)

    figures = read_json(FIGURES, "Fig. 2/Fig. 6 summary")
    comparison = read_json(COMPARISON, "lambda600/lambda1 comparison")
    validate_hash_map(comparison.get("artifacts", {}),
                      comparison.get("artifact_sha256", {}),
                      "lambda600/lambda1 comparison", COMPARISON.parent)
    validate_path_hash_map({
        str(Path(path_text)): digest
        for path_text, digest in comparison.get("input_sha256", {}).items()
        if Path(path_text).is_absolute()
    }, "comparison absolute inputs") if any(
        Path(path_text).is_absolute()
        for path_text in comparison.get("input_sha256", {})) else None

    terminal = read_json(TERMINAL_EVIDENCE, "terminal evidence manifest")
    endpoint_gate = final_promotion_gate(final_envelope)
    validate_terminal_evidence_manifest(
        terminal, protocol_hash, postfit_hash, nested_hash, final_hash,
        endpoint_gate)

    if proposed_status is None:
        decision, decision_hash = validate_complete_lambda600_comparison()
        outcome = str(decision["status"])
        decision_mode = "published_terminal_decision_revalidated"
    else:
        require(proposed_status in TERMINAL_STATUSES,
                f"invalid proposed terminal status: {proposed_status}")
        outcome = proposed_status
        decision = None
        decision_hash = None
        decision_mode = "prepublication_complete_graph_revalidated"
        if DECISION.is_file():
            existing = read_json(DECISION, "current fixed-comparison decision")
            if existing.get("status") in TERMINAL_STATUSES:
                decision, decision_hash = validate_complete_lambda600_comparison()
                require(decision.get("status") == outcome,
                        "published terminal status differs from requested status")
                decision_mode = "published_terminal_decision_revalidated"

    gates = validate_outcome_consistency(
        outcome, starts, replicas, stability, postfit, nested,
        final_envelope, figures, comparison, width_gate)
    figure_paths = validate_figure_names(figures, outcome)

    if decision is not None:
        require(decision.get("full24") == starts
                and decision.get("replicas") == replicas
                and decision.get("stability") == stability
                and decision.get("figures") == figures
                and decision.get("comparison") == comparison
                and decision.get("postfit_tail_transform_audit") == postfit
                and decision.get("nested_interaction_validation") == nested
                and decision.get("final_directional_envelope") == final_envelope,
                "published terminal decision embeds a different evidence graph")

    completion = None
    if outcome == PROMOTED:
        completion = read_json(COMPLETION, "promotion completion audit")
        validate_completion_audit(
            completion, final_envelope, final_hash, width_metrics,
            postfit_hash, nested_hash)

    frozen = read_json(FROZEN_AUDIT, "frozen-input audit")
    require(frozen.get("status") == "pass"
            and int(frozen.get("registered_input_count", -1)) > 0
            and int(frozen.get("unchanged_input_count", -2))
                == int(frozen.get("registered_input_count", -1))
            and not frozen.get("failures"),
            "frozen-input audit is not clean")
    incumbent = read_json(INCUMBENT, "immutable lambda1 incumbent")
    require(incumbent.get("champion_id") == INCUMBENT_ID,
            "immutable lambda1 incumbent identity changed")
    validate_hash_map(incumbent.get("artifacts", {}),
                      incumbent.get("artifact_sha256", {}),
                      "immutable lambda1 incumbent")
    for flavor in FLAVORS:
        exact_number(incumbent[
            "combined_fig6_max_active_relative_full_width"][flavor],
            LOCKED_INCUMBENT_WIDTHS[flavor],
            f"incumbent {flavor} width")

    input_paths = [
        PROTOCOL, FNP_REFERENCE, STARTS, START_CHAIN_AUDIT,
        Path(start_ancestry["seal"]), REPLICAS, PRODUCT, STABILITY,
        POSTFIT, NESTED, FINAL_ENVELOPE, FIGURES, COMPARISON,
        TERMINAL_EVIDENCE, FROZEN_AUDIT, INCUMBENT,
    ]
    if completion is not None:
        input_paths.append(COMPLETION)
    if decision is not None:
        input_paths.append(DECISION)
    input_sha256 = {str(path): sha256(path) for path in input_paths}
    return {
        "outcome": outcome,
        "decision_validation_mode": decision_mode,
        "decision_sha256": decision_hash,
        "protocol": protocol,
        "protocol_sha256": protocol_hash,
        "starts": starts,
        "start_ancestry": start_ancestry,
        "replicas": replicas,
        "product": product,
        "stability": stability,
        "postfit": postfit,
        "postfit_sha256": postfit_hash,
        "nested": nested,
        "nested_sha256": nested_hash,
        "final_envelope": final_envelope,
        "final_envelope_sha256": final_hash,
        "figures": figures,
        "comparison": comparison,
        "terminal_evidence": terminal,
        "completion": completion,
        "frozen": frozen,
        "incumbent": incumbent,
        "width_metrics": width_metrics,
        "gates": gates,
        "figure_paths": figure_paths,
        "input_sha256": input_sha256,
    }


def gate_word(value: bool) -> str:
    return "pass" if value else "fail"


def render_report(evidence: dict) -> str:
    outcome = str(evidence["outcome"])
    promoted = outcome == PROMOTED
    widths = evidence["width_metrics"]
    gates = evidence["gates"]
    starts = evidence["starts"]
    start_ancestry = evidence["start_ancestry"]
    replicas = evidence["replicas"]
    product = evidence["product"]
    postfit = evidence["postfit"]
    nested = evidence["nested"]
    final_envelope = evidence["final_envelope"]
    figures = evidence["figures"]
    comparison = evidence["comparison"]
    frozen = evidence["frozen"]
    figure_paths = evidence["figure_paths"]

    if promoted:
        decision_text = (
            "The fixed lambda=600 challenger passed every required gate and was "
            "promoted as the new isolated-study champion."
        )
        incumbent_text = (
            "The lambda=1 record was the immutable comparison threshold during "
            "the challenge; promotion occurred only after the complete evidence "
            "graph passed."
        )
    else:
        decision_text = (
            "The fixed lambda=600 challenger completed the requested evidence "
            "collection but failed at least one scientific promotion gate and was "
            "rejected. The lambda=1 incumbent remains the study baseline."
        )
        incumbent_text = (
            "The complete diagnostic figures are retained even though they are not "
            "eligible for promotion."
        )

    width_rows = []
    for flavor in FLAVORS:
        item = widths[flavor]
        width_rows.append(
            f"| {flavor} | {pct(item['terminal_product_raw_full_width'])} | "
            f"{pct(item['terminal_anchor_convergence_raw_full_width'])} | "
            f"{pct(item['joint_convergence_interaction_raw_full_width'])} | "
            f"{pct(item['direct_finite_sampling_full_width_margin'])} | "
            f"{pct(item['joint_raw_plus_direct_sampling_margin'])} | "
            f"{pct(item['immutable_lambda1_full_width'])} | "
            f"{gate_word(item['replacement_gate_pass'])} |"
        )
    gate_labels = {
        "start_stationarity_and_fit": "24-start FNP stationarity and fit preservation",
        "trained_central_stationarity": "trained-central FNP stationarity",
        "experimental_replica_stationarity_and_agreement":
            "50-replica stationarity and cross-optimizer agreement",
        "base_product_stability": "24x50 product coverage, integrity, and robustness",
        "postfit_tail_transform": "full-tail/checkpoint and transform audit",
        "nested_interaction": "six-pair nested interaction validation",
        "trained_central_containment": "trained-central containment",
        "joint_width_replacement": "u/d final-width replacement",
        "complete_promotion_gate": "complete promotion gate",
    }
    gate_rows = [
        f"| {gate_labels[key]} | {gate_word(bool(gates[key]))} |"
        for key in gate_labels
    ]
    failure_reasons = [
        str(value) for value in final_envelope.get(
            "scientific_failure_reasons", []) if str(value).strip()
    ]
    if promoted:
        reason_section = "All recorded scientific failure-reason lists are empty."
    else:
        require(failure_reasons,
                "rejected report has no explicit scientific failure reason")
        reason_section = "\n".join(f"- {reason}" for reason in failure_reasons)

    start_failures = starts.get("failed_seeds", [])
    replica_failures = replicas.get("failed_replica_seeds", [])
    central_tag = str(replicas["central_endpoint_tag"])
    nested_count = int(nested["completed_pair_count"])
    figure_rel = {
        role: path.relative_to(BASE) if BASE in path.resolve().parents else path
        for role, path in figure_paths.items()
    }
    final_artifacts = final_envelope["artifacts"]
    comparator_artifacts = comparison["artifacts"]
    figure_outcome_text = (
        "These are the validated promotion artifacts."
        if promoted else
        "A scientifically rejected challenger is deliberately shown as "
        "diagnostic-only rather than hidden."
    )

    report = f"""# Final fixed-challenger lambda=600 versus lambda=1 report

## Decision

{decision_text} {incumbent_text}

This was one pre-registered replacement test, not a lambda ladder or a search
for a numerically weakest constraint. The challenger prescription was the
production-capacity FiLM model with reference-distance lambda=600 on
0.1 <= bT <= 4 GeV^-1 and the one-sided power-2 fit-quality safeguard mu=100.
No other lambda, prior, or architecture is selected by this report. Even a
promoted result establishes only that this fixed challenger beat the incumbent
under the locked gates; it does not prove that lambda=600 is the minimum
sufficient non-physics constraint.

The immutable comparison is the registered lambda=1, bT<=2 incumbent. Its
historical training protocol is not identical to the challenger protocol.
Post-processing was harmonized as a diagnostic control, but the locked legacy
lambda=1 widths—not a candidate-dependent recomputation—remain the promotion
thresholds.

## Exact terminal coverage

- Independent production-capacity optimizer starts: 24; terminal status:
  `{starts['status']}`; failed seeds: `{start_failures}`.
- Start-checkpoint ancestry: {int(start_ancestry['launch_time_receipt_checkpoint_count'])}
  checkpoints have immutable launch-time parent-content receipts and
  {int(start_ancestry['legacy_pre_receipt_checkpoint_count'])} checkpoints predate
  that receipt mechanism. The complete terminal graph is frozen by the
  generation-addressed precentral current-byte seal
  `{Path(start_ancestry['seal']).relative_to(BASE)}`.
- Separately trained central model: one endpoint, `{central_tag}`, forced through
  the complete 300000 requested-capacity horizon. This trained endpoint is the
  central line in FNP, Fig. 2, and Fig. 6.
- Experimental pseudo-data replicas: 50/50; terminal status:
  `{replicas['status']}`; failed replica seeds: `{replica_failures}`.
- Centered whole-curve log-FNP Cartesian product: 24 x 50 =
  {int(product['combined_member_count'])} derived members. These are not 1200
  independently fitted nested models.
- Nested start-by-replica stress: {nested_count}/6 deterministic stratified pairs,
  each taken through its declared full horizon; status: `{nested['status']}`.

Requested LBFGS capacity labels are not interpreted as executed optimizer
updates. Closure-evaluation accounting remains in the terminal ledgers.

## Central curve and uncertainty construction

The plotted central is the separately trained terminal lambda=600 central
endpoint at the 300000 requested-capacity horizon, propagated directly in
b-space and through its exact paired expb2 finite-b transform. It is not the
pointwise median of the 24 starts, the 50 replicas, or the 1200 product members.
Those medians remain distribution diagnostics only.

The final displayed bounds are constructed in three explicit stages:

1. Form the empirical product q16--q84 band from the 24 centered optimizer-start
   residual curves crossed with the 50 centered experimental-replica residual
   curves around the trained central.
2. Enlarge it directionally by the terminal-versus-stationarity-anchor motion:
   take the lower of the two lower product endpoints and the upper of the two
   upper product endpoints.
3. Add the observed directional residual from the six full-horizon nested
   start-by-replica fits (multiplicatively in log-FNP/Fig. 2 and additively after
   the exact Fig. 6 transform).

The resulting product + convergence + interaction band is an empirical
directional envelope. Only the conditional experimental-replica marginal has
its conventional replica interpretation. Optimizer starts, checkpoint motion,
and the nested-interaction stress have no calibrated probability law. No formal
1sigma or confidence-level interpretation is assigned to the final envelope.

The post-fit audit extends FNP checks from the formal training domain through
bT=8 GeV^-1 and checks expb2, expb, and taper continuations. Expb2 remains the
locked Fig. 6 gate; the alternative modes test the raw decision sign and cannot
loosen that threshold. Post-fit status: `{postfit['status']}`.

## Fig. 6 fixed-incumbent comparison

Relative full width means (upper-lower)/abs(trained central), maximized on the
flavor-local union of the challenger and incumbent active regions through
kT=2.25 GeV. The finite-sampling margin is not drawn as an additional band.
The authoritative calculation resamples the same final trained-central-
normalized statistic shown in the table. Terminal and stationarity-anchor
arrays use identical resampled start/replica identities; each resample
recomputes both checkpoint q16/q84 pairs, applies the fixed nested-interaction
directions, divides by the fixed trained-300k central on the fixed challenger/
incumbent union mask, and maximizes the resulting full width. The flavor
allowance is the largest p95 bootstrap/start-split/replica-split/joint-split
movement. The earlier product-median sampling allowance is retained only as a
non-gating diagnostic; no endpoint-motion conversion factor is used.

| flavor | product raw | + convergence raw | + interaction raw (final plotted) | direct sampling margin | final raw + margin | immutable lambda=1 | replacement |
|:---:|---:|---:|---:|---:|---:|---:|:---:|
{chr(10).join(width_rows)}

Both u and d must pass separately; an improvement in one flavor cannot offset
a failure in the other. The margin is used only for the promotion decision and
does not convert the plotted directional envelope into a confidence interval.

## Scientific gates

| gate | result |
|:---|:---:|
{chr(10).join(gate_rows)}

Outcome-specific reasons:

{reason_section}

## Final Fig. 2 and Fig. 6 artifacts

The final/diagnostic figures contain one trained central line and one
product-plus-directional-envelope band, with no individual seed curves and no
legacy conditional result:

- FNP diagnostic: `{figure_rel['fnp']}`
- Fig. 2: `{figure_rel['fig2']}`
- Fig. 6: `{figure_rel['fig6']}`
- FNP final-envelope table:
  `{Path(final_artifacts['fnp_final_envelope']).relative_to(BASE)}`
- Fig. 2 final-envelope table:
  `{Path(final_artifacts['fig2_bspace_final_envelope']).relative_to(BASE)}`
- Fig. 6 final-envelope table:
  `{Path(final_artifacts['fig6_kspace_final_envelope']).relative_to(BASE)}`
- Explicit lambda600-versus-lambda1 comparison summary:
  `{COMPARISON.relative_to(BASE)}`
- Fig. 6 comparison plot:
  `{Path(comparator_artifacts['fig6_png']).relative_to(BASE)}`

Figure status: `{figures['status']}`. Comparator status:
`{comparison['status']}`. {figure_outcome_text}

## Provenance and isolation

The locked challenger protocol, fixed empirical FNP reference, fixed training
implementation hashes, exact 24+1+50 state-chain evidence, product tables,
post-fit tail/transform audit, nested-interaction audit, final directional
envelope, figures, and explicit incumbent comparison were revalidated before
this report was written. Terminal decision validation mode:
`{evidence['decision_validation_mode']}`.

Start-chain provenance limitation: {start_ancestry['historical_ancestry_limitation']}
This limitation is retained explicitly: the current-byte seal prevents later
mutation, but it is not described as retrospective proof for checkpoints that
were launched by the already-running pre-receipt controller. Every central and
experimental-replica checkpoint, which had not started when the issue was
identified, is required to carry a prospective launch-time receipt.

Frozen-input audit: {int(frozen['unchanged_input_count'])}/
{int(frozen['registered_input_count'])} registered inputs unchanged. The report
writer writes only `summaries/final_study_report`; it neither edits frozen
production files nor the campaign champion registry.
"""
    return report


def write_report(evidence: dict) -> dict:
    protected_before = snapshot_protected_state()
    report = render_report(evidence)
    report_path = TARGET / "FINAL_REPORT.md"
    atomic_write_text(report_path, report)
    report_hash = sha256(report_path)
    summary = {
        "status": (
            "complete_fixed_challenger_report_promoted"
            if evidence["outcome"] == PROMOTED
            else "complete_fixed_challenger_report_rejected"
        ),
        "outcome": evidence["outcome"],
        "report": str(report_path),
        "artifact_sha256": {"final_report": report_hash},
        "decision_validation_mode": evidence["decision_validation_mode"],
        "terminal_decision_sha256": evidence["decision_sha256"],
        "fixed_challenger_protocol": str(PROTOCOL),
        "fixed_challenger_protocol_sha256": evidence["protocol_sha256"],
        "fixed_fnp_reference": str(FNP_REFERENCE),
        "fixed_fnp_reference_sha256": EXPECTED_FNP_REFERENCE_SHA256,
        "fixed_implementation_sha256": dict(EXPECTED_IMPLEMENTATION_SHA256),
        "start_count": 24,
        "trained_central_count": 1,
        "trained_central_requested_capacity": 300_000,
        "experimental_replica_count": 50,
        "product_member_count": 1200,
        "nested_interaction_pair_count": 6,
        "central_line": "separately trained terminal 300k lambda600 central endpoint",
        "displayed_interval": (
            "empirical product band plus residual convergence/interaction envelope"
        ),
        "formal_confidence_level_assigned": False,
        "one_sigma_claimed": False,
        "width_metrics_by_flavor": evidence["width_metrics"],
        "scientific_gates": evidence["gates"],
        "start_checkpoint_ancestry": evidence["start_ancestry"],
        "input_sha256": evidence["input_sha256"],
        "protected_frozen_input_count": len(protected_before["frozen"]),
        "protected_registry_file_count": len(protected_before["registry"]),
        "registry_modified_by_report_writer": False,
        "frozen_sources_modified": False,
        "production_sources_modified": False,
    }
    atomic_write_json(TARGET / "summary.json", summary)
    protected_after = snapshot_protected_state()
    require(protected_after == protected_before,
            "frozen inputs or registry changed while writing final report")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--decision-status", choices=TERMINAL_STATUSES,
        help=(
            "proposed terminal status for prepublication use; omit to require "
            "an already published and fully validated terminal decision"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    evidence = collect_and_validate(args.decision_status)
    write_report(evidence)


if __name__ == "__main__":
    main()
