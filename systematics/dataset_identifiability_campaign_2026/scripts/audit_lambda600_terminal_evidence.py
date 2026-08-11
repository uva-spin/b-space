#!/usr/bin/env python3
"""Bind terminal lambda600 figures and the immutable-lambda1 comparison.

This audit is intentionally independent of promotion.  A scientifically
rejected but complete 24x50 candidate must retain the same immutable evidence
binding as a passing candidate: its combined Fig. 2/Fig. 6 and every explicit
lambda600-versus-lambda1 comparison artifact are hashed into one deterministic
manifest.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

from fixed_challenger_protocol import (
    EXPECTED_FNP_REFERENCE_SHA256,
    FNP_REFERENCE as FIXED_FNP_REFERENCE,
    IMPLEMENTATION_FILES as FIXED_IMPLEMENTATION_FILES,
    PROTOCOL as FIXED_PROTOCOL,
    fixed_implementation_binding,
    require_fixed_implementation_binding,
    validate_fixed_challenger_protocol,
)
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
SUM = BASE / "summaries"
STARTS = SUM / "replica_robust_reference_full24/summary.json"
START_CHAIN_AUDIT = SUM / "lambda600_start_chain_audit/summary.json"
START_SEAL_ROOT = SUM / "lambda600_start_chain_audit/current_byte_seals"
REPLICAS = SUM / "selected_reference_central_replicas/summary.json"
STABILITY = SUM / "final_combined_ensemble_stability/summary.json"
POSTFIT_TAIL_AUDIT = (
    SUM / "lambda600_postfit_tail_transform_audit/summary.json"
)
NESTED_INTERACTION = (
    SUM / "lambda600_nested_start_replica_interaction/summary.json"
)
FINAL_DIRECTIONAL_ENVELOPE = (
    SUM / "lambda600_final_directional_envelope/summary.json"
)
FIGURE_DIR = SUM / "final_fig2_fig6"
FIGURES = FIGURE_DIR / "summary.json"
COMPARISON_DIR = SUM / "lambda600_vs_lambda1_diagnostic"
COMPARISON = COMPARISON_DIR / "summary.json"
INCUMBENT = (
    SUM / "champion_registry/empirical_reference_lambda1_b0p1_2p0_full24.json"
)
COMBINED = SUM / "final_combined_tmd_ensemble"
HARMONIZED = SUM / "harmonized_lambda1_logfnp_24x50_comparator"
PIN_MANIFEST = BASE / "manifests/harmonized_lambda1_inputs.json"
REFERENCE_BSPACE = (
    SYSTEMATICS / "collins_factorization_validity/plots/"
    "rowidfix_stageFT_all_qmax0p20_lam0p50_central_exactx/"
    "v22_scheme_tmd_bspace_long.csv"
)
TARGET = SUM / "lambda600_terminal_evidence"
INCUMBENT_ID = "empirical_reference_lambda1_b0p1_2p0_full24"
LOCKED_INCUMBENT_WIDTHS = {
    "u": 0.11772613918747582,
    "d": 0.12490071924111977,
}
COMPARISON_ARTIFACT_NAMES = {
    "lambda600_vs_lambda1_fnp_x0p1.png",
    "lambda600_vs_lambda1_fnp_x0p1.pdf",
    "lambda600_vs_lambda1_fig2_bT_Q7p5.png",
    "lambda600_vs_lambda1_fig2_bT_Q7p5.pdf",
    "lambda600_vs_lambda1_fig6_kT_Q10.png",
    "lambda600_vs_lambda1_fig6_kT_Q10.pdf",
    "fnp_comparison.csv",
    "fig2_bT_comparison.csv",
    "fig6_kT_comparison.csv",
    "relative_full_width_metrics.csv",
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


def load(path: Path) -> dict:
    require(path.is_file() and path.stat().st_size > 0,
            f"missing terminal evidence input: {path}")
    payload = json.loads(path.read_text())
    require(isinstance(payload, dict), f"invalid JSON object: {path}")
    return payload


def explicit_bool(value: object, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise RuntimeError(f"{label} is not an explicit boolean")


def exact_number(observed: object, expected: float) -> bool:
    try:
        value = float(observed)
    except (TypeError, ValueError):
        return False
    return (math.isfinite(value)
            and math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12))


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def resolved_artifact(path_text: str, parent: Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else parent / path


def main() -> None:
    _, fixed_protocol_hash = validate_fixed_challenger_protocol()
    starts = load(STARTS)
    start_chain_audit = load(START_CHAIN_AUDIT)
    replicas = load(REPLICAS)
    stability = load(STABILITY)
    require_fixed_implementation_binding(
        stability, "terminal stability evidence")
    postfit_tail, postfit_tail_hash = validated_postfit_tail_audit(
        POSTFIT_TAIL_AUDIT, STABILITY
    )
    nested_interaction, nested_interaction_hash = validated_nested_interaction(
        NESTED_INTERACTION, POSTFIT_TAIL_AUDIT, STABILITY
    )
    final_envelope, final_envelope_hash = validated_final_directional_envelope(
        FINAL_DIRECTIONAL_ENVELOPE,
        NESTED_INTERACTION,
        POSTFIT_TAIL_AUDIT,
        STABILITY,
    )
    figures = load(FIGURES)
    comparison = load(COMPARISON)
    incumbent = load(INCUMBENT)

    start_tags = [str(value) for value in starts.get("endpoint_tags", [])]
    replica_tags = [str(value) for value in replicas.get(
        "replica_endpoint_tags", [])]
    require(starts.get("status") in {"complete", "verification_failed"}
            and exact_number(starts.get("selected_strength"), 600.0)
            and exact_number(starts.get("selected_bmax"), 4.0)
            and exact_number(starts.get(
                "fit_quality_barrier_strength"), 100.0)
            and int(starts.get("fit_quality_barrier_power", -1)) == 2
            and int(starts.get("member_count", -1)) == 24
            and len(start_tags) == 24 and len(set(start_tags)) == 24,
            "terminal start evidence is not exact 24-member lambda600")
    start_seal = start_chain_audit.get("terminal_start_current_byte_seal")
    require(start_chain_audit.get("status") == "pass"
            and int(start_chain_audit.get("start_chain_count", -1)) == 24
            and start_chain_audit.get(
                "central_initializer_selection", {}).get(
                    "ordered_start_tags") == start_tags
            and isinstance(start_seal, dict),
            "terminal start-chain audit is incomplete")
    start_seal_path = Path(str(start_seal.get("path", "")))
    require(start_seal_path.parent.resolve() == START_SEAL_ROOT.resolve()
            and start_seal_path.is_file()
            and start_seal.get("sha256") == sha256(start_seal_path)
            and int(start_seal.get(
                "legacy_pre_receipt_checkpoint_count", -1)) >= 0
            and bool(str(start_seal.get(
                "historical_ancestry_limitation", "")).strip()),
            "terminal start current-byte seal is invalid")
    require(replicas.get("status") in {
                "complete", "complete_with_scientific_failures",
                "central_stationarity_failed", "replica_stationarity_failed"}
            and exact_number(replicas.get("selected_strength"), 600.0)
            and exact_number(replicas.get("selected_bmax"), 4.0)
            and exact_number(replicas.get(
                "fit_quality_barrier_strength"), 100.0)
            and int(replicas.get("fit_quality_barrier_power", -1)) == 2
            and replicas.get("central_endpoint_tag")
            and int(replicas.get("completed_replica_count", -1)) == 50
            and len(replica_tags) == 50 and len(set(replica_tags)) == 50,
            "terminal central/replica evidence is not exact 1+50 lambda600")
    endpoint_gate = final_promotion_gate(final_envelope)
    base_endpoint_gate = explicit_bool(
        stability.get("endpoint_gate_pass"), "stability.endpoint_gate_pass")
    stationarity_gate = explicit_bool(
        stability.get("candidate_stationarity_gate_pass"),
        "stability.candidate_stationarity_gate_pass")
    require(stability.get("status") == "complete"
            and int(stability.get("start_count", -1)) == 24
            and int(stability.get("replica_count", -1)) == 50
            and stability.get("comparison_champion_id") == INCUMBENT_ID
            and explicit_bool(stability.get("coverage_gate_pass"),
                              "stability.coverage_gate_pass")
            and explicit_bool(stability.get("band_integrity_gate_pass"),
                              "stability.band_integrity_gate_pass")
            and explicit_bool(stability.get("state_chain_gate_pass"),
                              "stability.state_chain_gate_pass"),
            "terminal stability comparison is incomplete")
    require(Path(stability.get(
                "fixed_challenger_protocol", "")).resolve()
                == FIXED_PROTOCOL.resolve()
            and stability.get("fixed_challenger_protocol_sha256")
                == fixed_protocol_hash,
            "terminal stability evidence is not bound to the locked protocol")
    require(Path(stability.get("fixed_fnp_reference", "")).resolve()
                == FIXED_FNP_REFERENCE.resolve()
            and stability.get("fixed_fnp_reference_sha256")
                == EXPECTED_FNP_REFERENCE_SHA256,
            "terminal stability evidence is not bound to the fixed FNP reference")
    require(incumbent.get("champion_id") == INCUMBENT_ID,
            "immutable lambda1 incumbent record is invalid")
    locked = stability.get(
        "comparison_champion_max_active_relative_full_width", {})
    registered = incumbent.get(
        "combined_fig6_max_active_relative_full_width", {})
    union_diagnostic = stability.get(
        "comparison_champion_union_mask_relative_full_width", {})
    require(set(locked) == {"u", "d"}
            and set(registered) == {"u", "d"}
            and set(union_diagnostic) == {"u", "d"}
            and all(exact_number(locked[key], LOCKED_INCUMBENT_WIDTHS[key])
                    and exact_number(registered[key], LOCKED_INCUMBENT_WIDTHS[key])
                    and float(union_diagnostic[key]) > 0.0
                    for key in ("u", "d")),
            "lambda1 promotion thresholds are not immutable registered widths")

    expected_figure_status = (
        "final_validated_figures" if endpoint_gate
        else "diagnostic_figures_not_promotable")
    require(figures.get("status") == expected_figure_status
            and explicit_bool(figures.get("updated_only"),
                              "figures.updated_only")
            and not explicit_bool(figures.get(
                "contains_legacy_conditional_result"),
                "figures.contains_legacy_conditional_result")
            and explicit_bool(figures.get("endpoint_gate_pass"),
                              "figures.endpoint_gate_pass") == endpoint_gate
            and figures.get("source_final_directional_envelope_sha256")
                == final_envelope_hash
            and explicit_bool(figures.get("candidate_stationarity_gate_pass"),
                              "figures.candidate_stationarity_gate_pass")
                == stationarity_gate
            and not explicit_bool(figures.get(
                "formal_confidence_level_assigned"),
                "figures.formal_confidence_level_assigned")
            and not explicit_bool(figures.get("one_sigma_claimed"),
                                  "figures.one_sigma_claimed")
            and "empirical product band plus residual convergence/interaction envelope"
                in figures.get("uncertainty", ""),
            "combined Fig. 2/Fig. 6 summary is inconsistent")
    figure_roles = {}
    for role, key in (("figure_2", "figure_2"), ("figure_6", "figure_6")):
        pdf = resolved_artifact(str(figures.get(key, "")), FIGURE_DIR)
        png = pdf.with_suffix(".png")
        require(pdf.suffix == ".pdf"
                and pdf.is_file() and pdf.stat().st_size > 0
                and png.is_file() and png.stat().st_size > 0,
                f"combined {role} PDF/PNG pair is missing")
        if endpoint_gate:
            require(pdf.stem.endswith("product_plus_directional_envelope"),
                    f"validated {role} filename overstates probability semantics")
        figure_roles[role] = [str(pdf.resolve()), str(png.resolve())]

    comparison_statuses = {
        "complete_validated_candidate_comparison",
        "complete_diagnostic_scientific_failure_comparison",
    }
    require(comparison.get("status") in comparison_statuses
            and comparison.get("comparison_champion_id") == INCUMBENT_ID
            and explicit_bool(comparison.get("candidate_endpoint_gate_pass"),
                              "comparison.candidate_endpoint_gate_pass")
                == endpoint_gate
            and explicit_bool(comparison.get(
                "candidate_base_stability_endpoint_gate_pass"),
                "comparison.candidate_base_stability_endpoint_gate_pass")
                == base_endpoint_gate
            and explicit_bool(comparison.get(
                "candidate_final_directional_envelope_gate_pass"),
                "comparison.candidate_final_directional_envelope_gate_pass")
                == endpoint_gate
            and comparison.get(
                "candidate_final_directional_envelope_sha256")
                == final_envelope_hash
            and explicit_bool(comparison.get(
                "candidate_stationarity_gate_pass"),
                "comparison.candidate_stationarity_gate_pass")
                == stationarity_gate
            and explicit_bool(comparison.get("diagnostic_only"),
                              "comparison.diagnostic_only") == (not endpoint_gate)
            and explicit_bool(comparison.get(
                "immutable_incumbent_hashes_validated"),
                "comparison.immutable_incumbent_hashes_validated")
            and not explicit_bool(comparison.get("training_protocol_harmonized"),
                                  "comparison.training_protocol_harmonized")
            and explicit_bool(comparison.get(
                "legacy_lambda1_fig6_widths_remain_gating"),
                "comparison.legacy_lambda1_fig6_widths_remain_gating")
            and not explicit_bool(comparison.get("production_sources_modified"),
                                  "comparison.production_sources_modified")
            and not explicit_bool(comparison.get("registry_modified"),
                                  "comparison.registry_modified")
            and not explicit_bool(comparison.get(
                "candidate_or_canonical_figures_modified"),
                "comparison.candidate_or_canonical_figures_modified")
            and not explicit_bool(comparison.get("frozen_sources_modified"),
                                  "comparison.frozen_sources_modified")
            and "empirical product band plus residual convergence/interaction envelope"
                in comparison.get(
                "interval_probability_semantics", ""),
            "explicit lambda600-versus-lambda1 comparison is invalid")
    declared_artifacts = comparison.get("artifacts")
    declared_hashes = comparison.get("artifact_sha256")
    declared_inputs = comparison.get("input_sha256")
    require(isinstance(declared_artifacts, dict)
            and isinstance(declared_hashes, dict)
            and isinstance(declared_inputs, dict) and declared_inputs,
            "comparison summary lacks artifact/hash maps")
    expected_inputs = {
        "candidate_summary": COMBINED / "summary.json",
        "candidate_audit": STABILITY,
        "candidate_postfit_tail_transform_audit": POSTFIT_TAIL_AUDIT,
        "candidate_nested_interaction_validation": NESTED_INTERACTION,
        "candidate_final_directional_envelope": FINAL_DIRECTIONAL_ENVELOPE,
        "candidate_final_fnp_envelope": Path(
            final_envelope["artifacts"]["fnp_final_envelope"]),
        "candidate_final_fig2_bspace_envelope": Path(
            final_envelope["artifacts"]["fig2_bspace_final_envelope"]),
        "candidate_final_fig6_kspace_envelope": Path(
            final_envelope["artifacts"]["fig6_kspace_final_envelope"]),
        "candidate_fnp_bands": COMBINED / "fnp_bands.csv",
        "candidate_bspace_bands": COMBINED / "bT_tmd_bands.csv",
        "candidate_kspace_bands": COMBINED / "kT_tmd_bands.csv",
        "candidate_kspace_ensemble_long":
            COMBINED / "kT_tmd_ensemble_long.csv",
        "candidate_bootstrap_width_statistic_deviations":
            STABILITY.parent / "bootstrap_full_width_statistic_deviations.csv",
        "candidate_split_half_width_statistic_differences":
            STABILITY.parent / "split_half_full_width_statistic_differences.csv",
        "harmonized_summary": HARMONIZED / "summary.json",
        "harmonized_provenance": HARMONIZED / "input_provenance.json",
        "harmonized_fnp_bands": HARMONIZED / "fnp_combined_bands.csv",
        "harmonized_bspace_bands": HARMONIZED / "bspace_combined_bands.csv",
        "harmonized_kspace_bands": HARMONIZED / "kspace_combined_bands.csv",
        "pinned_input_manifest": PIN_MANIFEST,
        "pinned_incumbent_record": INCUMBENT,
        "frozen_perturbative_reference": REFERENCE_BSPACE,
    }
    require(set(declared_inputs) == set(expected_inputs),
            "comparison input hash map has unexpected coverage")
    for key, path in expected_inputs.items():
        require(path.is_file() and path.stat().st_size > 0
                and declared_inputs[key] == sha256(path),
                f"comparison input changed after rendering: {path}")
    resolved_comparison = {
        key: resolved_artifact(str(value), COMPARISON_DIR)
        for key, value in declared_artifacts.items()
    }
    require({path.name for path in resolved_comparison.values()}
            == COMPARISON_ARTIFACT_NAMES,
            "comparison summary does not declare the exact expected artifacts")
    for key, path in resolved_comparison.items():
        expected_hash = (declared_hashes.get(key)
                         or declared_hashes.get(str(declared_artifacts[key]))
                         or declared_hashes.get(str(path))
                         or declared_hashes.get(str(path.resolve())))
        require(path.is_file() and path.stat().st_size > 0
                and expected_hash == sha256(path),
                f"comparison artifact hash mismatch: {path}")

    artifact_roles = {
        **figure_roles,
        "lambda600_vs_lambda1_comparison": [
            str(COMPARISON.resolve()),
            *[str(path.resolve()) for _, path in sorted(
                resolved_comparison.items())],
        ],
        "supporting_terminal_evidence": [
            str(FIXED_PROTOCOL.resolve()),
            str(FIXED_FNP_REFERENCE.resolve()),
            *[
                str(path.resolve())
                for _, path in sorted(FIXED_IMPLEMENTATION_FILES.items())
            ],
            str(STARTS.resolve()), str(REPLICAS.resolve()),
            str(START_CHAIN_AUDIT.resolve()), str(start_seal_path.resolve()),
            str(STABILITY.resolve()), str(FIGURES.resolve()),
            str(POSTFIT_TAIL_AUDIT.resolve()),
            *[
                str(Path(path).resolve())
                for path in postfit_tail.get("artifacts", {}).values()
            ],
            str(NESTED_INTERACTION.resolve()),
            *[
                str(Path(path).resolve())
                for path in nested_interaction.get("artifacts", {}).values()
            ],
            str(FINAL_DIRECTIONAL_ENVELOPE.resolve()),
            *[
                str(Path(path).resolve())
                for path in final_envelope.get("artifacts", {}).values()
            ],
            *[str(path.resolve()) for _, path in sorted(
                expected_inputs.items())],
        ],
    }
    artifact_paths = sorted({
        Path(path_text)
        for paths in artifact_roles.values() for path_text in paths
    }, key=str)
    require(all(path.is_file() and path.stat().st_size > 0
                for path in artifact_paths),
            "terminal evidence contains a missing artifact")
    summary = {
        "status": "pass",
        "audit_scope": (
            "combined lambda600 Fig. 2/Fig. 6 plus explicit lambda600-versus-"
            "immutable-lambda1 comparison, independent of promotion outcome"
        ),
        "comparison_champion_id": INCUMBENT_ID,
        "candidate_endpoint_gate_pass": endpoint_gate,
        "candidate_base_stability_endpoint_gate_pass": base_endpoint_gate,
        "candidate_postfit_tail_transform_gate_pass": explicit_bool(
            postfit_tail.get("promotion_validation_gate_pass"),
            "postfit_tail.promotion_validation_gate_pass",
        ),
        "candidate_nested_interaction_gate_pass": explicit_bool(
            nested_interaction.get("interaction_validation_gate_pass"),
            "nested_interaction.interaction_validation_gate_pass",
        ),
        "candidate_final_directional_envelope_gate_pass": endpoint_gate,
        "postfit_tail_transform_audit": str(POSTFIT_TAIL_AUDIT),
        "postfit_tail_transform_audit_sha256": postfit_tail_hash,
        "nested_interaction_validation": str(NESTED_INTERACTION),
        "nested_interaction_validation_sha256": nested_interaction_hash,
        "final_directional_envelope": str(FINAL_DIRECTIONAL_ENVELOPE),
        "final_directional_envelope_sha256": final_envelope_hash,
        "candidate_stationarity_gate_pass": stationarity_gate,
        "fixed_challenger_protocol": str(FIXED_PROTOCOL),
        "fixed_challenger_protocol_sha256": fixed_protocol_hash,
        "fixed_fnp_reference": str(FIXED_FNP_REFERENCE),
        "fixed_fnp_reference_sha256": EXPECTED_FNP_REFERENCE_SHA256,
        **fixed_implementation_binding(),
        "diagnostic_only": not endpoint_gate,
        "start_count": 24,
        "terminal_start_current_byte_seal": start_seal,
        "central_count": 1,
        "replica_count": 50,
        "artifact_roles": artifact_roles,
        "artifact_sha256": {
            str(path): sha256(path) for path in artifact_paths
        },
        "comparison_input_sha256": declared_inputs,
        "production_sources_modified": False,
        "registry_modified": False,
    }
    atomic_write_json(TARGET / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
