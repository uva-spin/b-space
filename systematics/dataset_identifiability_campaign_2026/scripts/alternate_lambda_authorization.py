#!/usr/bin/env python3
"""Fail-closed authorization latch for any post-lambda600 constraint trial.

The complete lambda=600 24-start x 50-replica comparison is a scientific
precondition, but it is not itself permission to launch another constraint.
Every dormant launcher must additionally be named in a separately authored
authorization record bound to the exact terminal decision hash.

This module never creates an authorization.  Its default CLI is a read-only
status report; ``--require`` exits nonzero unless the latch is open.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from fixed_challenger_protocol import (
    EXPECTED_FNP_REFERENCE_SHA256,
    FNP_REFERENCE as FIXED_FNP_REFERENCE,
    IMPLEMENTATION_FILES as FIXED_IMPLEMENTATION_FILES,
    PROTOCOL as FIXED_PROTOCOL,
    require_fixed_implementation_binding,
    validate_fixed_challenger_protocol,
)
from postfit_tail_transform_validation import validated_postfit_tail_audit
from nested_interaction_and_final_envelope_validation import (
    final_promotion_gate,
    validated_final_directional_envelope,
    validated_nested_interaction,
)
from promote_validated_final_champion import validated_published_promotion


BASE = Path(__file__).resolve().parents[1]
SUM = BASE / "summaries"
DECISION = SUM / "lambda600_like_for_like_decision/summary.json"
AUTHORIZATION = SUM / "alternate_lambda_authorization/authorization.json"
CURRENT_CHAMPION = SUM / "champion_registry/current.json"
FINAL_REPORT_SUMMARY = SUM / "final_study_report/summary.json"
FINAL_REPORT_MARKDOWN = SUM / "final_study_report/FINAL_REPORT.md"
START_CHAIN_AUDIT = SUM / "lambda600_start_chain_audit/summary.json"
START_SEAL_ROOT = SUM / "lambda600_start_chain_audit/current_byte_seals"
REPORT_INPUTS = {
    SUM / "replica_robust_reference_full24/summary.json",
    START_CHAIN_AUDIT,
    SUM / "selected_reference_central_replicas/summary.json",
    SUM / "final_combined_tmd_ensemble/summary.json",
    SUM / "final_combined_ensemble_stability/summary.json",
    SUM / "lambda600_postfit_tail_transform_audit/summary.json",
    SUM / "lambda600_nested_start_replica_interaction/summary.json",
    SUM / "lambda600_final_directional_envelope/summary.json",
    SUM / "final_fig2_fig6/summary.json",
    SUM / "lambda600_vs_lambda1_diagnostic/summary.json",
    SUM / "lambda600_terminal_evidence/summary.json",
    SUM / "frozen_input_audit/summary.json",
    SUM / "champion_registry/empirical_reference_lambda1_b0p1_2p0_full24.json",
    FIXED_PROTOCOL,
    FIXED_FNP_REFERENCE,
}
PROMOTION_REPORT_INPUT = SUM / "campaign_completion_audit/summary.json"
IMMUTABLE_INCUMBENT_ID = "empirical_reference_lambda1_b0p1_2p0_full24"
LOCKED_INCUMBENT_WIDTHS = {
    "u": 0.11772613918747582,
    "d": 0.12490071924111977,
}
ALLOWED_TERMINAL_STATUSES = {
    "candidate_promoted_as_new_study_champion",
    "candidate_rejected",
}
LEGACY_GENERIC_PATH_LAUNCHERS = frozenset({
    "finish_campaign_automatically",
    "escalate_reference_strength_after_full50_failure",
    "recover_after_lower_survival",
    "continue_minimum_fitbar_search",
    "continue_minimum_barrier_search",
    "finish_after_boundary_confirmation",
    "recover_selected_replicas_extended80k",
    "continue_after_barrier_stress_with_recovery",
    "recover_barrier_reference_strength_after_stress_failure",
})
# These names are recognized only so old automation fails with an explicit
# scientific/provenance reason. A future post-verdict trial must register a new
# namespaced launcher and protocol; it must not reuse generic lambda600 paths.
KNOWN_LAUNCHERS = set(LEGACY_GENERIC_PATH_LAUNCHERS)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_number(observed: object, expected: float) -> bool:
    try:
        value = float(observed)
    except (TypeError, ValueError):
        return False
    return (math.isfinite(value)
            and math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12))


def explicit_bool(value: object, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise RuntimeError(f"{label} is not an explicit boolean")


def _load_object(path: Path, label: str) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"missing {label}: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} is not a JSON object: {path}")
    return payload


def require_terminal_gate_consistency(
        decision: dict, final_envelope: dict,
        terminal_evidence: dict) -> bool:
    """Require one Boolean promotion truth across all terminal endpoints."""
    final_gate = final_promotion_gate(final_envelope)
    decision_gate = explicit_bool(
        decision.get("promotion_gate_pass"), "decision.promotion_gate_pass")
    evidence_gate = explicit_bool(
        terminal_evidence.get("candidate_endpoint_gate_pass"),
        "terminal_evidence.candidate_endpoint_gate_pass",
    )
    status = decision.get("status")
    promoted = status == "candidate_promoted_as_new_study_champion"
    rejected = status == "candidate_rejected"
    if not (
            (promoted and final_gate and decision_gate and evidence_gate)
            or (rejected and not final_gate and not decision_gate
                and not evidence_gate)):
        raise RuntimeError(
            "terminal status, final promotion gate, decision promotion gate, "
            "and terminal-evidence endpoint flag disagree"
        )
    diagnostic = explicit_bool(
        terminal_evidence.get("diagnostic_only"),
        "terminal_evidence.diagnostic_only",
    )
    if diagnostic == final_gate:
        raise RuntimeError(
            "terminal-evidence diagnostic flag disagrees with promotion gate")
    return final_gate


def validated_prepublication_final_report(
        expected_outcome: str) -> tuple[dict, str]:
    """Validate the report written before the atomic terminal decision.

    The prepublication report cannot bind the not-yet-existing decision
    without creating a hash cycle.  It instead binds every upstream input;
    the terminal decision then binds both this summary and the rendered
    Markdown bytes.
    """
    if expected_outcome not in ALLOWED_TERMINAL_STATUSES:
        raise RuntimeError(f"invalid final-report outcome: {expected_outcome}")
    summary = _load_object(FINAL_REPORT_SUMMARY, "final study report summary")
    expected_status = (
        "complete_fixed_challenger_report_promoted"
        if expected_outcome == "candidate_promoted_as_new_study_champion"
        else "complete_fixed_challenger_report_rejected"
    )
    if not (
            summary.get("status") == expected_status
            and summary.get("outcome") == expected_outcome
            and summary.get("decision_validation_mode")
                == "prepublication_complete_graph_revalidated"
            and summary.get("terminal_decision_sha256") is None
            and Path(summary.get("report", "")).resolve()
                == FINAL_REPORT_MARKDOWN.resolve()
            and not explicit_bool(summary.get("registry_modified_by_report_writer"),
                                  "report.registry_modified_by_report_writer")
            and not explicit_bool(summary.get("frozen_sources_modified"),
                                  "report.frozen_sources_modified")
            and not explicit_bool(summary.get("production_sources_modified"),
                                  "report.production_sources_modified")):
        raise RuntimeError(
            "final study report does not match the prepublication outcome")
    artifacts = summary.get("artifact_sha256")
    if (not isinstance(artifacts, dict)
            or set(artifacts) != {"final_report"}
            or not FINAL_REPORT_MARKDOWN.is_file()
            or sha256(FINAL_REPORT_MARKDOWN) != artifacts["final_report"]):
        raise RuntimeError("final study report Markdown changed or is unbound")
    inputs = summary.get("input_sha256")
    expected_inputs = set(REPORT_INPUTS)
    start_ancestry = summary.get("start_checkpoint_ancestry")
    if not isinstance(start_ancestry, dict):
        raise RuntimeError(
            "final report omits the disclosed start-ancestry limitation")
    seal = Path(str(start_ancestry.get("seal", "")))
    if (seal.parent.resolve() != START_SEAL_ROOT.resolve()
            or not seal.is_file()
            or str(start_ancestry.get("seal_sha256")) != sha256(seal)
            or int(start_ancestry.get(
                "legacy_pre_receipt_checkpoint_count", -1)) < 0
            or not str(start_ancestry.get(
                "historical_ancestry_limitation", "")).strip()):
        raise RuntimeError("final report start-ancestry binding is invalid")
    expected_inputs.add(seal)
    if expected_outcome == "candidate_promoted_as_new_study_champion":
        expected_inputs.add(PROMOTION_REPORT_INPUT)
    if (not isinstance(inputs, dict)
            or {Path(path).resolve() for path in inputs}
                != {path.resolve() for path in expected_inputs}):
        raise RuntimeError(
            "final study report input binding is incomplete or excessive")
    for path_text, expected_hash in inputs.items():
        path = Path(path_text)
        if (not path.is_file() or not expected_hash
                or sha256(path) != expected_hash):
            raise RuntimeError(f"final study report input changed: {path}")
    return summary, sha256(FINAL_REPORT_SUMMARY)


def validate_complete_lambda600_comparison() -> tuple[dict, str]:
    """Require exact terminal 24+1+50 evidence and immutable-lambda1 comparison."""
    _, fixed_protocol_hash = validate_fixed_challenger_protocol()
    decision = _load_object(DECISION, "lambda600 terminal decision")
    require_fixed_implementation_binding(
        decision, "lambda600 terminal decision")
    candidate = decision.get("candidate", {})
    starts = decision.get("full24")
    replicas = decision.get("replicas")
    stability = decision.get("stability")
    figures = decision.get("figures")
    comparison = decision.get("comparison")
    postfit_tail, postfit_tail_hash = validated_postfit_tail_audit()
    nested_interaction, nested_interaction_hash = validated_nested_interaction()
    final_envelope, final_envelope_hash = validated_final_directional_envelope()
    if not (
            decision.get("status") in ALLOWED_TERMINAL_STATUSES
            and decision.get("stage") == "complete_like_for_like_comparison"
            and exact_number(candidate.get("reference_strength"), 600.0)
            and exact_number(candidate.get("reference_bmax"), 4.0)
            and exact_number(candidate.get(
                "fit_quality_barrier_strength"), 100.0)
            and exact_number(candidate.get("fit_quality_barrier_power"), 2.0)
            and isinstance(starts, dict)
            and starts.get("status") in {"complete", "verification_failed"}
            and exact_number(starts.get("selected_strength"), 600.0)
            and int(starts.get("member_count", -1)) == 24
            and len(starts.get("endpoint_tags", [])) == 24
            and len(set(starts.get("endpoint_tags", []))) == 24
            and isinstance(replicas, dict)
            and replicas.get("status") in {
                "complete", "complete_with_scientific_failures",
                "central_stationarity_failed", "replica_stationarity_failed"}
            and exact_number(replicas.get("selected_strength"), 600.0)
            and replicas.get("central_endpoint_tag")
            and int(replicas.get("completed_replica_count", -1)) == 50
            and len(replicas.get("replica_endpoint_tags", [])) == 50
            and len(set(replicas.get("replica_endpoint_tags", []))) == 50
            and isinstance(stability, dict)
            and stability.get("status") == "complete"
            and int(stability.get("start_count", -1)) == 24
            and int(stability.get("replica_count", -1)) == 50
            and stability.get("comparison_champion_id")
                == IMMUTABLE_INCUMBENT_ID
            and all(key in stability.get(
                "final_max_active_relative_full_width", {})
                    for key in ("u", "d"))
            and all(key in stability.get(
                "comparison_champion_max_active_relative_full_width", {})
                    for key in ("u", "d"))
            and all(exact_number(stability[
                "comparison_champion_max_active_relative_full_width"][key],
                LOCKED_INCUMBENT_WIDTHS[key]) for key in ("u", "d"))
            and all(key in stability.get(
                "comparison_champion_union_mask_relative_full_width", {})
                    for key in ("u", "d"))
            and isinstance(figures, dict)
            and figures.get("figure_2")
            and figures.get("figure_6")
            and explicit_bool(figures.get("updated_only"),
                              "figures.updated_only")
            and not explicit_bool(figures.get(
                "contains_legacy_conditional_result"),
                "figures.contains_legacy_conditional_result")
            and isinstance(comparison, dict)
            and comparison.get("status") in {
                "complete_validated_candidate_comparison",
                "complete_diagnostic_scientific_failure_comparison"}
            and comparison.get("comparison_champion_id")
                == IMMUTABLE_INCUMBENT_ID
            and decision.get("postfit_tail_transform_audit") == postfit_tail
            and decision.get("postfit_tail_transform_audit_sha256")
                == postfit_tail_hash
            and decision.get("nested_interaction_validation")
                == nested_interaction
            and decision.get("nested_interaction_validation_sha256")
                == nested_interaction_hash
            and decision.get("final_directional_envelope")
                == final_envelope
            and decision.get("final_directional_envelope_sha256")
                == final_envelope_hash
            and not explicit_bool(decision.get("production_sources_modified"),
                                  "decision.production_sources_modified")):
        raise RuntimeError(
            "lambda600 has not produced a complete exact 24+1+50 comparison")

    start_binding = decision.get("terminal_start_ancestry")
    start_audit = _load_object(
        START_CHAIN_AUDIT, "terminal start-chain ancestry audit")
    start_seal = start_audit.get("terminal_start_current_byte_seal")
    if not isinstance(start_seal, dict):
        raise RuntimeError("terminal start-chain audit lacks its current-byte seal")
    seal_path = Path(str(start_seal.get("path", "")))
    if (not isinstance(start_binding, dict)
            or start_binding.get("status") != "pass"
            or Path(start_binding.get("manifest", "")).resolve()
                != START_CHAIN_AUDIT.resolve()
            or start_binding.get("manifest_sha256")
                != sha256(START_CHAIN_AUDIT)
            or start_binding.get("central_initializer_selection")
                != start_audit.get("central_initializer_selection")
            or start_binding.get("terminal_start_current_byte_seal")
                != start_seal
            or seal_path.parent.resolve() != START_SEAL_ROOT.resolve()
            or not seal_path.is_file()
            or start_seal.get("sha256") != sha256(seal_path)
            or int(start_seal.get(
                "legacy_pre_receipt_checkpoint_count", -1)) < 0
            or not str(start_seal.get(
                "historical_ancestry_limitation", "")).strip()):
        raise RuntimeError(
            "terminal lambda600 decision lacks the exact disclosed start seal")

    evidence = decision.get("terminal_evidence")
    if not isinstance(evidence, dict) or evidence.get("status") != "pass":
        raise RuntimeError(
            "terminal lambda600 decision lacks a passing bound-evidence audit")
    manifest_text = evidence.get("manifest")
    manifest_hash = evidence.get("manifest_sha256")
    if not manifest_text or not manifest_hash:
        raise RuntimeError("terminal evidence manifest binding is incomplete")
    manifest = Path(manifest_text)
    if (not manifest.is_file() or manifest.resolve().parent !=
            (SUM / "lambda600_terminal_evidence").resolve()
            or sha256(manifest) != manifest_hash):
        raise RuntimeError("terminal evidence manifest changed or is misplaced")
    manifest_payload = _load_object(manifest, "terminal evidence manifest")
    require_fixed_implementation_binding(
        manifest_payload, "terminal evidence manifest")
    if (manifest_payload.get("status") != "pass"
            or manifest_payload.get("comparison_champion_id")
                != IMMUTABLE_INCUMBENT_ID
            or int(manifest_payload.get("start_count", -1)) != 24
            or int(manifest_payload.get("replica_count", -1)) != 50):
        raise RuntimeError("terminal evidence manifest schema is invalid")
    gate_pass = require_terminal_gate_consistency(
        decision, final_envelope, manifest_payload)
    if (Path(manifest_payload.get(
                "fixed_challenger_protocol", "")).resolve()
                != FIXED_PROTOCOL.resolve()
            or manifest_payload.get("fixed_challenger_protocol_sha256")
                != fixed_protocol_hash):
        raise RuntimeError(
            "terminal evidence is not bound to the locked challenger protocol")
    if (Path(manifest_payload.get("fixed_fnp_reference", "")).resolve()
                != FIXED_FNP_REFERENCE.resolve()
            or manifest_payload.get("fixed_fnp_reference_sha256")
                != EXPECTED_FNP_REFERENCE_SHA256):
        raise RuntimeError(
            "terminal evidence is not bound to the fixed FNP reference")
    bound = manifest_payload.get("artifact_sha256")
    roles = manifest_payload.get("artifact_roles")
    if not isinstance(bound, dict) or not isinstance(roles, dict):
        raise RuntimeError("terminal evidence manifest lacks artifact bindings")
    for path_text, expected_hash in bound.items():
        path = Path(path_text)
        if (not path.is_file() or not expected_hash
                or sha256(path) != expected_hash):
            raise RuntimeError(f"terminal evidence artifact changed: {path}")
    required_roles = {
        "figure_2", "figure_6", "lambda600_vs_lambda1_comparison",
        "supporting_terminal_evidence",
    }
    if not required_roles.issubset(roles):
        raise RuntimeError("terminal evidence lacks Fig2/Fig6/comparison roles")
    if str(FIXED_PROTOCOL.resolve()) not in roles["supporting_terminal_evidence"]:
        raise RuntimeError(
            "terminal evidence supporting role omits the locked protocol")
    if (str(FIXED_FNP_REFERENCE.resolve())
            not in roles["supporting_terminal_evidence"]):
        raise RuntimeError(
            "terminal evidence supporting role omits the fixed FNP reference")
    if (str(START_CHAIN_AUDIT.resolve())
            not in roles["supporting_terminal_evidence"]
            or str(seal_path.resolve())
            not in roles["supporting_terminal_evidence"]):
        raise RuntimeError(
            "terminal evidence omits the disclosed start-chain seal")
    for role, path in FIXED_IMPLEMENTATION_FILES.items():
        if str(path.resolve()) not in roles["supporting_terminal_evidence"]:
            raise RuntimeError(
                "terminal evidence supporting role omits fixed "
                f"implementation {role}")
    for role in required_roles:
        paths = roles[role]
        if not isinstance(paths, list) or not paths:
            raise RuntimeError(f"terminal evidence role {role} is empty")
        for path_text in paths:
            path = Path(path_text)
            if (path_text not in bound or not path.is_file()
                    or sha256(path) != bound[path_text]):
                raise RuntimeError(
                    f"terminal evidence artifact changed: {path}")
    if gate_pass:
        post_promotion = decision.get("post_promotion_validation")
        observed_promotion = validated_published_promotion()
        if (not isinstance(post_promotion, dict)
                or post_promotion != observed_promotion):
            raise RuntimeError(
                "promoted terminal decision lacks exact post-promotion validation")
    else:
        current = _load_object(CURRENT_CHAMPION, "current champion registry")
        if (current.get("champion_id") != IMMUTABLE_INCUMBENT_ID
                or decision.get("post_promotion_validation") not in (None, {})):
            raise RuntimeError(
                "rejected terminal decision changed the incumbent or carries "
                "promotion evidence")
    report, report_summary_hash = validated_prepublication_final_report(
        str(decision["status"]))
    report_binding = decision.get("final_study_report")
    expected_report_binding = {
        "status": "pass",
        "summary": str(FINAL_REPORT_SUMMARY),
        "summary_sha256": report_summary_hash,
        "report": str(FINAL_REPORT_MARKDOWN),
        "report_sha256": report["artifact_sha256"]["final_report"],
        "outcome": decision["status"],
    }
    if report_binding != expected_report_binding:
        raise RuntimeError(
            "terminal decision lacks the exact final study report binding")
    return decision, sha256(DECISION)


def authorization_status(launcher: str) -> dict:
    if launcher not in KNOWN_LAUNCHERS:
        return {
            "status": "blocked",
            "authorized": False,
            "launcher": launcher,
            "reason": "unknown alternate-constraint launcher identity",
        }
    if launcher in LEGACY_GENERIC_PATH_LAUNCHERS:
        return {
            "status": "blocked",
            "authorized": False,
            "launcher": launcher,
            "reason": (
                "legacy generic-path launcher is permanently prohibited after "
                "the fixed lambda600 challenge; register a new namespaced "
                "launcher and immutable protocol for any future trial"),
            "legacy_generic_path_launcher": True,
        }
    try:
        decision, decision_hash = validate_complete_lambda600_comparison()
    except Exception as error:
        return {
            "status": "blocked",
            "authorized": False,
            "launcher": launcher,
            "reason": str(error),
            "lambda600_comparison_complete": False,
        }
    try:
        authorization = _load_object(
            AUTHORIZATION, "alternate-lambda authorization")
        launchers = authorization.get("authorized_launchers")
        if not (
                authorization.get("status") == "authorized"
                and authorization.get("authorization_kind") ==
                    "alternate_constraint_trial_after_complete_lambda600_comparison"
                and Path(authorization.get(
                    "lambda600_decision", "")).resolve() == DECISION.resolve()
                and authorization.get("lambda600_decision_sha256")
                    == decision_hash
                and authorization.get("lambda600_terminal_status")
                    == decision.get("status")
                and isinstance(launchers, list)
                and len(launchers) == len(set(launchers))
                and launcher in launchers
                and set(launchers).issubset(KNOWN_LAUNCHERS)
                and str(authorization.get("authorized_by", "")).strip()
                and str(authorization.get(
                    "authorization_timestamp", "")).strip()
                and str(authorization.get("scientific_rationale", "")).strip()
                and not explicit_bool(authorization.get(
                    "production_sources_modified"),
                    "authorization.production_sources_modified")):
            raise RuntimeError(
                "authorization record is incomplete, stale, or excludes launcher")
    except Exception as error:
        return {
            "status": "blocked",
            "authorized": False,
            "launcher": launcher,
            "reason": str(error),
            "lambda600_comparison_complete": True,
            "lambda600_decision_sha256": decision_hash,
        }
    return {
        "status": "authorized",
        "authorized": True,
        "launcher": launcher,
        "lambda600_comparison_complete": True,
        "lambda600_decision_sha256": decision_hash,
        "authorization": str(AUTHORIZATION),
        "authorization_sha256": sha256(AUTHORIZATION),
    }


def require_alternate_lambda_authorization(launcher: str) -> dict:
    result = authorization_status(launcher)
    if not result["authorized"]:
        raise RuntimeError(
            f"alternate-lambda launch blocked for {launcher}: "
            f"{result['reason']}")
    return result


def authorization_template(launcher: str) -> dict:
    if launcher in LEGACY_GENERIC_PATH_LAUNCHERS:
        raise RuntimeError(
            "cannot authorize a legacy generic-path launcher; create a newly "
            "registered namespaced launcher/protocol")
    decision_hash = None
    terminal_status = None
    try:
        decision, decision_hash = validate_complete_lambda600_comparison()
        terminal_status = decision["status"]
    except Exception:
        pass
    return {
        "status": "authorized",
        "authorization_kind":
            "alternate_constraint_trial_after_complete_lambda600_comparison",
        "lambda600_decision": str(DECISION),
        "lambda600_decision_sha256": decision_hash or
            "REPLACE_AFTER_TERMINAL_COMPARISON",
        "lambda600_terminal_status": terminal_status or
            "REPLACE_AFTER_TERMINAL_COMPARISON",
        "authorized_launchers": [launcher],
        "authorized_by": "REQUIRED_EXPLICIT_IDENTITY",
        "authorization_timestamp": "REQUIRED_ISO8601_TIMESTAMP",
        "scientific_rationale": "REQUIRED_SCIENTIFIC_RATIONALE",
        "production_sources_modified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launcher", required=True,
                        choices=sorted(KNOWN_LAUNCHERS))
    parser.add_argument("--require", action="store_true",
                        help="exit nonzero unless the launch latch is open")
    parser.add_argument("--print-template", action="store_true",
                        help="print, but never write, a candidate record schema")
    args = parser.parse_args()
    if args.print_template:
        print(json.dumps(authorization_template(args.launcher), indent=2))
        return
    result = authorization_status(args.launcher)
    print(json.dumps(result, indent=2))
    if args.require and not result["authorized"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
