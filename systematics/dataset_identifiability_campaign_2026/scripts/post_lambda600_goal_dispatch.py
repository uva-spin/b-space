#!/usr/bin/env python3
"""Record the goal-level state after the fixed lambda600 verdict.

The lambda600 controller owns one pre-registered incumbent-replacement test.
Promotion or rejection completes that test, but it cannot establish the
minimum sufficient non-physics constraint by itself.  This dispatcher makes
that distinction machine-readable and hands the persistent study to a
failure-mode-driven analysis stage without choosing or launching another
prior.

Every legacy follow-on launcher writes to generic summary paths that belong to
the now hash-bound lambda600 evidence graph.  They are therefore explicitly
prohibited by this contract.  A future trial must first receive its own locked
protocol and disjoint output/summary namespace.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from alternate_lambda_authorization import (
    LEGACY_GENERIC_PATH_LAUNCHERS,
    validate_complete_lambda600_comparison,
)
from nested_interaction_and_final_envelope_validation import (
    final_promotion_gate,
)


BASE = Path(__file__).resolve().parents[1]
SUM = BASE / "summaries"
CLASSIFICATION = SUM / "lambda600_outcome_classification/summary.json"
TARGET = SUM / "post_lambda600_goal_dispatch"

PROMOTED = "candidate_promoted_as_new_study_champion"
REJECTED = "candidate_rejected"
PROMOTED_CLASSIFICATION = "validated_final_band_improvement"
REJECTED_CLASSIFICATIONS = {
    "complete_comparison_with_lambda600_stationarity_failure",
    "postfit_tail_or_transform_validation_failure",
    "nested_start_replica_interaction_failure",
    "trained_central_containment_failure",
    "finite_ensemble_resolution_failure_only",
    "final_joint_directional_band_not_better_than_incumbent",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def explicit_bool(value: object, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise RuntimeError(f"{label} is not an explicit boolean")


def load_object(path: Path, label: str) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"missing {label}: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} is not a JSON object: {path}")
    return payload


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(
            path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def required_analysis(classification: str, outcome: str) -> tuple[str, list[str]]:
    if outcome == PROMOTED:
        return (
            "minimum_constraint_necessity_and_ablation_design",
            [
                "freeze lambda600 as the new comparison champion before any new trial",
                "separate the necessity of reference strength, constrained bT extent, "
                "and fit-quality safeguard with one-factor controlled ablations",
                "use screening only to select one fixed challenger; require the complete "
                "end-to-end evidence graph before another replacement decision",
            ],
        )
    mapping = {
        "complete_comparison_with_lambda600_stationarity_failure": (
            "failure_mode_driven_stationarity_analysis",
            [
                "identify the exact failing start, central, or replica trajectories",
                "localize their terminal FNP motion in bT before choosing one response",
            ],
        ),
        "postfit_tail_or_transform_validation_failure": (
            "failure_mode_driven_tail_transform_analysis",
            [
                "separate bT<=4 stationarity from bT=4--8 motion",
                "identify which expb2, expb, or taper decision sign failed",
            ],
        ),
        "nested_start_replica_interaction_failure": (
            "failure_mode_driven_interaction_analysis",
            [
                "expand or diagnose the same-prior nested start-by-replica stress",
                "do not change a constraint merely to hide nonseparability",
            ],
        ),
        "trained_central_containment_failure": (
            "failure_mode_driven_central_envelope_analysis",
            [
                "locate the FNP, Fig.2, or Fig.6 containment failure",
                "separate an envelope-construction defect from a model failure",
            ],
        ),
        "finite_ensemble_resolution_failure_only": (
            "same_prior_sampling_extension_design",
            [
                "increase lambda600 start or replica resolution under a new locked "
                "extension protocol",
                "recompute the exact final-statistic allowance before changing the prior",
            ],
        ),
        "final_joint_directional_band_not_better_than_incumbent": (
            "failure_mode_driven_final_width_decomposition",
            [
                "separate product, convergence, interaction, and finite-sampling width",
                "choose at most one controlled constraint or architecture response from "
                "the dominant component",
            ],
        ),
    }
    if classification not in mapping:
        raise RuntimeError(
            f"no goal-level analysis contract for classification {classification}")
    return mapping[classification]


def dispatch(
    *, classification_path: Path = CLASSIFICATION,
    target: Path = TARGET,
    decision_validator=None,
    emit: bool = True,
) -> dict:
    """Validate the exact verdict and atomically publish the broader-goal state."""
    validator = (validate_complete_lambda600_comparison
                 if decision_validator is None else decision_validator)
    decision, decision_hash = validator()
    outcome = str(decision.get("status"))
    if outcome not in {PROMOTED, REJECTED}:
        raise RuntimeError("lambda600 decision is not scientifically terminal")
    final_envelope = decision.get("final_directional_envelope")
    if not isinstance(final_envelope, dict):
        raise RuntimeError("terminal decision lacks final directional-envelope evidence")
    final_gate = final_promotion_gate(final_envelope)
    if (outcome == PROMOTED) != final_gate:
        raise RuntimeError("terminal outcome disagrees with the exact final gate")

    classification_path = Path(classification_path)
    classifier = load_object(classification_path, "lambda600 outcome classification")
    classification = str(classifier.get("classification"))
    expected_classifications = (
        {PROMOTED_CLASSIFICATION} if outcome == PROMOTED
        else REJECTED_CLASSIFICATIONS
    )
    if not (
        classifier.get("status") == "complete"
        and classifier.get("lambda600_stage")
            == "complete_like_for_like_comparison"
        and classification in expected_classifications
        and classifier.get("terminal_decision_sha256") == decision_hash
        and explicit_bool(classifier.get("promotion_gate_pass"),
                          "classifier promotion_gate_pass") == final_gate
        and not explicit_bool(classifier.get("another_constraint_selected"),
                              "classifier another_constraint_selected")
        and not explicit_bool(classifier.get("production_sources_modified"),
                              "classifier production_sources_modified")
    ):
        raise RuntimeError(
            "lambda600 classifier is stale or not bound to the exact terminal decision")

    next_stage, analysis = required_analysis(classification, outcome)
    prohibited = sorted(LEGACY_GENERIC_PATH_LAUNCHERS)
    payload = {
        "status": "fixed_challenger_complete_broader_study_incomplete",
        "fixed_lambda600_challenger_complete": True,
        "lambda600_terminal_status": outcome,
        "lambda600_terminal_decision_sha256": decision_hash,
        "lambda600_classification": classification,
        "lambda600_classification_summary": str(classification_path),
        "lambda600_classification_summary_sha256": sha256(classification_path),
        "broader_study_complete": False,
        "minimum_nonphysics_constraint_established": False,
        "another_constraint_selected": False,
        "next_trial_launch_authorized": False,
        "authorization_record_created": False,
        "dispatcher_launches_processes": False,
        "next_stage": next_stage,
        "required_failure_mode_analysis": analysis,
        "legacy_generic_path_launchers_prohibited": prohibited,
        "legacy_launcher_prohibition_reason": (
            "these launchers reuse generic summary/output paths that are now part "
            "of the hash-bound lambda600 terminal graph"
        ),
        "required_before_any_next_trial": [
            "preserve the complete lambda600 terminal graph and current champion bytes",
            "select exactly one failure-mode-driven or one-factor ablation hypothesis; "
            "do not start a lambda ladder",
            "register a new fixed protocol bound to this terminal decision and "
            "classification hash",
            "use disjoint namespaced outputs, summaries, figures, and decision records",
            "define fit, stationarity, coverage, interaction, transform, and direct "
            "current-champion replacement gates before candidate evidence exists",
            "create a separate authorization only after the namespaced protocol and "
            "launcher have passed fail-closed review",
        ],
        "broader_completion_evidence_still_required": [
            "a defensible necessity/minimum bracket or lower-bound ablation for the "
            "non-physics constraint package",
            "a complete end-to-end comparison for every candidate proposed to replace "
            "the current champion",
            "final champion Fig.2 and Fig.6 with experimental and residual "
            "nonuniqueness propagation under the winning audited protocol",
        ],
        "production_sources_modified": False,
    }
    atomic_write_json(Path(target) / "summary.json", payload)
    if emit:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


if __name__ == "__main__":
    dispatch()
