#!/usr/bin/env python3
"""Promote the fully audited isolated result into the study champion registry.

This updates only campaign-local registry artifacts.  It deliberately runs
after the completion audit, which compares the candidate with the incumbent;
therefore replacing ``current.json`` cannot weaken its own comparison gate.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

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
SUMMARIES = BASE / "summaries"
REGISTRY = SUMMARIES / "champion_registry"
SOURCE = SUMMARIES / "final_combined_tmd_ensemble"
STABILITY = SUMMARIES / "final_combined_ensemble_stability/summary.json"
POSTFIT_TAIL_AUDIT = (
    SUMMARIES / "lambda600_postfit_tail_transform_audit/summary.json"
)
NESTED_INTERACTION = (
    SUMMARIES / "lambda600_nested_start_replica_interaction/summary.json"
)
FINAL_DIRECTIONAL_ENVELOPE = (
    SUMMARIES / "lambda600_final_directional_envelope/summary.json"
)
COMPLETION = SUMMARIES / "campaign_completion_audit/summary.json"
FIGURES = SUMMARIES / "final_fig2_fig6"
STARTS = SUMMARIES / "replica_robust_reference_full24/summary.json"
REPLICAS = SUMMARIES / "selected_reference_central_replicas/summary.json"
IMMUTABLE_INCUMBENT = (
    REGISTRY / "empirical_reference_lambda1_b0p1_2p0_full24.json")
STATE_CHAIN_SCRIPT = BASE / "scripts/audit_lambda600_state_chains.py"
STATE_CHAIN_AUDIT = SUMMARIES / "lambda600_state_chain_audit/summary.json"
TERMINAL_EVIDENCE = SUMMARIES / "lambda600_terminal_evidence/summary.json"
COMPARISON = SUMMARIES / "lambda600_vs_lambda1_diagnostic/summary.json"
PUBLIC = REGISTRY / "current_fig2_fig6"
CURRENT = REGISTRY / "current.json"
PROMOTION_TRANSACTION = REGISTRY / ".lambda600_promotion_transaction.json"
LOCKED_INCUMBENT_WIDTHS = {
    "u": 0.11772613918747582,
    "d": 0.12490071924111977,
}
INCUMBENT_ID = "empirical_reference_lambda1_b0p1_2p0_full24"
EXPECTED_CHAMPION_ID = (
    "empirical_reference_lambda600_fitbarp2mu100_b0p1_4_full24x50"
)
PUBLIC_COPIES = {
    "updated_fnp": (
        FIGURES / "updated_fnp_bspace_product_plus_directional_envelope.png",
        PUBLIC / "champion_fnp_bT_product_plus_directional_envelope.png"),
    "updated_only_fig2_space": (
        FIGURES / "updated_fig2_bspace_product_plus_directional_envelope.png",
        PUBLIC / "champion_fig2space_bT_product_plus_directional_envelope.png"),
    "updated_only_fig6": (
        FIGURES / "updated_fig6_kspace_ud_product_plus_directional_envelope.png",
        PUBLIC / "champion_fig6_kT_ud_product_plus_directional_envelope.png"),
}
REGISTERED_K = PUBLIC / "champion_kspace_combined_bands.csv"
PUBLIC_SUMMARY = PUBLIC / "summary.json"
REQUIRED_PROMOTION_ARTIFACT_KEYS = {
    "combined_summary",
    "fnp_combined_bands",
    "bspace_combined_bands",
    "kspace_combined_bands",
    "completion_audit",
    "state_chain_audit",
    "terminal_evidence_audit",
    "postfit_tail_transform_audit",
    "nested_start_replica_interaction_audit",
    "final_directional_envelope_audit",
    "final_fig6_component_width_curves",
    "lambda600_vs_lambda1_comparison",
    *PUBLIC_COPIES,
}


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_json(path: Path, payload: dict) -> None:
    """Publish a complete, durable JSON object without a torn target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_copy(source: Path, target: Path) -> None:
    """Copy one artifact atomically; a retry sees either old or complete bytes."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    try:
        shutil.copy2(source, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_frame(path: Path, frame: pd.DataFrame) -> None:
    """Publish a CSV atomically and durably."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        frame.to_csv(temporary, index=False)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def validated_artifact_bindings(
        record: dict, *, required_keys: set[str] | None = None) -> dict:
    """Validate exact role/hash coverage and every referenced artifact byte."""
    artifacts = record.get("artifacts")
    hashes = record.get("artifact_sha256")
    expected = set(artifacts) if required_keys is None and isinstance(
        artifacts, dict) else required_keys
    _require(
        isinstance(artifacts, dict) and expected is not None
        and set(artifacts) == set(expected)
        and isinstance(hashes, dict) and set(hashes) == set(artifacts),
        "promoted champion artifact/hash coverage is incomplete",
    )
    for role, path_text in artifacts.items():
        path = Path(path_text)
        _require(path.is_file() and path.stat().st_size > 0,
                 f"promoted artifact is missing: {role}: {path}")
        _require(hashes[role] == sha256(path),
                 f"promoted artifact hash changed: {role}: {path}")
    return artifacts


def canonical_kspace_frame(final_envelope: dict) -> pd.DataFrame:
    frame = pd.read_csv(Path(
        final_envelope["artifacts"]["fig6_kspace_final_envelope"]
    ))
    frame = frame[frame.flavor.astype(str).isin(("u", "d"))].copy()
    frame = frame.rename(columns={
        "final_envelope_low": "q16",
        "trained_central": "central",
        "final_envelope_high": "q84",
    })
    required = ["flavor", "kT", "q16", "central", "q84"]
    _require(set(required).issubset(frame.columns),
             "final envelope lacks canonical k-space columns")
    frame = frame[required].reset_index(drop=True)
    _require(
        set(frame.flavor.astype(str)) == {"u", "d"}
        and frame.groupby("flavor").kT.nunique().to_dict()
            == {"d": 401, "u": 401},
        "canonical champion k-space band is incomplete",
    )
    return frame


def publish_registry_commit(
        *, record: dict, public_summary: dict, transaction_core: dict,
        champion_record: Path, current: Path = CURRENT,
        public_summary_path: Path = PUBLIC_SUMMARY,
        transaction_path: Path = PROMOTION_TRANSACTION,
        step_hook=None) -> dict:
    """Commit registry metadata with ``current.json`` as the final marker.

    Every individual write is atomic.  The optional hook exists only to test
    crashes at every boundary; rerunning this function with the same inputs is
    safe after any such interruption.
    """
    hook = (lambda _stage: None) if step_hook is None else step_hook
    atomic_write_json(champion_record, record)
    hook("champion_record")
    record_hash = sha256(champion_record)

    published_summary = {
        **public_summary,
        "champion_record": str(champion_record),
        "champion_record_sha256": record_hash,
        "current_registry": str(current),
        "expected_current_registry_sha256": record_hash,
    }
    atomic_write_json(public_summary_path, published_summary)
    hook("public_summary")
    public_summary_hash = sha256(public_summary_path)

    ready = {
        **transaction_core,
        "status": "ready_for_atomic_current_commit",
        "transaction_version": 1,
        "champion_record": str(champion_record),
        "champion_record_sha256": record_hash,
        "public_summary": str(public_summary_path),
        "public_summary_sha256": public_summary_hash,
        "current_registry": str(current),
        "expected_current_registry_sha256": record_hash,
        "production_sources_modified": False,
    }
    atomic_write_json(transaction_path, ready)
    hook("ready_transaction")

    # This is the only commit point.  There are no promotion mutations after
    # it; a crash before it leaves the incumbent current, and a crash after it
    # leaves a completely staged candidate current.
    atomic_write_json(current, record)
    hook("current_commit")
    return {
        "record": record,
        "record_sha256": record_hash,
        "public_summary": published_summary,
        "public_summary_sha256": public_summary_hash,
        "transaction": ready,
        "transaction_sha256": sha256(transaction_path),
    }


def validate_registry_commit(
        *, expected_champion_id: str, champion_record: Path,
        current: Path = CURRENT, public_summary_path: Path = PUBLIC_SUMMARY,
        transaction_path: Path = PROMOTION_TRANSACTION) -> dict:
    """Validate the registry/current/public-summary atomic commit graph."""
    record = load(champion_record)
    current_payload = load(current)
    public_summary = load(public_summary_path)
    transaction = load(transaction_path)
    record_hash = sha256(champion_record)
    public_hash = sha256(public_summary_path)
    _require(record.get("champion_id") == expected_champion_id,
             "promoted champion record has the wrong identity")
    _require(current_payload == record and sha256(current) == record_hash,
             "current registry is not the committed champion record")
    _require(
        public_summary.get("status") == "complete"
        and public_summary.get("champion_id") == expected_champion_id
        and Path(public_summary.get("champion_record", "")).resolve()
            == champion_record.resolve()
        and public_summary.get("champion_record_sha256") == record_hash
        and Path(public_summary.get("current_registry", "")).resolve()
            == current.resolve()
        and public_summary.get("expected_current_registry_sha256") == record_hash
        and public_summary.get("production_sources_modified") is False,
        "public champion summary is not bound to the registry commit",
    )
    _require(
        transaction.get("status") == "ready_for_atomic_current_commit"
        and int(transaction.get("transaction_version", -1)) == 1
        and transaction.get("champion_id") == expected_champion_id
        and Path(transaction.get("champion_record", "")).resolve()
            == champion_record.resolve()
        and transaction.get("champion_record_sha256") == record_hash
        and Path(transaction.get("public_summary", "")).resolve()
            == public_summary_path.resolve()
        and transaction.get("public_summary_sha256") == public_hash
        and Path(transaction.get("current_registry", "")).resolve()
            == current.resolve()
        and transaction.get("expected_current_registry_sha256") == record_hash
        and transaction.get("production_sources_modified") is False,
        "promotion transaction marker is incomplete or stale",
    )
    return {
        "status": "pass",
        "champion_id": expected_champion_id,
        "champion_record": str(champion_record),
        "champion_record_sha256": record_hash,
        "current_registry": str(current),
        "current_registry_sha256": sha256(current),
        "public_summary": str(public_summary_path),
        "public_summary_sha256": public_hash,
        "promotion_transaction": str(transaction_path),
        "promotion_transaction_sha256": sha256(transaction_path),
        "production_sources_modified": False,
    }


def validated_published_promotion(
        expected_champion_id: str = EXPECTED_CHAMPION_ID) -> dict:
    """Revalidate every byte reachable from the promoted registry state.

    This is deliberately callable after the promoter process exits.  The
    continuation embeds its returned hashes in the terminal decision, so a
    successful subprocess return code alone is never promotion evidence.
    """
    champion_record = REGISTRY / f"{expected_champion_id}.json"
    validation = validate_registry_commit(
        expected_champion_id=expected_champion_id,
        champion_record=champion_record,
    )
    record = load(champion_record)
    hashes = record.get("artifact_sha256")
    _require(
        record.get("status") ==
            "complete_audited_study_champion_not_frozen_production"
        and record.get("previous_champion_id") == INCUMBENT_ID
        and record.get("promotion_gate_pass") is True
        and record.get("terminal_evidence_candidate_endpoint_gate_pass") is True
        and int(record.get("start_count", -1)) == 24
        and int(record.get("experimental_replica_count", -1)) == 50
        and int(record.get("combined_member_count_per_flavor", -1)) == 1200
        and record.get("production_sources_modified") is False,
        "promoted champion record does not describe the complete passing gate",
    )
    artifacts = validated_artifact_bindings(
        record, required_keys=REQUIRED_PROMOTION_ARTIFACT_KEYS)

    expected_paths = {
        "kspace_combined_bands": REGISTERED_K,
        "completion_audit": COMPLETION,
        "state_chain_audit": STATE_CHAIN_AUDIT,
        "terminal_evidence_audit": TERMINAL_EVIDENCE,
        "postfit_tail_transform_audit": POSTFIT_TAIL_AUDIT,
        "nested_start_replica_interaction_audit": NESTED_INTERACTION,
        "final_directional_envelope_audit": FINAL_DIRECTIONAL_ENVELOPE,
        "lambda600_vs_lambda1_comparison": COMPARISON,
        **{role: target for role, (_, target) in PUBLIC_COPIES.items()},
    }
    for role, expected in expected_paths.items():
        _require(Path(artifacts[role]).resolve() == expected.resolve(),
                 f"promoted artifact role points elsewhere: {role}")

    final_envelope, final_hash = validated_final_directional_envelope(
        FINAL_DIRECTIONAL_ENVELOPE,
        NESTED_INTERACTION,
        POSTFIT_TAIL_AUDIT,
        STABILITY,
    )
    terminal = load(TERMINAL_EVIDENCE)
    completion = load(COMPLETION)
    _require(
        final_promotion_gate(final_envelope)
        and terminal.get("status") == "pass"
        and terminal.get("candidate_endpoint_gate_pass") is True
        and terminal.get("diagnostic_only") is False
        and completion.get("status") == "complete"
        and completion.get("terminal_evidence_gate_pass") is True
        and completion.get("final_directional_envelope_gate_pass") is True
        and record.get("final_directional_envelope_audit_sha256") == final_hash
        and hashes["final_directional_envelope_audit"] == final_hash
        and hashes["terminal_evidence_audit"] == sha256(TERMINAL_EVIDENCE)
        and hashes["completion_audit"] == sha256(COMPLETION),
        "promoted registry disagrees with the complete final/terminal gate",
    )

    expected_k = canonical_kspace_frame(final_envelope)
    observed_k = pd.read_csv(REGISTERED_K)
    try:
        pd.testing.assert_frame_equal(
            observed_k, expected_k, check_exact=True, check_dtype=False)
    except AssertionError as error:
        raise RuntimeError(
            "registered champion k-space CSV differs from the final envelope"
        ) from error
    for role, (source, target) in PUBLIC_COPIES.items():
        _require(sha256(source) == sha256(target) == hashes[role],
                 f"public promoted copy differs from validated source: {role}")

    public_summary = load(PUBLIC_SUMMARY)
    expected_public_artifacts = {
        "kspace_combined_bands": str(REGISTERED_K),
        **{role: str(target) for role, (_, target) in PUBLIC_COPIES.items()},
    }
    _require(
        public_summary.get("promotion_gate_pass") is True
        and public_summary.get("terminal_evidence_candidate_endpoint_gate_pass")
            is True
        and public_summary.get("artifacts") == expected_public_artifacts
        and public_summary.get("artifact_sha256") == {
            role: hashes[role] for role in expected_public_artifacts
        },
        "public champion summary does not bind every public artifact",
    )

    transaction = load(PROMOTION_TRANSACTION)
    _require(
        transaction.get("immutable_incumbent_id") == INCUMBENT_ID
        and transaction.get("completion_audit_sha256") == sha256(COMPLETION)
        and transaction.get("terminal_evidence_audit_sha256")
            == sha256(TERMINAL_EVIDENCE)
        and transaction.get("final_directional_envelope_sha256") == final_hash
        and transaction.get("public_artifact_sha256") == {
            role: hashes[role] for role in expected_public_artifacts
        },
        "promotion transaction is not bound to the audited/public artifacts",
    )
    return {
        **validation,
        "promotion_gate_pass": True,
        "terminal_evidence_candidate_endpoint_gate_pass": True,
        "artifact_sha256": dict(hashes),
    }


def main() -> None:
    subprocess.run([sys.executable, str(STATE_CHAIN_SCRIPT)],
                   check=True, stdout=subprocess.DEVNULL)
    incumbent = load(IMMUTABLE_INCUMBENT)
    completion = load(COMPLETION)
    stability = load(STABILITY)
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
    starts = load(STARTS)
    replicas = load(REPLICAS)
    state_chain = load(STATE_CHAIN_AUDIT)
    terminal_evidence = load(TERMINAL_EVIDENCE)
    if completion.get("status") != "complete":
        raise RuntimeError("campaign completion audit has not passed")
    strength = float(completion["selected_strength"])
    bmax = float(completion["selected_bmax"])
    barrier_strength = float(starts.get("fit_quality_barrier_strength", 0.0))
    barrier_power = int(starts.get("fit_quality_barrier_power", 2))
    barrier_id = (f"_fitbarp{barrier_power}mu{barrier_strength:g}"
                  if barrier_strength > 0 else "")
    champion_id = (f"empirical_reference_lambda{strength:g}{barrier_id}_"
                   f"b0p1_{bmax:g}_full24x50")
    if champion_id != EXPECTED_CHAMPION_ID:
        raise RuntimeError("promotion resolved a non-fixed challenger identity")
    chain_hash = sha256(STATE_CHAIN_AUDIT)
    chain_prescription = state_chain.get("selected_prescription", {})
    if not (state_chain.get("status") == "pass"
            and int(state_chain.get("lambda300_source_count", 0)) == 24
            and int(state_chain.get("start_chain_count", 0)) == 24
            and int(state_chain.get("central_chain_count", 0)) == 1
            and int(state_chain.get(
                "experimental_replica_chain_count", 0)) == 50
            and math.isclose(float(chain_prescription.get(
                "reference_strength", -1)), 600.0,
                rel_tol=0.0, abs_tol=1e-12)
            and math.isclose(float(chain_prescription.get(
                "fit_quality_barrier_strength", -1)), 100.0,
                rel_tol=0.0, abs_tol=1e-12)
            and int(chain_prescription.get(
                "fit_quality_barrier_power", -1)) == 2
            and state_chain.get("production_sources_modified") is False
            and completion.get("state_chain_gate_pass") is True
            and Path(completion.get("state_chain_audit", "")).resolve()
                == STATE_CHAIN_AUDIT.resolve()
            and completion.get("state_chain_audit_sha256") == chain_hash
            and completion.get("evidence_sha256", {}).get(
                str(STATE_CHAIN_AUDIT)) == chain_hash
            and stability.get("state_chain_gate_pass") is True
            and stability.get("state_chain_audit_sha256") == chain_hash):
        raise RuntimeError("lambda=600 state-chain provenance is not bound")
    if incumbent.get("champion_id") != INCUMBENT_ID:
        raise RuntimeError("immutable lambda1 incumbent record is invalid")
    # A retry may observe the incumbent (pre-commit), the exact candidate
    # (post-commit), or a torn legacy current only when our durable transaction
    # marker already proves this promotion had begun.  A valid third champion
    # is never overwritten.
    transaction_marker = None
    try:
        transaction_marker = load(PROMOTION_TRANSACTION)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        pass
    try:
        current = load(CURRENT)
    except (FileNotFoundError, OSError, json.JSONDecodeError) as error:
        if (not isinstance(transaction_marker, dict)
                or transaction_marker.get("champion_id") != champion_id
                or transaction_marker.get("immutable_incumbent_id")
                    != INCUMBENT_ID):
            raise RuntimeError(
                "current registry is unreadable without a matching promotion "
                "transaction marker"
            ) from error
        current = None
    if (current is not None and current.get("champion_id")
            not in {INCUMBENT_ID, champion_id}):
        raise RuntimeError("an unrelated current champion cannot be overwritten")
    for path_text, expected in completion.get("evidence_sha256", {}).items():
        path = Path(path_text)
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"audited evidence changed before promotion: {path}")
    if not completion.get("evidence_sha256"):
        raise RuntimeError("completion audit did not bind its evidence artifacts")
    for name, expected in completion.get("figure_sha256", {}).items():
        path = FIGURES / name
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"audited figure changed before promotion: {path}")
    if len(completion.get("figure_sha256", {})) != 3:
        raise RuntimeError("completion audit did not bind all three figures")
    endpoint_gate = final_promotion_gate(final_envelope)
    if not endpoint_gate:
        raise RuntimeError("candidate robustness gate has not passed")
    if not (
            terminal_evidence.get("status") == "pass"
            and terminal_evidence.get("candidate_endpoint_gate_pass") is True
            and terminal_evidence.get("diagnostic_only") is False
            and completion.get("terminal_evidence_gate_pass") is True
            and completion.get("final_directional_envelope_gate_pass") is True
            and completion.get("terminal_evidence_audit_sha256")
                == sha256(TERMINAL_EVIDENCE)
            and completion.get("final_directional_envelope_sha256")
                == final_envelope_hash):
        raise RuntimeError(
            "completion, terminal evidence, and final promotion gate disagree")
    widths = {key: float(final_envelope["width_metrics_by_flavor"][key][
        "joint_convergence_interaction_raw_full_width"])
              for key in ("u", "d")}
    previous = {key: float(stability[
        "comparison_champion_max_active_relative_full_width"][key])
        for key in ("u", "d")}
    union_diagnostic = {key: float(stability[
        "comparison_champion_union_mask_relative_full_width"][key])
        for key in ("u", "d")}
    if stability.get("comparison_champion_id") != incumbent.get("champion_id"):
        raise RuntimeError("candidate audit compared against a different incumbent")
    allowance = {key: float(final_envelope["width_metrics_by_flavor"][key][
        "corrected_finite_sampling_full_width_margin"])
        for key in ("u", "d")}
    for key in ("u", "d"):
        if not (math.isclose(previous[key], LOCKED_INCUMBENT_WIDTHS[key],
                             rel_tol=0.0, abs_tol=1e-14)
                and math.isclose(previous[key], float(incumbent[
                    "combined_fig6_max_active_relative_full_width"][key]),
                                 rel_tol=0.0, abs_tol=1e-14)
                and math.isfinite(union_diagnostic[key])
                and union_diagnostic[key] > 0.0
                and math.isclose(float(completion[
                    "final_fig6_max_active_relative_full_width"][key]),
                                 widths[key], rel_tol=1e-12, abs_tol=1e-14)
                and math.isclose(float(completion[
                    "resampling_full_width_allowance_by_flavor"][key]),
                                 allowance[key], rel_tol=1e-12, abs_tol=1e-14)
                and math.isclose(float(completion[
                    "incumbent_fig6_max_active_relative_full_width"][key]),
                                 previous[key], rel_tol=1e-12, abs_tol=1e-14)):
            raise RuntimeError(
                f"completion and stability summaries disagree for {key}")
        if not widths[key] + allowance[key] < previous[key]:
            raise RuntimeError(
                f"candidate does not robustly improve incumbent {key}")

    transaction_core = {
        "champion_id": champion_id,
        "immutable_incumbent_id": INCUMBENT_ID,
        "completion_audit": str(COMPLETION),
        "completion_audit_sha256": sha256(COMPLETION),
        "terminal_evidence_audit": str(TERMINAL_EVIDENCE),
        "terminal_evidence_audit_sha256": sha256(TERMINAL_EVIDENCE),
        "final_directional_envelope": str(FINAL_DIRECTIONAL_ENVELOPE),
        "final_directional_envelope_sha256": final_envelope_hash,
    }
    atomic_write_json(PROMOTION_TRANSACTION, {
        **transaction_core,
        "status": "preparing_atomic_promotion",
        "transaction_version": 1,
        "production_sources_modified": False,
    })
    PUBLIC.mkdir(parents=True, exist_ok=True)
    for source, target in PUBLIC_COPIES.values():
        if not source.is_file() or source.stat().st_size == 0:
            raise RuntimeError(f"missing validated figure: {source}")
        atomic_copy(source, target)

    # Register a stable combined-only schema so the next comparison can read
    # either this champion or the legacy lambda1 champion uniformly.
    kframe = canonical_kspace_frame(final_envelope)
    atomic_write_frame(REGISTERED_K, kframe)
    artifacts = {
        "combined_summary": str(SOURCE / "summary.json"),
        "fnp_combined_bands": str(Path(final_envelope["artifacts"][
            "fnp_final_envelope"])),
        "bspace_combined_bands": str(Path(final_envelope["artifacts"][
            "fig2_bspace_final_envelope"])),
        "kspace_combined_bands": str(REGISTERED_K),
        "completion_audit": str(COMPLETION),
        "state_chain_audit": str(STATE_CHAIN_AUDIT),
        "terminal_evidence_audit": str(TERMINAL_EVIDENCE),
        "postfit_tail_transform_audit": str(POSTFIT_TAIL_AUDIT),
        "nested_start_replica_interaction_audit": str(NESTED_INTERACTION),
        "final_directional_envelope_audit": str(FINAL_DIRECTIONAL_ENVELOPE),
        "final_fig6_component_width_curves": str(Path(final_envelope[
            "artifacts"]["fig6_component_width_curves"])),
        "lambda600_vs_lambda1_comparison": str(COMPARISON),
        **{key: str(target) for key, (_, target) in PUBLIC_COPIES.items()},
    }
    historical = incumbent["historical_baseline_fig6_max_active_relative_full_width"]
    record = {
        "status": "complete_audited_study_champion_not_frozen_production",
        "champion_id": champion_id,
        "previous_champion_id": incumbent["champion_id"],
        "method": (f"pointwise distance to the full 24-start empirical "
                   f"baseline-median FNP on 0.1<=bT<={bmax:g}, "
                   f"lambda={strength:g}; reciprocal even/odd cross-fitting "
                   f"was used for the preceding leakage/selection validation; "
                   f"one-sided power-{barrier_power} fit-quality barrier "
                   f"mu={barrier_strength:g}"),
        "fit_quality_barrier_strength": barrier_strength,
        "fit_quality_barrier_power": barrier_power,
        "start_count": int(starts["member_count"]),
        "experimental_replica_count": int(replicas["replica_count"]),
        "combined_member_count_per_flavor": 1200,
        "promotion_gate_pass": True,
        "terminal_evidence_candidate_endpoint_gate_pass": True,
        "postfit_tail_transform_audit_sha256": postfit_tail_hash,
        "nested_start_replica_interaction_audit_sha256":
            nested_interaction_hash,
        "final_directional_envelope_audit_sha256": final_envelope_hash,
        "central_line_definition": (
            "separately trained 300k terminal lambda600 central endpoint propagated "
            "through the exact paired expb2 finite-b transform"
        ),
        "displayed_band_definition": (
            "empirical product band plus residual convergence/interaction "
            "envelope; the compatibility columns named q16 and q84 are final "
            "directional bounds, not quantiles of a single probability law"
        ),
        "combined_fig6_max_active_relative_full_width": widths,
        "fig6_component_widths_by_flavor": final_envelope[
            "width_metrics_by_flavor"],
        "combined_fig6_own_active_relative_full_width": stability[
            "candidate_own_active_relative_full_width"],
        "comparison_active_mask_definition": stability[
            "comparison_active_mask_definition"],
        "previous_champion_fig6_max_active_relative_full_width": previous,
        "previous_champion_union_mask_relative_full_width_diagnostic":
            union_diagnostic,
        "historical_baseline_fig6_max_active_relative_full_width": historical,
        "relative_width_reduction_from_previous_champion": {
            key: 1.0 - widths[key] / previous[key] for key in widths},
        "relative_width_reduction_from_historical_baseline": {
            key: 1.0 - widths[key] / float(historical[key]) for key in widths},
        "resampling_sensitivity": {
            "bootstrap_p95_absolute_full_width_statistic_deviation": stability[
                "bootstrap_p95_absolute_full_width_statistic_deviation"],
            "start_split_half_p95_absolute_full_width_statistic_difference": stability[
                "start_split_half_p95_absolute_full_width_statistic_difference"],
            "replica_split_half_p95_absolute_full_width_statistic_difference": stability[
                "replica_split_half_p95_absolute_full_width_statistic_difference"],
            "joint_split_half_p95_absolute_full_width_statistic_difference": stability[
                "joint_split_half_p95_absolute_full_width_statistic_difference"],
            "direct_full_width_allowance": stability[
                "resampling_full_width_allowance"],
            "corrected_finite_sampling_full_width_margin_by_flavor":
                allowance,
            "semantics": (
                "the final joint directional width pays the corrected "
                "flavor-specific finite-sampling full-width margin"
            ),
        },
        "promotion_semantics": "the one fixed lambda600 challenger completed the exact 24-start + trained-300k-central + 50-replica comparison, post-fit tail/transform audit, and six full-horizon nested interaction fits; its jointly expanded raw u/d width is evaluated on the conservative union active mask, pays the corrected flavor-specific finite-sampling margin, and remains below the immutable registered lambda1 threshold separately for u and d; no other lambda or prior is selected from intermediate evidence",
        "interval": "empirical product band plus residual convergence/interaction envelope",
        "formal_confidence_level_assigned": False,
        "one_sigma_claimed": False,
        "incumbent_and_candidate_training_protocols_identical": False,
        "artifacts": artifacts,
        "artifact_sha256": {key: sha256(Path(value)) for key, value in artifacts.items()},
        "production_sources_modified": False,
    }
    public_artifacts = {
        "kspace_combined_bands": str(REGISTERED_K),
        **{role: str(target) for role, (_, target) in PUBLIC_COPIES.items()},
    }
    public_summary = {
        "status": "complete", "champion_id": champion_id,
        "source": str(FIGURES),
        "central_line": "trained 300k lambda600 central endpoint",
        "interval": (
            "empirical product band plus residual convergence/interaction "
            "envelope"
        ),
        "formal_confidence_level_assigned": False,
        "one_sigma_claimed": False,
        "promotion_gate_pass": True,
        "terminal_evidence_candidate_endpoint_gate_pass": True,
        "artifacts": public_artifacts,
        "artifact_sha256": {
            role: record["artifact_sha256"][role]
            for role in public_artifacts
        },
        "production_sources_modified": False,
    }
    transaction_core["public_artifact_sha256"] = {
        role: record["artifact_sha256"][role]
        for role in public_artifacts
    }
    champion_record = REGISTRY / f"{champion_id}.json"
    publish_registry_commit(
        record=record,
        public_summary=public_summary,
        transaction_core=transaction_core,
        champion_record=champion_record,
    )
    validation = validated_published_promotion(champion_id)
    print(json.dumps({
        "record": record,
        "post_promotion_validation": validation,
    }, indent=2))


if __name__ == "__main__":
    main()
