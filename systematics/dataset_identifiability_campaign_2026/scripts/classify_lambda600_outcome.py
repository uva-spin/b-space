#!/usr/bin/env python3
"""Classify the completed lambda600 challenge without selecting another prior.

The classifier makes the post-campaign decision failure-driven.  It separates
optimizer/FNP stationarity, fit preservation, finite-ensemble resolution, and
technical audit failures.  While the challenge is running it records only an
in-progress state. Even after the final comparison it may record a
failure-driven scientific option, but selection and launch require a separate
recorded decision.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from alternate_lambda_authorization import (
    validate_complete_lambda600_comparison,
)
from nested_interaction_and_final_envelope_validation import (
    final_promotion_gate,
)


BASE = Path(__file__).resolve().parents[1]
SUM = BASE / "summaries"
DECISION = SUM / "lambda600_like_for_like_decision/summary.json"
STARTS = SUM / "replica_robust_reference_full24/summary.json"
START_LEDGER = SUM / "replica_robust_reference_full24/runs.csv"
REPLICAS = SUM / "selected_reference_central_replicas/summary.json"
REPLICA_LEDGER = SUM / "selected_reference_central_replicas/runs.csv"
STABILITY = SUM / "final_combined_ensemble_stability/summary.json"
START_SOURCE_ENSEMBLE = SUM / "selected_reference_method_full24/summary.json"
TARGET = SUM / "lambda600_outcome_classification"
EXPECTED = {
    "reference_strength": 600.0,
    "reference_bmax": 4.0,
    "fit_quality_barrier_strength": 100.0,
    "fit_quality_barrier_power": 2,
}
LOCKED_INCUMBENT_WIDTHS = {
    "u": 0.11772613918747582,
    "d": 0.12490071924111977,
}
FINAL_SCIENTIFIC_STATUSES = {
    "candidate_promoted_as_new_study_champion",
    "candidate_rejected",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def wait_for_scientific_terminal_decision(
        *, loader=None, sleeper=time.sleep,
        poll_seconds: float = 30.0) -> dict:
    """Wait through recoverable technical states for the final decision.

    The continuation publishes ``technical_failure`` before exiting nonzero;
    systemd then restarts it.  Treating that intermediate manifest as terminal
    would permanently retire the classifier before the recovered controller
    can publish promotion or rejection.
    """
    if loader is None:
        loader = lambda: load(DECISION)
    while True:
        try:
            decision = loader()
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            sleeper(poll_seconds)
            continue
        if not isinstance(decision, dict):
            raise RuntimeError("lambda600 decision manifest is not an object")
        status = str(decision.get("status"))
        if status in FINAL_SCIENTIFIC_STATUSES:
            return decision
        if status not in {"in_progress", "technical_failure"}:
            raise RuntimeError(
                f"unexpected nonterminal lambda600 decision status: {status}")
        sleeper(poll_seconds)


def exact_float(observed, expected: float, label: str) -> None:
    if (observed is None or not np.isfinite(float(observed))
            or not np.isclose(float(observed), float(expected),
                              rtol=0.0, atol=1.0e-12)):
        raise RuntimeError(f"exact lambda600 value mismatch for {label}")


def load_objective_auditor():
    path = BASE / "scripts/audit_like_for_like_completion.py"
    spec = importlib.util.spec_from_file_location(
        "lambda600_classifier_objective_auditor", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    stat = Path(f"/proc/{pid}/stat")
    return not stat.exists() or stat.read_text().split()[2] != "Z"


def write(payload: dict) -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    payload = {
        **payload,
        "classifier_semantics": (
            "diagnose the completed lambda600 challenge; this classifier never "
            "selects or launches another constraint"
        ),
        "production_sources_modified": False,
    }
    path = TARGET / "summary.json"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps(payload, indent=2))


def exact_candidate(decision: dict) -> None:
    candidate = decision.get("candidate", {})
    for key, expected in EXPECTED.items():
        observed = candidate.get(key)
        if isinstance(expected, float):
            valid = (observed is not None and np.isfinite(float(observed))
                     and np.isclose(float(observed), expected,
                                    rtol=0.0, atol=1.0e-12))
        else:
            valid = observed == expected
        if not valid:
            raise RuntimeError(f"lambda600 decision candidate mismatch for {key}")


def exact_bool(value, label: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise RuntimeError(f"{label} is not an explicit boolean")


def bool_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        raise RuntimeError(f"ledger lacks boolean column {column}")
    return frame[column].map(lambda value: exact_bool(value, column))


def exact_prescription(summary: dict) -> None:
    observed = {
        "reference_strength": summary.get("selected_strength"),
        "reference_bmax": summary.get("selected_bmax"),
        "fit_quality_barrier_strength": summary.get(
            "fit_quality_barrier_strength"),
        "fit_quality_barrier_power": summary.get("fit_quality_barrier_power"),
    }
    for key, expected in EXPECTED.items():
        value = observed[key]
        exact_float(value, float(expected), f"terminal summary {key}")


def terminal_rows(ledger: pd.DataFrame, group: str) -> pd.DataFrame:
    if ledger.empty or group not in ledger.columns:
        raise RuntimeError("invalid terminal ledger")
    order = "cumulative_lbfgs_iterations"
    return (ledger.sort_values(order).groupby(group, as_index=False,
                                               dropna=False).tail(1))


def classify(*, upstream_ended: bool = False) -> dict:
    decision = load(DECISION)
    exact_candidate(decision)
    status = str(decision.get("status"))
    stage = str(decision.get("stage"))
    if status == "in_progress":
        if upstream_ended:
            return {
                "status": "complete",
                "lambda600_stage": stage,
                "classification": "pipeline_controller_terminated_without_terminal_manifest",
                "next_action": (
                    "inspect logs and resume or repair the interrupted lambda600 "
                    "stage without changing the prior"
                ),
                "another_constraint_selected": False,
            }
        return {
            "status": "in_progress",
            "lambda600_stage": stage,
            "classification": None,
            "next_action": "wait_for_complete_lambda600_evidence",
            "another_constraint_selected": False,
        }
    if status == "technical_failure":
        return {
            "status": "complete",
            "lambda600_stage": stage,
            "classification": "technical_pipeline_failure",
            "next_action": (
                "repair or resume the recorded lambda600 stage without changing "
                "the prior"
            ),
            "another_constraint_selected": False,
            "pipeline_failure": decision,
        }
    if status == "candidate_promoted_as_new_study_champion":
        # A status label alone is not promotion evidence.  Revalidate the
        # exact 24+1+50 artifact graph, immutable incumbent comparison, locked
        # protocol, and terminal hashes before reporting success.
        validated_decision, terminal_decision_sha256 = (
            validate_complete_lambda600_comparison())
        if validated_decision != decision:
            raise RuntimeError(
                "validated terminal decision differs from classifier input")
        return {
            "status": "complete",
            "lambda600_stage": stage,
            "classification": "validated_final_band_improvement",
            "next_action": (
                "retain promoted lambda600 as the new fixed-incumbent champion; "
                "it is a validated candidate, not yet a proof that lambda600 is "
                "the minimum sufficient constraint"
            ),
            "another_constraint_selected": False,
            "terminal_decision_sha256": terminal_decision_sha256,
            "promotion_gate_pass": True,
        }
    if status != "candidate_rejected":
        raise RuntimeError(f"unknown lambda600 terminal status: {status}")

    if stage == "full24_long_horizon":
        starts = decision.get("full24")
        if not isinstance(starts, dict) or starts.get("status") != "verification_failed":
            raise RuntimeError("full24 rejection lacks explicit verification_failed evidence")
        exact_prescription(starts)
        endpoint_tags = [str(value) for value in starts.get("endpoint_tags", [])]
        failed = [int(value) for value in starts.get("failed_seeds", [])]
        if (int(starts.get("member_count", -1)) != 24
                or len(endpoint_tags) != 24 or len(set(endpoint_tags)) != 24
                or not failed or not set(failed).issubset(set(range(303, 327)))
                or exact_bool(starts.get(
                    "all_starts_fnp_plateaued_and_fit_preserved"),
                    "all_starts_fnp_plateaued_and_fit_preserved")):
            raise RuntimeError("full24 scientific-failure manifest is incomplete")
        ledger = pd.read_csv(START_LEDGER)
        strength_values = pd.to_numeric(ledger["strength"], errors="coerce")
        ledger = ledger[np.isclose(strength_values, 600.0,
                                   rtol=0.0, atol=1.0e-12)]
        rows = terminal_rows(ledger, "seed")
        if set(rows["seed"].astype(int)) != set(range(303, 327)):
            raise RuntimeError("lambda600 start ledger lacks exact seed coverage")
        rows = rows.sort_values("seed")
        if rows["tag"].astype(str).tolist() != endpoint_tags:
            raise RuntimeError("lambda600 start ledger endpoints differ from manifest")
        objective_audit = load_objective_auditor()
        source_summary = load(START_SOURCE_ENSEMBLE)
        source_tags = [str(value) for value in source_summary.get(
            "endpoint_tags", [])]
        if (source_summary.get("status") != "complete"
                or len(source_tags) != 24 or len(set(source_tags)) != 24):
            raise RuntimeError("lambda300 start-source evidence is incomplete")
        for seed, tag, source_tag in zip(
                range(303, 327), endpoint_tags, source_tags, strict=True):
            source_status = load(BASE / "outputs" / source_tag / "fit_status.json")
            expected_ceiling = (
                float(source_status["final"]["unpenalized_total_chi2"])
                + math.sqrt(2.0 * int(source_status["row_count"])))
            objective_audit.verify_objective(
                tag, strength=600.0, bmax=4.0,
                barrier_strength=100.0, barrier_power=2,
                replica_seed=None, fit_seed=seed,
                expected_barrier_ceiling=expected_ceiling)
        failed_rows = rows[rows["seed"].astype(int).isin(failed)].copy()
        fit_pass = bool_column(failed_rows, "passes_natural_chi2_scale")
        fit_failures = failed_rows.loc[~fit_pass, "seed"].astype(int).tolist()
        stationarity_failures = [value for value in failed
                                 if value not in fit_failures]
        if fit_failures:
            classification = "start_fit_preservation_failure"
            next_action = (
                "diagnose_the_fit_distortion_while_finishing_the_fixed_lambda600_"
                "central50_combined_figures_and_final_lambda1_comparison"
            )
            selected = False
        else:
            classification = "residual_start_fnp_stationarity_failure"
            next_action = (
                "finish_the_lambda600_central50_combined_figures_and_final_"
                "lambda1_comparison_before_recording_any_next_trial"
            )
            selected = False
        return {
            "status": "complete",
            "lambda600_stage": stage,
            "classification": classification,
            "fit_failure_seeds": fit_failures,
            "stationarity_failure_seeds": stationarity_failures,
            "next_action": next_action,
            "another_constraint_selected": selected,
        }

    if stage == "central_plus_50_replicas":
        replicas = decision.get("replicas")
        if (not isinstance(replicas, dict)
                or replicas.get("status") not in {
                    "central_stationarity_failed", "replica_stationarity_failed"}):
            raise RuntimeError("replica rejection lacks an explicit scientific manifest")
        exact_prescription(replicas)
        central_failed = not exact_bool(
            replicas.get("central_fnp_plateau_pass"),
            "central_fnp_plateau_pass")
        replica_tags = [str(value) for value in replicas.get(
            "replica_endpoint_tags", [])]
        completed = int(replicas.get("completed_replica_count", len(replica_tags)))
        failed = [int(value) for value in replicas.get(
            "failed_replica_seeds", [])]
        if (completed != len(replica_tags) or len(replica_tags) != len(set(replica_tags))
                or completed > 50
                or (replicas.get("status") == "replica_stationarity_failed"
                    and not failed)):
            raise RuntimeError("replica scientific-failure manifest is incomplete")
        ledger = pd.read_csv(REPLICA_LEDGER)
        central_rows = ledger[ledger["kind"].astype(str).eq("central")]
        if central_rows.empty:
            raise RuntimeError("current replica ledger lacks the central trajectory")
        central_terminal = central_rows.sort_values(
            "cumulative_lbfgs_iterations").iloc[-1]
        if str(central_terminal["tag"]) != str(replicas.get("central_endpoint_tag")):
            raise RuntimeError("central ledger endpoint differs from manifest")
        if "fit_quality_gate_pass" not in ledger.columns:
            raise RuntimeError("current replica ledger lacks the generic fit gate")
        central_fit_pass = exact_bool(
            central_terminal["fit_quality_gate_pass"], "central fit gate")
        replica_rows = ledger[
            ledger["kind"].astype(str).eq("experimental_replica")]
        expected_seeds = list(range(1001, 1001 + completed))
        if completed:
            terminal_replica_rows = terminal_rows(replica_rows, "replica_seed")
            terminal_replica_rows = terminal_replica_rows.sort_values(
                "replica_seed")
        else:
            terminal_replica_rows = replica_rows.iloc[0:0].copy()
        observed = terminal_replica_rows[
            "replica_seed"].dropna().astype(int).tolist()
        observed_tags = terminal_replica_rows["tag"].astype(str).tolist()
        if observed != expected_seeds or observed_tags != replica_tags:
            raise RuntimeError("replica ledger endpoints differ from manifest")
        if not set(failed).issubset(set(expected_seeds)):
            raise RuntimeError("failed replica identity is outside completed coverage")
        objective_audit = load_objective_auditor()
        objective_audit.verify_objective(
            str(replicas["central_endpoint_tag"]), strength=600.0,
            bmax=4.0, barrier_strength=100.0, barrier_power=2,
            replica_seed=None,
            expected_barrier_ceiling=float(
                replicas["central_fit_quality_ceiling_total_chi2"]))
        for offset, (replica_seed, tag) in enumerate(
                zip(expected_seeds, replica_tags, strict=True)):
            objective_audit.verify_objective(
                tag, strength=600.0, bmax=4.0,
                barrier_strength=100.0, barrier_power=2,
                replica_seed=replica_seed, fit_seed=2001 + offset)
        failed_rows = terminal_replica_rows[
            terminal_replica_rows["replica_seed"].astype(int).isin(failed)]
        fit_pass = bool_column(failed_rows, "fit_quality_gate_pass")
        fit_failures = failed_rows.loc[
            ~fit_pass, "replica_seed"].astype(int).tolist()
        stationarity_failures = [value for value in failed
                                 if value not in fit_failures]
        if central_failed and not central_fit_pass:
            classification = "central_fit_preservation_failure"
            next_action = (
                "diagnose_the_central_fit_distortion_while_finishing_all_50_"
                "lambda600_replicas_and_the_final_comparison"
            )
            selected = False
        elif fit_failures:
            classification = "experimental_replica_fit_failure"
            next_action = (
                "diagnose_the_replica_fit_failure_while_finishing_all_50_"
                "lambda600_replicas_and_the_final_comparison"
            )
            selected = False
        elif central_failed or stationarity_failures:
            classification = (
                "central_fnp_stationarity_failure" if central_failed else
                "residual_replica_fnp_stationarity_or_cross_optimizer_failure"
            )
            next_action = (
                "finish_all_50_lambda600_replicas_and_the_combined_figures_and_"
                "final_lambda1_comparison_before_recording_any_next_trial"
            )
            selected = False
        else:
            raise RuntimeError("scientific replica rejection has no failed gate")
        return {
            "status": "complete",
            "lambda600_stage": stage,
            "classification": classification,
            "central_failed": central_failed,
            "central_fit_quality_gate_pass": central_fit_pass,
            "fit_failure_replica_seeds": fit_failures,
            "stationarity_failure_replica_seeds": stationarity_failures,
            "next_action": next_action,
            "another_constraint_selected": selected,
        }

    if stage == "protocol_native_completion_audit":
        return {
            "status": "complete",
            "lambda600_stage": stage,
            "classification": "technical_or_provenance_completion_audit_failure",
            "next_action": (
                "repair or explain the failed audit and rerun it without changing "
                "the fitted lambda600 ensemble"
            ),
            "another_constraint_selected": False,
        }

    if stage == "complete_like_for_like_comparison":
        stability = decision.get("stability")
        starts = decision.get("full24")
        replicas = decision.get("replicas")
        figures = decision.get("figures")
        comparison = decision.get("comparison")
        start_status = starts.get("status") if isinstance(starts, dict) else None
        replica_status = (replicas.get("status")
                          if isinstance(replicas, dict) else None)
        if (not isinstance(stability, dict) or stability.get("status") != "complete"
                or not isinstance(starts, dict)
                or start_status not in {"complete", "verification_failed"}
                or not isinstance(replicas, dict)
                or replica_status not in {
                    "complete", "complete_with_scientific_failures",
                    "central_stationarity_failed", "replica_stationarity_failed"}
                or not isinstance(figures, dict)
                or not figures.get("figure_2") or not figures.get("figure_6")
                or not isinstance(comparison, dict)
                or comparison.get("status") not in {
                    "complete_validated_candidate_comparison",
                    "complete_diagnostic_scientific_failure_comparison"}
                or comparison.get("comparison_champion_id") !=
                    "empirical_reference_lambda1_b0p1_2p0_full24"
                or stability.get("comparison_champion_id") !=
                    "empirical_reference_lambda1_b0p1_2p0_full24"):
            raise RuntimeError("final rejection lacks complete exact-provenance evidence")
        validated_decision, terminal_decision_sha256 = (
            validate_complete_lambda600_comparison())
        if validated_decision != decision:
            raise RuntimeError(
                "validated terminal rejection differs from classifier input")
        exact_prescription(starts)
        exact_prescription(replicas)
        start_tags = [str(value) for value in starts.get("endpoint_tags", [])]
        replica_tags = [str(value) for value in replicas.get(
            "replica_endpoint_tags", [])]
        sampled_coverage = (
            int(stability.get("start_count", -1)) == 24
            and int(stability.get("replica_count", -1)) == 50
            and int(starts.get("member_count", len(start_tags))) == 24
            and len(start_tags) == 24 and len(set(start_tags)) == 24
            and int(replicas.get("completed_replica_count",
                                 len(replica_tags))) == 50
            and len(replica_tags) == 50 and len(set(replica_tags)) == 50
        )
        coverage = exact_bool(stability.get("coverage_gate_pass"),
                              "coverage_gate_pass")
        start_gate = exact_bool(stability.get(
            "start_stationarity_and_fit_gate_pass"),
            "start_stationarity_and_fit_gate_pass")
        central_gate = exact_bool(replicas.get("central_fnp_plateau_pass"),
                                  "central_fnp_plateau_pass")
        replica_gate = exact_bool(stability.get(
            "replica_stationarity_and_agreement_gate_pass"),
            "replica_stationarity_and_agreement_gate_pass")
        candidate_stationarity = exact_bool(stability.get(
            "candidate_stationarity_gate_pass"),
            "candidate_stationarity_gate_pass")
        integrity = exact_bool(stability.get("band_integrity_gate_pass"),
                               "band_integrity_gate_pass")
        if candidate_stationarity != (start_gate and central_gate and replica_gate):
            raise RuntimeError("lambda600 stationarity gates are logically inconsistent")
        if not sampled_coverage or not coverage or not integrity:
            # A published scientific terminal decision is required to carry
            # exact sampled coverage and band integrity.  Treat any contrary
            # state as a retryable validation failure, never as a successful
            # classifier terminal that could skip the broader-goal handoff.
            raise RuntimeError(
                "terminal lambda600 decision lacks exact sample coverage or "
                "band integrity")
        # The terminal decision is made from the five-component final
        # directional envelope, not the earlier product-median stability
        # endpoint.  A post-fit, nested-interaction, containment, or final
        # joint-width failure can therefore reject a candidate even when the
        # base stability endpoint is true.  The exact terminal graph was
        # revalidated above; classify that graph and the same width statistic
        # rather than imposing the obsolete condition that the base endpoint
        # must also be false.
        final_envelope = decision.get("final_directional_envelope")
        if not isinstance(final_envelope, dict):
            raise RuntimeError(
                "terminal decision lacks the final directional envelope")
        component_gates = {
            "base_product_stability": exact_bool(final_envelope.get(
                "base_product_stability_gate_pass"),
                "base_product_stability_gate_pass"),
            "postfit_tail_transform": exact_bool(final_envelope.get(
                "postfit_tail_convergence_gate_pass"),
                "postfit_tail_convergence_gate_pass"),
            "nested_interaction": exact_bool(final_envelope.get(
                "nested_interaction_validation_gate_pass"),
                "nested_interaction_validation_gate_pass"),
            "joint_width_replacement": exact_bool(final_envelope.get(
                "joint_width_replacement_gate_pass"),
                "joint_width_replacement_gate_pass"),
            "trained_central_containment": exact_bool(final_envelope.get(
                "trained_central_containment_gate_pass"),
                "trained_central_containment_gate_pass"),
        }
        terminal_gate = final_promotion_gate(final_envelope)
        if terminal_gate != all(component_gates.values()) or terminal_gate:
            raise RuntimeError(
                "rejected final directional-envelope gates are inconsistent")
        expected_base_gate = bool(
            exact_bool(stability.get("diagnostic_figure_gate_pass"),
                       "diagnostic_figure_gate_pass")
            and candidate_stationarity
        )
        if component_gates["base_product_stability"] != expected_base_gate:
            raise RuntimeError(
                "final directional envelope disagrees with base stability")

        final_metrics = final_envelope.get("width_metrics_by_flavor")
        if not isinstance(final_metrics, dict) or set(final_metrics) != {"u", "d"}:
            raise RuntimeError(
                "final directional envelope lacks exact u/d width metrics")
        widths = {
            key: float(final_metrics[key][
                "joint_convergence_interaction_raw_full_width"])
            for key in ("u", "d")
        }
        allowance = {
            key: float(final_metrics[key][
                "final_statistic_finite_sampling_full_width_margin"])
            for key in ("u", "d")
        }
        incumbent = {
            key: float(final_metrics[key]["immutable_lambda1_width"])
            for key in ("u", "d")
        }
        for key in ("u", "d"):
            exact_float(incumbent[key], LOCKED_INCUMBENT_WIDTHS[key],
                        f"immutable lambda1 {key} width")
        if (not all(np.isfinite(list(widths.values()) + list(allowance.values())
                                + list(incumbent.values())))
                or any(value < 0 for value in widths.values())
                or any(value < 0 for value in allowance.values())
                or any(value <= 0 for value in incumbent.values())):
            raise RuntimeError("final comparison contains invalid width metrics")
        raw = {key: widths[key] < incumbent[key] for key in ("u", "d")}
        robust = {key: widths[key] + allowance[key] < incumbent[key]
                  for key in ("u", "d")}
        declared = {
            key: exact_bool(final_metrics[key].get("replacement_gate_pass"),
                            f"final declared robust {key}")
            for key in ("u", "d")
        }
        if (declared != robust
                or component_gates["joint_width_replacement"]
                    != all(robust.values())):
            raise RuntimeError(
                "rejected final joint-width comparison is inconsistent")
        scientific_next_option = None
        if not component_gates["base_product_stability"]:
            classification = "complete_comparison_with_lambda600_stationarity_failure"
            next_action = (
                "analyze_the_exact_start_central_and_replica_fnp_failure_mode_"
                "before_designing_one_namespaced_controlled_next_trial"
            )
        elif not component_gates["postfit_tail_transform"]:
            classification = "postfit_tail_or_transform_validation_failure"
            next_action = (
                "analyze_the_full_bT_tail_checkpoint_and_transform_mode_failure_"
                "before_designing_one_namespaced_controlled_next_trial"
            )
        elif not component_gates["nested_interaction"]:
            classification = "nested_start_replica_interaction_failure"
            next_action = (
                "analyze_or_expand_the_same_prior_nested_interaction_evidence_"
                "before_changing_the_constraint"
            )
        elif not component_gates["trained_central_containment"]:
            classification = "trained_central_containment_failure"
            next_action = (
                "diagnose_the_trained_central_and_directional_envelope_"
                "inconsistency_before_selecting_any_new_constraint"
            )
        elif all(raw.values()) and not all(robust.values()):
            classification = "finite_ensemble_resolution_failure_only"
            next_action = (
                "extend lambda600 sampling or endpoint confirmation; do not change "
                "the prior merely to reduce resampling allowance"
            )
        else:
            classification = "final_joint_directional_band_not_better_than_incumbent"
            next_action = (
                "decompose_the_final_convergence_interaction_and_sampling_width_"
                "before_designing_one_namespaced_controlled_next_trial"
            )
        selected = False
        return {
            "status": "complete",
            "lambda600_stage": stage,
            "classification": classification,
            "candidate_width": widths,
            "resampling_allowance": allowance,
            "incumbent_width": incumbent,
            "raw_improvement_by_flavor": raw,
            "robust_improvement_by_flavor": robust,
            "sample_coverage_gate_pass": coverage,
            "start_stationarity_and_fit_gate_pass": start_gate,
            "central_stationarity_gate_pass": central_gate,
            "replica_stationarity_and_agreement_gate_pass": replica_gate,
            "candidate_stationarity_gate_pass": candidate_stationarity,
            "authoritative_final_component_gates": component_gates,
            "authoritative_width_source": "final_directional_envelope",
            "scientific_next_option": scientific_next_option,
            "next_action": next_action,
            "another_constraint_selected": selected,
            "terminal_decision_sha256": terminal_decision_sha256,
            "promotion_gate_pass": False,
        }

    raise RuntimeError(f"unclassified lambda600 rejection stage: {stage}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int)
    parser.add_argument(
        "--wait-summary", action="store_true",
        help=("wait for the like-for-like decision manifest to become terminal; "
              "this survives a bounded systemd restart of its controller"))
    args = parser.parse_args()
    if args.wait_pid is not None and args.wait_summary:
        raise ValueError("choose only one classifier wait mechanism")
    if args.wait_pid is not None:
        write({
            "status": "in_progress",
            "lambda600_stage": "waiting_for_like_for_like_controller",
            "classification": None,
            "next_action": "wait_for_complete_lambda600_evidence",
            "another_constraint_selected": False,
        })
        while running(args.wait_pid):
            time.sleep(30)
    elif args.wait_summary:
        write({
            "status": "in_progress",
            "lambda600_stage": "waiting_for_like_for_like_manifest",
            "classification": None,
            "next_action": "wait_for_complete_lambda600_evidence",
            "another_constraint_selected": False,
        })
        wait_for_scientific_terminal_decision()
    try:
        result = classify(upstream_ended=(
            args.wait_pid is not None or args.wait_summary))
    except Exception as error:
        write({
            "status": "complete",
            "lambda600_stage": "classifier_validation",
            "classification": "technical_classifier_validation_failure",
            "validation_error_type": type(error).__name__,
            "validation_error": str(error),
            "next_action": (
                "repair the evidence/provenance inconsistency and rerun the "
                "classifier without changing the lambda600 prior"
            ),
            "another_constraint_selected": False,
        })
        raise
    write(result)
    # Promotion/rejection completes only the fixed challenger.  Immediately
    # publish the separately validated goal-level handoff so unattended
    # automation cannot mistake that one verdict for completion of the wider
    # minimum-constraint study.  The dispatcher is status-only: it neither
    # selects nor launches another prior.
    terminal_comparison = (
        result.get("status") == "complete"
        and result.get("lambda600_stage")
            == "complete_like_for_like_comparison"
    )
    if not terminal_comparison:
        # Intermediate scientific failures and technical/controller failures
        # are diagnostics, not a completed incumbent-replacement test.  Keep
        # their evidence on disk, but force the service to retry instead of
        # silently retiring before the locked 24-start + central + 50-replica
        # comparison and its broader-goal handoff exist.
        raise RuntimeError(
            "lambda600 classifier cannot exit successfully before the exact "
            "complete like-for-like comparison and broader-goal handoff"
        )
    if terminal_comparison and not result.get("terminal_decision_sha256"):
        raise RuntimeError(
            "terminal lambda600 classification lacks a decision hash and "
            "cannot hand off the broader goal")
    from post_lambda600_goal_dispatch import dispatch
    dispatch()


if __name__ == "__main__":
    main()
