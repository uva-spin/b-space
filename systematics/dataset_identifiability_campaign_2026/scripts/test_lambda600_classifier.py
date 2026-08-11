#!/usr/bin/env python3
"""Focused fail-closed tests for terminal lambda600 success classification."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def load_module():
    path = HERE / "classify_lambda600_outcome.py"
    spec = importlib.util.spec_from_file_location(
        "tested_lambda600_outcome_classifier", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


classifier = load_module()
import post_lambda600_goal_dispatch as goal_dispatch


def promoted_decision() -> dict:
    return {
        "status": "candidate_promoted_as_new_study_champion",
        "stage": "complete_like_for_like_comparison",
        "candidate": {
            "reference_strength": 600.0,
            "reference_bmax": 4.0,
            "fit_quality_barrier_strength": 100.0,
            "fit_quality_barrier_power": 2,
        },
    }


def rejected_decision(failed_gate: str) -> dict:
    """Build a terminal rejection whose legacy base endpoint still passes."""
    locked = dict(classifier.LOCKED_INCUMBENT_WIDTHS)
    metrics = {
        "u": {
            "joint_convergence_interaction_raw_full_width": 0.080,
            "final_statistic_finite_sampling_full_width_margin": 0.005,
            "immutable_lambda1_width": locked["u"],
            "replacement_gate_pass": True,
        },
        "d": {
            "joint_convergence_interaction_raw_full_width": 0.090,
            "final_statistic_finite_sampling_full_width_margin": 0.005,
            "immutable_lambda1_width": locked["d"],
            "replacement_gate_pass": True,
        },
    }
    gates = {
        "base_product_stability_gate_pass": True,
        "postfit_tail_convergence_gate_pass": True,
        "nested_interaction_validation_gate_pass": True,
        "joint_width_replacement_gate_pass": True,
        "trained_central_containment_gate_pass": True,
    }
    gate_keys = {
        "postfit": "postfit_tail_convergence_gate_pass",
        "nested": "nested_interaction_validation_gate_pass",
        "containment": "trained_central_containment_gate_pass",
        "joint_width": "joint_width_replacement_gate_pass",
    }
    gates[gate_keys[failed_gate]] = False
    if failed_gate == "joint_width":
        metrics["d"]["joint_convergence_interaction_raw_full_width"] = 0.130
        metrics["d"]["replacement_gate_pass"] = False
    final_envelope = {
        **gates,
        "promotion_validation_gate_pass": False,
        "width_metrics_by_flavor": metrics,
    }
    return {
        "status": "candidate_rejected",
        "stage": "complete_like_for_like_comparison",
        "candidate": {
            "reference_strength": 600.0,
            "reference_bmax": 4.0,
            "fit_quality_barrier_strength": 100.0,
            "fit_quality_barrier_power": 2,
        },
        "full24": {
            "status": "complete",
            "selected_strength": 600.0,
            "selected_bmax": 4.0,
            "fit_quality_barrier_strength": 100.0,
            "fit_quality_barrier_power": 2,
            "member_count": 24,
            "endpoint_tags": [f"start_{seed}" for seed in range(303, 327)],
        },
        "replicas": {
            "status": "complete",
            "selected_strength": 600.0,
            "selected_bmax": 4.0,
            "fit_quality_barrier_strength": 100.0,
            "fit_quality_barrier_power": 2,
            "completed_replica_count": 50,
            "replica_endpoint_tags": [
                f"replica_{seed}" for seed in range(1001, 1051)
            ],
            "central_fnp_plateau_pass": True,
        },
        "stability": {
            "status": "complete",
            "start_count": 24,
            "replica_count": 50,
            "comparison_champion_id":
                "empirical_reference_lambda1_b0p1_2p0_full24",
            "coverage_gate_pass": True,
            "band_integrity_gate_pass": True,
            "diagnostic_figure_gate_pass": True,
            "start_stationarity_and_fit_gate_pass": True,
            "replica_stationarity_and_agreement_gate_pass": True,
            "candidate_stationarity_gate_pass": True,
            # This is deliberately true.  Final-only gates are allowed to
            # reject the complete challenger after this earlier gate passes.
            "endpoint_gate_pass": True,
        },
        "figures": {"figure_2": "fig2.pdf", "figure_6": "fig6.pdf"},
        "comparison": {
            "status": "complete_diagnostic_scientific_failure_comparison",
            "comparison_champion_id":
                "empirical_reference_lambda1_b0p1_2p0_full24",
        },
        "final_directional_envelope": final_envelope,
        "promotion_gate_pass": False,
    }


class PromotedClassificationTests(unittest.TestCase):
    def run_with_decision(self, decision: dict, validator):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decision.json"
            path.write_text(json.dumps(decision) + "\n")
            with mock.patch.object(classifier, "DECISION", path), \
                    mock.patch.object(
                        classifier,
                        "validate_complete_lambda600_comparison",
                        side_effect=validator if isinstance(validator, Exception) else None,
                        return_value=(validator if not isinstance(
                            validator, Exception) else None),
                    ) as check:
                result = classifier.classify()
        return result, check

    def test_promoted_status_requires_and_records_complete_validation(self) -> None:
        decision = promoted_decision()
        result, check = self.run_with_decision(
            decision, (decision, "terminal-decision-sha256"))
        check.assert_called_once_with()
        self.assertEqual(result["classification"],
                         "validated_final_band_improvement")
        self.assertEqual(result["terminal_decision_sha256"],
                         "terminal-decision-sha256")

    def test_tampered_terminal_graph_cannot_be_reported_as_promoted(self) -> None:
        decision = promoted_decision()
        with self.assertRaisesRegex(RuntimeError, "terminal artifact tamper"):
            self.run_with_decision(
                decision, RuntimeError("terminal artifact tamper"))


class FinalEnvelopeRejectionClassificationTests(unittest.TestCase):
    def classify(self, failed_gate: str) -> dict:
        decision = rejected_decision(failed_gate)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decision.json"
            path.write_text(json.dumps(decision) + "\n")
            with mock.patch.object(classifier, "DECISION", path), \
                    mock.patch.object(
                        classifier,
                        "validate_complete_lambda600_comparison",
                        return_value=(decision, "terminal-decision-sha256"),
                    ):
                result = classifier.classify()
        self.assertTrue(decision["stability"]["endpoint_gate_pass"])
        self.assertEqual(result["authoritative_width_source"],
                         "final_directional_envelope")
        self.assertFalse(result["promotion_gate_pass"])
        self.assertFalse(result["another_constraint_selected"])
        self.assertEqual(result["terminal_decision_sha256"],
                         "terminal-decision-sha256")
        return result

    def test_postfit_only_failure_is_scientifically_classified(self) -> None:
        result = self.classify("postfit")
        self.assertEqual(
            result["classification"],
            "postfit_tail_or_transform_validation_failure",
        )

    def test_nested_only_failure_is_scientifically_classified(self) -> None:
        result = self.classify("nested")
        self.assertEqual(
            result["classification"],
            "nested_start_replica_interaction_failure",
        )

    def test_containment_only_failure_is_scientifically_classified(self) -> None:
        result = self.classify("containment")
        self.assertEqual(
            result["classification"],
            "trained_central_containment_failure",
        )

    def test_final_joint_width_failure_uses_final_not_base_width(self) -> None:
        result = self.classify("joint_width")
        self.assertEqual(
            result["classification"],
            "final_joint_directional_band_not_better_than_incumbent",
        )
        self.assertFalse(result["robust_improvement_by_flavor"]["d"])

    def test_terminal_coverage_contradiction_is_retryable_error(self) -> None:
        decision = rejected_decision("postfit")
        decision["stability"]["coverage_gate_pass"] = False
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decision.json"
            path.write_text(json.dumps(decision) + "\n")
            with mock.patch.object(classifier, "DECISION", path), \
                    mock.patch.object(
                        classifier,
                        "validate_complete_lambda600_comparison",
                        return_value=(decision, "terminal-decision-sha256"),
                    ):
                with self.assertRaisesRegex(
                        RuntimeError, "sample coverage or band integrity"):
                    classifier.classify()


class GoalHandoffPublicationTests(unittest.TestCase):
    def terminal_result(self, *, include_hash: bool = True) -> dict:
        result = {
            "status": "complete",
            "lambda600_stage": "complete_like_for_like_comparison",
            "classification": "nested_start_replica_interaction_failure",
            "promotion_gate_pass": False,
            "another_constraint_selected": False,
        }
        if include_hash:
            result["terminal_decision_sha256"] = "terminal-decision-sha256"
        return result

    def test_terminal_result_always_invokes_goal_dispatcher(self) -> None:
        result = self.terminal_result()
        with mock.patch.object(sys, "argv", ["classify_lambda600_outcome.py"]), \
                mock.patch.object(classifier, "classify", return_value=result), \
                mock.patch.object(classifier, "write") as writer, \
                mock.patch.object(goal_dispatch, "dispatch") as dispatch:
            classifier.main()
        writer.assert_called_once_with(result)
        dispatch.assert_called_once_with()

    def test_terminal_result_without_hash_cannot_exit_successfully(self) -> None:
        result = self.terminal_result(include_hash=False)
        with mock.patch.object(sys, "argv", ["classify_lambda600_outcome.py"]), \
                mock.patch.object(classifier, "classify", return_value=result), \
                mock.patch.object(classifier, "write"), \
                mock.patch.object(goal_dispatch, "dispatch") as dispatch:
            with self.assertRaisesRegex(RuntimeError, "lacks a decision hash"):
                classifier.main()
        dispatch.assert_not_called()

    def test_technical_failure_diagnostic_cannot_exit_successfully(self) -> None:
        result = {
            "status": "complete",
            "lambda600_stage": "classifier_validation",
            "classification": "technical_classifier_validation_failure",
            "another_constraint_selected": False,
        }
        with mock.patch.object(sys, "argv", ["classify_lambda600_outcome.py"]), \
                mock.patch.object(classifier, "classify", return_value=result), \
                mock.patch.object(classifier, "write") as writer, \
                mock.patch.object(goal_dispatch, "dispatch") as dispatch:
            with self.assertRaisesRegex(
                    RuntimeError, "cannot exit successfully before"):
                classifier.main()
        writer.assert_called_once_with(result)
        dispatch.assert_not_called()

    def test_early_scientific_failure_cannot_replace_full_comparison(self) -> None:
        result = {
            "status": "complete",
            "lambda600_stage": "full24_long_horizon",
            "classification": "residual_start_fnp_stationarity_failure",
            "another_constraint_selected": False,
        }
        with mock.patch.object(sys, "argv", ["classify_lambda600_outcome.py"]), \
                mock.patch.object(classifier, "classify", return_value=result), \
                mock.patch.object(classifier, "write"), \
                mock.patch.object(goal_dispatch, "dispatch") as dispatch:
            with self.assertRaisesRegex(
                    RuntimeError, "exact complete like-for-like comparison"):
                classifier.main()
        dispatch.assert_not_called()

    def test_dispatch_failure_is_nonzero_retry_outcome(self) -> None:
        result = self.terminal_result()
        with mock.patch.object(sys, "argv", ["classify_lambda600_outcome.py"]), \
                mock.patch.object(classifier, "classify", return_value=result), \
                mock.patch.object(classifier, "write"), \
                mock.patch.object(
                    goal_dispatch, "dispatch",
                    side_effect=RuntimeError("handoff validation failed")):
            with self.assertRaisesRegex(RuntimeError, "handoff validation failed"):
                classifier.main()


if __name__ == "__main__":
    unittest.main()
