#!/usr/bin/env python3
"""Tests for the post-lambda600 broader-goal handoff contract."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def load_module():
    path = HERE / "post_lambda600_goal_dispatch.py"
    spec = importlib.util.spec_from_file_location(
        "tested_post_lambda600_goal_dispatch", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


dispatcher = load_module()


def decision(outcome: str) -> dict:
    promoted = outcome == dispatcher.PROMOTED
    return {
        "status": outcome,
        "final_directional_envelope": {
            "promotion_validation_gate_pass": promoted,
        },
    }


def classification(outcome: str, digest: str, *, selected: bool = False,
                   name: str | None = None) -> dict:
    promoted = outcome == dispatcher.PROMOTED
    default = (
        dispatcher.PROMOTED_CLASSIFICATION if promoted else
        "nested_start_replica_interaction_failure"
    )
    return {
        "status": "complete",
        "lambda600_stage": "complete_like_for_like_comparison",
        "classification": default if name is None else name,
        "terminal_decision_sha256": digest,
        "promotion_gate_pass": promoted,
        "another_constraint_selected": selected,
        "production_sources_modified": False,
    }


class PostLambda600GoalDispatchTests(unittest.TestCase):
    def run_dispatch(self, outcome: str, *, classifier_overrides=None) -> dict:
        terminal_hash = "terminal-decision-sha256"
        terminal = decision(outcome)
        classifier = classification(outcome, terminal_hash)
        if classifier_overrides:
            classifier.update(classifier_overrides)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            classifier_path = root / "classification.json"
            target = root / "dispatch"
            classifier_path.write_text(json.dumps(classifier) + "\n")
            result = dispatcher.dispatch(
                classification_path=classifier_path,
                target=target,
                decision_validator=lambda: (terminal, terminal_hash),
                emit=False,
            )
            persisted = json.loads((target / "summary.json").read_text())
        self.assertEqual(result, persisted)
        return result

    def test_promoted_challenger_does_not_complete_minimum_study(self) -> None:
        result = self.run_dispatch(dispatcher.PROMOTED)
        self.assertTrue(result["fixed_lambda600_challenger_complete"])
        self.assertFalse(result["broader_study_complete"])
        self.assertFalse(result["minimum_nonphysics_constraint_established"])
        self.assertFalse(result["another_constraint_selected"])
        self.assertFalse(result["next_trial_launch_authorized"])
        self.assertEqual(
            result["next_stage"],
            "minimum_constraint_necessity_and_ablation_design",
        )

    def test_rejection_dispatches_to_failure_mode_analysis_only(self) -> None:
        result = self.run_dispatch(dispatcher.REJECTED)
        self.assertEqual(
            result["next_stage"],
            "failure_mode_driven_interaction_analysis",
        )
        self.assertFalse(result["dispatcher_launches_processes"])
        self.assertFalse(result["authorization_record_created"])

    def test_all_current_legacy_launchers_are_explicitly_prohibited(self) -> None:
        result = self.run_dispatch(dispatcher.REJECTED)
        self.assertEqual(
            result["legacy_generic_path_launchers_prohibited"],
            sorted(dispatcher.LEGACY_GENERIC_PATH_LAUNCHERS),
        )
        self.assertTrue(result["legacy_generic_path_launchers_prohibited"])
        self.assertTrue(any(
            "disjoint namespaced" in item
            for item in result["required_before_any_next_trial"]
        ))

    def test_stale_classifier_decision_hash_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "stale or not bound"):
            self.run_dispatch(
                dispatcher.REJECTED,
                classifier_overrides={
                    "terminal_decision_sha256": "stale-decision-sha256"},
            )

    def test_classifier_cannot_preselect_another_constraint(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "stale or not bound"):
            self.run_dispatch(
                dispatcher.REJECTED,
                classifier_overrides={"another_constraint_selected": True},
            )

    def test_unregistered_rejection_classification_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "stale or not bound"):
            self.run_dispatch(
                dispatcher.REJECTED,
                classifier_overrides={"classification": "arbitrary_lambda_ladder"},
            )


if __name__ == "__main__":
    unittest.main()
