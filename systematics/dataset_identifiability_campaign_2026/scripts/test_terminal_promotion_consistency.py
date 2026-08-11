#!/usr/bin/env python3
"""Focused tests for one terminal promotion/rejection truth value."""

from __future__ import annotations

from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import alternate_lambda_authorization as authorization


def decision(status: str, gate: bool) -> dict:
    return {"status": status, "promotion_gate_pass": gate}


def envelope(gate: bool) -> dict:
    return {"promotion_validation_gate_pass": gate}


def evidence(gate: bool) -> dict:
    return {
        "candidate_endpoint_gate_pass": gate,
        "diagnostic_only": not gate,
    }


class TerminalPromotionConsistencyTests(unittest.TestCase):
    def test_every_legacy_generic_launcher_is_permanently_blocked(self) -> None:
        self.assertEqual(
            set(authorization.LEGACY_GENERIC_PATH_LAUNCHERS),
            set(authorization.KNOWN_LAUNCHERS))
        for launcher in authorization.LEGACY_GENERIC_PATH_LAUNCHERS:
            with self.subTest(launcher=launcher):
                result = authorization.authorization_status(launcher)
                self.assertFalse(result["authorized"])
                self.assertTrue(result["legacy_generic_path_launcher"])
                with self.assertRaisesRegex(RuntimeError, "legacy generic-path"):
                    authorization.authorization_template(launcher)

    def test_promoted_requires_all_three_gates_true(self) -> None:
        self.assertTrue(authorization.require_terminal_gate_consistency(
            decision("candidate_promoted_as_new_study_champion", True),
            envelope(True), evidence(True)))

    def test_rejected_requires_all_three_gates_false(self) -> None:
        self.assertFalse(authorization.require_terminal_gate_consistency(
            decision("candidate_rejected", False),
            envelope(False), evidence(False)))

    def test_every_mixed_or_opposite_terminal_state_fails_closed(self) -> None:
        cases = (
            ("candidate_promoted_as_new_study_champion", False, True, True),
            ("candidate_promoted_as_new_study_champion", True, False, True),
            ("candidate_promoted_as_new_study_champion", True, True, False),
            ("candidate_rejected", True, False, False),
            ("candidate_rejected", False, True, False),
            ("candidate_rejected", False, False, True),
        )
        for status, decision_gate, final_gate, evidence_gate in cases:
            with self.subTest(
                    status=status, decision_gate=decision_gate,
                    final_gate=final_gate, evidence_gate=evidence_gate):
                with self.assertRaisesRegex(RuntimeError, "disagree"):
                    authorization.require_terminal_gate_consistency(
                        decision(status, decision_gate),
                        envelope(final_gate), evidence(evidence_gate))

    def test_diagnostic_flag_must_be_exact_complement(self) -> None:
        terminal = evidence(True)
        terminal["diagnostic_only"] = True
        with self.assertRaisesRegex(RuntimeError, "diagnostic flag"):
            authorization.require_terminal_gate_consistency(
                decision("candidate_promoted_as_new_study_champion", True),
                envelope(True), terminal)

    def test_prepublication_report_contract_for_both_outcomes(self) -> None:
        for outcome, suffix in (
                ("candidate_promoted_as_new_study_champion", "promoted"),
                ("candidate_rejected", "rejected")):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                upstream = root / "upstream.json"
                completion = root / "completion.json"
                report = root / "FINAL_REPORT.md"
                summary_path = root / "summary.json"
                seal_root = root / "current_byte_seals"
                seal_root.mkdir()
                seal = seal_root / "generation.json"
                seal.write_text("seal\n")
                upstream.write_text("upstream\n")
                completion.write_text("completion\n")
                report.write_text("report\n")
                inputs = {str(upstream): authorization.sha256(upstream)}
                inputs[str(seal)] = authorization.sha256(seal)
                if suffix == "promoted":
                    inputs[str(completion)] = authorization.sha256(completion)
                summary = {
                    "status": f"complete_fixed_challenger_report_{suffix}",
                    "outcome": outcome,
                    "decision_validation_mode":
                        "prepublication_complete_graph_revalidated",
                    "terminal_decision_sha256": None,
                    "report": str(report),
                    "artifact_sha256": {
                        "final_report": authorization.sha256(report)},
                    "input_sha256": inputs,
                    "start_checkpoint_ancestry": {
                        "seal": str(seal),
                        "seal_sha256": authorization.sha256(seal),
                        "legacy_pre_receipt_checkpoint_count": 12,
                        "historical_ancestry_limitation": "disclosed legacy limit",
                    },
                    "registry_modified_by_report_writer": False,
                    "frozen_sources_modified": False,
                    "production_sources_modified": False,
                }
                summary_path.write_text(json.dumps(summary) + "\n")
                with mock.patch.multiple(
                        authorization,
                        FINAL_REPORT_SUMMARY=summary_path,
                        FINAL_REPORT_MARKDOWN=report,
                        REPORT_INPUTS={upstream},
                        START_SEAL_ROOT=seal_root,
                        PROMOTION_REPORT_INPUT=completion):
                    observed, digest = (
                        authorization.validated_prepublication_final_report(
                            outcome))
                self.assertEqual(observed, summary)
                self.assertEqual(digest, authorization.sha256(summary_path))

    def test_prepublication_report_rejects_tampered_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            upstream = root / "upstream.json"
            report = root / "FINAL_REPORT.md"
            summary_path = root / "summary.json"
            upstream.write_text("upstream\n")
            report.write_text("original\n")
            summary_path.write_text(json.dumps({
                "status": "complete_fixed_challenger_report_rejected",
                "outcome": "candidate_rejected",
                "decision_validation_mode":
                    "prepublication_complete_graph_revalidated",
                "terminal_decision_sha256": None,
                "report": str(report),
                "artifact_sha256": {
                    "final_report": authorization.sha256(report)},
                "input_sha256": {
                    str(upstream): authorization.sha256(upstream)},
                "registry_modified_by_report_writer": False,
                "frozen_sources_modified": False,
                "production_sources_modified": False,
            }) + "\n")
            report.write_text("tampered\n")
            with mock.patch.multiple(
                    authorization,
                    FINAL_REPORT_SUMMARY=summary_path,
                    FINAL_REPORT_MARKDOWN=report,
                    REPORT_INPUTS={upstream},
                    PROMOTION_REPORT_INPUT=root / "unused.json"):
                with self.assertRaisesRegex(RuntimeError, "Markdown changed"):
                    authorization.validated_prepublication_final_report(
                        "candidate_rejected")


if __name__ == "__main__":
    unittest.main()
