#!/usr/bin/env python3
"""Focused tests for unattended lambda600 technical-retry semantics."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import classify_lambda600_outcome as classifier
import continue_lambda600_like_for_like as continuation


class ContinuationExitTests(unittest.TestCase):
    @staticmethod
    def complete_replica_manifest() -> dict:
        return {
            "status": "complete_with_scientific_failures",
            "central_endpoint_tag": "lambda600_central_polish64_300000",
            "central_full_horizon_complete": True,
            "central_full_horizon_requested_capacity": 300000,
            "central_terminal_requested_capacity": 300000,
            "completed_replica_count": 50,
            "replica_endpoint_tags": [
                f"lambda600_replica_r{seed}" for seed in range(1001, 1051)
            ],
        }

    @staticmethod
    def materialize_endpoints(root: Path, tags: list[str]) -> None:
        for tag in tags:
            endpoint = root / tag
            endpoint.mkdir(parents=True, exist_ok=True)
            for name in continuation.TERMINAL_ENDPOINT_FILES:
                (endpoint / name).write_bytes(b"test\n")

    def test_final_report_is_written_prepublication_and_bound(self) -> None:
        calls = []
        summary = {"artifact_sha256": {"final_report": "report-hash"}}

        def runner(script, *, args=(), check=True):
            calls.append((script, args, check))
            return 0

        binding = continuation.prepare_final_study_report(
            "candidate_rejected", runner=runner,
            validator=lambda status: (summary, "summary-hash"),
        )
        self.assertEqual(calls, [(
            "write_final_study_report.py",
            ("--decision-status", "candidate_rejected"), False,
        )])
        self.assertEqual(binding["status"], "pass")
        self.assertEqual(binding["outcome"], "candidate_rejected")
        self.assertEqual(binding["summary_sha256"], "summary-hash")
        self.assertEqual(binding["report_sha256"], "report-hash")

    def test_final_report_failure_is_retryable_not_terminal(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "writer exited"):
            continuation.prepare_final_study_report(
                "candidate_rejected",
                runner=lambda *args, **kwargs: 7,
                validator=lambda status: ({}, "must-not-run"),
            )

    def test_failed_starts_must_exhaust_exact_300k_horizon(self) -> None:
        payload = {
            "failed_seeds": [306, 310],
            "failed_terminal_requested_capacity_by_seed": {
                "306": 300000, "310": 300000,
            },
            "full_requested_capacity_per_nonpassing_start": 300000,
            "failed_starts_exhausted_full_requested_horizon": True,
        }
        self.assertEqual(
            continuation.require_failed_start_horizon_exhaustion(payload),
            {306: 300000, 310: 300000},
        )

    def test_legacy_255k_failure_is_not_terminal_for_continuation(self) -> None:
        payload = {
            "failed_seeds": [306],
            "failed_terminal_requested_capacity_by_seed": {"306": 255000},
            "full_requested_capacity_per_nonpassing_start": 300000,
            "failed_starts_exhausted_full_requested_horizon": False,
        }
        with self.assertRaisesRegex(RuntimeError, "exhaust exactly 300000"):
            continuation.require_failed_start_horizon_exhaustion(payload)

    def test_passing_ensemble_has_vacuously_complete_failure_horizon(self) -> None:
        payload = {
            "failed_seeds": [],
            "failed_terminal_requested_capacity_by_seed": {},
            "full_requested_capacity_per_nonpassing_start": 300000,
            "failed_starts_exhausted_full_requested_horizon": True,
        }
        self.assertEqual(
            continuation.require_failed_start_horizon_exhaustion(payload), {})

    def test_updated_full24_ancestry_metadata_is_required(self) -> None:
        payload = {
            "launch_time_content_receipt_checkpoint_count": 7,
            "legacy_pre_receipt_checkpoint_count": 2,
            "legacy_pre_receipt_admission_tag_count": 3,
            "legacy_pre_receipt_admission_tags": ["a", "b", "unused"],
            "legacy_pre_receipt_used_tags": ["a", "b"],
            "legacy_checkpoint_ancestry_semantics": (
                "the seal does not retroactively prove bytes consumed"),
        }
        result = continuation.require_full24_launch_ancestry_metadata(payload)
        self.assertEqual(result[
            "launch_time_content_receipt_checkpoint_count"], 7)
        self.assertEqual(result["legacy_pre_receipt_checkpoint_count"], 2)

    def test_old_full24_manifest_requires_repair_before_central(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "lacks updated"):
            continuation.require_full24_launch_ancestry_metadata({})

    def test_full24_ancestry_used_tags_must_be_pre_admitted(self) -> None:
        payload = {
            "launch_time_content_receipt_checkpoint_count": 0,
            "legacy_pre_receipt_checkpoint_count": 1,
            "legacy_pre_receipt_admission_tag_count": 1,
            "legacy_pre_receipt_admission_tags": ["a"],
            "legacy_pre_receipt_used_tags": ["later"],
            "legacy_checkpoint_ancestry_semantics": (
                "the seal does not retroactively prove bytes consumed"),
        }
        with self.assertRaisesRegex(RuntimeError, "inconsistent"):
            continuation.require_full24_launch_ancestry_metadata(payload)

    def test_legacy_horizon_repair_precedes_central_ancestry_audit(self) -> None:
        source = Path(continuation.__file__).read_text()
        repair = source.index(
            '"full24_horizon_and_ancestry_generation_repair"')
        ancestry = source.index("start_audit_returncode = run")
        self.assertLess(repair, ancestry)
        self.assertIn(
            "run(\n                FULL24_SCRIPT, args=FULL24_ARGS, check=False)",
            source,
        )

    def test_partial_scientific_replica_status_replays_until_full50(self) -> None:
        partial = self.complete_replica_manifest()
        partial["status"] = "central_stationarity_failed"
        partial["completed_replica_count"] = 0
        partial["replica_endpoint_tags"] = []
        complete = self.complete_replica_manifest()
        with tempfile.TemporaryDirectory() as directory:
            outputs = Path(directory)
            self.materialize_endpoints(outputs, [
                complete["central_endpoint_tag"],
                *complete["replica_endpoint_tags"],
            ])
            state = {"payload": partial}
            launches = []

            def runner(script, *, args=(), check=True):
                launches.append((script, args, check))
                state["payload"] = complete
                return 0

            ready = lambda payload: continuation.replica_terminal_coverage_ready(
                payload, outputs_root=outputs)
            result, error, history = continuation.recover_terminal_summary(
                relative="selected_reference_central_replicas/summary.json",
                terminal_statuses=continuation.REPLICA_TERMINAL_STATUSES,
                stage="central_replicas",
                script="supervise_selected_reference_central_replicas.py",
                max_attempts=1,
                backoffs=(0,),
                summary_loader=lambda relative: (state["payload"], None),
                quiescence_waiter=lambda stage: [],
                runner=runner,
                sleeper=lambda seconds: None,
                terminal_evidence_ready=ready,
            )
            self.assertIsNone(error)
            self.assertTrue(ready(result))
            self.assertEqual(len(history), 1)
            self.assertEqual(launches, [(
                "supervise_selected_reference_central_replicas.py", (), False,
            )])

    def test_partial_scientific_full24_status_replays_until_exact24(self) -> None:
        partial = {
            "status": "verification_failed",
            "member_count": 11,
            "endpoint_tags": [f"start_{seed}" for seed in range(303, 314)],
        }
        complete = {
            "status": "verification_failed",
            "member_count": 24,
            "endpoint_tags": [f"start_{seed}" for seed in range(303, 327)],
        }
        with tempfile.TemporaryDirectory() as directory:
            outputs = Path(directory)
            self.materialize_endpoints(outputs, complete["endpoint_tags"])
            state = {"payload": partial}
            launches = []

            def runner(script, *, args=(), check=True):
                launches.append((script, args, check))
                state["payload"] = complete
                return 0

            ready = lambda payload: continuation.full24_terminal_coverage_ready(
                payload, outputs_root=outputs)
            result, error, history = continuation.recover_terminal_summary(
                relative="replica_robust_reference_full24/summary.json",
                terminal_statuses=continuation.FULL24_TERMINAL_STATUSES,
                stage="full24",
                script=continuation.FULL24_SCRIPT,
                script_args=continuation.FULL24_ARGS,
                max_attempts=1,
                backoffs=(0,),
                summary_loader=lambda relative: (state["payload"], None),
                quiescence_waiter=lambda stage: [],
                runner=runner,
                sleeper=lambda seconds: None,
                terminal_evidence_ready=ready,
            )
            self.assertIsNone(error)
            self.assertTrue(ready(result))
            self.assertEqual(len(history), 1)
            self.assertEqual(launches, [(
                continuation.FULL24_SCRIPT, continuation.FULL24_ARGS, False,
            )])

    def test_technical_failure_requires_nonzero_restart_exit(self) -> None:
        with self.assertRaisesRegex(
                RuntimeError, "retryable lambda600 technical failure"):
            continuation.require_restartable_controller_exit({
                "status": "technical_failure",
                "stage": "figure_render_failed",
            })

    def test_scientific_terminal_decisions_exit_normally(self) -> None:
        for status in sorted(continuation.FINAL_SCIENTIFIC_STATUSES):
            with self.subTest(status=status):
                payload = {
                    "status": status,
                    "stage": "complete_like_for_like_comparison",
                }
                continuation.require_restartable_controller_exit(
                    payload, validator=lambda: (payload, "decision-hash"))

    def test_terminal_label_without_validated_graph_cannot_exit_normally(self) -> None:
        payload = {
            "status": "candidate_promoted_as_new_study_champion",
            "stage": "complete_like_for_like_comparison",
        }
        with self.assertRaisesRegex(RuntimeError, "artifact graph invalid"):
            continuation.require_restartable_controller_exit(
                payload,
                validator=lambda: (_ for _ in ()).throw(
                    RuntimeError("artifact graph invalid")),
            )

    def test_nonterminal_return_cannot_exit_successfully(self) -> None:
        with self.assertRaisesRegex(
                RuntimeError, "without a complete scientific decision"):
            continuation.require_restartable_controller_exit({
                "status": "in_progress",
                "stage": "central_plus_50_replicas",
            })


class ClassifierWaitTests(unittest.TestCase):
    def test_transient_technical_decision_does_not_retire_classifier(self) -> None:
        sequence = iter((
            {"status": "in_progress"},
            {"status": "technical_failure"},
            {"status": "in_progress"},
            {
                "status": "candidate_rejected",
                "stage": "complete_like_for_like_comparison",
            },
        ))
        sleeps = []
        result = classifier.wait_for_scientific_terminal_decision(
            loader=lambda: next(sequence),
            sleeper=sleeps.append,
            poll_seconds=0.25,
        )
        self.assertEqual(result["status"], "candidate_rejected")
        self.assertEqual(sleeps, [0.25, 0.25, 0.25])

    def test_scientific_terminal_decision_returns_without_waiting(self) -> None:
        sleeps = []
        decision = {
            "status": "candidate_promoted_as_new_study_champion",
            "stage": "complete_like_for_like_comparison",
        }
        result = classifier.wait_for_scientific_terminal_decision(
            loader=lambda: decision,
            sleeper=sleeps.append,
            poll_seconds=0.25,
        )
        self.assertIs(result, decision)
        self.assertEqual(sleeps, [])


if __name__ == "__main__":
    unittest.main()
