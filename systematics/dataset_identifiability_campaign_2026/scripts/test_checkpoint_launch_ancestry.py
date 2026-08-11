#!/usr/bin/env python3
"""Focused tests for prospective checkpoint launch ancestry receipts."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import checkpoint_launch_ancestry as ancestry


class CheckpointLaunchAncestryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.receipts = self.root / "receipts"
        self.outputs = self.root / "outputs"
        self.parent = self.outputs / "parent"
        self.parent.mkdir(parents=True)
        self.state = self.parent / "model_state.pt"
        self.norms = self.parent / "dataset_norms.csv"
        self.state.write_bytes(b"state-v1")
        self.norms.write_text("dataset,control_norm\nA,1.0\n")
        self.tag = "child_polish64_5000"
        self.child = self.outputs / self.tag

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def expected(self, *, norms: Path | None = None) -> dict:
        norms = self.norms if norms is None else norms
        command = ancestry.build_continuation_command(
            python=Path("/python"), runner=Path("/runner.py"), seed=303,
            source_production=Path("/source"), w_grid=Path("/w.csv"),
            output_root=self.outputs, child_tag=self.tag,
            parent_state=self.state, parent_norms=norms,
            reference_strength=600.0,
            reference_csv=ancestry.FIXED_FNP_REFERENCE,
            reference_bmin=0.1, reference_bmax=4.0,
            barrier_strength=100.0, barrier_power=2,
            barrier_ceiling=150.25, replica_seed=None)
        return ancestry.build_launch_receipt(
            receipt_root=self.receipts, child_output=self.child,
            child_tag=self.tag, parent_state=self.state,
            parent_norms=norms, fit_seed=303, replica_seed=None,
            command=command, reference_strength=600.0,
            reference_bmin=0.1, reference_bmax=4.0,
            barrier_strength=100.0, barrier_power=2,
            barrier_ceiling=150.25)

    def test_receipt_is_durable_before_child_exists_and_can_resume(self) -> None:
        expected = self.expected()
        first = ancestry.prepare_launch_receipt(
            self.receipts, self.child, self.tag, expected)
        self.assertFalse(self.child.exists())
        self.assertEqual(first["kind"], "launch_time_content_receipt")
        second = ancestry.prepare_launch_receipt(
            self.receipts, self.child, self.tag, expected)
        self.assertEqual(first, second)

    def test_cached_child_without_receipt_fails_closed(self) -> None:
        self.child.mkdir(parents=True)
        (self.child / "fit_status.json").write_text("{}\n")
        with self.assertRaisesRegex(RuntimeError, "without a launch-time receipt"):
            ancestry.prepare_launch_receipt(
                self.receipts, self.child, self.tag, self.expected())
        with self.assertRaisesRegex(RuntimeError, "receipt is missing"):
            ancestry.classify_launch_ancestry(
                self.receipts, self.child, self.tag, self.expected(),
                allow_legacy_without_receipt=False)

    def test_parent_state_mutation_invalidates_cached_receipt(self) -> None:
        expected = self.expected()
        ancestry.prepare_launch_receipt(
            self.receipts, self.child, self.tag, expected)
        self.state.write_bytes(b"state-v2")
        changed = self.expected()
        with self.assertRaisesRegex(RuntimeError, "content/parent mismatch"):
            ancestry.validate_launch_receipt(
                self.receipts, self.tag, changed)

    def test_parent_norm_mutation_invalidates_cached_receipt(self) -> None:
        expected = self.expected()
        ancestry.prepare_launch_receipt(
            self.receipts, self.child, self.tag, expected)
        self.norms.write_text("dataset,control_norm\nA,1.2\n")
        changed = self.expected()
        with self.assertRaisesRegex(RuntimeError, "content/parent mismatch"):
            ancestry.validate_launch_receipt(
                self.receipts, self.tag, changed)

    def test_different_norm_path_is_bound_even_with_same_bytes(self) -> None:
        expected = self.expected()
        ancestry.prepare_launch_receipt(
            self.receipts, self.child, self.tag, expected)
        alternate = self.parent / "alternate_norms.csv"
        alternate.write_bytes(self.norms.read_bytes())
        changed = self.expected(norms=alternate)
        self.assertNotEqual(
            expected["parent_norms"]["path"],
            changed["parent_norms"]["path"])
        with self.assertRaisesRegex(RuntimeError, "content/parent mismatch"):
            ancestry.validate_launch_receipt(
                self.receipts, self.tag, changed)

    def test_receipt_tamper_is_rejected(self) -> None:
        expected = self.expected()
        ancestry.prepare_launch_receipt(
            self.receipts, self.child, self.tag, expected)
        path = ancestry.receipt_path(self.receipts, self.tag)
        tampered = json.loads(path.read_text())
        tampered["fit_seed"] = 999
        path.write_text(json.dumps(tampered) + "\n")
        with self.assertRaisesRegex(RuntimeError, "content/parent mismatch"):
            ancestry.validate_launch_receipt(
                self.receipts, self.tag, expected)

    def test_legacy_classification_is_explicitly_not_launch_proof(self) -> None:
        self.child.mkdir(parents=True)
        (self.child / "fit_status.json").write_text("{}\n")
        result = ancestry.classify_launch_ancestry(
            self.receipts, self.child, self.tag, self.expected(),
            allow_legacy_without_receipt=True)
        self.assertEqual(
            result["kind"],
            "legacy_pre_receipt_path_only_requires_contemporaneous_seal")
        self.assertIsNone(result["sha256"])

    def test_immutable_generation_rejects_changed_payload(self) -> None:
        path = self.root / "seal.json"
        ancestry.immutable_create_or_validate_json(path, {"value": 1})
        ancestry.immutable_create_or_validate_json(path, {"value": 1})
        with self.assertRaisesRegex(RuntimeError, "content mismatch"):
            ancestry.immutable_create_or_validate_json(path, {"value": 2})


if __name__ == "__main__":
    unittest.main()
