#!/usr/bin/env python3
"""Focused tests for the pre-registered lambda600 protocol binding."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent


def load_module():
    path = HERE / "fixed_challenger_protocol.py"
    spec = importlib.util.spec_from_file_location(
        "tested_fixed_challenger_protocol", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


protocol = load_module()


class FixedChallengerProtocolTests(unittest.TestCase):
    def test_registered_manifest_matches_pinned_digest_and_schema(self) -> None:
        payload, digest = protocol.validate_fixed_challenger_protocol()
        self.assertEqual(digest, protocol.EXPECTED_SHA256)
        self.assertEqual(
            protocol.sha256(protocol.FNP_REFERENCE),
            protocol.EXPECTED_FNP_REFERENCE_SHA256,
        )
        for role, path in protocol.IMPLEMENTATION_FILES.items():
            with self.subTest(implementation=role):
                self.assertEqual(
                    protocol.sha256(path),
                    protocol.EXPECTED_IMPLEMENTATION_SHA256[role],
                )
        binding = protocol.fixed_implementation_binding()
        protocol.require_fixed_implementation_binding(
            binding, "positive test binding")
        self.assertEqual(payload["candidate"]["start_seeds"], list(range(303, 327)))
        self.assertEqual(
            payload["candidate"]["experimental_replica_seeds"],
            list(range(1001, 1051)),
        )

    def test_byte_tamper_fails_before_schema_can_be_reinterpreted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory) / "protocol.json"
            copy.write_bytes(protocol.PROTOCOL.read_bytes() + b" ")
            with self.assertRaisesRegex(RuntimeError, "SHA256 changed"):
                protocol.validate_fixed_challenger_protocol(
                    copy, require_canonical_paths=False)

    def test_decision_critical_schema_fails_even_with_matching_test_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory) / "protocol.json"
            payload = json.loads(protocol.PROTOCOL.read_text())
            payload["candidate"]["reference_distance_lambda"] = 601.0
            copy.write_text(json.dumps(payload, indent=2) + "\n")
            with self.assertRaisesRegex(RuntimeError, "schema/value"):
                protocol.validate_fixed_challenger_protocol(
                    copy, expected_sha256=protocol.sha256(copy),
                    require_canonical_paths=False)

    def test_empirical_reference_byte_tamper_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory) / "fnp_median.csv"
            copy.write_bytes(protocol.FNP_REFERENCE.read_bytes() + b"\n")
            with self.assertRaisesRegex(
                    RuntimeError, "empirical FNP reference SHA256 changed"):
                protocol.validate_fixed_challenger_protocol(
                    fnp_reference=copy, require_canonical_paths=False)

    def test_external_implementation_byte_tamper_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            role = "fit_runner"
            copy = Path(directory) / "run_production_fnp_stability_control.py"
            copy.write_bytes(
                protocol.IMPLEMENTATION_FILES[role].read_bytes() + b"\n")
            implementations = dict(protocol.IMPLEMENTATION_FILES)
            implementations[role] = copy
            with self.assertRaisesRegex(
                    RuntimeError,
                    "fixed implementation SHA256 changed for fit_runner"):
                protocol.validate_fixed_challenger_protocol(
                    implementation_files=implementations,
                    require_canonical_paths=False)

    def test_evidence_implementation_binding_tamper_fails_closed(self) -> None:
        binding = protocol.fixed_implementation_binding()
        binding["fixed_implementation_sha256"]["film_trainer"] = "0" * 64
        with self.assertRaisesRegex(
                RuntimeError, "lacks the fixed implementation hashes"):
            protocol.require_fixed_implementation_binding(
                binding, "tampered evidence")


class DormantAlternateLauncherTests(unittest.TestCase):
    def test_legacy_recovery_entry_points_fail_before_wait_or_fit(self) -> None:
        commands = (
            [
                sys.executable,
                str(HERE / "continue_after_barrier_stress_with_recovery.py"),
                "--wait-pid",
                str(os.getpid()),
            ],
            [
                sys.executable,
                str(HERE / "recover_barrier_reference_strength_after_stress_failure.py"),
            ],
        )
        for command in commands:
            with self.subTest(script=Path(command[1]).name):
                result = subprocess.run(
                    command,
                    cwd=HERE.parent,
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(
                    "alternate-lambda launch blocked",
                    result.stdout + result.stderr,
                )


if __name__ == "__main__":
    unittest.main()
