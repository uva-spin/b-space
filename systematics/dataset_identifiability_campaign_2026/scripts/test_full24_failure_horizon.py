#!/usr/bin/env python3
"""Focused tests that scientific failure cannot truncate lambda600 starts."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def load_verifier():
    path = HERE / "verify_replica_robust_reference_full24.py"
    spec = importlib.util.spec_from_file_location(
        "tested_full24_failure_horizon", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


verifier = load_verifier()


class Full24FailureHorizonTests(unittest.TestCase):
    def records(self, failed_capacity: int) -> list[dict]:
        return [
            {
                "seed": seed,
                "cumulative_lbfgs_iterations": (
                    failed_capacity if seed == 306 else 250_000),
            }
            for seed in verifier.SEEDS
        ]

    def test_255k_failure_is_not_complete_distribution_evidence(self) -> None:
        metadata = verifier.terminal_capacity_metadata(
            self.records(255_000), [306])
        self.assertFalse(
            metadata["failed_starts_exhausted_full_requested_horizon"])
        self.assertEqual(
            metadata["failed_terminal_requested_capacity_by_seed"],
            {306: 255_000},
        )

    def test_300k_failure_exhausts_the_declared_horizon(self) -> None:
        metadata = verifier.terminal_capacity_metadata(
            self.records(300_000), [306])
        self.assertTrue(
            metadata["failed_starts_exhausted_full_requested_horizon"])
        self.assertEqual(
            metadata["full_requested_capacity_per_nonpassing_start"],
            300_000,
        )

    def test_no_impossibility_shortcut_remains_in_training_loop(self) -> None:
        text = (HERE / "verify_replica_robust_reference_full24.py").read_text()
        self.assertNotIn("remaining < confirmations_needed", text)
        self.assertIn("Continue every such", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
