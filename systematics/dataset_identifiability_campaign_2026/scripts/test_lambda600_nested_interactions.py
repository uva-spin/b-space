#!/usr/bin/env python3
"""Focused tests for the isolated lambda600 nested-interaction validator."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def load_module():
    path = HERE / "validate_lambda600_nested_interactions.py"
    spec = importlib.util.spec_from_file_location("tested_nested_interactions", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


nested = load_module()


def load_validation_module():
    path = HERE / "nested_interaction_and_final_envelope_validation.py"
    spec = importlib.util.spec_from_file_location("tested_nested_evidence", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


evidence_validation = load_validation_module()


class SelectionTests(unittest.TestCase):
    def test_start_extremes_are_deterministic_and_exclude_central(self) -> None:
        direction = np.asarray([1.0, -0.5, 0.25, 0.1])
        scores = np.asarray([-4.0, -2.0, 0.0, 1.0, 5.0])
        logs = scores[:, None] * direction[None, :]
        chosen, metadata = nested.select_start_extremes(
            logs, [303, 304, 305, 306, 307], excluded_index=2)
        self.assertEqual(chosen, [0, 4])
        self.assertEqual(metadata["excluded_seed"], 305)
        self.assertEqual(metadata["selected_seeds_low_high"], [303, 307])

    def test_replica_span_selects_low_middle_high(self) -> None:
        direction = np.asarray([1.0, 0.4, -0.2, 0.05])
        scores = np.asarray([-5.0, -1.0, 0.1, 0.2, 7.0])
        logs = scores[:, None] * direction[None, :]
        chosen, metadata = nested.select_replica_span(
            logs, [1001, 1002, 1003, 1004, 1005])
        self.assertEqual(chosen[0], 0)
        self.assertEqual(chosen[-1], 4)
        self.assertEqual(chosen[1], 2)
        self.assertEqual(
            metadata["selected_replicas_low_middle_high"],
            [1001, 1003, 1005])

    def test_exactly_degenerate_design_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "zero variation"):
            nested.select_replica_span(
                np.zeros((5, 4)), [1001, 1002, 1003, 1004, 1005])


class AlgebraTests(unittest.TestCase):
    def test_additive_prediction_counts_central_once(self) -> None:
        central = np.log(np.asarray([2.0, 3.0, 4.0]))
        starts = np.log(np.asarray([
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [4.0, 8.0, 12.0],
        ]))
        replicas = np.log(np.asarray([
            [0.5, 1.0, 2.0],
            [1.0, 2.0, 4.0],
            [2.0, 4.0, 8.0],
        ]))
        observed = nested.additive_log_prediction(
            central, starts, replicas, 2, 0)
        expected = (
            central
            + starts[2] - np.median(starts, axis=0)
            + replicas[0] - np.median(replicas, axis=0)
        )
        np.testing.assert_allclose(observed, expected, rtol=0.0, atol=0.0)

    def test_interaction_tag_is_pair_and_horizon_specific(self) -> None:
        tags = {
            nested.interaction_tag(start, replica, cumulative)
            for start in (303, 326)
            for replica in (1001, 1025, 1050)
            for cumulative in (5_000, 300_000)
        }
        self.assertEqual(len(tags), 12)
        self.assertTrue(all(tag.startswith("nestedint_") for tag in tags))


class StationarityTests(unittest.TestCase):
    def test_ten_wholly_post_200k_quiet_blocks_pass(self) -> None:
        tracker = nested.new_tracker()
        previous = np.ones(5)
        result = None
        for cumulative in range(5_000, 250_001, 5_000):
            current = previous.copy()
            result = nested.update_stationarity_tracker(
                tracker, previous, current, cumulative)
            previous = current
        self.assertIsNotNone(result)
        self.assertTrue(result["stationarity_pass"])

    def test_late_two_percent_failure_resets_confirmation(self) -> None:
        tracker = nested.new_tracker()
        previous = np.ones(5)
        result = None
        for cumulative in range(5_000, 250_001, 5_000):
            current = previous.copy()
            if cumulative == 230_000:
                current *= 1.03
            result = nested.update_stationarity_tracker(
                tracker, previous, current, cumulative)
            previous = current
        self.assertIsNotNone(result)
        self.assertFalse(result["stationarity_pass"])
        self.assertEqual(result["next_anchor_capacity"], 230_000)


class DecisionImpactTests(unittest.TestCase):
    @staticmethod
    def inputs(delta_scale: float):
        grid = np.linspace(0.0, 4.0, 401)
        candidate_rows = []
        incumbent_rows = []
        transform_rows = []
        for flavor in ("u", "d"):
            central = np.exp(-grid)
            for k_t, value in zip(grid, central):
                candidate_rows.append({
                    "flavor": flavor, "kT": k_t,
                    "q16": 0.97 * value, "central": value,
                    "q84": 1.03 * value,
                })
                incumbent_rows.append({
                    "flavor": flavor, "kT": k_t,
                    "q16": 0.94 * value, "central": value,
                    "q84": 1.06 * value,
                })
            for start in (303, 326):
                for replica in (1001, 1025, 1050):
                    sign = -1.0 if (start + replica) % 2 else 1.0
                    for k_t, value in zip(grid, central):
                        predicted = value
                        actual = value + sign * delta_scale * value
                        common = {
                            "seed": start, "pdf_member": replica,
                            "pid": 2 if flavor == "u" else 1,
                            "flavor": flavor, "x": 0.1, "Q": 10.0,
                            "kT": k_t, "quantity": "ftilde",
                        }
                        transform_rows.append({
                            **common, "interaction_component": "actual",
                            "value": actual,
                        })
                        transform_rows.append({
                            **common,
                            "interaction_component": "additive_prediction",
                            "value": predicted,
                        })
        audit = {
            "resampling_full_width_allowance_by_flavor": {"u": 0.005, "d": 0.005},
            "final_max_active_relative_full_width": {"u": 0.06, "d": 0.06},
        }
        return (
            pd.DataFrame(candidate_rows), pd.DataFrame(incumbent_rows), audit,
            pd.DataFrame(transform_rows),
        )

    def test_small_interaction_preserves_replacement_sign(self) -> None:
        envelope, summary = nested.compute_interaction_decision_impact(
            *self.inputs(0.005))
        self.assertFalse(summary["overall_replacement_decision_sign_changed"])
        self.assertTrue(summary[
            "overall_width_replacement_decision_after_interaction"])
        self.assertEqual(set(envelope["flavor"]), {"u", "d"})

    def test_large_interaction_flips_replacement_sign_without_tolerance(self) -> None:
        _, summary = nested.compute_interaction_decision_impact(
            *self.inputs(0.04))
        self.assertTrue(summary["overall_replacement_decision_sign_changed"])
        self.assertFalse(summary[
            "overall_width_replacement_decision_after_interaction"])
        for flavor in ("u", "d"):
            self.assertTrue(summary["by_flavor"][flavor][
                "replacement_decision_sign_changed"])


class RawEndpointEvidenceTests(unittest.TestCase):
    def test_legacy_product_median_sign_cannot_gate_nested_evidence(self) -> None:
        payload = {
            "all_nested_trajectories_stationary_and_fit_preserving": True,
            "observed_interaction_decision_sign_stable": False,
        }
        self.assertTrue(
            evidence_validation.reconstructed_nested_interaction_gate(payload)
        )

    def test_exact_six_endpoint_raw_files_are_hash_bound(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            endpoints = []
            pairs = []
            artifacts = {}
            for start in (303, 326):
                for replica in (1001, 1025, 1050):
                    tag = nested.interaction_tag(start, replica, 300_000)
                    run = root / tag
                    run.mkdir()
                    roles = {
                        "fit_status": run / "fit_status.json",
                        "fnp_grid": run / "fnp_grid.csv",
                        "model_state": run / "model_state.pt",
                        "dataset_norms": run / "dataset_norms.csv",
                    }
                    for role, path in roles.items():
                        path.write_text(f"{role}\n")
                    hashes = {
                        role: evidence_validation.sha256(path)
                        for role, path in roles.items()
                    }
                    endpoints.append({
                        "start_seed": start, "replica_seed": replica,
                        "endpoint_tag": tag,
                        "raw_artifacts": {
                            role: str(path) for role, path in roles.items()
                        },
                        "raw_artifact_sha256": hashes,
                    })
                    pairs.append({"start_seed": start, "replica_seed": replica})
                    prefix = f"terminal_s{start}_r{replica}"
                    artifacts.update({
                        f"{prefix}_{role}": path for role, path in roles.items()
                    })
            evidence_validation.validate_nested_raw_endpoints(
                endpoints, pairs, artifacts
            )
            tampered = Path(endpoints[0]["raw_artifacts"]["fnp_grid"])
            tampered.write_text("changed\n")
            with self.assertRaisesRegex(RuntimeError, "changed"):
                evidence_validation.validate_nested_raw_endpoints(
                    endpoints, pairs, artifacts
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
