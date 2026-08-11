#!/usr/bin/env python3
"""Focused tests for the final lambda600 directional-envelope handoff."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


builder = load_module(
    "tested_final_directional_envelope",
    "build_lambda600_final_directional_envelope.py",
)
validation = load_module(
    "tested_nested_final_validation",
    "nested_interaction_and_final_envelope_validation.py",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class EnvelopeAlgebraTests(unittest.TestCase):
    def test_fnp_uses_minmax_checkpoint_bounds_and_log_interaction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            b = np.linspace(0.0001, 8.0, 321)
            rows = []
            for checkpoint, low, median, high, central in (
                ("terminal", 0.80, 1.00, 1.20, 1.05),
                ("stationarity_anchor", 0.75, 0.98, 1.25, 0.99),
            ):
                rows.append(pd.DataFrame({
                    "checkpoint": checkpoint,
                    "bT": b,
                    "q16": low,
                    "median": median,
                    "q84": high,
                    "declared_trained_central": central,
                }))
            product = root / "product.csv"
            pd.concat(rows, ignore_index=True).to_csv(product, index=False)
            interaction = root / "interaction.csv"
            pd.DataFrame({
                "bT": b,
                "interaction_log_delta_low": -0.10,
                "interaction_log_delta_high": 0.20,
            }).to_csv(interaction, index=False)
            observed = builder.build_fnp(
                {"artifacts": {"fnp_checkpoint_bands": str(product)}},
                {"artifacts": {"logfnp_directional_envelope": str(interaction)}},
            )
        np.testing.assert_allclose(
            observed["final_envelope_low"], 0.75 * np.exp(-0.10))
        np.testing.assert_allclose(
            observed["final_envelope_high"], 1.25 * np.exp(0.20))
        np.testing.assert_allclose(observed["trained_central"], 1.05)

    def test_exact_final_statistic_uses_both_checkpoints_fixed_central_and_mask(self) -> None:
        k = np.linspace(0.0, 1.0, 7)
        central = np.asarray([
            3.0 * np.exp(-k), 2.0 * np.exp(-k),
        ])
        start = np.linspace(-0.02, 0.02, 4)[None, :, None, None]
        replica = np.linspace(-0.01, 0.01, 6)[None, None, :, None]
        base = central[:, None, None, :]
        terminal = base * (1.0 + start + replica)
        anchor = base * (1.0 + 1.5 * start - replica)
        low = -0.003 * central
        high = 0.004 * central
        mask = np.ones_like(central, dtype=bool)
        observed = builder.final_joint_width_statistic(
            terminal, anchor, low, high, central, mask
        )
        tq16, tq84 = np.quantile(terminal.reshape(2, -1, len(k)),
                                 (0.16, 0.84), axis=1)
        aq16, aq84 = np.quantile(anchor.reshape(2, -1, len(k)),
                                 (0.16, 0.84), axis=1)
        expected_curve = (
            np.maximum(tq84, aq84) + high
            - np.minimum(tq16, aq16) - low
        ) / central
        np.testing.assert_allclose(observed, np.max(expected_curve, axis=1))

    def test_final_statistic_resampling_is_deterministic_and_exact(self) -> None:
        k = np.linspace(0.0, 4.0, 401)
        central = np.asarray([
            3.0 * np.exp(-k**2), 2.0 * np.exp(-k**2),
        ])
        starts = np.linspace(-0.025, 0.025, 24)[None, :, None, None]
        replicas = np.linspace(-0.015, 0.015, 50)[None, None, :, None]
        terminal = central[:, None, None, :] * (1.0 + starts + replicas)
        anchor = central[:, None, None, :] * (
            1.0 + 0.7 * starts + 1.2 * replicas
        )
        interaction_low = -0.002 * central
        interaction_high = 0.003 * central
        rows = []
        incumbent_rows = []
        for flavor_index, flavor in enumerate(("u", "d")):
            tq16, tq84 = np.quantile(
                terminal[flavor_index].reshape(1200, 401), (0.16, 0.84), axis=0
            )
            aq16, aq84 = np.quantile(
                anchor[flavor_index].reshape(1200, 401), (0.16, 0.84), axis=0
            )
            final_low = np.minimum(tq16, aq16) + interaction_low[flavor_index]
            final_high = np.maximum(tq84, aq84) + interaction_high[flavor_index]
            rows.append(pd.DataFrame({
                "flavor": flavor, "kT": k,
                "interaction_delta_low": interaction_low[flavor_index],
                "interaction_delta_high": interaction_high[flavor_index],
                "final_envelope_low": final_low,
                "trained_central": central[flavor_index],
                "final_envelope_high": final_high,
            }))
            incumbent_rows.append(pd.DataFrame({
                "flavor": flavor, "kT": k,
                "central": central[flavor_index],
            }))
        arrays = {
            "flavors": np.asarray(("u", "d")), "kT": k,
            "terminal_values": terminal,
            "stationarity_anchor_values": anchor,
            "terminal_declared_central": central,
        }
        inputs = (pd.concat(rows), arrays, pd.concat(incumbent_rows))
        first = builder.exact_final_statistic_resampling(
            *inputs, bootstraps=3, splits=2, seed=17
        )
        second = builder.exact_final_statistic_resampling(
            *inputs, bootstraps=3, splits=2, seed=17
        )
        self.assertEqual(first[0], second[0])
        pd.testing.assert_frame_equal(first[1], second[1])
        pd.testing.assert_frame_equal(first[2], second[2])
        self.assertEqual(set(first[0]["allowance_by_flavor"]), {"u", "d"})
        for flavor in ("u", "d"):
            expected = max(
                first[0][key][flavor] for key in (
                    "bootstrap_p95_absolute_deviation_by_flavor",
                    "start_split_p95_absolute_difference_by_flavor",
                    "replica_split_p95_absolute_difference_by_flavor",
                    "joint_split_p95_absolute_difference_by_flavor",
                )
            )
            self.assertEqual(first[0]["allowance_by_flavor"][flavor], expected)

    def test_kspace_uses_additive_directional_interaction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            k = np.linspace(0.0, 4.0, 401)
            rows = []
            interaction_rows = []
            for flavor in ("u", "d"):
                for checkpoint, low, median, high, central in (
                    ("terminal", 0.80, 1.00, 1.20, 1.05),
                    ("stationarity_anchor", 0.75, 0.98, 1.25, 0.99),
                ):
                    rows.append(pd.DataFrame({
                        "model": "lambda600", "tail_mode": "expb2",
                        "flavor": flavor, "checkpoint": checkpoint,
                        "kT": k, "q16": low, "median": median,
                        "q84": high, "declared_central": central,
                    }))
                interaction_rows.append(pd.DataFrame({
                    "flavor": flavor, "kT": k,
                    "interaction_delta_low": -0.03,
                    "interaction_delta_high": 0.04,
                }))
            product = root / "product.csv"
            pd.concat(rows, ignore_index=True).to_csv(product, index=False)
            interaction = root / "interaction.csv"
            pd.concat(interaction_rows, ignore_index=True).to_csv(
                interaction, index=False)
            observed = builder.build_kspace(
                {"artifacts": {"kspace_checkpoint_expb2_bands": str(product)}},
                {"artifacts": {"kspace_directional_envelope": str(interaction)}},
            )
        np.testing.assert_allclose(observed["final_envelope_low"], 0.72)
        np.testing.assert_allclose(observed["final_envelope_high"], 1.29)
        np.testing.assert_allclose(observed["trained_central"], 1.05)


class EvidenceAndOrderingTests(unittest.TestCase):
    def test_every_declared_artifact_is_hash_bound_and_tampering_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "artifact.csv"
            artifact.write_text("x\n1\n")
            payload = {
                "artifacts": {"example": str(artifact)},
                "artifact_sha256": {"example": digest(artifact)},
            }
            validation._validate_artifacts(payload, "test")
            artifact.write_text("x\n2\n")
            with self.assertRaisesRegex(RuntimeError, "artifact changed"):
                validation._validate_artifacts(payload, "test")

    def test_unattended_order_is_fixed_and_figures_precede_promotion(self) -> None:
        text = (HERE / "continue_lambda600_like_for_like.py").read_text()
        ordered = [
            "build_final_combined_tmd_ensemble.py",
            "audit_final_combined_ensemble.py",
            "audit_lambda600_postfit_tail_transform.py",
            "validate_lambda600_nested_interactions.py",
            "build_lambda600_final_directional_envelope.py",
        ]
        positions = [text.index(name) for name in ordered]
        self.assertEqual(positions, sorted(positions))
        self.assertLess(
            text.index("plot_validated_final_fig2_fig6.py"),
            text.index("promote_validated_final_champion.py"),
        )

    def test_no_intermediate_audit_can_authorize_another_constraint(self) -> None:
        nested_text = (
            HERE / "validate_lambda600_nested_interactions.py"
        ).read_text()
        self.assertIn(
            '"promotion_eligible_after_observed_interaction_envelope": False',
            nested_text,
        )
        self.assertIn("never authorizes promotion or another lambda/prior", nested_text)
        final_text = (
            HERE / "build_lambda600_final_directional_envelope.py"
        ).read_text()
        self.assertIn('"promotion_eligible": False', final_text)
        self.assertIn('"diagnostic_figure_gate_pass": True', final_text)
        self.assertIn('"bspace_interaction_limitation"', final_text)

    def test_final_promotion_gate_is_exact_five_component_conjunction(self) -> None:
        payload = {
            "base_product_stability_gate_pass": True,
            "postfit_tail_convergence_gate_pass": True,
            "nested_interaction_validation_gate_pass": True,
            "joint_width_replacement_gate_pass": True,
            "trained_central_containment_gate_pass": True,
        }
        self.assertTrue(validation.reconstructed_final_promotion_gate(payload))
        for key in tuple(payload):
            changed = dict(payload)
            changed[key] = False
            self.assertFalse(validation.reconstructed_final_promotion_gate(changed))


if __name__ == "__main__":
    unittest.main(verbosity=2)
