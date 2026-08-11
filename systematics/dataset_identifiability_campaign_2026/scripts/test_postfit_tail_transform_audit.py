#!/usr/bin/env python3
"""Focused tests for the post-fit full-tail/transform promotion audit."""

from __future__ import annotations

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


audit = load_module(
    "tested_postfit_tail_transform_audit",
    "audit_lambda600_postfit_tail_transform.py",
)
transformer = audit.load_module("tested_frozen_transformer", audit.TRANSFORMER)
validation = load_module(
    "tested_postfit_tail_transform_validation",
    "postfit_tail_transform_validation.py",
)


def bands(width: float, *, shift: float = 0.0) -> pd.DataFrame:
    k = np.linspace(0.0, 4.0, 401)
    rows = []
    for flavor, height in (("u", 3.0), ("d", 2.0)):
        central = height * np.exp(-k**2) + shift
        half = 0.5 * width * central
        rows.append(
            pd.DataFrame(
                {
                    "flavor": flavor,
                    "kT": k,
                    "q16": central - half,
                    "median": central,
                    "q84": central + half,
                    "declared_central": central,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def incumbent(width: float) -> pd.DataFrame:
    frame = bands(width).rename(columns={"median": "central"})
    return frame[["flavor", "kT", "q16", "central", "q84"]]


class CheckpointSelectionTests(unittest.TestCase):
    def test_terminal_row_resolves_its_formal_window_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for cumulative in (200000, 205000, 210000):
                tag = f"run_s303_{cumulative}"
                run = root / tag
                run.mkdir()
                (run / "fnp_grid.csv").write_text("x,bT,F_NP\n")
                (run / "fit_status.json").write_text("{}\n")
                rows.append(
                    {
                        "tag": tag,
                        "seed": 303,
                        "cumulative_lbfgs_iterations": cumulative,
                        "stationarity_window_anchor_iterations": (
                            np.nan if cumulative == 200000 else 200000
                        ),
                    }
                )
            old = audit.OUTPUTS
            try:
                audit.OUTPUTS = root
                pair = audit.checkpoint_pair(
                    pd.DataFrame(rows), "run_s303_210000", family="start"
                )
            finally:
                audit.OUTPUTS = old
        self.assertEqual(pair["anchor_tag"], "run_s303_200000")
        self.assertEqual(pair["anchor_iterations"], 200000)
        self.assertEqual(pair["terminal_iterations"], 210000)

    def test_nonstationary_chain_uses_fixed_200k_not_late_reset_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for cumulative in (200000, 280000, 300000):
                tag = f"run_s306_{cumulative}"
                run = root / tag
                run.mkdir()
                (run / "fnp_grid.csv").write_text("x,bT,F_NP\n")
                (run / "fit_status.json").write_text("{}\n")
                rows.append({
                    "tag": tag, "seed": 306,
                    "cumulative_lbfgs_iterations": cumulative,
                    "stationarity_window_anchor_iterations": (
                        np.nan if cumulative == 200000 else 280000
                    ),
                })
            old = audit.OUTPUTS
            try:
                audit.OUTPUTS = root
                pair = audit.checkpoint_pair(
                    pd.DataFrame(rows), "run_s306_300000", family="start",
                    scientifically_nonstationary=True,
                )
            finally:
                audit.OUTPUTS = old
        self.assertEqual(pair["anchor_iterations"], 200000)
        self.assertEqual(
            pair["anchor_selection_rule"],
            "fixed_formal_200k_for_nonstationary_chain",
        )


class OutcomeStabilityTests(unittest.TestCase):
    def test_existing_allowance_covers_small_checkpoint_movement(self) -> None:
        terminal = bands(0.08)
        anchor = bands(0.081)
        result, passed = audit.checkpoint_outcome_metrics(
            terminal,
            anchor,
            incumbent(0.11),
            {"u": 0.01, "d": 0.01},
            {"u": True, "d": True},
        )
        self.assertTrue(passed)
        for flavor in audit.FLAVORS:
            self.assertLess(
                result[flavor][
                    "exact_union_envelope_plus_finite_ensemble_allowance"
                ],
                result[flavor]["immutable_lambda1_width"],
            )

    def test_excess_checkpoint_movement_is_charged_and_can_reject(self) -> None:
        terminal = bands(0.105)
        anchor = bands(0.07, shift=0.01)
        result, passed = audit.checkpoint_outcome_metrics(
            terminal,
            anchor,
            incumbent(0.11),
            {"u": 0.005, "d": 0.005},
            {"u": True, "d": True},
        )
        self.assertFalse(passed)
        self.assertGreater(
            result["u"]["tail_envelope_increment_beyond_terminal_width"], 0.0
        )
        self.assertFalse(result["u"]["promotion_validation_gate_pass"])

    def test_changed_incumbent_replacement_verdict_fails_closed(self) -> None:
        terminal = bands(0.08)
        anchor = bands(0.13)
        result, passed = audit.checkpoint_outcome_metrics(
            terminal,
            anchor,
            incumbent(0.11),
            {"u": 0.001, "d": 0.001},
            {"u": True, "d": True},
        )
        self.assertFalse(passed)
        self.assertFalse(
            result["u"][
                "incumbent_replacement_verdict_unchanged_across_checkpoint"
            ]
        )


class TransformAndCentralTests(unittest.TestCase):
    def test_batched_transform_uses_exact_frozen_extension_and_kernel(self) -> None:
        engine = audit.ExactTransformEngine(transformer)
        b = np.linspace(0.0001, 8.0, 321)
        curves = np.asarray(
            [np.exp(-0.2 * b**2), 1.1 * np.exp(-0.3 * b**2)]
        )
        for mode in audit.TAIL_MODES:
            observed = engine.transform(b, curves, mode, batch_size=1)
            expected = []
            for curve in curves:
                extended = transformer.extend_curve(
                    b,
                    curve,
                    engine.b_grid,
                    tail_mode=mode,
                    tail_fit_bmin=None,
                    eps=1.0e-300,
                )
                expected.append(engine.kernel @ extended)
            self.assertTrue(
                np.allclose(observed, np.asarray(expected), rtol=2e-13, atol=2e-13)
            )

    def test_declared_central_containment_is_exact_not_percent_gated(self) -> None:
        frame = bands(0.10)
        contained = audit.containment(frame, "kT", 2.25)
        self.assertTrue(audit.all_contained(contained))
        frame.loc[0, "declared_central"] = frame.loc[0, "q84"] + 0.01
        outside = audit.containment(frame, "kT", 2.25)
        self.assertFalse(audit.all_contained(outside))

    def test_only_expb2_is_the_declared_gating_tail_mode(self) -> None:
        self.assertEqual(audit.TAIL_MODES, ("expb2", "expb", "taper"))
        text = (HERE / "audit_lambda600_postfit_tail_transform.py").read_text()
        self.assertIn('"locked_gating_tail_mode": "expb2"', text)
        self.assertIn('"alternate_tail_modes_gate_promotion": True', text)
        self.assertIn(
            '"alternate_tail_modes_can_replace_or_loosen_locked_expb2_width_gate": False',
            text,
        )
        self.assertIn('"new_arbitrary_percent_tolerance_used": False', text)


class ExactCheckpointEvidenceTests(unittest.TestCase):
    @staticmethod
    def fixture(root: Path) -> dict:
        k = np.linspace(0.0, 4.0, 401)
        starts = np.linspace(-0.02, 0.02, 24)[None, :, None, None]
        replicas = np.linspace(-0.01, 0.01, 50)[None, None, :, None]
        base = np.asarray([3.0, 2.0])[:, None, None, None] * np.exp(
            -k[None, None, None, :] ** 2
        )
        terminal = (base * (1.0 + starts + replicas)).astype(np.float64)
        anchor = (base * (1.0 + 0.8 * starts + 1.1 * replicas)).astype(np.float64)
        central = base[:, 0, 0].astype(np.float64)
        npz = root / "exact.npz"
        np.savez_compressed(
            npz,
            schema=np.asarray(validation.EXACT_ARRAY_SCHEMA),
            checkpoints=np.asarray(("terminal", "stationarity_anchor")),
            flavors=np.asarray(("u", "d")),
            start_seeds=np.arange(303, 327, dtype=np.int64),
            replica_seeds=np.arange(1001, 1051, dtype=np.int64),
            kT=k,
            terminal_values=terminal,
            stationarity_anchor_values=anchor,
            terminal_declared_central=central,
            stationarity_anchor_declared_central=central,
        )
        rows = []
        for checkpoint, values in (
            ("terminal", terminal), ("stationarity_anchor", anchor)
        ):
            for flavor_index, flavor in enumerate(("u", "d")):
                quantiles = np.quantile(
                    values[flavor_index].reshape(1200, 401),
                    (0.16, 0.50, 0.84), axis=0,
                )
                rows.append(pd.DataFrame({
                    "model": "lambda600", "checkpoint": checkpoint,
                    "tail_mode": "expb2", "flavor": flavor, "kT": k,
                    "q16": quantiles[0], "median": quantiles[1],
                    "q84": quantiles[2], "declared_central": central[flavor_index],
                }))
        bands_path = root / "bands.csv"
        pd.concat(rows, ignore_index=True).to_csv(bands_path, index=False)
        return {
            "artifacts": {
                "exact_checkpoint_expb2_members": str(npz),
                "kspace_checkpoint_expb2_bands": str(bands_path),
            },
            "exact_checkpoint_transform_arrays": {
                "schema": validation.EXACT_ARRAY_SCHEMA,
                "shape": [2, 24, 50, 401],
                "axis_order": ["flavor", "start_seed", "replica_seed", "kT"],
                "checkpoints": ["terminal", "stationarity_anchor"],
                "flavors": ["u", "d"],
                "start_seed_range": [303, 326],
                "replica_seed_range": [1001, 1050],
                "quantiles_declared_from_exact_arrays": True,
            },
        }

    def test_hash_bound_exact_arrays_reproduce_declared_checkpoint_bands(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = self.fixture(Path(directory))
            observed = validation.validated_exact_checkpoint_transform_arrays(payload)
            self.assertEqual(observed["terminal_values"].shape, (2, 24, 50, 401))
            bands_path = Path(payload["artifacts"]["kspace_checkpoint_expb2_bands"])
            bands_frame = pd.read_csv(bands_path)
            bands_frame.loc[0, "q16"] += 1.0e-3
            bands_frame.to_csv(bands_path, index=False)
            with self.assertRaisesRegex(RuntimeError, "differ from exact arrays"):
                validation.validated_exact_checkpoint_transform_arrays(payload)

    def test_postfit_gate_is_exact_component_conjunction(self) -> None:
        payload = {
            "coverage_gate_pass": True,
            "candidate_stationarity_gate_pass": True,
            "central_line_containment_gate_pass": True,
            "transform_decision_robustness_gate_pass": True,
        }
        self.assertTrue(validation.reconstructed_postfit_promotion_gate(payload))
        payload["candidate_stationarity_gate_pass"] = False
        self.assertFalse(validation.reconstructed_postfit_promotion_gate(payload))


if __name__ == "__main__":
    unittest.main(verbosity=2)
