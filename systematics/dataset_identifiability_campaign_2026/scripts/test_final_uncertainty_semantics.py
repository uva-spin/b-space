#!/usr/bin/env python3
"""Focused fail-closed tests for final-band probability and promotion semantics."""

from __future__ import annotations

import importlib.util
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
BASE = HERE.parent
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


audit = load_module("tested_final_ensemble_audit", "audit_final_combined_ensemble.py")
renderer = load_module("tested_final_renderer", "plot_validated_final_fig2_fig6.py")
comparison = load_module(
    "tested_lambda600_lambda1_comparison", "plot_lambda600_vs_lambda1_diagnostic.py"
)


class LockedPromotionGateTests(unittest.TestCase):
    def test_manifest_and_code_pin_the_same_flavor_specific_thresholds(self) -> None:
        protocol = json.loads(
            (BASE / "manifests/lambda600_fixed_challenger_protocol.json").read_text()
        )
        expected = protocol["incumbent"]["strict_max_active_relative_full_width"]
        self.assertEqual(set(expected), {"u", "d"})
        for flavor in ("u", "d"):
            self.assertAlmostEqual(
                float(expected[flavor]), audit.LOCKED_INCUMBENT_WIDTHS[flavor], places=14
            )

    def test_candidate_dependent_union_width_cannot_loosen_locked_gate(self) -> None:
        candidate_width = 0.13
        candidate_allowance = 0.0
        enlarged_union_incumbent_width = 0.35
        self.assertLess(candidate_width + candidate_allowance,
                        enlarged_union_incumbent_width)
        self.assertFalse(
            audit.beats_locked_width(
                candidate_width,
                candidate_allowance,
                audit.LOCKED_INCUMBENT_WIDTHS["u"],
            )
        )

    def test_u_and_d_decisions_are_separate(self) -> None:
        observed = {
            "u": audit.beats_locked_width(0.10, 0.01,
                                           audit.LOCKED_INCUMBENT_WIDTHS["u"]),
            "d": audit.beats_locked_width(0.12, 0.01,
                                           audit.LOCKED_INCUMBENT_WIDTHS["d"]),
        }
        self.assertEqual(observed, {"u": True, "d": False})

    def test_invalid_width_inputs_fail_closed(self) -> None:
        for values in ((np.nan, 0.0, 0.1), (0.1, -0.01, 0.2), (0.1, 0.0, 0.0)):
            with self.assertRaises(RuntimeError):
                audit.beats_locked_width(*values)


class DirectWidthStatisticTests(unittest.TestCase):
    def test_statistic_recomputes_quantiles_and_sample_active_mask(self) -> None:
        kvals = np.asarray([0.0, 1.0, 2.0])
        sample = np.asarray([
            [[8.0, 0.8, 0.002]],
            [[9.0, 0.9, 0.006]],
            [[10.0, 1.0, 0.010]],
            [[11.0, 1.1, 0.014]],
            [[12.0, 1.2, 0.018]],
        ])
        incumbent_active = np.asarray([[False, False, True]])
        result = audit.flavor_local_width_statistic(
            sample, kvals, incumbent_active
        )
        expected_q = np.quantile(sample, [.16, .50, .84], axis=0)
        expected_curve = ((expected_q[2, 0] - expected_q[0, 0]) /
                          np.maximum(expected_q[1, 0], 1e-30))
        self.assertTrue(np.allclose(result["quantiles"], expected_q))
        self.assertEqual(
            result["candidate_active"].tolist(), [[True, True, False]]
        )
        self.assertEqual(result["union_active"].tolist(), [[True, True, True]])
        self.assertAlmostEqual(
            result["max_relative_full_width"][0],
            float(np.max(expected_curve)),
        )

    def test_each_sample_recomputes_its_own_five_percent_mask(self) -> None:
        kvals = np.asarray([0.0, 1.0, 2.0])
        incumbent_active = np.zeros((1, 3), dtype=bool)
        above = np.repeat(np.asarray([[[10.0, 0.6, 0.1]]]), 5, axis=0)
        below = np.repeat(np.asarray([[[10.0, 0.4, 0.1]]]), 5, axis=0)
        above_result = audit.flavor_local_width_statistic(
            above, kvals, incumbent_active
        )
        below_result = audit.flavor_local_width_statistic(
            below, kvals, incumbent_active
        )
        self.assertTrue(above_result["candidate_active"][0, 1])
        self.assertFalse(below_result["candidate_active"][0, 1])

    def test_split_difference_is_difference_of_exact_width_statistics(self) -> None:
        kvals = np.asarray([0.0, 1.0, 2.0])
        incumbent_active = np.asarray([[True, False, False]])
        first = np.asarray([[[8.0, 1.0, 0.1]], [[9.0, 1.0, 0.1]],
                            [[11.0, 1.0, 0.1]], [[12.0, 1.0, 0.1]]])
        second = np.asarray([[[9.5, 1.0, 0.1]], [[9.8, 1.0, 0.1]],
                             [[10.2, 1.0, 0.1]], [[10.5, 1.0, 0.1]]])
        first_width = audit.flavor_local_width_statistic(
            first, kvals, incumbent_active
        )["max_relative_full_width"]
        second_width = audit.flavor_local_width_statistic(
            second, kvals, incumbent_active
        )["max_relative_full_width"]
        observed = audit.absolute_width_statistic_difference(
            first, second, kvals, incumbent_active
        )
        self.assertTrue(np.allclose(observed, np.abs(first_width - second_width)))

    def test_allowance_is_direct_maximum_without_bootstrap_factor(self) -> None:
        bootstrap = np.asarray([[0.01, 0.02], [0.03, 0.04], [0.05, 0.06]])
        start = np.asarray([[0.005, 0.01], [0.01, 0.02], [0.015, 0.03]])
        replica = np.asarray([[0.004, 0.008], [0.008, 0.016], [0.012, 0.024]])
        joint = np.asarray([[0.006, 0.012], [0.012, 0.024], [0.018, 0.036]])
        result = audit.direct_resampling_allowance(
            bootstrap, start, replica, joint
        )
        expected_bootstrap = np.quantile(bootstrap, .95, axis=0)
        self.assertTrue(np.allclose(result["bootstrap_p95"], expected_bootstrap))
        self.assertTrue(np.allclose(result["allowance"], expected_bootstrap))
        self.assertTrue(np.all(result["allowance"] < 2 * expected_bootstrap))

    def test_stored_combined_band_must_match_long_quantiles_pointwise(self) -> None:
        flavors = ["u", "d"]
        kvals = np.asarray([0.0, 1.0, 2.0])
        sample = np.arange(1.0, 1.0 + 8 * 2 * 3).reshape(8, 2, 3)
        quantiles = tuple(np.quantile(sample, q, axis=0)
                          for q in (.16, .50, .84))
        rows = []
        for fi, flavor in enumerate(flavors):
            for ki, k_t in enumerate(kvals):
                rows.append({
                    "component": "combined", "quantity": "ftilde",
                    "Q": 10.0, "flavor": flavor, "kT": k_t,
                    "q16": quantiles[0][fi, ki],
                    "median": quantiles[1][fi, ki],
                    "q84": quantiles[2][fi, ki],
                })
        bands = pd.DataFrame(rows)
        result = audit.validate_stored_combined_kspace_quantiles(
            bands, flavors, kvals, quantiles
        )
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["compared_value_count"], 18)
        corrupted = bands.copy()
        corrupted.loc[
            (corrupted.flavor == "d") & np.isclose(corrupted.kT, 1.0), "q84"
        ] += 1e-6
        with self.assertRaisesRegex(RuntimeError, "disagree with long data"):
            audit.validate_stored_combined_kspace_quantiles(
                corrupted, flavors, kvals, quantiles
            )

    def test_stored_combined_band_missing_row_fails_closed(self) -> None:
        flavors = ["u", "d"]
        kvals = np.asarray([0.0, 1.0, 2.0])
        quantiles = tuple(np.ones((2, 3)) * value
                          for value in (0.9, 1.0, 1.1))
        rows = [
            {"component": "combined", "quantity": "ftilde", "Q": 10.0,
             "flavor": flavor, "kT": k_t, "q16": 0.9,
             "median": 1.0, "q84": 1.1}
            for flavor in flavors for k_t in kvals
        ]
        with self.assertRaisesRegex(RuntimeError, "exact renderer-row coverage"):
            audit.validate_stored_combined_kspace_quantiles(
                pd.DataFrame(rows[:-1]), flavors, kvals, quantiles
            )


class DirectionalComparisonFailureTests(unittest.TestCase):
    @staticmethod
    def fixture(root: Path, *, promotion: bool, invert: bool = False) -> dict:
        b = np.linspace(0.0001, 8.0, 321)
        k = np.linspace(0.0, 4.0, 401)
        fnp = pd.DataFrame({
            "x": 0.1,
            "bT": b,
            "final_envelope_low": 0.90,
            "trained_central": 1.00,
            "final_envelope_high": 1.10,
        })
        # This is the scientific failure the comparison must display rather
        # than reclassify as a technical failure.
        fnp.loc[10, "trained_central"] = 1.20
        if invert:
            fnp.loc[11, "final_envelope_low"] = 1.30
        bspace = pd.concat([
            pd.DataFrame({
                "Q": 7.5,
                "flavor": flavor,
                "bT": b,
                "final_envelope_low": 0.90,
                "trained_central": 1.00,
                "final_envelope_high": 1.10,
            })
            for flavor in comparison.FLAVORS_B
        ], ignore_index=True)
        kspace = pd.concat([
            pd.DataFrame({
                "Q": 10.0,
                "flavor": flavor,
                "kT": k,
                "final_envelope_low": 0.90,
                "trained_central": 1.00,
                "final_envelope_high": 1.10,
            })
            for flavor in comparison.FLAVORS_K
        ], ignore_index=True)
        paths = {
            "fnp_final_envelope": root / "fnp.csv",
            "fig2_bspace_final_envelope": root / "bspace.csv",
            "fig6_kspace_final_envelope": root / "kspace.csv",
        }
        fnp.to_csv(paths["fnp_final_envelope"], index=False)
        bspace.to_csv(paths["fig2_bspace_final_envelope"], index=False)
        kspace.to_csv(paths["fig6_kspace_final_envelope"], index=False)
        return {
            "promotion_validation_gate_pass": promotion,
            "artifacts": {key: str(path) for key, path in paths.items()},
        }

    def test_diagnostic_comparison_preserves_failed_central_containment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary = self.fixture(Path(directory), promotion=False)
            fnp, bspace, kspace = comparison.validate_final_envelope_tables(summary)
        self.assertGreater(float(fnp.loc[10, "central"]),
                           float(fnp.loc[10, "q84"]))
        self.assertEqual(len(bspace), 6 * 321)
        self.assertEqual(len(kspace), 2 * 401)

    def test_promotable_comparison_requires_central_containment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary = self.fixture(Path(directory), promotion=True)
            with self.assertRaisesRegex(RuntimeError, "does not contain"):
                comparison.validate_final_envelope_tables(summary)

    def test_diagnostic_comparison_still_rejects_inverted_interval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary = self.fixture(Path(directory), promotion=False, invert=True)
            with self.assertRaisesRegex(RuntimeError, "inverted"):
                comparison.validate_final_envelope_tables(summary)


class EmpiricalIntervalSemanticsTests(unittest.TestCase):
    def test_validated_artifact_names_make_no_sigma_claim(self) -> None:
        for stem in renderer.VALIDATED_STEMS.values():
            self.assertIn("product_plus_directional_envelope", stem)
            self.assertNotIn("sigma", stem.lower())
            self.assertNotIn("68", stem)

    def test_renderer_metadata_disclaims_formal_probability(self) -> None:
        for diagnostic in (False, True):
            metadata = renderer.interval_metadata(diagnostic)
            self.assertFalse(metadata["formal_confidence_level_assigned"])
            self.assertFalse(metadata["one_sigma_claimed"])
            self.assertIn(
                "empirical product band plus residual convergence/interaction envelope",
                metadata["uncertainty"],
            )
            self.assertIn("no calibrated probability law",
                          metadata["probability_semantics"])

    def test_negative_transform_lobe_is_not_an_active_signal_region(self) -> None:
        rows = []
        for method in ("lambda600_candidate", "lambda1_harmonized"):
            for k_t, central in ((0.0, 1.0), (1.0, 0.10), (2.0, -0.20)):
                rows.append({
                    "method": method,
                    "flavor": "u",
                    "kT": k_t,
                    "central": central,
                })
        marked = comparison.annotate_masks(pd.DataFrame(rows), "kT")
        tail = marked[np.isclose(marked["kT"], 2.0)]
        self.assertFalse(tail["own_active_mask"].any())
        self.assertFalse(tail["comparison_union_active_mask"].any())

    def test_renderer_publishes_validated_and_diagnostic_generations_safely(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            target = root / "figures"
            source.mkdir()
            audit_path = root / "audit.json"
            audit_path.write_text(json.dumps({
                "endpoint_gate_pass": True,
                "diagnostic_figure_gate_pass": True,
                "candidate_stationarity_gate_pass": True,
                "scientific_failure_reasons": [],
            }))
            grid = np.asarray([0.0, 1.0, 2.0])
            pd.DataFrame({
                "component": "combined", "x": 0.1, "bT": grid,
                "q16": [0.9, 0.7, 0.4], "median": [1.0, 0.8, 0.5],
                "q84": [1.1, 0.9, 0.6],
            }).to_csv(source / "fnp_bands.csv", index=False)
            b_rows = []
            for flavor in renderer.COLORS:
                for b_t, center in zip(grid, (1.0, 0.8, 0.5)):
                    b_rows.append({
                        "component": "combined", "Q": 7.5, "flavor": flavor,
                        "bT": b_t, "q16": 0.9 * center, "median": center,
                        "q84": 1.1 * center,
                    })
            pd.DataFrame(b_rows).to_csv(source / "bT_tmd_bands.csv", index=False)
            k_rows = []
            for flavor in ("u", "d"):
                for k_t, center in zip(grid, (3.0, 1.0, 0.2)):
                    k_rows.append({
                        "component": "combined", "Q": 10.0, "flavor": flavor,
                        "kT": k_t, "q16": 0.9 * center, "median": center,
                        "q84": 1.1 * center,
                    })
            pd.DataFrame(k_rows).to_csv(source / "kT_tmd_bands.csv", index=False)

            envelope_root = root / "final_envelope"
            envelope_root.mkdir()
            fnp_envelope = envelope_root / "fnp_final_envelope.csv"
            pd.DataFrame({
                "x": 0.1, "bT": grid,
                "final_envelope_low": [0.9, 0.7, 0.4],
                "trained_central": [1.0, 0.8, 0.5],
                "final_envelope_high": [1.1, 0.9, 0.6],
            }).to_csv(fnp_envelope, index=False)
            b_envelope = envelope_root / "fig2_bspace_final_envelope.csv"
            pd.DataFrame([
                {"Q": 7.5, "flavor": flavor, "bT": b_t,
                 "final_envelope_low": 0.9 * center,
                 "trained_central": center,
                 "final_envelope_high": 1.1 * center}
                for flavor in renderer.COLORS
                for b_t, center in zip(grid, (1.0, 0.8, 0.5))
            ]).to_csv(b_envelope, index=False)
            k_envelope = envelope_root / "fig6_kspace_final_envelope.csv"
            pd.DataFrame([
                {"Q": 10.0, "flavor": flavor, "kT": k_t,
                 "final_envelope_low": 0.9 * center,
                 "trained_central": center,
                 "final_envelope_high": 1.1 * center}
                for flavor in ("u", "d")
                for k_t, center in zip(grid, (3.0, 1.0, 0.2))
            ]).to_csv(k_envelope, index=False)
            final_envelope_path = envelope_root / "summary.json"
            final_payload = {
                "diagnostic_figure_gate_pass": True,
                "promotion_validation_gate_pass": True,
                "scientific_failure_reasons": [],
                "production_sources_modified": False,
                "artifacts": {
                    "fnp_final_envelope": str(fnp_envelope),
                    "fig2_bspace_final_envelope": str(b_envelope),
                    "fig6_kspace_final_envelope": str(k_envelope),
                },
            }
            final_envelope_path.write_text(json.dumps(final_payload))

            old = (
                renderer.SOURCE,
                renderer.AUDIT,
                renderer.FINAL_ENVELOPE,
                renderer.TARGET,
                renderer.validated_final_directional_envelope,
                renderer.final_promotion_gate,
            )
            try:
                (
                    renderer.SOURCE,
                    renderer.AUDIT,
                    renderer.FINAL_ENVELOPE,
                    renderer.TARGET,
                ) = (
                    source, audit_path, final_envelope_path, target
                )
                renderer.validated_final_directional_envelope = (
                    lambda *_args, **_kwargs: (final_payload, "fixture-sha256")
                )
                renderer.final_promotion_gate = lambda _payload: True
                with contextlib.redirect_stdout(io.StringIO()):
                    renderer.main()

                for stem in renderer.VALIDATED_STEMS.values():
                    self.assertTrue((target / f"{stem}.png").is_file())
                    self.assertTrue((target / f"{stem}.pdf").is_file())
                self.assertFalse(any("sigma" in path.name.lower()
                                     for path in target.iterdir()))
                summary = json.loads((target / "summary.json").read_text())
                self.assertFalse(summary["formal_confidence_level_assigned"])
                self.assertFalse(summary["one_sigma_claimed"])
                self.assertIn("experimental replicas", summary["displayed_band"])
                self.assertIn("nonuniqueness", summary["displayed_band"])

                # Exercise the scientific-failure renderer directly.  A
                # completed diagnostic generation must replace the manifest,
                # publish its own exact names, and retire only the stale
                # validated names from the opposite outcome class.
                final_payload["promotion_validation_gate_pass"] = False
                final_payload["scientific_failure_reasons"] = [
                    "trained central containment gate failed"
                ]
                renderer.final_promotion_gate = lambda _payload: False
                with contextlib.redirect_stdout(io.StringIO()):
                    renderer.main()

                for stem in renderer.DIAGNOSTIC_STEMS.values():
                    self.assertTrue((target / f"{stem}.png").is_file())
                    self.assertTrue((target / f"{stem}.pdf").is_file())
                for stem in renderer.VALIDATED_STEMS.values():
                    self.assertFalse((target / f"{stem}.png").exists())
                    self.assertFalse((target / f"{stem}.pdf").exists())
                self.assertFalse(any(".tmp." in path.name for path in target.iterdir()))
                summary = json.loads((target / "summary.json").read_text())
                self.assertEqual(summary["status"],
                                 "diagnostic_figures_not_promotable")
                self.assertTrue(summary["diagnostic_only"])
                self.assertFalse(summary["one_sigma_claimed"])
            finally:
                (
                    renderer.SOURCE,
                    renderer.AUDIT,
                    renderer.FINAL_ENVELOPE,
                    renderer.TARGET,
                    renderer.validated_final_directional_envelope,
                    renderer.final_promotion_gate,
                ) = old

if __name__ == "__main__":
    unittest.main(verbosity=2)
