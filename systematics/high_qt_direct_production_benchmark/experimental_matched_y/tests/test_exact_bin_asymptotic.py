import math
import unittest

import numpy as np
import pandas as pd

from systematics.high_qt_direct_production_benchmark.experimental_matched_y.backend.exact_bin_asymptotic import (
    integrate_exact_bin,
    make_resummed_w_point_evaluators,
    node_row,
    rapidity_interval,
)


class ExactBinQuadratureTests(unittest.TestCase):
    def setUp(self):
        self.row = pd.Series({
            "dataset": "TEST", "target": "pbar_p", "QM": 10.0, "SqrtS": 100.0,
            "qT_low": 2.0, "qT_high": 4.0, "qT": 3.0,
            "y_Low": -1.0, "y_High": 2.0, "observable_name": "inclusive",
        })

    def test_polynomial_quadrature_and_bin_average(self):
        # Integral_y[-1,2] (1+y) = 4.5; average_qT[2,4] qT^2 = 28/3.
        result = integrate_exact_bin(
            self.row, point_evaluator=lambda r: float(r.qT) ** 2 * (1.0 + float(r.y)),
            n_qT=3, n_y=2,
        )
        self.assertAlmostEqual(result.value_pb_per_GeV, 42.0, places=12)

    def test_node_uses_leading_power_x_and_removes_bin_jacobian(self):
        current = node_row(self.row, qT=3.25, y=0.7)
        self.assertAlmostEqual(current.x1, 0.1 * math.exp(0.7))
        self.assertAlmostEqual(current.x2, 0.1 * math.exp(-0.7))
        self.assertTrue(np.isnan(current.qT_low))
        self.assertTrue(np.isnan(current.qT_high))

    def test_rapidity_defaults_to_leading_power_kinematic_range(self):
        row = self.row.copy()
        row["y_Low"] = np.nan
        row["y_High"] = np.nan
        low, high = rapidity_interval(row)
        self.assertAlmostEqual(low, -math.log(10.0))
        self.assertAlmostEqual(high, math.log(10.0))

    def test_pp_fiducial_row_requires_explicit_acceptance(self):
        row = self.row.copy()
        row["target"] = "pp"
        with self.assertRaises(ValueError):
            integrate_exact_bin(row, point_evaluator=lambda _: 1.0)

    def test_explicit_acceptance_is_applied(self):
        row = self.row.copy()
        row["target"] = "pp"
        result = integrate_exact_bin(
            row, point_evaluator=lambda _: 2.0, acceptance=lambda _: 0.25,
            n_qT=2, n_y=2,
        )
        # Constant 0.5 integrated over y width 3; qT averaging leaves 1.5.
        self.assertAlmostEqual(result.value_pb_per_GeV, 1.5, places=12)

    def test_resummed_evaluators_share_kernel_and_apply_pair_factor(self):
        class Backend:
            calls = 0

            @staticmethod
            def make_b_grid(_cfg):
                return np.array([0.0, 1.0, 2.0])

            @classmethod
            def wpert_cs_for_row(cls, _row, b_grid, _pdf, _cfg):
                cls.calls += 1
                return np.ones_like(b_grid)

        row = pd.Series({"qT": 0.0, "y": 0.0, "x1": 0.2, "x2": 0.3})
        perturbative, fitted = make_resummed_w_point_evaluators(
            backend=Backend, pdf=object(), cfg=object(),
            np_pair_factor=lambda _x1, _x2, b: np.full_like(b, 2.0),
        )
        self.assertAlmostEqual(perturbative(row), 2.0)
        self.assertAlmostEqual(fitted(row), 4.0)
        self.assertEqual(Backend.calls, 1)

    def test_resummed_evaluator_can_remove_inclusive_rapidity_factor(self):
        class Backend:
            @staticmethod
            def make_b_grid(_cfg):
                return np.array([0.0, 1.0, 2.0])

            @staticmethod
            def wpert_cs_for_row(_row, b_grid, _pdf, _cfg):
                return np.full_like(b_grid, 3.0)

            @staticmethod
            def _tevatron_rapidity_factor(_row):
                return 3.0

        row = pd.Series({"qT": 0.0, "y": 0.0, "x1": 0.2, "x2": 0.3})
        perturbative, _ = make_resummed_w_point_evaluators(
            backend=Backend, pdf=object(), cfg=object(),
            remove_inclusive_rapidity_approximation=True,
        )
        self.assertAlmostEqual(perturbative(row), 2.0)


if __name__ == "__main__":
    unittest.main()
