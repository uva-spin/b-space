from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sidis_covariance import correlated_chi2, covariance_condition_number, whiten_residual  # noqa: E402


class SidisCovarianceTests(unittest.TestCase):
    def test_correlated_quadratic_form(self):
        covariance = np.array([[4.0, 1.0], [1.0, 3.0]])
        residual = np.array([2.0, -1.0])
        expected = float(residual @ np.linalg.solve(covariance, residual))
        self.assertAlmostEqual(correlated_chi2(residual, covariance), expected)
        whitened = whiten_residual(residual, covariance)
        self.assertAlmostEqual(float(whitened @ whitened), expected)

    def test_indefinite_or_singular_covariance_fails_closed(self):
        with self.assertRaises(ValueError):
            correlated_chi2(np.ones(2), np.array([[1.0, 1.0], [1.0, 1.0]]))
        with self.assertRaises(ValueError):
            correlated_chi2(np.ones(2), np.array([[1.0, 2.0], [2.0, 1.0]]))

    def test_shape_and_symmetry_are_checked(self):
        with self.assertRaises(ValueError):
            correlated_chi2(np.ones(2), np.eye(3))
        with self.assertRaises(ValueError):
            correlated_chi2(np.ones(2), np.array([[1.0, 0.2], [0.0, 1.0]]))

    def test_condition_number_is_finite(self):
        self.assertGreater(covariance_condition_number(np.eye(2)), 0.0)
