import unittest

import numpy as np

from systematics.finite_y_completion_2026.backend.unitary_finite_y import (
    smootherstep_profile,
    unitary_matched,
    unitary_y,
)


class UnitaryFiniteYTests(unittest.TestCase):
    def test_profile_endpoints_and_c2_shape(self):
        self.assertEqual(smootherstep_profile(0.19), 0.0)
        self.assertEqual(smootherstep_profile(0.31), 1.0)
        self.assertAlmostEqual(smootherstep_profile(0.25), 0.5)
        h = 1e-5
        for endpoint in (0.20, 0.30):
            left = smootherstep_profile(endpoint - h)
            center = smootherstep_profile(endpoint)
            right = smootherstep_profile(endpoint + h)
            self.assertLess(abs((right - left) / (2*h)), 1e-5)
            self.assertLess(abs((right - 2*center + left) / h**2), 0.2)

    def test_core_and_fixed_order_limits(self):
        w = np.array([2.0, 2.0, 2.0])
        fo = np.array([8.0, 8.0, 8.0])
        r = np.array([0.19, 0.25, 0.31])
        np.testing.assert_allclose(unitary_y(w=w, fixed_order=fo, r=r), [0.0, 3.0, 6.0])
        np.testing.assert_allclose(unitary_matched(w=w, fixed_order=fo, r=r), [2.0, 5.0, 8.0])

    def test_convexity(self):
        r = np.linspace(0.18, 0.32, 101)
        out = unitary_matched(w=2.0, fixed_order=8.0, r=r)
        self.assertTrue(np.all(out >= 2.0))
        self.assertTrue(np.all(out <= 8.0))

    def test_rejects_nonfinite(self):
        with self.assertRaises(ValueError):
            unitary_y(w=[1.0, np.nan], fixed_order=[2.0, 3.0], r=[0.2, 0.3])


if __name__ == "__main__":
    unittest.main()
