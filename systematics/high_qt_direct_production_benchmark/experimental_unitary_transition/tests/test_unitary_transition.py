import unittest

import numpy as np

from systematics.high_qt_direct_production_benchmark.experimental_unitary_transition.backend.unitary_transition import (
    bin_averaged_profile,
    smootherstep_profile,
    unitary_transition,
)


class UnitaryTransitionTests(unittest.TestCase):
    def test_endpoints_and_midpoint(self):
        self.assertEqual(smootherstep_profile(0.19), 0.0)
        self.assertEqual(smootherstep_profile(0.31), 1.0)
        self.assertAlmostEqual(smootherstep_profile(0.25), 0.5)

    def test_c2_endpoint_behavior(self):
        h = 1.0e-5
        for endpoint in (0.20, 0.30):
            center = smootherstep_profile(endpoint)
            left = smootherstep_profile(endpoint - h)
            right = smootherstep_profile(endpoint + h)
            self.assertLess(abs((right - left) / (2.0 * h)), 1.0e-5)
            # The centered stencil straddles the clipped endpoint, leaving an
            # O(h) third-derivative remainder; it must still tend to zero.
            self.assertLess(abs((right - 2.0 * center + left) / h**2), 0.2)

    def test_unitary_limits_and_convexity(self):
        self.assertEqual(unitary_transition(2.0, 8.0, 0.0), 2.0)
        self.assertEqual(unitary_transition(2.0, 8.0, 1.0), 8.0)
        values = unitary_transition(np.array([2.0, 2.0]), 8.0, np.array([0.25, 0.75]))
        np.testing.assert_allclose(values, [3.5, 6.5])

    def test_rejects_invalid_profile(self):
        with self.assertRaises(ValueError):
            unitary_transition(1.0, 2.0, 1.01)

    def test_bin_average_symmetry(self):
        # A symmetric bin around the midpoint of a symmetric smootherstep averages to 1/2.
        self.assertAlmostEqual(bin_averaged_profile(20.0, 30.0, 100.0), 0.5, places=12)


if __name__ == "__main__":
    unittest.main()
