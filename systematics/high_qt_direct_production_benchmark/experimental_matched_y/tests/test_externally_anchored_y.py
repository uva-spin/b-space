import unittest

import numpy as np

from systematics.high_qt_direct_production_benchmark.experimental_matched_y.backend.externally_anchored_y import (
    compose_additive_matched,
)


class AdditiveMatchingTests(unittest.TestCase):
    def test_matching_closes_to_fixed_order_when_w_equals_asymptotic(self):
        w = np.array([2.0, 3.0, 4.0])
        fo = np.array([2.5, 2.7, 5.0])
        result = compose_additive_matched(w=w, fixed_order=fo, asymptotic=w, profile=np.ones(3))
        np.testing.assert_allclose(result.matched, fo, rtol=0.0, atol=0.0)

    def test_matching_returns_w_when_fixed_order_equals_asymptotic(self):
        w = np.array([2.0, 3.0])
        common = np.array([1.0, 4.0])
        result = compose_additive_matched(w=w, fixed_order=common, asymptotic=common, profile=np.ones(2))
        np.testing.assert_allclose(result.matched, w, rtol=0.0, atol=0.0)

    def test_zero_profile_returns_w_and_unit_profile_returns_full_y(self):
        w = np.array([1.0, 1.0])
        fo = np.array([3.0, 5.0])
        asym = np.array([2.0, 2.0])
        zero = compose_additive_matched(w=w, fixed_order=fo, asymptotic=asym, profile=np.zeros(2))
        full = compose_additive_matched(w=w, fixed_order=fo, asymptotic=asym, profile=np.ones(2))
        np.testing.assert_allclose(zero.matched, w)
        np.testing.assert_allclose(full.y, fo - asym)

    def test_invalid_profiles_are_rejected(self):
        for profile in (np.array([-0.1]), np.array([1.1]), np.array([np.nan])):
            with self.subTest(profile=profile):
                with self.assertRaises(ValueError):
                    compose_additive_matched(w=[1.0], fixed_order=[1.0], asymptotic=[1.0], profile=profile)

    def test_shape_mismatch_is_rejected_without_broadcasting(self):
        with self.assertRaises(ValueError):
            compose_additive_matched(w=[1.0, 2.0], fixed_order=[1.0], asymptotic=[1.0], profile=[1.0])


if __name__ == "__main__":
    unittest.main()
