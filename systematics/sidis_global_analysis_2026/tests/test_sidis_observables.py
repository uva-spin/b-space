import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sidis_observables import RadialConvolutionConvention, SidisKinematics, multiplicity_ratio, uu_structure_function  # noqa: E402


class SidisObservableTests(unittest.TestCase):
    def test_kinematics_and_qt_mapping(self):
        SidisKinematics(0.1, 2.0, 0.3, 0.4, 0.5).validate()
        np.testing.assert_allclose(RadialConvolutionConvention().qT(0.4, 0.3), 4 / 3)

    def test_flavor_weighted_radial_convolution(self):
        b = np.linspace(0, 8, 4001)
        constant = lambda grid, _a, _q: np.exp(-grid**2)
        result = uu_structure_function(b, np.array([0.0, 0.3]), 0.5, {"u": constant, "d": constant}, {"u": constant, "d": constant}, {"u": 4 / 9, "d": 1 / 9}, x=0.1, q2=2.0)
        self.assertEqual(result.shape, (2,))
        self.assertGreater(result[0], result[1])

    def test_ratio_rejects_zero_denominator(self):
        with self.assertRaises(ValueError):
            multiplicity_ratio(1.0, 0.0)


if __name__ == "__main__":
    unittest.main()
