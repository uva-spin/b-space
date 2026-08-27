from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sidis_ff import LHAPDFFamily, LHAPDFFMember  # noqa: E402


class FakeSet:
    size = 3
    errorType = "replicas"

    def __init__(self):
        self.values = {
            "SetType": "fragfn",
            "NumMembers": "3",
            "OrderQCD": "2",
            "QMin": "1.0",
            "QMax": "100.0",
            "XMin": "0.01",
            "XMax": "1.0",
            "ErrorType": "replicas",
            "Flavors": [-1, 1, 21],
        }

    def get_entry(self, key):
        return self.values[key]


class FakePDF:
    def xfxQ(self, pid, z, q):
        return float(pid + 2) * z * (1.0 + q / 100.0)


class SidisFFTests(unittest.TestCase):
    def test_xfxq_is_converted_to_density_and_arrays_are_supported(self):
        member = LHAPDFFMember("fake", 0, backend=FakePDF(), pdfset=FakeSet())
        self.assertAlmostEqual(member.density(1, 0.25, 2.0), 3.0 * 1.02)
        np.testing.assert_allclose(member.density(1, np.array([0.2, 0.4]), 2.0), [3.0 * 1.02] * 2)
        self.assertEqual(member.metadata()["OrderQCD"], 2)

    def test_family_requires_configured_hadron(self):
        family = LHAPDFFamily.__new__(LHAPDFFamily)
        family.hadron_sets = {"pi+": "fake"}
        family.member = 0
        family._members = {"pi+": LHAPDFFMember("fake", 0, backend=FakePDF(), pdfset=FakeSet())}
        self.assertAlmostEqual(family.density("pi+", 1, 0.3, 2.0), 3.0 * 1.02)
        with self.assertRaises(KeyError):
            family.member_for("K+")


if __name__ == "__main__":
    unittest.main()
