"""Regression tests for the source-only global SIDIS inventory drivers."""

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.inventory_arxiv_tables import profile_clas  # noqa: E402
from scripts.reproduce_sidis_benchmark_count import interval  # noqa: E402


class GlobalInventoryTests(unittest.TestCase):
    def test_interval_uses_unsigned_separator(self):
        self.assertEqual(interval("1.7-3.0"), (1.7, 3.0))
        self.assertIsNone(interval("Q > 1.4"))

    def test_clas_profile_rejects_no_numeric_rows_by_reporting_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.dat"
            path.write_text(
                "description\n"
                "Q2 x z pT2 phi ds stat sys rad\n"
                "[GeV2] [1] [1] [GeV2] [deg] [u] [u] [u] [1]\n"
                "1.0 0.2 0.3 0.1 90 2.0 0.1 0.2 1.0\n"
            )
            profile = profile_clas(path, {"axes": ["Q2", "x", "z", "pT2", "phi"], "observable": "test"})
            self.assertEqual(profile["numeric_row_count"], 1)
            self.assertEqual(profile["malformed_numeric_row_count"], 0)
            self.assertEqual(profile["column_ranges"]["Q2"]["max"], 1.0)


if __name__ == "__main__":
    unittest.main()
