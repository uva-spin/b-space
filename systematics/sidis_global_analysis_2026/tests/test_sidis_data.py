import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sidis_data import profile_table, read_hepdata_csv  # noqa: E402


class SidisDataTests(unittest.TestCase):
    def test_duplicate_columns_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "table.csv"
            path.write_text("#: name: Example\nx,x,M,stat +,stat -,sys +,sys -\n0.1,0.2,1,0.1,-0.1,0.2,-0.2\n")
            table = read_hepdata_csv(path)
            self.assertEqual(table.columns[:3], ("x", "x__2", "M"))
            self.assertEqual(table.metadata["name"], "Example")

    def test_transverse_and_uncertainty_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "table.csv"
            path.write_text("#: name: Example\n$x$,$Q^2$,$z$,$P_{hT}^2$,M,stat +,stat -,sys +,sys -\n0.1,2,0.3,0.12,1,0.1,-0.1,0.2,-0.2\n")
            profile = profile_table(read_hepdata_csv(path))
            self.assertTrue(profile["has_transverse_momentum"])
            self.assertTrue(profile["has_statistical_columns"])
            self.assertTrue(profile["has_systematic_columns"])


if __name__ == "__main__":
    unittest.main()
