import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sidis_data import profile_table, read_hepdata_csv, read_covariance_matrix  # noqa: E402


class SidisDataTests(unittest.TestCase):
    def test_duplicate_columns_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "table.csv"
            path.write_text("#: name: Example\nx,x,M,stat +,stat -,sys +,sys -\n0.1,0.2,1,0.1,-0.1,0.2,-0.2\n")
            table = read_hepdata_csv(path)
            self.assertEqual(table.columns[:3], ("x", "x__2", "M"))
            self.assertEqual(table.metadata["name"], "Example")

    def test_repeated_headers_are_not_counted_as_rows_and_blocks_are_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blocked.csv"
            path.write_text(
                "#: RE,E P --> E PI+ X\n"
                "x,M,stat +,stat -\n"
                "0.1,1.0,0.1,-0.1\n"
                "#: RE,E DEUT --> E DEUT PI+ X\n"
                "x,M,stat +,stat -\n"
                "0.2,2.0,0.2,-0.2\n"
            )
            table = read_hepdata_csv(path)
            self.assertEqual(len(table.rows), 2)
            self.assertEqual(len(table.row_metadata), 2)
            self.assertEqual(table.row_metadata[0]["reaction"], "E P --> E PI+ X")
            self.assertEqual(table.row_metadata[1]["reaction"], "E DEUT --> E DEUT PI+ X")

    def test_transverse_and_uncertainty_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "table.csv"
            path.write_text("#: name: Example\n$x$,$Q^2$,$z$,$P_{hT}^2$,M,stat +,stat -,sys +,sys -\n0.1,2,0.3,0.12,1,0.1,-0.1,0.2,-0.2\n")
            profile = profile_table(read_hepdata_csv(path))
            self.assertTrue(profile["has_transverse_momentum"])
            self.assertTrue(profile["has_statistical_columns"])
            self.assertTrue(profile["has_systematic_columns"])

    def test_covariance_reader_accepts_compressed_symmetric_matrix(self):
        import gzip

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "covariance.list.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write("# covariance\n1.0,0.2\n0.2,2.0\n")
            matrix = read_covariance_matrix(path, expected_size=2)
            self.assertEqual(matrix.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
