from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sidis_data import read_hepdata_csv  # noqa: E402
from sidis_dataset import SidisColumnMap, canonicalize_table, parse_interval  # noqa: E402


class SidisDatasetTests(unittest.TestCase):
    def test_explicit_columns_and_components(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "table.csv"
            path.write_text("x,Q2,z,P_hT,M,stat +,stat -,sys +,sys -\n0.1,2,0.3,0.4,1,0.1,-0.1,0.2,-0.2\n")
            rows = canonicalize_table(read_hepdata_csv(path), SidisColumnMap(value="M", axis_columns={"x": "x", "q2": "Q2", "z": "z", "pht": "P_hT"}, stat_columns=("stat +", "stat -"), sys_columns=("sys +", "sys -"), required_axes=("x", "q2", "z", "pht")), source="synthetic")
            self.assertEqual(rows[0].uncertainties, {"stat": (0.1, 0.1), "sys": (0.2, 0.2)})

    def test_block_metadata_axes_and_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blocked.csv"
            path.write_text("#: X_BJ,,,0.1-0.2\n#: Z,,,0.3-0.4\nP_T**2,M,error +,error -\n0.2,1,0.1,-0.1\n")
            rows = canonicalize_table(read_hepdata_csv(path), SidisColumnMap(value="M", axis_columns={"pht2": "P_T**2"}, metadata_axes={"x": "x_bin", "z": "z_bin"}, total_columns=("error +", "error -"), required_axes=("x", "z", "pht2")), source="synthetic")
            self.assertAlmostEqual(rows[0].axes["x"], 0.15)
            bad = Path(directory) / "bad.csv"
            bad.write_text("x,M\n0.1,1\n")
            with self.assertRaises(ValueError):
                canonicalize_table(read_hepdata_csv(bad), SidisColumnMap(value="M", axis_columns={"x": "x"}), source="synthetic")

    def test_interval_parser(self):
        self.assertEqual(parse_interval("0.2-0.4"), (0.2, 0.4))
        with self.assertRaises(ValueError):
            parse_interval("z > 0.2")

    def test_block_filter_and_placeholder_policy_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blocked.csv"
            path.write_text(
                "#: RE,E P --> E PI+ X\nP_hT,M,stat +,stat -\n0.2,1,0.1,-0.1\n"
                "#: RE,E DEUT --> E DEUT PI+ X\nP_hT,M,stat +,stat -\n0.3,2,0.2,-0.2\n"
            )
            table = read_hepdata_csv(path)
            rows = canonicalize_table(
                table,
                SidisColumnMap(
                    value="M", axis_columns={"pht": "P_hT"},
                    stat_columns=("stat +", "stat -"),
                    required_axes=("pht",),
                    block_filters={"reaction": "E DEUT --> E DEUT PI+ X"},
                    target="D",
                ),
                source="synthetic",
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].target, "D")


if __name__ == "__main__":
    unittest.main()
