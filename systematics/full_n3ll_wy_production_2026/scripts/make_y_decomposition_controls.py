#!/usr/bin/env python3
"""Make isolated Y-grid decomposition controls without touching production inputs."""

from pathlib import Path
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
REPORTS = BASE / "reports/scope_353_fnp_inputs"
SOURCE = REPORTS / "scope_353_y.csv"


def main() -> None:
    y = pd.read_csv(SOURCE)
    variants = {
        "no_lhcb": ~y.row_id.astype(str).str.startswith("LHCb_7:"),
        "lhcb_only": y.row_id.astype(str).str.startswith("LHCb_7:"),
    }
    for tag, keep in variants.items():
        out = y.copy()
        out.loc[~keep, "Y_CS"] = 0.0
        path = REPORTS / f"scope_353_y_{tag}.csv"
        out.to_csv(path, index=False)
        print(path)


if __name__ == "__main__":
    main()
