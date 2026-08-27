#!/usr/bin/env python3
"""Normalize the text output of the external APFEL++ probe for fit diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--ratio-column", default="apfel_nlo_numerator_lo_den_ratio",
                    help="ratio column when input is a CSV table")
    ap.add_argument("--output-column", default="apfel_nlo_numerator_lo_den_ratio",
                    help="name written to the normalized ratio table")
    ap.add_argument("--csv", action="store_true",
                    help="read a row-level APFEL CSV rather than raw probe text")
    args = ap.parse_args()
    if args.csv:
        result = pd.read_csv(args.input)
        required = {"row_id", args.ratio_column}
        missing = required.difference(result.columns)
        if missing:
            raise ValueError(f"APFEL CSV missing columns: {sorted(missing)}")
        result = result[["row_id", args.ratio_column]].rename(
            columns={args.ratio_column: args.output_column})
        if result.empty or result.row_id.astype(str).duplicated().any():
            raise ValueError("APFEL CSV did not provide unique row-level ratios")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
        print(f"wrote {len(result)} rows to {args.output}")
        return
    rows = []
    for line in args.input.read_text().splitlines():
        if not line.startswith("hepdata:"):
            continue
        fields = line.split(",")
        if len(fields) != 8:
            raise ValueError(f"unexpected APFEL row with {len(fields)} fields")
        rows.append({"row_id": fields[0], "lo_ratio": float(fields[6]),
                     "apfel_nlo_numerator_lo_den_ratio": float(fields[7])})
    result = pd.DataFrame(rows)
    if result.empty or result.row_id.duplicated().any():
        raise ValueError("APFEL probe did not provide unique row-level ratios")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"wrote {len(result)} rows to {args.output}")


if __name__ == "__main__":
    main()
