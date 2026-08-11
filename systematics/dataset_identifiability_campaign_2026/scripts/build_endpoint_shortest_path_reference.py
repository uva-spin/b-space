#!/usr/bin/env python3
"""Build analytic shortest-path F_NP references from fixed endpoints.

This is an isolated diagnostic input generator.  For each x, the endpoint
values are taken from the declared reference table and the interior is filled
with the unique shortest path in either direct-F or log-F coordinates.  No
production fit or frozen result is changed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bmin", type=float, default=0.10)
    parser.add_argument("--bmax", type=float, default=2.0)
    parser.add_argument("--metric", choices=("F", "logF"), required=True)
    args = parser.parse_args()

    if not 0.0 <= args.bmin < args.bmax:
        raise ValueError("require 0 <= bmin < bmax")
    frame = pd.read_csv(args.reference)
    required = {"x", "bT", "F_NP"}
    if not required.issubset(frame.columns):
        raise ValueError(f"reference must contain {sorted(required)}")

    rows = []
    endpoint_rows = []
    for x, group in frame.groupby("x", sort=True):
        group = group.sort_values("bT")
        b_source = group["bT"].to_numpy(float)
        f_source = group["F_NP"].to_numpy(float)
        f0 = float(np.interp(args.bmin, b_source, f_source))
        f1 = float(np.interp(args.bmax, b_source, f_source))
        if f0 <= 0.0 or f1 <= 0.0:
            raise ValueError(f"non-positive endpoint at x={x}")
        # Include the declared endpoints explicitly; otherwise a source grid
        # whose nearest node is merely close to b_min/b_max would make the
        # exported diagnostic appear not to satisfy its own endpoint claim.
        b = np.unique(np.r_[b_source, args.bmin, args.bmax])
        f = np.interp(b, b_source, f_source)
        t = ((b - args.bmin) / (args.bmax - args.bmin)).clip(0.0, 1.0)
        if args.metric == "F":
            path = f0 + (f1 - f0) * t
        else:
            path = np.exp(np.log(f0) + (np.log(f1) - np.log(f0)) * t)
        # Preserve the reference outside the constrained interval.  The
        # endpoint values themselves are exact, while the continuation remains
        # a transparent diagnostic choice rather than an implicit tail prior.
        path = np.where(b < args.bmin, f, path)
        path = np.where(b > args.bmax, f, path)
        for bi, value in zip(b, path):
            rows.append({"x": float(x), "bT": float(bi), "F_NP": float(value)})
        endpoint_rows.append({
            "x": float(x), "F_NP_bmin": f0, "F_NP_bmax": f1,
            "metric": args.metric,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    summary = {
        "status": "isolated_analytic_endpoint_shortest_path_reference",
        "source_reference": str(args.reference),
        "output": str(args.output),
        "bmin": args.bmin,
        "bmax": args.bmax,
        "metric": args.metric,
        "mathematical_definition": (
            "linear interpolation in F_NP" if args.metric == "F"
            else "linear interpolation in log(F_NP)"
        ),
        "uniqueness": "unique minimizer for the corresponding fixed-endpoint path metric",
        "continuation_outside_interval": "source reference unchanged",
        "endpoints": endpoint_rows,
        "production_state_modified": False,
    }
    (args.output.with_suffix(".summary.json")).write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
