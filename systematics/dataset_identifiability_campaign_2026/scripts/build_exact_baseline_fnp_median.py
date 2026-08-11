#!/usr/bin/env python3
"""Build a pointwise FNP median from a declared baseline-seed subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
UNITARY = (
    BASE.parent
    / "high_qt_direct_production_benchmark/experimental_unitary_transition")
TARGET = BASE / "summaries/exact_baseline_fnp_median"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    parser.add_argument("--target-name", default="exact_baseline_fnp_median")
    args = parser.parse_args()
    seeds = tuple(args.seeds or range(303, 327))
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("reference requires at least two distinct seeds")
    target = BASE / "summaries" / args.target_name
    members = []
    for seed in seeds:
        frame = pd.read_csv(
            UNITARY / "outputs" / f"fig6_lbfgs_stationary_s{seed}"
            / "fnp_grid.csv").sort_values(["x", "bT"])
        members.append(frame["F_NP"].to_numpy(float))
    template = frame[["x", "bT"]].copy()
    matrix = np.asarray(members)
    template["F_NP"] = np.median(matrix, axis=0)
    template["q16"] = np.quantile(matrix, .16, axis=0)
    template["q84"] = np.quantile(matrix, .84, axis=0)
    target.mkdir(parents=True, exist_ok=True)
    template.to_csv(target / "fnp_median.csv", index=False)
    summary = {
        "status": "isolated_exact_baseline_FNP_median_not_production",
        "member_count": len(seeds),
        "seeds": list(seeds),
        "source": "exactly reproduced historical Fig. 6 baseline endpoints",
        "semantics": (
            "pointwise median is an explicit empirical central convention; "
            "q16/q84 are retained for sensitivity accounting"),
        "production_sources_modified": False,
    }
    (target / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
