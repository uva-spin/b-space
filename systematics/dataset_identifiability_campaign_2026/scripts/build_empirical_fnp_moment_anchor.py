#!/usr/bin/env python3
"""Build a robust FNP-width anchor from independent data-fit basins."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
SEEDS = tuple(range(303, 315))
BMIN = 0.10
BMAX = 2.0
TARGET = BASE / "summaries/empirical_fnp_moment_anchor"


def main() -> None:
    rows = []
    for seed in SEEDS:
        source = (
            BASE / "outputs"
            / f"independent_datafit_D020_E772_init{seed}"
            / "fnp_grid.csv"
        )
        frame = pd.read_csv(source)
        for x, group in frame[
            (frame["bT"] >= BMIN) & (frame["bT"] <= BMAX)
        ].groupby("x"):
            group = group.sort_values("bT")
            moment = np.trapezoid(
                group["F_NP"].to_numpy(float),
                group["bT"].to_numpy(float),
            ) / (BMAX - BMIN)
            rows.append({"seed": seed, "x": float(x), "moment": float(moment)})
    members = pd.DataFrame(rows)
    anchor = members.groupby("x", as_index=False).agg(
        moment=("moment", "median"),
        q16=("moment", lambda values: np.quantile(values, 0.16)),
        q84=("moment", lambda values: np.quantile(values, 0.84)),
        minimum=("moment", "min"),
        maximum=("moment", "max"),
    )
    TARGET.mkdir(parents=True, exist_ok=True)
    members.to_csv(TARGET / "member_moments.csv", index=False)
    anchor.to_csv(TARGET / "moment_anchor.csv", index=False)
    summary = {
        "status": "isolated_empirical_fnp_moment_anchor_not_production",
        "source": "twelve independent unregularized D020_E772 data-fit basins",
        "source_seeds": list(SEEDS),
        "definition": (
            "normalized integral of F_NP over 0.1<=bT<=2 GeV^-1; "
            "pointwise-in-x median defines the central anchor"
        ),
        "uncertainty_semantics": (
            "q16/q84 are descriptive independent-start quantiles and remain "
            "available for downstream nonuniqueness propagation"
        ),
        "production_sources_modified": False,
    }
    (TARGET / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
