#!/usr/bin/env python3
"""Build an all-x baseline-reference CSV with the promoted 96-start x=0.1 slice.

The historical reference-distance table contains the eight x knots needed by
the current FiLM regularizer, while the promoted production ensemble stores the
authoritative 96-start F_NP curve at x=0.1.  This isolated utility preserves the
historical x knots and replaces only their x=0.1 slice by the promoted median.
It creates a diagnostic input; it never writes into the production package.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
SYSTEMATICS = BASE.parent
OLD = SYSTEMATICS / "dataset_identifiability_campaign_2026/summaries/exact_baseline_fnp_median/fnp_median.csv"
PROMOTED = SYSTEMATICS / "dataset_identifiability_campaign_2026/summaries/lambda1_start_expansion96_bspace/bspace_tmd_ensemble_long.csv"
OUT = BASE / "reports/baseline_reference_promoted96_allx.csv"
META = BASE / "reports/baseline_reference_promoted96_allx.json"


def main() -> None:
    old = pd.read_csv(OLD)
    required = {"x", "bT", "F_NP"}
    if not required.issubset(old.columns):
        raise ValueError(f"historical reference missing {required - set(old.columns)}")
    old = old[["x", "bT", "F_NP"]].copy()
    promoted = pd.read_csv(PROMOTED)
    promoted = promoted[np.isclose(promoted["x"].to_numpy(float), 0.1)]
    if promoted.empty:
        raise ValueError("promoted 96-start ensemble has no x=0.1 rows")
    med = (promoted.groupby("bT", observed=False)["F_NP"].median()
           .sort_index())
    base_grid = old[np.isclose(old["x"].to_numpy(float), 0.1)].sort_values("bT")
    b = base_grid["bT"].to_numpy(float)
    promoted_f = np.interp(b, med.index.to_numpy(float), med.to_numpy(float))
    replaced = old[~np.isclose(old["x"].to_numpy(float), 0.1)].copy()
    x01 = pd.DataFrame({"x": 0.1, "bT": b, "F_NP": promoted_f})
    out = pd.concat([replaced, x01], ignore_index=True).sort_values(["x", "bT"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    meta = {
        "status": "isolated_promoted96_reference_input_not_production",
        "source_historical_all_x": str(OLD),
        "source_promoted_96_start_x0p1": str(PROMOTED),
        "replacement": "historical x=0.1 slice replaced by promoted 96-start median; other x knots retained",
        "x_values": sorted(float(v) for v in out.x.unique()),
        "rows": int(len(out)),
        "output": str(OUT),
        "frozen_production_modified": False,
        "promotion_authorized": False,
    }
    META.write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
