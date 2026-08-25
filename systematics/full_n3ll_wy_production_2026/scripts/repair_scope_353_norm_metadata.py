#!/usr/bin/env python3
"""Create candidate-only data copies with parsed normalization metadata.

The source data are untouched.  The generic v19 trainer's paper map only
contains the fixed-target groups, so this isolated copy parses the published
``sysNorm``/``sysP2P`` columns for all scope datasets before a CSV-driven fit.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
SRC = BASE / "reports/scope_353_fnp_inputs/data"
DST = BASE / "reports/scope_353_fnp_inputs/data_with_csv_uncertainties"


def parse_percent(value: object) -> float:
    if value is None or pd.isna(value):
        return 0.0
    s = str(value).strip().replace("%", "")
    if not s or s.lower() in {"nan", "none", "-"}:
        return 0.0
    # A signed point-to-point entry is still an uncertainty magnitude.
    return abs(float(s)) / 100.0


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    rows = 0
    datasets = {}
    for src in sorted(SRC.glob("*.csv")):
        df = pd.read_csv(src)
        if "sysNorm_rel" not in df:
            df["sysNorm_rel"] = df.get("sysNorm", pd.Series([0.0] * len(df))).map(parse_percent)
        else:
            parsed_norm = df.get("sysNorm", pd.Series([None] * len(df))).map(parse_percent)
            # Prefer an explicit numeric column when it is present and finite.
            df["sysNorm_rel"] = pd.to_numeric(df["sysNorm_rel"], errors="coerce").fillna(parsed_norm)
        if "sysP2P_rel" not in df:
            df["sysP2P_rel"] = df.get("sysP2P", pd.Series([0.0] * len(df))).map(parse_percent)
        else:
            parsed_ptp = df.get("sysP2P", pd.Series([None] * len(df))).map(parse_percent)
            df["sysP2P_rel"] = pd.to_numeric(df["sysP2P_rel"], errors="coerce").fillna(parsed_ptp)
        df.to_csv(DST / src.name, index=False)
        rows += len(df)
        datasets[src.stem] = {
            "rows": int(len(df)),
            "norm_rel": sorted(set(float(x) for x in df["sysNorm_rel"].to_numpy(float))),
            "ptp_rel": sorted(set(float(x) for x in df["sysP2P_rel"].to_numpy(float))),
        }
    status = {
        "status": "isolated_scope_353_candidate_uncertainty_metadata_complete",
        "rows": rows,
        "datasets": datasets,
        "source": str(SRC),
        "output": str(DST),
        "frozen_production_modified": False,
        "production_promotion_authorized": False,
    }
    (DST / "metadata_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
