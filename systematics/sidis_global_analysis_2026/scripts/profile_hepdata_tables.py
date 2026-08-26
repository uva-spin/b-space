#!/usr/bin/env python3
"""Profile downloaded HEPData tables without selecting fit rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

CAMPAIGN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAMPAIGN))
from sidis_data import profile_table, read_hepdata_csv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=CAMPAIGN / "data/raw/hepdata")
    parser.add_argument("--output", type=Path, default=CAMPAIGN / "reports/hepdata_table_inventory.json")
    args = parser.parse_args()
    files = sorted(args.raw_root.glob("*/tables/**/*.csv"))
    if not files:
        raise SystemExit(f"no CSV files below {args.raw_root}")
    profiles = [profile_table(read_hepdata_csv(path)) for path in files]
    report = {
        "campaign": "sidis_global_analysis_2026",
        "status": "public_hepdata_inventory_profiled_not_fit_ready",
        "counts": {
            "tables": len(profiles),
            "rows": sum(item["row_count"] for item in profiles),
            "with_transverse_momentum": sum(item["has_transverse_momentum"] for item in profiles),
            "with_statistical_columns": sum(item["has_statistical_columns"] for item in profiles),
            "with_systematic_columns": sum(item["has_systematic_columns"] for item in profiles),
        },
        "profiles": profiles,
        "selection_authorized": False,
        "production_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], **report["counts"]}, indent=2))


if __name__ == "__main__":
    main()
