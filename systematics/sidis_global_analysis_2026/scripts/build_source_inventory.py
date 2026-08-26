#!/usr/bin/env python3
"""Build a portable source inventory from downloaded/profiles HEPData records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CAMPAIGN = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=CAMPAIGN / "reports/public_source_inventory.json")
    parser.add_argument("--markdown", type=Path, default=CAMPAIGN / "reports/public_source_inventory.md")
    args = parser.parse_args()
    config = json.loads((CAMPAIGN / "config/public_sources.json").read_text())
    downloads = json.loads((CAMPAIGN / "data/hepdata_download_manifest.json").read_text())
    profiles = json.loads((CAMPAIGN / "reports/hepdata_table_inventory.json").read_text())["profiles"]
    records = []
    for source in config["records"]:
        recid = source["record"]
        downloaded = next(item for item in downloads["records"] if item["record"] == recid)
        items = [item for item in profiles if f"/hepdata/{recid}/" in item["path"]]
        profile = {
            "tables": len(items),
            "rows": sum(item["row_count"] for item in items),
            "transverse_momentum_tables": sum(item["has_transverse_momentum"] for item in items),
            "tables_with_statistical_columns": sum(item["has_statistical_columns"] for item in items),
            "tables_with_systematic_columns": sum(item["has_systematic_columns"] for item in items),
        }
        records.append({**source, "download": {"archive": downloaded["archive"], "archive_sha256": downloaded["archive_sha256"], "csv_table_count": downloaded["csv_table_count"]}, "profile": profile})
    report = {"campaign": "sidis_global_analysis_2026", "status": "public_sources_downloaded_and_profiled_not_fit_ready", "record_count": len(records), "records": records, "approved_rows": 0, "selection_authorized": False, "production_authorized": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Public SIDIS source inventory", "", "Status: downloaded and profiled; not fit-ready and no rows approved.", "", "| Record | Collaboration | Tables | Rows | pT tables | Status |", "| --- | --- | ---: | ---: | ---: | --- |"]
    for item in records:
        p = item["profile"]
        lines.append(f"| [{item['record']}]({item['source_url']}) | {item['collaboration']} | {p['tables']} | {p['rows']} | {p['transverse_momentum_tables']} | {item['status']} |")
    lines += ["", "Archive SHA256 values and table metadata are in `data/hepdata_download_manifest.json` and `reports/public_source_inventory.json`.", "The profiler does not select rows or combine uncertainties into a fit covariance."]
    args.markdown.write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": report["status"], "record_count": len(records), "approved_rows": 0}, indent=2))


if __name__ == "__main__":
    main()
