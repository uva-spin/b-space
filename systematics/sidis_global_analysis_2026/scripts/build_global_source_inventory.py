#!/usr/bin/env python3
"""Build the staged global SIDIS inventory without approving fit rows.

The report combines the candidate registry with whatever public files have
been harvested by ``fetch_global_sources.py``.  A missing archive is a
provenance state, not a reason to drop the source.  HEPData tables are
profiled with the same metadata-preserving reader used for the initial
HERMES/COMPASS harvest; arXiv source archives are counted and named but TeX
tables are not converted implicitly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

CAMPAIGN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAMPAIGN))
from sidis_data import profile_table, read_hepdata_csv  # noqa: E402


def profile_hepdata(root: Path) -> dict[str, Any]:
    files = sorted(root.glob("tables/**/*.csv")) if root.exists() else []
    profiles = [profile_table(read_hepdata_csv(path)) for path in files]
    return {
        "tables": len(profiles),
        "rows": sum(item["row_count"] for item in profiles),
        "primary_rows": sum(item.get("primary_row_count", item["row_count"]) for item in profiles),
        "auxiliary_rows": sum(item.get("auxiliary_row_count", 0) for item in profiles),
        "transverse_momentum_tables": sum(item["has_transverse_momentum"] for item in profiles),
        "tables_with_statistical_columns": sum(item["has_statistical_columns"] for item in profiles),
        "tables_with_systematic_columns": sum(item["has_systematic_columns"] for item in profiles),
        "profiled": bool(profiles),
    }


def profile_record(record: dict[str, Any], manifest_by_id: dict[str, Any]) -> dict[str, Any]:
    item: dict[str, Any] = dict(record)
    downloaded = manifest_by_id.get(record["id"])
    item["harvest"] = downloaded or {"status": "not_harvested"}
    if record["source_kind"] == "hepdata":
        table_root = CAMPAIGN / "data" / "raw" / "global" / record["id"].replace(":", "_")
        item["profile"] = profile_hepdata(table_root)
    elif record["source_kind"] in {"arxiv_eprint_with_ancillary", "arxiv_eprint_tex_tables"}:
        source_root = CAMPAIGN / "data" / "raw" / "global" / record["id"].replace(":", "_") / "source"
        files = sorted(str(path.relative_to(source_root)) for path in source_root.rglob("*") if path.is_file()) if source_root.exists() else []
        item["profile"] = {
            "source_files": len(files),
            "listed_data_paths_present": [path for path in record.get("data_paths", []) if (source_root / path).exists()],
            "listed_table_labels": record.get("table_labels", []),
            "source_extracted": bool(files),
        }
    else:
        item["profile"] = {"profiled": False}
    return item


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CAMPAIGN / "config/global_sources.json")
    parser.add_argument("--manifest", type=Path, default=CAMPAIGN / "data/global_source_download_manifest.json")
    parser.add_argument("--output", type=Path, default=CAMPAIGN / "reports/global_source_inventory.json")
    parser.add_argument("--markdown", type=Path, default=CAMPAIGN / "reports/global_source_inventory.md")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    manifest = json.loads(args.manifest.read_text()) if args.manifest.exists() else {"records": []}
    manifest_by_id = {item["id"]: item for item in manifest.get("records", [])}
    records = [profile_record(item, manifest_by_id) for item in config["records"]]
    counts = {
        "registry_records": len(records),
        "hepdata_records": sum(item["source_kind"] == "hepdata" for item in records),
        "arxiv_records": sum(item["source_kind"].startswith("arxiv_") for item in records),
        "pointer_records": sum(item["source_kind"] not in {"hepdata", "arxiv_eprint_with_ancillary", "arxiv_eprint_tex_tables"} for item in records),
        "harvested_records": sum(item["harvest"].get("status") == "downloaded" for item in records),
        "profiled_hepdata_tables": sum(item.get("profile", {}).get("tables", 0) for item in records),
        "profiled_hepdata_rows": sum(item.get("profile", {}).get("rows", 0) for item in records),
        "profiled_hepdata_primary_rows": sum(item.get("profile", {}).get("primary_rows", 0) for item in records),
        "profiled_hepdata_auxiliary_rows": sum(item.get("profile", {}).get("auxiliary_rows", 0) for item in records),
        "approved_rows": 0,
    }
    report = {
        "campaign": "sidis_global_analysis_2026",
        "status": "global_candidate_inventory_harvested_not_fit_ready",
        "registry": str(args.config.relative_to(CAMPAIGN) if args.config.is_absolute() else args.config),
        "counts": counts,
        "records": records,
        "staging_rule": "fit stage 1 first; add stage 2 one experiment/hadron/target family at a time after closure; retain diagnostics and deferred sources without merging them",
        "selection_authorized": False,
        "production_authorized": False,
        "production_files_modified": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Global unpolarized SIDIS candidate inventory",
        "",
        "Status: broad public-source registry and harvest; no rows approved for a fit.",
        "",
        "The registry deliberately separates the data universe from the staged fit scope. Stage 1 is the clean proton/deuteron multiplicity core; stage 2 adds older, identified, or absolute-cross-section data only after observable and covariance closure. Nuclear, current-region, jet/remnant, and source-only records remain diagnostics or deferred inputs.",
        "",
        f"Registry records: **{counts['registry_records']}**; HEPData records: **{counts['hepdata_records']}**; public arXiv source records: **{counts['arxiv_records']}**; pointer/deferred records: **{counts['pointer_records']}**.",
        f"Harvested HEPData profile: **{counts['profiled_hepdata_tables']}** tables and **{counts['profiled_hepdata_rows']}** rows (primary **{counts['profiled_hepdata_primary_rows']}**, auxiliary **{counts['profiled_hepdata_auxiliary_rows']}**).",
        "",
        "| ID | Collaboration | Stage | Role | Observable | Target | Readiness | Harvest/profile |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in records:
        p = item.get("profile", {})
        h = item.get("harvest", {})
        if item["source_kind"] == "hepdata":
            state = f"{p.get('tables', 0)} tables/{p.get('rows', 0)} rows" if p.get("profiled") else h.get("status", "not harvested")
        elif item["source_kind"].startswith("arxiv_"):
            state = f"{p.get('source_files', 0)} source files" if p.get("source_extracted") else h.get("status", "not harvested")
        else:
            state = h.get("status", "pointer only")
        lines.append(
            f"| [{item['id']}]({item.get('source_url', '')}) | {item['collaboration']} | {item['stage']} | {item['fit_role']} | {item['observable_class']} | {item['target']} | {item['readiness']} | {state} |"
        )
    lines += [
        "",
        "The HEPData archives and arXiv e-print sources are local ignored inputs. Their URLs and SHA256 hashes are in `data/global_source_download_manifest.json`; raw files are not part of the public source release. TeX tables and ancillary files require an explicit source-specific converter and unit test before a canonical observation is created.",
        "",
        "The historical global-fit benchmark of 1,547 SIDIS points is a selected HERMES/COMPASS subset after TMD-validity cuts, not a claim that all raw tables should be fitted. Our target is to reproduce that count under an explicit cut/observable lock, then test additional JLab, EMC, E665, and HERA families progressively.",
    ]
    args.markdown.write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": report["status"], **counts}, indent=2))


if __name__ == "__main__":
    main()
