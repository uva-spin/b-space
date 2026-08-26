#!/usr/bin/env python3
"""Fetch the staged public SIDIS candidate registry.

This is deliberately separate from ``fetch_hepdata_records.py``.  The first
seven HERMES/COMPASS records remain the reproducibility anchor for the initial
harvest; this driver adds older HEPData records and public arXiv source
archives without changing that baseline manifest.  Raw downloads live under
``data/raw/global`` (ignored by the source release).  Every downloaded object
gets a SHA256 hash and a manifest entry.  Source-only and access-restricted
records are recorded without pretending that a file was obtained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
from typing import Any

import requests


CAMPAIGN = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = CAMPAIGN / "config" / "global_sources.json"
RAW_ROOT = CAMPAIGN / "data" / "raw" / "global"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(url: str, destination: Path) -> None:
    response = requests.get(url, timeout=240)
    response.raise_for_status()
    destination.write_bytes(response.content)


def safe_extract(archive: Path, destination: Path) -> None:
    """Extract a tar archive after rejecting path traversal members."""

    destination = destination.resolve()
    with tarfile.open(archive, mode="r:*") as tar:
        members = tar.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if target != destination and destination not in target.parents:
                raise RuntimeError(f"unsafe archive member: {member.name}")
        # Python 3.12's data filter prevents links/devices from escaping the
        # destination.  Keep a fallback for older interpreters used by some
        # reproducibility environments.
        try:
            tar.extractall(destination, filter="data")
        except TypeError:
            tar.extractall(destination)


def record_path(record: dict[str, Any]) -> Path:
    safe = str(record["id"]).replace(":", "_").replace("/", "_")
    return RAW_ROOT / safe


def fetch_hepdata(record: dict[str, Any], force: bool) -> dict[str, Any]:
    out = record_path(record)
    out.mkdir(parents=True, exist_ok=True)
    archive = out / "submission_csv.tar.gz"
    metadata_path = out / "record.json"
    tables = out / "tables"
    recid = str(record["record"])
    if force or not metadata_path.exists():
        fetch(f"https://www.hepdata.net/record/{recid}?format=json", metadata_path)
    metadata = json.loads(metadata_path.read_text())
    if force or not archive.exists():
        fetch(f"https://www.hepdata.net/download/submission/{recid}/1/csv", archive)
    if force or not tables.exists():
        if tables.exists():
            shutil.rmtree(tables)
        tables.mkdir()
        safe_extract(archive, tables)
    csv_files = sorted(str(p.relative_to(tables)) for p in tables.rglob("*.csv"))
    return {
        "id": record["id"],
        "source_kind": record["source_kind"],
        "status": "downloaded",
        "source_url": record.get("source_url"),
        "archive": str(archive.relative_to(CAMPAIGN)),
        "archive_sha256": sha256(archive),
        "record_json": str(metadata_path.relative_to(CAMPAIGN)),
        "table_root": str(tables.relative_to(CAMPAIGN)),
        "csv_table_count": len(csv_files),
        "csv_tables": csv_files,
        "record_version": metadata.get("version"),
        "record_title": metadata.get("record", {}).get("title"),
    }


def fetch_arxiv(record: dict[str, Any], force: bool) -> dict[str, Any]:
    out = record_path(record)
    out.mkdir(parents=True, exist_ok=True)
    archive = out / "source.eprint"
    extracted = out / "source"
    if force or not archive.exists():
        fetch(record["download_url"], archive)
    # arXiv e-print responses are tar/gzip source archives.  Extract only when
    # the archive is readable; retaining the hash still documents a source
    # that needs manual conversion if a future arXiv format changes.
    extracted_status = "not_attempted"
    source_files: list[str] = []
    if force or not extracted.exists():
        if extracted.exists():
            shutil.rmtree(extracted)
        extracted.mkdir()
        try:
            safe_extract(archive, extracted)
            extracted_status = "extracted"
            source_files = sorted(str(p.relative_to(extracted)) for p in extracted.rglob("*") if p.is_file())
        except (tarfile.ReadError, OSError) as exc:
            extracted_status = f"not_tar_archive:{type(exc).__name__}"
    else:
        extracted_status = "existing"
        source_files = sorted(str(p.relative_to(extracted)) for p in extracted.rglob("*") if p.is_file())
    return {
        "id": record["id"],
        "source_kind": record["source_kind"],
        "status": "downloaded",
        "source_url": record.get("source_url"),
        "archive": str(archive.relative_to(CAMPAIGN)),
        "archive_sha256": sha256(archive),
        "extracted_root": str(extracted.relative_to(CAMPAIGN)),
        "extraction_status": extracted_status,
        "source_file_count": len(source_files),
        "source_files": source_files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--id", action="append", help="registry id; repeat for a subset")
    parser.add_argument("--kind", action="append", help="source kind; repeat to select kinds")
    parser.add_argument("--force", action="store_true", help="redownload/re-extract selected objects")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    requested_ids = set(args.id or [])
    selected_ids = requested_ids or {item["id"] for item in config["records"]}
    selected_kinds = set(args.kind or [])
    records = [
        item for item in config["records"]
        if item["id"] in selected_ids and (not selected_kinds or item["source_kind"] in selected_kinds)
    ]
    missing = requested_ids - {item["id"] for item in records}
    if missing:
        raise SystemExit(f"records not in config: {sorted(missing)}")
    results: list[dict[str, Any]] = []
    for record in records:
        kind = record["source_kind"]
        if kind == "hepdata":
            results.append(fetch_hepdata(record, args.force))
        elif kind in {"arxiv_eprint_with_ancillary", "arxiv_eprint_tex_tables"}:
            results.append(fetch_arxiv(record, args.force))
        else:
            results.append({
                "id": record["id"],
                "source_kind": kind,
                "status": "pointer_only",
                "source_url": record.get("source_url"),
            })
    target = CAMPAIGN / "data" / "global_source_download_manifest.json"
    # Targeted refreshes (for example HEPData first and arXiv later) must
    # preserve hashes already recorded for the other source kinds.
    if (args.id or args.kind) and target.exists():
        previous = json.loads(target.read_text())
        by_id = {item["id"]: item for item in previous.get("records", [])}
        by_id.update({item["id"]: item for item in results})
        results = [by_id[key] for key in sorted(by_id)]
    manifest = {
        "campaign": "sidis_global_analysis_2026",
        "status": "global_public_candidate_sources_harvested_or_registered",
        "registry": str(args.config.relative_to(CAMPAIGN) if args.config.is_absolute() else args.config),
        "records": results,
        "approved_rows": 0,
        "selection_authorized": False,
        "production_authorized": False,
        "production_files_modified": False,
    }
    target.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
