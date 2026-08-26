#!/usr/bin/env python3
"""Fetch versioned HEPData SIDIS CSV archives into a local campaign tree."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

import requests

CAMPAIGN = Path(__file__).resolve().parents[1]
CONFIG = CAMPAIGN / "config/public_sources.json"
RAW_ROOT = CAMPAIGN / "data/raw/hepdata"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, target: Path) -> None:
    response = requests.get(url, timeout=180)
    response.raise_for_status()
    target.write_bytes(response.content)


def safe_extract(archive: Path, target: Path) -> None:
    target = target.resolve()
    with tarfile.open(archive, "r:*") as tar:
        members = tar.getmembers()
        for member in members:
            resolved = (target / member.name).resolve()
            if resolved != target and target not in resolved.parents:
                raise RuntimeError(f"unsafe archive member: {member.name}")
        tar.extractall(target, filter="data")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--record", action="append")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    wanted = set(args.record or [item["record"] for item in config["records"]])
    records = [item for item in config["records"] if item["record"] in wanted]
    if wanted != {item["record"] for item in records}:
        raise SystemExit(f"unknown records: {sorted(wanted - {item['record'] for item in records})}")
    results = []
    for source in records:
        recid = source["record"]
        out = RAW_ROOT / recid
        out.mkdir(parents=True, exist_ok=True)
        record_json = out / "record.json"
        archive = out / "submission_csv.tar.gz"
        tables = out / "tables"
        if args.force or not record_json.exists():
            fetch(f"https://www.hepdata.net/record/{recid}?format=json", record_json)
        if args.force or not archive.exists():
            fetch(f"https://www.hepdata.net/download/submission/{recid}/1/csv", archive)
        if args.force or not tables.exists():
            if tables.exists():
                shutil.rmtree(tables)
            tables.mkdir()
            safe_extract(archive, tables)
        metadata = json.loads(record_json.read_text())
        csv_tables = sorted(str(p.relative_to(tables)) for p in tables.rglob("*.csv"))
        results.append({
            "record": recid,
            "version": metadata.get("version"),
            "title": metadata.get("record", {}).get("title"),
            "record_url": f"https://www.hepdata.net/record/{recid}",
            "download_url": f"https://www.hepdata.net/download/submission/{recid}/1/csv",
            "archive": str(archive.relative_to(CAMPAIGN)),
            "archive_sha256": digest(archive),
            "csv_table_count": len(csv_tables),
            "csv_tables": csv_tables,
        })
    target = CAMPAIGN / "data/hepdata_download_manifest.json"
    if args.record and target.exists():
        previous = json.loads(target.read_text()).get("records", [])
        by_record = {item["record"]: item for item in previous}
        by_record.update({item["record"]: item for item in results})
        results = [by_record[key] for key in sorted(by_record)]
    report = {"campaign": "sidis_global_analysis_2026", "status": "public_hepdata_records_downloaded", "records": results, "production_authorized": False}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "records": len(results)}, indent=2))


if __name__ == "__main__":
    main()
