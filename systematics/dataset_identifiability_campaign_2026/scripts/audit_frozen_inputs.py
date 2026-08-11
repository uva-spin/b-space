#!/usr/bin/env python3
"""Verify that every registered immutable campaign input is unchanged."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
MANIFEST = BASE / "manifests/input_files.json"
TARGET = BASE / "summaries/frozen_input_audit"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    registered = json.loads(MANIFEST.read_text())["files"]
    failures = []
    for name, expected in registered.items():
        path = Path(name)
        if not path.exists():
            failures.append({"path": name, "failure": "missing"})
            continue
        observed_hash = digest(path)
        observed_bytes = path.stat().st_size
        if (observed_hash != expected["sha256"]
                or observed_bytes != expected["bytes"]):
            failures.append({
                "path": name,
                "failure": "content_changed",
                "expected_sha256": expected["sha256"],
                "observed_sha256": observed_hash,
                "expected_bytes": expected["bytes"],
                "observed_bytes": observed_bytes,
            })
    summary = {
        "status": "pass" if not failures else "fail",
        "registered_input_count": len(registered),
        "unchanged_input_count": len(registered) - len(failures),
        "failures": failures,
        "manifest": str(MANIFEST),
        "production_sources_modified": bool(failures),
    }
    TARGET.mkdir(parents=True, exist_ok=True)
    (TARGET / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
