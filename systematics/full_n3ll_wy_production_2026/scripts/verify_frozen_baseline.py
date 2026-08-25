#!/usr/bin/env python3
"""Verify the hash-locked lambda=1 baseline without writing any files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


# Artifact paths in the freeze record are relative to the systematics root.
ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).resolve().parents[1] / "manifests" / "frozen_lambda1_baseline.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    record = json.loads(MANIFEST.read_text())
    results = []
    passed = True
    for item in record["artifacts"]:
        path = ROOT / item["path"]
        exists = path.is_file()
        actual = sha256(path) if exists else None
        ok = exists and actual == item["sha256"]
        passed = passed and ok
        results.append({
            "path": item["path"],
            "exists": exists,
            "expected_sha256": item["sha256"],
            "actual_sha256": actual,
            "pass": ok,
        })
    output = {
        "status": "frozen_baseline_verified" if passed else "frozen_baseline_hash_mismatch",
        "baseline_id": record["baseline_id"],
        "artifact_count": len(results),
        "pass_count": sum(int(row["pass"]) for row in results),
        "results": results,
        "writes_performed": False,
    }
    print(json.dumps(output, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
