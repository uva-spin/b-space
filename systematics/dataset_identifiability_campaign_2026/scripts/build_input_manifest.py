#!/usr/bin/env python3
"""Resolve and hash immutable inputs for the identifiability campaign."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
SYSTEMATICS = BASE.parent
PROJECT = SYSTEMATICS.parent
REGISTRY = BASE / "config/dataset_registry.json"
TARGET = BASE / "manifests"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def resolve(relative: str) -> Path:
    return (BASE / relative).resolve()


def main() -> None:
    registry = json.loads(REGISTRY.read_text())
    rows = []
    files = {}
    for candidate in registry["candidates"]:
        output_text = candidate.get("central_output")
        output = resolve(output_text) if output_text else None
        predictions = output / "predictions.csv" if output else None
        metrics = output / "metrics.json" if output else None
        observed_rows = None
        if predictions and predictions.exists():
            table = pd.read_csv(predictions)
            observed_rows = len(table)
            files[str(predictions)] = {
                "sha256": digest(predictions),
                "bytes": predictions.stat().st_size,
            }
        if metrics and metrics.exists():
            files[str(metrics)] = {
                "sha256": digest(metrics),
                "bytes": metrics.stat().st_size,
            }
        expected = candidate["expected_rows"]
        rows.append({
            "candidate_id": candidate["id"],
            "label": candidate["label"],
            "theory_model": candidate["theory_model"],
            "expected_rows": expected,
            "observed_rows": observed_rows,
            "row_count_pass": observed_rows == expected if observed_rows is not None else None,
            "central_output": str(output) if output else "",
            "central_output_exists": bool(output and output.exists()),
            "maturity": candidate.get("maturity", "central_output_available"),
        })

    auxiliary = [
        SYSTEMATICS / "high_qt_direct_production_benchmark/README.md",
        SYSTEMATICS / "high_qt_direct_production_benchmark/config/promotion_policy.json",
        SYSTEMATICS / "high_qt_direct_production_benchmark/summaries/high_qt_candidate_inventory.csv",
        SYSTEMATICS / "finite_y_tail_benchmark/summaries/tail_benchmark_row_gate.csv",
        SYSTEMATICS / "production_candidate_v23a_collins_q020_tailbench/PRODUCTION_CANDIDATE_MANIFEST.json",
    ]
    for path in auxiliary:
        files[str(path)] = {"sha256": digest(path), "bytes": path.stat().st_size}

    TARGET.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(TARGET / "dataset_candidates.csv", index=False)
    (TARGET / "input_files.json").write_text(json.dumps({
        "schema_version": 1,
        "project": str(PROJECT),
        "systematics": str(SYSTEMATICS),
        "files": files,
    }, indent=2) + "\n")

    failures = [
        row["candidate_id"] for row in rows
        if row["row_count_pass"] is False
    ]
    summary = {
        "candidate_count": len(rows),
        "resolved_central_output_count": sum(row["central_output_exists"] for row in rows),
        "row_count_failures": failures,
        "status": "pass" if not failures else "fail",
    }
    (TARGET / "manifest_status.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
