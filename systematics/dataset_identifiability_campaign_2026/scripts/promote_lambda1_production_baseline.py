#!/usr/bin/env python3
"""Atomically package the audited lambda=1 champion as production baseline.

The package is campaign-local and additive. Existing frozen production outputs
are never overwritten; the manifest records them as the rollback reference.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import time


BASE = Path(__file__).resolve().parents[1]
SUMMARIES = BASE / "summaries"
REGISTRY = SUMMARIES / "champion_registry"
AUDIT = SUMMARIES / "lambda1_productionization_audit/summary.json"
TARGET = BASE / "production_lambda1_empirical_reference"
SOURCE = SUMMARIES / "matched_baseline_reference_distance_lam1e00_full24_crossed_experimental"
FIGURES = REGISTRY / "current_fig2_fig6"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> None:
    if not AUDIT.is_file():
        raise RuntimeError("run audit_lambda1_productionization.py first")
    audit = json.loads(AUDIT.read_text())
    if audit["status"] != "pass_for_productionization_with_documented_limitations":
        raise RuntimeError(f"audit is not promotable: {audit['status']}")
    if TARGET.exists():
        manifest = TARGET / "PRODUCTION_MANIFEST.json"
        if manifest.is_file() and json.loads(manifest.read_text()).get("status") == "production_active":
            print(json.dumps(json.loads(manifest.read_text()), indent=2))
            return
        raise RuntimeError(f"refusing to overwrite existing target: {TARGET}")

    staging = TARGET.with_name(f".{TARGET.name}.staging.{os.getpid()}")
    staging.mkdir(parents=True, exist_ok=False)
    copied = {
        "combined_summary": SOURCE / "summary.json",
        "bspace_combined_bands": SOURCE / "bspace_combined_bands.csv",
        "kspace_combined_bands": SOURCE / "kspace_combined_bands.csv",
        "fig2_png": FIGURES / "champion_fig2space_bT_ud_combined_1sigma.png",
        "fig2_pdf": FIGURES / "champion_fig2space_bT_ud_combined_1sigma.pdf",
        "fig6_png": FIGURES / "champion_fig6_kT_ud_combined_1sigma.png",
        "fig6_pdf": FIGURES / "champion_fig6_kT_ud_combined_1sigma.pdf",
    }
    try:
        for name, source in copied.items():
            if not source.is_file():
                raise RuntimeError(f"missing source artifact: {source}")
            shutil.copy2(source, staging / source.name)
        manifest = {
            "status": "production_active",
            "production_id": "lambda1_empirical_reference_full24x50",
            "promoted_at_epoch": time.time(),
            "method": audit["method"],
            "audit": str(AUDIT),
            "audit_sha256": sha256(AUDIT),
            "artifact_sources": {name: str(path) for name, path in copied.items()},
            "artifact_sha256": {name: sha256(path) for name, path in copied.items()},
            "rollback_reference": {
                "prior_campaign_registry": str(REGISTRY / "current.json"),
                "prior_status": "provisional_complete_24start_champion_not_production",
                "frozen_production_outputs_overwritten": False,
            },
            "limitations": audit["limitations"],
            "production_sources_modified_by_transaction": False,
        }
        atomic_json(staging / "PRODUCTION_MANIFEST.json", manifest)
        atomic_json(staging / "PRODUCTION_AUDIT.json", audit)
        os.replace(staging, TARGET)

        current_path = REGISTRY / "current.json"
        current = json.loads(current_path.read_text())
        current["status"] = "production_active"
        current["production_manifest"] = str(TARGET / "PRODUCTION_MANIFEST.json")
        current["production_audit"] = str(TARGET / "PRODUCTION_AUDIT.json")
        current["prior_status"] = "provisional_complete_24start_champion_not_production"
        current["production_sources_modified"] = False
        atomic_json(current_path, current)
        champion_path = REGISTRY / "empirical_reference_lambda1_b0p1_2p0_full24.json"
        champion = json.loads(champion_path.read_text())
        champion.update({
            "status": "production_active",
            "production_manifest": str(TARGET / "PRODUCTION_MANIFEST.json"),
            "production_audit": str(TARGET / "PRODUCTION_AUDIT.json"),
            "prior_status": "provisional_complete_24start_champion_not_production",
            "production_sources_modified": False,
        })
        atomic_json(champion_path, champion)
        print(json.dumps(manifest, indent=2))
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


if __name__ == "__main__":
    main()
