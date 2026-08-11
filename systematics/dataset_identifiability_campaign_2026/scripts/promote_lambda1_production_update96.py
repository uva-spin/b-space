#!/usr/bin/env python3
"""Publish the validated 96-start lambda=1 package additively.

The existing 24-start package is retained as an immutable rollback reference;
this transaction creates a new active package and updates the study registry.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import time


BASE = Path(__file__).resolve().parents[1]
REGISTRY = BASE / "summaries/champion_registry"
START = BASE / "summaries/lambda1_start_expansion96/summary.json"
CROSSED = BASE / "summaries/matched_baseline_reference_distance_lam1e00_full96_crossed_experimental"
PREVIEW = REGISTRY / "lambda1_96start_update_preview.json"
FIGURES = REGISTRY / "current_fig2_fig6_96start"
OLD_PACKAGE = BASE / "production_lambda1_empirical_reference"
TARGET = BASE / "production_lambda1_empirical_reference_full96x50"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> None:
    start = json.loads(START.read_text())
    crossed = json.loads((CROSSED / "summary.json").read_text())
    preview = json.loads(PREVIEW.read_text())
    if start["status"] != "complete" or not start["all_new_starts_pass_fnp_stationarity_gate"]:
        raise RuntimeError("96-start stationarity audit is incomplete")
    if crossed["start_count"] != 96 or crossed["experimental_replica_count"] != 50:
        raise RuntimeError("crossed ensemble is not 96 starts x 50 replicas")
    required = {
        "combined_summary": CROSSED / "summary.json",
        "bspace_combined_bands": CROSSED / "bspace_combined_bands.csv",
        "kspace_combined_bands": CROSSED / "kspace_combined_bands.csv",
        "start_only_summary": START,
        "start_only_handoff": BASE / "summaries/lambda1_start_expansion96/HANDOFF.md",
        "fig2_png": FIGURES / "champion_fig2space_bT_ud_combined_1sigma.png",
        "fig2_pdf": FIGURES / "champion_fig2space_bT_ud_combined_1sigma.pdf",
        "fig6_png": FIGURES / "champion_fig6_kT_ud_combined_1sigma.png",
        "fig6_pdf": FIGURES / "champion_fig6_kT_ud_combined_1sigma.pdf",
    }
    destination_names = {
        "combined_summary": "combined_summary.json",
        "bspace_combined_bands": "bspace_combined_bands.csv",
        "kspace_combined_bands": "kspace_combined_bands.csv",
        "start_only_summary": "start_only_summary.json",
        "start_only_handoff": "start_only_HANDOFF.md",
        "fig2_png": "champion_fig2space_bT_ud_combined_1sigma.png",
        "fig2_pdf": "champion_fig2space_bT_ud_combined_1sigma.pdf",
        "fig6_png": "champion_fig6_kT_ud_combined_1sigma.png",
        "fig6_pdf": "champion_fig6_kT_ud_combined_1sigma.pdf",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing update artifacts: {missing}")
    if TARGET.exists():
        raise RuntimeError(f"refusing to overwrite existing package: {TARGET}")

    prior_registry = json.loads((REGISTRY / "current.json").read_text())
    prior_manifest_path = OLD_PACKAGE / "PRODUCTION_MANIFEST.json"
    prior_manifest = json.loads(prior_manifest_path.read_text()) if prior_manifest_path.is_file() else {}
    staging = TARGET.with_name(f".{TARGET.name}.staging.{os.getpid()}")
    staging.mkdir(parents=True, exist_ok=False)
    try:
        copied = {}
        for name, source in required.items():
            destination = staging / destination_names[name]
            shutil.copy2(source, destination)
            copied[name] = destination
        audit = {
            "status": "pass_for_96start_production_update",
            "production_id": "lambda1_empirical_reference_full96x50",
            "method": preview["method"],
            "start_count": 96,
            "experimental_replica_count": 50,
            "crossed_member_count_per_flavor": 4800,
            "start_only_stationarity": {
                "all_new_starts_pass_fnp_stationarity_gate": start["all_new_starts_pass_fnp_stationarity_gate"],
                "bspace_width_ratio_96_to_48": start["bspace_width_ratio_96_to_48"],
                "kspace_width_ratio_96_to_48": start["kspace_width_ratio_96_to_48"],
            },
            "combined_metrics": crossed["metrics"],
            "band_interpretation": crossed["band_interpretation"],
            "replicas_launched_by_update": False,
            "frozen_production_outputs_overwritten": False,
            "rollback_package": str(OLD_PACKAGE),
            "source_hashes": {name: sha256(path) for name, path in required.items()},
        }
        atomic_json(staging / "PRODUCTION_AUDIT.json", audit)
        manifest = {
            "status": "production_active",
            "production_id": "lambda1_empirical_reference_full96x50",
            "promoted_at_epoch": time.time(),
            "method": preview["method"],
            "audit": str(TARGET / "PRODUCTION_AUDIT.json"),
            "audit_sha256": sha256(staging / "PRODUCTION_AUDIT.json"),
            "artifact_sources": {name: str(path) for name, path in required.items()},
            "artifact_sha256": {name: sha256(path) for name, path in copied.items()},
            "rollback_reference": {
                "prior_production_package": str(OLD_PACKAGE),
                "prior_registry": str(REGISTRY / "current.json"),
                "prior_production_id": prior_manifest.get("production_id", "lambda1_empirical_reference_full24x50"),
                "frozen_production_outputs_overwritten": False,
            },
            "limitations": [
                "empirical reference median is not reciprocal-cross-fitted",
                "combined q16-q84 interval is operational and has no calibrated confidence-level interpretation",
                "96-start non-uniqueness completeness is established for the declared perturbation family and objective",
            ],
            "production_sources_modified_by_transaction": False,
        }
        atomic_json(staging / "PRODUCTION_MANIFEST.json", manifest)
        os.replace(staging, TARGET)

        record = dict(preview)
        record.update({
            "status": "production_active",
            "production_manifest": str(TARGET / "PRODUCTION_MANIFEST.json"),
            "production_audit": str(TARGET / "PRODUCTION_AUDIT.json"),
            "production_sources_modified": False,
            "prior_production_package": str(OLD_PACKAGE),
            "artifact_sha256": {name: sha256(path) for name, path in required.items()},
        })
        atomic_json(REGISTRY / "current.json", record)
        atomic_json(REGISTRY / "empirical_reference_lambda1_b0p1_2p0_full96.json", record)
        atomic_json(TARGET / "PRODUCTION_MANIFEST.json", manifest)
        print(json.dumps(manifest, indent=2))
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


if __name__ == "__main__":
    main()
