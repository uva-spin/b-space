#!/usr/bin/env python3
"""Build a read-only provenance manifest for perturbative completion work."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "systematics" / "perturbative_provenance_completion"
OUT = WORK / "manifests" / "input_manifest.json"

FILES = [
    ROOT / "v22" / "backends" / "bt_internal_css_backend_v22_full.py",
    ROOT / "v22" / "backends" / "bt_internal_css_backend_v22_scheme_y.py",
    ROOT / "bt_internal_css_backend_v19_smoothprofile.py",
    ROOT / "v22" / "src" / "dy_w_nlo_reference.py",
    ROOT / "v22" / "src" / "dy_hard_nlo.py",
    ROOT / "v22" / "src" / "css2_ope_nlo.py",
    ROOT / "v22" / "src" / "css2_ope_nlo_general.py",
    ROOT / "v22" / "src" / "small_b_profile.py",
    ROOT / "b-space-public" / "workflows" / "v22" / "utilities" / "export_v22_backend_cache.sh",
    ROOT / "production_frozen" / "v22_lambda3_50rep_DYonly_bspace" / "backend_cache" / "metadata_v22full_n3llp_nloQ96_b160_qToQ05.json",
    ROOT / "production_frozen" / "v22_lambda3_50rep_DYonly_bspace" / "backend_cache" / "cache_paths.json",
    ROOT / "production_frozen" / "v22_lambda3_50rep_DYonly_bspace" / "PRODUCTION_MANIFEST.json",
    ROOT / "systematics" / "dataset_identifiability_campaign_2026" / "production_lambda1_empirical_reference_full96x50" / "PRODUCTION_MANIFEST.json",
    ROOT / "systematics" / "dataset_identifiability_campaign_2026" / "production_lambda1_empirical_reference_full96x50" / "PRODUCTION_AUDIT.json",
    ROOT / "b-space-public" / "production" / "lambda1_empirical_reference_full96x50" / "PRODUCTION_MANIFEST.json",
    ROOT / "b-space-public" / "docs" / "MATCHING.md",
    ROOT / "b-space-public" / "README.md",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT / "b-space-public"), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    records = []
    missing = []
    for path in FILES:
        if not path.exists():
            missing.append(str(path))
            continue
        records.append(
            {
                "path": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    manifest = {
        "status": "input_manifest_only_no_production_changes",
        "created": "2026-08-15",
        "workspace": str(WORK.relative_to(ROOT)),
        "published_source_commit": git_commit(),
        "missing_inputs": missing,
        "files": records,
        "frozen_inputs_read_only": True,
        "production_outputs_modified": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
