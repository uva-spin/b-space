#!/usr/bin/env python3
"""Hash protected q020 production references for experimental isolation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
HERE = ROOT / "systematics/high_qt_direct_production_benchmark/experimental_matched_y"
REFERENCES = [
    ROOT / "v23/backends/bt_internal_css_backend_v22_tevatron.py",
    ROOT / "systematics/production_candidate_v23a_collins_q020_tailbench/PRODUCTION_CANDIDATE_MANIFEST.json",
    ROOT / "systematics/collins_factorization_validity/outputs/rowidfix_stageFT_all_qmax0p20_lam0p50_central_s303/predictions.csv",
    ROOT / "systematics/collins_factorization_validity/replicas/rowidfix_stageFT_all_qmax0p20_lam0p50_lambda3_50rep/audit_convergence_q95/lambda3_ensemble_q95_summary.json",
]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    missing = [str(path) for path in REFERENCES if not path.exists()]
    if missing:
        raise SystemExit(f"Missing protected references: {missing}")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "read_only_reference; experimental work must not overwrite these files",
        "references": [{"path": str(path.relative_to(ROOT)), "sha256": digest(path), "size": path.stat().st_size}
                       for path in REFERENCES],
    }
    out = HERE / "manifests/protected_production_references.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
