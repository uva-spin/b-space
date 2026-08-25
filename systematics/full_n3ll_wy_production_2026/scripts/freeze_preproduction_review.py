#!/usr/bin/env python3
"""Hash-lock the isolated review inputs before the final W+Y production run.

This manifest is deliberately separate from the frozen lambda=1 package and
does not copy, overwrite, or promote any production result.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "manifests/preproduction_review_freeze_20260819.json"

FILES = [
    "manifests/frozen_lambda1_baseline.json",
    "manifests/tevatron_n3ll_nnlo_wy_candidate_freeze.json",
    "manifests/tevatron_n3ll_nnlo_wy_final_launch.json",
    "reports/accuracy_closure_g1_1p017.json",
    "reports/accuracy_closure_v2.json",
    "reports/dyturbo_n3ll_source_map.json",
    "reports/dyturbo_term_decomposition_g1_1p017/term_decomposition_status.json",
    "reports/tevatron_n3ll_nnlo_wy_stationarity_g1_1p017_seed_20260820/stationarity_status.json",
    "reports/tevatron_external_seed_stability.json",
    "reports/tevatron_g1_direct_profile_decision.json",
    "reports/fixed_target_quadrature_probes/fixed_target_normalization_audit.json",
    "reports/tevatron_353_scope_audit.json",
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    artifacts = []
    missing = []
    for rel in FILES:
        path = BASE / rel
        if not path.exists():
            missing.append(rel)
            continue
        artifacts.append({"path": str(path), "sha256": digest(path)})
    if missing:
        raise SystemExit("missing freeze inputs: " + ", ".join(missing))
    payload = {
        "status": "isolated_preproduction_review_inputs_frozen_not_promoted",
        "freeze_date": "2026-08-19",
        "purpose": "hash-lock the validated candidate inputs before the new genuine Tevatron W+Y production run",
        "candidate": "unprimed N3LL+NNLO with conventional Y=FO_NNLO-ASY_NNLO",
        "scope_at_freeze": "122 Tevatron qT rows plus 24 Tevatron boundary oracle rows; fixed-target and LHCb remain fail-closed",
        "artifacts": artifacts,
        "frozen_lambda1_baseline_unchanged": True,
        "production_outputs_modified": False,
        "promotion_authorized": False,
        "active_run": "reports/tevatron_n3ll_nnlo_wy_final_g1_1p017/",
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
