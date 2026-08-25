#!/usr/bin/env python3
"""Read-only audit of the current backend/cache provenance boundary."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "systematics" / "perturbative_provenance_completion"
REPORT = WORK / "reports" / "backend_provenance_audit.json"
FULL = ROOT / "v22" / "backends" / "bt_internal_css_backend_v22_full.py"
BASE = ROOT / "bt_internal_css_backend_v19_smoothprofile.py"
META = ROOT / "production_frozen" / "v22_lambda3_50rep_DYonly_bspace" / "backend_cache" / "metadata_v22full_n3llp_nloQ96_b160_qToQ05.json"
PUBLIC_FULL = ROOT / "b-space-public" / "v22" / "backends" / "bt_internal_css_backend_v22_full.py"


def source_constant(name: str):
    tree = ast.parse(FULL.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == name:
                return ast.literal_eval(node.value)
    return None


def main() -> None:
    metadata = json.loads(META.read_text())
    full_text = FULL.read_text()
    base_text = BASE.read_text()
    metadata_config = metadata.get("config", {})
    metadata_args = metadata.get("args", {})

    report = {
        "status": "historical_cache_metadata_gap_diagnostic_complete",
        "workspace": str(WORK.relative_to(ROOT)),
        "backend_declared_w_organization": source_constant("V22_W_ORGANIZATION"),
        "backend_entrypoint_default_is_multiplicative": "Backend entry point: multiplicative v22 W by default." in full_text,
        "strict_branch_present": 'mode in {"strict", "strict_nlo"}' in full_text,
        "cache_records_explicit_w_organization": any(
            "w_organization" in str(key).lower()
            for key in [*metadata_config.keys(), *metadata_args.keys()]
        ),
        "cache_metadata_script": metadata.get("script"),
        "cache_resum_order": metadata_config.get("resum_order"),
        "cache_match_order": metadata_config.get("match_order"),
        "cache_y_mode": metadata_config.get("y_mode"),
        "cache_backend_script": metadata_args.get("backend_script"),
        "legacy_n3llp_alias_present": 'order in {"n3llp", "n3ll_prime", "n3llprime", "n3llp_pilot"}' in base_text,
        "legacy_n3llp_same_as_nnll": "Deliberately same as nnll until the N3LL' hard/OPE/matching audit is done." in base_text,
        "implemented_sudakov_symbols": sorted(set(re.findall(r"\b(A[1-4]|B[1-4])\b", base_text))),
        "public_backend_byte_identical": FULL.read_bytes() == PUBLIC_FULL.read_bytes(),
        "frozen_inputs_read_only": True,
        "production_outputs_modified": False,
        "next_required_action": "complete accuracy inventory and expansion closure; retain historical cache as read-only",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
