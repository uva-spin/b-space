#!/usr/bin/env python3
"""Build a read-only perturbative-accuracy inventory for the new W+Y campaign.

This deliberately inspects source text and records what is actually present.
Configuration labels such as ``n3llp`` are not treated as evidence that the
corresponding hard, OPE, and matching coefficients exist.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


SYSTEMATICS = Path(__file__).resolve().parents[2]
PROJECT = SYSTEMATICS.parent
TARGET = SYSTEMATICS / "full_n3ll_wy_production_2026" / "reports" / "accuracy_inventory_v1.json"

SOURCES = {
    "css_backend": PROJECT / "bt_internal_css_backend_v19_smoothprofile.py",
    "v22_full_w_backend": PROJECT / "v22/backends/bt_internal_css_backend_v22_full.py",
    "v22_scheme_y_backend": PROJECT / "v22/backends/bt_internal_css_backend_v22_scheme_y.py",
    "dy_hard_nlo": PROJECT / "v22/src/dy_hard_nlo.py",
    "css2_ope_nlo": PROJECT / "v22/src/css2_ope_nlo.py",
    "css2_ope_general_nlo": PROJECT / "v22/src/css2_ope_nlo_general.py",
    "dyturbo_rescoeff": Path("/home/dustin/src/dyturbo-1.4.2/resum/rescoeff.f"),
    "dyturbo_resconst": Path("/home/dustin/src/dyturbo-1.4.2/resum/resconst.h"),
    "dyturbo_hcoeff": Path("/home/dustin/src/dyturbo-1.4.2/resum/hcoeff.h"),
    "dyturbo_ccoeff": Path("/home/dustin/src/dyturbo-1.4.2/resum/ccoeff.h"),
    "dyturbo_gamma4": Path("/home/dustin/src/dyturbo-1.4.2/resum/gamma4sg.h"),
    "dyturbo_binary": Path("/home/dustin/src/dyturbo-1.4.2/bin/dyturbo"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def line_of(text: str, pattern: str) -> int | None:
    match = re.search(pattern, text, flags=re.MULTILINE)
    if match is None:
        return None
    return text.count("\n", 0, match.start()) + 1


def main() -> None:
    texts = {}
    for name, path in SOURCES.items():
        if not path.exists() or path.suffix in {"", ".so"}:
            continue
        try:
            texts[name] = path.read_text()
        except UnicodeDecodeError:
            continue
    missing = sorted(name for name, path in SOURCES.items() if not path.exists())

    css = texts["css_backend"]
    v22_full = texts["v22_full_w_backend"]
    scheme_y = texts["v22_scheme_y_backend"]
    dyturbo_resconst = texts.get("dyturbo_resconst", "")
    dyturbo_ccoeff = texts.get("dyturbo_ccoeff", "")

    source_records = {
        name: {
            "path": str(path.relative_to(PROJECT)) if path.is_relative_to(PROJECT) else str(path),
            "sha256": sha256(path) if path.exists() else None,
            "exists": path.exists(),
        }
        for name, path in SOURCES.items()
    }

    order_alias_line = line_of(css, r"if order in \{\"n3llp\"")
    pilot_same_line = line_of(css, r"Deliberately same as nnll")
    y_zero_line = line_of(css, r"y_mode: str = \"zero\"")
    y_warning_line = line_of(css, r"Not a final N3LL'")
    y_dev_line = line_of(css, r"First self-contained development Y_NLO")

    report = {
        "status": "accuracy_inventory_complete_for_current_source_not_target",
        "campaign": "full_n3ll_wy_production_2026",
        "candidate": "tevatron_n3ll_nnlo_wy_v1",
        "generated_at": "2026-08-18",
        "scope": {
            "primary_rows": 353,
            "low_qt_core_rows": 329,
            "tevatron_boundary_rows": 24,
            "excluded_rows": "LHCb high-qT rows 10-13 pending fixed-order observable/covariance closure",
        },
    "declared_target": {
        "resummation": "unprimed N3LL",
        "fixed_order": "NNLO",
        "matching": "conventional additive W_N3LL + (FO_NNLO - ASY_NNLO)",
            "primed": False,
        },
        "current_source_findings": {
            "sudakov": {
                "A_coefficients": ["A1", "A2", "A3"],
                "B_coefficients": ["B1", "B2"],
                "declared_n3ll_pilot_aliases": True,
                "n3ll_pilot_same_coefficients_as_nnll": True,
                "evidence": {
                    "alias_line": order_alias_line,
                    "same_as_nnll_line": pilot_same_line,
                },
            },
            "hard_and_ope": {
                "hard_source": "one-loop NLO hard module in the current backend",
                "ope_source": "one-loop NLO OPE module(s) in the current backend",
                "alpha_s2_hard_constants_verified": False,
                "alpha_s2_ope_coefficients_verified": False,
                "full_N3LL_hard_ope_inventory": False,
            },
            "w_organization": {
                "default_v22_organization": "multiplicative_nlo",
                "strict_one_loop_branch_present": True,
                "evidence": {
                    "default_line": line_of(v22_full, r"V22_W_ORGANIZATION"),
                    "strict_branch_line": line_of(v22_full, r"mode in \{\"strict\""),
                },
            },
            "finite_y": {
                "default_y_mode": "zero",
                "development_path": "FO_real_dev - singular_dev with a switch/damping prescription",
                "conventional_FO_minus_ASY_production_path": False,
                "same_scheme_ASY_closure_complete": False,
                "evidence": {
                    "y_mode_line": y_zero_line,
                    "development_y_line": y_dev_line,
                    "nonfinal_warning_line": y_warning_line,
                    "scheme_y_wexp_numeric": line_of(scheme_y, r"Complete strict one-loop expansion"),
                },
            },
            "external_dyturbo": {
                "engine_path": "/home/dustin/src/dyturbo-1.4.2/bin/dyturbo",
                "supports_unprimed_n3ll_and_nnlo_fixed_order": True,
                "four_loop_cusp_symbols_present": bool(re.search(r"A4q|A4g", dyturbo_resconst)),
                "three_loop_non_cusp_symbols_present": bool(re.search(r"B3q|B3g", dyturbo_resconst)),
                "alpha_s2_hard_and_matching_symbols_present": bool(re.search(r"C2qq|H2", dyturbo_ccoeff)),
                "note": "External engine is an observable-level NNLO/N3LL oracle; its coefficients are not yet imported into the fitted v22 W backend.",
            },
        },
        "required_before_promotion": [
            "freeze the explicit unprimed N3LL convention table and coefficient sources",
            "supply and test the missing higher-order hard/OPE/evolution ingredients required by that convention",
            "derive ASY_NNLO from the same W scheme and normalization used by the exact transform",
            "validate independent fiducial NNLO FO against the 24 Tevatron boundary observables",
            "demonstrate conventional W+Y continuity, positivity, refinement stability, and FO approach",
            "only then refit/propagate replicas and start variation",
        ],
        "known_nonproduction_labels": [
            "resum_order=n3llp/n3ll_pilot is explicitly a pilot alias, not proof of full N3LL",
            "match_order=nlo/nlo_dev is a development finite-tail path, not a promoted conventional Y",
            "y_mode=zero is W-only in the strict low-qT core",
        ],
        "source_records": source_records,
        "missing_sources": missing,
        "frozen_baseline_untouched": True,
        "production_outputs_modified": False,
        "promotion_authorized": False,
    }
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
