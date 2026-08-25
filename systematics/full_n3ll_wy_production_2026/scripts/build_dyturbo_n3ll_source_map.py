#!/usr/bin/env python3
"""Record the external DYTurbo coefficient and convention provenance.

The map is intentionally an audit artifact, not an imported production
implementation.  It identifies the source locations that must be translated
into the candidate W backend and records which pieces are observable-only in
the current stage.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


SYSTEMATICS = Path(__file__).resolve().parents[2]
PROJECT = SYSTEMATICS.parent
OUT = SYSTEMATICS / "full_n3ll_wy_production_2026/reports/dyturbo_n3ll_source_map.json"


def file_record(path: Path, line_ranges: list[list[int]], symbols: list[str], role: str) -> dict:
    text = path.read_text()
    lines = text.splitlines()
    excerpts = []
    for lo, hi in line_ranges:
        excerpts.append({"start": lo, "end": hi, "text": "\n".join(lines[lo - 1 : hi])})
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "line_ranges": line_ranges,
        "symbols": symbols,
        "role": role,
        "excerpts": excerpts,
    }


def main() -> None:
    sources = [
        file_record(
            Path("/home/dustin/src/dyturbo-1.4.2/resum/resconst.C"),
            [[70, 105], [133, 163], [175, 190], [217, 228], [260, 322]],
            ["beta3", "beta4", "A1q", "A2q", "A3q", "A4q", "B1q", "B2q", "B3q", "H2q", "C2qqn"],
            "quark Sudakov, hard, and delta matching constants",
        ),
        file_record(
            Path("/home/dustin/src/dyturbo-1.4.2/resum/resconst.h"),
            [[35, 52]],
            ["A1q-A6q", "B1q-B5q", "H1q-H4q"],
            "coefficient declarations and perturbative-order interface",
        ),
        file_record(
            Path("/home/dustin/src/dyturbo-1.4.2/resum/ccoeff.h"),
            [[1, 45]],
            ["C1qq_delta", "C2qq_delta", "C3qq_delta"],
            "TMD matching coefficient declarations",
        ),
        file_record(
            Path("/home/dustin/src/dyturbo-1.4.2/resum/hcoeff.h"),
            [[1, 55]],
            ["hard coefficient arrays"],
            "hard-function coefficient declarations",
        ),
        file_record(
            Path("/home/dustin/src/dyturbo-1.4.2/src/settings.C"),
            [[650, 700], [920, 935]],
            ["order_vjet", "primed", "doVJREAL", "doVJVIRT", "intDimVJ"],
            "DYTurbo perturbative-order and NNLO V+jet runtime convention",
        ),
    ]
    status = {
        "status": "external_dyturbo_n3ll_nnlo_source_map_complete",
        "target": "unprimed N3LL+NNLO W+Y",
        "standard_counting": {
            "cusp": "alpha_s^4",
            "non_cusp": "alpha_s^3",
            "hard_and_collinear_boundary": "alpha_s^2",
            "Y_matching": "FO_NNLO_minus_ASY_NNLO",
        },
        "sources": sources,
        "runtime_convention": {
            "unprimed_order": 3,
            "primed": False,
            "why": "DYTurbo sets order_vjet=max(0, order-1) for primed=false, capped at 2; order=3 is therefore NNLO V+jet and enables real/virtual NNLO pieces",
            "cut_integration": "NNLO V+jet with cuts requires non-quadrature integration and intDimVJ >= 7; candidate probe uses intDimVJ=-1",
        },
        "translation_state": {
            "external_coefficients_available": True,
            "imported_into_fitted_v22_w": False,
            "same_scheme_asy_nnlo_implemented": False,
            "promotion_authorized": False,
        },
        "production_outputs_modified": False,
    }
    OUT.write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps({k: status[k] for k in ("status", "target", "translation_state")}, indent=2))


if __name__ == "__main__":
    main()
