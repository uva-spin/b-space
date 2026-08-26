#!/usr/bin/env python3
"""Inventory public arXiv ancillary and TeX SIDIS tables.

This driver is intentionally an inventory/conversion boundary.  It parses the
machine-readable CLAS ancillary files and counts labelled TeX rows from the
Hall-C and Hall-A source packages, but it does not create fit rows.  Absolute
cross sections, radiative corrections, nuclear targets, and azimuthal
dependences need source-specific observable and covariance closure first.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any


CAMPAIGN = Path(__file__).resolve().parents[1]
RAW_ROOT = CAMPAIGN / "data" / "raw" / "global"
DEFAULT_OUTPUT = CAMPAIGN / "reports" / "arxiv_table_inventory.json"
DEFAULT_MARKDOWN = CAMPAIGN / "reports" / "arxiv_table_inventory.md"

CLAS_TABLES = {
    "pip_sidis_cs_table_zh_pt2.dat": {
        "axes": ["Q2", "x", "z", "pT2", "phi"],
        "observable": "ds/dx dQ2 dz dpT2 dphi",
    },
    "pip_sidis_cs_table_xf_pt2.dat": {
        "axes": ["Q2", "x", "xF", "pT2", "phi"],
        "observable": "ds/dx dQ2 dxF dpT2 dphi",
    },
    "pip_sidis_cs_table_zh_t.dat": {
        "axes": ["Q2", "x", "z", "|t|", "phi"],
        "observable": "ds/dx dQ2 dz dt dphi",
    },
    "pip_sidis_cs_table_zg_v.dat": {
        "axes": ["Q2", "x", "zG", "v", "phi"],
        "observable": "ds/dx dQ2 dz dv dphi",
    },
}

FLOAT_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?$")


def _float(token: str) -> float:
    return float(token.replace("D", "E").replace("d", "e"))


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(CAMPAIGN))
    except ValueError:
        return str(path)


def profile_clas(path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    """Parse a CLAS whitespace ancillary file without rewriting its rows."""

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header_index = next(
        (i for i, line in enumerate(lines) if line.strip().startswith("Q2") and "stat" in line),
        None,
    )
    if header_index is None:
        raise ValueError(f"CLAS header not found: {path}")
    header = lines[header_index].split()
    expected_fields = 9
    numeric_rows = 0
    malformed_numeric_rows = 0
    ranges: dict[str, list[float]] = {}
    for raw in lines[header_index + 2 :]:  # skip the units line
        tokens = raw.split()
        if not tokens:
            continue
        if len(tokens) < expected_fields or not all(FLOAT_RE.match(token) for token in tokens[:expected_fields]):
            if tokens and FLOAT_RE.match(tokens[0]):
                malformed_numeric_rows += 1
            continue
        values = [_float(token) for token in tokens[:expected_fields]]
        if any(not (value == value and abs(value) != float("inf")) for value in values):
            malformed_numeric_rows += 1
            continue
        numeric_rows += 1
        for name, value in zip(header[:expected_fields], values):
            ranges.setdefault(name, []).append(value)
    return {
        "source": _display_path(path),
        "header": header,
        "axis_names": spec["axes"],
        "observable": spec["observable"],
        "units_line": lines[header_index + 1].strip() if header_index + 1 < len(lines) else "",
        "line_count": len(lines),
        "numeric_row_count": numeric_rows,
        "malformed_numeric_row_count": malformed_numeric_rows,
        "column_ranges": {
            name: {"min": min(values), "max": max(values)} for name, values in ranges.items()
        },
        "fit_ready": False,
        "blocking_reason": "absolute cross section, radiative/acceptance convention, and covariance closure pending",
    }


def _table_block(text: str, label: str, begin: str = "\\begin{table") -> str:
    marker = f"\\label{{{label}}}"
    position = text.find(marker)
    if position < 0:
        raise ValueError(f"TeX label not found: {label}")
    start = text.rfind(begin, 0, position)
    stop = text.find("\\end{table", position)
    if start < 0 or stop < 0:
        raise ValueError(f"TeX table bounds not found: {label}")
    return text[start:stop]


def profile_hall_c(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    label = "tab:xsect-vrs-pt2"
    block = _table_block(text, label)
    rows = [
        line.strip()
        for line in block.splitlines()
        if "&" in line and "\\pm" in line and "\\hline" not in line and re.match(r"[0-9]", line.strip())
    ]
    return {
        "source": _display_path(path),
        "table_labels": [label],
        "physical_row_count": len(rows),
        "columns_per_physical_row": 8,
        "observations_per_row": 8,
        "expanded_observation_count": len(rows) * 8,
        "fit_ready": False,
        "blocking_reason": "absolute low-energy H/D cross sections require rho-subtraction, W/Mx, and covariance closure",
    }


def profile_hall_a(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    labels = ["xs_table_2510_p", "xs_table_2510_m", "xs_table_1010_p", "xs_table_1010_m"]
    tables: list[dict[str, Any]] = []
    for label in labels:
        block = _table_block(text, label, begin="\\begin{table*")
        rows = [
            line.strip()
            for line in block.splitlines()
            if "&" in line and "\\hline" not in line and any(char.isdigit() for char in line)
            and re.match(r"[0-9]", line.strip())
        ]
        tables.append({
            "label": label,
            "physical_row_count": len(rows),
            "observations_per_row": 2,
            "expanded_observation_count": len(rows) * 2,
        })
    return {
        "source": _display_path(path),
        "tables": tables,
        "physical_row_count": sum(item["physical_row_count"] for item in tables),
        "expanded_observation_count": sum(item["expanded_observation_count"] for item in tables),
        "fit_ready": False,
        "blocking_reason": "3He nuclear impulse-approximation/dilution model and absolute cross-section covariance closure pending",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()

    clas_root = RAW_ROOT / "arxiv_0809.1153" / "source" / "anc"
    clas = []
    for filename, spec in CLAS_TABLES.items():
        path = clas_root / filename
        clas.append(profile_clas(path, spec) if path.exists() else {
            "source": str(path.relative_to(CAMPAIGN)), "status": "missing"
        })
    hall_c_path = RAW_ROOT / "arxiv_1103.1649" / "source" / "prc2011_v2.tex"
    hall_a_path = RAW_ROOT / "arxiv_1610.02350" / "source" / "main_text_with_figs.tex"
    report = {
        "campaign": "sidis_global_analysis_2026",
        "status": "arxiv_tables_inventoried_not_fit_ready",
        "clas_0809_1153": clas,
        "hall_c_1103_1649": profile_hall_c(hall_c_path) if hall_c_path.exists() else {"status": "missing"},
        "hall_a_1610_02350": profile_hall_a(hall_a_path) if hall_a_path.exists() else {"status": "missing"},
        "selection_authorized": False,
        "production_authorized": False,
        "production_files_modified": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        "# Public arXiv SIDIS table inventory",
        "",
        "This is a source parser/inventory only. No arXiv rows are approved for a fit.",
        "",
        "## CLAS 0809.1153 ancillary tables",
        "",
        "| File | Numeric rows | Observable axes | Malformed numeric rows |",
        "| --- | ---: | --- | ---: |",
    ]
    for item in clas:
        lines.append(
            f"| `{Path(item['source']).name}` | {item.get('numeric_row_count', 0)} | {', '.join(item.get('axis_names', []))} | {item.get('malformed_numeric_row_count', 0)} |"
        )
    lines += [
        "",
        "The CLAS files contain absolute five-fold cross sections with statistical and systematic columns and a radiative-correction factor. They are not multiplicities; conversion requires a fixed SIDIS cross-section convention, bin integration, acceptance/radiative treatment, and covariance model.",
        "",
        "## Hall C E00-108 and Hall A E06-010 TeX tables",
        "",
        f"Hall C `tab:xsect-vrs-pt2`: {report['hall_c_1103_1649'].get('physical_row_count', 0)} physical pT² rows, {report['hall_c_1103_1649'].get('expanded_observation_count', 0)} target/charge entries in the two rho-subtraction columns.",
        f"Hall A: {report['hall_a_1610_02350'].get('physical_row_count', 0)} physical table rows and {report['hall_a_1610_02350'].get('expanded_observation_count', 0)} pi+/pi- entries across the four labelled tables.",
        "",
        "Hall C remains a stage-2 low-energy absolute-cross-section candidate pending rho-subtraction and covariance closure. Hall A is a 3He nuclear diagnostic until a nuclear impulse-approximation/dilution interface is explicitly validated. The TeX rows are counted for provenance and are not silently converted.",
    ]
    args.markdown.write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "status": report["status"],
        "clas_numeric_rows": sum(item.get("numeric_row_count", 0) for item in clas),
        "hall_c_expanded_entries": report["hall_c_1103_1649"].get("expanded_observation_count", 0),
        "hall_a_expanded_entries": report["hall_a_1610_02350"].get("expanded_observation_count", 0),
        "approved_rows": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
