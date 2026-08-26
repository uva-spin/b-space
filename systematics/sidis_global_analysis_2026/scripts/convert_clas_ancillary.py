#!/usr/bin/env python3
"""Convert CLAS 0809.1153 ancillary files to an auditable CSV.

The conversion is deliberately lossless for the published numeric columns and
adds only source/table metadata.  It does not apply cuts, combine errors,
divide by a DIS cross section, or approve observations for a fit.  The output
defaults to the ignored ``data/derived/global`` tree and can be regenerated
from the raw arXiv archive at any time.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
from typing import Any, Iterable


CAMPAIGN = Path(__file__).resolve().parents[1]
CLAS_ROOT = CAMPAIGN / "data/raw/global/arxiv_0809.1153/source/anc"
DEFAULT_OUTPUT = CAMPAIGN / "data/derived/global/arxiv_0809.1153"
NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?$")

SPECS = {
    "pip_sidis_cs_table_zh_pt2.dat": ("z", "pT2", "ds/dx dQ2 dz dpT2 dphi"),
    "pip_sidis_cs_table_xf_pt2.dat": ("xF", "pT2", "ds/dx dQ2 dxF dpT2 dphi"),
    "pip_sidis_cs_table_zh_t.dat": ("z", "|t|", "ds/dx dQ2 dz dt dphi"),
    "pip_sidis_cs_table_zg_v.dat": ("zG", "v", "ds/dx dQ2 dz dv dphi"),
}


def _numeric(token: str) -> bool:
    return bool(NUMBER.match(token))


def read_clas_rows(path: Path, variable: str, transverse_variable: str, observable: str) -> tuple[list[str], list[list[str]], dict[str, Any]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header_index = next(
        (index for index, line in enumerate(lines) if line.strip().startswith("Q2") and "stat" in line),
        None,
    )
    if header_index is None:
        raise ValueError(f"CLAS header not found: {path}")
    source_columns = lines[header_index].split()
    if len(source_columns) != 9:
        raise ValueError(f"unexpected CLAS header width {len(source_columns)}: {path}")
    rows: list[list[str]] = []
    malformed = 0
    for line_number, line in enumerate(lines[header_index + 2 :], start=header_index + 3):
        tokens = line.split()
        if not tokens:
            continue
        if len(tokens) < len(source_columns) or not all(_numeric(token) for token in tokens[: len(source_columns)]):
            if _numeric(tokens[0]):
                malformed += 1
            continue
        rows.append(tokens[: len(source_columns)])
    columns = ["Q2_GeV2", "x", variable, transverse_variable, "phi_deg", "cross_section", "stat_err", "sys_err", "rad_cor"]
    converted = [row for row in rows]
    metadata = {
        "source": str(path.relative_to(CAMPAIGN)),
        "source_columns": source_columns,
        "canonical_columns": columns,
        "variable": variable,
        "transverse_variable": transverse_variable,
        "observable": observable,
        "units": {
            "Q2_GeV2": "GeV^2",
            "x": "dimensionless",
            variable: "dimensionless" if variable in {"z", "xF", "zG"} else "GeV^2",
            transverse_variable: "GeV^2" if transverse_variable == "pT2" else "GeV^2",
            "phi_deg": "degree",
            "cross_section": "microbarn / GeV^4 / sr",
            "stat_err": "microbarn / GeV^4 / sr",
            "sys_err": "microbarn / GeV^4 / sr",
            "rad_cor": "dimensionless multiplicative factor",
        },
        "line_count": len(lines),
        "converted_row_count": len(converted),
        "malformed_numeric_row_count": malformed,
        "fit_ready": False,
        "selection_authorized": False,
        "blocking_reason": "absolute-cross-section conversion and covariance/radiative/acceptance closure pending",
    }
    return columns, converted, metadata


def convert(paths: Iterable[Path], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for path in paths:
        try:
            variable, transverse_variable, observable = SPECS[path.name]
        except KeyError as exc:
            raise ValueError(f"unsupported CLAS ancillary filename: {path.name}") from exc
        columns, rows, metadata = read_clas_rows(path, variable, transverse_variable, observable)
        target = output_dir / f"{path.stem}.csv"
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            writer.writerows(rows)
        metadata["output"] = str(target.relative_to(CAMPAIGN))
        records.append(metadata)
    manifest = {
        "campaign": "sidis_global_analysis_2026",
        "source": "arxiv:0809.1153",
        "status": "source_specific_conversion_complete_not_fit_ready",
        "records": records,
        "converted_rows": sum(item["converted_row_count"] for item in records),
        "approved_rows": 0,
        "production_authorized": False,
        "production_files_modified": False,
    }
    (output_dir / "conversion_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", action="append", type=Path, help="one CLAS ancillary file; repeat or omit for all four")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    paths = args.file or [CLAS_ROOT / name for name in SPECS]
    manifest = convert(paths, args.output_dir)
    print(json.dumps({
        "status": manifest["status"],
        "converted_rows": manifest["converted_rows"],
        "files": len(manifest["records"]),
        "approved_rows": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
