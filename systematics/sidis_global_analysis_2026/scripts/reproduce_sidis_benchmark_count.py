#!/usr/bin/env python3
"""Audit the 1,547-point HERMES/COMPASS literature benchmark.

The literature count is a post-cut selection, not the number of rows in a
HEPData archive.  This script applies only the kinematic information that is
actually present in the public tables and reports unresolved pieces explicitly.
It never approves rows or constructs a fit dataset.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from typing import Any

from sidis_data import read_hepdata_csv
from sidis_dataset import parse_interval


CAMPAIGN = Path(__file__).resolve().parents[1]
RAW_ROOT = CAMPAIGN / "data" / "raw" / "global"
DEFAULT_OUTPUT = CAMPAIGN / "reports" / "sidis_1547_benchmark_audit.json"
DEFAULT_MARKDOWN = CAMPAIGN / "reports" / "sidis_1547_benchmark_audit.md"
NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?")


def first_number(value: str | None) -> float | None:
    if value is None:
        return None
    match = NUMBER.search(value)
    return float(match.group()) if match else None


def interval(value: str | None) -> tuple[float, float] | None:
    if value is None:
        return None
    try:
        return parse_interval(value)
    except ValueError:
        return None


def _column(table: Any, token: str) -> str | None:
    for column in table.columns:
        normalized = re.sub(r"[^a-z0-9]", "", column.lower())
        if normalized == token:
            return column
    return None


def hermes_projection_audit() -> dict[str, Any]:
    root = RAW_ROOT / "hepdata_ins1208547" / "tables"
    tables = []
    for path in sorted(root.rglob("*.csv")):
        table = read_hepdata_csv(path)
        if "MULT" not in table.columns:
            continue
        pt_columns = [column for column in table.columns if re.search(r"\bPT\b", column, re.IGNORECASE)]
        if not pt_columns:
            continue
        rows = sum(
            bool(row.get("MULT", "").strip()) and (not metadata or metadata.get("data_block", "primary") == "primary")
            for row, metadata in zip(table.rows, table.row_metadata or ({},) * len(table.rows))
        )
        tables.append({
            "table": str(path.relative_to(CAMPAIGN)),
            "description": table.metadata.get("description", ""),
            "primary_rows": rows,
            "axis_columns": pt_columns,
        })
    return {
        "source": "hepdata:ins1208547",
        "available_transverse_projection_tables": len(tables),
        "available_primary_rows": sum(item["primary_rows"] for item in tables),
        "reported_literature_rows": 344,
        "status": "incomplete_projection",
        "blocking_reason": "the public HEPData projection has no Q or x axes and does not expose the full HERMES vector-meson-subtracted zxpt-3D supplemental archive/covariance sidecars used for the literature count",
        "tables": tables,
    }


def compass_cut_audit() -> dict[str, Any]:
    root = RAW_ROOT / "hepdata_ins1624692" / "tables"
    q2_col = "$P_{hT}^2 (GeV/c)^{2}$"
    q2_low_col = q2_col + " LOW"
    q2_high_col = q2_col + " HIGH"
    rows: list[dict[str, float]] = []
    for path in sorted(root.rglob("*.csv")):
        table = read_hepdata_csv(path)
        if q2_col not in table.columns:
            continue
        for row, metadata in zip(table.rows, table.row_metadata or ({},) * len(table.rows)):
            if metadata.get("data_block", "primary") != "primary":
                continue
            try:
                rows.append({
                    "pht2": float(row[q2_col]),
                    "pht2_low": float(row[q2_low_col]),
                    "pht2_high": float(row[q2_high_col]),
                    "q2_value": first_number(metadata.get("q2_value")),
                    "q2_low": interval(metadata.get("q2_bin"))[0],
                    "q2_high": interval(metadata.get("q2_bin"))[1],
                    "z_low": interval(metadata.get("z_bin"))[0],
                    "z_high": interval(metadata.get("z_bin"))[1],
                    "y": first_number(metadata.get("y_value")),
                })
            except (KeyError, TypeError, ValueError, IndexError):
                continue

    def count(*, q2_mode: str, pht_mode: str, z_mode: str) -> int:
        selected = 0
        for row in rows:
            q2 = row["q2_value"] if q2_mode == "published" else 0.5 * (row["q2_low"] + row["q2_high"])
            z = {
                "mid": 0.5 * (row["z_low"] + row["z_high"]),
                "high": row["z_high"],
                "low": row["z_low"],
            }[z_mode]
            pht2 = {
                "low": row["pht2_low"],
                "center": row["pht2"],
                "high": row["pht2_high"],
            }[pht_mode]
            if q2 is None or row["y"] is None:
                continue
            q = math.sqrt(q2)
            limit = min(min(0.2 * q, 0.5 * z * q) + 0.3, z * q)
            if q > 1.4 and 0.2 < z < 0.7 and 0.1 < row["y"] < 0.9 and math.sqrt(pht2) < limit:
                selected += 1
        return selected

    policies = {
        "published_Q2_value_z_mid_Pht_center": count(q2_mode="published", pht_mode="center", z_mode="mid"),
        "published_Q2_value_z_mid_Pht_low": count(q2_mode="published", pht_mode="low", z_mode="mid"),
        "published_Q2_value_z_mid_Pht_high": count(q2_mode="published", pht_mode="high", z_mode="mid"),
        "Q2_bin_midpoint_z_mid_Pht_center": count(q2_mode="midpoint", pht_mode="center", z_mode="mid"),
        "Q2_bin_midpoint_z_mid_Pht_high": count(q2_mode="midpoint", pht_mode="high", z_mode="mid"),
    }
    return {
        "source": "hepdata:ins1624692",
        "available_primary_rows": len(rows),
        "reported_literature_rows": 1203,
        "cut_equation": "Q > 1.4 GeV; 0.2 < z < 0.7; |P_hT| < min[min(0.2 Q, 0.5 z Q)+0.3 GeV, z Q]; 0.1 < y < 0.9",
        "representative_bin_policies": policies,
        "status": "selection_convention_not_closed",
        "blocking_reason": "the HEPData archive exposes bin centers/edges and metadata, but the literature's exact point-level representative values, W^2 treatment, duplicate policy, and source conversion are not encoded as a canonical selection manifest",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    hermes = hermes_projection_audit()
    compass = compass_cut_audit()
    report = {
        "campaign": "sidis_global_analysis_2026",
        "status": "1547_benchmark_audited_not_reproduced",
        "literature_reference": "arXiv:2206.07598, Table 2/data.tex",
        "reported_sidIs_total": 1547,
        "reported_breakdown": {"HERMES": 344, "COMPASS": 1203},
        "cut_lock": {
            "Q_min_GeV": 1.4,
            "z_range": [0.2, 0.7],
            "pht_limit": "min[min(0.2 Q, 0.5 z Q)+0.3 GeV, z Q]",
            "HERMES_choice": "vector-meson-subtracted zxpt-3D",
            "COMPASS_choice": "vector-boson-subtracted release",
        },
        "hermes": hermes,
        "compass": compass,
        "available_projection_rows": hermes["available_primary_rows"],
        "available_compass_rows": compass["available_primary_rows"],
        "approved_rows": 0,
        "selection_authorized": False,
        "production_authorized": False,
        "production_files_modified": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# HERMES/COMPASS 1,547-point benchmark audit",
        "",
        "Status: **not reproduced and not fit-authorized**. The 1,547 value is the post-cut HERMES/COMPASS count in arXiv:2206.07598, not a raw HEPData row total.",
        "",
        "Locked literature cuts: `Q > 1.4 GeV`, `0.2 < z < 0.7`, `|P_hT| < min[min(0.2 Q, 0.5 z Q)+0.3 GeV, z Q]`, with the published HERMES vector-meson-subtracted `zxpt-3D` scope and COMPASS vector-boson-subtracted release.",
        "",
        f"- HERMES target count: **344**; available public HEPData pT projection rows: **{hermes['available_primary_rows']}** across **{hermes['available_transverse_projection_tables']}** tables. The HEPData projection lacks Q/x axes and the supplemental archive/covariance sidecars, so it cannot certify the 344 selection.",
        f"- COMPASS target count: **1203**; available HEPData primary rows: **{compass['available_primary_rows']}**. Deterministic representative-value counts range across the recorded policies: **{min(compass['representative_bin_policies'].values())}--{max(compass['representative_bin_policies'].values())}**, demonstrating why a source selection manifest is required.",
        "",
        "No rows are approved. The next gate is to obtain/mirror the HERMES `zxpt-3D` value and covariance files, define the COMPASS point-level bin convention, then freeze a row-level selection manifest before any fit.",
    ]
    args.markdown.write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "status": report["status"],
        "hermes_available_rows": hermes["available_primary_rows"],
        "compass_available_primary_rows": compass["available_primary_rows"],
        "compass_policy_counts": compass["representative_bin_policies"],
        "approved_rows": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
