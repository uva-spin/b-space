#!/usr/bin/env python3
"""Build a conservative row/table provenance audit from HEPData profiles.

This is deliberately an audit, not a converter.  It records likely observable
classes and candidate transverse-momentum tables while preserving ambiguity in
axis/value columns, target composition, normalization, and covariance.  No
row is approved by this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


CAMPAIGN = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = CAMPAIGN / "reports/hepdata_table_inventory.json"
DEFAULT_CONFIG = CAMPAIGN / "config/public_sources.json"
DEFAULT_OUTPUT = CAMPAIGN / "reports/row_level_provenance_audit.json"
DEFAULT_MARKDOWN = CAMPAIGN / "reports/row_level_provenance_audit.md"


def normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def record_id(path: str) -> str:
    match = re.search(r"/hepdata/([^/]+)/", path)
    if match is None:
        raise ValueError(f"cannot identify HEPData record from {path!r}")
    return match.group(1)


def observable_class(profile: dict) -> str:
    keyword = normalized(profile.get("metadata", {}).get("keyword observables", ""))
    description = normalized(profile.get("metadata", {}).get("description", ""))
    if "mult" in keyword or "multiplic" in description:
        return "multiplicity"
    if "asym" in keyword or "asymmetr" in description:
        return "asymmetry"
    return "other_or_unresolved"


def axis_candidates(columns: list[str]) -> dict[str, list[str]]:
    result = {"x": [], "q2": [], "y": [], "z": [], "pht": [], "pht2": []}
    for column in columns:
        token = normalized(column)
        if token in {"x", "xb", "xbj"} or token.startswith("x("):
            result["x"].append(column)
        if token == "y":
            result["y"].append(column)
        if token == "z" or token.startswith("z(") or token.startswith("z{"):
            result["z"].append(column)
        if "q2" in token or "q2" in token:
            result["q2"].append(column)
        if not any(bound in token for bound in ("low", "high")) and ("pht2" in token or "phperp2" in token or token.startswith("pt2")):
            result["pht2"].append(column)
        elif not any(bound in token for bound in ("low", "high")) and ("pht" in token or "phperp" in token or token.startswith("pt")):
            result["pht"].append(column)
    return result


def table_audit(profile: dict, source: dict) -> dict:
    columns = list(profile.get("columns", []))
    metadata = profile.get("metadata", {})
    candidates = axis_candidates(columns)
    bounds = [c for c in columns if normalized(c).endswith("low") or normalized(c).endswith("high")]
    uncertainty = [c for c in columns if any(token in normalized(c) for token in ("stat", "sys", "error", "uncert"))]
    value_columns = [
        c for c in columns
        if c not in uncertainty and c not in bounds
        and all(c not in values for values in candidates.values())
    ]
    warnings = list(profile.get("warnings", []))
    for axis, names in candidates.items():
        if len(names) > 1:
            warnings.append(f"multiple {axis} columns require published bin/value mapping")
    warnings.append("point-to-point and correlated covariance treatment not resolved")
    warnings.append("normalization correlations and nuisance treatment not resolved")
    if profile.get("has_transverse_momentum") and observable_class(profile) != "multiplicity":
        warnings.append("transverse table is not yet confirmed as an unpolarized multiplicity")
    candidate = bool(profile.get("has_transverse_momentum")) and observable_class(profile) == "multiplicity"
    status = "tmd_candidate_pending_review" if candidate else (
        "collinear_multiplicity_complement_pending_review"
        if observable_class(profile) == "multiplicity" else "not_in_first_unpolarized_multiplicity_scope"
    )
    return {
        "record": source["record"],
        "collaboration": source["collaboration"],
        "source_url": source["source_url"],
        "path": profile["path"],
        "table_name": metadata.get("name"),
        "table_doi": metadata.get("table_doi"),
        "description": metadata.get("description", ""),
        "keyword_observables": metadata.get("keyword observables"),
        "keyword_reactions": metadata.get("keyword reactions"),
        "target_record_scope": source["target"],
        "observable_class": observable_class(profile),
        "row_count": profile["row_count"],
        "primary_row_count": profile.get("primary_row_count", profile["row_count"]),
        "auxiliary_row_count": profile.get("auxiliary_row_count", 0),
        "data_block_counts": profile.get("data_block_counts", {"primary": profile["row_count"]}),
        "axis_columns_profiled": profile.get("axes", {}),
        "axis_columns_candidates": candidates,
        "bin_bound_columns": bounds,
        "value_columns_candidates": value_columns,
        "uncertainty_columns": uncertainty,
        "has_transverse_momentum": bool(profile.get("has_transverse_momentum")),
        "candidate_tmd_multiplicity": candidate,
        "covariance_status": "not_resolved_from_CSV_submission",
        "normalization_status": "not_resolved_from_CSV_submission",
        "row_approval": "not_approved",
        "review_status": status,
        "warnings": sorted(set(warnings)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    profiles = json.loads(args.profiles.read_text())["profiles"]
    sources = json.loads(args.config.read_text())["records"]
    by_record = {item["record"]: item for item in sources}
    audits = []
    for profile in profiles:
        recid = record_id(profile["path"])
        audits.append(table_audit(profile, by_record[recid]))
    counts = {
        "tables": len(audits),
        "rows": sum(item["row_count"] for item in audits),
        "tmd_candidate_tables": sum(item["candidate_tmd_multiplicity"] for item in audits),
        "tmd_candidate_rows": sum(item["primary_row_count"] for item in audits if item["candidate_tmd_multiplicity"]),
        "multiplicity_tables": sum(item["observable_class"] == "multiplicity" for item in audits),
        "asymmetry_tables": sum(item["observable_class"] == "asymmetry" for item in audits),
        "unresolved_tables": sum(item["observable_class"] == "other_or_unresolved" for item in audits),
    }
    report = {
        "campaign": "sidis_global_analysis_2026",
        "status": "table_and_row_provenance_audit_complete_selection_not_authorized",
        "method": "metadata/column audit only; no values transformed and no covariance inferred",
        "counts": counts,
        "records": sorted({item["record"] for item in audits}),
        "audits": audits,
        "approved_rows": 0,
        "selection_authorized": False,
        "production_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# SIDIS row/table provenance audit",
        "",
        "Status: table/column audit complete; no rows approved and no covariance inferred.",
        "",
        "| Record | TMD-candidate tables | TMD-candidate rows | Multiplicity tables | Asymmetry tables | Unresolved tables |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for recid in sorted(by_record):
        items = [item for item in audits if item["record"] == recid]
        lines.append(
            f"| {recid} | {sum(item['candidate_tmd_multiplicity'] for item in items)} "
            f"| {sum(item['primary_row_count'] for item in items if item['candidate_tmd_multiplicity'])} "
            f"| {sum(item['observable_class'] == 'multiplicity' for item in items)} "
            f"| {sum(item['observable_class'] == 'asymmetry' for item in items)} "
            f"| {sum(item['observable_class'] == 'other_or_unresolved' for item in items)} |"
        )
    lines += [
        "",
        "Every table records candidate axes, value columns, bin-edge columns, and uncertainty columns.",
        "For multi-block CSVs, candidate rows count only the explicitly marked primary data block; auxiliary correction-factor rows remain in the raw audit.",
        "Tables with multiple possible axis columns require a published mapping review before conversion.",
        "The CSV submissions do not by themselves close correlated covariance or normalization treatment.",
        "See `row_level_provenance_audit.json` for the complete machine-readable audit.",
    ]
    args.markdown.write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": report["status"], **counts}, indent=2))


if __name__ == "__main__":
    main()
