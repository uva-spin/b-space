#!/usr/bin/env python3
"""Summarize auditable SIDIS scope options without selecting fit rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CAMPAIGN = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=CAMPAIGN / "reports/row_level_provenance_audit.json")
    parser.add_argument("--output", type=Path, default=CAMPAIGN / "reports/candidate_scope_options.json")
    parser.add_argument("--markdown", type=Path, default=CAMPAIGN / "reports/candidate_scope_options.md")
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text())
    by_record = {}
    for item in audit["audits"]:
        record = item["record"]
        entry = by_record.setdefault(record, {"tables": 0, "rows": 0, "raw_rows": 0, "auxiliary_rows": 0, "tmd_tables": 0, "tmd_rows": 0})
        entry["tables"] += 1
        entry["rows"] += item.get("primary_row_count", item["row_count"])
        entry["raw_rows"] += item["row_count"]
        entry["auxiliary_rows"] += item.get("auxiliary_row_count", 0)
        entry["tmd_tables"] += int(item["candidate_tmd_multiplicity"])
        entry["tmd_rows"] += item.get("primary_row_count", item["row_count"]) if item["candidate_tmd_multiplicity"] else 0

    def totals(records, key):
        return sum(by_record[r][key] for r in records)

    scopes = [
        {"name": "hermes_identified_transverse", "records": ["ins1208547"], "role": "first identified-hadron TMD conversion candidate", "observable": "pi+/pi-/K+/K- multiplicities versus P_hperp with H/D target projections", "tables": by_record["ins1208547"]["tmd_tables"], "rows": by_record["ins1208547"]["tmd_rows"], "uncertainty_state": "explicit stat/sys columns; supplemental HERMES database advertises statistical covariance, archive not mirrored", "blocking_items": ["mirror and hash the HERMES covariance archive", "resolve proton/deuteron block mapping and bin integration"], "decision": "candidate_after_covariance_and_bin_audit"},
        {"name": "compass_2013_transverse", "records": ["ins1236358"], "role": "historical broad pT^2 transverse candidate", "observable": "charged-hadron differential multiplicities versus pT^2 in x_B, Q^2, and z bins", "tables": by_record["ins1236358"]["tmd_tables"], "rows": by_record["ins1236358"]["tmd_rows"], "uncertainty_state": "generic asymmetric error columns only; statistical/systematic decomposition and correlated normalization are not identified in the CSV", "blocking_items": ["map comment-defined x/Q^2/z blocks to rows", "resolve error semantics and overlap with COMPASS 2018"], "decision": "candidate_for_independent_cross_check_not_first_fit"},
        {"name": "compass_2018_transverse", "records": ["ins1624692"], "role": "primary modern charged-hadron TMD grid candidate", "observable": "h+/h- multiplicities versus P_hT^2 in x, Q^2, and z bins", "tables": by_record["ins1624692"]["tmd_tables"], "rows": by_record["ins1624692"]["tmd_rows"], "uncertainty_state": "explicit stat/sys columns; correlated covariance and normalization treatment are not supplied by the CSV audit", "blocking_items": ["verify table-level bin integration and correction-factor convention", "resolve correlated systematics before likelihood construction"], "decision": "candidate_for_first_scalar_conversion_after_convention_lock"},
        {"name": "combined_transverse_candidate", "records": ["ins1208547", "ins1236358", "ins1624692"], "role": "future combined transverse-momentum scope", "observable": "identified and unidentified charged-hadron multiplicities", "tables": totals(["ins1208547", "ins1236358", "ins1624692"], "tmd_tables"), "rows": totals(["ins1208547", "ins1236358", "ins1624692"], "tmd_rows"), "uncertainty_state": "not closed; source overlap, covariance, target composition, and bin conventions must be resolved jointly", "blocking_items": ["complete each source audit", "define overlap/leave-one-experiment-out protocol"], "decision": "deferred_until_individual_scopes_close"},
        {"name": "collinear_ff_complements", "records": ["46860", "ins1444985", "ins1483098", "ins2840545"], "role": "fragmentation-normalization complements, not direct transverse fit inputs", "observable": "z/x/y multiplicity projections without a transverse-momentum axis", "tables": totals(["46860", "ins1444985", "ins1483098", "ins2840545"], "tables"), "rows": totals(["46860", "ins1444985", "ins1483098", "ins2840545"], "rows"), "uncertainty_state": "value and stat/sys/error columns present in HEPData, but normalization/correlation conventions remain open", "blocking_items": ["lock multiplicity denominator and FF convention", "test compatibility with transverse datasets"], "decision": "deferred_complement"},
    ]
    for scope in scopes:
        scope["raw_rows"] = totals(scope["records"], "raw_rows")
        scope["auxiliary_rows"] = totals(scope["records"], "auxiliary_rows")
    report = {"campaign": "sidis_global_analysis_2026", "status": "candidate_scope_options_summarized_no_rows_selected", "audit_source": str(args.audit), "scopes": scopes, "selection_authorized": False, "production_authorized": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# SIDIS candidate scope options", "", "Status: options summarized from the row/table audit; no rows selected or approved.", "", "| Scope | Role | TMD tables | Primary rows | Raw rows | Decision |", "| --- | --- | ---: | ---: | ---: | --- |"]
    for item in scopes:
        lines.append(f"| `{item['name']}` | {item['role']} | {item['tables']} | {item['rows']} | {item['raw_rows']} | {item['decision']} |")
    lines += ["", "The modern COMPASS 2018 grid is the best first scalar-conversion candidate once its bin/covariance convention is locked. HERMES is the strongest identified-hadron cross-check because its supplemental database advertises statistical covariance. COMPASS 2013 is retained as an independent pT^2 cross-check, not silently merged with the 2018 grid.", "Collinear sources remain fragmentation complements until the multiplicity denominator and normalization conventions are implemented and tested.", "Primary rows exclude explicitly marked auxiliary correction-factor blocks; raw row counts remain available in the JSON report."]
    args.markdown.write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": report["status"], "scope_count": len(scopes)}, indent=2))


if __name__ == "__main__":
    main()
