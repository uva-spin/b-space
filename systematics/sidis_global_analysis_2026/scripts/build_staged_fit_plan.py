#!/usr/bin/env python3
"""Validate and render the progressive SIDIS fitting protocol.

This is a planning/provenance driver only.  It does not select rows, create a
likelihood, start a fit, or modify the frozen DY production package.
"""

from __future__ import annotations

import json
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
PLAN = BASE / "config/staged_fit_plan.json"
REGISTRY = BASE / "config/global_sources.json"
OUT_JSON = BASE / "reports/staged_fit_plan.json"
OUT_MD = BASE / "reports/staged_fit_plan.md"


def _source_ids(plan: dict) -> list[str]:
    ids: list[str] = []
    for stage in plan["stages"]:
        value = stage["source_ids"]
        if isinstance(value, str):
            continue
        ids.extend(value)
    return ids


def _render(plan: dict, registry: dict) -> str:
    records = {row["id"]: row for row in registry["records"]}
    lines = [
        "# Progressive SIDIS fit plan",
        "",
        "Status: **staged discovery only; no rows approved and no production fit authorized**.",
        "",
        plan["principle"],
        "",
        "## External benchmark",
        "",
        "The literature checkpoint is **1,547 post-cut HERMES/COMPASS points** "
        "(344 HERMES + 1,203 COMPASS), not a raw-table count. It remains "
        "blocked until the source-specific row and covariance conventions are "
        "closed. See `reports/sidis_1547_benchmark_audit.md`.",
        "",
        "## Stages",
        "",
        "| ID | Scope | Sources | Status | Entry/exit rule |",
        "| --- | --- | --- | --- | --- |",
    ]
    for stage in plan["stages"]:
        source_ids = stage["source_ids"]
        if isinstance(source_ids, str):
            scope = "all registry records"
            source_text = source_ids
        else:
            scope = stage["name"]
            source_text = ", ".join(source_ids)
        rule = stage.get("entry_gate", "")
        if stage.get("exit_gate"):
            rule += ("; " if rule else "") + stage["exit_gate"]
        lines.append(
            f"| `{stage['id']}` | {scope} | {source_text} | "
            f"{stage['status']} | {rule} |"
        )
    lines.extend([
        "",
        "## Source coverage check",
        "",
        f"Registry records: **{len(records)}**. Explicitly assigned non-S0 records: "
        f"**{len(set(_source_ids(plan)))}**.",
        "",
        "A source may be fit only after its stage entry gate passes. Sources in "
        "the diagnostic or deferred classes are never silently merged into the "
        "multiplicity likelihood.",
        "",
        "## Required record for every trial",
        "",
    ])
    for item in plan["per_trial_record"]["required"]:
        lines.append(f"- `{item}`")
    lines.extend([
        "",
        "Experimental replicas, start non-uniqueness, TMDFF parameterization, "
        "dataset selection, and perturbative/theory variations must remain "
        "separately labelled before any combined envelope is shown.",
        "",
        "The machine-readable source is `config/staged_fit_plan.json`.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    plan = json.loads(PLAN.read_text())
    registry = json.loads(REGISTRY.read_text())
    registry_ids = {row["id"] for row in registry["records"]}
    assigned = set(_source_ids(plan))
    unknown = sorted(assigned - registry_ids)
    if unknown:
        raise RuntimeError(f"staged plan contains unknown source ids: {unknown}")
    if plan["approved_rows"] != 0 or plan["production_authorized"]:
        raise RuntimeError("discovery plan cannot approve rows or authorize production")
    plan["registry_record_count"] = len(registry_ids)
    plan["explicitly_assigned_non_inventory_records"] = len(assigned)
    plan["unassigned_non_inventory_records"] = sorted(registry_ids - assigned)
    OUT_JSON.write_text(json.dumps(plan, indent=2) + "\n")
    OUT_MD.write_text(_render(plan, registry))
    print(json.dumps({
        "status": plan["status"],
        "registry_records": len(registry_ids),
        "explicitly_assigned_non_inventory_records": len(assigned),
        "unassigned_non_inventory_records": plan["unassigned_non_inventory_records"],
        "approved_rows": plan["approved_rows"],
        "production_authorized": plan["production_authorized"],
    }, indent=2))


if __name__ == "__main__":
    main()
