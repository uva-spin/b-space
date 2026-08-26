#!/usr/bin/env python3
"""Validate explicit source-specific SIDIS mappings without selecting rows."""

from __future__ import annotations

import json
from pathlib import Path
import sys

CAMPAIGN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAMPAIGN))
from sidis_data import read_hepdata_csv  # noqa: E402
from sidis_dataset import SidisColumnMap, canonicalize_table  # noqa: E402

RAW = CAMPAIGN / "data/raw/hepdata"
OUTPUT = CAMPAIGN / "reports/candidate_mapping_validation.json"
MARKDOWN = CAMPAIGN / "reports/candidate_mapping_validation.md"


def table_path(record: str, number: int) -> Path:
    return RAW / record / "tables" / f"HEPData-{record}-v1-csv" / f"Table{number}.csv"


def reactions(table) -> list[str]:
    return list(dict.fromkeys(metadata.get("reaction", "") for metadata in table.row_metadata))


def hadron_from_reaction(reaction: str) -> str:
    upper = reaction.upper()
    for token, label in (("PI+", "pi+"), ("PI-", "pi-"), ("K+", "K+"), ("K-", "K-"),
                         ("HADRON+", "h+"), ("HADRON-", "h-"), ("H$^{+}$", "h+"),
                         ("H$^{−}$", "h-"), ("H$^{-}$", "h-"), ("H^{+}", "h+"),
                         ("H^{-}", "h-")):
        if token.upper() in upper:
            return label
    raise ValueError(f"cannot determine hadron from reaction {reaction!r}")


def hermes(table):
    block_reactions = reactions(table)
    if len(block_reactions) != 2:
        raise ValueError(f"expected proton/deuteron blocks, found {block_reactions}")
    total = 0
    targets = []
    for reaction in block_reactions:
        target = "D" if "DEUT" in reaction.upper() else "H"
        pht = "PT (deuteron) [GEV]" if target == "D" else "PT (proton) [GEV]"
        mapping = SidisColumnMap(
            value="MULT", axis_columns={"pht": pht}, metadata_axes={"z": "z_bin"},
            stat_columns=("stat +", "stat -"), sys_columns=("sys +", "sys -"),
            required_axes=("z", "pht"), block_filters={"reaction": reaction},
            target=target, hadron=hadron_from_reaction(reaction),
        )
        rows = canonicalize_table(table, mapping, source="ins1208547")
        total += len(rows)
        targets.append({"target": target, "hadron": mapping.hadron, "reaction": reaction, "rows": len(rows)})
    return {"mapping_id": "hermes_identified_pht_projection_target_blocks", "fit_readiness": "integrated_projection_requires_acceptance_or_bin_average_convention", "targets": targets}, total


def compass_2013(table):
    block_reactions = reactions(table)
    if len(block_reactions) != 1:
        raise ValueError(f"expected one reaction block, found {block_reactions}")
    reaction = block_reactions[0]
    mapping = SidisColumnMap(
        value="D(N(P=4))/DZ/DPT**2 [GEV**-2]", axis_columns={"pht2": "PT**2 [GEV**2]"},
        metadata_axes={"x": "x_bin", "q2": "q2_bin", "z": "z_bin"},
        bin_columns={"pht2": ("PT**2 [GEV**2] LOW", "PT**2 [GEV**2] HIGH")},
        total_columns=("error +", "error -"), required_axes=("x", "q2", "z", "pht2"),
        block_filters={"data_block": "primary"}, skip_missing_values=True,
        target="D", hadron=hadron_from_reaction(reaction),
    )
    rows = canonicalize_table(table, mapping, source="ins1236358")
    return {"mapping_id": "compass_2013_pt2_generic_total_error", "fit_readiness": "candidate_after_generic_error_and_bin_covariance_review", "reaction": reaction, "hadron": mapping.hadron, "rows": len(rows)}, len(rows)


def compass_2018(table):
    block_reactions = reactions(table)
    if len(block_reactions) != 1:
        raise ValueError(f"expected one reaction block, found {block_reactions}")
    reaction = block_reactions[0]
    hadron = hadron_from_reaction(reaction)
    value = "$M^{h^{+}}$" if hadron == "h+" else "$M^{h^{-}}$"
    mapping = SidisColumnMap(
        value=value, axis_columns={"pht2": "$P_{hT}^2 (GeV/c)^{2}$"},
        metadata_axes={"x": "x_bin", "q2": "q2_bin", "z": "z_bin"},
        bin_columns={"pht2": ("$P_{hT}^2 (GeV/c)^{2}$ LOW", "$P_{hT}^2 (GeV/c)^{2}$ HIGH")},
        stat_columns=("stat +", "stat -"), sys_columns=("sys +", "sys -"),
        required_axes=("x", "q2", "z", "pht2"), block_filters={"data_block": "primary"},
        target="D", hadron=hadron,
    )
    rows = canonicalize_table(table, mapping, source="ins1624692")
    return {"mapping_id": "compass_2018_pt2_stat_sys", "fit_readiness": "candidate_after_covariance_and_bin_convention_review", "reaction": reaction, "hadron": hadron, "rows": len(rows)}, len(rows)


def main() -> None:
    sources = {"ins1208547": (range(17, 33), hermes), "ins1236358": (range(1, 47), compass_2013), "ins1624692": (range(1, 163), compass_2018)}
    records = []
    for record, (numbers, validator) in sources.items():
        item = {"record": record, "expected_table_count": len(list(numbers)), "tables": [], "failures": []}
        for number in numbers:
            path = table_path(record, number)
            table = read_hepdata_csv(path)
            try:
                mapping, count = validator(table)
                item["tables"].append({"table": path.name, "published_rows": len(table.rows), "canonical_rows": count, **mapping})
            except (KeyError, ValueError) as exc:
                item["failures"].append({"table": path.name, "error": str(exc)})
        item["table_count"] = len(item["tables"])
        item["published_row_count"] = sum(entry["published_rows"] for entry in item["tables"]) + sum(
            len(read_hepdata_csv(table_path(record, int(failure["table"][5:-4]))).rows) for failure in item["failures"]
        )
        item["canonical_observation_count"] = sum(entry["canonical_rows"] for entry in item["tables"])
        item["failure_count"] = len(item["failures"])
        item["covariance_status"] = "not_resolved"
        records.append(item)
    report = {"campaign": "sidis_global_analysis_2026", "status": "explicit_candidate_mappings_validated_no_rows_approved", "method": "source-specific caller-declared mappings through sidis_dataset.canonicalize_table", "records": records, "approved_rows": 0, "production_authorized": False}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Candidate SIDIS mapping validation", "", "Status: explicit mappings validate downloaded candidate tables; no rows are approved and no covariance is inferred.", "", "| Record | Tables | Published rows | Canonical observations | Failures |", "| --- | ---: | ---: | ---: | ---: |"]
    for item in records:
        lines.append(f"| {item['record']} | {item['table_count']}/{item['expected_table_count']} | {item['published_row_count']} | {item['canonical_observation_count']} | {item['failure_count']} |")
    lines += ["", "The HERMES projection tables remain integrated over unreported x and Q2.", "COMPASS 2013 dash placeholders are skipped only by an explicit mapping switch.", "COMPASS 2018 correction-factor rows are excluded only by an explicit primary-block filter.", "See candidate_mapping_validation.json for the complete table-level record."]
    MARKDOWN.write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": report["status"], "records": [{"record": x["record"], "tables": f"{x['table_count']}/{x['expected_table_count']}", "canonical_rows": x["canonical_observation_count"], "failures": x["failure_count"]} for x in records]}, indent=2))


if __name__ == "__main__":
    main()
