#!/usr/bin/env python3
"""Build the high-qT row inventory and dense external-benchmark work plan."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "systematics" / "high_qt_direct_production_benchmark"
SOURCE = ROOT / "systematics" / "finite_y_tail_benchmark" / "summaries" / "tail_benchmark_row_gate.csv"
POLICY_PATH = STUDY / "config" / "promotion_policy.json"


def tier_for(value: float, tiers: list[dict]) -> str:
    for tier in tiers:
        upper = tier["max_inclusive"]
        if value > tier["min_exclusive"] and (upper is None or value <= upper):
            return tier["name"]
    raise ValueError(f"No tier configured for qT/Q={value}")


def main() -> None:
    policy = json.loads(POLICY_PATH.read_text())
    rows = pd.read_csv(SOURCE)
    candidates = rows.loc[rows["region"].eq("high_qT_candidate")].copy()
    allowed = set(policy["candidate_region"]["datasets"])
    candidates = candidates.loc[candidates["dataset"].isin(allowed)].copy()
    candidates["benchmark_tier"] = [tier_for(float(v), policy["tiers"]) for v in candidates["qT_over_Q"]]

    available = candidates["external_benchmark_available"].fillna(False).astype(bool)
    agreement = candidates["external_code_agreement_pass"].fillna(False).astype(bool)
    candidates["row_level_external_gate_pass"] = available & agreement
    candidates["observable_identity_review"] = np.where(available, "partial_from_existing_artifact", "pending")
    candidates["remaining_required_gates"] = np.where(
        candidates["row_level_external_gate_pass"],
        "numerical_precision;scale;pdf;matched_tail;fit_impact;replica_bspace",
        "observable_identity;external_pair;numerical_precision;scale;pdf;matched_tail;fit_impact;replica_bspace",
    )
    candidates["current_decision"] = np.where(
        candidates["row_level_external_gate_pass"],
        "external_pair_pass_only_not_production_approved",
        "benchmark_pending_not_production_approved",
    )

    keep = [
        "benchmark_tier", "dataset", "row_id", "qT", "qT_low", "qT_high", "qT_over_Q",
        "QM_Low", "QM_High", "y_Low", "y_High", "CS", "error", "source_table",
        "external_benchmark_available", "dyturbo_pb_per_GeV", "dyturbo_pb_per_GeV_unc",
        "mcfm_pb_per_GeV", "mcfm_pb_per_GeV_unc", "dyturbo_mcfm_rel_diff",
        "external_code_agreement_pass", "row_level_external_gate_pass",
        "observable_identity_review", "remaining_required_gates", "current_decision",
    ]
    keep = [column for column in keep if column in candidates.columns]
    candidates = candidates[keep].sort_values(["benchmark_tier", "dataset", "qT_over_Q", "row_id"])

    summary_dir = STUDY / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = summary_dir / "high_qt_candidate_inventory.csv"
    candidates.to_csv(inventory_path, index=False)

    pending = candidates.loc[~candidates["row_level_external_gate_pass"]].copy()
    pending["run_dyturbo"] = True
    pending["run_mcfm"] = True
    pending["output_group"] = pending["dataset"].str.lower()
    batch_columns = [
        "benchmark_tier", "dataset", "row_id", "qT_low", "qT_high", "qT_over_Q",
        "QM_Low", "QM_High", "y_Low", "y_High", "run_dyturbo", "run_mcfm", "output_group",
    ]
    pending[[c for c in batch_columns if c in pending.columns]].to_csv(
        summary_dir / "benchmark_batch_plan.csv", index=False
    )

    by_dataset = []
    for (tier, dataset), group in candidates.groupby(["benchmark_tier", "dataset"], sort=True):
        by_dataset.append({
            "benchmark_tier": tier,
            "dataset": dataset,
            "n_candidates": int(len(group)),
            "n_external_pairs": int(group["external_benchmark_available"].fillna(False).sum()),
            "n_external_pass": int(group["row_level_external_gate_pass"].sum()),
            "n_pending_pairs": int((~group["row_level_external_gate_pass"]).sum()),
            "min_qT_over_Q": float(group["qT_over_Q"].min()),
            "max_qT_over_Q": float(group["qT_over_Q"].max()),
        })
    pd.DataFrame(by_dataset).to_csv(summary_dir / "coverage_by_tier_and_dataset.csv", index=False)

    status = {
        "study": policy["study"],
        "source_gate": str(SOURCE.relative_to(ROOT)),
        "production_reference": policy["production_reference"],
        "n_high_qt_candidates": int(len(candidates)),
        "n_existing_external_pairs": int(candidates["external_benchmark_available"].fillna(False).sum()),
        "n_existing_external_pass": int(candidates["row_level_external_gate_pass"].sum()),
        "n_pending_external_pairs": int((~candidates["row_level_external_gate_pass"]).sum()),
        "n_direct_production_approved": 0,
        "direct_approval_blockers": policy["required_followup_gates"],
        "policy_note": policy["policy"],
        "outputs": {
            "inventory": str(inventory_path.relative_to(ROOT)),
            "batch_plan": str((summary_dir / "benchmark_batch_plan.csv").relative_to(ROOT)),
            "coverage": str((summary_dir / "coverage_by_tier_and_dataset.csv").relative_to(ROOT)),
        },
    }
    (summary_dir / "baseline_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
