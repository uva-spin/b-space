#!/usr/bin/env python3
"""Canonicalize row-isolated external-code outputs and evaluate central gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "systematics" / "high_qt_direct_production_benchmark"
POLICY = json.loads((STUDY / "config" / "promotion_policy.json").read_text())


def one_pair(dy_path: Path, mc_path: Path) -> dict:
    dy = pd.read_csv(dy_path).iloc[-1]
    mc = pd.read_csv(mc_path).iloc[-1]
    if dy.row_id != mc.row_id:
        raise ValueError(f"Row mismatch: {dy.row_id} vs {mc.row_id}")
    width = float(dy.qT_high) - float(dy.qT_low)
    # Existing DYTurbo V+jet text output is fb/bin; MCFM summary is canonical pb/bin.
    dy_pb_bin = float(dy.dyturbo_raw) / 1000.0
    dy_unc_pb_bin = float(dy.dyturbo_raw_unc) / 1000.0
    mc_scale = 0.5 if dy.dataset == "LHCb_7" else 1.0
    mc_pb_bin = float(mc.mcfm_pb_bin) * mc_scale
    mc_unc_pb_bin = float(mc.mcfm_pb_bin_unc) * mc_scale
    average = 0.5 * (abs(dy_pb_bin) + abs(mc_pb_bin))
    difference = abs(dy_pb_bin - mc_pb_bin) / average if average else np.nan
    dy_rel_mc = dy_unc_pb_bin / abs(dy_pb_bin) if dy_pb_bin else np.nan
    mc_rel_mc = mc_unc_pb_bin / abs(mc_pb_bin) if mc_pb_bin else np.nan
    max_diff = float(POLICY["external_code_gate"]["symmetric_relative_difference_max"])
    max_mc = float(POLICY["external_code_gate"]["max_relative_mc_uncertainty_each_code"])
    return {
        "dataset": dy.dataset, "row_id": dy.row_id, "qT": float(dy.qT),
        "qT_low": float(dy.qT_low), "qT_high": float(dy.qT_high),
        "qT_over_Q": float(dy.qT_over_Q), "QM_Low": float(dy.QM_Low), "QM_High": float(dy.QM_High),
        "bin_width_GeV": width, "data_pb_per_GeV": float(dy.data_pb_per_GeV),
        "dyturbo_pb_per_GeV": dy_pb_bin / width,
        "dyturbo_pb_per_GeV_unc": dy_unc_pb_bin / width,
        "mcfm_pb_per_GeV": mc_pb_bin / width,
        "mcfm_pb_per_GeV_unc": mc_unc_pb_bin / width,
        "mcfm_convention_scale": mc_scale,
        "dyturbo_relative_mc_uncertainty": dy_rel_mc,
        "mcfm_relative_mc_uncertainty": mc_rel_mc,
        "symmetric_relative_difference": difference,
        "numerical_precision_pass": bool(dy_rel_mc <= max_mc and mc_rel_mc <= max_mc),
        "central_external_agreement_pass": bool(difference <= max_diff),
        "central_pair_gate_pass": bool(difference <= max_diff and dy_rel_mc <= max_mc and mc_rel_mc <= max_mc),
        "dyturbo_summary": str(dy_path.relative_to(ROOT)), "mcfm_summary": str(mc_path.relative_to(ROOT)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True, choices=["tier1_boundary", "tier2_high_qt", "tier3_exceptional"])
    ap.add_argument("--stage", default="central", choices=["central", "precision"])
    args = ap.parse_args()
    base = STUDY / "outputs" / args.stage / args.tier
    records = []
    for dy_path in sorted(base.glob("*/*/dyturbo/dyturbo_benchmark_summary.csv")):
        mc_path = dy_path.parent.parent / "mcfm" / "mcfm_benchmark_summary.csv"
        if mc_path.exists():
            records.append(one_pair(dy_path, mc_path))
    # Include validated legacy pairs recorded by the source gate when the dense
    # campaign intentionally skipped them. Keep their provenance explicit.
    inventory = pd.read_csv(STUDY / "summaries" / "high_qt_candidate_inventory.csv")
    legacy = inventory.loc[
        inventory["benchmark_tier"].eq(args.tier)
        & inventory["external_benchmark_available"].fillna(False).astype(bool)
    ].copy() if args.stage == "central" else inventory.iloc[0:0].copy()
    completed_ids = {record["row_id"] for record in records}
    max_diff = float(POLICY["external_code_gate"]["symmetric_relative_difference_max"])
    max_mc = float(POLICY["external_code_gate"]["max_relative_mc_uncertainty_each_code"])
    for _, row in legacy.loc[~legacy["row_id"].isin(completed_ids)].iterrows():
        dy = float(row.dyturbo_pb_per_GeV)
        mc = float(row.mcfm_pb_per_GeV)
        dy_unc = float(row.dyturbo_pb_per_GeV_unc)
        mc_unc = float(row.mcfm_pb_per_GeV_unc)
        dy_rel = dy_unc / abs(dy)
        mc_rel = mc_unc / abs(mc)
        difference = float(row.dyturbo_mcfm_rel_diff)
        records.append({
            "dataset": row.dataset, "row_id": row.row_id, "qT": float(row.qT),
            "qT_low": float(row.qT_low), "qT_high": float(row.qT_high),
            "qT_over_Q": float(row.qT_over_Q), "QM_Low": float(row.QM_Low), "QM_High": float(row.QM_High),
            "bin_width_GeV": float(row.qT_high) - float(row.qT_low), "data_pb_per_GeV": float(row.CS),
            "dyturbo_pb_per_GeV": dy, "dyturbo_pb_per_GeV_unc": dy_unc,
            "mcfm_pb_per_GeV": mc, "mcfm_pb_per_GeV_unc": mc_unc,
            "mcfm_convention_scale": 0.5 if row.dataset == "LHCb_7" else 1.0,
            "dyturbo_relative_mc_uncertainty": dy_rel, "mcfm_relative_mc_uncertainty": mc_rel,
            "symmetric_relative_difference": difference,
            "numerical_precision_pass": bool(dy_rel <= max_mc and mc_rel <= max_mc),
            "central_external_agreement_pass": bool(difference <= max_diff),
            "central_pair_gate_pass": bool(difference <= max_diff and dy_rel <= max_mc and mc_rel <= max_mc),
            "dyturbo_summary": "legacy_source_gate", "mcfm_summary": "legacy_source_gate",
        })
    out_dir = STUDY / "summaries" / args.tier / args.stage
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(records).sort_values(["dataset", "qT_over_Q", "row_id"])
    table.to_csv(out_dir / "external_pairs.csv", index=False)
    status = {
        "tier": args.tier, "stage": args.stage, "n_complete_pairs": int(len(table)),
        "n_precision_pass": int(table["numerical_precision_pass"].sum()) if len(table) else 0,
        "n_external_agreement_pass": int(table["central_external_agreement_pass"].sum()) if len(table) else 0,
        "n_central_pair_gate_pass": int(table["central_pair_gate_pass"].sum()) if len(table) else 0,
        "max_symmetric_relative_difference": float(table["symmetric_relative_difference"].max()) if len(table) else None,
        "direct_production_approved": 0,
        "note": "Central external-pair passage is necessary but does not satisfy scale, PDF, matching, or fit-impact gates.",
    }
    (out_dir / "external_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
