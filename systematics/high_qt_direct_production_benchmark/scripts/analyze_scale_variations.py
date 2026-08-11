#!/usr/bin/env python3
"""Canonicalize representative seven-point scale results and summarize envelopes."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "systematics/high_qt_direct_production_benchmark"
POLICY = json.loads((STUDY / "config/promotion_policy.json").read_text())


def decode(tag: str) -> tuple[float, float]:
    match = re.fullmatch(r"mur([0-9p]+)_muf([0-9p]+)", tag)
    if not match:
        raise ValueError(tag)
    return tuple(float(value.replace("p", ".")) for value in match.groups())


def main() -> None:
    central = pd.read_csv(STUDY / "summaries/tier1_boundary/central/external_pairs.csv").set_index("row_id")
    base = STUDY / "outputs/variations/tier1_boundary/scale"
    records = []
    for dy_path in sorted(base.glob("*/*/*/dyturbo/dyturbo_benchmark_summary.csv")):
        mc_path = dy_path.parent.parent / "mcfm/mcfm_benchmark_summary.csv"
        if not mc_path.exists():
            continue
        tag = dy_path.parents[3].name
        mur, muf = decode(tag)
        dy = pd.read_csv(dy_path).iloc[-1]
        mc = pd.read_csv(mc_path).iloc[-1]
        precision_mc_path = (STUDY / "outputs/precision_variations/tier1_boundary/scale" / tag /
                             str(dy.dataset).lower() / str(dy.row_id).replace(":", "_").lower() /
                             "mcfm/mcfm_benchmark_summary.csv")
        mc_source = str(mc_path.relative_to(ROOT))
        if precision_mc_path.exists():
            mc = pd.read_csv(precision_mc_path).iloc[-1]
            mc_source = str(precision_mc_path.relative_to(ROOT))
        width = float(dy.qT_high) - float(dy.qT_low)
        dy_value = float(dy.dyturbo_raw) / 1000.0 / width
        mc_value = float(mc.mcfm_pb_per_GeV) * (0.5 if dy.dataset == "LHCb_7" else 1.0)
        average = 0.5 * (abs(dy_value) + abs(mc_value))
        ref = central.loc[dy.row_id]
        records.append({
            "variant": tag, "mu_r_factor": mur, "mu_f_factor": muf,
            "dataset": dy.dataset, "row_id": dy.row_id, "qT_over_Q": float(dy.qT_over_Q),
            "dyturbo_pb_per_GeV": dy_value, "mcfm_pb_per_GeV": mc_value,
            "dyturbo_over_central": dy_value / float(ref.dyturbo_pb_per_GeV),
            "mcfm_over_central": mc_value / float(ref.mcfm_pb_per_GeV),
            "symmetric_code_difference": abs(dy_value - mc_value) / average,
            "mcfm_source": mc_source,
        })
    table = pd.DataFrame(records).sort_values(["dataset", "qT_over_Q", "row_id", "mu_r_factor", "mu_f_factor"])
    threshold = float(POLICY["external_code_gate"]["symmetric_relative_difference_max"])
    table["scale_point_code_agreement_pass"] = table.symmetric_code_difference <= threshold
    out = STUDY / "summaries/tier1_boundary/scale_dense"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "scale_variation_pairs.csv", index=False)
    rows = []
    for (dataset, row_id), group in table.groupby(["dataset", "row_id"], sort=True):
        rows.append({
            "dataset": dataset, "row_id": row_id, "qT_over_Q": float(group.qT_over_Q.iloc[0]),
            "n_noncentral_points": int(len(group)),
            "max_code_difference": float(group.symmetric_code_difference.max()),
            "n_code_agreement_pass": int(group.scale_point_code_agreement_pass.sum()),
            "dyturbo_ratio_min": float(group.dyturbo_over_central.min()),
            "dyturbo_ratio_max": float(group.dyturbo_over_central.max()),
            "mcfm_ratio_min": float(group.mcfm_over_central.min()),
            "mcfm_ratio_max": float(group.mcfm_over_central.max()),
            "all_scale_point_code_agreement_pass": bool(group.scale_point_code_agreement_pass.all()),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "scale_envelope_by_row.csv", index=False)
    status = {
        "n_rows": int(len(summary)), "n_noncentral_pairs": int(len(table)),
        "expected_noncentral_pairs": int(6 * len(summary)),
        "n_rows_all_scale_points_code_agree": int(summary.all_scale_point_code_agreement_pass.sum()),
        "max_code_difference": float(table.symmetric_code_difference.max()),
        "dyturbo_global_ratio_min": float(table.dyturbo_over_central.min()),
        "dyturbo_global_ratio_max": float(table.dyturbo_over_central.max()),
        "mcfm_global_ratio_min": float(table.mcfm_over_central.min()),
        "mcfm_global_ratio_max": float(table.mcfm_over_central.max()),
        "dense_scale_code_agreement_pass": bool(
            len(summary) == 25 and len(table) == 150 and summary.all_scale_point_code_agreement_pass.all()
        ),
        "note": "Envelope sizes are diagnostics, not a direct-production pass until the perturbative order and matched prediction are frozen.",
    }
    (out / "scale_variation_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
