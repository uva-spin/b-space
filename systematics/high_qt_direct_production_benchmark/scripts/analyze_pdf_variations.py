#!/usr/bin/env python3
"""Summarize the 50-member DYTurbo PDF propagation and optional MCFM checks."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "systematics/high_qt_direct_production_benchmark"


def main() -> None:
    central = pd.read_csv(STUDY / "summaries/tier1_boundary/central/external_pairs.csv").set_index("row_id")
    base = STUDY / "outputs/variations/tier1_boundary/pdf"
    records = []
    for path in sorted(base.glob("member_*/*/*/dyturbo/dyturbo_benchmark_summary.csv")):
        member = int(re.search(r"member_(\d+)", str(path)).group(1))
        row = pd.read_csv(path).iloc[-1]
        width = float(row.qT_high) - float(row.qT_low)
        records.append({"pdf_member": member, "dataset": row.dataset, "row_id": row.row_id,
                        "qT_over_Q": float(row.qT_over_Q),
                        "dyturbo_pb_per_GeV": float(row.dyturbo_raw) / 1000.0 / width})
    long = pd.DataFrame(records).sort_values(["dataset", "qT_over_Q", "row_id", "pdf_member"])
    out = STUDY / "summaries/tier1_boundary/pdf"
    out.mkdir(parents=True, exist_ok=True)
    long.to_csv(out / "dyturbo_pdf_members_long.csv", index=False)
    summaries = []
    for (dataset, row_id), group in long.groupby(["dataset", "row_id"], sort=True):
        values = group.dyturbo_pb_per_GeV
        ref = float(central.loc[row_id].dyturbo_pb_per_GeV)
        q16, q84 = values.quantile([0.16, 0.84])
        q025, q975 = values.quantile([0.025, 0.975])
        summaries.append({"dataset": dataset, "row_id": row_id,
                          "qT_over_Q": float(group.qT_over_Q.iloc[0]), "n_members": int(len(group)),
                          "central_member0_pb_per_GeV": ref, "member_mean_pb_per_GeV": float(values.mean()),
                          "q16_pb_per_GeV": float(q16), "q84_pb_per_GeV": float(q84),
                          "relative_68_halfwidth": float((q84 - q16) / (2 * abs(ref))),
                          "q025_pb_per_GeV": float(q025), "q975_pb_per_GeV": float(q975),
                          "relative_95_halfwidth": float((q975 - q025) / (2 * abs(ref)))})
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / "dyturbo_pdf_envelope_by_row.csv", index=False)
    checks = []
    for mc_path in sorted(base.glob("member_*/*/*/mcfm/mcfm_benchmark_summary.csv")):
        member = int(re.search(r"member_(\d+)", str(mc_path)).group(1))
        mc = pd.read_csv(mc_path).iloc[-1]
        dy_match = long[(long.pdf_member == member) & (long.row_id == mc.row_id)]
        if dy_match.empty:
            continue
        dy_value = float(dy_match.dyturbo_pb_per_GeV.iloc[-1])
        mc_value = float(mc.mcfm_pb_per_GeV) * (0.5 if mc.dataset == "LHCb_7" else 1.0)
        average = 0.5 * (abs(dy_value) + abs(mc_value))
        difference = abs(dy_value - mc_value) / average
        checks.append({"pdf_member": member, "dataset": mc.dataset, "row_id": mc.row_id,
                       "qT_over_Q": float(mc.qT_over_Q), "dyturbo_pb_per_GeV": dy_value,
                       "mcfm_pb_per_GeV": mc_value, "symmetric_code_difference": difference,
                       "pdf_member_code_agreement_pass": bool(difference <= 0.05)})
    check_table = pd.DataFrame(checks).sort_values(["pdf_member", "dataset", "qT_over_Q", "row_id"])
    check_table.to_csv(out / "selected_pdf_member_code_checks.csv", index=False)
    status = {"n_rows": int(len(summary)), "n_members_per_row_min": int(summary.n_members.min()),
              "n_members_per_row_max": int(summary.n_members.max()),
              "max_relative_68_halfwidth": float(summary.relative_68_halfwidth.max()),
              "median_relative_68_halfwidth": float(summary.relative_68_halfwidth.median()),
              "max_relative_95_halfwidth": float(summary.relative_95_halfwidth.max()),
              "full_dyturbo_pdf_propagation_complete": bool(len(summary) == 25 and (summary.n_members == 50).all()),
              "n_selected_mcfm_checks": int(len(check_table)),
              "n_selected_mcfm_checks_pass": int(check_table.pdf_member_code_agreement_pass.sum()),
              "max_selected_member_code_difference": float(check_table.symmetric_code_difference.max()),
              "external_pdf_gate_pass": bool(len(check_table) == 28 and check_table.pdf_member_code_agreement_pass.all()),
              "matched_prediction_pdf_gate_pass": False,
              "note": "External PDF gate does not approve direct use; matched-prediction PDF propagation remains required."}
    (out / "pdf_variation_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
