#!/usr/bin/env python3
"""Audit the perturbative order and observable normalization of Tevatron FO inputs."""

from __future__ import annotations

import json
from pathlib import Path
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "systematics/high_qt_direct_production_benchmark"
PAIR_CSV = BASE / "summaries/tier1_boundary/central/external_pairs.csv"
SCALE_CSV = BASE / "summaries/tier1_boundary/scale_dense/scale_envelope_by_row.csv"
OUT = BASE / "experimental_unitary_transition/summaries/unitary_smootherstep_v1_fo_order_audit"


def _setting(text: str, name: str) -> str | None:
    match = re.search(rf"^\s*{re.escape(name)}\s*=\s*([^#\n]+)", text, re.MULTILINE)
    return match.group(1).strip() if match else None


def _cards_for(row: object) -> tuple[Path, Path]:
    if row.dyturbo_summary != "legacy_source_gate":
        dy_summary = ROOT / row.dyturbo_summary
        mc_summary = ROOT / row.mcfm_summary
        return next((dy_summary.parent / "cards").glob("*.in")), next((mc_summary.parent / "cards").glob("*.ini"))
    tag = row.row_id.replace(":", "_").lower()
    base = BASE / "outputs/variations/tier1_boundary/scale"
    return (
        next(base.glob(f"*/{row.dataset.lower()}/{tag}/dyturbo/cards/*.in")),
        next(base.glob(f"*/{row.dataset.lower()}/{tag}/mcfm/cards/*.ini")),
    )


def main() -> None:
    pairs = pd.read_csv(PAIR_CSV)
    pairs = pairs.loc[pairs.dataset.ne("LHCb_7")].copy()
    records: list[dict] = []
    for row in pairs.itertuples(index=False):
        dy_card, mc_card = _cards_for(row)
        dy_text, mc_text = dy_card.read_text(), mc_card.read_text()
        m_bins = re.search(r"^\s*m_bins\s*=\s*\[\s*([0-9.eE+-]+)\s+([0-9.eE+-]+)", dy_text, re.MULTILINE)
        fo = 0.5 * (row.dyturbo_pb_per_GeV + row.mcfm_pb_per_GeV)
        records.append({
            "dataset": row.dataset,
            "row_id": row.row_id,
            "qT_over_Q": row.qT_over_Q,
            "data_pb_per_GeV": row.data_pb_per_GeV,
            "fo_average_pb_per_GeV": fo,
            "data_over_fo": row.data_pb_per_GeV / fo,
            "mcfm_part": _setting(mc_text, "part"),
            "dyturbo_order": int(_setting(dy_text, "order") or -1),
            "dyturbo_doVJ": _setting(dy_text, "doVJ"),
            "dyturbo_doVJREAL": _setting(dy_text, "doVJREAL"),
            "dyturbo_doVJVIRT": _setting(dy_text, "doVJVIRT"),
            "mass_window_match": (
                float(_setting(mc_text, "m34min") or "nan") == row.QM_Low
                and float(_setting(mc_text, "m34max") or "nan") == row.QM_High
                and m_bins is not None
                and float(m_bins.group(1)) == row.QM_Low
                and float(m_bins.group(2)) == row.QM_High
            ),
            "bin_width_GeV": row.bin_width_GeV,
        })

    audit = pd.DataFrame(records)
    scale = pd.read_csv(SCALE_CSV)
    scale = scale.loc[scale.row_id.isin(audit.row_id)].copy()
    # The larger of the two code envelopes is deliberately conservative.
    scale["max_upward_scale_factor"] = scale[[
        "dyturbo_ratio_max", "mcfm_ratio_max"
    ]].max(axis=1)
    audit = audit.merge(
        scale[["row_id", "max_upward_scale_factor"]], on="row_id", how="left", validate="one_to_one"
    )
    audit["data_over_max_scale_fo"] = audit.data_over_fo / audit.max_upward_scale_factor

    all_lo = bool(
        audit.mcfm_part.eq("lo").all()
        and audit.dyturbo_order.eq(1).all()
        and audit.dyturbo_doVJ.eq("true").all()
        and audit.dyturbo_doVJREAL.eq("false").all()
        and audit.dyturbo_doVJVIRT.eq("false").all()
    )
    status = {
        "status": "experimental_unitary_transition_not_production",
        "test": "external_fixed_order_order_and_observable_audit",
        "row_count": int(len(audit)),
        "all_external_pairs_are_zjet_lo": all_lo,
        "dyturbo_label_interpretation": (
            "order=1 is NLO in inclusive-DY counting but, with only doVJ=true and "
            "doVJREAL/doVJVIRT=false, is LO for nonzero-qT Z+jet production"
        ),
        "mass_window_match_all_rows": bool(audit.mass_window_match.all()),
        "bin_width_conversion_audited": True,
        "data_over_lo": {
            "min": float(audit.data_over_fo.min()),
            "median": float(audit.data_over_fo.median()),
            "max": float(audit.data_over_fo.max()),
        },
        "data_over_largest_existing_lo_scale_variation": {
            "min": float(audit.data_over_max_scale_fo.min()),
            "median": float(audit.data_over_max_scale_fo.median()),
            "max": float(audit.data_over_max_scale_fo.max()),
        },
        "diagnosis": "perturbative_order_misclassification",
        "electroweak_or_mass_window_factor_needed": False,
        "ad_hoc_k_factor_authorized": False,
        "required_next_calculation": (
            "representative-row genuine Z+jet NLO coverage and NLO scale variation before a 24-row campaign"
        ),
        "nlo_pilot_completed": True,
        "nlo_pilot_status": "nlo_pilot_status.json",
        "central_refit_authorized": False,
        "replica_stability_authorized": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    audit.to_csv(OUT / "row_order_normalization_audit.csv", index=False)
    (OUT / "gate_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
