#!/usr/bin/env python3
"""Compare alternate-seed predictions with the Tier-1 central campaign."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "systematics/high_qt_direct_production_benchmark"
POLICY = json.loads((STUDY / "config/promotion_policy.json").read_text())


def symmetric(a: float, b: float) -> float:
    average = 0.5 * (abs(a) + abs(b))
    return abs(a - b) / average if average else np.nan


def main() -> None:
    central = pd.read_csv(STUDY / "summaries/tier1_boundary/central/external_pairs.csv").set_index("row_id")
    base = STUDY / "outputs/variations/tier1_boundary/seed/alternate_seed"
    records = []
    for dy_path in sorted(base.glob("*/*/dyturbo/dyturbo_benchmark_summary.csv")):
        mc_path = dy_path.parent.parent / "mcfm/mcfm_benchmark_summary.csv"
        if not mc_path.exists():
            continue
        dy = pd.read_csv(dy_path).iloc[-1]
        mc = pd.read_csv(mc_path).iloc[-1]
        ref = central.loc[dy.row_id]
        width = float(dy.qT_high) - float(dy.qT_low)
        dy_value = float(dy.dyturbo_raw) / 1000.0 / width
        mc_value = float(mc.mcfm_pb_per_GeV) * (0.5 if dy.dataset == "LHCb_7" else 1.0)
        seed_ref_path = (STUDY / "outputs/seed_reference/tier1_boundary" /
                         str(dy.dataset).lower() / str(dy.row_id).replace(":", "_").lower() /
                         "mcfm/mcfm_benchmark_summary.csv")
        mc_central = float(ref.mcfm_pb_per_GeV)
        mc_reference = "central_external_pairs"
        if seed_ref_path.exists():
            seed_ref = pd.read_csv(seed_ref_path).iloc[-1]
            mc_central = float(seed_ref.mcfm_pb_per_GeV) * (0.5 if dy.dataset == "LHCb_7" else 1.0)
            mc_reference = str(seed_ref_path.relative_to(ROOT))
        records.append({
            "dataset": dy.dataset, "row_id": dy.row_id, "qT_over_Q": float(dy.qT_over_Q),
            "dyturbo_central_pb_per_GeV": float(ref.dyturbo_pb_per_GeV),
            "dyturbo_alternate_pb_per_GeV": dy_value,
            "dyturbo_seed_symmetric_shift": symmetric(float(ref.dyturbo_pb_per_GeV), dy_value),
            "mcfm_central_pb_per_GeV": mc_central,
            "mcfm_alternate_pb_per_GeV": mc_value,
            "mcfm_seed_symmetric_shift": symmetric(mc_central, mc_value),
            "mcfm_seed_reference": mc_reference,
        })
    table = pd.DataFrame(records).sort_values(["dataset", "qT_over_Q", "row_id"])
    limit = float(POLICY["seed_reproducibility"]["symmetric_relative_shift_max"])
    table["dyturbo_seed_pass"] = table["dyturbo_seed_symmetric_shift"] <= limit
    table["mcfm_seed_pass"] = table["mcfm_seed_symmetric_shift"] <= limit
    table["seed_pair_pass"] = table["dyturbo_seed_pass"] & table["mcfm_seed_pass"]
    out = STUDY / "summaries/tier1_boundary/seed"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "seed_reproducibility.csv", index=False)
    status = {
        "n_rows": int(len(table)), "threshold": limit,
        "n_dyturbo_pass": int(table.dyturbo_seed_pass.sum()),
        "n_mcfm_pass": int(table.mcfm_seed_pass.sum()),
        "n_pair_pass": int(table.seed_pair_pass.sum()),
        "max_dyturbo_shift": float(table.dyturbo_seed_symmetric_shift.max()),
        "max_mcfm_shift": float(table.mcfm_seed_symmetric_shift.max()),
        "p95_dyturbo_shift": float(table.dyturbo_seed_symmetric_shift.quantile(0.95)),
        "p95_mcfm_shift": float(table.mcfm_seed_symmetric_shift.quantile(0.95)),
        "pass": bool(table.seed_pair_pass.all()),
    }
    (out / "seed_reproducibility_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
