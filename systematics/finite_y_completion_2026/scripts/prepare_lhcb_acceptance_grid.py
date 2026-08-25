#!/usr/bin/env python3
"""Merge validated DYTurbo node acceptance grids for the four LHCb tail bins."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "systematics/finite_y_completion_2026/reports"
GRID_10_12 = WORK / "lhcb_node_acceptance_grid_highqt_q2y6/lhcb_node_acceptance.csv"
GRID_13 = WORK / "lhcb_node_acceptance_grid_lhcb13_q4y6/lhcb_node_acceptance.csv"
BIN_CHECK = WORK / "lhcb_bin_acceptance_check/lhcb_bin_acceptance.csv"
OUT = WORK / "lhcb_node_acceptance_grid_production_candidate"


def weighted(frame: pd.DataFrame) -> float:
    weight = frame.q_weight * frame.y_weight
    return float((frame.fiducial_pb * weight).sum() / (frame.inclusive_pb * weight).sum())


def main() -> None:
    a = pd.read_csv(GRID_10_12)
    b = pd.read_csv(GRID_13)
    frame = pd.concat([a[a.row_id.ne("LHCb_7:13")], b], ignore_index=True)
    checks = pd.read_csv(BIN_CHECK).set_index("row_id")
    rows = []
    for row_id, group in frame.groupby("row_id", sort=False):
        integrated = float(checks.loc[row_id, "acceptance"])
        node = weighted(group)
        rows.append({
            "row_id": row_id, "qT_node_order": int(group.qT.nunique()),
            "rapidity_node_order": int(group.y.nunique()),
            "node_count": int(len(group)), "node_weighted_acceptance": node,
            "full_bin_dyturbo_acceptance": integrated,
            "relative_grid_vs_bin": (node - integrated) / integrated,
            "min_node_acceptance": float(group.acceptance.min()),
            "max_node_acceptance": float(group.acceptance.max()),
            "zero_cross_section_nodes": int((group.inclusive_pb == 0.0).sum()),
        })
    summary = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "lhcb_node_acceptance.csv", index=False)
    summary.to_csv(OUT / "per_row_validation.csv", index=False)
    report = {
        "status": "lhcb_node_acceptance_grid_candidate_prepared",
        "rows": summary.to_dict(orient="records"),
        "max_abs_grid_vs_full_bin_relative": float(np.max(np.abs(summary.relative_grid_vs_bin))),
        "all_nonzero_nodes_finite": bool(np.isfinite(frame.acceptance).all()),
        "acceptance_range": [float(frame.acceptance.min()), float(frame.acceptance.max())],
        "interpretation": "DYTurbo NLO fixed-order fiducial/inclusive ratios are now available at the explicit finite-Y qT/y nodes. Rows 10--12 use 2 qT x 6 y nodes; the broad row 13 uses 4 qT x 6 y nodes. This remains an isolated acceptance input until the LHCb W kernels and unitary fit impact are audited.",
        "production_outputs_modified": False,
    }
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
