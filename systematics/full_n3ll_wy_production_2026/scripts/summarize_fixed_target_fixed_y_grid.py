#!/usr/bin/env python3
"""Summarize the isolated fixed-y fixed-target W+Y conversion grid."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
GRID_ROOT = BASE / "reports/fixed_target_fixed_y_grid_full"
OUT = BASE / "reports/fixed_target_fixed_y_grid_summary"
EXPECTED = {"E288_200": 31, "E288_300": 41, "E288_400": 63, "E605": 54, "E772": 54}


def main() -> None:
    frames = []
    status = {}
    for dataset, expected in EXPECTED.items():
        path = GRID_ROOT / dataset / "fixed_target_full_wy_grid.csv"
        if not path.exists():
            status[dataset] = {"complete": False, "rows": 0, "expected": expected}
            continue
        frame = pd.read_csv(path)
        frames.append(frame)
        ratios = frame["predicted_CS_to_data"].to_numpy(float)
        status[dataset] = {
            "complete": bool(len(frame) == expected),
            "rows": int(len(frame)),
            "expected": expected,
            "all_finite": bool(np.isfinite(ratios).all()),
            "all_positive": bool((frame["predicted_CS"].to_numpy(float) > 0).all()),
            "median_predicted_CS_to_data": float(np.median(ratios)),
            "p16_predicted_CS_to_data": float(np.quantile(ratios, 0.16)),
            "p84_predicted_CS_to_data": float(np.quantile(ratios, 0.84)),
            "min_predicted_CS_to_data": float(np.min(ratios)),
            "max_predicted_CS_to_data": float(np.max(ratios)),
        }
    OUT.mkdir(parents=True, exist_ok=True)
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(OUT / "fixed_target_fixed_y_grid.csv", index=False)
    complete = all(v["complete"] for v in status.values())
    summary = {
        "status": "isolated_fixed_target_fixed_y_n3ll_nnlo_wy_grid_complete_not_production" if complete else "isolated_fixed_target_fixed_y_n3ll_nnlo_wy_grid_in_progress",
        "formula": "W_N3LL + (FO_NNLO - ASY_NNLO)",
        "observable_convention": "fixed-y Eq. (3.3): A=I/[1000*A*pi*Delta(qT^2)*Delta(y)], CS=A/PreFactor",
        "native_dsdxf_used": False,
        "expected_rows": int(sum(EXPECTED.values())),
        "observed_rows": int(sum(v["rows"] for v in status.values())),
        "datasets": status,
        "interpretation": "diagnostic only; no fixed-target production promotion",
        "fixed_target_xf_identity_failure": "native dsdxf path failed fixed-y identity and remains excluded",
        "frozen_baseline_unchanged": True,
        "production_outputs_modified": False,
        "promotion_authorized": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
