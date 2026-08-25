#!/usr/bin/env python3
"""Collect the isolated positive-arm NNLO DYTurbo boundary rows.

This is a derived diagnostic input only.  The four LHCb boundary rows were
run in separate, independently sampled DYTurbo jobs; this file gives the
unitary endpoint audit one explicit, immutable source table without touching
the released data or frozen production outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "systematics/finite_y_completion_2026/reports"
OUT = BASE / "lhcb7_external_true_nnlo_positive_y_10m_combined"
ROWS = ("10", "11", "12", "13")


def main() -> None:
    frames = []
    sources = {}
    for row in ROWS:
        source = BASE / f"lhcb7_external_true_nnlo_positive_y_10m_{row}" / "dyturbo_true_nlo_summary.csv"
        if not source.exists():
            raise FileNotFoundError(source)
        frame = pd.read_csv(source)
        expected = f"LHCb_7:{row}"
        frame = frame[frame["row_id"].eq(expected)].copy()
        if len(frame) != 1:
            raise RuntimeError(f"expected one NNLO row for {expected}, found {len(frame)}")
        frames.append(frame)
        sources[expected] = str(source)
    combined = pd.concat(frames, ignore_index=True).sort_values("row_id")
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "dyturbo_true_nnlo_summary.csv"
    combined.to_csv(output, index=False)
    summary = {
        "status": "isolated_lhcb_positive_arm_nnlo_source_combined",
        "order": "DYTurbo order=2 with doVJREAL/doVJVIRT=true",
        "rapidity_scope": "positive boson rapidity arm 0<y_Z<6; LHCb fiducial lepton cuts retained",
        "row_ids": combined["row_id"].tolist(),
        "source_files": sources,
        "output": str(output),
        "production_outputs_modified": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
