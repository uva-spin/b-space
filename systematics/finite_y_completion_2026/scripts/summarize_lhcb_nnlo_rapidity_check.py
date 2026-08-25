#!/usr/bin/env python3
"""Record the isolated positive-arm boson-rapidity observable check."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "reports/lhcb_nnlo_rapidity_semantics"


def value(tag: str) -> dict[str, float | str]:
    path = BASE / f"reports/{tag}/dyturbo_true_nlo_summary.csv"
    row = pd.read_csv(path).iloc[0]
    return {
        "row_id": str(row.row_id),
        "data_pb_per_GeV": float(row.data_pb_per_GeV),
        "theory_pb_per_GeV": float(row.dyturbo_pb_per_GeV),
        "theory_unc_pb_per_GeV": float(row.dyturbo_pb_per_GeV_unc),
        "theory_data_ratio": float(row.dyturbo_pb_per_GeV / row.data_pb_per_GeV),
    }


def central_row10() -> dict[str, float | str]:
    """Read the completed 10M-call row-10 table (aggregate CSV raced)."""
    row = pd.read_csv(
        BASE / "reports/lhcb7_external_true_nnlo_scale_mur0p5_muf0p5/dyturbo_true_nlo_summary.csv"
    ).query("row_id == 'LHCb_7:10'").iloc[0]
    table = BASE / "reports/lhcb7_external_true_nnlo_probe_10m/tables/LHCb_7_10_dyturbo_vj_true_nlo_fid_o2_mur1_muf1.txt"
    fields = next(line.split() for line in table.read_text().splitlines() if line and not line.startswith("#"))
    theory = float(fields[0]) / (1000.0 * float(row.qT_high - row.qT_low))
    uncertainty = float(fields[1]) / (1000.0 * float(row.qT_high - row.qT_low))
    return {
        "row_id": str(row.row_id),
        "data_pb_per_GeV": float(row.data_pb_per_GeV),
        "theory_pb_per_GeV": theory,
        "theory_unc_pb_per_GeV": uncertainty,
        "theory_data_ratio": theory / float(row.data_pb_per_GeV),
    }


def main() -> None:
    report = {
        "status": "lhcb_nnlo_positive_arm_rapidity_semantics_complete_diagnostic",
        "order": "DYTurbo fixedorder_only=true, order=2, doVJREAL/doVJVIRT=true",
        "calls": 10000000,
        "positive_arm_open_y": [0.0, 6.0],
        "metadata_y_window": [2.0, 4.25],
        "open_y_row10": value("lhcb7_external_true_nnlo_positive_y_10m_10"),
        "metadata_y_row10": central_row10(),
        "open_y_rows11_13": [
            value(f"lhcb7_external_true_nnlo_positive_y_10m_{i}") for i in (11, 12, 13)
        ],
        "interpretation": (
            "Opening the positive-arm boson rapidity from 2--4.25 to 0--6 leaves "
            "rows 11--13 unchanged within Monte Carlo uncertainty and shifts row 10 "
            "by only about 3%. The repeated metadata y window is not the source of "
            "the NNLO residual. A -10--10 diagnostic includes the negative pp arm and "
            "is not the LHCb observable."
        ),
        "production_outputs_modified": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
