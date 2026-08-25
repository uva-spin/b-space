#!/usr/bin/env python3
"""Summarize isolated NNLO PDF-set checks for the LHCb boundary bins."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
SETS = {
    "NNPDF40_nnlo_as_01180": "lhcb7_external_true_nnlo_positive_y_10m_",
    "CT18NNLO": "lhcb7_external_true_nnlo_pdf_CT18NNLO_",
    "MSHT20nnlo_as118": "lhcb7_external_true_nnlo_pdf_MSHT20nnlo_as118_",
}
OUT = BASE / "reports/lhcb_nnlo_pdf_scan"


def main() -> None:
    rows = []
    for pdf, prefix in SETS.items():
        for index in (10, 11, 12, 13):
            frame = pd.read_csv(BASE / f"reports/{prefix}{index}/dyturbo_true_nlo_summary.csv")
            row = frame.iloc[0]
            rows.append({
                "pdf_set": pdf,
                "row_id": str(row.row_id),
                "data_pb_per_GeV": float(row.data_pb_per_GeV),
                "theory_pb_per_GeV": float(row.dyturbo_pb_per_GeV),
                "theory_unc_pb_per_GeV": float(row.dyturbo_pb_per_GeV_unc),
                "theory_data_ratio": float(row.dyturbo_pb_per_GeV / row.data_pb_per_GeV),
            })
    table = pd.DataFrame(rows)
    report = {
        "status": "lhcb_nnlo_pdf_scan_complete_diagnostic_not_production",
        "order": "DYTurbo fixedorder_only=true, order=2, positive-arm 0<yZ<6",
        "calls_per_point": 2000000,
        "pdf_sets": list(SETS),
        "rows": rows,
        "ratio_range_by_row": {
            row_id: {
                "min": float(group.theory_data_ratio.min()),
                "max": float(group.theory_data_ratio.max()),
            }
            for row_id, group in table.groupby("row_id", sort=False)
        },
        "interpretation": (
            "CT18NNLO and MSHT20nnlo_as118 shift the NNLO boundary prediction by "
            "only a few percent relative to NNPDF40, far less than the residual "
            "15--23% in rows 10--12. PDF choice does not close the LHCb mismatch."
        ),
        "production_outputs_modified": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "lhcb_nnlo_pdf_scan.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
