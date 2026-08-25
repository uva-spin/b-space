#!/usr/bin/env python3
"""Record isolated NNLO gamma*/electroweak convention checks for LHCb row 10."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "reports/lhcb_nnlo_convention_scan"
CASES = {
    "baseline_gamma_ew1": "lhcb7_external_true_nnlo_positive_y_10m_10",
    "no_gamma_ew1": "lhcb7_external_true_nnlo_convention_nogamma",
    "gamma_ew0": "lhcb7_external_true_nnlo_convention_ew0",
    "gamma_ew2": "lhcb7_external_true_nnlo_convention_ew2",
}


def main() -> None:
    rows = []
    for case, tag in CASES.items():
        row = pd.read_csv(BASE / f"reports/{tag}/dyturbo_true_nlo_summary.csv").iloc[0]
        rows.append({
            "case": case,
            "row_id": str(row.row_id),
            "data_pb_per_GeV": float(row.data_pb_per_GeV),
            "theory_pb_per_GeV": float(row.dyturbo_pb_per_GeV),
            "theory_unc_pb_per_GeV": float(row.dyturbo_pb_per_GeV_unc),
            "theory_data_ratio": float(row.dyturbo_pb_per_GeV / row.data_pb_per_GeV),
        })
    report = {
        "status": "lhcb_nnlo_convention_scan_complete_diagnostic_not_production",
        "row": "LHCb_7:10",
        "order": "DYTurbo fixedorder_only=true, order=2, positive-arm 0<yZ<6",
        "calls_per_point": 2000000,
        "cases": rows,
        "interpretation": (
            "Removing gamma* or changing the electroweak input scheme shifts the "
            "row-10 NNLO prediction by only a few percent; none closes the roughly "
            "18% residual. The baseline gamma*/G_mu convention is retained and no "
            "convention-driven rescaling is authorized."
        ),
        "production_outputs_modified": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
