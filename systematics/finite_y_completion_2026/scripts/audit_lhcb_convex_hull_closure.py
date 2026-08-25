#!/usr/bin/env python3
"""Bound what the unitary interpolation can achieve on LHCb rows 10--13.

For fixed-order input FO and an allowed transition profile p in [0,1], the
unitary prediction is a convex combination of W and FO.  This audit computes
an optimistic upper bound using all 96 lambda=1 endpoints, the NNLO central
input, and the largest positive NNLO scale excursion.  If that bound remains
below the data, changing F_NP endpoints or the profile cannot close the row.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
ENDPOINTS = BASE / "reports/lambda1_lhcb_unitary_nnlo/lhcb_unitary_lambda1_endpoints.csv"
NNLO_SCALE = BASE / "reports/lhcb_true_nnlo_scale_scan/summary.json"
OUT = BASE / "reports/lhcb_convex_hull_closure"
ROW_IDS = [f"LHCb_7:{i}" for i in (10, 11, 12, 13)]


def main() -> None:
    values = pd.read_csv(ENDPOINTS)
    scale_rows = {row["row_id"]: row for row in json.loads(NNLO_SCALE.read_text())["rows"]}
    records = []
    for row_id in ROW_IDS:
        sub = values[values.row_id.eq(row_id)]
        data = float(sub.CS.iloc[0])
        fo = float(sub.FO_DYTurbo.iloc[0])
        fo_high = float(scale_rows[row_id]["theory_max_pb_per_GeV"])
        w_min = float(sub.W_lambda1_fiducial.min())
        w_max = float(sub.W_lambda1_fiducial.max())
        profile_max = float(sub.matched_lambda1.max())
        # Optimistic arbitrary-profile bound: choose the larger of W and the
        # NNLO scale-high FO value, allowing any p in [0,1].
        optimistic = max(w_max, fo_high)
        records.append({
            "row_id": row_id,
            "data_pb_per_GeV": data,
            "FO_NNLO_central_pb_per_GeV": fo,
            "FO_NNLO_scale_high_pb_per_GeV": fo_high,
            "W_endpoint_min_pb_per_GeV": w_min,
            "W_endpoint_max_pb_per_GeV": w_max,
            "profiled_lambda1_max_pb_per_GeV": profile_max,
            "optimistic_convex_hull_upper_bound_pb_per_GeV": optimistic,
            "upper_bound_over_data": optimistic / data,
            "remaining_fraction_below_data": 1.0 - optimistic / data,
            "all_W_and_FO_below_data": bool(w_max < data and fo_high < data),
        })
    result = pd.DataFrame(records)
    OUT.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT / "lhcb_convex_hull_bound.csv", index=False)
    report = {
        "status": "isolated_lhcb_convex_hull_closure_complete_not_production",
        "construction": "matched=(1-p)W+p*FO with p in [0,1]",
        "endpoint_count": int(values.endpoint.nunique()),
        "rows": result.to_dict(orient="records"),
        "all_rows_have_W_and_FO_below_data": bool(result.all_W_and_FO_below_data.all()),
        "production_outputs_modified": False,
        "interpretation": (
            "For every high-qT row, even an optimistic arbitrary-profile convex "
            "combination of the endpoint W range and the NNLO scale-high FO term "
            "does not reach the data. Therefore F_NP endpoint selection or a "
            "different unitary profile cannot solve the residual; the missing "
            "closure must be in the fixed-order prediction, observable input, "
            "or an explicitly documented experimental issue."
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
