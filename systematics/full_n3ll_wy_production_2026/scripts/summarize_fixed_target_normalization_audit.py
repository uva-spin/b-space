#!/usr/bin/env python3
"""Record the fixed-target nuclear-card normalization boundary.

This is an audit of already-completed quadrature probes.  It deliberately does
not choose a fixed-target normalization or create production rows.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
PROJECT = BASE.parents[1]
DATA_ROOT = PROJECT / "Data/v23a_fixed_target_plus_tevatron_absolute_fit_ready"
PROBES = BASE / "reports/fixed_target_quadrature_probes"
OUT = PROBES / "fixed_target_normalization_audit.json"


def table_value(path: Path) -> tuple[float, float]:
    lines = [line.split() for line in path.read_text().splitlines() if line and not line.startswith("#")]
    if not lines or len(lines[0]) < 2:
        raise RuntimeError(f"missing numeric table row: {path}")
    return float(lines[0][0]), float(lines[0][1])


def main() -> None:
    rows = []
    for status_path in sorted(PROBES.glob("*_nuc_*/probe_status.json")):
        status = json.loads(status_path.read_text())
        table = Path(status["table"])
        value, unc = table_value(table)
        dataset, row_id = status["dataset"], status["row_id"]
        data = pd.read_csv(DATA_ROOT / f"{dataset}.csv")
        source = data[data.row_id.eq(row_id)]
        if len(source) != 1:
            raise RuntimeError(f"source row is not unique: {row_id}")
        source = source.iloc[0]
        A = float(status["target"]["A"])
        rows.append({
            "dataset": dataset,
            "row_id": row_id,
            "Z": float(status["target"]["Z"]),
            "A": A,
            "qT_GeV": float(source.qT),
            "QM_GeV": float(source.QM),
            "data_CS": float(source.CS),
            "data_error": float(source.error),
            "whole_nucleus_prediction": value,
            "whole_nucleus_mc_uncertainty": unc,
            "per_nucleon_prediction": value / A,
            "whole_nucleus_ratio_to_data": value / float(source.CS),
            "per_nucleon_ratio_to_data": value / A / float(source.CS),
            "probe": str(status_path),
        })
    if not rows:
        raise SystemExit("no explicit nuclear-card probes found")
    result = {
        "status": "fixed_target_normalization_unresolved_not_production",
        "scope": "completed explicit Be/Cu nuclear-card quadrature probes only",
        "interpretation": "DYTurbo whole-nucleus output and the obvious per-nucleon division are both retained; neither is selected as a production convention",
        "rows": rows,
        "required_before_production": [
            "documented target composition for every fixed-target dataset, especially E772",
            "published-unit and per-nucleon normalization reconciliation",
            "same-scheme W+Y closure after the convention is fixed",
        ],
        "frozen_baseline_unchanged": True,
        "production_outputs_modified": False,
        "promotion_authorized": False,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
