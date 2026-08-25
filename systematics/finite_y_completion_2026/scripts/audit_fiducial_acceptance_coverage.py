#!/usr/bin/env python3
"""Record the finite-Y candidate's fiducial-acceptance coverage and gaps."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "systematics/finite_y_completion_2026"
GATE = ROOT / "systematics/finite_y_tail_benchmark/summaries/tail_benchmark_row_gate.csv"
UNITARY = ROOT / "systematics/high_qt_direct_production_benchmark/experimental_unitary_transition/summaries/unitary_smootherstep_v1_nlo_24row/nlo_frozen_unitary_pulls.csv"


def main() -> None:
    gate = pd.read_csv(GATE)
    unitary = pd.read_csv(UNITARY)
    coverage = {}
    for dataset, frame in gate.groupby("dataset", sort=True):
        high = frame[frame.region.eq("high_qT_candidate")]
        coverage[dataset] = {
            "candidate_rows": int(len(high)),
            "candidate_qT_range_GeV": [float(high.qT.min()), float(high.qT.max())],
            "external_benchmark_rows": int(high.external_benchmark_available.fillna(False).sum()),
            "external_benchmark_qT_GeV": [float(high.loc[high.external_benchmark_available.fillna(False), "qT"].min()),
                                            float(high.loc[high.external_benchmark_available.fillna(False), "qT"].max())]
            if high.external_benchmark_available.fillna(False).any() else [],
            "unitary_node_rows": int(sum(unitary.dataset.astype(str).eq(dataset))),
        }
    report = {
        "status": "fiducial_acceptance_coverage_audit_complete",
        "source_row_gate": str(GATE),
        "unitary_boundary_source": str(UNITARY),
        "candidate_high_qT_rows": int(gate.region.eq("high_qT_candidate").sum()),
        "candidate_high_qT_external_benchmark_rows": int((gate.region.eq("high_qT_candidate") & gate.external_benchmark_available.fillna(False)).sum()),
        "unitary_boundary_rows": int(len(unitary)),
        "unitary_boundary_datasets": sorted(unitary.dataset.astype(str).unique().tolist()),
        "coverage_by_dataset": coverage,
        "conclusion": "The unitary finite-Y node campaign is complete for 24 Tevatron boundary rows only. Existing external fiducial benchmarks cover selected high-qT rows, including one LHCb row, but no LHCb unitary node/kernel campaign exists; therefore broad Tevatron+LHCb finite-Y production remains unapproved until LHCb fiducial node integration is generated and audited.",
        "production_promotion": False,
    }
    out = WORK / "reports/fiducial_acceptance_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
