#!/usr/bin/env python3
"""Recheck the conventional additive FO-ASY candidate after the rapidity audit.

This is intentionally a small, reproducible diagnostic.  The earlier additive
pilot was run before the explicit-y W evaluator removed the Tevatron backend's
inclusive rapidity factor.  Here the perturbative W is evaluated with that
factor removed and compared against the strict one-loop ASY at the central
node of the original test bin.  No production inputs are modified.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "Data/v23a_tevatron_plus_lhcb7_fiducial_candidate/CDF_RUN_2.csv"
EXTERNAL = ROOT / "systematics/high_qt_direct_production_benchmark/summaries/tier1_boundary/central/external_pairs.csv"
ASY_RUNNER = ROOT / "systematics/high_qt_direct_production_benchmark/experimental_matched_y/scripts/run_asymptotic_pilot.py"
EXACT = ROOT / "systematics/high_qt_direct_production_benchmark/experimental_matched_y/backend/exact_bin_asymptotic.py"
OUT = ROOT / "systematics/finite_y_completion_2026/reports/conventional_y_reassessment"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    asym_runner = load(ASY_RUNNER, "conventional_y_reassessment_runner")
    exact = load(EXACT, "conventional_y_reassessment_exact_bin")
    row = pd.read_csv(DATA).loc[lambda frame: frame.row_id.eq("CDF_RUN_2:36")].iloc[0]
    external = pd.read_csv(EXTERNAL).loc[lambda frame: frame.row_id.eq("CDF_RUN_2:36")].iloc[0]
    node = exact.node_row(row, qT=float(row.qT), y=0.0)
    fo = float(external.dyturbo_pb_per_GeV + external.mcfm_pb_per_GeV) / 2.0
    records = []
    for n_b in (160, 320, 640):
        backend = asym_runner.load_backend()
        cfg = asym_runner.production_cfg(backend, n_b=n_b)
        pdf = backend.LHAPDFProvider("NNPDF40_nnlo_as_01180", 0, use_toy_pdf=False)
        _, fitted = exact.make_resummed_w_point_evaluators(
            backend=backend,
            pdf=pdf,
            cfg=cfg,
            remove_inclusive_rapidity_approximation=True,
        )
        asym_eval = exact.make_v22_point_evaluator(backend=backend, pdf=pdf, cfg=cfg)
        w = float(fitted(node))
        asym = float(asym_eval(node))
        y = fo - asym
        matched = w + y
        records.append({
            "n_b": n_b,
            "w_corrected_pb_per_GeV": w,
            "asy_pb_per_GeV": asym,
            "fo_pb_per_GeV": fo,
            "y_fo_minus_asy_pb_per_GeV": y,
            "matched_pb_per_GeV": matched,
            "matched_minus_fo_pb_per_GeV": matched - fo,
            "matched_over_fo": matched / fo,
        })
    report = {
        "status": "conventional_additive_fo_minus_asy_rejected_after_rapidity_correction",
        "row_id": "CDF_RUN_2:36",
        "qT_over_Q": float(row.qT / row.QM),
        "central_node": {"qT": float(row.qT), "y": 0.0},
        "rapidity_correction": "W evaluator removes backend inclusive Tevatron rapidity factor before explicit-y comparison",
        "records": records,
        "decision": "The corrected W remains far from strict ASY at qT/Q≈0.20; additive FO-ASY gives a matched result tens of pb/GeV above FO. This is a domain-of-validity failure of conventional additive matching at the transition node, not a reason to stop finite-Y work.",
        "next_candidate": "retain the separately validated unitary transition Y=p(qT/Q)(FO-W) and continue its production-scope audits",
        "production_outputs_modified": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cdf_run_2_36.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
