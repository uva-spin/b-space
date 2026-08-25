#!/usr/bin/env python3
"""Run the isolated lambda=1 LHCb unitary endpoint audit with NNLO FO input."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "systematics/finite_y_completion_2026"
SOURCE = BASE / "reports/lhcb7_external_true_nnlo_positive_y_10m_combined/dyturbo_true_nnlo_summary.csv"
OUT = BASE / "reports/lambda1_lhcb_unitary_nnlo"
SOURCE_SCRIPT = BASE / "scripts/recompute_lambda1_lhcb_unitary.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    source = load_module("recompute_lambda1_lhcb_unitary_nnlo_source", SOURCE_SCRIPT)
    source.DY_TRUE = SOURCE
    source.OUT = OUT
    source.main()
    summary_path = OUT / "summary.json"
    report = json.loads(summary_path.read_text())
    report.update({
        "status": "lambda1_lhcb_unitary_nnlo_endpoint_recompute_complete_diagnostic_not_production",
        "fixed_order_order": "DYTurbo order=2 (NNLO), doVJREAL/doVJVIRT=true",
        "rapidity_scope": "positive boson rapidity arm 0<y_Z<6",
        "production_outputs_modified": False,
        "next_step": "Audit the NNLO endpoint ensemble with published correlated LHCb covariance and scale variations.",
    })
    summary_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
