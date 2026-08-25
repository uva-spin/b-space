#!/usr/bin/env python3
"""Audit the LHCb 7 TeV pT observable and local data conversion.

This records checks that can be established from the released table and the
candidate manifest without modifying the source data.  The publication
identification and fiducial cuts are recorded explicitly so the remaining
finite-Y blocker is not confused with a hidden per-bin normalization issue.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]
RAW = ROOT / "Data/global_dy_raw/LHCb_7.csv"
CANDIDATE = ROOT / "Data/v23a_tevatron_plus_lhcb7_fiducial_candidate/LHCb_7.csv"
OUT = BASE / "reports/lhcb_observable_provenance_audit.json"


def load_covariance_module():
    path = BASE / "scripts/audit_lhcb_correlated_covariance.py"
    spec = importlib.util.spec_from_file_location("lhcb_covariance_provenance", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    raw = pd.read_csv(RAW).iloc[:14].copy()
    candidate = pd.read_csv(CANDIDATE).copy()
    expected_a = raw["A_raw_cs"] / raw["dPT"]
    candidate_high = candidate[candidate.row_id.isin([f"LHCb_7:{i}" for i in range(10, 14)])].sort_values("source_row")
    corr = load_covariance_module().correlation_matrix()
    checks = {
        "raw_row_count": int(len(raw)),
        "candidate_row_count": int(len(candidate)),
        "raw_bin_edges_contiguous": bool(np.allclose(raw["PT Low"].to_numpy(float)[1:], raw["PT High"].to_numpy(float)[:-1])),
        "raw_A_is_bin_integrated_over_dPT": bool(np.allclose(raw["A"].to_numpy(float), expected_a.to_numpy(float), rtol=0.0, atol=1e-9)),
        "candidate_A_matches_raw_per_GeV": bool(np.allclose(candidate["A"].to_numpy(float), np.repeat(raw["A"].to_numpy(float), 1), rtol=0.0, atol=1e-12)) if len(candidate) == len(raw) else False,
        "sum_raw_bin_cross_section_pb": float(raw["A_raw_cs"].sum()),
        "published_total_cross_section_pb": 76.0,
        "sum_matches_published_total_within_rounding": bool(abs(float(raw["A_raw_cs"].sum()) - 76.0) < 0.1),
        "high_qT_rows": candidate_high[["row_id", "qT", "qT_low", "qT_high", "qT_over_Q", "A", "error"]].to_dict(orient="records"),
        "published_correlation_matrix_shape": list(corr.shape),
        "published_correlation_matrix_symmetric": bool(np.allclose(corr, corr.T)),
        "published_correlation_matrix_diagonal_one": bool(np.allclose(np.diag(corr), 1.0)),
        "fiducial_definition": {
            "lepton_pT_GeV_min": 20.0,
            "lepton_eta": [2.0, 4.5],
            "dimuon_mass_GeV": [60.0, 120.0],
            "sqrt_s_GeV": 7000.0,
            "publication": "LHCb, arXiv:1505.07024, JHEP 08 (2015) 039",
        },
        "source_paths": {"raw": str(RAW), "candidate": str(CANDIDATE)},
        "production_outputs_modified": False,
    }
    checks["interpretation"] = (
        "The local LHCb table is an absolute fiducial cross-section table: its "
        "bin-integrated values sum to the published 76.0 pb total, and the "
        "candidate per-GeV values are the bin-integrated values divided by dPT. "
        "This rules out a hidden global normalized-spectrum conversion as the "
        "source of the high-qT residual. The covariance reconstruction remains "
        "an isolated audit until its full publication manifest is formally "
        "promoted."
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
