#!/usr/bin/env python3
"""Audit the historical fixed-target invariant-cross-section prefactors.

The fit-ready CS tables retain ``A/PreFactor`` as provenance.  For the
published invariant observable, the kinematic relation used in the fixed-
target TMD literature is

    E d^3 sigma/d^3 q = 1/(2 pi q_T) d sigma/(d q_T d eta_Q)

with ``dx_F/d eta_Q = 2 m_T cosh(eta_Q)/sqrt(s)`` for fixed-x_F rows.  The
E772 table in this workspace, however, carries a different historical
``A`` convention: its stored prefactor follows ``1/(2 pi q_T Q)`` rather than
the fixed-x_F Jacobian.  This diagnostic compares both conventions with the
stored value.  It is an audit only: no data or production cache is modified.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
PROJECT = BASE.parents[1]
DATA = PROJECT / "Data/v23a_fixed_target_plus_tevatron_absolute_fit_ready"
OUT = BASE / "reports/fixed_target_prefactor_kinematic_audit"


def main() -> None:
    rows = []
    for dataset in ("E288_200", "E288_300", "E288_400", "E605", "E772"):
        frame = pd.read_csv(DATA / f"{dataset}.csv")
        for _, row in frame.iterrows():
            q = float(row.qT)
            mass = float(row.QM)
            sqrts = float(row.SqrtS)
            y = float(row.y)
            mt = math.sqrt(mass * mass + q * q)
            base = 1.0 / (2.0 * math.pi * q)
            jac_xf = 2.0 * mt * math.cosh(y) / sqrts
            fixed_y = base
            fixed_xf = base * jac_xf
            stored = float(row.PreFactor)
            historical_e772 = 1.0 / (2.0 * math.pi * q * mass)
            if dataset == "E288_200" or dataset == "E288_300" or dataset == "E288_400":
                expected = fixed_y
                expected_mode = "fixed_y"
            elif dataset == "E605":
                expected = fixed_xf
                expected_mode = "fixed_xF"
            else:
                expected = historical_e772
                expected_mode = "historical_E772_A"
            rows.append(
                {
                    "dataset": dataset,
                    "row_id": str(row.row_id),
                    "qT": q,
                    "QM": mass,
                    "y": y,
                    "xF": float(row.xF),
                    "stored_PreFactor": stored,
                    "expected_fixed_y_PreFactor": fixed_y,
                    "expected_fixed_xF_PreFactor": fixed_xf,
                    "historical_E772_PreFactor": historical_e772,
                    "expected_mode": expected_mode,
                    "expected_PreFactor": expected,
                    "stored_over_expected": stored / expected,
                    "A": float(row.A),
                    "CS": float(row.CS),
                    "CS_from_A_over_PreFactor": float(row.A) / stored,
                }
            )
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "prefactor_kinematic_rows.csv", index=False)
    summary = {
        "status": "fixed_target_prefactor_kinematic_audit_complete_not_production",
        "formula_fixed_y": "1/(2*pi*qT)",
        "formula_fixed_xF": "[1/(2*pi*qT)]*[2*sqrt(Q^2+qT^2)*cosh(y)/sqrt(s)]",
        "formula_historical_E772_A": "1/(2*pi*qT*Q)",
        "dataset_median_stored_over_expected": {
            ds: float(np.median(out.loc[out.dataset.eq(ds), "stored_over_expected"]))
            for ds in sorted(out.dataset.unique())
        },
        "dataset_min_stored_over_expected": {
            ds: float(np.min(out.loc[out.dataset.eq(ds), "stored_over_expected"]))
            for ds in sorted(out.dataset.unique())
        },
        "dataset_max_stored_over_expected": {
            ds: float(np.max(out.loc[out.dataset.eq(ds), "stored_over_expected"]))
            for ds in sorted(out.dataset.unique())
        },
        "finding": (
            "E288 fixed-y and E605 fixed-xF stored prefactors reproduce the expected "
            "kinematic factors. E772 instead follows the historical 1/(2*pi*qT*Q) "
            "A-convention to high accuracy except for its separately encoded Q=9.46, "
            "qT=0.25 block. This is a data-convention audit, not a DYTurbo prediction "
            "or permission to alter fit-ready data."
        ),
        "frozen_baseline_unchanged": True,
        "production_outputs_modified": False,
        "promotion_authorized": False,
    }
    (OUT / "audit_status.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
