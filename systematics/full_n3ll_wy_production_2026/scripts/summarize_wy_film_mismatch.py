#!/usr/bin/env python3
"""Summarize the isolated W versus W+Y FiLM interface diagnosis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
REPORTS = BASE / "reports"
Y_SOURCE = REPORTS / "lhcb7_y_fullminusres_expcreg2p0_1m/lhcb7_y_full_minus_res.csv"
OUT = REPORTS / "w_y_film_mismatch_audit.json"


def metrics(tag: str) -> dict[str, object]:
    vals = []
    for seed in (303, 304, 305):
        path = REPORTS / f"{tag}_s{seed}/metrics.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        vals.append({
            "seed": seed,
            "best_chi2_like_per_row": data["train"]["best_chi2_like"],
            "best_epoch": data["train"]["best_epoch"],
            "per_dataset": {x["dataset"]: x["chi2_like"] for x in data["per_dataset"]},
        })
    return {"runs": vals,
            "range": [min((x["best_chi2_like_per_row"] for x in vals), default=np.nan),
                      max((x["best_chi2_like_per_row"] for x in vals), default=np.nan)]}


def main() -> None:
    y = pd.read_csv(Y_SOURCE)
    y_rel = y.Y_pb_per_GeV / y.CS
    y_rel_unc = y.Y_unc_pb_per_GeV / y.Y_pb_per_GeV.abs().clip(lower=1e-300)
    payload = {
        "status": "isolated_wy_film_interface_diagnosis_complete_not_production",
        "conclusion": (
            "The original large W+Y band was not a FiLM flexibility result. It combined "
            "stale row-ID pairing, raw rather than frozen-baseline effective errors, and "
            "a duplicate LHCb fiducial factor in the external W grid. After repairing those "
            "interfaces, W-only reproduces the frozen objective scale. The remaining W+Y "
            "fit degradation is localized to the six LHCb Y rows, whose full-minus-RES "
            "Monte-Carlo cancellation uncertainty is larger than their central values."
        ),
        "matched_control": {
            "wonly": metrics("scope_329_exactprotocol_aligned_wonly"),
            "wy_all": metrics("scope_329_exactprotocol_aligned_wy"),
            "wy_no_lhcb": metrics("scope_329_y_no_lhcb_decomp_wy"),
            "wy_lhcb_only": metrics("scope_329_y_lhcb_only_decomp_wy"),
            "wy_long_seed303": {
                "best_chi2_like_per_row": json.loads(
                    (REPORTS / "scope_329_exactprotocol_aligned_wy_long_wy_s303/metrics.json").read_text()
                )["train"]["best_chi2_like"],
                "best_epoch": json.loads(
                    (REPORTS / "scope_329_exactprotocol_aligned_wy_long_wy_s303/metrics.json").read_text()
                )["train"]["best_epoch"],
            },
        },
        "lhcb_y_audit": {
            "rows": int(len(y)),
            "Y_over_data": y_rel.tolist(),
            "Y_relative_subtraction_uncertainty": y_rel_unc.tolist(),
            "mean_abs_Y_over_data": float(np.mean(np.abs(y_rel))),
            "max_abs_Y_over_data": float(np.max(np.abs(y_rel))),
            "mean_relative_subtraction_uncertainty": float(np.mean(y_rel_unc)),
            "max_relative_subtraction_uncertainty": float(np.max(y_rel_unc)),
            "positive_Y_rows": int((y.Y_pb_per_GeV > 0).sum()),
            "negative_Y_rows": int((y.Y_pb_per_GeV < 0).sum()),
            "source_status": "isolated_lhcb7_y_full_minus_res_complete_not_production",
        },
        "interface_failures_repaired": [
            "scope data were remapped to dense trainer row IDs after W selection, so stale source row IDs paired some W curves with the wrong kinematics",
            "the candidate controls initially used raw table errors rather than the Collins effective errors used by the frozen baseline",
            "the external W cache already followed the frozen fiducial convention; multiplying LHCb rows by theory_fiducial_factor again suppressed W by about 0.45",
        ],
        "corrected_inputs": {
            "data": str(REPORTS / "scope_353_fnp_inputs/data_with_old_effective_errors"),
            "w_grid": str(REPORTS / "scope_353_fnp_inputs/scope_353_bspace_w_kinematic_corrected_nofidfactor.csv"),
            "y_grid": str(REPORTS / "scope_353_fnp_inputs/scope_353_y.csv"),
            "backend": "/home/dustin/work/project/bT-TMD/v23/backends/bt_internal_css_backend_v22_tevatron.py",
        },
        "decision": (
            "Do not interpret the old multi-hundred-percent W+Y band as FiLM non-uniqueness. "
            "For Tevatron/fixed-target rows the corrected W+Y fit is numerically near the baseline. "
            "LHCb Y remains a diagnostic-only input until its cancellation uncertainty and fiducial "
            "W/Y convention are independently reduced/validated; it must not be treated as an exact "
            "production constraint in the current global propagation."
        ),
        "frozen_production_modified": False,
        "promotion_authorized": False,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
