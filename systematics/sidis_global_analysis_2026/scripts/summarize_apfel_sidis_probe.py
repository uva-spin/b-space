#!/usr/bin/env python3
"""Summarize the isolated APFEL SIDIS coefficient/denominator probes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "outputs/apfel_sidis_nlo_full_den_probe.csv"
FIT = ROOT / "outputs/initial_joint_dy_compass_apfel_nlo_full_den_probe_validated_named/fit_summary.json"
OUT_JSON = ROOT / "reports/apfel_sidis_interface_probe.json"
OUT_MD = ROOT / "reports/apfel_sidis_interface_probe.md"


def main() -> None:
    table = pd.read_csv(PROBE)
    fit = json.loads(FIT.read_text()) if FIT.exists() else None
    finite = np.isfinite(table.nlo_full_den_ratio.to_numpy(float))
    positive = finite & (table.nlo_full_den_ratio.to_numpy(float) > 0)
    ratio_shift = table.nlo_full_den_ratio / table.nlo_numerator_lo_den_ratio
    representative = table.iloc[0]
    report = {
        "status": "exploratory_apfel_sidis_coefficient_and_denominator_probe_complete_not_production",
        "backend": "APFEL++ InitializeSIDIS plus BuildStructureFunctions (local build outside source tree)",
        "scope": f"{len(table)} identified COMPASS 2026 pi/K collinear rows",
        "pdf": "NNPDF40_nlo_as_01180 member 0",
        "ff": "NNFF10 identified pion/kaon NLO grids member 0",
        "expansion": "alpha_s/(4 pi)",
        "operators": ["C20qq", "C21qq", "C21qg", "C21gq", "CL1qq", "CL1qg", "CL1gq"],
        "lo_positive_rows": int(np.sum(table.lo_ratio > 0)),
        "nlo_numerator_positive_rows": int(np.sum(table.nlo_numerator_lo_den_ratio > 0)),
        "nlo_full_den_positive_rows": int(np.sum(positive)),
        "full_to_lo_den_numerator_ratio_quantiles": {
            "0.05": float(np.nanpercentile(ratio_shift, 5)),
            "0.50": float(np.nanpercentile(ratio_shift, 50)),
            "0.95": float(np.nanpercentile(ratio_shift, 95)),
        },
        "representative_row": {
            "row_id": str(representative.row_id),
            "x": float(representative.x),
            "z": float(representative.z),
            "Q_GeV": float(representative.Q_reconstructed),
            "y": float(representative.y),
            "lo_ratio": float(representative.lo_ratio),
            "nlo_numerator_lo_den_ratio": float(representative.nlo_numerator_lo_den_ratio),
            "nlo_full_den_ratio": float(representative.nlo_full_den_ratio),
        },
        "denominator_scope": "NLO massless inclusive DIS F2 and FL through APFEL Observable interface; bin integration and heavy-quark scheme are not yet validated",
        "fit": None if fit is None else {
            "path": str(FIT.parent),
            "dy_chi2_per_row": fit["objective"]["dy_chi2_per_row"],
            "sidis_chi2_per_row": fit["objective"]["sidis_chi2_per_row"],
            "sidis_rows_fit": fit["sidis_rows_fit"],
            "sidis_rows_excluded": len(fit["sidis_rows_excluded"]),
            "sidis_normalizations": fit["sidis_normalizations"],
        },
        "fit_used": fit is not None,
        "production_files_modified": False,
        "promotion_authorized": False,
    }
    OUT_JSON.write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# APFEL++ SIDIS coefficient and denominator probe (2026-08-26)",
        "",
        "This isolated diagnostic evaluates the massless NLO SIDIS C20/C21/CL1 operators and the NLO inclusive-DIS F2/FL denominator with the same PDF member. It is not a production prediction.",
        "",
        f"The row-level table contains {len(table)} identified COMPASS 2026 pi/K rows. The LO ratio is positive for {report['lo_positive_rows']} rows; the SIDIS-NLO numerator over an LO denominator is positive for {report['nlo_numerator_positive_rows']}; using the full NLO denominator leaves {report['nlo_full_den_positive_rows']} positive rows.",
        f"Relative to the LO-denominator numerator diagnostic, the full-denominator ratio has median multiplicative shift {report['full_to_lo_den_numerator_ratio_quantiles']['0.50']:.4f} (5--95% range {report['full_to_lo_den_numerator_ratio_quantiles']['0.05']:.4f}--{report['full_to_lo_den_numerator_ratio_quantiles']['0.95']:.4f}).",
        "",
        "The full denominator is now assembled through APFEL's Observable path, which includes the NLO coefficient-function and PDF-evolution terms. The remaining validation gates are bin-averaged phase-space integration, scale/threshold choices, and covariance-consistent normalization. The eight non-positive rows are retained in the manifest and excluded from the positive-ratio pilot rather than positivity-clipped.",
    ]
    if fit is not None:
        lines += [
            "",
            f"The corresponding isolated joint-fit diagnostic gives DY chi2/row = {fit['objective']['dy_chi2_per_row']:.4f} and SIDIS chi2/row = {fit['objective']['sidis_chi2_per_row']:.4f} on {fit['sidis_rows_fit']} rows, with {len(fit['sidis_rows_excluded'])} rows excluded for non-positive theory ratios. This is an interface test, not a promotion candidate.",
        ]
    lines += ["", "No frozen production files were modified."]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
