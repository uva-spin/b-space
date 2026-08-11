#!/usr/bin/env python3
"""Build diagnostic fit-ready CERN/LHC pp DY tables from Data/global_dy_raw."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DATASETS = [
    "CMS_7",
    "CMS_8",
    "CMS_13",
    "ATLAS_7",
    "ATLAS_8",
    "ATLAS_13",
    "LHCb_7",
    "LHCb_8",
    "LHCb_13",
]


def first_number(row: pd.Series, names: list[str], default: float = float("nan")) -> float:
    for name in names:
        if name not in row.index:
            continue
        try:
            value = float(row[name])
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return default


def convert_dataset(name: str, raw_dir: Path, out_dir: Path) -> dict:
    src = raw_dir / f"{name}.csv"
    raw = pd.read_csv(src)
    rows: list[dict] = []
    for i, row in raw.iterrows():
        qt = first_number(row, ["PT", "qT"])
        qlo = first_number(row, ["PT Low", "qT_low"], qt)
        qhi = first_number(row, ["PT High", "qT_high"], qt)
        q = first_number(row, ["QM", "Q"], 91.1876)
        y = first_number(row, ["y"], 0.0)
        ylo = first_number(row, ["y_Low"], y)
        yhi = first_number(row, ["y_High"], y)
        sqrts = first_number(row, ["SqrtS"])
        if not all(math.isfinite(v) for v in (qt, q, sqrts)):
            continue
        tau = q / sqrts
        x1 = first_number(row, ["x1"], tau * math.exp(y))
        x2 = first_number(row, ["x2"], tau * math.exp(-y))
        cs = first_number(row, ["A", "calc_abs_CS", "A_raw_cs"])
        err = first_number(row, ["dA", "dA_norm", "dA_raw_cs"])
        if not (math.isfinite(cs) and math.isfinite(err) and err > 0.0):
            continue
        qm_low = first_number(row, ["QM_Low", "Q_Low"], q)
        qm_high = first_number(row, ["QM_High", "Q_High"], q)
        fidcs = first_number(row, ["fidcs", "fid_CS"])
        observable_type = "absolute_differential_cross_section"
        fit_mode = "absolute_cross_section_with_released_errors"
        if math.isfinite(fidcs) and "NormCS" in row.index:
            observable_type = "normalized_spectrum_with_fidcs_conversion"
            fit_mode = "normalized_spectrum_converted_with_published_fidcs_prefactor"
        rows.append(
            {
                "dataset": name,
                "row_id": f"{name}:{len(rows)}",
                "source_file": str(src),
                "source_row": int(i),
                "qT": qt,
                "qT_low": qlo,
                "qT_high": qhi,
                "qT_bin_width": qhi - qlo if qhi > qlo else np.nan,
                "QM": q,
                "QM_Low": qm_low,
                "QM_High": qm_high,
                "y": y,
                "y_Low": ylo,
                "y_High": yhi,
                "x1": x1,
                "x2": x2,
                "SqrtS": sqrts,
                "BeamE": "",
                "CS": cs,
                "error": err,
                "A": cs,
                "dA": err,
                "PreFactor": first_number(row, ["PreFactor", "prefactor"], 1.0),
                "sysNorm": "",
                "sysNorm_rel": 0.0,
                "sysP2P": "",
                "target": "pp",
                "unit": "pb/GeV",
                "observable_type": observable_type,
                "observable_name": "Z/gamma* transverse-momentum spectrum in pp collisions",
                "unit_status": "diagnostic_raw_table_convention_needs_publication_audit",
                "bin_width_convention": "per_GeV",
                "fit_mode": fit_mode,
                "norm_group": "",
                "covariance_group": f"{name}:covariance",
                "covariance_status": "released_errors_as_diagonal_for_diagnostic; correlated_components_not_yet_profiled",
                "final_state": "lepton_pair",
                "experiment": name.split("_", 1)[0],
                "run_period": name,
                "qT_over_Q": qt / q,
                "fit_region": "diagnostic_pp_collider",
                "diagnostic_fit_candidate": True,
                "review_status": "diagnostic_standardized_from_global_dy_raw",
                "production_ready": False,
                "source_publication": "",
                "source_doi": "",
                "source_arxiv": "",
                "review_notes": "CERN/LHC diagnostic pp table; requires backend and publication-level audit before production inclusion.",
            }
        )
    if not rows:
        raise ValueError(f"No usable rows converted for {name}")
    out_path = out_dir / f"{name}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return {"path": str(out_path), "source": str(src), "rows": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="Data/global_dy_raw")
    parser.add_argument("--out-data-dir", default="Data/v23_cern_lhc_pp_diagnostic_fit_ready")
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "purpose": "Diagnostic fit-ready CERN/LHC pp DY data directory.",
        "status": "diagnostic_not_production",
        "recommended_backend": "v23/backends/bt_internal_css_backend_v22_tevatron.py",
        "datasets": list(args.datasets),
        "files": {},
    }
    for dataset in args.datasets:
        manifest["files"][dataset] = convert_dataset(dataset, raw_dir, out_dir)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
