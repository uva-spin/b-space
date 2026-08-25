#!/usr/bin/env python3
"""v23 global-DY data intake audit.

Purpose
-------
Inventory newly collected DY data files before changing the fit.

This script does *not* decide physics cuts or convert units.  It creates a
canonical candidate table and a provenance/quality report so we can decide
what needs manual normalization, bin-width, target, or covariance handling.

Supported directly:
  .csv, .tsv, .txt, .dat, .xlsx, .xls

YAML/HEPData files are inventoried as unsupported-by-parser for now unless
they are already converted to tables.  That is intentional: HEPData YAML tables
need dataset-specific handling of independent/dependent variables and errors.

Outputs
-------
  file_inventory.csv
  canonical_candidate_rows.csv
  dataset_summary.csv
  issues.csv
  current_baseline_summary.csv
  global_dy_intake_manifest.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd


ALIASES: dict[str, list[str]] = {
    "dataset": [
        "dataset", "experiment", "exp", "name", "table", "source",
    ],
    "process": [
        "process", "reaction", "channel",
    ],
    "sqrtS": [
        "sqrts", "sqrt_s", "sqrt{s}", "roots", "root_s", "sqrtS", "SqrtS",
        "cms_energy", "ecm", "Ecm", "energy_cm",
    ],
    "beamE": [
        "BeamE", "beam_e", "beamenergy", "beam_energy", "Ebeam",
    ],
    "target": [
        "target", "hadron", "nucleus", "beam_target",
    ],
    "Q": [
        "Q", "QM", "M", "mass", "m", "mll", "m_ll", "Qmid", "mass_mid",
        "Q_center", "M_center", "Qmean", "Mmean",
    ],
    "Q_low": [
        "QM_Low", "Q_low", "Qmin", "M_low", "Mmin", "mass_low",
    ],
    "Q_high": [
        "QM_High", "Q_high", "Qmax", "M_high", "Mmax", "mass_high",
    ],
    "qT": [
        "qT", "qt", "pT", "pt", "p_T", "q_T", "QT", "PT", "qT_mid",
        "pt_mid", "qT_center", "pT_center",
    ],
    "qT_low": [
        "qT_Low", "qT_low", "qt_low", "pT_low", "pt_low", "qTmin", "ptmin",
    ],
    "qT_high": [
        "qT_High", "qT_high", "qt_high", "pT_high", "pt_high", "qTmax", "ptmax",
    ],
    "y": [
        "y", "rapidity", "Y", "y_cm", "ycenter", "y_mid", "y_center",
    ],
    "y_low": [
        "y_Low", "y_low", "ymin", "rapidity_low",
    ],
    "y_high": [
        "y_High", "y_high", "ymax", "rapidity_high",
    ],
    "eta": [
        "eta", "pseudorapidity", "abs_eta", "eta_center",
    ],
    "xF": [
        "xF", "xf", "x_F", "xF_mid", "xf_center",
    ],
    "x1": [
        "x1", "x_1",
    ],
    "x2": [
        "x2", "x_2",
    ],
    "cross_section": [
        "CS", "sigma", "xsec", "cross_section", "crosssection",
        "dsigma", "d_sigma", "value", "central_value", "measurement",
        "d2sigma", "d3sigma", "differential_cross_section",
    ],
    "stat_unc": [
        "stat", "stat_unc", "stat_error", "staterr", "statistical",
    ],
    "syst_unc": [
        "syst", "sys", "syst_unc", "sys_unc", "syst_error", "syserr",
        "systematic",
    ],
    "total_unc": [
        "error", "err", "unc", "uncertainty", "total_unc", "total_error",
        "combined_unc", "dCS", "dA",
    ],
    "norm_unc": [
        "sysNorm", "norm_unc", "normalization_unc", "lumi_unc", "luminosity_unc",
    ],
    "unit": [
        "unit", "units", "xsec_unit", "cross_section_unit",
    ],
}


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def alias_lookup(columns: list[str]) -> dict[str, str]:
    norm_to_original = {normalize_name(c): c for c in columns}
    found: dict[str, str] = {}
    for canonical, aliases in ALIASES.items():
        for alias in aliases:
            key = normalize_name(alias)
            if key in norm_to_original:
                found[canonical] = norm_to_original[key]
                break
    return found


def as_numeric(series: pd.Series) -> pd.Series:
    if series.dtype.kind in "biufc":
        return pd.to_numeric(series, errors="coerce")
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def try_read_table(path: Path) -> tuple[pd.DataFrame | None, str, str]:
    suffix = path.suffix.lower()

    if suffix in {".yaml", ".yml", ".json"}:
        return None, "unsupported_structured", "YAML/JSON require dataset-specific HEPData parser"

    if suffix in {".xlsx", ".xls"}:
        try:
            frame = pd.read_excel(path)
            return frame, "excel", ""
        except Exception as exc:
            return None, "read_error", repr(exc)

    if suffix not in {".csv", ".tsv", ".txt", ".dat"}:
        return None, "unsupported_extension", f"unsupported extension {suffix}"

    attempts: list[tuple[str, dict[str, Any]]] = [
        ("csv_auto", dict(sep=None, engine="python", comment="#")),
        ("csv_comma", dict(sep=",", engine="python", comment="#")),
        ("tsv", dict(sep="\t", engine="python", comment="#")),
        ("whitespace", dict(sep=r"\s+", engine="python", comment="#")),
    ]

    for label, kwargs in attempts:
        try:
            frame = pd.read_csv(path, **kwargs)
            if frame.shape[1] >= 2 and frame.shape[0] > 0:
                return frame, label, ""
        except Exception:
            continue

    return None, "read_error", "all table read attempts failed"


def min_max_numeric(frame: pd.DataFrame, col: str | None) -> tuple[float | None, float | None, int]:
    if col is None or col not in frame.columns:
        return None, None, 0
    values = as_numeric(frame[col])
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return None, None, 0
    return float(finite.min()), float(finite.max()), int(len(finite))


def value_or_nan(frame: pd.DataFrame, col: str | None) -> pd.Series:
    if col is None or col not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return as_numeric(frame[col])


def string_value_or_empty(frame: pd.DataFrame, col: str | None, default: str = "") -> pd.Series:
    if col is None or col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=object)
    return frame[col].astype(str)


def dataset_name_from_path(path: Path) -> str:
    stem = path.stem
    stem = re.sub(r"[^A-Za-z0-9_.+-]+", "_", stem).strip("_")
    return stem or "unknown_dataset"


def build_canonical_rows(path: Path, frame: pd.DataFrame, aliases: dict[str, str]) -> pd.DataFrame:
    n = len(frame)
    out = pd.DataFrame({
        "source_file": str(path),
        "source_row": np.arange(n, dtype=int),
    })

    dataset_col = aliases.get("dataset")
    out["dataset"] = string_value_or_empty(frame, dataset_col, dataset_name_from_path(path))
    out["process"] = string_value_or_empty(frame, aliases.get("process"), "")
    out["target"] = string_value_or_empty(frame, aliases.get("target"), "")

    for canonical in [
        "sqrtS", "beamE", "Q", "Q_low", "Q_high", "qT", "qT_low", "qT_high",
        "y", "y_low", "y_high", "eta", "xF", "x1", "x2",
        "cross_section", "stat_unc", "syst_unc", "total_unc", "norm_unc",
    ]:
        out[canonical] = value_or_nan(frame, aliases.get(canonical))

    out["unit"] = string_value_or_empty(frame, aliases.get("unit"), "")

    # Fill bin centers from low/high if no explicit center exists.
    for center, lo, hi in [
        ("Q", "Q_low", "Q_high"),
        ("qT", "qT_low", "qT_high"),
        ("y", "y_low", "y_high"),
    ]:
        missing = ~np.isfinite(out[center])
        can_fill = np.isfinite(out[lo]) & np.isfinite(out[hi])
        out.loc[missing & can_fill, center] = 0.5 * (
            out.loc[missing & can_fill, lo] + out.loc[missing & can_fill, hi]
        )

    # Compute qT/Q and x1/x2 when enough information is present.
    out["qT_over_Q"] = out["qT"] / out["Q"]

    missing_x = ~np.isfinite(out["x1"]) | ~np.isfinite(out["x2"])
    can_x = np.isfinite(out["Q"]) & np.isfinite(out["sqrtS"]) & np.isfinite(out["y"])
    out.loc[missing_x & can_x, "x1"] = (
        out.loc[missing_x & can_x, "Q"]
        / out.loc[missing_x & can_x, "sqrtS"]
        * np.exp(out.loc[missing_x & can_x, "y"])
    )
    out.loc[missing_x & can_x, "x2"] = (
        out.loc[missing_x & can_x, "Q"]
        / out.loc[missing_x & can_x, "sqrtS"]
        * np.exp(-out.loc[missing_x & can_x, "y"])
    )

    return out


def summarize_canonical(canon: pd.DataFrame) -> pd.DataFrame:
    if canon.empty:
        return pd.DataFrame()

    def finite_min(s: pd.Series):
        v = s[np.isfinite(s)]
        return float(v.min()) if len(v) else np.nan

    def finite_max(s: pd.Series):
        v = s[np.isfinite(s)]
        return float(v.max()) if len(v) else np.nan

    def finite_median(s: pd.Series):
        v = s[np.isfinite(s)]
        return float(v.median()) if len(v) else np.nan

    return (
        canon.groupby("dataset", dropna=False)
        .agg(
            n_rows=("source_row", "size"),
            n_files=("source_file", lambda s: int(pd.Series(s).nunique())),
            Q_min=("Q", finite_min),
            Q_max=("Q", finite_max),
            qT_min=("qT", finite_min),
            qT_max=("qT", finite_max),
            qT_over_Q_min=("qT_over_Q", finite_min),
            qT_over_Q_median=("qT_over_Q", finite_median),
            qT_over_Q_max=("qT_over_Q", finite_max),
            y_min=("y", finite_min),
            y_max=("y", finite_max),
            x1_min=("x1", finite_min),
            x1_max=("x1", finite_max),
            x2_min=("x2", finite_min),
            x2_max=("x2", finite_max),
            n_missing_Q=("Q", lambda s: int((~np.isfinite(s)).sum())),
            n_missing_qT=("qT", lambda s: int((~np.isfinite(s)).sum())),
            n_missing_cross_section=("cross_section", lambda s: int((~np.isfinite(s)).sum())),
            n_tmd_core=("qT_over_Q", lambda s: int((np.isfinite(s) & (s <= 0.20)).sum())),
            n_matched_candidate=("qT_over_Q", lambda s: int((np.isfinite(s) & (s <= 0.50)).sum())),
        )
        .reset_index()
        .sort_values(["dataset"])
    )


def collect_issues(canon: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx, row in canon.iterrows():
        base = {
            "source_file": row["source_file"],
            "source_row": int(row["source_row"]),
            "dataset": row["dataset"],
        }

        checks = [
            ("missing_Q", not np.isfinite(row["Q"])),
            ("missing_qT", not np.isfinite(row["qT"])),
            ("missing_cross_section", not np.isfinite(row["cross_section"])),
            ("nonpositive_cross_section", np.isfinite(row["cross_section"]) and row["cross_section"] <= 0),
            ("missing_uncertainty", not any(np.isfinite(row[c]) for c in ["total_unc", "stat_unc", "syst_unc"])),
            ("missing_energy_or_rapidity_for_x1x2", not (np.isfinite(row["x1"]) and np.isfinite(row["x2"]))),
            ("outside_matched_qT_over_Q_gt_0p5", np.isfinite(row["qT_over_Q"]) and row["qT_over_Q"] > 0.50),
            ("unit_missing", str(row.get("unit", "")).strip() == ""),
        ]

        for issue, bad in checks:
            if bad:
                rows.append({**base, "issue": issue})

    return pd.DataFrame(rows)


def inventory_file(path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    frame, reader, read_note = try_read_table(path)
    record: dict[str, Any] = {
        "source_file": str(path),
        "file_name": path.name,
        "suffix": path.suffix.lower(),
        "reader": reader,
        "read_note": read_note,
        "readable": frame is not None,
        "n_rows": 0,
        "n_columns": 0,
        "columns": "",
        "inferred_columns_json": "{}",
    }

    if frame is None:
        return record, pd.DataFrame()

    # Drop fully empty columns and rows.
    frame = frame.dropna(axis=0, how="all").dropna(axis=1, how="all")
    aliases = alias_lookup(list(frame.columns))

    record.update({
        "n_rows": int(len(frame)),
        "n_columns": int(len(frame.columns)),
        "columns": "|".join(map(str, frame.columns)),
        "inferred_columns_json": json.dumps(aliases, sort_keys=True),
    })

    for canonical in ["Q", "qT", "y", "xF", "x1", "x2", "sqrtS", "cross_section"]:
        lo, hi, nfinite = min_max_numeric(frame, aliases.get(canonical))
        record[f"{canonical}_min"] = lo
        record[f"{canonical}_max"] = hi
        record[f"{canonical}_nfinite"] = nfinite

    canonical_rows = build_canonical_rows(path, frame, aliases)

    return record, canonical_rows


def expand_globs(patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        matches = [Path(p) for p in glob.glob(pattern, recursive=True)]
        files.extend(matches)

    # Deduplicate and keep files only.
    unique = []
    seen = set()
    for path in files:
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        if path.is_file():
            unique.append(path)
    return sorted(unique, key=lambda p: str(p))


def baseline_current_summary(current_data_dir: Path) -> pd.DataFrame:
    if not current_data_dir.exists():
        return pd.DataFrame()

    files = []
    for name in ["E288_200.csv", "E288_300.csv", "E288_400.csv", "E605.csv"]:
        p = current_data_dir / name
        if p.exists():
            files.append(p)

    pieces = []
    for path in files:
        record, canon = inventory_file(path)
        if not canon.empty:
            pieces.append(canon)

    if not pieces:
        return pd.DataFrame()

    return summarize_canonical(pd.concat(pieces, ignore_index=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-glob",
        nargs="+",
        required=True,
        help="One or more glob patterns, e.g. 'Data/global_dy_raw/**/*.csv'",
    )
    parser.add_argument(
        "--current-data-dir",
        default="Data",
        help="Existing FNAL-only data directory for comparison.",
    )
    parser.add_argument(
        "--out",
        default="v23/outputs/global_dy_data_intake",
    )
    args = parser.parse_args()

    files = expand_globs(args.input_glob)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if not files:
        raise SystemExit(
            "No files matched --input-glob. Put new DY files in a directory and rerun."
        )

    inventory_records = []
    canonical_pieces = []

    for path in files:
        record, canon = inventory_file(path)
        inventory_records.append(record)
        if not canon.empty:
            canonical_pieces.append(canon)

    inventory = pd.DataFrame(inventory_records)
    canonical = (
        pd.concat(canonical_pieces, ignore_index=True)
        if canonical_pieces
        else pd.DataFrame()
    )

    summary = summarize_canonical(canonical)
    issues = collect_issues(canonical) if not canonical.empty else pd.DataFrame()
    baseline = baseline_current_summary(Path(args.current_data_dir))

    inventory.to_csv(out / "file_inventory.csv", index=False)
    canonical.to_csv(out / "canonical_candidate_rows.csv", index=False)
    summary.to_csv(out / "dataset_summary.csv", index=False)
    issues.to_csv(out / "issues.csv", index=False)
    baseline.to_csv(out / "current_baseline_summary.csv", index=False)

    manifest = {
        "input_globs": args.input_glob,
        "n_files_matched": len(files),
        "n_files_readable": int(inventory["readable"].sum()) if not inventory.empty else 0,
        "n_canonical_rows": int(len(canonical)),
        "n_datasets_inferred": int(summary["dataset"].nunique()) if not summary.empty else 0,
        "n_issues": int(len(issues)),
        "outputs": {
            "file_inventory": str(out / "file_inventory.csv"),
            "canonical_candidate_rows": str(out / "canonical_candidate_rows.csv"),
            "dataset_summary": str(out / "dataset_summary.csv"),
            "issues": str(out / "issues.csv"),
            "current_baseline_summary": str(out / "current_baseline_summary.csv"),
        },
        "notes": [
            "This is an intake/provenance audit, not a final training table.",
            "Unit conversion, bin-width convention, covariance treatment, and normalization priors still require dataset-level review.",
            "Rows with qT/Q <= 0.20 are TMD-core candidates; rows with qT/Q <= 0.50 are matched-fit candidates by the current convention.",
            "YAML/HEPData files are currently inventoried but not automatically canonicalized.",
        ],
    }
    (out / "global_dy_intake_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print("\n=== v23 global-DY intake summary ===")
    print(json.dumps(manifest, indent=2))

    print("\n=== Dataset summary preview ===")
    if summary.empty:
        print("(no canonical rows)")
    else:
        print(summary.head(80).to_string(index=False))

    print("\n=== Top issue counts ===")
    if issues.empty:
        print("(no issues)")
    else:
        print(
            issues.groupby("issue")
            .size()
            .sort_values(ascending=False)
            .rename("n")
            .reset_index()
            .to_string(index=False)
        )

    print("\nwrote:", out)


if __name__ == "__main__":
    main()
