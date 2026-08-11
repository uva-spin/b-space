#!/usr/bin/env python3
"""Build v23a fixed-target low-Q DY candidate training tables.

This is a *candidate* table builder.  It is intended to prepare the first
global-DY extension after the frozen v22 FNAL-only baseline.

Default v23a datasets:
  E288_200, E288_300, E288_400, E605, E772

Conversion convention
---------------------
The FNAL-style raw files often contain:
  A, dA, PreFactor

where the training cross section used by the v22 code is

  CS    = A  / PreFactor
  error = dA / PreFactor

If an explicit CS/error column already exists, that is preferred.  Otherwise
the script falls back to the A/PreFactor rule, then to A/dA directly.

This script does not decide final covariance/norm handling.  It writes
dataset cards and issue reports for review before any production refit.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd


DATASET_DEFAULTS = ["E288_200", "E288_300", "E288_400", "E605", "E772"]


ALIASES: dict[str, list[str]] = {
    "dataset": ["dataset", "experiment", "exp", "name"],
    "row_id": ["row_id", "rowid", "id"],
    "target": ["target", "hadron", "nucleus"],
    "qT": ["qT", "qt", "QT", "pT", "pt", "q_T", "p_T"],
    "qT_low": ["qT_Low", "qT_low", "qt_low", "pT_low", "pt_low", "qTmin", "ptmin"],
    "qT_high": ["qT_High", "qT_high", "qt_high", "pT_high", "pt_high", "qTmax", "ptmax"],
    "QM": ["QM", "Q", "M", "mass", "mll", "m_ll", "Qmid", "Mmid"],
    "QM_Low": ["QM_Low", "Q_low", "Qmin", "M_low", "Mmin", "mass_low"],
    "QM_High": ["QM_High", "Q_high", "Qmax", "M_high", "Mmax", "mass_high"],
    "y": ["y", "rapidity", "Y", "y_cm", "y_mid", "ycenter"],
    "y_Low": ["y_Low", "y_low", "ymin", "rapidity_low"],
    "y_High": ["y_High", "y_high", "ymax", "rapidity_high"],
    "xF": ["xF", "xf", "x_F"],
    "xF_Low": ["xF_Low", "xf_low", "xFmin", "xfmin"],
    "xF_High": ["xF_High", "xf_high", "xFmax", "xfmax"],
    "x1": ["x1", "x_1"],
    "x2": ["x2", "x_2"],
    "SqrtS": ["SqrtS", "sqrts", "sqrt_s", "sqrtS", "root_s", "Ecm", "ecm"],
    "BeamE": ["BeamE", "beamE", "beam_energy", "Ebeam"],
    "CS": ["CS", "cross_section", "crosssection", "sigma", "xsec"],
    "error": ["error", "err", "unc", "uncertainty", "total_unc", "total_error"],
    "A": ["A", "value", "central_value", "measurement", "observable"],
    "dA": ["dA", "dA_total", "dA_tot", "dA_unc", "stat_tot", "total_dA"],
    "PreFactor": ["PreFactor", "prefactor", "pre_factor", "factor"],
    "sysNorm": ["sysNorm", "norm_unc", "normalization_unc", "lumi_unc"],
    "sysP2P": ["sysP2P", "ptp_unc", "point_to_point", "sys_unc", "syst_unc"],
    "stat": ["stat", "stat_unc", "stat_error"],
    "unit": ["unit", "units", "xsec_unit"],
}


def norm_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def alias_lookup(columns: list[str]) -> dict[str, str]:
    lookup = {norm_name(c): c for c in columns}
    found: dict[str, str] = {}
    for canonical, aliases in ALIASES.items():
        for alias in aliases:
            key = norm_name(alias)
            if key in lookup:
                found[canonical] = lookup[key]
                break
    return found


def to_num(series: pd.Series | None, index: pd.Index) -> pd.Series:
    if series is None:
        return pd.Series(np.nan, index=index, dtype=float)
    if series.dtype.kind in "biufc":
        return pd.to_numeric(series, errors="coerce")
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def to_str(series: pd.Series | None, index: pd.Index, default: str = "") -> pd.Series:
    if series is None:
        return pd.Series(default, index=index, dtype=object)
    return series.astype(str)


def col(frame: pd.DataFrame, aliases: dict[str, str], key: str) -> pd.Series | None:
    name = aliases.get(key)
    if name is None or name not in frame.columns:
        return None
    return frame[name]


def read_table(path: Path) -> pd.DataFrame | None:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        try:
            return pd.read_excel(path)
        except Exception:
            return None
    if suffix not in {".csv", ".tsv", ".txt", ".dat"}:
        return None
    attempts = [
        dict(sep=None, engine="python", comment="#"),
        dict(sep=",", engine="python", comment="#"),
        dict(sep="\t", engine="python", comment="#"),
        dict(sep=r"\s+", engine="python", comment="#"),
    ]
    for kwargs in attempts:
        try:
            frame = pd.read_csv(path, **kwargs)
            if frame.shape[0] > 0 and frame.shape[1] >= 2:
                return frame.dropna(axis=0, how="all").dropna(axis=1, how="all")
        except Exception:
            continue
    return None


def find_dataset_file(raw_dir: Path, dataset: str) -> Path | None:
    patterns = [
        str(raw_dir / f"{dataset}.csv"),
        str(raw_dir / f"{dataset}.*"),
        str(raw_dir / "**" / f"{dataset}.csv"),
        str(raw_dir / "**" / f"{dataset}.*"),
    ]
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(Path(p) for p in glob.glob(pattern, recursive=True))
    matches = [p for p in matches if p.is_file()]
    if not matches:
        return None
    return sorted(matches, key=lambda p: str(p))[0]


def derive_sqrts(beamE: pd.Series) -> pd.Series:
    # Fixed-target proton beam on nucleon at rest: s ~= 2 m_p E_beam + 2 m_p^2.
    mp = 0.9382720813
    return np.sqrt(2.0 * mp * beamE + 2.0 * mp * mp)


def canonicalize_file(path: Path, dataset_name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = read_table(path)
    if frame is None:
        raise RuntimeError(f"Could not read {path}")

    aliases = alias_lookup(list(frame.columns))
    idx = frame.index

    out = pd.DataFrame(index=idx)
    out["dataset"] = to_str(col(frame, aliases, "dataset"), idx, dataset_name)
    out["dataset"] = out["dataset"].replace({"": dataset_name, "nan": dataset_name}).fillna(dataset_name)
    out["target"] = to_str(col(frame, aliases, "target"), idx, "")

    row_id_raw = col(frame, aliases, "row_id")
    if row_id_raw is not None:
        out["row_id"] = row_id_raw.astype(str)
    else:
        out["row_id"] = [f"{dataset_name}:{i}" for i in range(len(frame))]

    for key in [
        "qT", "qT_low", "qT_high",
        "QM", "QM_Low", "QM_High",
        "y", "y_Low", "y_High",
        "xF", "xF_Low", "xF_High",
        "x1", "x2", "SqrtS", "BeamE",
        "A", "dA", "PreFactor", "CS", "error",
        "stat",
    ]:
        out[key] = to_num(col(frame, aliases, key), idx)

    out["sysNorm"] = to_str(col(frame, aliases, "sysNorm"), idx, "")
    out["sysP2P"] = to_str(col(frame, aliases, "sysP2P"), idx, "")
    out["unit"] = to_str(col(frame, aliases, "unit"), idx, "")

    # Fill centers from bin limits where possible.
    for center, lo, hi in [
        ("qT", "qT_low", "qT_high"),
        ("QM", "QM_Low", "QM_High"),
        ("y", "y_Low", "y_High"),
        ("xF", "xF_Low", "xF_High"),
    ]:
        mask = ~np.isfinite(out[center]) & np.isfinite(out[lo]) & np.isfinite(out[hi])
        out.loc[mask, center] = 0.5 * (out.loc[mask, lo] + out.loc[mask, hi])

    # Fill sqrtS from beam energy for fixed-target rows if missing.
    mask_s = ~np.isfinite(out["SqrtS"]) & np.isfinite(out["BeamE"])
    out.loc[mask_s, "SqrtS"] = derive_sqrts(out.loc[mask_s, "BeamE"])

    # Fill x1/x2 from Q,y,sqrtS if missing.
    can_x = np.isfinite(out["QM"]) & np.isfinite(out["SqrtS"]) & np.isfinite(out["y"])
    mask_x = can_x & (~np.isfinite(out["x1"]) | ~np.isfinite(out["x2"]))
    out.loc[mask_x, "x1"] = out.loc[mask_x, "QM"] / out.loc[mask_x, "SqrtS"] * np.exp(out.loc[mask_x, "y"])
    out.loc[mask_x, "x2"] = out.loc[mask_x, "QM"] / out.loc[mask_x, "SqrtS"] * np.exp(-out.loc[mask_x, "y"])

    # Choose CS/error conversion.
    explicit_cs = np.isfinite(out["CS"])
    explicit_error = np.isfinite(out["error"])
    has_prefactor = np.isfinite(out["PreFactor"]) & (out["PreFactor"] != 0)
    has_A = np.isfinite(out["A"])
    has_dA = np.isfinite(out["dA"])

    out["CS_training"] = np.nan
    out["error_training"] = np.nan
    out["conversion_mode"] = ""

    out.loc[explicit_cs, "CS_training"] = out.loc[explicit_cs, "CS"]
    out.loc[explicit_cs, "conversion_mode"] = "explicit_CS"

    mask = ~explicit_cs & has_A & has_prefactor
    out.loc[mask, "CS_training"] = out.loc[mask, "A"] / out.loc[mask, "PreFactor"]
    out.loc[mask, "conversion_mode"] = "A_over_PreFactor"

    mask = ~explicit_cs & ~has_prefactor & has_A
    out.loc[mask, "CS_training"] = out.loc[mask, "A"]
    out.loc[mask, "conversion_mode"] = "A_direct_no_PreFactor"

    out.loc[explicit_error, "error_training"] = out.loc[explicit_error, "error"]
    mask = ~explicit_error & has_dA & has_prefactor
    out.loc[mask, "error_training"] = out.loc[mask, "dA"] / out.loc[mask, "PreFactor"]
    mask = ~explicit_error & ~has_prefactor & has_dA
    out.loc[mask, "error_training"] = out.loc[mask, "dA"]

    # If no total dA/error, combine stat/sysP2P only when numeric sysP2P exists.
    # Keep conservative: do not parse percent strings here as absolute errors.
    out["qT_over_Q"] = out["qT"] / out["QM"]
    out["source_file"] = str(path)
    out["source_row"] = np.arange(len(out), dtype=int)

    # Rename to the columns expected by the existing trainer.
    out["CS"] = out["CS_training"]
    out["error"] = out["error_training"]

    # Retain A/dA/PreFactor for provenance.
    ordered = [
        "dataset", "row_id", "target",
        "qT", "qT_low", "qT_high",
        "QM", "QM_Low", "QM_High",
        "y", "y_Low", "y_High",
        "xF", "xF_Low", "xF_High",
        "x1", "x2", "SqrtS", "BeamE",
        "CS", "error", "A", "dA", "PreFactor",
        "sysNorm", "sysP2P", "unit",
        "qT_over_Q", "conversion_mode",
        "source_file", "source_row",
    ]

    meta = {
        "source_file": str(path),
        "dataset": dataset_name,
        "n_rows": int(len(out)),
        "aliases": aliases,
        "conversion_modes": out["conversion_mode"].value_counts(dropna=False).to_dict(),
    }
    return out[ordered], meta


def issue_rows(frame: pd.DataFrame, qT_max_over_Q: float) -> pd.DataFrame:
    rows = []
    for _, row in frame.iterrows():
        base = {
            "dataset": row["dataset"],
            "row_id": row["row_id"],
            "source_file": row["source_file"],
            "source_row": int(row["source_row"]),
        }
        checks = [
            ("missing_QM", not np.isfinite(row["QM"])),
            ("missing_qT", not np.isfinite(row["qT"])),
            ("missing_y", not np.isfinite(row["y"])),
            ("missing_x1x2", not (np.isfinite(row["x1"]) and np.isfinite(row["x2"]))),
            ("missing_CS", not np.isfinite(row["CS"])),
            ("missing_error", not np.isfinite(row["error"])),
            ("nonpositive_CS", np.isfinite(row["CS"]) and row["CS"] <= 0.0),
            ("nonpositive_error", np.isfinite(row["error"]) and row["error"] <= 0.0),
            ("outside_matched_qT_over_Q", np.isfinite(row["qT_over_Q"]) and row["qT_over_Q"] > qT_max_over_Q),
            ("conversion_A_direct_without_PreFactor", row["conversion_mode"] == "A_direct_no_PreFactor"),
            ("target_missing", str(row.get("target", "")).strip() in {"", "nan", "None"}),
            ("unit_missing", str(row.get("unit", "")).strip() in {"", "nan", "None"}),
        ]
        for issue, is_bad in checks:
            if is_bad:
                rows.append({**base, "issue": issue})
    return pd.DataFrame(rows)


def dataset_summary(frame: pd.DataFrame) -> pd.DataFrame:
    def fmin(s):
        v = s[np.isfinite(s)]
        return float(v.min()) if len(v) else np.nan

    def fmax(s):
        v = s[np.isfinite(s)]
        return float(v.max()) if len(v) else np.nan

    def fmed(s):
        v = s[np.isfinite(s)]
        return float(v.median()) if len(v) else np.nan

    return (
        frame.groupby("dataset", observed=False)
        .agg(
            n_rows=("row_id", "size"),
            n_matched=("qT_over_Q", lambda s: int((np.isfinite(s) & (s <= 0.50)).sum())),
            n_tmd_core=("qT_over_Q", lambda s: int((np.isfinite(s) & (s <= 0.20)).sum())),
            Q_min=("QM", fmin),
            Q_max=("QM", fmax),
            qT_min=("qT", fmin),
            qT_max=("qT", fmax),
            qT_over_Q_median=("qT_over_Q", fmed),
            qT_over_Q_max=("qT_over_Q", fmax),
            x1_min=("x1", fmin),
            x1_max=("x1", fmax),
            x2_min=("x2", fmin),
            x2_max=("x2", fmax),
            CS_min=("CS", fmin),
            CS_max=("CS", fmax),
            rel_error_median=("error", lambda s: np.nan),
        )
        .reset_index()
    )


def add_relative_error(summary: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    rel = frame.copy()
    rel["rel_error"] = rel["error"] / rel["CS"].replace(0.0, np.nan)
    rel_summary = (
        rel.groupby("dataset", observed=False)
        .agg(
            rel_error_median=("rel_error", lambda s: float(np.nanmedian(s))),
            rel_error_max=("rel_error", lambda s: float(np.nanmax(s))),
        )
        .reset_index()
    )
    summary = summary.drop(columns=[c for c in ["rel_error_median"] if c in summary.columns])
    return summary.merge(rel_summary, on="dataset", how="left")


def compare_to_current(data_out: Path, current_data_dir: Path, datasets: list[str]) -> pd.DataFrame:
    rows = []

    for dataset in datasets:
        new_path = data_out / f"{dataset}.csv"
        old_path = current_data_dir / f"{dataset}.csv"

        row = {
            "dataset": dataset,
            "new_file": str(new_path),
            "current_file": str(old_path) if old_path.exists() else "",
            "current_exists": old_path.exists(),
        }

        if not new_path.exists() or not old_path.exists():
            rows.append(row)
            continue

        new = pd.read_csv(new_path)
        old = pd.read_csv(old_path)

        # Canonicalize old CS/error if necessary.
        aliases = alias_lookup(list(old.columns))
        idx = old.index
        old_work = pd.DataFrame(index=idx)
        for key in ["qT", "QM", "y", "xF", "CS", "error", "A", "dA", "PreFactor"]:
            old_work[key] = to_num(col(old, aliases, key), idx)
        if old_work["CS"].isna().all() and np.isfinite(old_work["A"]).any() and np.isfinite(old_work["PreFactor"]).any():
            old_work["CS"] = old_work["A"] / old_work["PreFactor"].replace(0.0, np.nan)
        if old_work["error"].isna().all() and np.isfinite(old_work["dA"]).any() and np.isfinite(old_work["PreFactor"]).any():
            old_work["error"] = old_work["dA"] / old_work["PreFactor"].replace(0.0, np.nan)

        def key_frame(df: pd.DataFrame) -> pd.DataFrame:
            out = df.copy()
            for c in ["qT", "QM", "y", "xF"]:
                if c not in out.columns:
                    out[c] = np.nan
                out[f"{c}_key"] = pd.to_numeric(out[c], errors="coerce").round(8)
            return out

        nk = key_frame(new)
        ok = key_frame(old_work)

        merged = nk.merge(
            ok,
            on=["qT_key", "QM_key", "y_key", "xF_key"],
            suffixes=("_new", "_old"),
            how="inner",
        )

        row.update({
            "n_new": int(len(new)),
            "n_current": int(len(old)),
            "n_common_key": int(len(merged)),
        })

        if len(merged):
            row["max_rel_delta_CS_common"] = float(
                np.nanmax(
                    np.abs(merged["CS_new"] - merged["CS_old"])
                    / np.maximum(np.abs(merged["CS_old"]), 1.0e-300)
                )
            )
            row["max_rel_delta_error_common"] = float(
                np.nanmax(
                    np.abs(merged["error_new"] - merged["error_old"])
                    / np.maximum(np.abs(merged["error_old"]), 1.0e-300)
                )
            )
        rows.append(row)

    return pd.DataFrame(rows)


def write_dataset_cards(frame: pd.DataFrame, meta: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_by_dataset = {m["dataset"]: m for m in meta}

    for dataset, group in frame.groupby("dataset", observed=False):
        card = {
            "dataset": dataset,
            "source_file": meta_by_dataset.get(dataset, {}).get("source_file"),
            "include_in_v23a_first_refit": dataset in DATASET_DEFAULTS,
            "process_class": "fixed_target_lowQ_DY",
            "target_review_required": True,
            "unit_review_required": True,
            "normalization_review_required": True,
            "conversion_modes": group["conversion_mode"].value_counts(dropna=False).to_dict(),
            "n_rows_total": int(len(group)),
            "n_rows_matched_qT_over_Q_le_0p5": int((group["qT_over_Q"] <= 0.5).sum()),
            "n_rows_tmd_core_qT_over_Q_le_0p2": int((group["qT_over_Q"] <= 0.2).sum()),
            "columns_used": {
                "CS": "explicit CS if present, otherwise A/PreFactor",
                "error": "explicit error if present, otherwise dA/PreFactor",
            },
            "manual_checks_before_production": [
                "confirm target material/isospin/nuclear handling",
                "confirm A/PreFactor convention against original paper/table",
                "confirm dA/PreFactor error convention",
                "confirm sysNorm/sysP2P treatment and normalization priors",
                "confirm qT/Q <= 0.5 matched and qT/Q <= 0.2 TMD-core cuts",
            ],
        }
        (out_dir / f"{dataset}.json").write_text(json.dumps(card, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="Data/global_dy_raw")
    parser.add_argument("--datasets", nargs="+", default=DATASET_DEFAULTS)
    parser.add_argument("--current-data-dir", default="Data")
    parser.add_argument("--data-out", default="Data/v23a_fixed_target_lowQ_candidate")
    parser.add_argument("--out", default="v23/outputs/v23a_fixed_target_lowQ_table")
    parser.add_argument("--qT-max-over-Q", type=float, default=0.5)
    parser.add_argument("--tmd-qT-max-over-Q", type=float, default=0.2)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out = Path(args.out)
    data_out = Path(args.data_out)
    out.mkdir(parents=True, exist_ok=True)
    data_out.mkdir(parents=True, exist_ok=True)

    pieces = []
    meta = []
    missing = []

    for dataset in args.datasets:
        path = find_dataset_file(raw_dir, dataset)
        if path is None:
            missing.append(dataset)
            continue
        table, m = canonicalize_file(path, dataset)
        pieces.append(table)
        meta.append(m)

    if not pieces:
        raise SystemExit("No requested datasets were found/readable.")

    all_rows = pd.concat(pieces, ignore_index=True)
    matched = all_rows[
        np.isfinite(all_rows["qT_over_Q"])
        & (all_rows["qT_over_Q"] <= float(args.qT_max_over_Q))
    ].copy()
    tmd_core = all_rows[
        np.isfinite(all_rows["qT_over_Q"])
        & (all_rows["qT_over_Q"] <= float(args.tmd_qT_max_over_Q))
    ].copy()

    # Write per-dataset matched files for direct trainer use.
    for dataset, group in matched.groupby("dataset", observed=False):
        group.to_csv(data_out / f"{dataset}.csv", index=False)

    all_rows.to_csv(out / "v23a_fixed_target_all_rows.csv", index=False)
    matched.to_csv(out / "v23a_fixed_target_matched_qToQ_le_0p5.csv", index=False)
    tmd_core.to_csv(out / "v23a_fixed_target_tmd_core_qToQ_le_0p2.csv", index=False)

    summary = add_relative_error(dataset_summary(all_rows), all_rows)
    matched_summary = add_relative_error(dataset_summary(matched), matched)
    summary.to_csv(out / "dataset_summary_all_rows.csv", index=False)
    matched_summary.to_csv(out / "dataset_summary_matched.csv", index=False)

    issues = issue_rows(all_rows, qT_max_over_Q=float(args.qT_max_over_Q))
    issues.to_csv(out / "issues_all_rows.csv", index=False)

    comparison = compare_to_current(data_out, Path(args.current_data_dir), args.datasets)
    comparison.to_csv(out / "comparison_to_current_Data.csv", index=False)

    write_dataset_cards(all_rows, meta, out / "dataset_cards")

    manifest = {
        "raw_dir": str(raw_dir),
        "datasets_requested": args.datasets,
        "datasets_missing": missing,
        "data_out": str(data_out),
        "n_rows_all": int(len(all_rows)),
        "n_rows_matched_qT_over_Q_le_0p5": int(len(matched)),
        "n_rows_tmd_core_qT_over_Q_le_0p2": int(len(tmd_core)),
        "conversion_summary": all_rows.groupby(["dataset", "conversion_mode"], observed=False).size().rename("n").reset_index().to_dict(orient="records"),
        "outputs": {
            "trainer_data_dir": str(data_out),
            "all_rows": str(out / "v23a_fixed_target_all_rows.csv"),
            "matched": str(out / "v23a_fixed_target_matched_qToQ_le_0p5.csv"),
            "tmd_core": str(out / "v23a_fixed_target_tmd_core_qToQ_le_0p2.csv"),
            "summary_all": str(out / "dataset_summary_all_rows.csv"),
            "summary_matched": str(out / "dataset_summary_matched.csv"),
            "issues": str(out / "issues_all_rows.csv"),
            "comparison_to_current_Data": str(out / "comparison_to_current_Data.csv"),
            "dataset_cards": str(out / "dataset_cards"),
        },
        "next_steps": [
            "Inspect comparison_to_current_Data.csv; v23a should not silently alter E288/E605 conventions.",
            "Inspect dataset_cards/E772.json for target/unit/norm handling.",
            "Only after review, run a v23a check-only backend cache build with datasets E288_200 E288_300 E288_400 E605 E772.",
        ],
    }

    (out / "v23a_fixed_target_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print("\n=== v23a fixed-target low-Q candidate table ===")
    print(json.dumps(manifest, indent=2))

    print("\n=== Matched dataset summary ===")
    print(matched_summary.to_string(index=False))

    print("\n=== Comparison to current Data ===")
    print(comparison.to_string(index=False))

    print("\n=== Issue counts ===")
    if issues.empty:
        print("(none)")
    else:
        print(issues.groupby("issue").size().sort_values(ascending=False).rename("n").reset_index().to_string(index=False))

    print("\nwrote:", out)
    print("trainer data dir:", data_out)


if __name__ == "__main__":
    main()
