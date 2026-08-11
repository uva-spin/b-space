#!/usr/bin/env python3
"""Build v23a fixed-target low-Q DY table from verified trainer-format data.

Use this when raw global_dy_raw files contain A/dA but not the PreFactor
needed to convert into the trainer's CS/error convention.  For the FNAL-style
fixed-target files, the verified Data/*.csv files already contain CS/error and
usually A/dA/PreFactor provenance.

Default v23a datasets:
  E288_200, E288_300, E288_400, E605, E772

The output data directory is intended for the v23a check-only backend cache
build and later central refit.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DATASETS = ["E288_200", "E288_300", "E288_400", "E605", "E772"]


ALIASES = {
    "qT": ["qT", "qt", "QT", "pT", "pt"],
    "QM": ["QM", "Q", "M", "mass", "mll"],
    "y": ["y", "Y", "rapidity"],
    "xF": ["xF", "xf", "x_F"],
    "x1": ["x1", "x_1"],
    "x2": ["x2", "x_2"],
    "SqrtS": ["SqrtS", "sqrts", "sqrt_s", "sqrtS"],
    "BeamE": ["BeamE", "beamE", "beam_energy", "Ebeam"],
    "CS": ["CS", "cross_section", "crosssection", "sigma", "xsec"],
    "error": ["error", "err", "unc", "uncertainty", "total_unc"],
    "A": ["A", "value", "central_value"],
    "dA": ["dA", "dA_total", "total_dA"],
    "PreFactor": ["PreFactor", "prefactor", "pre_factor", "factor"],
    "sysNorm": ["sysNorm", "norm_unc"],
    "sysP2P": ["sysP2P", "ptp_unc"],
    "row_id": ["row_id", "rowid", "id"],
}


def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def alias_lookup(columns: list[str]) -> dict[str, str]:
    by_norm = {norm(c): c for c in columns}
    found = {}
    for key, aliases in ALIASES.items():
        for a in aliases:
            if norm(a) in by_norm:
                found[key] = by_norm[norm(a)]
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


def col(frame: pd.DataFrame, aliases: dict[str, str], key: str) -> pd.Series | None:
    name = aliases.get(key)
    if name is None or name not in frame.columns:
        return None
    return frame[name]


def relerr(a: pd.Series, b: pd.Series) -> pd.Series:
    return np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-300)


def finite_count(s: pd.Series) -> int:
    return int(np.isfinite(s).sum())


def canonicalize_verified(path: Path, dataset: str) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(path)
    aliases = alias_lookup(list(raw.columns))
    idx = raw.index

    work = raw.copy()

    # Ensure core columns exist under the trainer names.
    for key in ["qT", "QM", "y", "xF", "x1", "x2", "SqrtS", "BeamE", "CS", "error", "A", "dA", "PreFactor"]:
        if key not in work.columns:
            work[key] = to_num(col(raw, aliases, key), idx)
        else:
            work[key] = to_num(work[key], idx)

    if "row_id" not in work.columns:
        row_id_src = col(raw, aliases, "row_id")
        if row_id_src is not None:
            work["row_id"] = row_id_src.astype(str)
        else:
            work["row_id"] = [f"{dataset}:{i}" for i in range(len(work))]

    if "dataset" not in work.columns:
        work["dataset"] = dataset
    else:
        work["dataset"] = work["dataset"].fillna(dataset).astype(str)
        work.loc[work["dataset"].isin(["", "nan", "None"]), "dataset"] = dataset

    for text_col in ["sysNorm", "sysP2P", "target", "unit"]:
        if text_col not in work.columns:
            src = col(raw, aliases, text_col)
            work[text_col] = src.astype(str) if src is not None else ""
        else:
            work[text_col] = work[text_col].astype(str)

    # If CS/error are missing but A/dA/PreFactor are present, fill them.
    has_pref = np.isfinite(work["PreFactor"]) & (work["PreFactor"] != 0)
    missing_cs = ~np.isfinite(work["CS"])
    missing_err = ~np.isfinite(work["error"])

    fill_cs = missing_cs & np.isfinite(work["A"]) & has_pref
    fill_err = missing_err & np.isfinite(work["dA"]) & has_pref

    work.loc[fill_cs, "CS"] = work.loc[fill_cs, "A"] / work.loc[fill_cs, "PreFactor"]
    work.loc[fill_err, "error"] = work.loc[fill_err, "dA"] / work.loc[fill_err, "PreFactor"]

    # Provenance consistency checks when A/dA/PreFactor exist.
    work["CS_from_A_over_PreFactor"] = np.nan
    work["error_from_dA_over_PreFactor"] = np.nan
    work.loc[has_pref & np.isfinite(work["A"]), "CS_from_A_over_PreFactor"] = (
        work.loc[has_pref & np.isfinite(work["A"]), "A"]
        / work.loc[has_pref & np.isfinite(work["A"]), "PreFactor"]
    )
    work.loc[has_pref & np.isfinite(work["dA"]), "error_from_dA_over_PreFactor"] = (
        work.loc[has_pref & np.isfinite(work["dA"]), "dA"]
        / work.loc[has_pref & np.isfinite(work["dA"]), "PreFactor"]
    )

    work["relerr_CS_vs_A_over_PreFactor"] = relerr(work["CS"], work["CS_from_A_over_PreFactor"])
    work["relerr_error_vs_dA_over_PreFactor"] = relerr(work["error"], work["error_from_dA_over_PreFactor"])

    work["qT_over_Q"] = work["qT"] / work["QM"]
    work["source_file"] = str(path)
    work["source_row"] = np.arange(len(work), dtype=int)

    # Keep original useful columns plus provenance.
    preferred = [
        "dataset", "row_id",
        "qT", "QM", "y", "xF", "x1", "x2", "SqrtS", "BeamE",
        "CS", "error", "A", "dA", "PreFactor",
        "sysNorm", "sysP2P", "target", "unit",
        "qT_over_Q",
        "CS_from_A_over_PreFactor", "error_from_dA_over_PreFactor",
        "relerr_CS_vs_A_over_PreFactor", "relerr_error_vs_dA_over_PreFactor",
        "source_file", "source_row",
    ]
    cols = [c for c in preferred if c in work.columns] + [c for c in work.columns if c not in preferred]

    meta = {
        "dataset": dataset,
        "source_file": str(path),
        "n_rows": int(len(work)),
        "aliases": aliases,
        "n_CS": finite_count(work["CS"]),
        "n_error": finite_count(work["error"]),
        "n_PreFactor": finite_count(work["PreFactor"]),
        "max_relerr_CS_vs_A_over_PreFactor": (
            float(np.nanmax(work["relerr_CS_vs_A_over_PreFactor"]))
            if np.isfinite(work["relerr_CS_vs_A_over_PreFactor"]).any()
            else None
        ),
        "max_relerr_error_vs_dA_over_PreFactor": (
            float(np.nanmax(work["relerr_error_vs_dA_over_PreFactor"]))
            if np.isfinite(work["relerr_error_vs_dA_over_PreFactor"]).any()
            else None
        ),
    }
    return work[cols], meta


def issues_for(frame: pd.DataFrame, qT_max_over_Q: float) -> pd.DataFrame:
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
            ("nonpositive_CS", np.isfinite(row["CS"]) and row["CS"] <= 0),
            ("nonpositive_error", np.isfinite(row["error"]) and row["error"] <= 0),
            ("outside_matched_qT_over_Q", np.isfinite(row["qT_over_Q"]) and row["qT_over_Q"] > qT_max_over_Q),
        ]
        if np.isfinite(row.get("relerr_CS_vs_A_over_PreFactor", np.nan)):
            checks.append(("CS_prefactor_inconsistency", row["relerr_CS_vs_A_over_PreFactor"] > 1e-8))
        if np.isfinite(row.get("relerr_error_vs_dA_over_PreFactor", np.nan)):
            checks.append(("error_prefactor_inconsistency", row["relerr_error_vs_dA_over_PreFactor"] > 1e-8))
        for issue, bad in checks:
            if bad:
                rows.append({**base, "issue": issue})
    return pd.DataFrame(rows)


def summary(frame: pd.DataFrame) -> pd.DataFrame:
    def fmin(s):
        v = pd.to_numeric(s, errors="coerce")
        v = v[np.isfinite(v)]
        return float(v.min()) if len(v) else np.nan
    def fmax(s):
        v = pd.to_numeric(s, errors="coerce")
        v = v[np.isfinite(v)]
        return float(v.max()) if len(v) else np.nan
    def fmed(s):
        v = pd.to_numeric(s, errors="coerce")
        v = v[np.isfinite(v)]
        return float(v.median()) if len(v) else np.nan
    tmp = frame.copy()
    tmp["rel_error"] = tmp["error"] / tmp["CS"].replace(0, np.nan)
    return (
        tmp.groupby("dataset", observed=False)
        .agg(
            n_rows=("row_id", "size"),
            n_tmd_core=("qT_over_Q", lambda s: int((np.isfinite(s) & (s <= 0.2)).sum())),
            n_matched=("qT_over_Q", lambda s: int((np.isfinite(s) & (s <= 0.5)).sum())),
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
            rel_error_median=("rel_error", fmed),
            rel_error_max=("rel_error", fmax),
        )
        .reset_index()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-data-dir", default="Data")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--out-data-dir", default="Data/v23a_fixed_target_lowQ_verified")
    parser.add_argument("--out", default="v23/outputs/v23a_fixed_target_lowQ_verified_table")
    parser.add_argument("--qT-max-over-Q", type=float, default=0.5)
    parser.add_argument("--tmd-qT-max-over-Q", type=float, default=0.2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source = Path(args.source_data_dir)
    out_data = Path(args.out_data_dir)
    out = Path(args.out)

    if out_data.exists() and any(out_data.iterdir()) and not args.force:
        raise SystemExit(f"Refusing to overwrite nonempty {out_data}; use --force")

    out_data.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    pieces = []
    metas = []
    missing = []

    for dataset in args.datasets:
        path = source / f"{dataset}.csv"
        if not path.exists():
            missing.append(dataset)
            continue
        table, meta = canonicalize_verified(path, dataset)
        pieces.append(table)
        metas.append(meta)

    if not pieces:
        raise SystemExit("No requested datasets found.")

    all_rows = pd.concat(pieces, ignore_index=True)
    matched = all_rows[
        np.isfinite(all_rows["qT_over_Q"])
        & (all_rows["qT_over_Q"] <= float(args.qT_max_over_Q))
    ].copy()
    tmd = all_rows[
        np.isfinite(all_rows["qT_over_Q"])
        & (all_rows["qT_over_Q"] <= float(args.tmd_qT_max_over_Q))
    ].copy()

    # Write per-dataset matched files.
    for dataset, group in matched.groupby("dataset", observed=False):
        group.to_csv(out_data / f"{dataset}.csv", index=False)

    all_rows.to_csv(out / "v23a_verified_all_rows.csv", index=False)
    matched.to_csv(out / "v23a_verified_matched_qToQ_le_0p5.csv", index=False)
    tmd.to_csv(out / "v23a_verified_tmd_core_qToQ_le_0p2.csv", index=False)

    summary(all_rows).to_csv(out / "dataset_summary_all_rows.csv", index=False)
    summary(matched).to_csv(out / "dataset_summary_matched.csv", index=False)

    issues = issues_for(all_rows, qT_max_over_Q=float(args.qT_max_over_Q))
    issues.to_csv(out / "issues_all_rows.csv", index=False)

    pd.DataFrame(metas).to_csv(out / "source_validation_summary.csv", index=False)

    manifest = {
        "source_data_dir": str(source),
        "datasets_requested": args.datasets,
        "datasets_missing": missing,
        "out_data_dir": str(out_data),
        "n_rows_all": int(len(all_rows)),
        "n_rows_matched_qT_over_Q_le_0p5": int(len(matched)),
        "n_rows_tmd_core_qT_over_Q_le_0p2": int(len(tmd)),
        "max_prefactor_validation_errors": {
            m["dataset"]: {
                "CS": m["max_relerr_CS_vs_A_over_PreFactor"],
                "error": m["max_relerr_error_vs_dA_over_PreFactor"],
                "n_PreFactor": m["n_PreFactor"],
            }
            for m in metas
        },
        "outputs": {
            "trainer_data_dir": str(out_data),
            "summary_matched": str(out / "dataset_summary_matched.csv"),
            "source_validation_summary": str(out / "source_validation_summary.csv"),
            "issues": str(out / "issues_all_rows.csv"),
        },
        "next_steps": [
            "Confirm source_validation_summary has tiny prefactor consistency errors where PreFactor exists.",
            "Run v23a check-only backend cache using out_data_dir.",
            "Do not use the raw global_dy_raw A_direct candidate table for fitting."
        ],
    }
    (out / "v23a_verified_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print("\n=== v23a verified fixed-target table ===")
    print(json.dumps(manifest, indent=2))

    print("\n=== Matched summary ===")
    print(summary(matched).to_string(index=False))

    print("\n=== Source validation ===")
    print(pd.DataFrame(metas).to_string(index=False))

    print("\n=== Issue counts ===")
    if issues.empty:
        print("(none)")
    else:
        print(issues.groupby("issue").size().sort_values(ascending=False).rename("n").reset_index().to_string(index=False))

    print("\nwrote:", out)
    print("trainer data dir:", out_data)


if __name__ == "__main__":
    main()
