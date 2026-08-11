#!/usr/bin/env python3
"""Patch/variant builder for the v23a E288_300:99 suspect row.

Context
-------
The verified v23a table found one matched/TMD-core row with a factor-10
mismatch between explicit CS and A/PreFactor:

  E288_300:99, qT=0.9, Q=10.5

Neighboring rows and A/PreFactor indicate the explicit CS value is likely a
decimal-place typo.  This script creates two clean v23a variants:

  1. corrected:
       keep the row but set CS=A/PreFactor and error=dA/PreFactor.

  2. drop_suspect:
       remove the row completely.

Both variants write trainer-ready per-dataset matched CSVs and audit manifests.
No original files are modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ROW_ID = "E288_300:99"


def finite(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def relerr(a: pd.Series, b: pd.Series) -> pd.Series:
    return np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), 1.0e-300)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    tmp = frame.copy()
    tmp["rel_error"] = tmp["error"] / tmp["CS"].replace(0.0, np.nan)

    def fmin(s):
        v = finite(s)
        v = v[np.isfinite(v)]
        return float(v.min()) if len(v) else np.nan

    def fmax(s):
        v = finite(s)
        v = v[np.isfinite(v)]
        return float(v.max()) if len(v) else np.nan

    def fmed(s):
        v = finite(s)
        v = v[np.isfinite(v)]
        return float(v.median()) if len(v) else np.nan

    return (
        tmp.groupby("dataset", observed=False)
        .agg(
            n_rows=("row_id", "size"),
            n_tmd_core=("qT_over_Q", lambda s: int((finite(s) <= 0.2).sum())),
            n_matched=("qT_over_Q", lambda s: int((finite(s) <= 0.5).sum())),
            Q_min=("QM", fmin),
            Q_max=("QM", fmax),
            qT_min=("qT", fmin),
            qT_max=("qT", fmax),
            qT_over_Q_median=("qT_over_Q", fmed),
            qT_over_Q_max=("qT_over_Q", fmax),
            CS_min=("CS", fmin),
            CS_max=("CS", fmax),
            rel_error_median=("rel_error", fmed),
            rel_error_max=("rel_error", fmax),
        )
        .reset_index()
    )


def recompute_prefactor_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for c in ["CS", "error", "A", "dA", "PreFactor", "qT_over_Q"]:
        if c in out.columns:
            out[c] = finite(out[c])

    has_pref = np.isfinite(out["PreFactor"]) & (out["PreFactor"] != 0.0)
    out["A_over_PreFactor"] = np.nan
    out["dA_over_PreFactor"] = np.nan

    mask = has_pref & np.isfinite(out["A"])
    out.loc[mask, "A_over_PreFactor"] = out.loc[mask, "A"] / out.loc[mask, "PreFactor"]

    mask = has_pref & np.isfinite(out["dA"])
    out.loc[mask, "dA_over_PreFactor"] = out.loc[mask, "dA"] / out.loc[mask, "PreFactor"]

    out["relerr_CS_vs_A_over_PreFactor"] = relerr(out["CS"], out["A_over_PreFactor"])
    out["relerr_error_vs_dA_over_PreFactor"] = relerr(out["error"], out["dA_over_PreFactor"])
    out["ratio_CS_to_A_over_PreFactor"] = out["CS"] / out["A_over_PreFactor"].replace(0.0, np.nan)
    out["ratio_error_to_dA_over_PreFactor"] = out["error"] / out["dA_over_PreFactor"].replace(0.0, np.nan)
    return out


def write_variant(
    *,
    frame: pd.DataFrame,
    variant: str,
    out_root: Path,
    data_root: Path,
    qT_max_over_Q: float,
    tmd_qT_max_over_Q: float,
    correction_record: dict,
) -> dict:
    variant_out = out_root / variant
    variant_data = data_root / variant
    variant_out.mkdir(parents=True, exist_ok=True)
    variant_data.mkdir(parents=True, exist_ok=True)

    frame = recompute_prefactor_columns(frame)

    matched = frame[
        np.isfinite(frame["qT_over_Q"])
        & (frame["qT_over_Q"] <= float(qT_max_over_Q))
        & np.isfinite(frame["CS"])
        & np.isfinite(frame["error"])
        & (frame["CS"] > 0.0)
        & (frame["error"] > 0.0)
    ].copy()

    tmd = matched[
        np.isfinite(matched["qT_over_Q"])
        & (matched["qT_over_Q"] <= float(tmd_qT_max_over_Q))
    ].copy()

    for dataset, group in matched.groupby("dataset", observed=False):
        group.to_csv(variant_data / f"{dataset}.csv", index=False)

    frame.to_csv(variant_out / "all_rows.csv", index=False)
    matched.to_csv(variant_out / "matched_qToQ_le_0p5_positive.csv", index=False)
    tmd.to_csv(variant_out / "tmd_core_qToQ_le_0p2_positive.csv", index=False)
    summarize(frame).to_csv(variant_out / "dataset_summary_all_rows.csv", index=False)
    summarize(matched).to_csv(variant_out / "dataset_summary_matched.csv", index=False)

    issue_rows = []
    for _, row in frame.iterrows():
        base = {
            "dataset": row.get("dataset", ""),
            "row_id": row.get("row_id", ""),
            "qT": row.get("qT", np.nan),
            "QM": row.get("QM", np.nan),
            "qT_over_Q": row.get("qT_over_Q", np.nan),
        }
        checks = [
            ("outside_matched_qT_over_Q", np.isfinite(row.get("qT_over_Q", np.nan)) and row["qT_over_Q"] > qT_max_over_Q),
            ("nonpositive_CS", np.isfinite(row.get("CS", np.nan)) and row["CS"] <= 0.0),
            ("nonpositive_error", np.isfinite(row.get("error", np.nan)) and row["error"] <= 0.0),
            ("CS_prefactor_relerr_gt_1e_3", np.isfinite(row.get("relerr_CS_vs_A_over_PreFactor", np.nan)) and row["relerr_CS_vs_A_over_PreFactor"] > 1e-3),
            ("error_prefactor_relerr_gt_1e_3", np.isfinite(row.get("relerr_error_vs_dA_over_PreFactor", np.nan)) and row["relerr_error_vs_dA_over_PreFactor"] > 1e-3),
        ]
        for issue, bad in checks:
            if bad:
                issue_rows.append({**base, "issue": issue})

    issues = pd.DataFrame(issue_rows)
    issues.to_csv(variant_out / "issues.csv", index=False)

    matched_bad = issues[
        issues["issue"].isin(["nonpositive_CS", "nonpositive_error", "CS_prefactor_relerr_gt_1e_3"])
        & (pd.to_numeric(issues["qT_over_Q"], errors="coerce") <= float(qT_max_over_Q))
    ] if not issues.empty else pd.DataFrame()

    manifest = {
        "variant": variant,
        "trainer_data_dir": str(variant_data),
        "n_rows_all": int(len(frame)),
        "n_rows_matched_positive_qT_over_Q_le_0p5": int(len(matched)),
        "n_rows_tmd_core_positive_qT_over_Q_le_0p2": int(len(tmd)),
        "datasets": sorted(str(x) for x in matched["dataset"].unique()),
        "correction_record": correction_record,
        "n_matched_bad_rows_after_variant": int(len(matched_bad)),
        "max_matched_CS_prefactor_relerr": (
            float(np.nanmax(matched["relerr_CS_vs_A_over_PreFactor"]))
            if np.isfinite(matched["relerr_CS_vs_A_over_PreFactor"]).any()
            else None
        ),
        "max_matched_error_prefactor_relerr": (
            float(np.nanmax(matched["relerr_error_vs_dA_over_PreFactor"]))
            if np.isfinite(matched["relerr_error_vs_dA_over_PreFactor"]).any()
            else None
        ),
        "outputs": {
            "all_rows": str(variant_out / "all_rows.csv"),
            "matched": str(variant_out / "matched_qToQ_le_0p5_positive.csv"),
            "tmd_core": str(variant_out / "tmd_core_qToQ_le_0p2_positive.csv"),
            "dataset_summary_matched": str(variant_out / "dataset_summary_matched.csv"),
            "issues": str(variant_out / "issues.csv"),
        },
    }
    (variant_out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-table",
        default="v23/outputs/v23a_fixed_target_lowQ_verified_table/v23a_verified_all_rows.csv",
    )
    parser.add_argument(
        "--row-id",
        default=DEFAULT_ROW_ID,
    )
    parser.add_argument(
        "--out-root",
        default="v23/outputs/v23a_fixed_target_lowQ_row99_variants",
    )
    parser.add_argument(
        "--data-root",
        default="Data/v23a_fixed_target_lowQ_row99_variants",
    )
    parser.add_argument("--qT-max-over-Q", type=float, default=0.5)
    parser.add_argument("--tmd-qT-max-over-Q", type=float, default=0.2)
    args = parser.parse_args()

    input_path = Path(args.input_table)
    if not input_path.exists():
        raise SystemExit(f"Missing input table: {input_path}")

    df = pd.read_csv(input_path)
    for c in ["CS", "error", "A", "dA", "PreFactor", "qT_over_Q"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    target_mask = df["row_id"].astype(str).eq(str(args.row_id))
    if int(target_mask.sum()) != 1:
        raise SystemExit(f"Expected exactly one row_id={args.row_id}; found {int(target_mask.sum())}")

    row_before = recompute_prefactor_columns(df[target_mask].copy()).iloc[0].to_dict()

    # Corrected variant.
    corrected = df.copy()
    target = corrected.loc[target_mask].copy()
    pf = float(target["PreFactor"].iloc[0])
    if not np.isfinite(pf) or pf == 0.0:
        raise SystemExit("Target row has missing/nonzero PreFactor; cannot correct.")
    new_cs = float(target["A"].iloc[0]) / pf
    new_err = float(target["dA"].iloc[0]) / pf
    old_cs = float(target["CS"].iloc[0])
    old_err = float(target["error"].iloc[0])

    corrected.loc[target_mask, "CS"] = new_cs
    corrected.loc[target_mask, "error"] = new_err
    corrected.loc[target_mask, "row_correction_note"] = (
        f"Corrected {args.row_id}: CS {old_cs} -> A/PreFactor {new_cs}; "
        f"error {old_err} -> dA/PreFactor {new_err}."
    )

    correction_record = {
        "row_id": args.row_id,
        "old_CS": old_cs,
        "new_CS_A_over_PreFactor": new_cs,
        "old_error": old_err,
        "new_error_dA_over_PreFactor": new_err,
        "factor_old_CS_over_new_CS": old_cs / new_cs,
        "row_before": {k: (None if pd.isna(v) else v) for k, v in row_before.items()},
    }

    # Drop-row variant.
    dropped = df.loc[~target_mask].copy()

    out_root = Path(args.out_root)
    data_root = Path(args.data_root)
    out_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)

    corrected_manifest = write_variant(
        frame=corrected,
        variant="corrected_E288_300_99",
        out_root=out_root,
        data_root=data_root,
        qT_max_over_Q=float(args.qT_max_over_Q),
        tmd_qT_max_over_Q=float(args.tmd_qT_max_over_Q),
        correction_record=correction_record,
    )

    drop_manifest = write_variant(
        frame=dropped,
        variant="drop_E288_300_99",
        out_root=out_root,
        data_root=data_root,
        qT_max_over_Q=float(args.qT_max_over_Q),
        tmd_qT_max_over_Q=float(args.tmd_qT_max_over_Q),
        correction_record=correction_record,
    )

    summary = {
        "input_table": str(input_path),
        "row_id": args.row_id,
        "recommend_primary_variant": "corrected_E288_300_99",
        "recommend_robustness_variant": "drop_E288_300_99",
        "corrected_manifest": corrected_manifest,
        "drop_manifest": drop_manifest,
        "next_steps": [
            "Run v23a check-only cache with corrected_E288_300_99 as the primary data directory.",
            "If central refit is sensitive, also refit drop_E288_300_99 as a robustness check.",
            "Do not use the uncorrected verified table for v23a fitting."
        ],
    }

    (out_root / "row99_variant_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("\n=== v23a E288_300:99 variants created ===")
    print(json.dumps(summary, indent=2))
    print("\nCorrected trainer data dir:", corrected_manifest["trainer_data_dir"])
    print("Drop-row trainer data dir:", drop_manifest["trainer_data_dir"])


if __name__ == "__main__":
    main()
