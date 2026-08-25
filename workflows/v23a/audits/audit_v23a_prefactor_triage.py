#!/usr/bin/env python3
"""Triage v23a fixed-target verified-data prefactor/convention warnings.

This script answers:
  * Are the A/PreFactor vs CS inconsistencies localized to excluded rows?
  * Are nonpositive CS rows inside the actual matched/TMD-core candidate region?
  * What are the worst rows and ratio patterns by dataset?

It does not modify any data.  Use before running the v23a check-only cache.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def finite(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def relerr(a: pd.Series, b: pd.Series) -> pd.Series:
    return np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), 1.0e-300)


def summarize_region(df: pd.DataFrame, region_name: str, mask: pd.Series) -> dict:
    g = df[mask].copy()
    out = {
        "region": region_name,
        "n": int(len(g)),
    }
    if g.empty:
        return out

    for col in [
        "CS",
        "error",
        "A_over_PreFactor",
        "dA_over_PreFactor",
        "ratio_CS_to_A_over_PreFactor",
        "ratio_error_to_dA_over_PreFactor",
        "relerr_CS",
        "relerr_error",
    ]:
        vals = finite(g[col]) if col in g.columns else pd.Series(dtype=float)
        vals = vals[np.isfinite(vals)]
        out[f"{col}_nfinite"] = int(len(vals))
        if len(vals):
            out[f"{col}_min"] = float(vals.min())
            out[f"{col}_median"] = float(vals.median())
            out[f"{col}_max"] = float(vals.max())

    out["n_nonpositive_CS"] = int((np.isfinite(g["CS"]) & (g["CS"] <= 0)).sum())
    out["n_nonpositive_error"] = int((np.isfinite(g["error"]) & (g["error"] <= 0)).sum())
    out["n_CS_prefactor_relerr_gt_1e_5"] = int((np.isfinite(g["relerr_CS"]) & (g["relerr_CS"] > 1e-5)).sum())
    out["n_error_prefactor_relerr_gt_1e_5"] = int((np.isfinite(g["relerr_error"]) & (g["relerr_error"] > 1e-5)).sum())
    out["n_CS_prefactor_relerr_gt_1e_3"] = int((np.isfinite(g["relerr_CS"]) & (g["relerr_CS"] > 1e-3)).sum())
    out["n_error_prefactor_relerr_gt_1e_3"] = int((np.isfinite(g["relerr_error"]) & (g["relerr_error"] > 1e-3)).sum())

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--table",
        default="v23/outputs/v23a_fixed_target_lowQ_verified_table/v23a_verified_all_rows.csv",
    )
    parser.add_argument(
        "--out",
        default="v23/outputs/v23a_fixed_target_lowQ_verified_table/prefactor_triage",
    )
    parser.add_argument("--qT-max-over-Q", type=float, default=0.5)
    parser.add_argument("--tmd-qT-max-over-Q", type=float, default=0.2)
    args = parser.parse_args()

    path = Path(args.table)
    if not path.exists():
        raise SystemExit(f"Missing table: {path}")

    df = pd.read_csv(path)

    required = ["dataset", "row_id", "qT", "QM", "CS", "error", "A", "dA", "PreFactor", "qT_over_Q"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns: {missing}")

    for col in ["qT", "QM", "CS", "error", "A", "dA", "PreFactor", "qT_over_Q"]:
        df[col] = finite(df[col])

    has_pref = np.isfinite(df["PreFactor"]) & (df["PreFactor"] != 0)
    df["A_over_PreFactor"] = np.nan
    df["dA_over_PreFactor"] = np.nan
    df.loc[has_pref & np.isfinite(df["A"]), "A_over_PreFactor"] = (
        df.loc[has_pref & np.isfinite(df["A"]), "A"] / df.loc[has_pref & np.isfinite(df["A"]), "PreFactor"]
    )
    df.loc[has_pref & np.isfinite(df["dA"]), "dA_over_PreFactor"] = (
        df.loc[has_pref & np.isfinite(df["dA"]), "dA"] / df.loc[has_pref & np.isfinite(df["dA"]), "PreFactor"]
    )

    df["ratio_CS_to_A_over_PreFactor"] = df["CS"] / df["A_over_PreFactor"].replace(0.0, np.nan)
    df["ratio_error_to_dA_over_PreFactor"] = df["error"] / df["dA_over_PreFactor"].replace(0.0, np.nan)
    df["relerr_CS"] = relerr(df["CS"], df["A_over_PreFactor"])
    df["relerr_error"] = relerr(df["error"], df["dA_over_PreFactor"])
    df["matched_candidate"] = np.isfinite(df["qT_over_Q"]) & (df["qT_over_Q"] <= float(args.qT_max_over_Q))
    df["tmd_core_candidate"] = np.isfinite(df["qT_over_Q"]) & (df["qT_over_Q"] <= float(args.tmd_qT_max_over_Q))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    region_rows = []
    for dataset, group in df.groupby("dataset", observed=False):
        for region, mask in [
            ("all", pd.Series(True, index=group.index)),
            ("matched_qToQ_le_0p5", group["matched_candidate"]),
            ("tmd_core_qToQ_le_0p2", group["tmd_core_candidate"]),
            ("outside_matched", ~group["matched_candidate"]),
        ]:
            region_rows.append({
                "dataset": dataset,
                **summarize_region(group, region, mask),
            })

    by_region = pd.DataFrame(region_rows)

    worst_cs = (
        df[np.isfinite(df["relerr_CS"])]
        .sort_values("relerr_CS", ascending=False)
        .head(80)
    )
    worst_err = (
        df[np.isfinite(df["relerr_error"])]
        .sort_values("relerr_error", ascending=False)
        .head(80)
    )
    nonpositive = df[
        (np.isfinite(df["CS"]) & (df["CS"] <= 0))
        | (np.isfinite(df["error"]) & (df["error"] <= 0))
    ].copy()

    cols = [
        "dataset", "row_id", "source_row", "qT", "QM", "qT_over_Q",
        "matched_candidate", "tmd_core_candidate",
        "CS", "A", "PreFactor", "A_over_PreFactor",
        "ratio_CS_to_A_over_PreFactor", "relerr_CS",
        "error", "dA", "dA_over_PreFactor",
        "ratio_error_to_dA_over_PreFactor", "relerr_error",
    ]
    cols = [c for c in cols if c in df.columns]

    df.to_csv(out / "prefactor_triage_long.csv", index=False)
    by_region.to_csv(out / "prefactor_triage_by_dataset_region.csv", index=False)
    worst_cs[cols].to_csv(out / "worst_CS_prefactor_rows.csv", index=False)
    worst_err[cols].to_csv(out / "worst_error_prefactor_rows.csv", index=False)
    nonpositive[cols].to_csv(out / "nonpositive_rows.csv", index=False)

    # Compact decision logic.
    matched = df[df["matched_candidate"]].copy()
    tmd = df[df["tmd_core_candidate"]].copy()

    summary = {
        "input_table": str(path),
        "n_rows": int(len(df)),
        "n_matched": int(len(matched)),
        "n_tmd_core": int(len(tmd)),
        "n_nonpositive_CS_all": int((np.isfinite(df["CS"]) & (df["CS"] <= 0)).sum()),
        "n_nonpositive_CS_matched": int((np.isfinite(matched["CS"]) & (matched["CS"] <= 0)).sum()),
        "n_nonpositive_CS_tmd_core": int((np.isfinite(tmd["CS"]) & (tmd["CS"] <= 0)).sum()),
        "n_CS_prefactor_relerr_gt_1e_3_all": int((np.isfinite(df["relerr_CS"]) & (df["relerr_CS"] > 1e-3)).sum()),
        "n_CS_prefactor_relerr_gt_1e_3_matched": int((np.isfinite(matched["relerr_CS"]) & (matched["relerr_CS"] > 1e-3)).sum()),
        "n_CS_prefactor_relerr_gt_1e_3_tmd_core": int((np.isfinite(tmd["relerr_CS"]) & (tmd["relerr_CS"] > 1e-3)).sum()),
        "n_error_prefactor_relerr_gt_1e_3_all": int((np.isfinite(df["relerr_error"]) & (df["relerr_error"] > 1e-3)).sum()),
        "n_error_prefactor_relerr_gt_1e_3_matched": int((np.isfinite(matched["relerr_error"]) & (matched["relerr_error"] > 1e-3)).sum()),
        "n_error_prefactor_relerr_gt_1e_3_tmd_core": int((np.isfinite(tmd["relerr_error"]) & (tmd["relerr_error"] > 1e-3)).sum()),
        "max_relerr_CS_matched": (
            float(np.nanmax(matched["relerr_CS"])) if np.isfinite(matched["relerr_CS"]).any() else None
        ),
        "max_relerr_error_matched": (
            float(np.nanmax(matched["relerr_error"])) if np.isfinite(matched["relerr_error"]).any() else None
        ),
        "recommendation": (
            "Proceed with explicit CS/error columns only if nonpositive matched rows are zero and any CS-vs-A/PreFactor mismatches are understood as provenance-only. "
            "Do not replace explicit CS by A/PreFactor for datasets with large CS mismatch."
        ),
    }

    (out / "prefactor_triage_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("\n=== v23a prefactor triage summary ===")
    print(json.dumps(summary, indent=2))

    print("\n=== By dataset/region ===")
    show_cols = [
        "dataset", "region", "n", "n_nonpositive_CS",
        "n_CS_prefactor_relerr_gt_1e_3", "n_error_prefactor_relerr_gt_1e_3",
        "ratio_CS_to_A_over_PreFactor_median",
        "ratio_error_to_dA_over_PreFactor_median",
    ]
    show_cols = [c for c in show_cols if c in by_region.columns]
    print(by_region[show_cols].to_string(index=False))

    print("\n=== Worst CS prefactor rows ===")
    print(worst_cs[cols].head(20).to_string(index=False))

    print("\nwrote:", out)


if __name__ == "__main__":
    main()
