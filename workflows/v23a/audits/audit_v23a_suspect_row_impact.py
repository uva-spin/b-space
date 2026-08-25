#!/usr/bin/env python3
"""v23a suspect-row impact audit.

Use this after audit_v23a_prefactor_triage.py identifies any matched/TMD-core
rows where explicit CS disagrees strongly with A/PreFactor.

It checks whether existing v22 central predictions prefer:
  * the explicit trainer CS column, or
  * the A/PreFactor provenance reconstruction.

It also writes a local neighborhood table by dataset/QM so the suspect row can
be inspected against adjacent qT points.

No data are modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def relerr(a: pd.Series, b: pd.Series) -> pd.Series:
    return np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), 1.0e-300)


def first_col(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in frame.columns:
            return c
    return None


def load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    pred = pd.read_csv(path)
    if "dataset" not in pred.columns or "row_id" not in pred.columns:
        return pd.DataFrame()

    pred_col = first_col(pred, ["pred_match_CS", "prediction", "pred", "theory", "W_plus_Y"])
    data_col = first_col(pred, ["CS", "target_used", "data", "target"])
    sigma_col = first_col(pred, ["sigma_used", "error", "sigma_uncorr", "unc"])

    keep = ["dataset", "row_id"]
    rename = {}
    if pred_col:
        keep.append(pred_col)
        rename[pred_col] = "prediction"
    if data_col:
        keep.append(data_col)
        rename[data_col] = "prediction_file_data"
    if sigma_col:
        keep.append(sigma_col)
        rename[sigma_col] = "prediction_file_sigma"

    return pred[keep].rename(columns=rename)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--table",
        default="v23/outputs/v23a_fixed_target_lowQ_verified_table/v23a_verified_all_rows.csv",
    )
    parser.add_argument(
        "--predictions",
        default="outputs/v22_full_backend_central_refit_stage1_s303/predictions.csv",
    )
    parser.add_argument(
        "--relerr-threshold",
        type=float,
        default=1.0e-3,
    )
    parser.add_argument(
        "--qT-max-over-Q",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--out",
        default="v23/outputs/v23a_fixed_target_lowQ_verified_table/suspect_row_impact",
    )
    args = parser.parse_args()

    table_path = Path(args.table)
    if not table_path.exists():
        raise SystemExit(f"Missing table: {table_path}")

    df = pd.read_csv(table_path)

    for col in ["qT", "QM", "qT_over_Q", "CS", "error", "A", "dA", "PreFactor"]:
        if col not in df.columns:
            raise SystemExit(f"Missing required column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    has_pref = np.isfinite(df["PreFactor"]) & (df["PreFactor"] != 0)
    df["A_over_PreFactor"] = np.nan
    df["dA_over_PreFactor"] = np.nan
    df.loc[has_pref & np.isfinite(df["A"]), "A_over_PreFactor"] = (
        df.loc[has_pref & np.isfinite(df["A"]), "A"]
        / df.loc[has_pref & np.isfinite(df["A"]), "PreFactor"]
    )
    df.loc[has_pref & np.isfinite(df["dA"]), "dA_over_PreFactor"] = (
        df.loc[has_pref & np.isfinite(df["dA"]), "dA"]
        / df.loc[has_pref & np.isfinite(df["dA"]), "PreFactor"]
    )

    df["relerr_CS_vs_A_over_PreFactor"] = relerr(df["CS"], df["A_over_PreFactor"])
    df["relerr_error_vs_dA_over_PreFactor"] = relerr(df["error"], df["dA_over_PreFactor"])
    df["ratio_CS_to_A_over_PreFactor"] = df["CS"] / df["A_over_PreFactor"].replace(0.0, np.nan)
    df["ratio_error_to_dA_over_PreFactor"] = df["error"] / df["dA_over_PreFactor"].replace(0.0, np.nan)
    df["matched_candidate"] = np.isfinite(df["qT_over_Q"]) & (df["qT_over_Q"] <= float(args.qT_max_over_Q))

    suspect = df[
        df["matched_candidate"]
        & np.isfinite(df["relerr_CS_vs_A_over_PreFactor"])
        & (df["relerr_CS_vs_A_over_PreFactor"] > float(args.relerr_threshold))
    ].copy()

    pred = load_predictions(Path(args.predictions))
    if not pred.empty:
        suspect = suspect.merge(pred, on=["dataset", "row_id"], how="left", validate="one_to_one")

        if "prediction" in suspect.columns:
            suspect["pull_vs_explicit_CS"] = (
                suspect["prediction"] - suspect["CS"]
            ) / suspect["error"].replace(0.0, np.nan)
            suspect["pull_vs_A_over_PreFactor"] = (
                suspect["prediction"] - suspect["A_over_PreFactor"]
            ) / suspect["error"].replace(0.0, np.nan)
            suspect["relative_prediction_to_explicit_CS"] = (
                suspect["prediction"] / suspect["CS"].replace(0.0, np.nan)
            )
            suspect["relative_prediction_to_A_over_PreFactor"] = (
                suspect["prediction"] / suspect["A_over_PreFactor"].replace(0.0, np.nan)
            )

    # Neighborhoods around each suspect: same dataset and same QM if possible.
    neighborhoods = []
    for _, row in suspect.iterrows():
        dataset = row["dataset"]
        qm = row["QM"]
        mask = df["dataset"].astype(str).eq(str(dataset))
        if np.isfinite(qm):
            mask &= np.isclose(df["QM"], qm, rtol=0.0, atol=1.0e-10)
        local = df[mask].copy().sort_values("qT")
        local["suspect_target_row_id"] = row["row_id"]
        local["is_suspect_row"] = local["row_id"].astype(str).eq(str(row["row_id"]))
        neighborhoods.append(local)

    neighborhood = pd.concat(neighborhoods, ignore_index=True) if neighborhoods else pd.DataFrame()
    if not neighborhood.empty and not pred.empty:
        neighborhood = neighborhood.merge(pred, on=["dataset", "row_id"], how="left")

    # Compact columns.
    compact_cols = [
        "dataset", "row_id", "source_row", "qT", "QM", "qT_over_Q",
        "matched_candidate", "CS", "error", "A", "dA", "PreFactor",
        "A_over_PreFactor", "dA_over_PreFactor",
        "ratio_CS_to_A_over_PreFactor",
        "ratio_error_to_dA_over_PreFactor",
        "relerr_CS_vs_A_over_PreFactor",
        "relerr_error_vs_dA_over_PreFactor",
        "prediction", "prediction_file_data", "prediction_file_sigma",
        "pull_vs_explicit_CS", "pull_vs_A_over_PreFactor",
        "relative_prediction_to_explicit_CS",
        "relative_prediction_to_A_over_PreFactor",
        "source_file",
    ]
    compact_cols = [c for c in compact_cols if c in suspect.columns]

    neigh_cols = [
        "suspect_target_row_id", "is_suspect_row",
        "dataset", "row_id", "source_row", "qT", "QM", "qT_over_Q",
        "CS", "error", "A", "dA", "PreFactor",
        "A_over_PreFactor", "ratio_CS_to_A_over_PreFactor",
        "relerr_CS_vs_A_over_PreFactor",
        "prediction", "prediction_file_data", "prediction_file_sigma",
    ]
    neigh_cols = [c for c in neigh_cols if c in neighborhood.columns]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    suspect.to_csv(out / "suspect_rows_full.csv", index=False)
    suspect[compact_cols].to_csv(out / "suspect_rows_compact.csv", index=False)
    if not neighborhood.empty:
        neighborhood[neigh_cols].to_csv(out / "suspect_neighborhoods.csv", index=False)

    summary = {
        "table": str(table_path),
        "predictions": str(Path(args.predictions)),
        "relerr_threshold": float(args.relerr_threshold),
        "n_suspect_matched_rows": int(len(suspect)),
        "suspect_row_ids": suspect[["dataset", "row_id"]].to_dict(orient="records") if len(suspect) else [],
        "prediction_columns_found": list(pred.columns) if not pred.empty else [],
        "recommendation": (
            "If prediction and neighboring rows favor explicit CS, keep explicit CS and record A/PreFactor as faulty provenance for that row. "
            "If they favor A/PreFactor, quarantine or correct the row before v23a fitting. "
            "If ambiguous, run v23a central refits with keep-row and drop-row variants before replicas."
        ),
    }
    (out / "suspect_row_impact_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("\n=== v23a suspect-row impact summary ===")
    print(json.dumps(summary, indent=2))

    print("\n=== Suspect rows ===")
    if suspect.empty:
        print("(none)")
    else:
        print(suspect[compact_cols].to_string(index=False))

    print("\n=== Neighborhoods ===")
    if neighborhood.empty:
        print("(none)")
    else:
        print(neighborhood[neigh_cols].to_string(index=False))

    print("\nwrote:", out)


if __name__ == "__main__":
    main()
