#!/usr/bin/env python3
"""Audit v23a central-refit normalization and nonfinite prediction columns.

This diagnoses failures like:
  * gigantic E772 dataset_norm_pull,
  * "all_prediction_numeric_values_finite": false.

It does not modify the run.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as handle:
        return json.load(handle)


def parse_percent_or_float(value: Any) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none"}:
        return np.nan
    is_percent = "%" in text
    text = text.replace("%", "").replace(",", "").strip()
    try:
        x = float(text)
    except Exception:
        return np.nan
    return x / 100.0 if is_percent else x


def summarize_data_norm_columns(data_dir: Path, datasets: list[str]) -> pd.DataFrame:
    rows = []
    for dataset in datasets:
        path = data_dir / f"{dataset}.csv"
        if not path.exists():
            rows.append({"dataset": dataset, "data_file": str(path), "exists": False})
            continue
        df = pd.read_csv(path)
        row = {
            "dataset": dataset,
            "data_file": str(path),
            "exists": True,
            "n_rows": int(len(df)),
        }
        for col in ["sysNorm", "sysP2P", "frac_error"]:
            if col in df.columns:
                vals = df[col].map(parse_percent_or_float)
                finite = vals[np.isfinite(vals)]
                row[f"{col}_nfinite"] = int(len(finite))
                if len(finite):
                    row[f"{col}_min"] = float(finite.min())
                    row[f"{col}_median"] = float(finite.median())
                    row[f"{col}_max"] = float(finite.max())
                    row[f"{col}_unique_preview"] = ",".join(
                        f"{x:.8g}" for x in sorted(pd.Series(finite).drop_duplicates().head(8).tolist())
                    )
                else:
                    row[f"{col}_min"] = np.nan
                    row[f"{col}_median"] = np.nan
                    row[f"{col}_max"] = np.nan
                    row[f"{col}_unique_preview"] = ""
            else:
                row[f"{col}_nfinite"] = 0
                row[f"{col}_unique_preview"] = ""
        rows.append(row)
    return pd.DataFrame(rows)


def nonfinite_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        n_nonfinite = int((~np.isfinite(values)).sum())
        if n_nonfinite:
            rows.append({
                "column": col,
                "n_nonfinite": n_nonfinite,
                "n_rows": int(len(df)),
                "fraction_nonfinite": n_nonfinite / max(len(df), 1),
                "example_rows": ",".join(map(str, df.loc[~np.isfinite(values), "row_id"].head(8).tolist()))
                if "row_id" in df.columns else "",
            })
    return pd.DataFrame(rows).sort_values("n_nonfinite", ascending=False) if rows else pd.DataFrame()


def first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def core_finite_report(pred: pd.DataFrame) -> dict[str, Any]:
    pred_col = first_existing(pred, ["pred_match_CS", "prediction", "pred", "matched_CS", "W_plus_Y"])
    data_col = first_existing(pred, ["CS", "target", "data", "target_CS"])
    err_col = first_existing(pred, ["sigma_used", "error", "sigma_uncorr", "unc"])
    core_cols = [c for c in [pred_col, data_col, err_col] if c is not None]
    out = {
        "prediction_column": pred_col,
        "data_column": data_col,
        "error_column": err_col,
        "core_columns": core_cols,
    }
    for c in core_cols:
        values = pd.to_numeric(pred[c], errors="coerce")
        out[f"{c}_n_nonfinite"] = int((~np.isfinite(values)).sum())
        out[f"{c}_finite"] = bool(np.isfinite(values).all())
    out["core_prediction_values_finite"] = bool(
        core_cols and all(out.get(f"{c}_finite", False) for c in core_cols)
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--data-dir", default="Data/v23a_fixed_target_lowQ_row99_variants/corrected_E288_300_99")
    parser.add_argument("--out", default="v23/outputs/v23a_norm_finite_audit_s303")
    args = parser.parse_args()

    run = Path(args.run)
    data_dir = Path(args.data_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    metrics = load_json(run / "metrics.json")
    pred_path = run / "predictions.csv"
    norms_path = run / "dataset_norms.csv"

    if not pred_path.exists():
        raise SystemExit(f"Missing predictions: {pred_path}")

    pred = pd.read_csv(pred_path)
    datasets = sorted(pred["dataset"].astype(str).unique()) if "dataset" in pred.columns else []

    norm_pulls = metrics.get("dataset_norm_pulls", {})
    norm_pull_rows = [
        {"dataset": str(k), "norm_pull": float(v), "abs_norm_pull": abs(float(v))}
        for k, v in norm_pulls.items()
    ]
    norm_pull_df = pd.DataFrame(norm_pull_rows).sort_values("abs_norm_pull", ascending=False) if norm_pull_rows else pd.DataFrame()

    dataset_norms = pd.read_csv(norms_path) if norms_path.exists() else pd.DataFrame()
    data_norm_summary = summarize_data_norm_columns(data_dir, datasets)
    nonfinite = nonfinite_report(pred)
    core_report = core_finite_report(pred)

    norm_pull_df.to_csv(out / "metrics_dataset_norm_pulls.csv", index=False)
    dataset_norms.to_csv(out / "dataset_norms_file_copy.csv", index=False)
    data_norm_summary.to_csv(out / "data_norm_columns_summary.csv", index=False)
    nonfinite.to_csv(out / "prediction_nonfinite_numeric_columns.csv", index=False)

    summary = {
        "run": str(run),
        "data_dir": str(data_dir),
        "metrics_dataset_norm_pulls": norm_pulls,
        "max_abs_norm_pull": float(norm_pull_df["abs_norm_pull"].max()) if not norm_pull_df.empty else None,
        "dataset_norms_columns": list(dataset_norms.columns),
        "prediction_nonfinite_numeric_column_count": int(len(nonfinite)),
        "core_finite_report": core_report,
        "likely_issue": (
            "If E772 has a huge pull while data sysNorm is finite/nonzero, rerun with --norm-source csv. "
            "If E772 sysNorm is zero/missing, define an E772 normalization prior or run a no-fit-norm sensitivity test."
        ),
    }
    (out / "norm_finite_audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("\n=== v23a normalization / finite audit ===")
    print(json.dumps(summary, indent=2))

    print("\n=== Metrics norm pulls ===")
    print(norm_pull_df.to_string(index=False) if not norm_pull_df.empty else "(none)")

    print("\n=== dataset_norms.csv ===")
    print(dataset_norms.to_string(index=False) if not dataset_norms.empty else "(missing/empty)")

    print("\n=== Data sysNorm/sysP2P summary ===")
    print(data_norm_summary.to_string(index=False))

    print("\n=== Nonfinite numeric columns in predictions ===")
    print(nonfinite.to_string(index=False) if not nonfinite.empty else "(none)")

    print("\nwrote:", out)


if __name__ == "__main__":
    main()
