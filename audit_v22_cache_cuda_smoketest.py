#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def first_existing(frame: pd.DataFrame, columns: list[str]) -> str | None:
    for column in columns:
        if column in frame.columns:
            return column
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-run", required=True)
    parser.add_argument("--test-run", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-rel-pred", type=float, default=1.0e-4)
    parser.add_argument("--max-delta-pull", type=float, default=1.0e-3)
    args = parser.parse_args()

    ref_run = Path(args.reference_run)
    test_run = Path(args.test_run)

    ref_path = ref_run / "predictions.csv"
    test_path = test_run / "predictions.csv"

    for path in [ref_path, test_path]:
        if not path.exists():
            raise SystemExit(f"Missing {path}")

    ref = pd.read_csv(ref_path)
    test = pd.read_csv(test_path)

    keys = ["dataset", "row_id"]
    merged = ref.merge(
        test,
        on=keys,
        suffixes=("_ref", "_test"),
        validate="one_to_one",
    )

    pred_ref = "pred_match_CS_ref"
    pred_test = "pred_match_CS_test"
    sigma = first_existing(merged, ["sigma_used_ref", "sigma_uncorr_ref", "error_ref"])

    if pred_ref not in merged.columns or pred_test not in merged.columns:
        raise SystemExit("Prediction columns missing after merge.")

    merged["abs_delta_pred"] = np.abs(merged[pred_test] - merged[pred_ref])
    merged["rel_delta_pred"] = merged["abs_delta_pred"] / np.maximum(
        np.maximum(np.abs(merged[pred_ref]), np.abs(merged[pred_test])),
        1.0e-300,
    )

    if sigma is not None:
        merged["delta_pull_units"] = (merged[pred_test] - merged[pred_ref]) / merged[sigma].replace(0.0, np.nan)
        max_delta_pull = float(np.nanmax(np.abs(merged["delta_pull_units"])))
    else:
        merged["delta_pull_units"] = np.nan
        max_delta_pull = float("nan")

    by_dataset = (
        merged.groupby("dataset", observed=False)
        .agg(
            n=("row_id", "size"),
            max_rel_delta_pred=("rel_delta_pred", "max"),
            median_rel_delta_pred=("rel_delta_pred", "median"),
            max_abs_delta_pull=("delta_pull_units", lambda x: float(np.nanmax(np.abs(x)))),
        )
        .reset_index()
    )

    max_rel = float(np.nanmax(merged["rel_delta_pred"]))

    pass_status = bool(
        np.isfinite(max_rel)
        and max_rel < float(args.max_rel_pred)
        and (
            not np.isfinite(max_delta_pull)
            or max_delta_pull < float(args.max_delta_pull)
        )
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out / "cache_cuda_prediction_comparison.csv", index=False)
    by_dataset.to_csv(out / "cache_cuda_by_dataset.csv", index=False)

    summary = {
        "reference_run": str(ref_run),
        "test_run": str(test_run),
        "n_rows": int(len(merged)),
        "max_rel_delta_pred": max_rel,
        "max_abs_delta_pull": max_delta_pull,
        "max_rel_pred_threshold": float(args.max_rel_pred),
        "max_delta_pull_threshold": float(args.max_delta_pull),
        "CACHE_CUDA_SMOKETEST_PASS": pass_status,
    }
    (out / "cache_cuda_smoketest_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("\n=== Cache/CUDA smoke-test summary ===")
    print(json.dumps(summary, indent=2))
    print("\n=== By dataset ===")
    print(by_dataset.to_string(index=False))
    print("\nwrote:", out)

    if not pass_status:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
