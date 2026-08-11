#!/usr/bin/env python3
"""Audit a v23a data-replica x PDF-replica TMD ensemble."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--band-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-replicas", type=int, default=20)
    ap.add_argument("--min-unique-pdf-members", type=int, default=5)
    args = ap.parse_args()

    plan = pd.read_csv(args.plan)
    band_dir = Path(args.band_dir)
    long_path = band_dir / "v23a_dataPDF_tmd_replica_bspace_long.csv"
    band_path = band_dir / "v23a_dataPDF_tmd_replica_bspace_bands.csv"
    rel_path = band_dir / "v23a_dataPDF_relative_band_summary.csv"
    if not long_path.exists():
        raise SystemExit(f"Missing {long_path}")
    if not band_path.exists():
        raise SystemExit(f"Missing {band_path}")

    long = pd.read_csv(long_path)
    bands = pd.read_csv(band_path)
    rel = pd.read_csv(rel_path) if rel_path.exists() else pd.DataFrame()

    # Audit run predictions and target sampling.
    frames = []
    for _, r in plan.iterrows():
        p = Path(str(r["run_dir"])) / "predictions.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df["seed"] = int(r["seed"])
        df["pdf_member"] = int(r["pdf_member"])
        keep = [c for c in ["seed", "pdf_member", "dataset", "row_id", "CS", "sigma_used", "target_used", "pred_match_CS", "dataset_norm_factor"] if c in df.columns]
        frames.append(df[keep])
    pred = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    pred_audit = {}
    if not pred.empty and {"target_used", "sigma_used", "CS"}.issubset(pred.columns):
        g = pred.groupby(["dataset", "row_id"], observed=False)
        row = g.agg(
            CS=("CS", "first"),
            sigma_used=("sigma_used", "first"),
            target_std=("target_used", "std"),
            pred_std=("pred_match_CS", "std") if "pred_match_CS" in pred.columns else ("target_used", "std"),
        ).reset_index()
        eps = 1e-300
        row["target_std_over_sigma"] = row["target_std"] / row["sigma_used"].abs().clip(lower=eps)
        row["target_std_over_CS"] = row["target_std"] / row["CS"].abs().clip(lower=eps)
        pred_audit = {
            "n_prediction_rows_long": int(len(pred)),
            "n_rows": int(len(row)),
            "target_std_over_sigma_median": float(row["target_std_over_sigma"].median()),
            "target_std_over_sigma_q10": float(row["target_std_over_sigma"].quantile(0.10)),
            "target_std_over_sigma_q90": float(row["target_std_over_sigma"].quantile(0.90)),
            "target_std_over_CS_median": float(row["target_std_over_CS"].median()),
        }
    else:
        pred_audit = {"warning": "prediction files missing target_used/sigma_used/CS; target sampling not audited"}

    n_reps = int(long[["seed", "pdf_member"]].drop_duplicates().shape[0])
    n_pdf = int(long["pdf_member"].nunique())

    rel_summary = {}
    if not rel.empty:
        rel_summary = (
            rel.groupby("quantity")
            .agg(
                n_curves=("quantity", "size"),
                rel_halfwidth_median=("relative_68_halfwidth_median_active", "median"),
                rel_halfwidth_p90=("relative_68_halfwidth_p90_active", "max"),
                rel_halfwidth_max=("relative_68_halfwidth_max_active", "max"),
            )
            .reset_index()
            .to_dict(orient="records")
        )

    decision = {
        "n_replicas_in_plan": int(len(plan)),
        "n_replicas_in_tmd_long": n_reps,
        "n_unique_pdf_members": n_pdf,
        "replica_count_pass": bool(n_reps >= int(args.min_replicas)),
        "pdf_member_diversity_pass": bool(n_pdf >= int(args.min_unique_pdf_members)),
        "all_tmd_values_finite": bool(np.isfinite(long.select_dtypes(include=[np.number]).to_numpy()).all()),
        "prediction_target_sampling_audit": pred_audit,
        "band_by_quantity": rel_summary,
        "V23A_DATA_PDF_TMD_ENSEMBLE_TECHNICAL_PASS": bool(
            n_reps >= int(args.min_replicas)
            and n_pdf >= int(args.min_unique_pdf_members)
            and np.isfinite(long.select_dtypes(include=[np.number]).to_numpy()).all()
        ),
        "interpretation": (
            "This is a technical gate for the joint data x PDF b-space TMD ensemble. "
            "It checks that multiple PDF members are actually present and that target_used varies. "
            "It does not certify scale/profile/nuclear/model-form uncertainty."
        ),
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "v23a_dataPDF_tmd_ensemble_audit.json").write_text(json.dumps(decision, indent=2) + "\n")
    if not rel.empty:
        rel.to_csv(out / "relative_band_summary.csv", index=False)
    if not pred.empty:
        pred.to_csv(out / "prediction_long_for_target_sampling.csv", index=False)

    print("\n=== v23a data x PDF TMD ensemble audit ===")
    print(json.dumps(decision, indent=2))
    print("\nwrote:", out)


if __name__ == "__main__":
    main()
