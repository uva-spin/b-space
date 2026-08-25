#!/usr/bin/env python3
"""Audit v22 b-space TMD replica bands.

This audit answers three practical questions:

  1. Did the three-replica pilot create numerically finite b-space bands?
  2. Are the plotted bands meaningful, or are they effectively collapsed onto
     the central curve?
  3. Which requested x values are exact F_NP support points and which are
     interpolation points?

It deliberately does not audit k-space transforms.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def rel_halfwidth(q16: np.ndarray, q84: np.ndarray, med: np.ndarray) -> np.ndarray:
    return 0.5 * (q84 - q16) / np.maximum(np.abs(med), 1.0e-300)


def active_mask(b: np.ndarray, med: np.ndarray, *, frac: float = 0.05) -> np.ndarray:
    scale = np.nanmax(np.abs(med))
    if not np.isfinite(scale) or scale <= 0.0:
        return np.zeros_like(med, dtype=bool)
    return np.abs(med) > frac * scale


def second_peak_metrics(b: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    if len(y) < 5:
        return (np.nan, 0.0, 0.0)

    peak_idx = int(np.nanargmax(y))
    peak = float(y[peak_idx])
    if peak <= 0.0 or not np.isfinite(peak):
        return (np.nan, 0.0, 0.0)

    maxima = np.where((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]))[0] + 1
    maxima_after = maxima[maxima > peak_idx]

    if len(maxima_after) == 0:
        return (np.nan, 0.0, 0.0)

    idx = int(maxima_after[np.argmax(y[maxima_after])])
    valley = float(np.nanmin(y[peak_idx:idx + 1]))
    height_ratio = float(y[idx] / peak)
    prominence_ratio = float((y[idx] - valley) / peak)

    return (float(b[idx]), height_ratio, prominence_ratio)


def summarize_quantity(
    bands: pd.DataFrame,
    central: pd.DataFrame,
    *,
    quantity: str,
    exact_x_values: set[float],
) -> pd.DataFrame:
    rows = []

    med_col = f"{quantity}_median"
    q16_col = f"{quantity}_q16"
    q84_col = f"{quantity}_q84"

    needed = {"pid", "flavor", "x", "Q", "bT", med_col, q16_col, q84_col}
    missing = needed.difference(bands.columns)
    if missing:
        raise SystemExit(f"bands table missing {quantity} columns: {sorted(missing)}")

    for (pid, flavor, x, Q), group in bands.groupby(["pid", "flavor", "x", "Q"], observed=False):
        g = group.sort_values("bT")
        b = g["bT"].to_numpy(float)
        med = g[med_col].to_numpy(float)
        q16 = g[q16_col].to_numpy(float)
        q84 = g[q84_col].to_numpy(float)

        active = active_mask(b, med)
        rel = rel_halfwidth(q16, q84, med)

        c = central[
            (central["pid"].astype(int) == int(pid))
            & np.isclose(central["x"], float(x))
            & np.isclose(central["Q"], float(Q))
        ].sort_values("bT")

        central_vs_median_active_median = np.nan
        central_vs_median_active_p90 = np.nan

        if not c.empty and quantity in c.columns and len(c) == len(g):
            central_values = c[quantity].to_numpy(float)
            diff = np.abs(med - central_values) / np.maximum(np.abs(central_values), 1.0e-300)
            if active.any():
                central_vs_median_active_median = float(np.nanmedian(diff[active]))
                central_vs_median_active_p90 = float(np.nanquantile(diff[active], 0.90))

        b2, h2, p2 = second_peak_metrics(b, med)

        endpoint_over_peak = (
            float(np.abs(med[-1]) / max(np.nanmax(np.abs(med)), 1.0e-300))
            if len(med) else np.nan
        )

        rows.append({
            "quantity": quantity,
            "pid": int(pid),
            "flavor": str(flavor),
            "x": float(x),
            "Q": float(Q),
            "x_exact_support": any(abs(float(x) - v) < 5.0e-10 for v in exact_x_values),
            "finite": bool(np.isfinite(med).all() and np.isfinite(q16).all() and np.isfinite(q84).all()),
            "active_points": int(np.sum(active)),
            "relative_68_halfwidth_median_active": float(np.nanmedian(rel[active])) if active.any() else np.nan,
            "relative_68_halfwidth_p90_active": float(np.nanquantile(rel[active], 0.90)) if active.any() else np.nan,
            "relative_68_halfwidth_max_active": float(np.nanmax(rel[active])) if active.any() else np.nan,
            "central_vs_replica_median_rel_median_active": central_vs_median_active_median,
            "central_vs_replica_median_rel_p90_active": central_vs_median_active_p90,
            "endpoint_abs_over_peak": endpoint_over_peak,
            "secondary_b": b2,
            "secondary_height_ratio": h2,
            "secondary_prominence_ratio": p2,
        })

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--band-dir",
        default="replica_pilot_v22_stage1_profiled/tmd_bspace_bands",
    )
    parser.add_argument(
        "--central-grid",
        default="plots/v22_scheme_tmd_stage1_s303_bandgrid/v22_scheme_tmd_bspace_long.csv",
    )
    parser.add_argument(
        "--exact-x-values",
        nargs="+",
        type=float,
        default=[0.10, 0.20, 0.30, 0.50],
        help="Exact x support of fnp_debug_grid.csv; other x values are interpolation points.",
    )
    parser.add_argument(
        "--min-useful-band-p90",
        type=float,
        default=0.02,
        help="If p90 relative half-width is below this for all curves, call the pilot band collapsed.",
    )
    parser.add_argument(
        "--out",
        default="replica_pilot_v22_stage1_profiled/tmd_bspace_bands/audit",
    )
    args = parser.parse_args()

    band_dir = Path(args.band_dir)
    bands_path = band_dir / "v22_tmd_replica_bspace_bands.csv"
    long_path = band_dir / "v22_tmd_replica_bspace_long.csv"
    central_path = Path(args.central_grid)

    for path in [bands_path, long_path, central_path]:
        if not path.exists():
            raise SystemExit(f"Missing {path}")

    bands = pd.read_csv(bands_path)
    long = pd.read_csv(long_path)
    central = pd.read_csv(central_path)

    exact_x_values = {float(v) for v in args.exact_x_values}

    quantities = [
        "F_NP",
        "ftilde",
        "x_ftilde",
        "b_ftilde",
        "b_x_ftilde",
    ]

    summaries = [
        summarize_quantity(
            bands,
            central,
            quantity=q,
            exact_x_values=exact_x_values,
        )
        for q in quantities
    ]
    summary_by_curve = pd.concat(summaries, ignore_index=True)

    by_quantity = (
        summary_by_curve.groupby("quantity", observed=False)
        .agg(
            n_curves=("x", "size"),
            n_exact_x_curves=("x_exact_support", "sum"),
            finite_all=("finite", "all"),
            rel_halfwidth_median=("relative_68_halfwidth_median_active", "median"),
            rel_halfwidth_p90=("relative_68_halfwidth_p90_active", "median"),
            rel_halfwidth_max=("relative_68_halfwidth_max_active", "max"),
            central_vs_median_rel_p90_max=("central_vs_replica_median_rel_p90_active", "max"),
            endpoint_abs_over_peak_max=("endpoint_abs_over_peak", "max"),
            secondary_prominence_max=("secondary_prominence_ratio", "max"),
        )
        .reset_index()
    )

    interpolation_x = sorted(
        float(x)
        for x in bands["x"].unique()
        if not any(abs(float(x) - v) < 5.0e-10 for v in exact_x_values)
    )

    collapsed = bool(
        by_quantity["rel_halfwidth_max"].max() < float(args.min_useful_band_p90)
    )

    finite_all = bool(summary_by_curve["finite"].all())

    # This pass means "the files are numerically sound"; it does not mean
    # "uncertainty model is complete".
    technical_pass = bool(finite_all)

    summary = {
        "band_dir": str(band_dir),
        "central_grid": str(central_path),
        "n_replicas": int(long["seed"].nunique()) if "seed" in long.columns else None,
        "replica_seeds": sorted([str(v) for v in long["seed"].unique()]) if "seed" in long.columns else [],
        "x_values_in_band_grid": sorted(float(x) for x in bands["x"].unique()),
        "exact_x_values_declared": sorted(exact_x_values),
        "interpolated_x_values": interpolation_x,
        "all_values_finite": finite_all,
        "max_relative_68_halfwidth_active": float(by_quantity["rel_halfwidth_max"].max()),
        "max_central_vs_replica_median_rel_p90_active": float(by_quantity["central_vs_median_rel_p90_max"].max()),
        "bands_collapsed_below_threshold": collapsed,
        "min_useful_band_threshold": float(args.min_useful_band_p90),
        "BSPACE_TMD_BAND_TECHNICAL_PASS": technical_pass,
        "BSPACE_TMD_BAND_UNCERTAINTY_USEFUL_PASS": bool(technical_pass and not collapsed),
        "interpretation": (
            "Technical pass checks finite files only. Uncertainty-useful pass checks that "
            "the 68% bands are not visually/quantitatively collapsed. Very narrow bands "
            "usually mean the replica protocol is dominated by the log-F_NP anchor and "
            "should be treated as a conservative pilot, not a final TMD uncertainty."
        ),
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    summary_by_curve.to_csv(out / "bspace_band_audit_by_curve.csv", index=False)
    by_quantity.to_csv(out / "bspace_band_audit_by_quantity.csv", index=False)
    (out / "bspace_band_audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("\n=== v22 b-space TMD band audit summary ===")
    print(json.dumps(summary, indent=2))

    print("\n=== By quantity ===")
    print(by_quantity.to_string(index=False))

    print("\n=== Widest relative bands ===")
    print(
        summary_by_curve.sort_values("relative_68_halfwidth_p90_active", ascending=False)
        .head(20)
        .to_string(index=False)
    )

    print("\nwrote:", out)

    if not technical_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
