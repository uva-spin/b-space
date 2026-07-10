#!/usr/bin/env python3
"""Audit and compare a v23a central b-space TMD grid.

Inputs are b-space long CSV files produced by:
  v22/tools/construct_v22_scheme_tmd_grid.py

The script is intentionally version-neutral: it audits any grid with columns
like F_NP, ftilde, x_ftilde, b_ftilde, b_x_ftilde.

It does not inspect kT-space transforms.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


QUANTITY_CANDIDATES = [
    "F_NP",
    "ftilde",
    "x_ftilde",
    "b_ftilde",
    "b_x_ftilde",
    "f_eff",
    "xf_eff",
    "b_f_eff",
    "b_xf_eff",
]


def as_float_set(values: list[float]) -> set[float]:
    return {float(v) for v in values}


def local_shape_metrics(b: np.ndarray, y: np.ndarray) -> dict:
    finite = np.isfinite(b) & np.isfinite(y)
    b = b[finite]
    y = y[finite]

    if len(b) == 0:
        return {
            "finite": False,
            "n_points": 0,
            "peak_b": np.nan,
            "peak_value": np.nan,
            "endpoint_abs_over_peak": np.nan,
            "secondary_b": np.nan,
            "secondary_height_ratio": 0.0,
            "secondary_prominence_ratio": 0.0,
            "post_peak_rebound_ratio": 0.0,
        }

    # Sort just in case.
    order = np.argsort(b)
    b = b[order]
    y = y[order]

    finite_all = bool(np.isfinite(y).all())
    abs_y = np.abs(y)
    peak_idx = int(np.nanargmax(abs_y))
    peak_value = float(abs_y[peak_idx])
    denom = max(peak_value, 1.0e-300)

    endpoint_abs_over_peak = float(abs_y[-1] / denom)

    # After the primary peak, look for a rebound above the valley.
    after_start = min(peak_idx + 2, len(y))
    secondary_b = np.nan
    secondary_height_ratio = 0.0
    secondary_prominence_ratio = 0.0
    post_peak_rebound_ratio = 0.0

    if after_start < len(y):
        y_after = abs_y[after_start:]
        sec_rel_idx = int(np.nanargmax(y_after))
        sec_idx = after_start + sec_rel_idx
        sec_height = float(abs_y[sec_idx])
        valley = float(np.nanmin(abs_y[peak_idx:sec_idx + 1])) if sec_idx > peak_idx else sec_height
        rebound = max(0.0, sec_height - valley)

        secondary_b = float(b[sec_idx])
        secondary_height_ratio = float(sec_height / denom)
        secondary_prominence_ratio = float(rebound / denom)
        post_peak_rebound_ratio = secondary_prominence_ratio

    return {
        "finite": finite_all,
        "n_points": int(len(b)),
        "peak_b": float(b[peak_idx]),
        "peak_value": float(y[peak_idx]),
        "endpoint_abs_over_peak": endpoint_abs_over_peak,
        "secondary_b": secondary_b,
        "secondary_height_ratio": secondary_height_ratio,
        "secondary_prominence_ratio": secondary_prominence_ratio,
        "post_peak_rebound_ratio": post_peak_rebound_ratio,
    }


def audit_grid(
    grid: pd.DataFrame,
    *,
    exact_x_values: set[float],
    endpoint_warn: float,
    secondary_warn: float,
    rebound_warn: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    quantities = [q for q in QUANTITY_CANDIDATES if q in grid.columns]
    if not quantities:
        raise SystemExit("No known TMD quantity columns found in grid.")

    rows = []
    required = ["pid", "flavor", "x", "Q", "bT"]
    missing = [c for c in required if c not in grid.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    for quantity in quantities:
        for (pid, flavor, x, Q), group in grid.groupby(["pid", "flavor", "x", "Q"], observed=False):
            metrics = local_shape_metrics(
                group["bT"].to_numpy(float),
                group[quantity].to_numpy(float),
            )
            x_float = float(x)
            is_exact = any(np.isclose(x_float, xv, rtol=0.0, atol=1.0e-12) for xv in exact_x_values)
            row = {
                "quantity": quantity,
                "pid": int(pid),
                "flavor": str(flavor),
                "x": x_float,
                "Q": float(Q),
                "x_exact": bool(is_exact),
                **metrics,
            }
            row["flag_endpoint"] = bool(row["endpoint_abs_over_peak"] > endpoint_warn)
            row["flag_secondary"] = bool(row["secondary_prominence_ratio"] > secondary_warn)
            row["flag_rebound"] = bool(row["post_peak_rebound_ratio"] > rebound_warn)

            if quantity == "F_NP":
                g = group.sort_values("bT")
                diffs = np.diff(g[quantity].to_numpy(float))
                row["F_NP_max_increase_step"] = float(np.nanmax(diffs)) if len(diffs) else 0.0
                row["F_NP_monotone_nonincreasing"] = bool(np.nanmax(diffs) <= 1.0e-8) if len(diffs) else True
                row["flag_F_NP_nonmonotone"] = not row["F_NP_monotone_nonincreasing"]
            else:
                row["F_NP_max_increase_step"] = np.nan
                row["F_NP_monotone_nonincreasing"] = True
                row["flag_F_NP_nonmonotone"] = False

            rows.append(row)

    per_curve = pd.DataFrame(rows)

    exact = per_curve[per_curve["x_exact"]].copy()
    by_quantity = (
        exact.groupby("quantity", observed=False)
        .agg(
            n_curves=("x", "size"),
            finite_all=("finite", "all"),
            endpoint_abs_over_peak_max=("endpoint_abs_over_peak", "max"),
            secondary_prominence_max=("secondary_prominence_ratio", "max"),
            post_peak_rebound_max=("post_peak_rebound_ratio", "max"),
            n_endpoint_flags=("flag_endpoint", "sum"),
            n_secondary_flags=("flag_secondary", "sum"),
            n_rebound_flags=("flag_rebound", "sum"),
            n_F_NP_nonmonotone_flags=("flag_F_NP_nonmonotone", "sum"),
        )
        .reset_index()
    )

    summary = {
        "n_rows": int(len(grid)),
        "quantities": quantities,
        "x_values_in_grid": sorted(float(x) for x in grid["x"].dropna().unique()),
        "exact_x_values_declared": sorted(float(x) for x in exact_x_values),
        "n_curves_total": int(len(per_curve)),
        "n_curves_exact_x": int(len(exact)),
        "all_values_finite": bool(grid[quantities].replace([np.inf, -np.inf], np.nan).notna().all().all()),
        "exact_endpoint_abs_over_peak_max": float(exact["endpoint_abs_over_peak"].max()) if not exact.empty else None,
        "exact_secondary_prominence_max": float(exact["secondary_prominence_ratio"].max()) if not exact.empty else None,
        "exact_post_peak_rebound_max": float(exact["post_peak_rebound_ratio"].max()) if not exact.empty else None,
        "n_exact_endpoint_flags": int(exact["flag_endpoint"].sum()) if not exact.empty else 0,
        "n_exact_secondary_flags": int(exact["flag_secondary"].sum()) if not exact.empty else 0,
        "n_exact_rebound_flags": int(exact["flag_rebound"].sum()) if not exact.empty else 0,
        "n_exact_F_NP_nonmonotone_flags": int(exact["flag_F_NP_nonmonotone"].sum()) if not exact.empty else 0,
        "endpoint_warn": float(endpoint_warn),
        "secondary_warn": float(secondary_warn),
        "rebound_warn": float(rebound_warn),
    }

    summary["BSPACE_CENTRAL_TECHNICAL_PASS"] = bool(summary["all_values_finite"])
    summary["BSPACE_CENTRAL_SHAPE_PASS"] = bool(
        summary["all_values_finite"]
        and summary["n_exact_endpoint_flags"] == 0
        and summary["n_exact_secondary_flags"] == 0
        and summary["n_exact_rebound_flags"] == 0
        and summary["n_exact_F_NP_nonmonotone_flags"] == 0
    )

    return per_curve, by_quantity, summary


def compare_to_reference(grid: pd.DataFrame, ref: pd.DataFrame, exact_x_values: set[float]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    quantities = [q for q in QUANTITY_CANDIDATES if q in grid.columns and q in ref.columns]
    if not quantities:
        return pd.DataFrame(), pd.DataFrame(), {"comparison_available": False}

    rows = []
    for quantity in quantities:
        lhs = grid[["pid", "flavor", "x", "Q", "bT", quantity]].rename(columns={quantity: "v23a"})
        rhs = ref[["pid", "flavor", "x", "Q", "bT", quantity]].rename(columns={quantity: "ref"})
        merged = lhs.merge(rhs, on=["pid", "flavor", "x", "Q", "bT"], how="inner")
        if merged.empty:
            continue
        merged["quantity"] = quantity
        merged["x_exact"] = merged["x"].map(lambda x: any(np.isclose(float(x), xv, rtol=0.0, atol=1e-12) for xv in exact_x_values))
        merged["rel_delta"] = np.abs(merged["v23a"] - merged["ref"]) / np.maximum(
            np.maximum(np.abs(merged["v23a"]), np.abs(merged["ref"])),
            1.0e-300,
        )
        rows.append(merged)

    if not rows:
        return pd.DataFrame(), pd.DataFrame(), {"comparison_available": False}

    comp = pd.concat(rows, ignore_index=True)
    curve_rows = []
    for (quantity, pid, flavor, x, Q), group in comp[comp["x_exact"]].groupby(["quantity", "pid", "flavor", "x", "Q"], observed=False):
        mid = 0.5 * (np.abs(group["v23a"]) + np.abs(group["ref"]))
        active = mid > 0.05 * max(float(np.nanmax(mid)), 1.0e-300)
        g = group[active]
        curve_rows.append({
            "quantity": quantity,
            "pid": int(pid),
            "flavor": str(flavor),
            "x": float(x),
            "Q": float(Q),
            "active_points": int(len(g)),
            "rel_delta_median_active": float(np.nanmedian(g["rel_delta"])) if len(g) else np.nan,
            "rel_delta_p90_active": float(np.nanquantile(g["rel_delta"], 0.90)) if len(g) else np.nan,
            "rel_delta_max_active": float(np.nanmax(g["rel_delta"])) if len(g) else np.nan,
        })

    curve = pd.DataFrame(curve_rows)
    summary = {
        "comparison_available": True,
        "n_pointwise_common": int(len(comp)),
        "n_curve_common_exact_x": int(len(curve)),
        "rel_delta_p90_active_max": float(curve["rel_delta_p90_active"].max()) if not curve.empty else None,
        "rel_delta_median_active_median": float(curve["rel_delta_median_active"].median()) if not curve.empty else None,
        "note": "Large deltas are not automatic vetoes; this is a v23a-vs-v22 central-shift diagnostic after adding E772.",
    }
    return comp, curve, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", required=True)
    parser.add_argument("--reference-grid", default="")
    parser.add_argument("--exact-x-values", nargs="+", type=float, default=[0.10, 0.20, 0.30, 0.50])
    parser.add_argument("--endpoint-warn", type=float, default=0.12)
    parser.add_argument("--secondary-warn", type=float, default=0.05)
    parser.add_argument("--rebound-warn", type=float, default=0.05)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    grid_path = Path(args.grid)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    grid = pd.read_csv(grid_path)
    exact_x = as_float_set(args.exact_x_values)

    per_curve, by_quantity, summary = audit_grid(
        grid,
        exact_x_values=exact_x,
        endpoint_warn=float(args.endpoint_warn),
        secondary_warn=float(args.secondary_warn),
        rebound_warn=float(args.rebound_warn),
    )

    comparison_summary = {"comparison_available": False}
    if args.reference_grid:
        ref_path = Path(args.reference_grid)
        if ref_path.exists():
            ref = pd.read_csv(ref_path)
            comp_point, comp_curve, comparison_summary = compare_to_reference(grid, ref, exact_x)
            comp_point.to_csv(out / "v23a_vs_reference_pointwise.csv", index=False)
            comp_curve.to_csv(out / "v23a_vs_reference_by_curve.csv", index=False)
        else:
            comparison_summary = {"comparison_available": False, "missing_reference_grid": str(ref_path)}

    summary["grid"] = str(grid_path)
    summary["reference_grid"] = args.reference_grid
    summary["comparison"] = comparison_summary

    per_curve.to_csv(out / "central_bspace_shape_by_curve.csv", index=False)
    by_quantity.to_csv(out / "central_bspace_shape_by_quantity.csv", index=False)
    (out / "central_bspace_shape_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("\n=== v23a central b-space TMD grid audit ===")
    print(json.dumps(summary, indent=2))

    print("\n=== By quantity ===")
    print(by_quantity.to_string(index=False))

    print("\n=== Worst exact-x shape diagnostics ===")
    worst = per_curve[per_curve["x_exact"]].sort_values(
        ["flag_secondary", "secondary_prominence_ratio", "endpoint_abs_over_peak"],
        ascending=[False, False, False],
    )
    show_cols = [
        "quantity", "pid", "flavor", "x", "Q", "peak_b", "endpoint_abs_over_peak",
        "secondary_b", "secondary_height_ratio", "secondary_prominence_ratio",
        "post_peak_rebound_ratio", "flag_endpoint", "flag_secondary", "flag_rebound",
        "flag_F_NP_nonmonotone",
    ]
    show_cols = [c for c in show_cols if c in worst.columns]
    print(worst[show_cols].head(30).to_string(index=False))

    if comparison_summary.get("comparison_available"):
        curve_path = out / "v23a_vs_reference_by_curve.csv"
        curve = pd.read_csv(curve_path)
        print("\n=== Largest v23a-vs-reference b-space shifts ===")
        print(
            curve.sort_values("rel_delta_p90_active", ascending=False)
            .head(30)
            .to_string(index=False)
        )

    print("\nwrote:", out)

    if not summary["BSPACE_CENTRAL_TECHNICAL_PASS"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
