#!/usr/bin/env python3
"""Audit central v22 scheme-TMD b-space shapes and kT-transform diagnostics.

Inputs are produced by construct_v22_scheme_tmd_grid.py.

This is a diagnostic gate, not a physics theorem.  It separates exact
F_NP-grid points from interpolation/extrapolation status using the chosen
--x-supported-min.  In the current DY-only fit, x=0.1 is an exact F_NP grid
point but is below the data-supported x range, so it is treated as
extrapolative for pass/fail by default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def local_maxima_indices(y: np.ndarray) -> np.ndarray:
    if len(y) < 3:
        return np.array([], dtype=int)
    return np.where((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]))[0] + 1


def safe_div(num: float, den: float) -> float:
    return float(num) / max(abs(float(den)), 1.0e-300)


def audit_bspace_group(group: pd.DataFrame, *, x_supported_min: float) -> dict:
    g = group.sort_values("bT")
    b = g["bT"].to_numpy(float)
    f = g["ftilde"].to_numpy(float)
    F = g["F_NP"].to_numpy(float)

    finite = bool(np.isfinite(b).all() and np.isfinite(f).all() and np.isfinite(F).all())
    absf = np.abs(f)
    peak_idx = int(np.nanargmax(absf))
    peak = float(f[peak_idx])
    abs_peak = float(absf[peak_idx])
    b_peak = float(b[peak_idx])
    endpoint = float(f[-1])
    endpoint_abs_over_peak = safe_div(abs(endpoint), abs_peak)

    maxima = local_maxima_indices(f)
    maxima_after = maxima[maxima > peak_idx]
    secondary_b = np.nan
    secondary_height_ratio = 0.0
    secondary_prominence_ratio = 0.0

    if len(maxima_after) > 0:
        secondary_idx = int(maxima_after[np.argmax(f[maxima_after])])
        secondary_b = float(b[secondary_idx])
        secondary_height_ratio = safe_div(f[secondary_idx], peak)
        valley = float(np.min(f[peak_idx:secondary_idx + 1]))
        secondary_prominence_ratio = safe_div(f[secondary_idx] - valley, peak)

    post = f[peak_idx:]
    post_peak_rebound_ratio = 0.0
    if len(post) >= 3:
        running_min = np.minimum.accumulate(post)
        rebound = np.max(post - running_min)
        post_peak_rebound_ratio = safe_div(rebound, peak)

    F_diff = np.diff(F)
    max_F_increase = float(np.max(F_diff)) if len(F_diff) else 0.0

    x = float(g["x"].iloc[0])
    data_supported = bool(x >= float(x_supported_min) - 1.0e-12)

    return {
        "pid": int(g["pid"].iloc[0]),
        "flavor": str(g["flavor"].iloc[0]),
        "x": x,
        "Q": float(g["Q"].iloc[0]),
        "data_supported_x": data_supported,
        "finite": finite,
        "b_peak": b_peak,
        "ftilde_b0": float(f[0]),
        "ftilde_peak": peak,
        "ftilde_endpoint": endpoint,
        "endpoint_abs_over_peak": endpoint_abs_over_peak,
        "secondary_b": secondary_b,
        "secondary_height_ratio": float(secondary_height_ratio),
        "secondary_prominence_ratio": float(secondary_prominence_ratio),
        "post_peak_rebound_ratio": float(post_peak_rebound_ratio),
        "F_NP_b0": float(F[0]),
        "F_NP_endpoint": float(F[-1]),
        "max_F_NP_increase_step": max_F_increase,
        "F_NP_monotone_nonincreasing": bool(max_F_increase <= 1.0e-6),
        "flag_large_endpoint_tail": bool(endpoint_abs_over_peak > 0.05 and data_supported),
        "flag_resolved_secondary": bool(secondary_prominence_ratio > 0.02 and data_supported),
        "flag_large_rebound": bool(post_peak_rebound_ratio > 0.05 and data_supported),
        "flag_F_NP_nonmonotone": bool(max_F_increase > 1.0e-6 and data_supported),
    }


def audit_kspace_group(group: pd.DataFrame, *, ftilde_b0: float | None, x_supported_min: float) -> dict:
    g = group.sort_values("kT")
    k = g["kT"].to_numpy(float)
    f = g["f_kT"].to_numpy(float)

    finite = bool(np.isfinite(k).all() and np.isfinite(f).all())
    peak = float(np.max(f))
    min_value = float(np.min(f))
    min_over_peak = safe_div(min_value, peak)

    negative = f < 0.0
    negative_point_fraction = float(np.mean(negative)) if len(f) else 0.0

    abs_area = float(np.trapezoid(np.abs(f) * k, x=k))
    neg_area = float(np.trapezoid(np.maximum(-f, 0.0) * k, x=k))
    negative_area_fraction = safe_div(neg_area, abs_area)

    closure_ratio_to_b0 = np.nan
    if ftilde_b0 is not None and np.isfinite(ftilde_b0) and abs(ftilde_b0) > 1.0e-300:
        closure_ratio_to_b0 = float(2.0 * np.pi * np.trapezoid(f * k, x=k) / ftilde_b0)

    x = float(g["x"].iloc[0])
    data_supported = bool(x >= float(x_supported_min) - 1.0e-12)

    severe = bool((min_over_peak < -0.10 or negative_area_fraction > 0.10) and data_supported)
    moderate = bool((min_over_peak < -0.02 or negative_area_fraction > 0.03) and data_supported)

    return {
        "pid": int(g["pid"].iloc[0]),
        "flavor": str(g["flavor"].iloc[0]),
        "x": x,
        "Q": float(g["Q"].iloc[0]),
        "data_supported_x": data_supported,
        "finite": finite,
        "kT_peak": float(k[int(np.argmax(f))]),
        "f_kT_peak": peak,
        "f_kT_min": min_value,
        "min_over_peak": float(min_over_peak),
        "negative_point_fraction": negative_point_fraction,
        "negative_area_fraction": float(negative_area_fraction),
        "closure_ratio_2pi_int_kdk_over_ftilde_b0_kmax_limited": closure_ratio_to_b0,
        "flag_moderate_ringing": moderate,
        "flag_severe_ringing": severe,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-dir", default="plots/v22_scheme_tmd_stage1_s303_exactx")
    parser.add_argument("--x-supported-min", type=float, default=0.15)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    grid = Path(args.grid_dir)
    if not grid.exists():
        raise SystemExit(f"Missing grid directory: {grid}")

    b_path = grid / "v22_scheme_tmd_bspace_long.csv"
    if not b_path.exists():
        raise SystemExit(f"Missing {b_path}")

    if args.out:
        out = Path(args.out)
    else:
        out = grid / "audit"
    out.mkdir(parents=True, exist_ok=True)

    b_long = pd.read_csv(b_path)

    required_b = {"pid", "flavor", "x", "Q", "bT", "ftilde", "F_NP"}
    missing_b = required_b.difference(b_long.columns)
    if missing_b:
        raise SystemExit(f"b-space table missing columns: {sorted(missing_b)}")

    b_rows = []
    ftilde_b0_map = {}
    for key, group in b_long.groupby(["pid", "x", "Q"], observed=False):
        row = audit_bspace_group(group, x_supported_min=float(args.x_supported_min))
        b_rows.append(row)
        ftilde_b0_map[key] = row["ftilde_b0"]

    b_audit = pd.DataFrame(b_rows).sort_values(["Q", "pid", "x"])
    b_audit.to_csv(out / "bspace_shape_audit.csv", index=False)

    k_audit = pd.DataFrame()
    k_path = grid / "v22_scheme_tmd_kspace_long.csv"
    if k_path.exists():
        k_long = pd.read_csv(k_path)
        required_k = {"pid", "flavor", "x", "Q", "kT", "f_kT"}
        missing_k = required_k.difference(k_long.columns)
        if missing_k:
            raise SystemExit(f"k-space table missing columns: {sorted(missing_k)}")

        k_rows = []
        for key, group in k_long.groupby(["pid", "x", "Q"], observed=False):
            k_rows.append(
                audit_kspace_group(
                    group,
                    ftilde_b0=ftilde_b0_map.get(key),
                    x_supported_min=float(args.x_supported_min),
                )
            )
        k_audit = pd.DataFrame(k_rows).sort_values(["Q", "pid", "x"])
        k_audit.to_csv(out / "kspace_transform_audit.csv", index=False)

    supported_b = b_audit[b_audit["data_supported_x"]].copy()
    supported_k = k_audit[k_audit["data_supported_x"]].copy() if not k_audit.empty else pd.DataFrame()

    b_pass = bool(
        supported_b["finite"].all()
        and not supported_b["flag_F_NP_nonmonotone"].any()
        and not supported_b["flag_large_endpoint_tail"].any()
    )

    # Rebounds/secondary features are not immediate vetoes because current
    # b-space curves intentionally include fitted structure. Keep them
    # diagnostic unless very large.
    no_severe_b_topology = bool(
        (supported_b["secondary_prominence_ratio"].max() < 0.10)
        and (supported_b["post_peak_rebound_ratio"].max() < 0.15)
    )

    k_pass = True
    if not supported_k.empty:
        k_pass = bool(
            supported_k["finite"].all()
            and not supported_k["flag_severe_ringing"].any()
        )

    summary = {
        "grid_dir": str(grid),
        "x_supported_min": float(args.x_supported_min),
        "n_bspace_curves": int(len(b_audit)),
        "n_supported_bspace_curves": int(len(supported_b)),
        "bspace_all_finite": bool(b_audit["finite"].all()),
        "supported_max_endpoint_abs_over_peak": (
            float(supported_b["endpoint_abs_over_peak"].max()) if not supported_b.empty else None
        ),
        "supported_max_secondary_prominence_ratio": (
            float(supported_b["secondary_prominence_ratio"].max()) if not supported_b.empty else None
        ),
        "supported_max_post_peak_rebound_ratio": (
            float(supported_b["post_peak_rebound_ratio"].max()) if not supported_b.empty else None
        ),
        "supported_F_NP_monotone_all": (
            bool(supported_b["F_NP_monotone_nonincreasing"].all()) if not supported_b.empty else None
        ),
        "bspace_n_large_endpoint_tail_flags": int(supported_b["flag_large_endpoint_tail"].sum()) if not supported_b.empty else 0,
        "bspace_n_resolved_secondary_flags": int(supported_b["flag_resolved_secondary"].sum()) if not supported_b.empty else 0,
        "bspace_n_large_rebound_flags": int(supported_b["flag_large_rebound"].sum()) if not supported_b.empty else 0,
        "n_kspace_curves": int(len(k_audit)) if not k_audit.empty else 0,
        "n_supported_kspace_curves": int(len(supported_k)) if not supported_k.empty else 0,
        "supported_kspace_worst_min_over_peak": (
            float(supported_k["min_over_peak"].min()) if not supported_k.empty else None
        ),
        "supported_kspace_max_negative_area_fraction": (
            float(supported_k["negative_area_fraction"].max()) if not supported_k.empty else None
        ),
        "kspace_n_moderate_ringing_flags": int(supported_k["flag_moderate_ringing"].sum()) if not supported_k.empty else 0,
        "kspace_n_severe_ringing_flags": int(supported_k["flag_severe_ringing"].sum()) if not supported_k.empty else 0,
        "BSPACE_SHAPE_PASS": b_pass and no_severe_b_topology,
        "KSPACE_DIAGNOSTIC_PASS": k_pass,
        "CENTRAL_TMD_GRID_AUDIT_PASS": bool(b_pass and no_severe_b_topology and k_pass),
        "notes": [
            "x below x_supported_min is audited but excluded from pass/fail.",
            "k-space closure is kmax-limited and should not be interpreted as a full normalization closure.",
            "Negative k-space lobes are diagnostic; positivity in kT is not imposed."
        ],
    }

    (out / "central_tmd_grid_audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("\n=== Central v22 TMD grid audit summary ===")
    print(json.dumps(summary, indent=2))

    print("\n=== Worst b-space endpoint tails ===")
    print(
        b_audit.sort_values("endpoint_abs_over_peak", ascending=False)
        .head(12)
        .to_string(index=False)
    )

    if not k_audit.empty:
        print("\n=== Worst k-space ringing ===")
        print(
            k_audit.sort_values("negative_area_fraction", ascending=False)
            .head(12)
            .to_string(index=False)
        )

    print("\nwrote:", out)

    if not summary["CENTRAL_TMD_GRID_AUDIT_PASS"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
