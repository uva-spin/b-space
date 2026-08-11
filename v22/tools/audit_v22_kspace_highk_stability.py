#!/usr/bin/env python3
"""High-k Hankel-transform stability audit for v22 scheme-TMD grids.

This script does NOT impose positivity.  It checks whether the k-space
transform is numerically stable and whether the k-integral approaches the
b-space value at b=0 when the k range is extended.

Input:
  plots/v22_scheme_tmd_stage1_s303_exactx/v22_scheme_tmd_bspace_long.csv

Output:
  kspace_highk_closure_by_curve.csv
  kspace_highk_closure_by_kmax.csv
  kspace_highk_summary.json
  optional kspace_highk_long.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import j0


def transform_curve(
    b: np.ndarray,
    f_b: np.ndarray,
    k: np.ndarray,
) -> np.ndarray:
    # f(k) = 1/(2pi) int_0^inf b db J0(kb) f(b)
    kb = np.outer(k, b)
    integrand = j0(kb) * (b[None, :] * f_b[None, :])
    return np.trapezoid(integrand, x=b, axis=1) / (2.0 * np.pi)


def cumulative_closure(
    k: np.ndarray,
    f_k: np.ndarray,
    f_b0: float,
    K: float,
) -> float:
    mask = k <= float(K) + 1.0e-12
    kk = k[mask]
    ff = f_k[mask]
    if len(kk) < 2 or abs(f_b0) <= 1.0e-300:
        return np.nan
    return float(2.0 * np.pi * np.trapezoid(kk * ff, x=kk) / f_b0)


def negative_metrics(
    k: np.ndarray,
    f_k: np.ndarray,
    K: float,
) -> dict[str, float]:
    mask = k <= float(K) + 1.0e-12
    kk = k[mask]
    ff = f_k[mask]
    if len(kk) < 2:
        return {
            "min_over_peak_kleK": np.nan,
            "negative_point_fraction_kleK": np.nan,
            "negative_area_fraction_kleK": np.nan,
        }

    peak = float(np.max(ff))
    min_value = float(np.min(ff))
    abs_area = float(np.trapezoid(np.abs(ff) * kk, x=kk))
    neg_area = float(np.trapezoid(np.maximum(-ff, 0.0) * kk, x=kk))

    return {
        "min_over_peak_kleK": (
            min_value / max(abs(peak), 1.0e-300)
        ),
        "negative_point_fraction_kleK": float(np.mean(ff < 0.0)),
        "negative_area_fraction_kleK": (
            neg_area / max(abs_area, 1.0e-300)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--grid-dir",
        default="plots/v22_scheme_tmd_stage1_s303_exactx",
    )
    parser.add_argument(
        "--x-supported-min",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--k-max",
        type=float,
        default=60.0,
    )
    parser.add_argument(
        "--n-k",
        type=int,
        default=6001,
    )
    parser.add_argument(
        "--closure-k-values",
        nargs="+",
        type=float,
        default=[4.0, 10.0, 20.0, 40.0, 60.0],
    )
    parser.add_argument(
        "--closure-final-tol",
        type=float,
        default=0.08,
    )
    parser.add_argument(
        "--closure-stability-tol",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--write-long",
        action="store_true",
    )
    parser.add_argument(
        "--out",
        default="",
    )
    args = parser.parse_args()

    grid = Path(args.grid_dir)
    b_path = grid / "v22_scheme_tmd_bspace_long.csv"
    if not b_path.exists():
        raise SystemExit(f"Missing {b_path}")

    out = Path(args.out) if args.out else grid / "audit_highk"
    out.mkdir(parents=True, exist_ok=True)

    b_long = pd.read_csv(b_path)
    required = {"pid", "flavor", "x", "Q", "bT", "ftilde"}
    missing = required.difference(b_long.columns)
    if missing:
        raise SystemExit(f"Missing columns in {b_path}: {sorted(missing)}")

    k = np.linspace(0.0, float(args.k_max), int(args.n_k))
    closure_values = sorted(float(v) for v in args.closure_k_values)
    final_K = max(closure_values)
    penultimate_K = closure_values[-2] if len(closure_values) >= 2 else closure_values[-1]

    curve_rows = []
    closure_rows = []
    long_rows = []

    for (pid, x, Q), group in b_long.groupby(["pid", "x", "Q"], observed=False):
        g = group.sort_values("bT")
        b = g["bT"].to_numpy(float)
        f_b = g["ftilde"].to_numpy(float)
        flavor = str(g["flavor"].iloc[0])

        if not np.all(np.diff(b) > 0.0):
            raise SystemExit(f"b grid is not strictly increasing for pid={pid}, x={x}, Q={Q}")

        f_k = transform_curve(b, f_b, k)
        f_b0 = float(f_b[0])
        data_supported = bool(float(x) >= float(args.x_supported_min) - 1.0e-12)

        closures = {
            f"closure_K{K:g}": cumulative_closure(k, f_k, f_b0, K)
            for K in closure_values
        }

        neg4 = negative_metrics(k, f_k, 4.0)

        final_closure = closures[f"closure_K{final_K:g}"]
        penultimate_closure = closures[f"closure_K{penultimate_K:g}"]

        final_error = abs(final_closure - 1.0) if np.isfinite(final_closure) else np.nan
        stability_error = abs(final_closure - penultimate_closure) if np.isfinite(final_closure) and np.isfinite(penultimate_closure) else np.nan

        row = {
            "pid": int(pid),
            "flavor": flavor,
            "x": float(x),
            "Q": float(Q),
            "data_supported_x": data_supported,
            "finite": bool(np.isfinite(f_k).all()),
            "ftilde_b0": f_b0,
            "f_k0": float(f_k[0]),
            "k_max": float(args.k_max),
            "n_k": int(args.n_k),
            **closures,
            "final_closure_error": float(final_error),
            "closure_stability_error": float(stability_error),
            **neg4,
            "flag_final_closure_bad": bool(final_error > float(args.closure_final_tol) and data_supported),
            "flag_closure_stability_bad": bool(stability_error > float(args.closure_stability_tol) and data_supported),
        }
        curve_rows.append(row)

        for K in closure_values:
            closure_rows.append({
                "pid": int(pid),
                "flavor": flavor,
                "x": float(x),
                "Q": float(Q),
                "data_supported_x": data_supported,
                "K": K,
                "closure_ratio": closures[f"closure_K{K:g}"],
                "closure_error": (
                    abs(closures[f"closure_K{K:g}"] - 1.0)
                    if np.isfinite(closures[f"closure_K{K:g}"])
                    else np.nan
                ),
            })

        if args.write_long:
            # Keep the file moderately sized by writing every point; 32*6001 rows is still okay.
            for kk, ff in zip(k, f_k):
                long_rows.append({
                    "pid": int(pid),
                    "flavor": flavor,
                    "x": float(x),
                    "Q": float(Q),
                    "kT": float(kk),
                    "f_kT": float(ff),
                    "x_f_kT": float(x) * float(ff),
                })

    curves = pd.DataFrame(curve_rows).sort_values(["Q", "pid", "x"])
    closures = pd.DataFrame(closure_rows).sort_values(["Q", "pid", "x", "K"])

    curves.to_csv(out / "kspace_highk_closure_by_curve.csv", index=False)
    closures.to_csv(out / "kspace_highk_closure_by_kmax.csv", index=False)

    if args.write_long:
        pd.DataFrame(long_rows).to_csv(out / "kspace_highk_long.csv", index=False)

    supported = curves[curves["data_supported_x"]].copy()
    pass_numeric = bool(
        not supported.empty
        and supported["finite"].all()
        and not supported["flag_final_closure_bad"].any()
        and not supported["flag_closure_stability_bad"].any()
    )

    summary = {
        "grid_dir": str(grid),
        "x_supported_min": float(args.x_supported_min),
        "k_max": float(args.k_max),
        "n_k": int(args.n_k),
        "closure_k_values": closure_values,
        "n_curves": int(len(curves)),
        "n_supported_curves": int(len(supported)),
        "supported_final_closure_error_median": (
            float(supported["final_closure_error"].median()) if not supported.empty else None
        ),
        "supported_final_closure_error_max": (
            float(supported["final_closure_error"].max()) if not supported.empty else None
        ),
        "supported_closure_stability_error_median": (
            float(supported["closure_stability_error"].median()) if not supported.empty else None
        ),
        "supported_closure_stability_error_max": (
            float(supported["closure_stability_error"].max()) if not supported.empty else None
        ),
        "supported_negative_area_fraction_kle4_max": (
            float(supported["negative_area_fraction_kleK"].max()) if not supported.empty else None
        ),
        "supported_min_over_peak_kle4_min": (
            float(supported["min_over_peak_kleK"].min()) if not supported.empty else None
        ),
        "n_supported_final_closure_flags": (
            int(supported["flag_final_closure_bad"].sum()) if not supported.empty else 0
        ),
        "n_supported_closure_stability_flags": (
            int(supported["flag_closure_stability_bad"].sum()) if not supported.empty else 0
        ),
        "KSPACE_HIGHK_NUMERICAL_PASS": pass_numeric,
        "notes": [
            "Negative k-space values are reported but are not a veto.",
            "The pass/fail is based on high-k closure and closure stability, not positivity.",
            "If closure fails while b-space endpoint tails are tiny, inspect k-grid density and small-b behavior."
        ],
    }

    (out / "kspace_highk_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("\n=== High-k k-space transform summary ===")
    print(json.dumps(summary, indent=2))

    print("\n=== Worst supported final closure errors ===")
    print(
        supported.sort_values("final_closure_error", ascending=False)
        .head(12)
        .to_string(index=False)
    )

    print("\n=== Worst supported k<=4 negative-area diagnostics ===")
    print(
        supported.sort_values("negative_area_fraction_kleK", ascending=False)
        .head(12)
        .to_string(index=False)
    )

    print("\nwrote:", out)

    # Deliberately do not raise SystemExit on failure; this is a diagnostic gate.
    # Downstream scripts should read KSPACE_HIGHK_NUMERICAL_PASS.


if __name__ == "__main__":
    main()
