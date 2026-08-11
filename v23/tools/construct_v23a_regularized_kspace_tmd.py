#!/usr/bin/env python3
"""Regularized Hankel/Fourier-Bessel transform of v23a b-space TMD ensembles.

This constructs a k_T-space companion to the b_T-space fixed-target DY result.
It is designed for b-space ensembles produced by
construct_v23a_data_pdf_bspace_tmd_bands_v2.py, especially the PDF-overlay
ensemble:

  v23a_dataPDF_tmd_replica_bspace_long.csv

The transform convention is

  f(k_T) = 1/(2*pi) int_0^inf db_T b_T J0(k_T b_T) ftilde(b_T)

so that, formally, ftilde(b_T=0) = 2*pi int_0^inf dk_T k_T f(k_T).

Because the available b_T grid is finite, this tool uses an explicit large-b
tail continuation and a smooth end taper.  The output should be treated as a
REGULARIZED k_T representation over a stated k_T range, not as a high-k
perturbative tail calculation.

Recommended first pass:
  PYTHONPATH=. python3 v23/tools/construct_v23a_regularized_kspace_tmd.py \
    --bspace-long replica_v23a_expPDF_overlay_lambda3_normpriors15_p2p5_50rep/tmd_bspace_bands_expPDF_overlay/v23a_dataPDF_tmd_replica_bspace_long.csv \
    --out replica_v23a_expPDF_overlay_lambda3_normpriors15_p2p5_50rep/kspace_regularized_expPDF_overlay_expb2 \
    --quantities ftilde x_ftilde \
    --tail-mode expb2 \
    --b-transform-max 24 \
    --n-b-transform 6001 \
    --k-max 4 \
    --n-k 401 \
    --thin-replicas 10

Run sensitivity checks with --tail-mode expb and --tail-mode taper and compare
the resulting bands before freezing k_T plots.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from scipy.interpolate import PchipInterpolator
from scipy.special import j0
try:
    from numpy import trapezoid as _trapezoid
except Exception:
    from scipy.integrate import trapezoid as _trapezoid
try:
    from numpy import trapezoid as _trapezoid
except Exception:
    from scipy.integrate import trapezoid as _trapezoid


QUANTITY_LABELS = {
    "ftilde": r"$f_1(x,k_T)$",
    "x_ftilde": r"$x\,f_1(x,k_T)$",
    "F_NP": r"${\cal F}[F_{\rm NP}](k_T)$",
}


def parse_float_list(vals: list[str] | None) -> list[float] | None:
    if not vals:
        return None
    out: list[float] = []
    for v in vals:
        for p in str(v).replace(",", " ").split():
            out.append(float(p))
    return out


def parse_int_list(vals: list[str] | None) -> list[int] | None:
    if not vals:
        return None
    out: list[int] = []
    for v in vals:
        for p in str(v).replace(",", " ").split():
            out.append(int(p))
    return out


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "ftilde" in df.columns:
        if "x_ftilde" not in df.columns:
            df["x_ftilde"] = pd.to_numeric(df["x"], errors="coerce") * pd.to_numeric(df["ftilde"], errors="coerce")
        if "b_ftilde" not in df.columns:
            df["b_ftilde"] = pd.to_numeric(df["bT"], errors="coerce") * pd.to_numeric(df["ftilde"], errors="coerce")
        if "b_x_ftilde" not in df.columns:
            df["b_x_ftilde"] = (
                pd.to_numeric(df["bT"], errors="coerce")
                * pd.to_numeric(df["x"], errors="coerce")
                * pd.to_numeric(df["ftilde"], errors="coerce")
            )
    return df


def filter_df(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    if args.x_values:
        xs = parse_float_list(args.x_values)
        mask = np.zeros(len(out), dtype=bool)
        arr = pd.to_numeric(out["x"], errors="coerce").to_numpy(float)
        for x in xs:
            mask |= np.isclose(arr, x, rtol=0, atol=1e-10)
        out = out[mask]
    if args.Q_values:
        Qs = parse_float_list(args.Q_values)
        mask = np.zeros(len(out), dtype=bool)
        arr = pd.to_numeric(out["Q"], errors="coerce").to_numpy(float)
        for Q in Qs:
            mask |= np.isclose(arr, Q, rtol=0, atol=1e-10)
        out = out[mask]
    if args.pids:
        pids = parse_int_list(args.pids)
        out = out[out["pid"].astype(int).isin(pids)]
    if args.flavors:
        out = out[out["flavor"].astype(str).isin(args.flavors)]
    return out


def replica_key(df: pd.DataFrame) -> pd.Series:
    seed = df["seed"].astype(str) if "seed" in df.columns else pd.Series(["unknown"] * len(df), index=df.index)
    pdf = df["pdf_member"].astype(str) if "pdf_member" in df.columns else pd.Series([""] * len(df), index=df.index)
    return seed + "|pdf" + pdf


def trapezoid_weights_uniform(x: np.ndarray) -> np.ndarray:
    if len(x) < 2:
        raise ValueError("need at least two grid points")
    dx = float(x[1] - x[0])
    w = np.full_like(x, dx, dtype=float)
    w[0] *= 0.5
    w[-1] *= 0.5
    return w


def taper_window(b: np.ndarray, start_fraction: float) -> np.ndarray:
    if start_fraction >= 1:
        return np.ones_like(b)
    B = float(np.max(b))
    start = float(start_fraction) * B
    w = np.ones_like(b)
    m = b > start
    if np.any(m):
        t = (b[m] - start) / max(B - start, 1e-300)
        w[m] = 0.5 * (1.0 + np.cos(np.pi * t))
    w[b >= B] = 0.0
    return w


def fit_tail_expb(b: np.ndarray, y: np.ndarray, b_fit_min: float, eps: float) -> float:
    m = (b >= b_fit_min) & np.isfinite(y) & (np.abs(y) > eps)
    if np.sum(m) < 3:
        return 1.0
    yy = np.log(np.maximum(np.abs(y[m]), eps))
    xx = b[m]
    slope, _intercept = np.polyfit(xx, yy, 1)
    lam = max(0.0, -float(slope))
    return max(lam, 1e-4)


def fit_tail_expb2(b: np.ndarray, y: np.ndarray, b_fit_min: float, eps: float) -> float:
    m = (b >= b_fit_min) & np.isfinite(y) & (np.abs(y) > eps)
    if np.sum(m) < 3:
        return 0.10
    yy = np.log(np.maximum(np.abs(y[m]), eps))
    xx = b[m] ** 2
    slope, _intercept = np.polyfit(xx, yy, 1)
    a = max(0.0, -float(slope))
    return max(a, 1e-5)


def extend_curve(
    b_in: np.ndarray,
    y_in: np.ndarray,
    b_grid: np.ndarray,
    tail_mode: str,
    tail_fit_bmin: float | None,
    eps: float,
) -> np.ndarray:
    order = np.argsort(b_in)
    b = np.asarray(b_in[order], dtype=float)
    y = np.asarray(y_in[order], dtype=float)

    good = np.isfinite(b) & np.isfinite(y)
    b = b[good]
    y = y[good]
    if len(b) < 3:
        raise ValueError("not enough b points in curve")

    # Collapse duplicate b values.
    tmp = pd.DataFrame({"b": b, "y": y}).groupby("b", observed=False)["y"].mean().reset_index()
    b = tmp["b"].to_numpy(float)
    y = tmp["y"].to_numpy(float)

    b0 = float(np.min(b))
    bN = float(np.max(b))
    yN = float(y[-1])
    signN = 1.0 if yN >= 0 else -1.0
    yN_abs = max(abs(yN), eps)
    fit_min = float(tail_fit_bmin) if tail_fit_bmin is not None else max(0.65 * bN, bN - 2.0)

    # PCHIP inside the original domain.  Extrapolate below first point flat.
    interp = PchipInterpolator(b, y, extrapolate=False)
    out = interp(np.clip(b_grid, b0, bN))
    out[b_grid < b0] = y[0]

    beyond = b_grid > bN
    if np.any(beyond):
        if tail_mode == "zero":
            out[beyond] = 0.0
        elif tail_mode == "hold":
            out[beyond] = yN
        elif tail_mode == "expb":
            lam = fit_tail_expb(b, y, fit_min, eps)
            out[beyond] = signN * yN_abs * np.exp(-lam * (b_grid[beyond] - bN))
        elif tail_mode == "expb2":
            a = fit_tail_expb2(b, y, fit_min, eps)
            out[beyond] = signN * yN_abs * np.exp(-a * (b_grid[beyond] ** 2 - bN ** 2))
        elif tail_mode == "taper":
            out[beyond] = 0.0
        else:
            raise ValueError(f"unknown tail_mode={tail_mode}")

    # For pure taper mode, taper down to zero over the last part of the original domain too.
    if tail_mode == "taper":
        start = fit_min
        m = b_grid > start
        if np.any(m):
            t = np.clip((b_grid[m] - start) / max(bN - start, 1e-300), 0, 1)
            out[m] *= 0.5 * (1 + np.cos(np.pi * t))
            out[b_grid >= bN] = 0.0

    return np.asarray(out, dtype=float)


def transform_curves(df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    quantities = args.quantities
    missing = [q for q in quantities if q not in df.columns]
    if missing:
        raise SystemExit(f"Missing requested quantities in input long CSV: {missing}")

    b_grid = np.linspace(0.0, float(args.b_transform_max), int(args.n_b_transform))
    k_grid = np.linspace(0.0, float(args.k_max), int(args.n_k))
    win = taper_window(b_grid, float(args.end_taper_start_fraction))
    trap_w = trapezoid_weights_uniform(b_grid)
    quad_w = b_grid * win * trap_w / (2.0 * np.pi)

    # Shared Bessel matrix.  This is the expensive but reusable part.
    J = j0(np.outer(k_grid, b_grid))

    rows = []
    group_cols = ["_replica_key", "seed", "pdf_member", "pid", "flavor", "x", "Q"]
    existing_group_cols = [c for c in group_cols if c in df.columns]

    n_curves = 0
    for key, g in df.groupby(existing_group_cols, observed=False):
        meta = dict(zip(existing_group_cols, key if isinstance(key, tuple) else (key,)))
        b_in = pd.to_numeric(g["bT"], errors="coerce").to_numpy(float)
        for q in quantities:
            y_in = pd.to_numeric(g[q], errors="coerce").to_numpy(float)
            y_ext = extend_curve(
                b_in,
                y_in,
                b_grid,
                tail_mode=args.tail_mode,
                tail_fit_bmin=args.tail_fit_bmin,
                eps=float(args.eps),
            )
            fk = J @ (y_ext * quad_w)
            for k, val in zip(k_grid, fk):
                row = meta.copy()
                row["kT"] = float(k)
                row["quantity"] = q
                row["value"] = float(val)
                rows.append(row)
            n_curves += 1

    out = pd.DataFrame(rows)
    meta = {
        "n_input_groups_times_quantities": int(n_curves),
        "b_transform_max": float(args.b_transform_max),
        "n_b_transform": int(args.n_b_transform),
        "k_max": float(args.k_max),
        "n_k": int(args.n_k),
        "tail_mode": args.tail_mode,
        "tail_fit_bmin": args.tail_fit_bmin,
        "end_taper_start_fraction": float(args.end_taper_start_fraction),
        "transform_convention": "f(k)=1/(2*pi) int db b J0(kb) ftilde(b)",
    }
    return out, meta


def make_bands(k_long: pd.DataFrame) -> pd.DataFrame:
    bands = (
        k_long.groupby(["quantity", "pid", "flavor", "x", "Q", "kT"], observed=False)["value"]
        .quantile([0.16, 0.50, 0.84])
        .unstack()
        .rename(columns={0.16: "q16", 0.50: "median", 0.84: "q84"})
        .reset_index()
    )
    return bands


def audit_bands(bands: pd.DataFrame, b0_table: pd.DataFrame | None, args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    rows = []
    for (quantity, pid, flavor, x, Q), g in bands.groupby(["quantity", "pid", "flavor", "x", "Q"], observed=False):
        g = g.sort_values("kT")
        k = g["kT"].to_numpy(float)
        med = g["median"].to_numpy(float)
        q16 = g["q16"].to_numpy(float)
        q84 = g["q84"].to_numpy(float)
        peak = float(np.nanmax(np.abs(med))) if len(med) else np.nan
        min_val = float(np.nanmin(med)) if len(med) else np.nan
        min_over_peak = min_val / peak if peak and np.isfinite(peak) else np.nan
        neg = med < 0
        neg_point_fraction = float(np.mean(neg)) if len(med) else np.nan
        # Area using absolute negative contribution over absolute total area in k dk measure.
        weights = k
        abs_area = float(_trapezoid(np.abs(med) * weights, k)) if len(k) > 1 else np.nan
        neg_area = float(_trapezoid(np.clip(-med, 0, None) * weights, k)) if len(k) > 1 else np.nan
        neg_area_fraction = neg_area / abs_area if abs_area and np.isfinite(abs_area) else np.nan
        rel_hw = 0.5 * (q84 - q16) / np.maximum(np.abs(med), 1e-300)
        active = np.abs(med) > 0.05 * peak if peak and np.isfinite(peak) else np.zeros_like(med, dtype=bool)

        closure = np.nan
        b0 = np.nan
        if b0_table is not None:
            m = (
                (b0_table["quantity"].astype(str) == str(quantity))
                & (b0_table["pid"].astype(int) == int(pid))
                & np.isclose(pd.to_numeric(b0_table["x"], errors="coerce"), float(x), rtol=0, atol=1e-10)
                & np.isclose(pd.to_numeric(b0_table["Q"], errors="coerce"), float(Q), rtol=0, atol=1e-10)
            )
            if np.any(m):
                b0 = float(b0_table.loc[m, "b0_median"].iloc[0])
                integ = 2.0 * np.pi * float(_trapezoid(k * med, k))
                closure = integ / b0 if b0 != 0 else np.nan

        rows.append({
            "quantity": quantity,
            "pid": int(pid),
            "flavor": flavor,
            "x": float(x),
            "Q": float(Q),
            "kT_min": float(np.nanmin(k)),
            "kT_max": float(np.nanmax(k)),
            "peak_abs_median": peak,
            "min_median": min_val,
            "min_over_peak": min_over_peak,
            "negative_point_fraction": neg_point_fraction,
            "negative_area_fraction": neg_area_fraction,
            "relative_68_halfwidth_median_active": float(np.nanmedian(rel_hw[active])) if np.any(active) else np.nan,
            "relative_68_halfwidth_p90_active": float(np.nanquantile(rel_hw[active], 0.90)) if np.any(active) else np.nan,
            "relative_68_halfwidth_max_active": float(np.nanmax(rel_hw[active])) if np.any(active) else np.nan,
            "bandlimited_closure_ratio_to_b0": closure,
            "b0_median": b0,
        })

    summary = pd.DataFrame(rows)
    decision = {
        "n_curves": int(len(summary)),
        "tail_mode": args.tail_mode,
        "all_values_finite": bool(np.isfinite(bands.select_dtypes(include=[np.number]).to_numpy()).all()),
        "min_over_peak_min": float(summary["min_over_peak"].min()) if len(summary) else np.nan,
        "negative_area_fraction_max": float(summary["negative_area_fraction"].max()) if len(summary) else np.nan,
        "relative_68_halfwidth_p90_max": float(summary["relative_68_halfwidth_p90_active"].max()) if len(summary) else np.nan,
        "REGULARIZED_KT_TECHNICAL_PASS": bool(
            np.isfinite(bands.select_dtypes(include=[np.number]).to_numpy()).all()
            and (len(summary) > 0)
        ),
        "interpretation": (
            "This is a regularized finite-b Hankel transform.  Negative lobes and bandlimited closure "
            "are diagnostics, not automatic vetoes.  Final kT range should be set only after comparing "
            "tail-mode/profile stability."
        ),
    }
    return summary, decision


def b0_from_bspace(df: pd.DataFrame, quantities: list[str]) -> pd.DataFrame:
    rows = []
    for (pid, flavor, x, Q), g in df.groupby(["pid", "flavor", "x", "Q"], observed=False):
        # Use nearest available b to zero for each replica, then median over replicas.
        for q in quantities:
            vals = []
            for _rk, rg in g.groupby("_replica_key", observed=False):
                idx = pd.to_numeric(rg["bT"], errors="coerce").abs().idxmin()
                vals.append(float(rg.loc[idx, q]))
            rows.append({
                "quantity": q,
                "pid": int(pid),
                "flavor": flavor,
                "x": float(x),
                "Q": float(Q),
                "b0_median": float(np.nanmedian(vals)),
            })
    return pd.DataFrame(rows)


def plot_bands(bands: pd.DataFrame, k_long: pd.DataFrame, out: Path, args: argparse.Namespace) -> None:
    for quantity in args.quantities:
        pdf = out / f"{quantity}_kspace_regularized_bands.pdf"
        with PdfPages(pdf) as pages:
            subset = bands[bands["quantity"].eq(quantity)]
            for Q in sorted(subset["Q"].unique()):
                for pid in sorted(subset["pid"].unique()):
                    page = subset[np.isclose(subset["Q"], Q) & subset["pid"].astype(int).eq(int(pid))]
                    if page.empty:
                        continue
                    flavor = str(page["flavor"].iloc[0])
                    fig, ax = plt.subplots(figsize=(8.8, 5.6))
                    for x in sorted(page["x"].unique()):
                        g = page[np.isclose(page["x"], x)].sort_values("kT")
                        line, = ax.plot(g["kT"], g["median"], lw=2.0, label=f"x={x:g}")
                        color = line.get_color()
                        ax.fill_between(g["kT"].to_numpy(float), g["q16"].to_numpy(float), g["q84"].to_numpy(float), color=color, alpha=0.22)
                    ax.axhline(0, color="0.4", lw=0.8)
                    ax.set_title(f"{quantity} regularized kT: {flavor}, Q={Q:g} GeV")
                    ax.set_xlabel(r"$k_T\,[\mathrm{GeV}]$")
                    ax.set_ylabel(QUANTITY_LABELS.get(quantity, quantity))
                    ax.grid(True, alpha=0.28)
                    ax.legend(ncol=2)
                    fig.tight_layout()
                    pages.savefig(fig)
                    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bspace-long", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quantities", nargs="+", default=["ftilde", "x_ftilde"], choices=["ftilde", "x_ftilde", "F_NP"])
    ap.add_argument("--x-values", nargs="*", default=None)
    ap.add_argument("--Q-values", nargs="*", default=None)
    ap.add_argument("--pids", nargs="*", default=None)
    ap.add_argument("--flavors", nargs="*", default=None)
    ap.add_argument("--tail-mode", choices=["expb2", "expb", "taper", "zero", "hold"], default="expb2")
    ap.add_argument("--tail-fit-bmin", type=float, default=None)
    ap.add_argument("--b-transform-max", type=float, default=24.0)
    ap.add_argument("--n-b-transform", type=int, default=6001)
    ap.add_argument("--end-taper-start-fraction", type=float, default=0.92)
    ap.add_argument("--k-max", type=float, default=4.0)
    ap.add_argument("--n-k", type=int, default=401)
    ap.add_argument("--eps", type=float, default=1e-300)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--thin-replicas", type=int, default=0, help="Reserved for later; current plots show bands only.")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.bspace_long)
    df = add_derived(df)
    df = filter_df(df, args)
    if df.empty:
        raise SystemExit("No b-space rows remain after filters.")
    df["_replica_key"] = replica_key(df)

    b0_table = b0_from_bspace(df, args.quantities)
    k_long, meta = transform_curves(df, args)
    bands = make_bands(k_long)
    curve_audit, decision = audit_bands(bands, b0_table, args)

    k_long_path = out / "v23a_regularized_kspace_replica_long.csv"
    band_path = out / "v23a_regularized_kspace_bands.csv"
    audit_path = out / "v23a_regularized_kspace_curve_audit.csv"
    summary_path = out / "v23a_regularized_kspace_summary.json"
    b0_path = out / "bspace_b0_medians.csv"

    k_long.to_csv(k_long_path, index=False)
    bands.to_csv(band_path, index=False)
    curve_audit.to_csv(audit_path, index=False)
    b0_table.to_csv(b0_path, index=False)

    manifest = {
        **meta,
        **decision,
        "input": str(args.bspace_long),
        "out": str(out),
        "quantities": args.quantities,
        "outputs": {
            "long": str(k_long_path),
            "bands": str(band_path),
            "curve_audit": str(audit_path),
            "summary": str(summary_path),
            "b0_medians": str(b0_path),
        },
        "recommended_next_step": (
            "Run at least two alternate tail modes, e.g. expb and taper, then compare stability "
            "before freezing a kT range."
        ),
    }
    summary_path.write_text(json.dumps(manifest, indent=2) + "\n")

    if not args.no_plots:
        plot_bands(bands, k_long, out, args)

    print("\n=== v23a regularized kT TMD constructed ===")
    print(json.dumps({k: manifest[k] for k in [
        "tail_mode", "b_transform_max", "k_max", "n_curves", "all_values_finite",
        "min_over_peak_min", "negative_area_fraction_max", "relative_68_halfwidth_p90_max",
        "REGULARIZED_KT_TECHNICAL_PASS", "interpretation"
    ]}, indent=2))
    print("\nwrote:", out)


if __name__ == "__main__":
    main()
