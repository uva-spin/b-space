#!/usr/bin/env python3
"""Paper-style 3D TMDPDF plot in (x, b_T) space.

The default output is a two-panel u/d-quark figure at fixed Q, inspired by the
traditional 3D wireframe representation used in TMD phenomenology:

  horizontal axis  : b_T [GeV^{-1}]
  depth axis       : x (displayed logarithmically)
  vertical axis    : x * ftilde_1^q(x,b_T;Q) by default
  line/surface color: relative 68% half-width in percent

The input is the b-space exp+PDF band table produced in this project, e.g.

  v23a_dataPDF_tmd_replica_bspace_bands.csv

with columns such as

  pid, flavor, x, Q, bT,
  ftilde_median, ftilde_q16, ftilde_q84,
  x_ftilde_median, x_ftilde_q16, x_ftilde_q84.

Important:
  The current fixed-target result has exact x support at x = 0.1, 0.2, 0.3,
  and 0.5.  The smooth surface between those values is an interpolation for
  visualization.  This script does not extrapolate beyond the available x
  support unless --allow-x-extrapolation is explicitly supplied.

Example:

  PYTHONPATH=. python3 v23/tools/plot_v23a_paper_bspace_3d_tmd.py \
    --band-dir replica_v23a_expPDF_overlay_lambda3_normpriors15_p2p5_50rep/tmd_bspace_bands_expPDF_overlay \
    --quantity x_ftilde \
    --flavors u d \
    --Q 10 \
    --b-max 4 \
    --x-min 0.10 \
    --x-max 0.50 \
    --out plots/v23a_paper_bspace_3D_xftilde_ud_Q10.pdf
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.interpolate import PchipInterpolator


FLAVOR_TO_PID = {"dbar": -1, "ubar": -2, "d": 1, "u": 2}
FLAVOR_TEX = {"u": "u", "d": "d", "ubar": r"\bar u", "dbar": r"\bar d"}


def first_col(df: pd.DataFrame, names: Iterable[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for name in names:
        if name in df.columns:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def find_bands_csv(band_dir: Path, explicit: str | None = None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise SystemExit(f"Requested --bands-csv does not exist: {p}")
        return p

    if band_dir.is_file():
        return band_dir

    preferred = [
        "v23a_dataPDF_tmd_replica_bspace_bands.csv",
        "v22_tmd_replica_bspace_bands.csv",
        "v23a_tmd_replica_bspace_bands.csv",
        "tmd_replica_bspace_bands.csv",
    ]
    for name in preferred:
        p = band_dir / name
        if p.exists():
            return p

    candidates = sorted(band_dir.glob("*bspace*bands*.csv"))
    if not candidates:
        candidates = sorted(band_dir.glob("*bands*.csv"))
    if candidates:
        return candidates[0]

    raise SystemExit(f"Could not find a b-space band CSV in {band_dir}")


def quantity_columns(df: pd.DataFrame, quantity: str) -> tuple[str, str, str]:
    med = first_col(df, [f"{quantity}_median", f"{quantity}_q50", f"{quantity}_p50"])
    lo = first_col(df, [f"{quantity}_q16", f"{quantity}_p16", f"{quantity}_lo68"])
    hi = first_col(df, [f"{quantity}_q84", f"{quantity}_p84", f"{quantity}_hi68"])
    if med is None or lo is None or hi is None:
        raise SystemExit(
            f"Could not identify {quantity} median/q16/q84 columns.\n"
            f"Available columns: {list(df.columns)}"
        )
    return med, lo, hi


def round_uncertainty_vmax(value: float) -> float:
    """Choose a readable percentage ceiling without exaggerating precision."""
    if not np.isfinite(value) or value <= 0:
        return 5.0
    choices = [2, 5, 10, 15, 20, 25, 35, 50, 75, 100, 150, 200]
    for c in choices:
        if value <= c:
            return float(c)
    return float(math.ceil(value / 100.0) * 100.0)


def interpolate_surface(
    sub: pd.DataFrame,
    *,
    med_col: str,
    lo_col: str,
    hi_col: str,
    b_max: float,
    x_min: float | None,
    x_max: float | None,
    n_b: int,
    n_x: int,
    allow_x_extrapolation: bool,
) -> dict[str, np.ndarray | list[float]]:
    # Numeric cleanup.
    g = sub.copy()
    for c in ["x", "bT", med_col, lo_col, hi_col]:
        g[c] = pd.to_numeric(g[c], errors="coerce")
    g = g.dropna(subset=["x", "bT", med_col, lo_col, hi_col])
    g = g[(g["bT"] >= 0) & (g["bT"] <= float(b_max))]
    if g.empty:
        raise ValueError("No finite rows remain after b_T filtering.")

    exact_x = np.array(sorted(g["x"].unique()), dtype=float)
    if len(exact_x) < 2:
        raise ValueError("At least two x values are needed for a 3D x-b_T surface.")

    xmin_data = float(exact_x.min())
    xmax_data = float(exact_x.max())
    xmin = xmin_data if x_min is None else float(x_min)
    xmax = xmax_data if x_max is None else float(x_max)

    if xmin <= 0 or xmax <= 0:
        raise ValueError("x must be positive for the logarithmic depth axis.")
    if xmin >= xmax:
        raise ValueError("--x-min must be below --x-max.")
    if not allow_x_extrapolation and (xmin < xmin_data - 1e-12 or xmax > xmax_data + 1e-12):
        raise ValueError(
            f"Requested x range [{xmin}, {xmax}] exceeds available support "
            f"[{xmin_data}, {xmax_data}]. Use --allow-x-extrapolation only if intentional."
        )

    # Common b grid within the data range.
    bmin_data = max(0.0, float(g["bT"].min()))
    bmax_data = min(float(b_max), float(g["bT"].max()))
    b_grid = np.linspace(bmin_data, bmax_data, int(n_b))

    # First interpolate every exact-x curve in b_T.
    exact_med = np.empty((len(exact_x), len(b_grid)))
    exact_lo = np.empty_like(exact_med)
    exact_hi = np.empty_like(exact_med)

    for ix, xval in enumerate(exact_x):
        gx = g[np.isclose(g["x"], xval, rtol=0, atol=1e-12)].copy()
        # Average accidental duplicate b points.
        gx = (
            gx.groupby("bT", observed=False)[[med_col, lo_col, hi_col]]
            .mean()
            .reset_index()
            .sort_values("bT")
        )
        b = gx["bT"].to_numpy(float)
        if len(b) < 3:
            raise ValueError(f"Not enough b_T points at x={xval:g}.")
        for arr, col in [(exact_med, med_col), (exact_lo, lo_col), (exact_hi, hi_col)]:
            interp = PchipInterpolator(b, gx[col].to_numpy(float), extrapolate=False)
            vals = interp(b_grid)
            if np.any(~np.isfinite(vals)):
                raise ValueError(
                    f"Nonfinite b_T interpolation at x={xval:g}; "
                    "check whether all curves share the requested b range."
                )
            arr[ix] = vals

    # Then interpolate each b_T column in log(x).
    logx_exact = np.log10(exact_x)
    logx_grid = np.linspace(np.log10(xmin), np.log10(xmax), int(n_x))
    x_grid = 10.0 ** logx_grid

    med = np.empty((len(x_grid), len(b_grid)))
    lo = np.empty_like(med)
    hi = np.empty_like(med)

    for ib in range(len(b_grid)):
        for out, exact in [(med, exact_med), (lo, exact_lo), (hi, exact_hi)]:
            interp = PchipInterpolator(logx_exact, exact[:, ib], extrapolate=allow_x_extrapolation)
            out[:, ib] = interp(logx_grid)

    # Enforce quantile ordering against tiny interpolation crossings.
    lo2 = np.minimum(lo, hi)
    hi2 = np.maximum(lo, hi)

    return {
        "b": b_grid,
        "x": x_grid,
        "logx": logx_grid,
        "median": med,
        "q16": lo2,
        "q84": hi2,
        "exact_x": exact_x.tolist(),
        "exact_median": exact_med,
        "exact_q16": exact_lo,
        "exact_q84": exact_hi,
    }


def uncertainty_percent(median: np.ndarray, q16: np.ndarray, q84: np.ndarray) -> np.ndarray:
    half = 0.5 * (q84 - q16)
    peak_by_x = np.nanmax(np.abs(median), axis=1, keepdims=True)
    floor = np.maximum(0.01 * peak_by_x, 1e-300)
    return 100.0 * half / np.maximum(np.abs(median), floor)


def add_colored_polyline(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    c: np.ndarray,
    *,
    cmap,
    norm,
    linewidth: float,
    alpha: float = 1.0,
    zorder: int = 3,
) -> None:
    pts = np.column_stack([x, y, z])
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    cseg = 0.5 * (c[:-1] + c[1:])
    coll = Line3DCollection(segs, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
    coll.set_array(cseg)
    coll.set_zorder(zorder)
    ax.add_collection3d(coll)


def set_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "savefig.dpi": 300,
        }
    )


def z_label(quantity: str, flavor: str) -> str:
    f = FLAVOR_TEX.get(flavor, flavor)
    if quantity == "x_ftilde":
        return rf"$x\,\widetilde f_1^{{\,{f}}}(x,b_T;Q)$"
    if quantity == "ftilde":
        return rf"$\widetilde f_1^{{\,{f}}}(x,b_T;Q)$"
    if quantity == "F_NP":
        return r"$F_{\rm NP}(x,b_T)$"
    return quantity


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--band-dir", required=True)
    ap.add_argument("--bands-csv", default=None)
    ap.add_argument("--quantity", default="x_ftilde", choices=["ftilde", "x_ftilde", "F_NP"])
    ap.add_argument("--flavors", nargs="+", default=["u", "d"], choices=["u", "d", "ubar", "dbar"])
    ap.add_argument("--Q", type=float, default=10.0)
    ap.add_argument("--b-max", type=float, default=4.0)
    ap.add_argument("--x-min", type=float, default=None)
    ap.add_argument("--x-max", type=float, default=None)
    ap.add_argument("--n-b", type=int, default=161)
    ap.add_argument("--n-x", type=int, default=41)
    ap.add_argument("--x-ridges", type=int, default=11, help="Number of x-direction ridge curves.")
    ap.add_argument("--b-cross-lines", type=int, default=9, help="Number of cross-lines at fixed b_T.")
    ap.add_argument("--allow-x-extrapolation", action="store_true")
    ap.add_argument("--uncertainty-vmax", type=float, default=None, help="Colorbar maximum in percent. Default: automatic.")
    ap.add_argument("--uncertainty-quantile", type=float, default=0.99, help="Quantile used for automatic color scale.")
    ap.add_argument("--cmap", default="inferno")
    ap.add_argument("--view-elev", type=float, default=25.0)
    ap.add_argument("--view-azim", type=float, default=-58.0)
    ap.add_argument("--show-band-surfaces", action="store_true")
    ap.add_argument("--show-support-note", action="store_true", help="Draw the interpolation/support note below the panels.")
    ap.add_argument("--title", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    band_dir = Path(args.band_dir)
    bands_csv = find_bands_csv(band_dir, args.bands_csv)
    df = pd.read_csv(bands_csv)

    required = ["pid", "flavor", "x", "Q", "bT"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Band file missing columns {missing}. Available: {list(df.columns)}")

    med_col, lo_col, hi_col = quantity_columns(df, args.quantity)

    surfaces: dict[str, dict] = {}
    all_unc: list[np.ndarray] = []

    for flavor in args.flavors:
        pid = FLAVOR_TO_PID[flavor]
        m = (
            df["pid"].astype(int).eq(pid)
            & df["flavor"].astype(str).eq(flavor)
            & np.isclose(pd.to_numeric(df["Q"], errors="coerce"), float(args.Q), rtol=0, atol=1e-10)
        )
        sub = df[m].copy()
        if sub.empty:
            raise SystemExit(f"No rows found for flavor={flavor}, pid={pid}, Q={args.Q:g}.")

        surf = interpolate_surface(
            sub,
            med_col=med_col,
            lo_col=lo_col,
            hi_col=hi_col,
            b_max=args.b_max,
            x_min=args.x_min,
            x_max=args.x_max,
            n_b=args.n_b,
            n_x=args.n_x,
            allow_x_extrapolation=args.allow_x_extrapolation,
        )
        unc = uncertainty_percent(surf["median"], surf["q16"], surf["q84"])
        surf["uncertainty_percent"] = unc
        surfaces[flavor] = surf
        all_unc.append(unc[np.isfinite(unc)])

    unc_concat = np.concatenate(all_unc) if all_unc else np.array([0.0])
    q = float(np.clip(args.uncertainty_quantile, 0.5, 1.0))
    unc_scale_value = float(np.nanquantile(unc_concat, q))
    vmax = float(args.uncertainty_vmax) if args.uncertainty_vmax is not None else round_uncertainty_vmax(unc_scale_value)
    vmax = max(vmax, 1e-6)

    cmap = matplotlib.colormaps.get_cmap(args.cmap)
    norm = colors.Normalize(vmin=0.0, vmax=vmax, clip=True)

    set_paper_style()
    n_panels = len(args.flavors)
    fig = plt.figure(figsize=(7.2 * n_panels, 6.2))
    axes = []

    for i, flavor in enumerate(args.flavors, start=1):
        surf = surfaces[flavor]
        b = np.asarray(surf["b"], dtype=float)
        x = np.asarray(surf["x"], dtype=float)
        logx = np.asarray(surf["logx"], dtype=float)
        med = np.asarray(surf["median"], dtype=float)
        q16 = np.asarray(surf["q16"], dtype=float)
        q84 = np.asarray(surf["q84"], dtype=float)
        unc = np.asarray(surf["uncertainty_percent"], dtype=float)
        exact_x = np.asarray(surf["exact_x"], dtype=float)
        exact_med = np.asarray(surf["exact_median"], dtype=float)

        ax = fig.add_subplot(1, n_panels, i, projection="3d")
        axes.append(ax)

        # Colored median ridges at selected interpolated x values.
        ridge_indices = np.unique(np.linspace(0, len(x) - 1, min(args.x_ridges, len(x))).round().astype(int))
        for ix in ridge_indices:
            add_colored_polyline(
                ax,
                b,
                np.full_like(b, logx[ix]),
                med[ix],
                unc[ix],
                cmap=cmap,
                norm=norm,
                linewidth=1.75,
                alpha=1.0,
            )

        # Cross-lines at selected fixed-b positions.
        b_indices = np.unique(np.linspace(0, len(b) - 1, min(args.b_cross_lines, len(b))).round().astype(int))
        for ib in b_indices:
            add_colored_polyline(
                ax,
                np.full_like(logx, b[ib]),
                logx,
                med[:, ib],
                unc[:, ib],
                cmap=cmap,
                norm=norm,
                linewidth=1.15,
                alpha=0.92,
            )

        # Exact-x support curves as thin dark outlines, so interpolation is visually explicit.
        for j, xval in enumerate(exact_x):
            ax.plot(
                b,
                np.full_like(b, np.log10(xval)),
                exact_med[j],
                color="black",
                lw=0.65,
                alpha=0.55,
                zorder=5,
            )

        if args.show_band_surfaces:
            B, X = np.meshgrid(b, logx)
            ax.plot_surface(B, X, q16, color="0.45", alpha=0.06, linewidth=0, shade=False)
            ax.plot_surface(B, X, q84, color="0.45", alpha=0.06, linewidth=0, shade=False)

        # Floor outline and guide lines.
        zmin = 0.0 if np.nanmin(q16) >= 0 else float(np.nanmin(q16))
        zmax = float(np.nanmax(q84))
        zspan = max(zmax - zmin, 1e-12)
        ax.set_zlim(zmin, zmax + 0.08 * zspan)

        for xval in exact_x:
            ax.plot(
                [b.min(), b.max()],
                [np.log10(xval), np.log10(xval)],
                [zmin, zmin],
                color="0.6",
                lw=0.45,
                ls="--",
                alpha=0.55,
            )
        for bv in np.linspace(b.min(), b.max(), 6):
            ax.plot(
                [bv, bv],
                [logx.min(), logx.max()],
                [zmin, zmin],
                color="0.7",
                lw=0.4,
                ls="--",
                alpha=0.45,
            )

        ax.set_xlim(b.min(), b.max())
        ax.set_ylim(logx.max(), logx.min())  # low x farther/back, similar to the reference figure
        ax.set_xlabel(r"$b_T\;[\mathrm{GeV}^{-1}]$", labelpad=10, fontsize=13)
        ax.set_ylabel(r"$x$", labelpad=13, fontsize=13)
        ax.set_zlabel("")
        ax.text2D(
            0.51,
            0.86,
            z_label(args.quantity, flavor),
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=13,
        )

        # Use exact support points as the y-axis tick labels.
        ax.set_yticks(np.log10(exact_x))
        ax.set_yticklabels([rf"${v:g}$" for v in exact_x], fontsize=9)

        flavor_tex = FLAVOR_TEX.get(flavor, flavor)
        ax.set_title(
            rf"${flavor_tex}$ quark, $Q={args.Q:g}\,\mathrm{{GeV}}$",
            fontsize=15,
            pad=13,
        )
        ax.view_init(elev=args.view_elev, azim=args.view_azim)
        ax.tick_params(axis="x", labelsize=9, pad=1)
        ax.tick_params(axis="z", labelsize=9, pad=2)

        # Transparent panes for a cleaner paper look.
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.pane.set_facecolor((1, 1, 1, 0))
            axis.pane.set_edgecolor((0.65, 0.65, 0.65, 0.7))
        ax.grid(False)

    if args.title:
        fig.suptitle(args.title, fontsize=18, y=0.98)

    # Shared uncertainty colorbar.
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cax = fig.add_axes([0.925, 0.23, 0.016, 0.54])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(r"relative 68\% half-width", fontsize=12, labelpad=8)
    ticks = np.linspace(0, vmax, 5)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:g}%" for t in ticks])
    cbar.ax.tick_params(labelsize=9)

    if args.show_support_note:
        fig.text(
            0.47,
            0.025,
            "Median surface interpolated in log(x) between exact support points; color shows the 68% exp+PDF overlay half-width.",
            ha="center",
            va="bottom",
            fontsize=9.5,
        )

    bottom = 0.09 if args.show_support_note else 0.045
    fig.subplots_adjust(left=0.02, right=0.90, bottom=bottom, top=0.93, wspace=0.00)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")

    # Export the interpolated surfaces for reproducibility.
    surface_rows = []
    panel_diag = {}
    for flavor, surf in surfaces.items():
        b = np.asarray(surf["b"], dtype=float)
        x = np.asarray(surf["x"], dtype=float)
        med = np.asarray(surf["median"], dtype=float)
        q16 = np.asarray(surf["q16"], dtype=float)
        q84 = np.asarray(surf["q84"], dtype=float)
        unc = np.asarray(surf["uncertainty_percent"], dtype=float)
        for ix, xv in enumerate(x):
            for ib, bv in enumerate(b):
                surface_rows.append(
                    {
                        "flavor": flavor,
                        "pid": FLAVOR_TO_PID[flavor],
                        "Q": float(args.Q),
                        "x": float(xv),
                        "bT": float(bv),
                        "median": float(med[ix, ib]),
                        "q16": float(q16[ix, ib]),
                        "q84": float(q84[ix, ib]),
                        "relative_68_halfwidth_percent": float(unc[ix, ib]),
                    }
                )
        panel_diag[flavor] = {
            "exact_x_support": [float(v) for v in surf["exact_x"]],
            "x_plot_min": float(x.min()),
            "x_plot_max": float(x.max()),
            "b_plot_min": float(b.min()),
            "b_plot_max": float(b.max()),
            "median_max": float(np.nanmax(med)),
            "relative_68_halfwidth_percent_median": float(np.nanmedian(unc)),
            "relative_68_halfwidth_percent_p90": float(np.nanquantile(unc, 0.90)),
            "relative_68_halfwidth_percent_p99": float(np.nanquantile(unc, 0.99)),
            "relative_68_halfwidth_percent_max": float(np.nanmax(unc)),
        }

    pd.DataFrame(surface_rows).to_csv(out.with_suffix(".surface.csv"), index=False)

    diagnostics = {
        "bands_csv": str(bands_csv),
        "quantity": args.quantity,
        "flavors": args.flavors,
        "Q": float(args.Q),
        "b_max": float(args.b_max),
        "x_interpolation": "PCHIP in log10(x), PCHIP in bT",
        "x_extrapolation_used": bool(args.allow_x_extrapolation),
        "uncertainty_definition": "100 * (q84-q16)/(2*max(|median|, 0.01*curve_peak))",
        "uncertainty_colorbar_vmax_percent": float(vmax),
        "uncertainty_auto_scale_quantile": q,
        "panels": panel_diag,
        "outputs": {
            "pdf": str(out),
            "png": str(out.with_suffix(".png")),
            "surface_csv": str(out.with_suffix(".surface.csv")),
            "diagnostics_json": str(out.with_suffix(".diagnostics.json")),
        },
    }
    out.with_suffix(".diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")

    print(json.dumps(diagnostics, indent=2))
    print("wrote:", out)
    print("wrote:", out.with_suffix(".png"))
    print("wrote:", out.with_suffix(".surface.csv"))
    print("wrote:", out.with_suffix(".diagnostics.json"))


if __name__ == "__main__":
    main()
