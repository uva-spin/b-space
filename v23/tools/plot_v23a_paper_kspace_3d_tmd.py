#!/usr/bin/env python3
"""Paper-style 3D regularized k_T-space TMDPDF plot in (x, k_T).

This is the k_T companion to plot_v23a_paper_bspace_3d_tmd.py.  It reads the
regularized finite-b_T Hankel-transform band table produced by

  construct_v23a_regularized_kspace_tmd.py

Expected input:
  <band-dir>/v23a_regularized_kspace_bands.csv

with columns:
  quantity, pid, flavor, x, Q, kT, q16, median, q84

Default figure:
  two panels (u and d quarks), fixed Q=10 GeV,
  x*f_1(x,k_T;Q), exact x support 0.1, 0.2, 0.3, 0.5,
  wireframe color = relative 68% exp+PDF-overlay half-width.

The continuous surface between exact x points is PCHIP-interpolated in log10(x)
for visualization only.  The script does not extrapolate beyond the available
x support unless --allow-x-extrapolation is explicitly supplied.

Small negative k_T lobes are retained by default.  A zero plane is drawn so
their location remains visible.  The relative-uncertainty color uses a
curve-local floor because the median can cross zero in the regularized tail.

Example:

  PYTHONPATH=. python3 v23/tools/plot_v23a_paper_kspace_3d_tmd.py \
    --band-dir replica_v23a_expPDF_overlay_lambda3_normpriors15_p2p5_50rep/kspace_regularized_expPDF_overlay_expb2 \
    --quantity x_ftilde \
    --flavors u d \
    --Q 10 \
    --k-max 3 \
    --x-min 0.10 \
    --x-max 0.50 \
    --out plots/v23a_paper_kspace_3D_xftilde_ud_Q10.pdf
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.interpolate import PchipInterpolator


FLAVOR_TO_PID = {"dbar": -1, "ubar": -2, "d": 1, "u": 2}
FLAVOR_TEX = {"u": "u", "d": "d", "ubar": r"\bar u", "dbar": r"\bar d"}


def find_bands_csv(band_dir: Path, explicit: str | None = None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise SystemExit(f"Requested --bands-csv does not exist: {p}")
        return p

    if band_dir.is_file():
        return band_dir

    preferred = [
        "v23a_regularized_kspace_bands.csv",
        "regularized_kspace_bands.csv",
        "kspace_bands.csv",
    ]
    for name in preferred:
        p = band_dir / name
        if p.exists():
            return p

    candidates = sorted(band_dir.glob("*kspace*bands*.csv"))
    if not candidates:
        candidates = sorted(band_dir.glob("*bands*.csv"))
    if candidates:
        return candidates[0]

    raise SystemExit(f"Could not find a k-space band CSV in {band_dir}")


def round_uncertainty_vmax(value: float) -> float:
    if not np.isfinite(value) or value <= 0:
        return 10.0
    choices = [2, 5, 10, 15, 20, 25, 35, 50, 75, 100, 150, 200, 300, 500]
    for c in choices:
        if value <= c:
            return float(c)
    return float(math.ceil(value / 100.0) * 100.0)


def interpolate_surface(
    sub: pd.DataFrame,
    *,
    k_max: float,
    x_min: float | None,
    x_max: float | None,
    n_k: int,
    n_x: int,
    allow_x_extrapolation: bool,
) -> dict[str, np.ndarray | list[float]]:
    g = sub.copy()
    for c in ["x", "kT", "median", "q16", "q84"]:
        g[c] = pd.to_numeric(g[c], errors="coerce")
    g = g.dropna(subset=["x", "kT", "median", "q16", "q84"])
    g = g[(g["kT"] >= 0) & (g["kT"] <= float(k_max))]
    if g.empty:
        raise ValueError("No finite rows remain after k_T filtering.")

    exact_x = np.array(sorted(g["x"].unique()), dtype=float)
    if len(exact_x) < 2:
        raise ValueError("At least two x values are needed for a 3D x-k_T surface.")

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
            f"[{xmin_data}, {xmax_data}]."
        )

    kmin_data = max(0.0, float(g["kT"].min()))
    kmax_data = min(float(k_max), float(g["kT"].max()))
    k_grid = np.linspace(kmin_data, kmax_data, int(n_k))

    exact_med = np.empty((len(exact_x), len(k_grid)))
    exact_lo = np.empty_like(exact_med)
    exact_hi = np.empty_like(exact_med)

    for ix, xval in enumerate(exact_x):
        gx = g[np.isclose(g["x"], xval, rtol=0, atol=1e-12)].copy()
        gx = (
            gx.groupby("kT", observed=False)[["median", "q16", "q84"]]
            .mean()
            .reset_index()
            .sort_values("kT")
        )
        kvals = gx["kT"].to_numpy(float)
        if len(kvals) < 3:
            raise ValueError(f"Not enough k_T points at x={xval:g}.")

        for arr, col in [(exact_med, "median"), (exact_lo, "q16"), (exact_hi, "q84")]:
            interp = PchipInterpolator(kvals, gx[col].to_numpy(float), extrapolate=False)
            vals = interp(k_grid)
            if np.any(~np.isfinite(vals)):
                raise ValueError(
                    f"Nonfinite k_T interpolation at x={xval:g}; "
                    "check whether all curves cover the requested k_T range."
                )
            arr[ix] = vals

    logx_exact = np.log10(exact_x)
    logx_grid = np.linspace(np.log10(xmin), np.log10(xmax), int(n_x))
    x_grid = 10.0 ** logx_grid

    med = np.empty((len(x_grid), len(k_grid)))
    lo = np.empty_like(med)
    hi = np.empty_like(med)

    for ik in range(len(k_grid)):
        for out, exact in [(med, exact_med), (lo, exact_lo), (hi, exact_hi)]:
            interp = PchipInterpolator(logx_exact, exact[:, ik], extrapolate=allow_x_extrapolation)
            out[:, ik] = interp(logx_grid)

    lo2 = np.minimum(lo, hi)
    hi2 = np.maximum(lo, hi)

    return {
        "k": k_grid,
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


def uncertainty_percent(
    median: np.ndarray,
    q16: np.ndarray,
    q84: np.ndarray,
    *,
    floor_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    half = 0.5 * (q84 - q16)
    peak_by_x = np.nanmax(np.abs(median), axis=1, keepdims=True)
    floor = np.maximum(float(floor_fraction) * peak_by_x, 1e-300)
    active = np.abs(median) >= floor
    rel = 100.0 * half / np.maximum(np.abs(median), floor)
    return rel, active


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
    coll = Line3DCollection(
        segs,
        cmap=cmap,
        norm=norm,
        linewidth=linewidth,
        alpha=alpha,
    )
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
        return rf"$x\,f_1^{{\,{f}}}(x,k_T;Q)\;[\mathrm{{GeV}}^{{-2}}]$"
    if quantity == "ftilde":
        return rf"$f_1^{{\,{f}}}(x,k_T;Q)\;[\mathrm{{GeV}}^{{-2}}]$"
    return quantity


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--band-dir", required=True)
    ap.add_argument("--bands-csv", default=None)
    ap.add_argument("--quantity", default="x_ftilde", choices=["ftilde", "x_ftilde"])
    ap.add_argument("--flavors", nargs="+", default=["u", "d"], choices=["u", "d", "ubar", "dbar"])
    ap.add_argument("--Q", type=float, default=10.0)
    ap.add_argument("--k-max", type=float, default=3.0)
    ap.add_argument("--x-min", type=float, default=None)
    ap.add_argument("--x-max", type=float, default=None)
    ap.add_argument("--n-k", type=int, default=181)
    ap.add_argument("--n-x", type=int, default=41)
    ap.add_argument("--x-ridges", type=int, default=11)
    ap.add_argument("--k-cross-lines", type=int, default=9)
    ap.add_argument("--allow-x-extrapolation", action="store_true")
    ap.add_argument(
        "--uncertainty-floor-frac",
        type=float,
        default=0.05,
        help="Denominator floor as a fraction of each fixed-x curve peak.",
    )
    ap.add_argument("--uncertainty-vmax", type=float, default=None)
    ap.add_argument("--uncertainty-quantile", type=float, default=0.99)
    ap.add_argument("--cmap", default="inferno")
    ap.add_argument("--view-elev", type=float, default=25.0)
    ap.add_argument("--view-azim", type=float, default=-58.0)
    ap.add_argument("--show-band-surfaces", action="store_true")
    ap.add_argument("--no-zero-plane", dest="show_zero_plane", action="store_false")
    ap.set_defaults(show_zero_plane=True)
    ap.add_argument(
        "--show-footer",
        action="store_true",
        help="Draw the methodological footer below the axes. Omit for a cleaner paper figure.",
    )
    ap.add_argument(
        "--figure-width",
        type=float,
        default=None,
        help="Figure width in inches. Default: 9.6 for a single flavor.",
    )
    ap.add_argument(
        "--colorbar-width-ratio",
        type=float,
        default=0.040,
        help="Colorbar column width relative to one 3D panel (multi-panel layout).",
    )
    ap.add_argument(
        "--colorbar-gap",
        type=float,
        default=0.10,
        help="Horizontal GridSpec gap for multi-panel plots.",
    )
    ap.add_argument("--title", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.uncertainty_floor_frac <= 0:
        raise SystemExit("--uncertainty-floor-frac must be positive.")

    band_dir = Path(args.band_dir)
    bands_csv = find_bands_csv(band_dir, args.bands_csv)
    df = pd.read_csv(bands_csv)

    required = ["quantity", "pid", "flavor", "x", "Q", "kT", "q16", "median", "q84"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Band file missing columns {missing}. Available: {list(df.columns)}")

    surfaces: dict[str, dict] = {}
    active_unc_values: list[np.ndarray] = []

    for flavor in args.flavors:
        pid = FLAVOR_TO_PID[flavor]
        m = (
            df["quantity"].astype(str).eq(args.quantity)
            & df["pid"].astype(int).eq(pid)
            & df["flavor"].astype(str).eq(flavor)
            & np.isclose(pd.to_numeric(df["Q"], errors="coerce"), float(args.Q), rtol=0, atol=1e-10)
        )
        sub = df[m].copy()
        if sub.empty:
            raise SystemExit(
                f"No rows found for quantity={args.quantity}, flavor={flavor}, "
                f"pid={pid}, Q={args.Q:g}."
            )

        surf = interpolate_surface(
            sub,
            k_max=args.k_max,
            x_min=args.x_min,
            x_max=args.x_max,
            n_k=args.n_k,
            n_x=args.n_x,
            allow_x_extrapolation=args.allow_x_extrapolation,
        )
        unc, active = uncertainty_percent(
            np.asarray(surf["median"]),
            np.asarray(surf["q16"]),
            np.asarray(surf["q84"]),
            floor_fraction=args.uncertainty_floor_frac,
        )
        surf["uncertainty_percent"] = unc
        surf["uncertainty_active_mask"] = active
        surfaces[flavor] = surf
        active_unc_values.append(unc[active & np.isfinite(unc)])

    unc_concat = np.concatenate(active_unc_values) if active_unc_values else np.array([0.0])
    q = float(np.clip(args.uncertainty_quantile, 0.5, 1.0))
    unc_scale_value = float(np.nanquantile(unc_concat, q))
    vmax = (
        float(args.uncertainty_vmax)
        if args.uncertainty_vmax is not None
        else round_uncertainty_vmax(unc_scale_value)
    )
    vmax = max(vmax, 1e-6)

    cmap = matplotlib.colormaps.get_cmap(args.cmap)
    norm = colors.Normalize(vmin=0.0, vmax=vmax, clip=True)

    set_paper_style()
    n_panels = len(args.flavors)

    # 3D z-axis labels project beyond the nominal axes rectangle.  For a
    # single-flavor paper plot, use explicit axes rectangles and reserve a
    # broad right gutter.  This keeps the colorbar completely separate from
    # the z ticks and TMD axis label in both PDF and PNG output.
    fig_width = (
        float(args.figure_width)
        if args.figure_width is not None
        else (9.6 if n_panels == 1 else 7.0 * n_panels + 1.50)
    )
    fig = plt.figure(figsize=(fig_width, 6.2))
    axes = []

    if n_panels == 1:
        panel_positions = [
            [0.035, 0.105 if not args.show_footer else 0.135, 0.690, 0.805]
        ]
        colorbar_position = [0.75, 0.245, 0.024, 0.525]
        gs = None
    else:
        width_ratios = [1.0] * n_panels + [float(args.colorbar_width_ratio)]
        gs = GridSpec(
            1,
            n_panels + 1,
            figure=fig,
            width_ratios=width_ratios,
            wspace=max(float(args.colorbar_gap), 0.45),
            left=0.025,
            right=0.965,
            bottom=0.10 if not args.show_footer else 0.13,
            top=0.92,
        )
        panel_positions = []
        colorbar_position = None

    for i, flavor in enumerate(args.flavors):
        surf = surfaces[flavor]
        k = np.asarray(surf["k"], dtype=float)
        x = np.asarray(surf["x"], dtype=float)
        logx = np.asarray(surf["logx"], dtype=float)
        med = np.asarray(surf["median"], dtype=float)
        q16 = np.asarray(surf["q16"], dtype=float)
        q84 = np.asarray(surf["q84"], dtype=float)
        unc = np.asarray(surf["uncertainty_percent"], dtype=float)
        exact_x = np.asarray(surf["exact_x"], dtype=float)
        exact_med = np.asarray(surf["exact_median"], dtype=float)

        if n_panels == 1:
            ax = fig.add_axes(panel_positions[i], projection="3d")
        else:
            ax = fig.add_subplot(gs[0, i], projection="3d")
        axes.append(ax)

        ridge_indices = np.unique(
            np.linspace(0, len(x) - 1, min(args.x_ridges, len(x))).round().astype(int)
        )
        for ix in ridge_indices:
            add_colored_polyline(
                ax,
                k,
                np.full_like(k, logx[ix]),
                med[ix],
                unc[ix],
                cmap=cmap,
                norm=norm,
                linewidth=1.8,
                alpha=1.0,
            )

        k_indices = np.unique(
            np.linspace(0, len(k) - 1, min(args.k_cross_lines, len(k))).round().astype(int)
        )
        for ik in k_indices:
            add_colored_polyline(
                ax,
                np.full_like(logx, k[ik]),
                logx,
                med[:, ik],
                unc[:, ik],
                cmap=cmap,
                norm=norm,
                linewidth=1.15,
                alpha=0.92,
            )

        # Exact-x support curves remain visible as thin dark outlines.
        for j, xval in enumerate(exact_x):
            ax.plot(
                k,
                np.full_like(k, np.log10(xval)),
                exact_med[j],
                color="black",
                lw=0.65,
                alpha=0.58,
                zorder=5,
            )

        if args.show_band_surfaces:
            K, X = np.meshgrid(k, logx)
            ax.plot_surface(K, X, q16, color="0.45", alpha=0.055, linewidth=0, shade=False)
            ax.plot_surface(K, X, q84, color="0.45", alpha=0.055, linewidth=0, shade=False)

        zmin = float(np.nanmin(q16))
        zmax = float(np.nanmax(q84))
        if zmin >= 0:
            zmin = 0.0
        zspan = max(zmax - zmin, 1e-12)
        ax.set_zlim(zmin - 0.03 * zspan, zmax + 0.08 * zspan)

        # Zero plane/grid is important because small negative regularized tails are retained.
        if args.show_zero_plane:
            for xval in exact_x:
                ax.plot(
                    [k.min(), k.max()],
                    [np.log10(xval), np.log10(xval)],
                    [0.0, 0.0],
                    color="0.55",
                    lw=0.5,
                    ls="--",
                    alpha=0.60,
                )
            for kval in np.linspace(k.min(), k.max(), 7):
                ax.plot(
                    [kval, kval],
                    [logx.min(), logx.max()],
                    [0.0, 0.0],
                    color="0.68",
                    lw=0.4,
                    ls="--",
                    alpha=0.50,
                )

        ax.set_xlim(k.min(), k.max())
        ax.set_ylim(logx.max(), logx.min())
        ax.set_xlabel(r"$k_T\;[\mathrm{GeV}]$", labelpad=10, fontsize=13)
        ax.set_ylabel(r"$x$", labelpad=13, fontsize=13)
        ax.set_zlabel(z_label(args.quantity, flavor), labelpad=7, fontsize=11.5)
        ax.set_box_aspect((1.35, 1.00, 0.82))

        ax.set_yticks(np.log10(exact_x))
        ax.set_yticklabels([rf"${v:g}$" for v in exact_x], fontsize=9)

        #flavor_tex = FLAVOR_TEX.get(flavor, flavor)
        #ax.set_title(
        #    rf"${flavor_tex}$ quark, $Q={args.Q:g}\,\mathrm{{GeV}}$",
        #    fontsize=15,
        #    pad=13,
        #)
        ax.view_init(elev=args.view_elev, azim=args.view_azim)
        ax.tick_params(axis="x", labelsize=9, pad=1)
        ax.tick_params(axis="z", labelsize=9, pad=2)

        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.pane.set_facecolor((1, 1, 1, 0))
            axis.pane.set_edgecolor((0.65, 0.65, 0.65, 0.7))
        ax.grid(False)

    if args.title:
        fig.suptitle(args.title, fontsize=18, y=0.98)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    if n_panels == 1:
        cax = fig.add_axes(colorbar_position)
    else:
        cax = fig.add_subplot(gs[0, -1])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(r"Relative 68% half-width", fontsize=11, labelpad=10)
    ticks = np.linspace(0, vmax, 5)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:g}%" for t in ticks])
    cbar.ax.yaxis.set_ticks_position("right")
    cbar.ax.yaxis.set_label_position("right")
    cbar.ax.tick_params(labelsize=9, pad=4)

    if args.show_footer:
        fig.text(
            0.46,
            0.025,
            (
                "Regularized finite-$b_T$ Hankel-transform median; "
                r"interpolated in $\log x$ between exact support points. "
                "Color shows the 68% exp+PDF-overlay half-width."
            ),
            ha="center",
            va="bottom",
            fontsize=9.0,
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")

    surface_rows = []
    panel_diag = {}

    for flavor, surf in surfaces.items():
        k = np.asarray(surf["k"], dtype=float)
        x = np.asarray(surf["x"], dtype=float)
        med = np.asarray(surf["median"], dtype=float)
        q16 = np.asarray(surf["q16"], dtype=float)
        q84 = np.asarray(surf["q84"], dtype=float)
        unc = np.asarray(surf["uncertainty_percent"], dtype=float)
        active = np.asarray(surf["uncertainty_active_mask"], dtype=bool)

        for ix, xv in enumerate(x):
            for ik, kv in enumerate(k):
                surface_rows.append(
                    {
                        "quantity": args.quantity,
                        "flavor": flavor,
                        "pid": FLAVOR_TO_PID[flavor],
                        "Q": float(args.Q),
                        "x": float(xv),
                        "kT": float(kv),
                        "median": float(med[ix, ik]),
                        "q16": float(q16[ix, ik]),
                        "q84": float(q84[ix, ik]),
                        "relative_68_halfwidth_percent": float(unc[ix, ik]),
                        "uncertainty_active": bool(active[ix, ik]),
                    }
                )

        active_vals = unc[active & np.isfinite(unc)]
        peak = float(np.nanmax(np.abs(med)))
        panel_diag[flavor] = {
            "exact_x_support": [float(v) for v in surf["exact_x"]],
            "x_plot_min": float(x.min()),
            "x_plot_max": float(x.max()),
            "k_plot_min": float(k.min()),
            "k_plot_max": float(k.max()),
            "median_abs_peak": peak,
            "median_min": float(np.nanmin(med)),
            "min_over_peak": float(np.nanmin(med) / max(peak, 1e-300)),
            "negative_point_fraction": float(np.mean(med < 0)),
            "relative_68_halfwidth_percent_median_active": (
                float(np.nanmedian(active_vals)) if len(active_vals) else None
            ),
            "relative_68_halfwidth_percent_p90_active": (
                float(np.nanquantile(active_vals, 0.90)) if len(active_vals) else None
            ),
            "relative_68_halfwidth_percent_p99_active": (
                float(np.nanquantile(active_vals, 0.99)) if len(active_vals) else None
            ),
            "relative_68_halfwidth_percent_max_active": (
                float(np.nanmax(active_vals)) if len(active_vals) else None
            ),
        }

    pd.DataFrame(surface_rows).to_csv(out.with_suffix(".surface.csv"), index=False)

    diagnostics = {
        "bands_csv": str(bands_csv),
        "quantity": args.quantity,
        "flavors": args.flavors,
        "Q": float(args.Q),
        "k_max": float(args.k_max),
        "x_interpolation": "PCHIP in log10(x), PCHIP in kT",
        "x_extrapolation_used": bool(args.allow_x_extrapolation),
        "negative_values_clipped": False,
        "zero_plane_drawn": bool(args.show_zero_plane),
        "uncertainty_definition": (
            "100*(q84-q16)/(2*max(|median|, floor_frac*fixed-x curve peak))"
        ),
        "uncertainty_floor_fraction": float(args.uncertainty_floor_frac),
        "uncertainty_colorbar_vmax_percent": float(vmax),
        "uncertainty_auto_scale_quantile": q,
        "layout": {
            "figure_width_inches": float(fig_width),
            "figure_height_inches": 6.2,
            "single_panel_explicit_axes": bool(n_panels == 1),
            "dedicated_colorbar_axes": True,
            "colorbar_width_ratio": float(args.colorbar_width_ratio),
            "colorbar_gap": float(args.colorbar_gap),
            "footer_drawn": bool(args.show_footer),
        },
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
