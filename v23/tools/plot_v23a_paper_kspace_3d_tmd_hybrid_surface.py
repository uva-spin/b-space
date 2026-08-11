#!/usr/bin/env python3
"""Hybrid smooth-surface/wireframe k_T-space TMDPDF plot.

This presentation-only renderer consumes the interpolated surface table written
by plot_v23a_paper_kspace_3d_tmd.py.  The surface height is the median
regularized k_T-space TMD, the smooth face color is the relative 68% replica
half-width, and the overlaid wire emphasizes the shape.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors


FLAVOR_TEX = {"u": "u", "d": "d", "ubar": r"\bar u", "dbar": r"\bar d"}


def set_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "savefig.dpi": 450,
        }
    )


def grid_from_surface(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    need = ["x", "kT", "median", "relative_68_halfwidth_percent"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"Surface CSV missing columns {missing}; available={list(df.columns)}")

    g = df.copy()
    for col in need:
        g[col] = pd.to_numeric(g[col], errors="coerce")
    g = g.dropna(subset=need)
    g = g.sort_values(["x", "kT"])
    if g.empty:
        raise SystemExit("No finite rows remain in surface table.")

    x = np.array(sorted(g["x"].unique()), dtype=float)
    k = np.array(sorted(g["kT"].unique()), dtype=float)
    if len(x) < 2 or len(k) < 3:
        raise SystemExit("Need at least two x values and three kT values for a 3D surface.")

    med = (
        g.pivot_table(index="x", columns="kT", values="median", aggfunc="mean")
        .reindex(index=x, columns=k)
        .to_numpy(float)
    )
    rel = (
        g.pivot_table(index="x", columns="kT", values="relative_68_halfwidth_percent", aggfunc="mean")
        .reindex(index=x, columns=k)
        .to_numpy(float)
    )
    if not np.isfinite(med).all() or not np.isfinite(rel).all():
        raise SystemExit("Surface table does not form a complete finite x-kT grid.")
    return x, k, med, rel


def z_label(quantity: str, flavor: str) -> str:
    f = FLAVOR_TEX.get(flavor, flavor)
    if quantity == "x_ftilde":
        return rf"$x\,f_1^{{\,{f}}}(x,k_T;Q)\;[\mathrm{{GeV}}^{{-2}}]$"
    return rf"$f_1^{{\,{f}}}(x,k_T;Q)\;[\mathrm{{GeV}}^{{-2}}]$"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--surface-csv", required=True)
    ap.add_argument("--flavor", required=True, choices=["u", "d", "ubar", "dbar"])
    ap.add_argument("--quantity", default="x_ftilde", choices=["x_ftilde", "ftilde"])
    ap.add_argument("--Q", type=float, default=7.5)
    ap.add_argument("--k-max", type=float, default=None)
    ap.add_argument("--uncertainty-vmax", type=float, default=10.0)
    ap.add_argument("--cmap", default="magma")
    ap.add_argument("--view-elev", type=float, default=24.0)
    ap.add_argument("--view-azim", type=float, default=-54.0)
    ap.add_argument("--wire-x-lines", type=int, default=4)
    ap.add_argument("--wire-k-lines", type=int, default=0)
    ap.add_argument("--figure-width", type=float, default=7.35)
    ap.add_argument("--figure-height", type=float, default=5.6)
    ap.add_argument("--title", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.surface_csv)
    if "flavor" in df.columns:
        df = df[df["flavor"].astype(str).eq(args.flavor)]
    if "quantity" in df.columns:
        df = df[df["quantity"].astype(str).eq(args.quantity)]
    if "Q" in df.columns:
        df = df[np.isclose(pd.to_numeric(df["Q"], errors="coerce"), args.Q, rtol=0, atol=1e-10)]
    if args.k_max is not None:
        df = df[pd.to_numeric(df["kT"], errors="coerce") <= float(args.k_max)]

    x, k, med, rel = grid_from_surface(df)
    logx = np.log10(x)
    K, LOGX = np.meshgrid(k, logx)

    set_paper_style()
    cmap = matplotlib.colormaps.get_cmap(args.cmap)
    norm = colors.Normalize(vmin=0.0, vmax=float(args.uncertainty_vmax), clip=True)
    facecolors = cmap(norm(rel))
    facecolors[..., -1] = 0.96

    fig = plt.figure(figsize=(args.figure_width, args.figure_height))
    ax = fig.add_axes([0.02, 0.06, 0.70, 0.86], projection="3d", computed_zorder=False)

    ax.plot_surface(
        K,
        LOGX,
        med,
        facecolors=facecolors,
        linewidth=0.08,
        edgecolor=(1, 1, 1, 0.16),
        antialiased=True,
        shade=True,
        rcount=len(x),
        ccount=len(k),
        zorder=1,
    )

    zspan_for_wire = max(float(np.nanmax(med) - np.nanmin(med)), 1e-12)
    wire_lift = 0.008 * zspan_for_wire

    exact_x_for_wires = np.array([0.1, 0.2, 0.3, 0.5], dtype=float)
    exact_x_for_wires = exact_x_for_wires[
        (exact_x_for_wires >= x.min() - 1e-12) & (exact_x_for_wires <= x.max() + 1e-12)
    ]
    if len(exact_x_for_wires):
        x_indices = np.array(
            [int(np.argmin(np.abs(x - xv))) for xv in exact_x_for_wires],
            dtype=int,
        )
    else:
        x_indices = np.unique(
            np.linspace(0, len(x) - 1, min(args.wire_x_lines, len(x))).round().astype(int)
        )
    if args.wire_x_lines > 0 and len(x_indices) > args.wire_x_lines:
        x_indices = x_indices[: args.wire_x_lines]

    for ix in x_indices:
        ax.plot(
            k,
            np.full_like(k, logx[ix]),
            med[ix] + wire_lift,
            color="0.22",
            lw=1.75,
            alpha=0.88,
            solid_capstyle="round",
            zorder=20,
        )

    if args.wire_k_lines > 0:
        k_indices = np.unique(
            np.linspace(0, len(k) - 1, min(args.wire_k_lines, len(k))).round().astype(int)
        )
        for ik in k_indices:
            ax.plot(
                np.full_like(logx, k[ik]),
                logx,
                med[:, ik] + wire_lift,
                color="0.35",
                lw=0.65,
                alpha=0.50,
                solid_capstyle="round",
                zorder=19,
            )

    zmin = float(np.nanmin(med))
    zmax = float(np.nanmax(med))
    if zmin >= 0.0:
        zmin = 0.0
    zspan = max(zmax - zmin, 1e-12)
    ax.set_zlim(zmin - 0.03 * zspan, zmax + 0.08 * zspan)
    ax.set_xlim(float(k.min()), float(k.max()))
    ax.set_ylim(float(logx.max()), float(logx.min()))

    ax.set_xlabel(r"$k_T\;[\mathrm{GeV}]$", labelpad=10, fontsize=14)
    ax.set_ylabel(r"$x$", labelpad=13, fontsize=14)
    ax.set_zlabel(z_label(args.quantity, args.flavor), labelpad=1, fontsize=10.5)
    exact_x = np.array([0.1, 0.2, 0.3, 0.5], dtype=float)
    exact_x = exact_x[(exact_x >= x.min() - 1e-12) & (exact_x <= x.max() + 1e-12)]
    ax.set_yticks(np.log10(exact_x))
    ax.set_yticklabels([rf"${v:g}$" for v in exact_x], fontsize=10)
    ax.tick_params(axis="x", labelsize=10, pad=1)
    ax.tick_params(axis="z", labelsize=10, pad=2)
    ax.set_box_aspect((1.35, 1.0, 0.78))
    ax.view_init(elev=args.view_elev, azim=args.view_azim)

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor((0.70, 0.70, 0.70, 0.75))
    ax.grid(False)

    flavor_tex = FLAVOR_TEX.get(args.flavor, args.flavor)
    title = args.title or rf"${flavor_tex}$, $Q={args.Q:g}\,\mathrm{{GeV}}$"
    fig.suptitle(title, fontsize=20, y=0.965)

    cax = fig.add_axes([0.805, 0.25, 0.024, 0.48])
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    ticks = np.linspace(0.0, float(args.uncertainty_vmax), 5)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:g}%" for t in ticks])
    cbar.set_label("Relative 68% half-width", fontsize=12, labelpad=11)
    cbar.ax.tick_params(labelsize=10, pad=4)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")

    print(
        {
            "out": str(out),
            "flavor": args.flavor,
            "x_min": float(x.min()),
            "x_max": float(x.max()),
            "k_min": float(k.min()),
            "k_max": float(k.max()),
            "median_min": float(np.nanmin(med)),
            "median_max": float(np.nanmax(med)),
            "relative_68_percent_p90": float(np.nanquantile(rel, 0.90)),
            "relative_68_percent_max": float(np.nanmax(rel)),
        }
    )


if __name__ == "__main__":
    main()
