#!/usr/bin/env python3
"""Render PRD-style Fig. 7/8 surfaces for the isolated refdist-3 candidate.

The completed candidate propagation contains 96 optimizer starts and 50
experimental-replica fits.  Their centered log(F_NP) cross is reconstructed at
the four x knots that are present in every fit diagnostic grid (0.1, 0.2,
0.3, 0.5).  Each crossed member is multiplied by the frozen perturbative
reference at Q=7.5 GeV and transformed with the same regularized finite-b
Hankel settings used for the candidate Fig. 6.  The surface color encodes the
pointwise empirical q16--q84 half-width; it is not a Gaussian confidence
interval.

This is a diagnostic surface for the isolated W+Y candidate.  It never writes
to the frozen production package.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.special import j0


BASE = Path(__file__).resolve().parents[1]
REPORTS = BASE / "reports"
START_ROOT = REPORTS / "scope_329_refdist3_full96x50_long50k_start_s"
REPLICA_ROOT = REPORTS / "scope_329_refdist3_full96x50_long50k_replica_r"
REFERENCE = BASE / "reports/baseline_reference_promoted96_allx.csv"
PERTURBATIVE = (
    BASE.parent / "collins_factorization_validity/plots/"
    "rowidfix_stageFT_all_qmax0p20_lam0p50_central_exactx/"
    "v22_scheme_tmd_bspace_long.csv"
)
TARGET = REPORTS / "scope_329_refdist3_full96x50_long50k_final_fig2_fig6"
STARTS = tuple(range(303, 399))
REPLICAS = tuple(range(1001, 1051))
X_VALUES = np.asarray([0.1, 0.2, 0.3, 0.5], dtype=float)
FLAVORS = ("u", "d")


def lightened_cmap(name: str = "magma", mix: float = 0.30):
    base = matplotlib.colormaps.get_cmap(name)
    points = np.linspace(0.0, 1.0, 256)
    rgba = base(points)
    rgba[:, :3] = rgba[:, :3] + mix * (1.0 - rgba[:, :3])
    return colors.LinearSegmentedColormap.from_list("lightened_magma_candidate", rgba)


def load_fnp(kind: str, members: tuple[int, ...], x: float, b_in: np.ndarray) -> np.ndarray:
    curves = []
    root = START_ROOT if kind == "start" else REPLICA_ROOT
    for member in members:
        suffix = f"{member}"
        path = root.parent / (
            f"scope_329_refdist3_full96x50_long50k_{'start_s' if kind == 'start' else 'replica_r'}{suffix}"
        ) / "fnp_debug_grid.csv"
        frame = pd.read_csv(path)
        frame = frame[np.isclose(frame["x"].to_numpy(float), x)].sort_values("bT")
        if len(frame) < 3:
            raise RuntimeError(f"missing {kind} F_NP curve at x={x:g}: {path}")
        b = frame["bT"].to_numpy(float)
        f = frame["F_NP"].to_numpy(float)
        if not np.all(np.isfinite(f)) or np.any(f <= 0.0):
            raise RuntimeError(f"non-positive/non-finite F_NP in {path}")
        curves.append(np.interp(b_in, b, f))
    return np.asarray(curves, dtype=float)


def crossed_fnp(x: float, b_in: np.ndarray) -> np.ndarray:
    starts = load_fnp("start", STARTS, x, b_in)
    replicas = load_fnp("replica", REPLICAS, x, b_in)
    residuals = np.log(replicas) - np.median(np.log(replicas), axis=0)
    crossed = np.exp(np.log(starts)[:, None, :] + residuals[None, :, :])
    return crossed.reshape(len(STARTS) * len(REPLICAS), len(b_in))


def perturbative_curve(x: float, flavor: str, b_in: np.ndarray) -> np.ndarray:
    ref = pd.read_csv(PERTURBATIVE)
    ref = ref[
        np.isclose(ref["x"].to_numpy(float), x)
        & np.isclose(ref["Q"].to_numpy(float), 7.5)
        & ref["flavor"].astype(str).eq(flavor)
    ].sort_values("bT")
    if len(ref) < 3:
        raise RuntimeError(f"missing Q=7.5 perturbative reference for x={x:g}, flavor={flavor}")
    return np.interp(b_in, ref["bT"].to_numpy(float), ref["ftilde_no_np"].to_numpy(float))


def transform_batch(values: np.ndarray, b_in: np.ndarray, k_grid: np.ndarray) -> np.ndarray:
    """Transform rows of b-space curves using the production regularization."""
    b_grid = np.linspace(0.0, 24.0, 6001)
    interp = PchipInterpolator(b_in, values, axis=1, extrapolate=False)
    y = interp(np.clip(b_grid, b_in[0], b_in[-1]))
    # PCHIP is undefined below the first input point; the production
    # implementation holds that endpoint flat.
    y[:, b_grid < b_in[0]] = values[:, :1]
    # Match construct_v23a_regularized_kspace_tmd_v2.py's exp(-a b^2) tail.
    tail = b_in >= max(0.65 * b_in[-1], b_in[-1] - 2.0)
    z = b_in[tail] ** 2
    z0 = z - z.mean()
    logv = np.log(np.maximum(values[:, tail], 1.0e-300))
    lv0 = logv - logv.mean(axis=1, keepdims=True)
    slope = (lv0 @ z0) / np.maximum(np.sum(z0 * z0), 1.0e-300)
    a = np.maximum(-slope, 1.0e-5)
    beyond = b_grid > b_in[-1]
    if np.any(beyond):
        y[:, beyond] = values[:, -1, None] * np.exp(
            -a[:, None] * (b_grid[beyond][None, :] ** 2 - b_in[-1] ** 2)
        )
    win = np.ones_like(b_grid)
    taper_start = 0.92 * b_grid[-1]
    mask = b_grid > taper_start
    t = (b_grid[mask] - taper_start) / (b_grid[-1] - taper_start)
    win[mask] = 0.5 * (1.0 + np.cos(np.pi * t))
    win[-1] = 0.0
    trap = np.full_like(b_grid, b_grid[1] - b_grid[0])
    trap[[0, -1]] *= 0.5
    quad = b_grid * win * trap / (2.0 * np.pi)
    J = j0(np.outer(k_grid, b_grid))
    # Batch multiplication limits peak memory while retaining exact member
    # quantiles after the transform.
    out = np.empty((values.shape[0], len(k_grid)), dtype=float)
    for i in range(0, values.shape[0], 256):
        out[i:i + 256] = (y[i:i + 256] * quad[None, :]) @ J.T
    return out


def build_surface() -> tuple[dict[str, pd.DataFrame], dict]:
    b_in = np.linspace(0.0001, 8.0, 160)
    k_grid = np.linspace(0.0, 3.0, 181)
    surfaces: dict[str, list[pd.DataFrame]] = {flavor: [] for flavor in FLAVORS}
    for x in X_VALUES:
        fnp = crossed_fnp(float(x), b_in)
        for flavor in FLAVORS:
            p = perturbative_curve(float(x), flavor, b_in)
            transformed = transform_batch(fnp * p[None, :], b_in, k_grid)
            q16, q50, q84 = np.quantile(transformed, [0.16, 0.50, 0.84], axis=0)
            rows = pd.DataFrame({
                "quantity": "x_ftilde",
                "flavor": flavor,
                "pid": 2,
                "Q": 7.5,
                "x": float(x),
                "kT": k_grid,
                "median": float(x) * q50,
                "q16": float(x) * q16,
                "q84": float(x) * q84,
            })
            surfaces[flavor].append(rows)
    out = {flavor: pd.concat(parts, ignore_index=True) for flavor, parts in surfaces.items()}
    for flavor, frame in out.items():
        med = frame.pivot(index="x", columns="kT", values="median").sort_index()
        q16 = frame.pivot(index="x", columns="kT", values="q16").loc[med.index, med.columns]
        q84 = frame.pivot(index="x", columns="kT", values="q84").loc[med.index, med.columns]
        denom = np.maximum(np.abs(med.to_numpy()), 0.05 * np.max(np.abs(med.to_numpy()), axis=1, keepdims=True))
        rel = 100.0 * 0.5 * (q84.to_numpy() - q16.to_numpy()) / denom
        active = med.to_numpy() > 0.05 * np.max(med.to_numpy(), axis=1, keepdims=True)
        frame["relative_68_halfwidth_percent"] = rel.ravel(order="C")
        frame["uncertainty_active"] = active.ravel(order="C")
    return out, {"b_input_points": len(b_in), "k_points": len(k_grid)}


def render(frame: pd.DataFrame, flavor: str, figure: int) -> None:
    xknots = np.sort(frame.x.unique())
    kvals = np.sort(frame.kT.unique())
    med_knots = frame.pivot(index="x", columns="kT", values="median").loc[xknots, kvals].to_numpy()
    rel_knots = frame.pivot(index="x", columns="kT", values="relative_68_halfwidth_percent").loc[xknots, kvals].to_numpy()
    # The fit diagnostics are available at four x knots.  Interpolate only
    # for a smooth display mesh; the accompanying CSV remains on the direct
    # fitted knots and is the authoritative numerical output.
    xvals = np.linspace(float(xknots.min()), float(xknots.max()), 41)
    med = np.stack([np.interp(xvals, xknots, med_knots[:, j]) for j in range(len(kvals))], axis=1)
    rel = np.stack([np.interp(xvals, xknots, rel_knots[:, j]) for j in range(len(kvals))], axis=1)
    signal_fraction = med / np.maximum(np.max(med, axis=1, keepdims=True), 1.0e-30)
    # Fade the color smoothly into the numerically negligible tail rather than
    # drawing a hard active/inactive boundary across the surface.
    color_weight = np.clip((signal_fraction - 0.01) / 0.05, 0.0, 1.0)
    rel_display = rel * color_weight
    log_grid = np.log10(xvals)
    # A fixed robust scale keeps u and d visually comparable.  The numerical
    # CSV still contains the unclipped widths, including the high-x tail.
    vmax = 50.0
    cmap = lightened_cmap(); norm = colors.Normalize(0.0, vmax, clip=True)
    K, LX = np.meshgrid(kvals, log_grid)
    face = cmap(norm(rel_display)); face[..., -1] = 0.96
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 13})
    fig = plt.figure(figsize=(7.5, 5.8))
    ax = fig.add_axes([.02, .06, .70, .86], projection="3d", computed_zorder=False)
    ax.plot_surface(K, LX, med, facecolors=face, linewidth=.08,
                    edgecolor=(1, 1, 1, .16), antialiased=True, shade=False,
                    rcount=len(xvals), ccount=len(kvals))
    lift = .008 * max(float(med.max() - med.min()), 1e-12)
    for xv in xknots:
        ix = int(np.argmin(abs(xvals - xv)))
        ax.plot(kvals, np.full_like(kvals, log_grid[ix]), med[ix] + lift,
                color=".22", lw=1.9, alpha=.9, zorder=20)
    zmin, zmax = float(med.min()), float(med.max()); zspan = max(zmax - zmin, 1e-12)
    ax.set_xlim(0, 3); ax.set_ylim(log_grid.max(), log_grid.min())
    ax.set_zlim(min(0., zmin) - .03 * zspan, zmax + .08 * zspan)
    ax.set_xlabel(r"$k_T\ [\mathrm{GeV}]$", labelpad=11, fontsize=16)
    ax.set_ylabel(r"$x$", labelpad=14, fontsize=16)
    ax.set_zlabel(rf"$x f_1^{{{flavor}}}(x,k_T;Q)\ [\mathrm{{GeV}}^{{-2}}]$", labelpad=2, fontsize=13)
    ax.set_yticks(np.log10(xknots)); ax.set_yticklabels([f"{x:g}" for x in xknots], fontsize=12)
    ax.tick_params(axis="x", labelsize=12); ax.tick_params(axis="z", labelsize=11)
    ax.set_box_aspect((1.35, 1, .78)); ax.view_init(elev=24, azim=-54); ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0)); axis.pane.set_edgecolor((.7, .7, .7, .75))
    ax.set_title(rf"${flavor}$, $Q=7.5\ \mathrm{{GeV}}$", fontsize=21, pad=12)
    cax = fig.add_axes([.805, .24, .027, .50]); sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax); ticks = np.linspace(0, vmax, 5); cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{v:g}%" for v in ticks]); cbar.ax.tick_params(labelsize=11)
    cbar.set_label("Combined relative 68% half-width", fontsize=13, labelpad=11)
    fig.savefig(TARGET / f"fig{figure}.png", dpi=300, bbox_inches="tight")
    fig.savefig(TARGET / f"fig{figure}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    surfaces, transform_meta = build_surface()
    TARGET.mkdir(parents=True, exist_ok=True)
    for flavor, frame in surfaces.items():
        figure = 7 if flavor == "u" else 8
        frame.to_csv(TARGET / f"fig{figure}_refdist3_full96x50_surface.csv", index=False)
        render(frame, flavor, figure)
    summary = {
        "status": "isolated_refdist3_full96x50_surface_figures_not_production",
        "scope": "lambda_ref=3; non-LHCb finite-Y; LHCb retained W-only",
        "Q_GeV": 7.5,
        "x_knots": [float(x) for x in X_VALUES],
        "start_count": len(STARTS), "replica_count": len(REPLICAS),
        "crossed_member_count": len(STARTS) * len(REPLICAS),
        "band": "pointwise empirical q16--q84 of transformed crossed members",
        "surface_quantity": "x * ftilde transformed to x * f_1(x,kT;Q)",
        "uncertainty_color": "relative propagated q16--q84 half-width; not a Gaussian confidence interval",
        "transform": transform_meta,
        "figures": {
            "fig7": str(TARGET / "fig7.png"),
            "fig8": str(TARGET / "fig8.png"),
        },
        "frozen_production_modified": False,
        "promotion_authorized": False,
    }
    (TARGET / "surface_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
