#!/usr/bin/env python3
"""Make lighter visual aliases for the standardized PRD Fig. 7 and Fig. 8."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import numpy as np
import pandas as pd


OUT = Path.home() / "Downloads" / "prd_lambda1_full96x50_figures_2026-08-11"


def lightened_cmap(name: str = "magma", mix: float = .30):
    base = matplotlib.colormaps.get_cmap(name)
    points = np.linspace(0., 1., 256)
    rgba = base(points)
    rgba[:, :3] = rgba[:, :3] + mix * (1. - rgba[:, :3])
    return colors.LinearSegmentedColormap.from_list("lightened_magma", rgba)


def render(flavor: str, figure: int) -> None:
    path = OUT / f"fig{figure}_lambda1_full96_{flavor}_combined_surface.csv"
    data = pd.read_csv(path)
    xvals = np.sort(data.x.unique())
    kvals = np.sort(data.kT.unique())
    med = data.pivot(index="x", columns="kT", values="median").loc[xvals, kvals].to_numpy()
    q16 = data.pivot(index="x", columns="kT", values="q16").loc[xvals, kvals].to_numpy()
    q84 = data.pivot(index="x", columns="kT", values="q84").loc[xvals, kvals].to_numpy()
    rel = 100. * .5 * (q84 - q16) / np.maximum(np.abs(med), .05 * np.max(np.abs(med), axis=1, keepdims=True))
    log_grid = np.log10(xvals)
    active = data.pivot(index="x", columns="kT", values="uncertainty_active").loc[xvals, kvals].to_numpy(bool)
    vmax = max(5., float(np.ceil(np.quantile(rel[active], .99) / 5.) * 5.))
    cmap = lightened_cmap(); norm = colors.Normalize(0., vmax, clip=True)
    K, LX = np.meshgrid(kvals, log_grid)
    face = cmap(norm(rel)); face[..., -1] = .96
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 13})
    fig = plt.figure(figsize=(7.5, 5.8))
    ax = fig.add_axes([.02, .06, .70, .86], projection="3d", computed_zorder=False)
    ax.plot_surface(K, LX, med, facecolors=face, linewidth=.08,
                    edgecolor=(1, 1, 1, .16), antialiased=True, shade=False,
                    rcount=len(xvals), ccount=len(kvals))
    lift = .008 * max(float(med.max() - med.min()), 1e-12)
    for xv in (.1, .2, .3, .5):
        ix = int(np.argmin(abs(xvals - xv)))
        ax.plot(kvals, np.full_like(kvals, log_grid[ix]), med[ix] + lift,
                color=".22", lw=1.9, alpha=.9, zorder=20)
    zmin, zmax = float(med.min()), float(med.max()); zspan = max(zmax - zmin, 1e-12)
    ax.set_xlim(0, 3); ax.set_ylim(log_grid.max(), log_grid.min())
    ax.set_zlim(min(0., zmin) - .03 * zspan, zmax + .08 * zspan)
    ax.set_xlabel(r"$k_T\ [\mathrm{GeV}]$", labelpad=11, fontsize=16)
    ax.set_ylabel(r"$x$", labelpad=14, fontsize=16)
    ax.set_zlabel(rf"$x f_1^{{{flavor}}}(x,k_T;Q)\ [\mathrm{{GeV}}^{{-2}}]$", labelpad=2, fontsize=13)
    ax.set_yticks(np.log10([.1, .2, .3, .5])); ax.set_yticklabels([r"$0.1$", r"$0.2$", r"$0.3$", r"$0.5$"], fontsize=12)
    ax.tick_params(axis="x", labelsize=12); ax.tick_params(axis="z", labelsize=11)
    ax.set_box_aspect((1.35, 1, .78)); ax.view_init(elev=24, azim=-54); ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0)); axis.pane.set_edgecolor((.7, .7, .7, .75))
    ax.set_title(rf"${flavor}$, $Q=7.5\ \mathrm{{GeV}}$", fontsize=21, pad=12)
    cax = fig.add_axes([.805, .24, .027, .50]); sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax); ticks = np.linspace(0, vmax, 5); cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{v:g}%" for v in ticks]); cbar.ax.tick_params(labelsize=11)
    cbar.set_label("Combined relative 68% half-width", fontsize=13, labelpad=11)
    fig.savefig(OUT / f"fig{figure}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    render("u", 7)
    render("d", 8)
    print("wrote", OUT / "fig7.png", "and", OUT / "fig8.png")


if __name__ == "__main__":
    main()
