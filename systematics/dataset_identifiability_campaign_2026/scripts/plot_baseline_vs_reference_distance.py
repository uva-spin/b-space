#!/usr/bin/env python3
"""Compare the reproduced baseline with the selected empirical-distance case."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
STATUS_PLOTTER = BASE / "scripts/plot_fig6_model_status_comparison.py"
OLD_K = (
    BASE.parent / "high_qt_direct_production_benchmark/experimental_unitary_transition/"
    "summaries/fig6_updated_ud_band/fig6_updated_ud_central_1sigma.csv")
NEW = BASE / "summaries/matched_baseline_reference_distance_lam1e00_full24_crossed_experimental"
TARGET = BASE / "summaries/baseline_vs_reference_distance_lam1e00"
COLORS = {"u": "#0072B2", "d": "#D55E00"}


def load_status_plotter():
    spec = importlib.util.spec_from_file_location("status_plotter", STATUS_PLOTTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    old_b = load_status_plotter().baseline_bspace()
    old_k_frame = pd.read_csv(OLD_K)
    new_b = pd.read_csv(NEW / "bspace_combined_bands.csv")
    new_k = pd.read_csv(NEW / "kspace_combined_bands.csv")
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.1), sharey="row", constrained_layout=True)
    for col, title in enumerate(("Reproduced baseline", r"Median-distance stabilizer, $\lambda=1$")):
        axes[0, col].set_title(title)
        for flavor in ("u", "d"):
            color = COLORS[flavor]
            if col == 0:
                x, q = old_b[flavor]
                kf = old_k_frame[old_k_frame.flavor.eq(flavor)].sort_values("kT")
                k, kq = kf.kT.to_numpy(), kf[["q16", "central", "q84"]].to_numpy().T
            else:
                bf = new_b[new_b.flavor.eq(flavor)].sort_values("bT")
                x, q = bf.bT.to_numpy(), bf[["q16", "central", "q84"]].to_numpy().T
                kf = new_k[new_k.flavor.eq(flavor)].sort_values("kT")
                k, kq = kf.kT.to_numpy(), kf[["q16", "central", "q84"]].to_numpy().T
            mb, mk = x <= 4, k <= 2.25
            axes[0, col].fill_between(x[mb], q[0, mb], q[2, mb], color=color, alpha=.18)
            axes[0, col].plot(x[mb], q[1, mb], color=color, lw=1.7, label=rf"${flavor}$ quark")
            axes[1, col].fill_between(k[mk], kq[0, mk], kq[2, mk], color=color, alpha=.18)
            axes[1, col].plot(k[mk], kq[1, mk], color=color, lw=1.7)
        axes[0, col].axvline(2, color=".35", ls=":", lw=.9)
        axes[0, col].legend(frameon=False)
        for row in range(2):
            axes[row, col].grid(alpha=.17)
            axes[row, col].set_ylim(bottom=0)
            axes[row, col].text(.97,.96,r"$x=0.1,\ Q=10\ \mathrm{GeV}$",ha="right",va="top",transform=axes[row,col].transAxes)
    axes[0,0].set_ylabel(r"$\widetilde f_1^q(x,b_T;Q)$")
    axes[1,0].set_ylabel(r"$f_1^q(x,k_T;Q)$")
    for ax in axes[0]: ax.set_xlabel(r"$b_T\ (\mathrm{GeV}^{-1})$")
    for ax in axes[1]: ax.set_xlabel(r"$k_T\ (\mathrm{GeV})$")
    fig.suptitle("Combined experimental + non-uniqueness intervals (diagnostic)")
    TARGET.mkdir(parents=True, exist_ok=True)
    fig.savefig(TARGET / "baseline_vs_reference_distance.png", dpi=220)
    fig.savefig(TARGET / "baseline_vs_reference_distance.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
