#!/usr/bin/env python3
"""Show FNP, b-space TMD, and k-space TMD for the exact baseline replay."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
SYSTEMATICS = BASE.parent
UNITARY = SYSTEMATICS / "high_qt_direct_production_benchmark/experimental_unitary_transition"
STATUS_PLOTTER = BASE / "scripts/plot_fig6_model_status_comparison.py"
BASELINE_K = UNITARY / "summaries/fig6_updated_ud_band/fig6_updated_ud_central_1sigma.csv"
TARGET = BASE / "summaries/exact_reproduced_baseline_overview"
COLORS = {"u": "#0072B2", "d": "#D55E00"}


def baseline_bspace():
    spec = importlib.util.spec_from_file_location("baseline_helper", STATUS_PLOTTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.baseline_bspace()


def main() -> None:
    fnp_members = []
    for seed in range(303, 327):
        frame = pd.read_csv(
            BASE / "outputs" / f"exact_original_baseline_replay_with_norms_s{seed}"
            / "fnp_grid.csv")
        frame = frame[np.isclose(frame["x"], .1)].sort_values("bT")
        fnp_members.append(frame["F_NP"].to_numpy(float))
    b_fnp = frame["bT"].to_numpy(float)
    fnp_q = np.quantile(np.asarray(fnp_members), [.16, .50, .84], axis=0)
    bspace = baseline_bspace()
    kframe = pd.read_csv(BASELINE_K)

    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm",
        "axes.linewidth": .9, "xtick.direction": "in",
        "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    })
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)

    mask = b_fnp <= 4
    axes[0].fill_between(b_fnp[mask], fnp_q[0, mask], fnp_q[2, mask],
                         color=".45", alpha=.25, linewidth=0)
    axes[0].plot(b_fnp[mask], fnp_q[1, mask], color="black", lw=1.8)
    axes[0].axvline(2, color=".35", ls=":", lw=.9)
    axes[0].set_title(r"Baseline $F_{\rm NP}$")
    axes[0].set_xlabel(r"$b_T\ (\mathrm{GeV}^{-1})$")
    axes[0].set_ylabel(r"$F_{\rm NP}(x,b_T)$")
    axes[0].text(.96,.94,r"$x=0.1$",ha="right",va="top",transform=axes[0].transAxes)

    for flavor in ("u", "d"):
        color = COLORS[flavor]
        b, q = bspace[flavor]
        selected = b <= 4
        axes[1].fill_between(b[selected], q[0, selected], q[2, selected],
                             color=color, alpha=.18, linewidth=0)
        axes[1].plot(b[selected], q[1, selected], color=color, lw=1.7,
                     label=rf"${flavor}$ quark")
        k = kframe[kframe.flavor.eq(flavor)].sort_values("kT")
        selected = k.kT <= 2.25
        axes[2].fill_between(k.loc[selected,"kT"], k.loc[selected,"q16"],
                             k.loc[selected,"q84"], color=color, alpha=.18,
                             linewidth=0)
        axes[2].plot(k.loc[selected,"kT"], k.loc[selected,"central"],
                     color=color, lw=1.7, label=rf"${flavor}$ quark")
    axes[1].axvline(2, color=".35", ls=":", lw=.9)
    axes[1].set_title(r"Fig. 2 space: $\widetilde f_1^q$")
    axes[1].set_xlabel(r"$b_T\ (\mathrm{GeV}^{-1})$")
    axes[1].set_ylabel(r"$\widetilde f_1^q(x,b_T;Q)$")
    axes[2].set_title(r"Fig. 6 space: $f_1^q$")
    axes[2].set_xlabel(r"$k_T\ (\mathrm{GeV})$")
    axes[2].set_ylabel(r"$f_1^q(x,k_T;Q)$")
    for ax in axes[1:]:
        ax.text(.96,.94,r"$x=0.1,\ Q=10\ \mathrm{GeV}$",ha="right",va="top",transform=ax.transAxes)
        ax.legend(frameon=False, loc="lower left")
    for ax in axes:
        ax.grid(alpha=.17)
        ax.set_ylim(bottom=0)
    fig.suptitle(
        "Exactly reproduced 24-start baseline; shaded TMD bands include experimental residuals",
        fontsize=12)
    TARGET.mkdir(parents=True, exist_ok=True)
    fig.savefig(TARGET / "exact_reproduced_baseline_FNP_Fig2_Fig6.png", dpi=220)
    fig.savefig(TARGET / "exact_reproduced_baseline_FNP_Fig2_Fig6.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
