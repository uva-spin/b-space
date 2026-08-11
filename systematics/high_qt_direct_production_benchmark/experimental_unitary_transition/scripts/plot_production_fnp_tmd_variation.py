#!/usr/bin/env python3
"""Plot production-only local-start FNP variation and one corresponding TMD."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "systematics/high_qt_direct_production_benchmark/experimental_unitary_transition"
SOURCE = BASE / "summaries/production_fnp_stability_control/fnp_grid_comparison.csv"
CENTRAL_TMD = ROOT / "systematics/collins_factorization_validity/plots/rowidfix_stageFT_all_qmax0p20_lam0p50_central_exactx/v22_scheme_tmd_bspace_long.csv"
TARGET = BASE / "summaries/production_fnp_stability_control/plots"
SEEDS = (303, 304, 305)
COLORS = ("#0072B2", "#D55E00", "#009E73")


def main():
    grid = pd.read_csv(SOURCE)
    tmd_all = pd.read_csv(CENTRAL_TMD)
    tmd = tmd_all[(tmd_all.x == 0.1) & (tmd_all.Q == 5.0) & (tmd_all.flavor == "u")].copy()
    if tmd.empty:
        raise RuntimeError("Representative central TMD curve is missing")

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.4), constrained_layout=True)
    for ax, x_value in zip(axes[0], (0.01, 0.1)):
        subset = grid[np.isclose(grid.x, x_value)].sort_values("bT")
        for seed, color in zip(SEEDS, COLORS):
            ax.plot(subset.bT, subset[f"fnp_s{seed}"], color=color, lw=2.0, label=f"start s{seed}")
        ax.set_title(rf"$F_{{\rm NP}}(x,b_T)$ at $x={x_value:g}$")
        ax.set_xlabel(r"$b_T\ [{\rm GeV}^{-1}]$")
        ax.set_ylabel(r"$F_{\rm NP}$")
        ax.set_xlim(0, 3.0)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.22)
    axes[0, 0].legend(frameon=False, fontsize=9)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylim(1.0e-4, 1.1)
    axes[0, 0].set_title(r"$F_{\rm NP}(x,b_T)$ at $x=0.01$ (log scale)")

    fnp_x01 = grid[np.isclose(grid.x, 0.1)].sort_values("bT")
    tmd_b = tmd.bT.to_numpy(float)
    tmd_curves = {}
    for seed, color in zip(SEEDS, COLORS):
        fnp_interp = np.interp(tmd_b, fnp_x01.bT, fnp_x01[f"fnp_s{seed}"])
        tmd_curves[seed] = tmd.ftilde_no_np.to_numpy(float) * fnp_interp
        axes[1, 0].plot(tmd_b, tmd_curves[seed], color=color, lw=2.0, label=f"start s{seed}")
    axes[1, 0].plot(tmd_b, tmd.ftilde, color="black", lw=1.5, ls="--", label="accepted production")
    axes[1, 0].set_title(r"Example TMD: $\widetilde f_u(x=0.1,b_T;Q=5\,{\rm GeV})$")
    axes[1, 0].set_xlabel(r"$b_T\ [{\rm GeV}^{-1}]$")
    axes[1, 0].set_ylabel(r"$\widetilde f_u$")
    axes[1, 0].set_xlim(0, 3.0)
    axes[1, 0].grid(alpha=0.22)
    axes[1, 0].legend(frameon=False, fontsize=8)

    accepted = tmd.ftilde.to_numpy(float)
    active = np.abs(accepted) > 1.0e-10
    for seed, color in zip(SEEDS, COLORS):
        ratio = np.full_like(accepted, np.nan)
        ratio[active] = tmd_curves[seed][active] / accepted[active] - 1.0
        axes[1, 1].plot(tmd_b, 100.0 * ratio, color=color, lw=2.0, label=f"start s{seed}")
    axes[1, 1].axhline(0.0, color="black", lw=1.0, ls="--")
    axes[1, 1].set_title("Example TMD change from accepted production")
    axes[1, 1].set_xlabel(r"$b_T\ [{\rm GeV}^{-1}]$")
    axes[1, 1].set_ylabel("relative change [%]")
    axes[1, 1].set_xlim(0, 3.0)
    axes[1, 1].grid(alpha=0.22)

    fig.suptitle("Production-only FiLM local-start variation", fontsize=15)
    TARGET.mkdir(parents=True, exist_ok=True)
    fig.savefig(TARGET / "production_fnp_and_tmd_variation.png", dpi=180)
    fig.savefig(TARGET / "production_fnp_and_tmd_variation.pdf")
    plt.close(fig)

    output = tmd[["x", "Q", "pid", "flavor", "bT", "ftilde_no_np", "ftilde"]].copy()
    output = output.rename(columns={"ftilde": "accepted_production_ftilde"})
    for seed in SEEDS:
        output[f"control_s{seed}_ftilde"] = tmd_curves[seed]
    output.to_csv(TARGET / "example_tmd_curves.csv", index=False)
    summary = {
        "fnp_source": str(SOURCE), "perturbative_tmd_source": str(CENTRAL_TMD),
        "tmd_definition": "central production ftilde_no_np multiplied by each control FNP",
        "example": {"x": 0.1, "Q_GeV": 5.0, "flavor": "u", "bT_plot_max_GeV_inverse": 3.0},
        "note": "The TMD ratio to accepted production equals the FNP ratio because the perturbative factor is held fixed.",
    }
    (TARGET / "plot_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
