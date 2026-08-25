#!/usr/bin/env python3
"""Plot the isolated external Tevatron N3LL+NNLO W+Y grid diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    grid = pd.read_csv(args.grid)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.0), sharex=False, constrained_layout=True)
    colors = {"CDF_RUN_1": "#0072B2", "CDF_RUN_2": "#D55E00", "D0_RUN_1": "#009E73"}
    labels = {"CDF_RUN_1": "CDF Run I", "CDF_RUN_2": "CDF Run II", "D0_RUN_1": "D0 Run I"}
    for dataset, data in grid.groupby("dataset", sort=False):
        data = data.sort_values("qT_low")
        x = 0.5 * (data.qT_low + data.qT_high)
        xerr = 0.5 * (data.qT_high - data.qT_low)
        c = colors.get(dataset, "0.2")
        axes[0].errorbar(x, data.data_pb_per_GeV, yerr=data.data_unc_pb_per_GeV, xerr=xerr, fmt="o", ms=3.0, lw=0.8, color=c, alpha=0.75, label=f"{labels.get(dataset, dataset)} data")
        axes[0].errorbar(x, data.full_wy_pb_per_GeV, yerr=data.full_wy_unc_pb_per_GeV, fmt="-", lw=1.2, color=c, alpha=0.95, label=f"{labels.get(dataset, dataset)} W+Y")
        axes[1].errorbar(x, data.full_wy_to_data_ratio, yerr=data.full_wy_unc_pb_per_GeV / data.data_pb_per_GeV, fmt="o", ms=3.0, color=c, alpha=0.85)
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$d\sigma/dq_T$ [pb/GeV]")
    axes[0].set_title(r"Unprimed N$^3$LL+NNLO conventional $W+Y$ Tevatron candidate")
    axes[0].grid(alpha=0.2, which="both")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    axes[1].axhline(1.0, color="0.2", lw=1.0)
    axes[1].axhspan(0.95, 1.05, color="0.85", alpha=0.35, zorder=-1, label="±5% guide")
    axes[1].set_xlabel(r"$q_T$ [GeV]")
    axes[1].set_ylabel("W+Y / data")
    axes[1].set_ylim(0, max(1.35, float(np.nanmax(grid.full_wy_to_data_ratio + grid.full_wy_unc_pb_per_GeV / grid.data_pb_per_GeV) * 1.05)))
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(out / "tevatron_n3ll_nnlo_wy_grid_comparison.png", dpi=220)
    fig.savefig(out / "tevatron_n3ll_nnlo_wy_grid_comparison.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
