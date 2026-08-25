#!/usr/bin/env python3
"""Plot the isolated seven-point perturbative scale envelope."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {"CDF_RUN_1": "#0072B2", "CDF_RUN_2": "#D55E00", "D0_RUN_1": "#009E73"}
LABELS = {"CDF_RUN_1": "CDF Run I", "CDF_RUN_2": "CDF Run II", "D0_RUN_1": "D0 Run I"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data = pd.read_csv(args.csv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    central = data[(data.scale_muR == 1.0) & (data.scale_muF == 1.0)].set_index("row_id")
    wide = data.pivot(index="row_id", columns=["scale_muR", "scale_muF"], values="wy_pb_per_GeV").loc[central.index]
    low, high = wide.min(axis=1), wide.max(axis=1)
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.0), sharex=False, constrained_layout=True)
    for dataset, group in central.reset_index().groupby("dataset", sort=False):
        group = group.sort_values("qT_low").set_index("row_id")
        x = 0.5 * (group.qT_low + group.qT_high)
        ids = group.index
        c = COLORS.get(dataset, "0.2")
        label = LABELS.get(dataset, dataset)
        axes[0].fill_between(x, low.loc[ids], high.loc[ids], color=c, alpha=0.18)
        axes[0].plot(x, central.loc[ids, "wy_pb_per_GeV"], color=c, lw=1.2, label=label)
        axes[1].fill_between(x, low.loc[ids] / central.loc[ids, "wy_pb_per_GeV"],
                             high.loc[ids] / central.loc[ids, "wy_pb_per_GeV"], color=c, alpha=0.18)
        axes[1].plot(x, central.loc[ids, "wy_pb_per_GeV"] / group.CS, color=c, lw=1.0)
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$d\sigma/dq_T$ [pb/GeV]")
    axes[0].set_title(r"Unprimed N$^3$LL+NNLO Tevatron $W+Y$: scale envelope")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(alpha=0.2, which="both")
    axes[1].axhline(1.0, color="0.2", lw=1.0)
    axes[1].set_xlabel(r"$q_T$ [GeV]")
    axes[1].set_ylabel("central W+Y / data\n(shaded: 7-point scale envelope / central)")
    axes[1].set_ylim(0, max(1.35, float(np.nanmax(high / central.wy_pb_per_GeV) * 1.05)))
    axes[1].grid(alpha=0.2)
    fig.savefig(out / "tevatron_scale_envelope.png", dpi=220)
    fig.savefig(out / "tevatron_scale_envelope.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
