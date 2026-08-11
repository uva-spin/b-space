#!/usr/bin/env python3
"""Plot all canonical external finite-tail benchmark ratios."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
SYS = ROOT / "systematics" / "finite_y_tail_benchmark"


def main() -> None:
    summary = SYS / "summaries" / "external_tail_benchmarks_canonical.csv"
    if not summary.exists():
        raise FileNotFoundError(f"Run summarize_existing_tail_benchmarks.py first: {summary}")
    df = pd.read_csv(summary).sort_values(["dataset", "qT_over_Q"])

    df["ratio"] = df["dyturbo_pb_per_GeV"] / df["mcfm_pb_per_GeV"]
    df["ratio_unc"] = df["ratio"] * np.sqrt(
        (df["dyturbo_pb_per_GeV_unc"] / df["dyturbo_pb_per_GeV"]) ** 2
        + (df["mcfm_pb_per_GeV_unc"] / df["mcfm_pb_per_GeV"]) ** 2
    )

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "legend.fontsize": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.linewidth": 1.25,
            "xtick.major.width": 1.15,
            "ytick.major.width": 1.15,
            "xtick.major.size": 5.0,
            "ytick.major.size": 5.0,
            "mathtext.fontset": "stix",
        }
    )

    fig, ax = plt.subplots(figsize=(6.6, 4.25))
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.17, top=0.90)

    ax.axhspan(0.95, 1.05, color="#2f7d4b", alpha=0.12, lw=0, label=r"$\pm 5\%$ gate")
    ax.axhline(1.0, color="0.25", lw=1.1)

    styles = {
        "CDF_RUN_1": {"marker": "s", "color": "#b85f1d", "label": "CDF Run I"},
        "CDF_RUN_2": {"marker": "o", "color": "#1f5f99", "label": "CDF Run II"},
        "D0_RUN_1": {"marker": "^", "color": "#2f7d4b", "label": "D0 Run I"},
        "LHCb_7": {"marker": "D", "color": "#7b4ca0", "label": "LHCb 7 TeV"},
    }
    for dataset, g in df.groupby("dataset"):
        style = styles.get(dataset, {"marker": "D", "color": "0.3", "label": dataset})
        ax.errorbar(
            g["qT_over_Q"],
            g["ratio"],
            yerr=g["ratio_unc"],
            fmt=style["marker"],
            ms=6.5,
            color=style["color"],
            ecolor=style["color"],
            elinewidth=1.0,
            capsize=2.5,
            label=style["label"],
        )

    ax.set_xlim(0.085, 0.335)
    ax.set_ylim(0.985, 1.015)
    ax.set_xlabel(r"$q_T/Q$")
    ax.set_ylabel("DYTurbo / MCFM")
    ax.set_title("Finite-tail external-code benchmark")
    ax.grid(True, axis="y", color="0.88", lw=0.8)
    ax.tick_params(direction="in", top=True, right=True)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="0.85")

    out = SYS / "plots" / "external_tail_benchmark_ratio"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=220)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {out.with_suffix('.png')}")
    print(f"wrote {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
