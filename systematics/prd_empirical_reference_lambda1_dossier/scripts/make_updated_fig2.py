#!/usr/bin/env python3
"""Render PRD Fig. 2 from the registered lambda=1 crossed ensemble.

The archived crossed summary stores u,d at Q=10. Because F_NP is shared by
all flavors, dividing its u quantiles by the frozen Q=10 perturbative u curve
recovers the pointwise F_NP quantiles exactly. Multiplication by each frozen
Q=7.5 perturbative curve then gives the six-flavor Fig. 2 bands.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CAMPAIGN = ROOT / "systematics/dataset_identifiability_campaign_2026"
CROSSED = CAMPAIGN / (
    "summaries/matched_baseline_reference_distance_lam1e00_"
    "full24_crossed_experimental/bspace_combined_bands.csv"
)
REFERENCE = ROOT / (
    "systematics/collins_factorization_validity/plots/"
    "rowidfix_stageFT_all_qmax0p20_lam0p50_central_exactx/"
    "v22_scheme_tmd_bspace_long.csv"
)

FLAVORS = ("u", "d", "s", "ubar", "dbar", "sbar")
COLORS = {
    "u": "#1f77b4", "d": "#ff7f0e", "s": "#2ca02c",
    "ubar": "#d62728", "dbar": "#9467bd", "sbar": "#8c564b",
}
STYLES = {"u": "-", "d": "--", "s": "--", "ubar": "-.",
          "dbar": ":", "sbar": ":"}
LABELS = {"u": r"$u$", "d": r"$d$", "s": r"$s$",
          "ubar": r"$\bar u$", "dbar": r"$\bar d$", "sbar": r"$\bar s$"}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    crossed = [row for row in read_rows(CROSSED) if row["flavor"] == "u"]
    crossed.sort(key=lambda row: float(row["bT"]))

    reference = read_rows(REFERENCE)
    q10_u = {
        round(float(row["bT"]), 10): float(row["ftilde_no_np"])
        for row in reference
        if row["flavor"] == "u"
        and abs(float(row["x"]) - 0.1) < 1e-12
        and abs(float(row["Q"]) - 10.0) < 1e-12
    }
    fnp_band = {}
    for row in crossed:
        key = round(float(row["bT"]), 10)
        perturbative = q10_u[key]
        fnp_band[key] = tuple(float(row[name]) / perturbative
                              for name in ("q16", "central", "q84"))

    fig2_rows: list[dict[str, float | str]] = []
    for flavor in FLAVORS:
        selected = [
            row for row in reference
            if row["flavor"] == flavor
            and abs(float(row["x"]) - 0.1) < 1e-12
            and abs(float(row["Q"]) - 7.5) < 1e-12
        ]
        selected.sort(key=lambda row: float(row["bT"]))
        for row in selected:
            b = float(row["bT"])
            key = round(b, 10)
            q16_f, median_f, q84_f = fnp_band[key]
            perturbative = float(row["ftilde_no_np"])
            fig2_rows.append({
                "flavor": flavor, "bT": b,
                "q16": perturbative * q16_f,
                "median": perturbative * median_f,
                "q84": perturbative * q84_f,
            })

    data_path = HERE / "fig2_lambda1_combined_six_flavor_bands.csv"
    with data_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream,
                                fieldnames=("flavor", "bT", "q16", "median", "q84"))
        writer.writeheader()
        writer.writerows(fig2_rows)

    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm",
        "font.size": 13, "axes.linewidth": 1.15,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
    })
    fig, ax = plt.subplots(figsize=(7.1, 4.35), constrained_layout=True)
    for flavor in FLAVORS:
        rows = [row for row in fig2_rows
                if row["flavor"] == flavor and float(row["bT"]) <= 4.0]
        b = [float(row["bT"]) for row in rows]
        lo = [float(row["q16"]) for row in rows]
        med = [float(row["median"]) for row in rows]
        hi = [float(row["q84"]) for row in rows]
        ax.fill_between(b, lo, hi, color=COLORS[flavor], alpha=0.18,
                        linewidth=0, zorder=1)
        ax.plot(b, med, color=COLORS[flavor], ls=STYLES[flavor], lw=2.35,
                zorder=2)

    handles = [Line2D([0], [0], color=COLORS[f], ls=STYLES[f], lw=2.35,
                      label=LABELS[f]) for f in FLAVORS]
    handles.append(Patch(facecolor="0.45", edgecolor="none", alpha=0.18,
                         label=r"combined central 68% interval"))
    ax.legend(handles=handles, ncol=4, frameon=False, loc="upper right",
              fontsize=11.5, columnspacing=1.15, handlelength=2.35,
              handletextpad=0.55)
    ax.set_title(r"$x=0.1,\quad Q=7.5\ \mathrm{GeV}$", loc="left",
                 fontsize=15, pad=7)
    ax.set_xlim(0.0, 4.0)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel(r"$b_T\ [\mathrm{GeV}^{-1}]$", fontsize=17)
    ax.set_ylabel(r"$\widetilde f_1^{q}(x,b_T;Q)$", fontsize=17)
    ax.tick_params(which="major", labelsize=14, length=6, width=1.15)
    ax.tick_params(which="minor", length=3.5, width=0.9)
    ax.minorticks_on()

    fig.savefig(HERE / "fig2_lambda1_combined_six_flavor.pdf")
    fig.savefig(HERE / "fig2_lambda1_combined_six_flavor.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
