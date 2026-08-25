#!/usr/bin/env python3
"""Render PRD Fig. 3 with the lambda=1 combined-ensemble median F_NP."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REFERENCE = ROOT / (
    "systematics/collins_factorization_validity/plots/"
    "rowidfix_stageFT_all_qmax0p20_lam0p50_central_exactx/"
    "v22_scheme_tmd_bspace_long.csv"
)
FIG2_BANDS = HERE / "fig2_lambda1_combined_six_flavor_bands.csv"
FLAVORS = ("u", "d", "s", "ubar", "dbar", "sbar")
TITLES = {"u": r"$u$", "d": r"$d$", "s": r"$s$",
          "ubar": r"$\bar u$", "dbar": r"$\bar d$", "sbar": r"$\bar s$"}


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    bands = read(FIG2_BANDS)
    reference = read(REFERENCE)
    fnp_median = {}
    for row in bands:
        if row["flavor"] != "u":
            continue
        key = round(float(row["bT"]), 10)
        source = next(
            item for item in reference
            if item["flavor"] == "u"
            and abs(float(item["x"]) - 0.1) < 1e-12
            and abs(float(item["Q"]) - 7.5) < 1e-12
            and round(float(item["bT"]), 10) == key
        )
        fnp_median[key] = float(row["median"]) / float(source["ftilde_no_np"])

    components = []
    for flavor in FLAVORS:
        rows = [row for row in reference
                if row["flavor"] == flavor
                and abs(float(row["x"]) - 0.1) < 1e-12
                and abs(float(row["Q"]) - 7.5) < 1e-12
                and float(row["bT"]) <= 1.2]
        rows.sort(key=lambda row: float(row["bT"]))
        for row in rows:
            b = float(row["bT"])
            fnp = fnp_median[round(b, 10)]
            no_np = float(row["ftilde_no_np"])
            components.append({
                "flavor": flavor, "bT": b,
                "full_tmd": no_np * fnp,
                "perturbative_ope_evolution": no_np,
                "ope_boundary_nlo": float(row["ope_boundary_nlo"]),
                "evolution_half": float(row["evol_half"]),
                "fnp_lambda1_median": fnp,
                "mu_b": float(row["mu_b"]),
            })

    out_csv = HERE / "fig3_lambda1_lowb_components.csv"
    with out_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=components[0].keys())
        writer.writeheader()
        writer.writerows(components)

    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm", "font.size": 15,
        "axes.linewidth": 1.2, "xtick.direction": "in", "ytick.direction": "in",
    })
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 7.9), sharex=True)
    styles = (
        ("full_tmd", r"$\widetilde f_1^q$", "black", "-", 2.6),
        ("perturbative_ope_evolution", r"$(C\!\otimes\! f)e^{-S/2}$", "#1f77b4", "--", 2.15),
        ("ope_boundary_nlo", r"$C\!\otimes\! f$", "#ff7f0e", "-.", 1.9),
        ("evolution_half", r"$e^{-S/2}$", "#2ca02c", ":", 2.0),
        ("fnp_lambda1_median", r"$F_{\rm NP}$", "#d62728", "-", 2.0),
    )
    for ax, flavor in zip(axes.ravel(), FLAVORS):
        rows = [row for row in components if row["flavor"] == flavor]
        b = [float(row["bT"]) for row in rows]
        for key, label, color, ls, lw in styles:
            values = [float(row[key]) for row in rows]
            base = values[0]
            ax.plot(b, [value / base for value in values], label=label,
                    color=color, ls=ls, lw=lw)
        full = [float(row["full_tmd"]) for row in rows]
        peak = b[max(range(len(full)), key=lambda i: full[i])]
        first_uncapped = next((b[i] for i, row in enumerate(rows)
                               if float(row["mu_b"]) < 7.5 * (1 - 1e-10)), None)
        if first_uncapped is not None:
            ax.axvline(first_uncapped, color="0.68", lw=1.2, ls="--")
        ax.axvline(peak, color="0.30", lw=1.2, ls=":")
        ax.set_title(TITLES[flavor], fontsize=19, pad=3)
        ax.set_xlim(0, 1.2)
        ax.grid(color="0.88", lw=0.6)
        ax.minorticks_on()
        ax.tick_params(which="major", top=True, right=True, labelsize=15,
                       length=6, width=1.15)
        ax.tick_params(which="minor", top=True, right=True, length=3.2, width=.9)

    axes[0, 0].legend(frameon=False, fontsize=13, handlelength=2.5,
                      loc="lower left")
    for ax in axes[1, :]:
        ax.set_xlabel(r"$b_T\ [\mathrm{GeV}^{-1}]$", fontsize=18)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"normalized to $b_T=0$", fontsize=18)
    fig.suptitle(r"Low-$b_T$ decomposition, $x=0.1$, $Q=7.5\ \mathrm{GeV}$",
                 fontsize=20, y=.99)
    fig.tight_layout(rect=(0, 0, 1, .955), h_pad=1.0, w_pad=.7)
    fig.savefig(HERE / "fig3_lambda1_lowb_decomposition.pdf",
                bbox_inches="tight", pad_inches=.04)
    fig.savefig(HERE / "fig3_lambda1_lowb_decomposition.png", dpi=300,
                bbox_inches="tight", pad_inches=.04)
    plt.close(fig)


if __name__ == "__main__":
    main()
