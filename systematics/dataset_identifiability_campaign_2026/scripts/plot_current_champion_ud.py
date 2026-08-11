#!/usr/bin/env python3
"""Render the current provisional champion without historical overlays.

The combined pointwise q16--q84 interval includes optimizer-start variation
and therefore has no calibrated confidence-level interpretation.
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
REGISTRY = BASE / "summaries/champion_registry/current.json"
TARGET = BASE / "summaries/champion_registry/current_fig2_fig6"
COLORS = {"u": "#1f77b4", "d": "#d95f02"}
LABELS = {"u": r"$u$ quark", "d": r"$d$ quark"}


def save(fig, target: Path, stem: str) -> None:
    fig.savefig(target / f"{stem}.png", dpi=240)
    fig.savefig(target / f"{stem}.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--target-dir", type=Path, default=TARGET)
    args = parser.parse_args()
    champion = json.loads(args.registry.read_text())
    b = pd.read_csv(champion["artifacts"]["bspace_combined_bands"])
    k = pd.read_csv(champion["artifacts"]["kspace_combined_bands"])
    target = args.target_dir
    target.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm",
        "axes.linewidth": .9, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
    })

    fig, ax = plt.subplots(figsize=(7.4, 5.0), constrained_layout=True)
    for flavor in ("u", "d"):
        curve = b[b.flavor.astype(str).eq(flavor)].sort_values("bT")
        ax.fill_between(curve.bT, curve.q16, curve.q84,
                        color=COLORS[flavor], alpha=.20, linewidth=0)
        ax.plot(curve.bT, curve.central, color=COLORS[flavor], lw=1.8,
                label=LABELS[flavor])
    ax.text(.98, .96, r"$x=0.1\qquad Q=10\ \mathrm{GeV}$",
            ha="right", va="top", transform=ax.transAxes, fontsize=12)
    ax.set(xlabel=r"$b_T\ [\mathrm{GeV}^{-1}]$",
           ylabel=r"$\widetilde f_1^q(x,b_T;Q)$", xlim=(0, 4))
    ax.grid(alpha=.15)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="0.5", alpha=.22, edgecolor="none"))
    labels.append(r"empirical q16--q84 combined ensemble")
    ax.legend(handles, labels, frameon=False, fontsize=10)
    save(fig, target, "champion_fig2space_bT_ud_combined_1sigma")

    fig, ax = plt.subplots(figsize=(7.4, 5.0), constrained_layout=True)
    for flavor in ("u", "d"):
        curve = k[(k.flavor.astype(str).eq(flavor)) & (k.kT <= 2.25)].sort_values("kT")
        ax.fill_between(curve.kT, curve.q16, curve.q84,
                        color=COLORS[flavor], alpha=.20, linewidth=0)
        ax.plot(curve.kT, curve.central, color=COLORS[flavor], lw=1.8,
                label=LABELS[flavor])
    ax.text(.98, .96, r"$x=0.1\qquad Q=10\ \mathrm{GeV}$",
            ha="right", va="top", transform=ax.transAxes, fontsize=12)
    ax.set(xlabel=r"$k_T\ [\mathrm{GeV}]$", ylabel=r"$f_1^q(x,k_T;Q)$",
           xlim=(0, 2.25), ylim=(0, None))
    ax.grid(alpha=.15)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="0.5", alpha=.22, edgecolor="none"))
    labels.append(r"empirical q16--q84 combined ensemble")
    ax.legend(handles, labels, frameon=False, fontsize=10)
    save(fig, target, "champion_fig6_kT_ud_combined_1sigma")

    summary = {
        "status": "provisional_champion_visual_baseline_not_production",
        "champion_id": champion["champion_id"],
        "figure_2_space": "champion_fig2space_bT_ud_combined_1sigma.pdf",
        "figure_6": "champion_fig6_kT_ud_combined_1sigma.pdf",
        "band": "combined start plus experimental operational q16-q84 ensemble",
        "formal_confidence_level_assigned": False,
        "legacy_filename_note": (
            "the retained _1sigma stems are compatibility names only; the "
            "combined interval is not a calibrated one-sigma band"),
        "contains_historical_overlay": False,
        "contains_individual_start_curves": False,
        "production_sources_modified": False,
    }
    (target / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
