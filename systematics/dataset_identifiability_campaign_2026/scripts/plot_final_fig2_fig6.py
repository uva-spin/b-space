#!/usr/bin/env python3
"""Render guarded final-style Fig. 2 and Fig. 6 from validated ensemble bands.

The shaded interval is the pointwise empirical q16--q84 range of the combined
ensemble.  It is descriptive: start non-uniqueness is not a calibrated random
variable, so the renderer must not label this range as a formal 68% or 1-sigma
confidence interval.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
COLORS = {
    "u": "#1f77b4", "d": "#d95f02", "s": "#2ca02c",
    "ubar": "#9467bd", "dbar": "#8c564b", "sbar": "#e377c2",
}
LABELS = {
    "u": r"$u$ quark", "d": r"$d$ quark", "s": r"$s$ quark",
    "ubar": r"$\bar{u}$ quark", "dbar": r"$\bar{d}$ quark",
    "sbar": r"$\bar{s}$ quark",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bspace-stability-dir", type=Path, required=True)
    parser.add_argument("--kspace-stability-dir", type=Path, required=True)
    parser.add_argument("--target-name", default="final_fig2_fig6")
    parser.add_argument("--diagnostic-override", action="store_true")
    parser.add_argument(
        "--final-authorized", action="store_true",
        help="Mark output final only after all scientific and uncertainty gates pass.")
    return parser.parse_args()


def require_gate(path: Path, override: bool) -> dict:
    status = json.loads((path / "summary.json").read_text())
    if not status["endpoint_gate_pass"] and not override:
        raise RuntimeError(
            f"{path} has not passed the ensemble endpoint gate; refusing final render")
    return status


def save(fig, target: Path, stem: str) -> None:
    fig.savefig(target / f"{stem}.png", dpi=240)
    fig.savefig(target / f"{stem}.pdf")
    plt.close(fig)


def main() -> None:
    args = arguments()
    b_status = require_gate(args.bspace_stability_dir, args.diagnostic_override)
    k_status = require_gate(args.kspace_stability_dir, args.diagnostic_override)
    b = pd.read_csv(args.bspace_stability_dir / "bT_tmd_bands.csv")
    k = pd.read_csv(args.kspace_stability_dir / "kT_tmd_bands.csv")
    target = BASE / "summaries" / args.target_name
    target.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.linewidth": 0.9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
    })

    fig, ax = plt.subplots(figsize=(7.4, 5.0), constrained_layout=True)
    selected_b = b[np.isclose(b["Q"], 7.5) & (b["bT"] <= 4.0)]
    for flavor in ("u", "d", "s", "ubar", "dbar", "sbar"):
        curve = selected_b[selected_b["flavor"].astype(str).eq(flavor)].sort_values("bT")
        if not len(curve):
            raise RuntimeError(f"Fig. 2 missing {flavor}")
        color = COLORS[flavor]
        ax.fill_between(
            curve["bT"], curve["q16"], curve["q84"],
            color=color, alpha=0.17, linewidth=0)
        ax.plot(curve["bT"], curve["median"], color=color, lw=1.7,
                label=LABELS[flavor])
    ax.text(0.98, 0.96, r"$x=0.1\qquad Q=7.5\ \mathrm{GeV}$",
            ha="right", va="top", transform=ax.transAxes, fontsize=12)
    ax.set_xlabel(r"$b_T\ [\mathrm{GeV}^{-1}]$", fontsize=13)
    ax.set_ylabel(r"$\widetilde f_1^q(x,b_T;Q)$", fontsize=13)
    ax.set_xlim(0, 4.0)
    ax.grid(alpha=0.15)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="0.6", alpha=0.25, edgecolor="none"))
    labels.append(r"empirical q16--q84 combined ensemble")
    ax.legend(handles, labels, frameon=False, fontsize=9, ncol=2)
    save(fig, target, "updated_fig2_bspace_full_uncertainty")

    fig, ax = plt.subplots(figsize=(7.4, 5.0), constrained_layout=True)
    selected_k = k[np.isclose(k["Q"], 10.0) & (k["kT"] <= 2.25)]
    for flavor in ("u", "d"):
        curve = selected_k[selected_k["flavor"].astype(str).eq(flavor)].sort_values("kT")
        if not len(curve):
            raise RuntimeError(f"Fig. 6 missing {flavor}")
        color = COLORS[flavor]
        ax.fill_between(
            curve["kT"], curve["q16"], curve["q84"],
            color=color, alpha=0.20, linewidth=0)
        ax.plot(curve["kT"], curve["median"], color=color, lw=1.8,
                label=LABELS[flavor])
    ax.text(0.98, 0.96, r"$x=0.1\qquad Q=10\ \mathrm{GeV}$",
            ha="right", va="top", transform=ax.transAxes, fontsize=12)
    ax.set_xlabel(r"$k_T\ [\mathrm{GeV}]$", fontsize=13)
    ax.set_ylabel(r"$f_1^q(x,k_T;Q)$", fontsize=13)
    ax.set_xlim(0, 2.25)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.15)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="0.6", alpha=0.25, edgecolor="none"))
    labels.append(r"empirical q16--q84 combined ensemble")
    ax.legend(handles, labels, frameon=False, fontsize=10)
    save(fig, target, "updated_fig6_kspace_ud_full_uncertainty")

    summary = {
        "status": (
            "final_validated_figures" if (
                args.final_authorized
                and b_status["endpoint_gate_pass"]
                and k_status["endpoint_gate_pass"])
            else "diagnostic_render_not_final"),
        "bspace_endpoint_gate_pass": b_status["endpoint_gate_pass"],
        "kspace_endpoint_gate_pass": k_status["endpoint_gate_pass"],
        "diagnostic_override": args.diagnostic_override,
        "final_authorized": args.final_authorized,
        "figure_2": "updated_fig2_bspace_full_uncertainty.pdf",
        "figure_6": "updated_fig6_kspace_ud_full_uncertainty.pdf",
        "contains_legacy_conditional_result": False,
        "contains_individual_start_curves": False,
        "band_definition": "pointwise empirical q16--q84 combined ensemble",
        "formal_confidence_level_assigned": False,
        "production_sources_modified": False,
    }
    (target / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
