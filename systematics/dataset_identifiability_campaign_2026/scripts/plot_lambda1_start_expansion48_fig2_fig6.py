#!/usr/bin/env python3
"""Render isolated 48-start Fig. 2/Fig. 6 diagnostics.

The fresh-start expansion has no experimental replicas.  These figures show
the empirical q16--q84 distribution of the 24 incumbent plus 24 fresh
lambda=1 endpoints, with one median curve per flavor and no individual starts.
The band is descriptive model/start non-uniqueness, not a calibrated 68%
confidence interval.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
SYSTEMATICS = BASE.parent
TARGET = BASE / "summaries/lambda1_start_expansion48"
FIGURE_TARGET = TARGET / "figures"
REFERENCE_BSPACE = (
    SYSTEMATICS
    / "collins_factorization_validity/plots/rowidfix_stageFT_all_qmax0p20_lam0p50_central_exactx"
    / "v22_scheme_tmd_bspace_long.csv"
)
OUTPUTS = BASE / "outputs"
COLORS = {
    "u": "#1f77b4", "d": "#d95f02", "s": "#2ca02c",
    "ubar": "#9467bd", "dbar": "#8c564b", "sbar": "#e377c2",
}
LABELS = {
    "u": r"$u$ quark", "d": r"$d$ quark", "s": r"$s$ quark",
    "ubar": r"$\bar{u}$ quark", "dbar": r"$\bar{d}$ quark",
    "sbar": r"$\bar{s}$ quark",
}
FLAVORS_FIG2 = ("u", "d", "s", "ubar", "dbar", "sbar")
FLAVORS_FIG6 = ("u", "d")


def endpoint_paths() -> list[Path]:
    summary = json.loads((TARGET / "summary.json").read_text())
    tags = summary["endpoint_tags"]
    paths = [OUTPUTS / tag for tag in tags[24:]]
    paths = [OUTPUTS / tag for tag in tags[:24]] + paths
    missing = [str(path) for path in paths if not (path / "fnp_grid.csv").exists()]
    if missing:
        raise RuntimeError(f"missing endpoint F_NP grids: {missing[:3]}")
    if len(paths) != 48:
        raise RuntimeError(f"expected 48 endpoints, found {len(paths)}")
    return paths


def fnp_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path / "fnp_grid.csv")
    return frame[np.isclose(frame["x"], 0.1)].sort_values("bT")


def build_bspace(paths: list[Path]) -> pd.DataFrame:
    reference = pd.read_csv(REFERENCE_BSPACE)
    reference = reference[
        np.isclose(reference["x"], 0.1)
        & np.isclose(reference["Q"], 7.5)
        & reference["flavor"].astype(str).isin(FLAVORS_FIG2)
    ].copy()
    rows = []
    for path in paths:
        fnp = fnp_frame(path)
        for flavor, group in reference.groupby("flavor", sort=False):
            curve = group.sort_values("bT").copy()
            curve["value"] = curve["ftilde_no_np"] * np.interp(
                curve["bT"], fnp["bT"], fnp["F_NP"])
            curve["endpoint"] = path.name
            rows.append(curve[["flavor", "bT", "value", "endpoint"]])
    long = pd.concat(rows, ignore_index=True)
    bands = []
    for flavor, group in long.groupby("flavor", sort=False):
        wide = group.pivot(index="bT", columns="endpoint", values="value")
        values = wide.to_numpy(float)
        bands.append(pd.DataFrame({
            "flavor": flavor,
            "bT": wide.index.to_numpy(float),
            "q16": np.quantile(values, 0.16, axis=1),
            "median": np.median(values, axis=1),
            "q84": np.quantile(values, 0.84, axis=1),
        }))
    return pd.concat(bands, ignore_index=True)


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm",
        "axes.linewidth": 0.9, "xtick.direction": "in",
        "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    })


def save(fig: plt.Figure, stem: str) -> None:
    FIGURE_TARGET.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_TARGET / f"{stem}.png", dpi=240)
    fig.savefig(FIGURE_TARGET / f"{stem}.pdf")
    plt.close(fig)


def main() -> None:
    paths = endpoint_paths()
    bspace = build_bspace(paths)
    kspace = pd.read_csv(TARGET / "kspace_start_only_bands.csv")
    bspace.to_csv(TARGET / "bspace_tmd_start_only_bands.csv", index=False)
    set_style()

    fig, ax = plt.subplots(figsize=(7.4, 5.0), constrained_layout=True)
    for flavor in FLAVORS_FIG2:
        curve = bspace[bspace["flavor"].astype(str).eq(flavor)].sort_values("bT")
        ax.fill_between(curve["bT"], curve["q16"], curve["q84"],
                        color=COLORS[flavor], alpha=0.17, linewidth=0)
        ax.plot(curve["bT"], curve["median"], color=COLORS[flavor],
                lw=1.7, label=LABELS[flavor])
    ax.text(0.98, 0.96, r"$x=0.1\qquad Q=7.5\ \mathrm{GeV}$",
            ha="right", va="top", transform=ax.transAxes, fontsize=12)
    ax.set(xlabel=r"$b_T\ [\mathrm{GeV}^{-1}]$",
           ylabel=r"$\widetilde f_1^q(x,b_T;Q)$", xlim=(0, 4.0))
    ax.grid(alpha=0.15)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="0.6", alpha=0.25, edgecolor="none"))
    labels.append(r"48-start q16--q84 range")
    ax.legend(handles, labels, frameon=False, fontsize=9, ncol=2)
    save(fig, "fig2_bspace_48start_only")

    fig, ax = plt.subplots(figsize=(7.4, 5.0), constrained_layout=True)
    selected = kspace[
        np.isclose(kspace["Q"], 10.0)
        & np.isclose(kspace["x"], 0.1)
        & kspace["flavor"].astype(str).isin(FLAVORS_FIG6)
        & (kspace["kT"] <= 2.25)
    ]
    for flavor in FLAVORS_FIG6:
        curve = selected[selected["flavor"].astype(str).eq(flavor)].sort_values("kT")
        if curve.empty:
            raise RuntimeError(f"Fig. 6 missing {flavor}")
        ax.fill_between(curve["kT"], curve["q16"], curve["q84"],
                        color=COLORS[flavor], alpha=0.20, linewidth=0)
        ax.plot(curve["kT"], curve["median"], color=COLORS[flavor],
                lw=1.8, label=LABELS[flavor])
    ax.text(0.98, 0.96, r"$x=0.1\qquad Q=10\ \mathrm{GeV}$",
            ha="right", va="top", transform=ax.transAxes, fontsize=12)
    ax.set(xlabel=r"$k_T\ [\mathrm{GeV}]$", ylabel=r"$f_1^q(x,k_T;Q)$",
           xlim=(0, 2.25), ylim=(0, None))
    ax.grid(alpha=0.15)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="0.6", alpha=0.25, edgecolor="none"))
    labels.append(r"48-start q16--q84 range")
    ax.legend(handles, labels, frameon=False, fontsize=10)
    save(fig, "fig6_kspace_48start_only")

    result = {
        "status": "isolated_48_start_fig2_fig6_diagnostic_not_production",
        "figure_2": "fig2_bspace_48start_only.png",
        "figure_6": "fig6_kspace_48start_only.png",
        "endpoint_count": 48,
        "band": "pointwise empirical q16--q84 over 24 incumbent plus 24 fresh starts",
        "formal_confidence_level_assigned": False,
        "experimental_replicas_included": False,
        "contains_individual_start_curves": False,
        "production_sources_modified": False,
        "source_summary": str(TARGET / "summary.json"),
    }
    (FIGURE_TARGET / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
