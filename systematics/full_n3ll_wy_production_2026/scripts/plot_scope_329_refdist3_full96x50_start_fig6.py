#!/usr/bin/env python3
"""Render a start-only Fig. 6 diagnostic from the completed 96-start W+Y fit.

The fitted W+Y starts do not carry a production-comparable absolute
normalization.  Each curve is therefore placed on the promoted production
normalization by multiplying the baseline central b-space TMD by the
candidate/baseline F_NP ratio.  Only the 96 independent starts are propagated;
experimental replicas are deliberately omitted from this diagnostic.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
SYSTEMATICS = BASE.parent
REPORTS = BASE / "reports"
TRANSFORMER = SYSTEMATICS.parent / "construct_v23a_regularized_kspace_tmd_v2.py"
BASELINE_BSPACE = (
    SYSTEMATICS / "dataset_identifiability_campaign_2026"
    / "production_lambda1_empirical_reference_full96x50/bspace_combined_bands.csv"
)
BASELINE_FNP = REPORTS / "baseline_reference_promoted96_allx.csv"
START_ROOT = REPORTS / "scope_329_refdist3_full96x50_long50k_start_s"
TARGET = REPORTS / "scope_329_refdist3_full96x50_long50k_start_fig6"
STARTS = tuple(range(303, 399))


def load_transformer():
    spec = importlib.util.spec_from_file_location("fig6_start_transform", TRANSFORMER)
    if spec is None or spec.loader is None:
        raise RuntimeError(TRANSFORMER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def completed_starts() -> list[int]:
    out = [seed for seed in STARTS if (Path(f"{START_ROOT}{seed}") / "metrics.json").exists()]
    if len(out) != len(STARTS):
        missing = sorted(set(STARTS) - set(out))
        raise RuntimeError(f"expected 96 completed starts; missing {missing}")
    return out


def build_bspace(starts: list[int]) -> pd.DataFrame:
    baseline = pd.read_csv(BASELINE_BSPACE)
    baseline = baseline[baseline.flavor.astype(str).isin(["u", "d"])].copy()
    fnp = pd.read_csv(BASELINE_FNP)
    fnp = fnp[np.isclose(fnp.x.to_numpy(float), 0.1)].sort_values("bT")
    if fnp.empty:
        raise RuntimeError("promoted baseline F_NP x=0.1 slice is missing")

    rows = []
    for seed in starts:
        grid = pd.read_csv(Path(f"{START_ROOT}{seed}") / "fnp_debug_grid.csv")
        grid = grid[np.isclose(grid.x.to_numpy(float), 0.1)].sort_values("bT")
        if grid.empty:
            raise RuntimeError(f"missing x=0.1 F_NP curve for seed {seed}")
        for flavor in ("u", "d"):
            ref = baseline[baseline.flavor.astype(str).eq(flavor)].sort_values("bT")
            b = ref.bT.to_numpy(float)
            cand = np.interp(b, grid.bT.to_numpy(float), grid.F_NP.to_numpy(float))
            base_fnp = np.interp(b, fnp.bT.to_numpy(float), fnp.F_NP.to_numpy(float))
            ratio = cand / np.maximum(base_fnp, 1.0e-30)
            for bi, val in zip(b, ref.central.to_numpy(float) * ratio):
                rows.append({
                    "_replica_key": f"s{seed}|pdf0",
                    "seed": str(seed), "pdf_member": 0, "pid": 0,
                    "flavor": flavor, "x": 0.1, "Q": 10.0,
                    "bT": float(bi), "ftilde": float(val),
                })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=TARGET)
    args = ap.parse_args()
    starts = completed_starts()
    bspace = build_bspace(starts)
    transform = load_transformer()
    settings = argparse.Namespace(
        quantities=["ftilde"], tail_mode="expb2", tail_fit_bmin=None,
        eps=1.0e-300, b_transform_max=24.0, n_b_transform=6001,
        k_max=4.0, n_k=401, end_taper_start_fraction=0.92,
    )
    k_long, transform_meta = transform.transform_curves(bspace, settings)
    bands = (
        k_long.groupby(["flavor", "kT"], observed=False)["value"]
        .quantile([0.16, 0.50, 0.84]).unstack()
        .rename(columns={0.16: "q16", 0.50: "central", 0.84: "q84"})
        .reset_index()
    )
    args.out.mkdir(parents=True, exist_ok=True)
    bspace.to_csv(args.out / "bspace_start_members_long.csv", index=False)
    k_long.to_csv(args.out / "kspace_start_members_long.csv", index=False)
    bands.to_csv(args.out / "kspace_start_only_bands.csv", index=False)

    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm",
        "axes.linewidth": 0.9, "xtick.direction": "in",
        "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    })
    fig, ax = plt.subplots(figsize=(7.4, 5.0), constrained_layout=True)
    colors = {"u": "#1f77b4", "d": "#d95f02"}
    flavor_labels = {"u": r"$u$ quark", "d": r"$d$ quark"}
    view = bands[bands.kT <= 2.25]
    for flavor in ("u", "d"):
        g = view[view.flavor.eq(flavor)].sort_values("kT")
        ax.fill_between(g.kT, g.q16, g.q84, color=colors[flavor], alpha=0.30,
                        linewidth=0, label="_nolegend_")
        ax.plot(g.kT, g.q16, color=colors[flavor], lw=0.55, alpha=0.72,
                label="_nolegend_")
        ax.plot(g.kT, g.q84, color=colors[flavor], lw=0.55, alpha=0.72,
                label="_nolegend_")
        ax.plot(g.kT, g.central, color=colors[flavor], lw=1.8,
                label=flavor_labels[flavor])
    ax.text(0.98, 0.96, r"$x=0.1\qquad Q=10\ \mathrm{GeV}$",
            ha="right", va="top", transform=ax.transAxes, fontsize=12)
    ax.set_xlabel(r"$k_T\ [\mathrm{GeV}]$")
    ax.set_ylabel(r"$f_1^q(x,k_T;Q)$")
    ax.set_xlim(0, 2.25)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.15)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="0.45", alpha=0.22, edgecolor="none"))
    labels.append("96-start model envelope (16--84%)")
    ax.legend(handles, labels, frameon=False, fontsize=8.5)
    fig.savefig(args.out / "fig6_start_only_ud.png", dpi=240)
    fig.savefig(args.out / "fig6_start_only_ud.pdf")
    plt.close(fig)

    # Companion view: the absolute band is necessarily hard to see where the
    # TMD itself is small, so show its fractional half-width in a lower panel.
    fig, axes = plt.subplots(
        2, 1, figsize=(7.4, 6.8), sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )
    ax = axes[0]
    for flavor in ("u", "d"):
        g = view[view.flavor.eq(flavor)].sort_values("kT")
        ax.fill_between(g.kT, g.q16, g.q84, color=colors[flavor], alpha=0.30,
                        linewidth=0, label="_nolegend_")
        ax.plot(g.kT, g.q16, color=colors[flavor], lw=0.55, alpha=0.72,
                label="_nolegend_")
        ax.plot(g.kT, g.q84, color=colors[flavor], lw=0.55, alpha=0.72,
                label="_nolegend_")
        ax.plot(g.kT, g.central, color=colors[flavor], lw=1.8,
                label=flavor_labels[flavor])
    ax.text(0.98, 0.96, r"$x=0.1\qquad Q=10\ \mathrm{GeV}$",
            ha="right", va="top", transform=ax.transAxes, fontsize=12)
    ax.set_ylabel(r"$f_1^q(x,k_T;Q)$")
    ax.set_xlim(0, 2.25)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.15)
    handles, labels_legend = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="0.45", alpha=0.30, edgecolor="none"))
    labels_legend.append("96-start model envelope (16--84%)")
    ax.legend(handles, labels_legend, frameon=False, fontsize=8.5)

    ax = axes[1]
    for flavor in ("u", "d"):
        g = view[view.flavor.eq(flavor)].sort_values("kT").copy()
        peak = float(g.central.max())
        active = g.central.to_numpy(float) > 0.05 * peak
        rel = 50.0 * (g.q84.to_numpy(float) - g.q16.to_numpy(float)) / np.maximum(
            np.abs(g.central.to_numpy(float)), 1.0e-30)
        rel[~active] = np.nan
        ax.plot(g.kT, rel, color=colors[flavor], lw=1.7, label=flavor_labels[flavor])
    ax.set_xlabel(r"$k_T\ [\mathrm{GeV}]$")
    ax.set_ylabel("relative half-width [%]")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.15)
    ax.legend(frameon=False, fontsize=8.5, ncol=2)
    fig.savefig(args.out / "fig6_start_only_ud_with_fractional_width.png", dpi=240)
    fig.savefig(args.out / "fig6_start_only_ud_with_fractional_width.pdf")
    plt.close(fig)

    metrics = {}
    for flavor in ("u", "d"):
        g = bands[(bands.flavor == flavor) & (bands.kT <= 2.25)].sort_values("kT")
        peak = float(g.central.max())
        active = g.central.to_numpy(float) > 0.05 * peak
        rel = 0.5 * (g.q84.to_numpy(float) - g.q16.to_numpy(float)) / np.maximum(g.central.to_numpy(float), 1.0e-30)
        metrics[flavor] = {
            "max_relative_q16_q84_halfwidth_active": float(np.max(rel[active])),
            "median_relative_q16_q84_halfwidth_active": float(np.median(rel[active])),
            "active_kT_max_GeV": float(g.kT.to_numpy(float)[active].max()),
        }
    summary = {
        "status": "isolated_96_start_only_fig6_not_production",
        "start_count": len(starts), "start_seeds": [min(starts), max(starts)],
        "experimental_replicas_included": False,
        "normalization": "promoted production b-space central times candidate/baseline F_NP ratio",
        "scope": "corrected non-LHCb-Y W+Y, lambda_ref=3, full bT reference interval",
        "x": 0.1, "Q_GeV": 10.0, "display_kT_max_GeV": 2.25,
        "bands": metrics, "transform": transform_meta,
        "artifacts": {
            "png": str(args.out / "fig6_start_only_ud.png"),
            "pdf": str(args.out / "fig6_start_only_ud.pdf"),
            "bands": str(args.out / "kspace_start_only_bands.csv"),
            "fractional_width_png": str(args.out / "fig6_start_only_ud_with_fractional_width.png"),
            "fractional_width_pdf": str(args.out / "fig6_start_only_ud_with_fractional_width.pdf"),
        },
        "interpretation": "pointwise empirical q16--q84 start/model envelope; not a Gaussian confidence interval",
        "frozen_production_modified": False, "promotion_authorized": False,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
