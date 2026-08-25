#!/usr/bin/env python3
"""Render candidate Fig. 2/Fig. 6 from the full 24-start x 50-replica cross.

The perturbative flavor factors are taken from the frozen b-space reference
only to provide a common visual TMD normalization.  The new candidate-side
information is the full-scope F_NP start/replica ensemble; no frozen output is
overwritten and the figures remain explicitly diagnostic until W/TMD source
consistency is promoted.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent
CENTRAL = ROOT / "collins_factorization_validity/plots/rowidfix_stageFT_all_qmax0p20_lam0p50_central_exactx/v22_scheme_tmd_bspace_long.csv"
TRANSFORMER = ROOT.parent / "workflows/v23a/construction/construct_v23a_regularized_kspace_tmd_v2.py"
DEFAULT_SOURCE = BASE / "reports/scope_353_start_replica_propagation_final/fnp_start_replica_crossed_long_x0p1.csv"
DEFAULT_TARGET = BASE / "reports/scope_353_final_fig2_fig6"


def load_transformer():
    spec = importlib.util.spec_from_file_location("scope353_transform", TRANSFORMER)
    if spec is None or spec.loader is None:
        raise RuntimeError(TRANSFORMER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    ap.add_argument("--start-count", type=int, default=None)
    ap.add_argument("--replica-count", type=int, default=None)
    ap.add_argument("--title-label", default=None,
                    help="Optional diagnostic label appended to the plot title.")
    args = ap.parse_args()
    fnp = pd.read_csv(args.source)
    central_all = pd.read_csv(CENTRAL)
    rows = []
    bspace = []
    for flavor in ("u", "d"):
        ref = central_all[np.isclose(central_all.x, .1) & np.isclose(central_all.Q, 10.0)
                          & central_all.flavor.astype(str).eq(flavor)].sort_values("bT").copy()
        if len(ref) < 3:
            raise RuntimeError(f"missing perturbative reference for {flavor}")
        b = ref.bT.to_numpy(float)
        perturbative = ref.ftilde_no_np.to_numpy(float)
        for member, g in fnp.groupby("member", sort=True):
            f = np.interp(b, g.bT.to_numpy(float), g.F_NP.to_numpy(float))
            trial = ref.copy()
            trial["F_NP"] = f
            trial["ftilde"] = perturbative * f
            trial["seed"] = str(member)
            trial["pdf_member"] = 0
            trial["_replica_key"] = f"cross{member}|pdf0"
            trial["flavor"] = flavor
            trial["x"] = .1
            trial["Q"] = 10.0
            bspace.append(trial[["x", "Q", "flavor", "bT", "F_NP", "ftilde", "ftilde_no_np", "seed", "pdf_member", "_replica_key"]])
        wide = pd.DataFrame({"bT": b})
        vals = np.stack([np.interp(b, g.bT.to_numpy(float), g.F_NP.to_numpy(float)) * perturbative
                         for _, g in fnp.groupby("member", sort=True)])
        for quant, q in (("q16", .16), ("central", .50), ("q84", .84)):
            wide[quant] = np.quantile(vals, q, axis=0)
        wide["flavor"] = flavor
        bspace.append(None)
        rows.append(wide)
    b_frame = pd.concat(rows, ignore_index=True)
    b_long = pd.concat([x for x in bspace if x is not None], ignore_index=True)
    transform = load_transformer()
    settings = argparse.Namespace(
        quantities=["ftilde"], tail_mode="expb2", tail_fit_bmin=None,
        eps=1e-300, b_transform_max=24.0, n_b_transform=6001,
        k_max=4.0, n_k=401, end_taper_start_fraction=.92,
    )
    k_long, transform_meta = transform.transform_curves(b_long, settings)
    k_rows = []
    for flavor in ("u", "d"):
        g = k_long[k_long.flavor.astype(str).eq(flavor)].copy()
        wide = g.pivot(index="kT", columns="seed", values="value").sort_index()
        vals = wide.to_numpy(float).T
        out = pd.DataFrame({"kT": wide.index.to_numpy(float), "flavor": flavor,
                            "q16": np.quantile(vals, .16, axis=0),
                            "central": np.quantile(vals, .50, axis=0),
                            "q84": np.quantile(vals, .84, axis=0)})
        k_rows.append(out)
    k_frame = pd.concat(k_rows, ignore_index=True)
    args.target.mkdir(parents=True, exist_ok=True)
    b_frame.to_csv(args.target / "fig2_bspace_start_replica_bands.csv", index=False)
    k_frame.to_csv(args.target / "fig6_kspace_start_replica_bands.csv", index=False)
    b_long.to_csv(args.target / "fig2_bspace_crossed_members_long.csv", index=False)
    k_long.to_csv(args.target / "fig6_kspace_crossed_members_long.csv", index=False)
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "axes.linewidth": .9,
                         "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True})
    colors = {"u": "#1f77b4", "d": "#d95f02"}
    labels = {"u": r"$u$ quark", "d": r"$d$ quark"}
    fig, ax = plt.subplots(figsize=(7.4, 5.0), constrained_layout=True)
    for flavor in ("u", "d"):
        g = b_frame[b_frame.flavor.eq(flavor) & (b_frame.bT <= 4.0)].sort_values("bT")
        ax.fill_between(g.bT, g.q16, g.q84, color=colors[flavor], alpha=.20, linewidth=0)
        ax.plot(g.bT, g.central, color=colors[flavor], lw=1.8, label=labels[flavor])
    title = r"$x=0.1\qquad Q=10\,\mathrm{GeV}$"
    if args.title_label:
        title += "\n" + str(args.title_label)
    ax.text(.98, .96, title, ha="right", va="top", transform=ax.transAxes, fontsize=11)
    ax.set(xlabel=r"$b_T\ [\mathrm{GeV}^{-1}]$", ylabel=r"$\widetilde f_1^q(x,b_T;Q)$", xlim=(0, 4))
    ax.grid(alpha=.15)
    h, l = ax.get_legend_handles_labels(); h.append(Patch(facecolor="0.5", alpha=.22, edgecolor="none")); l.append("propagated q16--q84 envelope")
    ax.legend(h, l, frameon=False, fontsize=9, ncol=2)
    fig.savefig(args.target / "fig2.png", dpi=240); fig.savefig(args.target / "fig2.pdf"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.4, 5.0), constrained_layout=True)
    for flavor in ("u", "d"):
        g = k_frame[k_frame.flavor.eq(flavor) & (k_frame.kT <= 2.25)].sort_values("kT")
        ax.fill_between(g.kT, g.q16, g.q84, color=colors[flavor], alpha=.20, linewidth=0)
        ax.plot(g.kT, g.central, color=colors[flavor], lw=1.8, label=labels[flavor])
    ax.text(.98, .96, title, ha="right", va="top", transform=ax.transAxes, fontsize=11)
    ax.set(xlabel=r"$k_T\ [\mathrm{GeV}]$", ylabel=r"$f_1^q(x,k_T;Q)$", xlim=(0, 2.25)); ax.set_ylim(bottom=0); ax.grid(alpha=.15)
    h, l = ax.get_legend_handles_labels(); h.append(Patch(facecolor="0.5", alpha=.22, edgecolor="none")); l.append("propagated q16--q84 envelope")
    ax.legend(h, l, frameon=False, fontsize=9)
    fig.savefig(args.target / "fig6.png", dpi=240); fig.savefig(args.target / "fig6.pdf"); plt.close(fig)
    inferred_replica_count = 50 if args.replica_count is None else int(args.replica_count)
    inferred_start_count = (int(fnp.member.nunique()) // inferred_replica_count
                            if args.start_count is None else int(args.start_count))
    summary = {"status": "isolated_scope_353_full_start_replica_figures_not_production",
               "source_fnp_cross": str(args.source), "perturbative_reference": str(CENTRAL),
               "start_count": inferred_start_count, "replica_count": inferred_replica_count,
               "crossed_member_count": int(fnp.member.nunique()), "band": "pointwise empirical q16-q84",
               "title_label": args.title_label,
               "transform": transform_meta, "frozen_production_modified": False, "promotion_authorized": False}
    (args.target / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
