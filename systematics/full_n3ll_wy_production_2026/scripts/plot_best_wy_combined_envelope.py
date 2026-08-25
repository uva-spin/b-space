#!/usr/bin/env python3
"""Make isolated PRD-style W+Y candidate b- and k-space envelopes.

The candidate members are the baseline-anchored starts from
plot_wy_baseline_anchored_diagnostics.py.  In k-space, the existing 50-member
experimental replica excursions are transferred from their own median onto
the candidate-start median and then added directionally to the start spread.
This is an operational Minkowski envelope, not a calibrated 68% interval.
The b-space panel shows the start envelope only because a like-for-like
Q=10, x=0.1 experimental b-space replica table is not available in the
current isolated W+Y scope.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
SYSTEMATICS = BASE.parent
REPORTS = BASE / "reports"
DEFAULT_FAMILY = "refdist3_b8_promoted96_long10k"
DEFAULT_CASE = "non_lhcb_y"
EXP_U = SYSTEMATICS.parent / "plots/prd_q020_figures/kspace_fixedx_q10_u_current/v23a_regularized_kspace_bands.csv"
EXP_D = SYSTEMATICS.parent / "plots/prd_q020_figures/kspace_fixedx_q10_d_current/v23a_regularized_kspace_bands.csv"


def quantile_bands(frame: pd.DataFrame, xcol: str) -> pd.DataFrame:
    rows = []
    for flavor, gfl in frame.groupby(frame["flavor"].astype(str), observed=False):
        for x, g in gfl.groupby(xcol, observed=False):
            v = g["value"].to_numpy(float)
            rows.append({"flavor": flavor, xcol: float(x),
                         "q16": float(np.quantile(v, .16)),
                         "median": float(np.quantile(v, .50)),
                         "q84": float(np.quantile(v, .84))})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default=DEFAULT_FAMILY)
    ap.add_argument("--case", default=DEFAULT_CASE)
    ap.add_argument("--target", type=Path, default=None)
    args = ap.parse_args()
    source = REPORTS / "wy_baseline_anchored_diagnostics" / args.family / args.case
    target = args.target or (REPORTS / "wy_final_candidate_envelope" / args.family / args.case)
    target.mkdir(parents=True, exist_ok=True)

    members = pd.read_csv(source / "kspace_members_long.csv")
    members = members[members["quantity"].eq("baseline_anchored")].copy()
    members["flavor"] = members["flavor"].astype(str)
    kb = quantile_bands(members, "kT")

    exp = pd.concat([pd.read_csv(EXP_U), pd.read_csv(EXP_D)], ignore_index=True)
    exp = exp[exp["quantity"].eq("ftilde")].copy()
    exp["flavor"] = exp["flavor"].astype(str)
    exp = exp[["flavor", "kT", "q16", "median", "q84"]]
    exp = exp.rename(columns={"q16": "exp_q16", "median": "exp_median", "q84": "exp_q84"})
    out = kb.merge(exp, on=["flavor", "kT"], how="inner", validate="one_to_one")
    if len(out) != len(kb):
        raise RuntimeError("experimental k-space grid does not cover candidate grid")
    out["combined_low"] = out["q16"] + (out["exp_q16"] - out["exp_median"])
    out["combined_high"] = out["q84"] + (out["exp_q84"] - out["exp_median"])
    out["start_low"] = out["q16"]
    out["start_high"] = out["q84"]
    out.to_csv(target / "kspace_combined_envelope.csv", index=False)

    bmembers = pd.read_csv(source / "bspace_members.csv")
    bmembers = bmembers[bmembers["quantity"].eq("baseline_anchored")].copy()
    bmembers["flavor"] = bmembers["flavor"].astype(str)
    bb = quantile_bands(bmembers.rename(columns={"ftilde": "value"}), "bT")
    bb["combined_low"] = bb["q16"]
    bb["combined_high"] = bb["q84"]
    bb["experimental_component"] = "not_available_like_for_like_Q10_x0p1_bspace"
    bb.to_csv(target / "bspace_start_envelope.csv", index=False)

    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                         "axes.linewidth": .9, "xtick.direction": "in",
                         "ytick.direction": "in", "xtick.top": True, "ytick.right": True})
    colors = {"u": "#1f77b4", "d": "#d95f02"}
    labels = {"u": r"$u$ quark", "d": r"$d$ quark"}

    fig, ax = plt.subplots(figsize=(7.5, 5.1), constrained_layout=True)
    for flavor in ("u", "d"):
        g = bb[bb["flavor"].eq(flavor)].sort_values("bT")
        g = g[g["bT"] <= 4.0]
        ax.fill_between(g["bT"], g["combined_low"], g["combined_high"],
                        color=colors[flavor], alpha=.22, linewidth=0)
        ax.plot(g["bT"], g["median"], color=colors[flavor], lw=1.9, label=labels[flavor])
    ax.text(.98, .96, "x=0.1   Q=10 GeV\ncorrected non-LHCb-Y W+Y candidate\nstart envelope",
            ha="right", va="top", transform=ax.transAxes, fontsize=10)
    ax.set(xlabel=r"$b_T\ [\mathrm{GeV}^{-1}]$", ylabel=r"$\widetilde f_1^q(x,b_T;Q)$",
           xlim=(0, 4), ylim=(0, None))
    ax.grid(alpha=.15); ax.legend(frameon=False, fontsize=9)
    fig.savefig(target / "fig2_candidate_start_envelope.png", dpi=240)
    fig.savefig(target / "fig2_candidate_start_envelope.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.1), constrained_layout=True)
    for flavor in ("u", "d"):
        g = out[out["flavor"].eq(flavor)].sort_values("kT")
        g = g[g["kT"] <= 2.25]
        ax.fill_between(g["kT"], g["combined_low"], g["combined_high"],
                        color=colors[flavor], alpha=.22, linewidth=0)
        ax.plot(g["kT"], g["median"], color=colors[flavor], lw=1.9, label=labels[flavor])
    ax.text(.98, .96, "x=0.1   Q=10 GeV\ncorrected non-LHCb-Y W+Y candidate\nstart + experimental envelope",
            ha="right", va="top", transform=ax.transAxes, fontsize=10)
    ax.set(xlabel=r"$k_T\ [\mathrm{GeV}]$", ylabel=r"$f_1^q(x,k_T;Q)$",
           xlim=(0, 2.25), ylim=(0, None))
    ax.grid(alpha=.15); ax.legend(frameon=False, fontsize=9)
    fig.savefig(target / "fig6_candidate_combined_envelope.png", dpi=240)
    fig.savefig(target / "fig6_candidate_combined_envelope.pdf")
    plt.close(fig)

    active = out[out["kT"] <= 2.0].copy()
    metrics = {
        "status": "isolated_wy_candidate_envelope_not_production",
        "family": args.family, "case": args.case,
        "candidate_source": str(source),
        "experimental_k_source": [str(EXP_U), str(EXP_D)],
        "start_count": int(members["seed"].nunique()),
        "combination_rule": "candidate start q16/q84 plus directional experimental q16/q84 excursions about experimental median",
        "interpretation": "operational envelope; non-uniqueness is not assigned a calibrated confidence level",
        "bspace_experimental_component": "not included; no like-for-like Q=10 x=0.1 b-space replica table in this scope",
        "relative_combined_half_width_max_kT_le_2": float(np.max(.5 * (active["combined_high"]-active["combined_low"]) / np.maximum(active["median"], 1e-12))),
        "relative_start_half_width_max_kT_le_2": float(np.max(.5 * (active["q84"]-active["q16"]) / np.maximum(active["median"], 1e-12))),
        "frozen_production_modified": False,
        "promotion_authorized": False,
    }
    (target / "summary.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
