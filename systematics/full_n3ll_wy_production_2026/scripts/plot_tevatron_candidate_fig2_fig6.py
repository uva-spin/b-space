#!/usr/bin/env python3
"""Render isolated Tevatron full-N3LL+NNLO Gaussian candidate Fig. 2/6.

This is deliberately a candidate diagnostic.  The external DYTurbo grid fixes
the observable-level W+Y accuracy, while the TMD curves use the frozen
perturbative b-space factors multiplied by a common Gaussian NP factor.  The
band is the pointwise q16--q84 spread of the 50 diagonal pseudo-replica
profiles and is not a promoted production uncertainty.
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


SYSTEMATICS = Path(__file__).resolve().parents[2]
ROOT = SYSTEMATICS.parent
CENTRAL = ROOT / "systematics/collins_factorization_validity/plots/rowidfix_stageFT_all_qmax0p20_lam0p50_central_exactx/v22_scheme_tmd_bspace_long.csv"
REPLICAS = SYSTEMATICS / "full_n3ll_wy_production_2026/reports/tevatron_gaussian_np_replica_profile/replica_members.json"
TRANSFORMER = ROOT / "construct_v23a_regularized_kspace_tmd_v2.py"
TARGET = SYSTEMATICS / "full_n3ll_wy_production_2026/reports/tevatron_candidate_fig2_fig6_g1_1p0"


def load_transformer():
    spec = importlib.util.spec_from_file_location("candidate_fig_transform", TRANSFORMER)
    if spec is None or spec.loader is None:
        raise RuntimeError(TRANSFORMER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g1", type=float, default=1.0)
    parser.add_argument("--target-dir", type=Path, default=TARGET)
    parser.add_argument("--replica-source", type=Path, default=REPLICAS)
    args = parser.parse_args()
    central_all = pd.read_csv(CENTRAL)
    members = pd.DataFrame(json.loads(args.replica_source.read_text()))
    if len(members) < 20 or "g1_GeV2" not in members:
        raise RuntimeError("replica Gaussian profile is missing or too small")
    g1_values = members.g1_GeV2.to_numpy(float)
    curves = []
    b_bands = {}
    for flavor in ("u", "d"):
        central = central_all[
            np.isclose(central_all.x, 0.1)
            & np.isclose(central_all.Q, 10.0)
            & central_all.flavor.astype(str).eq(flavor)
        ].sort_values("bT").copy()
        if len(central) < 3:
            raise RuntimeError(f"missing frozen perturbative curve for {flavor}")
        b = central.bT.to_numpy(float)
        fpert = central.ftilde_no_np.to_numpy(float)
        b_members = []
        for index, g1 in enumerate(g1_values):
            trial = central.copy()
            trial["F_NP"] = np.exp(-g1 * b * b)
            trial["ftilde"] = fpert * trial.F_NP.to_numpy(float)
            trial["seed"] = str(index)
            trial["pdf_member"] = 0
            trial["_replica_key"] = f"g1rep{index}|pdf0"
            curves.append(trial)
            b_members.append(trial.ftilde.to_numpy(float))
        b_members = np.asarray(b_members)
        accepted = central.copy()
        accepted["F_NP"] = np.exp(-args.g1 * b * b)
        accepted["ftilde"] = fpert * accepted.F_NP.to_numpy(float)
        b_bands[flavor] = pd.DataFrame({
            "flavor": flavor, "bT": b,
            "central": accepted.ftilde.to_numpy(float),
            "q16": np.quantile(b_members, .16, axis=0),
            "q84": np.quantile(b_members, .84, axis=0),
        })

    transform = load_transformer()
    transform_args = argparse.Namespace(
        quantities=["ftilde"], tail_mode="expb2", tail_fit_bmin=None,
        eps=1.0e-300, b_transform_max=24.0, n_b_transform=6001,
        k_max=4.0, n_k=401, end_taper_start_fraction=0.92,
    )
    long, transform_meta = transform.transform_curves(pd.concat(curves, ignore_index=True), transform_args)
    long.kT = long.kT.round(10)
    k_bands = {}
    for flavor in ("u", "d"):
        subset = long[long.flavor.astype(str).eq(flavor)].copy()
        wide = subset.pivot(index="kT", columns="seed", values="value").sort_index()
        vals = wide.to_numpy(float).T
        central_b = b_bands[flavor]
        # The direct candidate is transformed separately to define the central line.
        direct = central_all[
            np.isclose(central_all.x, .1) & np.isclose(central_all.Q, 10.)
            & central_all.flavor.astype(str).eq(flavor)
        ].sort_values("bT").copy()
        direct["F_NP"] = np.exp(-args.g1 * direct.bT.to_numpy(float) ** 2)
        direct["ftilde"] = direct.ftilde_no_np * direct.F_NP
        direct["seed"] = "direct"
        direct["pdf_member"] = 0
        direct["_replica_key"] = "direct|pdf0"
        direct_long, _ = transform.transform_curves(direct, transform_args)
        direct_long.kT = direct_long.kT.round(10)
        direct_wide = direct_long.sort_values("kT")
        k_bands[flavor] = pd.DataFrame({
            "flavor": flavor,
            "kT": wide.index.to_numpy(float),
            "central": direct_wide.value.to_numpy(float),
            "q16": np.quantile(vals, .16, axis=0),
            "q84": np.quantile(vals, .84, axis=0),
        })

    target = args.target_dir.resolve()
    target.mkdir(parents=True, exist_ok=True)
    b_frame = pd.concat(list(b_bands.values()), ignore_index=True)
    k_frame = pd.concat(list(k_bands.values()), ignore_index=True)
    b_frame.to_csv(target / "candidate_fig2space_bT_ud.csv", index=False)
    k_frame.to_csv(target / "candidate_fig6_kT_ud.csv", index=False)
    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm",
        "axes.linewidth": .9, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
    })
    colors = {"u": "#1f77b4", "d": "#d95f02"}
    labels = {"u": r"$u$ quark", "d": r"$d$ quark"}
    g1_tag = "1p0" if np.isclose(args.g1, 1.0) else f"{args.g1:.6g}".replace(".", "p").replace("-", "m")
    for frame, xcol, xlabel, stem, xlim in (
        (b_frame, "bT", r"$b_T\ [\mathrm{GeV}^{-1}]$", f"candidate_fig2space_bT_ud_g1_{g1_tag}", (0, 4)),
        (k_frame[k_frame.kT <= 2.25], "kT", r"$k_T\ [\mathrm{GeV}]$", f"candidate_fig6_kT_ud_g1_{g1_tag}", (0, 2.25)),
    ):
        fig, ax = plt.subplots(figsize=(7.4, 5.0), constrained_layout=True)
        for flavor in ("u", "d"):
            curve = frame[frame.flavor.eq(flavor)].sort_values(xcol)
            ax.fill_between(curve[xcol], curve.q16, curve.q84, color=colors[flavor], alpha=.20, linewidth=0)
            ax.plot(curve[xcol], curve.central, color=colors[flavor], lw=1.8, label=labels[flavor])
        ax.text(.98, .96, r"$x=0.1\qquad Q=10\ \mathrm{GeV}$", ha="right", va="top", transform=ax.transAxes, fontsize=12)
        ax.set(xlabel=xlabel, ylabel=r"$\widetilde f_1^q(x,b_T;Q)$" if xcol == "bT" else r"$f_1^q(x,k_T;Q)$", xlim=xlim)
        if xcol == "kT":
            ax.set_ylim(bottom=0)
        ax.grid(alpha=.15)
        handles, legend_labels = ax.get_legend_handles_labels()
        handles.append(Patch(facecolor="0.5", alpha=.22, edgecolor="none"))
        legend_labels.append(r"candidate q16--q84 profile spread")
        ax.legend(handles, legend_labels, frameon=False, fontsize=10)
        fig.savefig(target / f"{stem}.png", dpi=240)
        fig.savefig(target / f"{stem}.pdf")
        plt.close(fig)
    summary = {
        "status": "isolated_tevatron_full_n3ll_nnlo_candidate_figures_not_production",
        "candidate_g1_GeV2": float(args.g1),
        "replica_count": int(len(g1_values)),
        "replica_source": str(args.replica_source),
        "central_source": str(CENTRAL),
        "band": "pointwise q16-q84 over diagonal Gaussian-g1 pseudo-replica profiles",
        "formal_confidence_level_assigned": False,
        "transform": transform_meta,
        "production_outputs_modified": False,
    }
    (target / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
