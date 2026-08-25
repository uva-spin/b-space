#!/usr/bin/env python3
"""Baseline-anchored W+Y shape diagnostics.

The isolated W+Y fits do not carry a directly comparable perturbative
normalization: they fit an external W grid, while the old Fig. 2/Fig. 6
plotter multiplied the fitted F_NP by a frozen reference curve.  This script
therefore uses the promoted lambda=1 production b-space central curve as the
common normalization and applies only the candidate-to-baseline F_NP ratio.

The resulting curves are shape/model diagnostics, not production replacements.
An additional b-anchor diagnostic forces the candidate-to-baseline ratio to
one at b_T=1 GeV^-1; it is explicitly not used as a physical prediction.
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
SYSTEMATICS = BASE.parent
REPORTS = BASE / "reports"
PRODUCTION = SYSTEMATICS / "dataset_identifiability_campaign_2026/production_lambda1_empirical_reference_full96x50"
BASE_BANDS = PRODUCTION / "bspace_combined_bands.csv"
BASE_LONG = SYSTEMATICS / "dataset_identifiability_campaign_2026/summaries/lambda1_start_expansion96_bspace/bspace_tmd_ensemble_long.csv"
BASE_K = PRODUCTION / "kspace_combined_bands.csv"
TRANSFORMER = SYSTEMATICS.parent / "construct_v23a_regularized_kspace_tmd_v2.py"
SEEDS = tuple(range(303, 311))

CASES = {
    "all_y": {
        "label": "all W+Y (1% perturbed starts)",
        "no_tail": "scope_329_perturbed1pct_all_y",
        "tail1": "scope_329_perturbed1pct_all_y_tail1",
    },
    "non_lhcb_y": {
        "label": "W+Y with LHCb Y set to zero (1% perturbed starts)",
        "no_tail": "scope_329_perturbed1pct_non_lhcb_y",
        "tail1": "scope_329_perturbed1pct_non_lhcb_y_tail1",
        "refdist1": "scope_329_perturbed1pct_non_lhcb_y_tail1_refdist1",
        "refdist3_b8": "scope_329_perturbed1pct_non_lhcb_y_tail1_refdist3_b8",
        "refdist3_b8_long10k": "scope_329_perturbed1pct_non_lhcb_y_tail1_refdist3_b8_long10k",
        "refdist3_b8_promoted96_long10k": "scope_329_perturbed1pct_non_lhcb_y_tail1_refdist3_b8_promoted96_long10k",
        "refdist1_b8_promoted96_long10k": "scope_329_perturbed1pct_non_lhcb_y_tail1_refdist1_b8_promoted96_long10k",
        "refdist2_b8_promoted96_long10k": "scope_329_perturbed1pct_non_lhcb_y_tail1_refdist2_b8_promoted96_long10k",
        "refdist4_b8_promoted96_long10k": "scope_329_perturbed1pct_non_lhcb_y_tail1_refdist4_b8_promoted96_long10k",
    },
    "lhcb_only_y": {
        "label": "W+Y with LHCb Y only (1% perturbed starts)",
        "no_tail": "scope_329_perturbed1pct_lhcb_only_y",
        "tail1": "scope_329_perturbed1pct_lhcb_only_y_tail1",
    },
}


def load_transformer():
    spec = importlib.util.spec_from_file_location("anchored_transform", TRANSFORMER)
    if spec is None or spec.loader is None:
        raise RuntimeError(TRANSFORMER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def baseline_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    bands = pd.read_csv(BASE_BANDS)
    long = pd.read_csv(BASE_LONG)
    long = long[np.isclose(long["x"].to_numpy(float), 0.1)
                & np.isclose(long["Q"].to_numpy(float), 10.0)].copy()
    # The production b-space bands are already the authoritative central and
    # experimental envelope.  The long ensemble supplies the corresponding
    # median F_NP needed to form candidate/baseline ratios.
    fnp = (long.groupby(["flavor", "bT"], observed=False)["F_NP"]
           .median().reset_index(name="baseline_F_NP"))
    return bands, fnp


def make_members(label: str, baseline: pd.DataFrame, baseline_fnp: pd.DataFrame,
                 b_anchor: float) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    members = []
    ratio_rows = []
    for member, seed in enumerate(SEEDS):
        directory = REPORTS / f"{label}_wy_s{seed}"
        grid = pd.read_csv(directory / "fnp_debug_grid.csv")
        grid = grid[np.isclose(grid["x"].to_numpy(float), 0.1)].sort_values("bT")
        if grid.empty:
            raise RuntimeError(f"missing x=0.1 F_NP grid: {directory}")
        for flavor in ("u", "d"):
            ref = baseline[baseline["flavor"].astype(str).eq(flavor)].sort_values("bT").copy()
            ref_fnp = baseline_fnp[baseline_fnp["flavor"].astype(str).eq(flavor)].sort_values("bT")
            b = ref["bT"].to_numpy(float)
            cand = np.interp(b, grid["bT"].to_numpy(float), grid["F_NP"].to_numpy(float))
            base_f = np.interp(b, ref_fnp["bT"].to_numpy(float), ref_fnp["baseline_F_NP"].to_numpy(float))
            ratio = cand / np.maximum(base_f, 1.0e-12)
            anchor_ratio = float(np.interp(b_anchor, b, ratio))
            shape_ratio = ratio / max(anchor_ratio, 1.0e-12)
            for kind, values in (("baseline_anchored", ratio), ("b1_shape_only", shape_ratio)):
                row = ref[["flavor", "bT"]].copy()
                row["x"] = 0.1
                row["Q"] = 10.0
                row["pid"] = 0
                row["seed"] = f"m{member}"
                row["pdf_member"] = 0
                row["_replica_key"] = f"m{member}|{kind}|{flavor}"
                row["quantity"] = kind
                row["ratio_to_baseline"] = values
                row["F_NP_candidate"] = cand
                row["F_NP_baseline"] = base_f
                row["ftilde"] = ref["central"].to_numpy(float) * values
                members.append(row)
            ratio_rows.append(pd.DataFrame({
                "member": member, "seed": seed, "flavor": flavor, "bT": b,
                "candidate_F_NP": cand, "baseline_F_NP": base_f,
                "ratio_to_baseline": ratio, "ratio_b1_shape_only": shape_ratio,
            }))
    return pd.concat(members, ignore_index=True), pd.concat(ratio_rows, ignore_index=True), {
        "start_count": len(SEEDS), "b_anchor_GeV_inv": b_anchor,
        "normalization": "promoted lambda=1 production b-space central multiplied by candidate/baseline F_NP ratio",
    }


def bands_from_members(members: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (flavor, quantity, b), g in members.groupby(["flavor", "quantity", "bT"], observed=False):
        vals = g["ftilde"].to_numpy(float)
        rows.append({"flavor": flavor, "quantity": quantity, "bT": float(b),
                     "q16": np.quantile(vals, .16), "central": np.quantile(vals, .50),
                     "q84": np.quantile(vals, .84)})
    return pd.DataFrame(rows)


def main() -> None:
    global SEEDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=("no_tail", "tail1", "refdist1", "refdist3_b8", "refdist3_b8_long10k", "refdist3_b8_promoted96_long10k", "refdist1_b8_promoted96_long10k", "refdist2_b8_promoted96_long10k", "refdist4_b8_promoted96_long10k"), default="tail1")
    ap.add_argument("--cases", nargs="+", choices=tuple(CASES), default=tuple(CASES))
    ap.add_argument("--target-root", type=Path,
                    default=REPORTS / "wy_baseline_anchored_diagnostics")
    ap.add_argument("--b-anchor", type=float, default=1.0)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS,
                    help="Start seeds represented by the selected diagnostic family.")
    args = ap.parse_args()
    SEEDS = tuple(int(v) for v in args.seeds)
    baseline, baseline_fnp = baseline_tables()
    baseline = baseline[np.isclose(baseline["bT"].to_numpy(float), baseline["bT"].to_numpy(float))]
    transformer = load_transformer()
    target_root = args.target_root / args.family
    target_root.mkdir(parents=True, exist_ok=True)
    baseline_fnp.to_csv(target_root / "baseline_fnp_reference_x0p1.csv", index=False)
    manifest = {"status": "isolated_baseline_anchored_wy_shape_diagnostic_not_production",
                "family": args.family, "cases": {}, "baseline_package": str(PRODUCTION),
                "baseline_fnp_reference": str(target_root / "baseline_fnp_reference_x0p1.csv"),
                "frozen_production_modified": False, "promotion_authorized": False}
    base_k = pd.read_csv(BASE_K)
    for name in args.cases:
        label = CASES[name][args.family]
        members, ratios, meta = make_members(label, baseline, baseline_fnp, args.b_anchor)
        out = target_root / name
        out.mkdir(parents=True, exist_ok=True)
        b_bands = bands_from_members(members)
        # Keep only the two requested transform quantities and let the existing
        # regularized transform handle the common b-to-k convention.
        transform_in = members.copy()
        transform_in["ftilde_anchor"] = np.where(
            transform_in["quantity"].eq("baseline_anchored"), transform_in["ftilde"], np.nan)
        transform_in["ftilde_shape"] = np.where(
            transform_in["quantity"].eq("b1_shape_only"), transform_in["ftilde"], np.nan)
        # A separate row per quantity is simpler and avoids NaN groups.
        chunks = []
        for q, col in (("baseline_anchored", "ftilde_anchor"), ("b1_shape_only", "ftilde_shape")):
            g = transform_in[transform_in["quantity"].eq(q)].copy()
            g["ftilde"] = g[col]
            chunks.append(g.drop(columns=["ftilde_anchor", "ftilde_shape"]))
        transform_in = pd.concat(chunks, ignore_index=True)
        settings = argparse.Namespace(
            quantities=["ftilde"], tail_mode="expb2", tail_fit_bmin=None,
            eps=1e-300, b_transform_max=24.0, n_b_transform=6001,
            k_max=4.0, n_k=401, end_taper_start_fraction=.92,
        )
        k_long, transform_meta = transformer.transform_curves(transform_in, settings)
        k_long["quantity"] = k_long["_replica_key"].str.extract(r"\|(baseline_anchored|b1_shape_only)\|")[0]
        k_bands = (k_long.groupby(["flavor", "quantity", "kT"], observed=False)["value"]
                   .quantile([.16, .50, .84]).unstack()
                   .rename(columns={.16: "q16", .50: "central", .84: "q84"})
                   .reset_index())
        members.to_csv(out / "bspace_members.csv", index=False)
        ratios.to_csv(out / "fnp_ratio_audit.csv", index=False)
        b_bands.to_csv(out / "bspace_anchored_bands.csv", index=False)
        k_long.to_csv(out / "kspace_members_long.csv", index=False)
        k_bands.to_csv(out / "kspace_anchored_bands.csv", index=False)

        plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                             "axes.linewidth": .9, "xtick.direction": "in",
                             "ytick.direction": "in", "xtick.top": True, "ytick.right": True})
        colors = {"u": "#1f77b4", "d": "#d95f02"}
        labels = {"u": r"$u$ quark", "d": r"$d$ quark"}
        base_cut = base_k[base_k["flavor"].astype(str).isin(["u", "d"])]
        for space, frame, xcol, base_frame, xmax, xlabel, ylabel in (
            ("bspace", b_bands, "bT", baseline, 4.0,
             r"$b_T\ [\mathrm{GeV}^{-1}]$", r"$\widetilde f_1^q(x,b_T;Q)$"),
            ("kspace", k_bands, "kT", base_cut, 2.25,
             r"$k_T\ [\mathrm{GeV}]$", r"$f_1^q(x,k_T;Q)$"),
        ):
            fig, ax = plt.subplots(figsize=(7.5, 5.1), constrained_layout=True)
            for flavor in ("u", "d"):
                b0 = base_frame[base_frame["flavor"].astype(str).eq(flavor)].sort_values(xcol)
                if space == "bspace":
                    b0 = b0[b0[xcol] <= xmax]
                    ax.fill_between(b0[xcol], b0.q16, b0.q84, color="0.55", alpha=.20, linewidth=0)
                    ax.plot(b0[xcol], b0.central, color="0.25", lw=1.2, ls="--")
                else:
                    b0 = b0[b0[xcol] <= xmax]
                    ax.fill_between(b0[xcol], b0.q16, b0.q84, color="0.55", alpha=.20, linewidth=0)
                    ax.plot(b0[xcol], b0.central, color="0.25", lw=1.2, ls="--")
                g = frame[frame.flavor.astype(str).eq(flavor) & frame.quantity.eq("baseline_anchored")].sort_values(xcol)
                ax.fill_between(g[xcol], g.q16, g.q84, color=colors[flavor], alpha=.20, linewidth=0)
                ax.plot(g[xcol], g.central, color=colors[flavor], lw=1.8, label=labels[flavor])
            ax.text(.98, .96, f"x=0.1   Q=10 GeV\n{CASES[name]['label']}\nbaseline-anchored shape diagnostic",
                    ha="right", va="top", transform=ax.transAxes, fontsize=10)
            ax.set(xlabel=xlabel, ylabel=ylabel, xlim=(0, xmax))
            if space == "kspace": ax.set_ylim(bottom=0)
            ax.grid(alpha=.15)
            handles, labs = ax.get_legend_handles_labels()
            handles += [Patch(facecolor="0.55", alpha=.20, edgecolor="none"),
                        plt.Line2D([], [], color="0.25", ls="--", lw=1.2)]
            labs += ["baseline empirical q16--q84", "baseline central"]
            ax.legend(handles, labs, frameon=False, fontsize=8.5)
            fig.savefig(out / f"{space}_baseline_anchored.png", dpi=240)
            fig.savefig(out / f"{space}_baseline_anchored.pdf")
            plt.close(fig)

            # A second view removes any candidate/baseline ratio at b_T=1.
            # This is useful for diagnosing shape changes, but is not a
            # physical normalization prescription.
            fig, ax = plt.subplots(figsize=(7.5, 5.1), constrained_layout=True)
            for flavor in ("u", "d"):
                b0 = base_frame[base_frame["flavor"].astype(str).eq(flavor)].sort_values(xcol)
                b0 = b0[b0[xcol] <= xmax]
                ax.fill_between(b0[xcol], b0.q16, b0.q84, color="0.55", alpha=.20, linewidth=0)
                ax.plot(b0[xcol], b0.central, color="0.25", lw=1.2, ls="--")
                g = frame[frame.flavor.astype(str).eq(flavor) & frame.quantity.eq("b1_shape_only")].sort_values(xcol)
                ax.fill_between(g[xcol], g.q16, g.q84, color=colors[flavor], alpha=.20, linewidth=0)
                ax.plot(g[xcol], g.central, color=colors[flavor], lw=1.8, label=labels[flavor])
            ax.text(.98, .96, f"x=0.1   Q=10 GeV\n{CASES[name]['label']}\nb_T=1 shape-only diagnostic",
                    ha="right", va="top", transform=ax.transAxes, fontsize=10)
            ax.set(xlabel=xlabel, ylabel=ylabel, xlim=(0, xmax))
            if space == "kspace": ax.set_ylim(bottom=0)
            ax.grid(alpha=.15)
            handles, labs = ax.get_legend_handles_labels()
            handles += [Patch(facecolor="0.55", alpha=.20, edgecolor="none"),
                        plt.Line2D([], [], color="0.25", ls="--", lw=1.2)]
            labs += ["baseline empirical q16--q84", "baseline central"]
            ax.legend(handles, labs, frameon=False, fontsize=8.5)
            fig.savefig(out / f"{space}_b1_shape_only.png", dpi=240)
            fig.savefig(out / f"{space}_b1_shape_only.pdf")
            plt.close(fig)

        # Ratio panel directly diagnoses whether the apparent change is a
        # normalization shift or a b-dependent shape change.
        fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
        for flavor in ("u", "d"):
            g = ratios[ratios.flavor.astype(str).eq(flavor)].sort_values("bT")
            vals = g.groupby("bT")["ratio_to_baseline"].quantile([.16, .50, .84]).unstack()
            vals = vals.rename(columns={.16: "q16", .50: "central", .84: "q84"}).reset_index()
            vals = vals[vals.bT <= 4.0]
            ax.fill_between(vals.bT, vals.q16, vals.q84, color=colors[flavor], alpha=.20)
            ax.plot(vals.bT, vals.central, color=colors[flavor], lw=1.8, label=labels[flavor])
        ax.axhline(1, color="0.25", ls="--", lw=1)
        ax.axvline(args.b_anchor, color="0.5", ls=":", lw=1)
        ax.set(xlabel=r"$b_T\ [\mathrm{GeV}^{-1}]$", ylabel=r"candidate $F_{\rm NP}$/baseline $F_{\rm NP}$",
               xlim=(0, 4))
        ax.set_title("baseline ratio: shape versus normalization", fontsize=11)
        ax.grid(alpha=.15); ax.legend(frameon=False, fontsize=9)
        fig.savefig(out / "fnp_ratio_to_baseline.png", dpi=240)
        fig.savefig(out / "fnp_ratio_to_baseline.pdf")
        plt.close(fig)
        (out / "summary.json").write_text(json.dumps({
            "status": "isolated_baseline_anchored_wy_shape_diagnostic_not_production",
            "case": name, "fit_label": label, **meta, "transform": transform_meta,
            "ratio_to_baseline_at_bT": {
                str(b): {
                    "q16": float(np.quantile(ratios.loc[np.isclose(ratios["bT"], b), "ratio_to_baseline"], .16)),
                    "median": float(np.quantile(ratios.loc[np.isclose(ratios["bT"], b), "ratio_to_baseline"], .50)),
                    "q84": float(np.quantile(ratios.loc[np.isclose(ratios["bT"], b), "ratio_to_baseline"], .84)),
                }
                for b in (0.0, 1.0, 2.0, 4.0, 8.0)
            },
            "interpretation": "baseline_anchored is physical b=0 normalization; b1_shape_only forces ratio=1 at b=1 and is diagnostic only",
            "baseline_package": str(PRODUCTION), "frozen_production_modified": False,
            "promotion_authorized": False,
        }, indent=2) + "\n")
        manifest["cases"][name] = {"fit_label": label, "target": str(out),
                                    "bspace": str(out / "bspace_baseline_anchored.png"),
                                    "kspace": str(out / "kspace_baseline_anchored.png"),
                                    "ratio": str(out / "fnp_ratio_to_baseline.png")}
    (target_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
