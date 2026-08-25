#!/usr/bin/env python3
"""Revisit PRD Fig. 6 for u, adding production-model non-uniqueness.

This is an isolated diagnostic.  It uses the exact experimental-replica band
and regularized finite-b transform settings used for the production figure,
then transforms the three unanchored production-objective local starts.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "systematics/high_qt_direct_production_benchmark/experimental_unitary_transition"
CENTRAL_BSPACE = ROOT / "systematics/collins_factorization_validity/plots/rowidfix_stageFT_all_qmax0p20_lam0p50_central_exactx/v22_scheme_tmd_bspace_long.csv"
EXPERIMENTAL_BAND = ROOT / "plots/prd_q020_figures/kspace_fixedx_q10_u_current/v23a_regularized_kspace_bands.csv"
FNP_COMPARISON = BASE / "summaries/production_fnp_stability_control/fnp_grid_comparison.csv"
TRANSFORMER = ROOT / "workflows/v23a/construction/construct_v23a_regularized_kspace_tmd_v2.py"
TARGET = BASE / "summaries/production_fnp_stability_control/kspace_u_x0p1_Q10"
SEEDS = (303, 304, 305)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    transform = load_module("production_kspace_transform", TRANSFORMER)
    central_all = pd.read_csv(CENTRAL_BSPACE)
    central = central_all[
        np.isclose(central_all["x"], 0.1)
        & np.isclose(central_all["Q"], 10.0)
        & central_all["flavor"].astype(str).eq("u")
    ].sort_values("bT").copy()
    if len(central) < 3:
        raise RuntimeError("accepted central u curve at x=0.1, Q=10 is missing")

    fnp_all = pd.read_csv(FNP_COMPARISON)
    fnp = fnp_all[np.isclose(fnp_all["x"], 0.1)].sort_values("bT")
    if len(fnp) < 3:
        raise RuntimeError("production-control FNP grid at x=0.1 is missing")

    curves = []
    accepted = central.copy()
    accepted["seed"] = "accepted"
    accepted["pdf_member"] = 0
    accepted["_replica_key"] = "accepted|pdf0"
    curves.append(accepted)
    b = central["bT"].to_numpy(float)
    perturbative = central["ftilde_no_np"].to_numpy(float)
    for seed in SEEDS:
        trial = central.copy()
        trial["F_NP"] = np.interp(b, fnp["bT"], fnp[f"fnp_s{seed}"])
        trial["ftilde"] = perturbative * trial["F_NP"].to_numpy(float)
        trial["seed"] = str(seed)
        trial["pdf_member"] = 0
        trial["_replica_key"] = f"s{seed}|pdf0"
        curves.append(trial)
    bspace = pd.concat(curves, ignore_index=True)

    args = argparse.Namespace(
        quantities=["ftilde"], tail_mode="expb2", tail_fit_bmin=None,
        eps=1.0e-300, b_transform_max=24.0, n_b_transform=6001,
        k_max=4.0, n_k=401, end_taper_start_fraction=0.92,
    )
    k_long, transform_meta = transform.transform_curves(bspace, args)
    wide = k_long.pivot(index="kT", columns="seed", values="value").reset_index()
    for col in ("accepted", "303", "304", "305"):
        if col not in wide:
            raise RuntimeError(f"transformed curve {col} is missing")

    exp = pd.read_csv(EXPERIMENTAL_BAND)
    exp = exp[
        exp["quantity"].eq("ftilde") & exp["flavor"].astype(str).eq("u")
        & np.isclose(exp["x"], 0.1) & np.isclose(exp["Q"], 10.0)
    ].sort_values("kT")
    out = wide.merge(exp[["kT", "q16", "median", "q84"]], on="kT", validate="one_to_one")
    central_k = out["accepted"].to_numpy(float)
    alternatives = out[["303", "304", "305"]].to_numpy(float)
    delta = alternatives - central_k[:, None]
    out["uniqueness_delta_low"] = np.minimum(np.min(delta, axis=1), 0.0)
    out["uniqueness_delta_high"] = np.maximum(np.max(delta, axis=1), 0.0)
    # Transfer replica excursions about their median onto the accepted curve.
    out["experimental_low"] = out["accepted"] + (out["q16"] - out["median"])
    out["experimental_high"] = out["accepted"] + (out["q84"] - out["median"])
    out["uniqueness_low"] = out["accepted"] + out["uniqueness_delta_low"]
    out["uniqueness_high"] = out["accepted"] + out["uniqueness_delta_high"]
    # Conservative envelope/Minkowski sum; this is not assigned a confidence level.
    out["combined_low"] = out["experimental_low"] + out["uniqueness_delta_low"]
    out["combined_high"] = out["experimental_high"] + out["uniqueness_delta_high"]

    display = out[out["kT"] <= 2.25].copy()
    peak = float(display["accepted"].max())
    active = display["accepted"] > 0.05 * peak
    denom = display["accepted"].to_numpy(float)
    exp_half = 0.5 * (display["experimental_high"] - display["experimental_low"]).to_numpy(float)
    uniq_excursion = np.maximum(
        (display["uniqueness_high"] - display["accepted"]).to_numpy(float),
        (display["accepted"] - display["uniqueness_low"]).to_numpy(float),
    )
    comb_excursion = np.maximum(
        (display["combined_high"] - display["accepted"]).to_numpy(float),
        (display["accepted"] - display["combined_low"]).to_numpy(float),
    )

    fig, axes = plt.subplots(2, 1, figsize=(8.1, 7.5), sharex=True,
                             gridspec_kw={"height_ratios": [2.0, 1.0]}, constrained_layout=True)
    ax = axes[0]
    ax.fill_between(display.kT, display.combined_low, display.combined_high,
                    color="#D55E00", alpha=0.20, label="experimental + non-uniqueness envelope")
    ax.fill_between(display.kT, display.uniqueness_low, display.uniqueness_high,
                    color="#CC79A7", alpha=0.30, label="production non-uniqueness")
    ax.fill_between(display.kT, display.experimental_low, display.experimental_high,
                    color="0.60", alpha=0.35, label="68% experimental replicas")
    for seed, color in zip(SEEDS, ("#0072B2", "#009E73", "#E69F00")):
        ax.plot(display.kT, display[str(seed)], color=color, lw=0.9, alpha=0.85,
                label=f"unanchored start s{seed}")
    ax.plot(display.kT, display.accepted, color="black", lw=2.0, label="accepted central")
    ax.set_ylabel(r"$f_1^u(x,k_T;Q)$")
    ax.set_title(r"Production-model diagnostic: $u$, $x=0.1$, $Q=10\,\mathrm{GeV}$")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.20)
    ax.legend(loc="upper right", frameon=True, framealpha=0.92, edgecolor="0.85", fontsize=7.5)

    ax = axes[1]
    ax.plot(display.kT, 100 * exp_half / denom, color="0.35", lw=2, label="experimental 68% half-width")
    ax.plot(display.kT, 100 * uniq_excursion / denom, color="#CC79A7", lw=2, label="max non-uniqueness excursion")
    ax.plot(display.kT, 100 * comb_excursion / denom, color="#D55E00", lw=2, label="max combined excursion")
    ax.axvspan(display.loc[~active, "kT"].min() if np.any(~active) else 2.25, 2.25,
               color="0.85", alpha=0.35, zorder=-2, label="below 5% of peak")
    ax.set_xlabel(r"$k_T\ [\mathrm{GeV}]$")
    ax.set_ylabel("relative width [%]")
    ax.set_xlim(0, 2.25)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.20)
    ax.legend(frameon=False, fontsize=8, ncol=2)

    TARGET.mkdir(parents=True, exist_ok=True)
    fig.savefig(TARGET / "u_kspace_experimental_plus_nonuniqueness.png", dpi=220)
    fig.savefig(TARGET / "u_kspace_experimental_plus_nonuniqueness.pdf")
    plt.close(fig)
    out.to_csv(TARGET / "u_kspace_uncertainty_components.csv", index=False)

    active_array = active.to_numpy(bool)
    metrics = {
        "status": "isolated_production_model_diagnostic_not_production",
        "figure_revisited": "b_space_PRD.pdf Fig. 6, u-quark panel",
        "x": 0.1, "Q_GeV": 10.0, "flavor": "u",
        "display_kT_max_GeV": 2.25,
        "active_definition": "accepted curve > 5% of its displayed peak",
        "active_kT_max_GeV": float(display.loc[active, "kT"].max()),
        "experimental_band": "central 68% experimental replica band from accepted anchored 50-replica ensemble",
        "nonuniqueness_band": "pointwise envelope of three unanchored local starts relative to accepted central",
        "combined_rule": "directional Minkowski sum of experimental excursions and nonuniqueness excursions; no confidence-level interpretation",
        "active_region": {
            "max_experimental_relative_halfwidth": float(np.max(exp_half[active_array] / denom[active_array])),
            "max_nonuniqueness_relative_excursion": float(np.max(uniq_excursion[active_array] / denom[active_array])),
            "max_combined_relative_excursion": float(np.max(comb_excursion[active_array] / denom[active_array])),
            "median_experimental_relative_halfwidth": float(np.median(exp_half[active_array] / denom[active_array])),
            "median_nonuniqueness_relative_excursion": float(np.median(uniq_excursion[active_array] / denom[active_array])),
        },
        "transform": transform_meta,
        "sources": {
            "central_bspace": str(CENTRAL_BSPACE), "experimental_band": str(EXPERIMENTAL_BAND),
            "production_start_FNPs": str(FNP_COMPARISON), "transform_implementation": str(TRANSFORMER),
        },
        "production_sources_modified": False,
    }
    (TARGET / "summary.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
