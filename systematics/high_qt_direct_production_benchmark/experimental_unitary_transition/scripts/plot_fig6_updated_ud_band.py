#!/usr/bin/env python3
"""Clean isolated Fig. 6 pilot with updated u/d centers and 68% bands."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "systematics/high_qt_direct_production_benchmark/experimental_unitary_transition"
CENTRAL_BSPACE = (
    ROOT / "systematics/collins_factorization_validity/plots/"
    "rowidfix_stageFT_all_qmax0p20_lam0p50_central_exactx/"
    "v22_scheme_tmd_bspace_long.csv"
)
FNP_COMPARISON = BASE / "summaries/production_fnp_stability_control/fnp_grid_comparison.csv"
TRANSFORMER = ROOT / "workflows/v23a/construction/construct_v23a_regularized_kspace_tmd_v2.py"
REPLICA_DIR = ROOT / "plots/prd_q020_figures"
TARGET = BASE / "summaries/fig6_updated_ud_band"
SEEDS = tuple(range(303, 327))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def fnp_for_seed(seed: int) -> pd.DataFrame:
    path = BASE / f"outputs/fig6_lbfgs_stationary_s{seed}/fnp_grid.csv"
    grid = pd.read_csv(path)
    return grid[np.isclose(grid["x"], 0.1)].sort_values("bT")[["bT", "F_NP"]]


def transformed_starts(flavor: str) -> tuple[np.ndarray, np.ndarray, dict]:
    transform = load_module(f"fig6_updated_transform_{flavor}", TRANSFORMER)
    all_b = pd.read_csv(CENTRAL_BSPACE)
    central = all_b[
        np.isclose(all_b["x"], 0.1)
        & np.isclose(all_b["Q"], 10.0)
        & all_b["flavor"].astype(str).eq(flavor)
    ].sort_values("bT").copy()
    if len(central) < 3:
        raise RuntimeError(f"central b-space curve missing for flavor {flavor}")

    b = central["bT"].to_numpy(float)
    perturbative = central["ftilde_no_np"].to_numpy(float)
    curves = []
    for seed in SEEDS:
        grid = fnp_for_seed(seed)
        trial = central.copy()
        trial["F_NP"] = np.interp(b, grid["bT"], grid["F_NP"])
        trial["ftilde"] = perturbative * trial["F_NP"].to_numpy(float)
        trial["seed"] = seed
        trial["pdf_member"] = 0
        trial["_replica_key"] = f"s{seed}|pdf0"
        curves.append(trial)

    args = argparse.Namespace(
        quantities=["ftilde"], tail_mode="expb2", tail_fit_bmin=None,
        eps=1.0e-300, b_transform_max=24.0, n_b_transform=6001,
        k_max=4.0, n_k=401, end_taper_start_fraction=0.92,
    )
    long, meta = transform.transform_curves(pd.concat(curves, ignore_index=True), args)
    long["kT"] = long["kT"].round(10)
    wide = long.pivot(index="kT", columns="seed", values="value").sort_index()
    return wide.index.to_numpy(float), wide.to_numpy(float).T, meta


def updated_band(flavor: str) -> tuple[pd.DataFrame, dict]:
    k, starts, transform_meta = transformed_starts(flavor)
    replica_path = (
        REPLICA_DIR / f"kspace_fixedx_q10_{flavor}_current/"
        "v23a_regularized_kspace_replica_long.csv"
    )
    replicas = pd.read_csv(replica_path)
    replicas = replicas[
        replicas["quantity"].eq("ftilde")
        & replicas["flavor"].astype(str).eq(flavor)
        & np.isclose(replicas["x"], 0.1)
        & np.isclose(replicas["Q"], 10.0)
    ].copy()
    replicas["kT"] = replicas["kT"].round(10)
    wide = replicas.pivot(index="kT", columns="seed", values="value").sort_index()
    if len(wide.columns) != 50 or not np.array_equal(wide.index.to_numpy(float), k):
        raise RuntimeError(f"unexpected {flavor} replica ensemble or kT grid")

    replica_values = wide.to_numpy(float).T
    residuals = replica_values - np.median(replica_values, axis=0)
    crossed = (starts[:, None, :] + residuals[None, :, :]).reshape(-1, len(k))
    q16, median, q84 = np.quantile(crossed, [0.16, 0.50, 0.84], axis=0)
    out = pd.DataFrame({
        "flavor": flavor,
        "kT": k,
        "central": median,
        "q16": q16,
        "q84": q84,
    })
    active = (k <= 2.25) & (median > 0.05 * np.max(median[k <= 2.25]))
    half = len(SEEDS) // 2
    split_summaries = []
    for subset in (starts[:half], starts[half:]):
        subset_crossed = (subset[:, None, :] + residuals[None, :, :]).reshape(-1, len(k))
        split_summaries.append(np.quantile(subset_crossed, [0.16, 0.50, 0.84], axis=0))

    rng = np.random.default_rng(20260723)
    bootstrap = np.empty((300, 3, int(np.count_nonzero(active))), dtype=float)
    for index in range(len(bootstrap)):
        sampled = starts[rng.integers(0, len(starts), size=len(starts))]
        sampled_crossed = (sampled[:, None, :] + residuals[None, :, :]).reshape(-1, len(k))
        bootstrap[index] = np.quantile(sampled_crossed[:, active], [0.16, 0.50, 0.84], axis=0)
    bootstrap_low, bootstrap_high = np.quantile(bootstrap, [0.16, 0.84], axis=0)
    nominal_active = np.stack((q16[active], median[active], q84[active]))
    metrics = {
        "active_kT_max_GeV": float(k[active].max()),
        "max_lower_relative_68pct_excursion": float(np.max((median[active] - q16[active]) / median[active])),
        "max_upper_relative_68pct_excursion": float(np.max((q84[active] - median[active]) / median[active])),
        "split_half_start_stability": {
            "max_q16_difference_relative": float(np.max(
                np.abs(split_summaries[0][0, active] - split_summaries[1][0, active]) / median[active]
            )),
            "max_median_difference_relative": float(np.max(
                np.abs(split_summaries[0][1, active] - split_summaries[1][1, active]) / median[active]
            )),
            "max_q84_difference_relative": float(np.max(
                np.abs(split_summaries[0][2, active] - split_summaries[1][2, active]) / median[active]
            )),
        },
        "seed_bootstrap_68pct_numerical_uncertainty": {
            "resamples": len(bootstrap),
            "max_q16_endpoint_uncertainty_relative": float(np.max(
                np.maximum(nominal_active[0] - bootstrap_low[0],
                           bootstrap_high[0] - nominal_active[0]) / median[active]
            )),
            "max_median_uncertainty_relative": float(np.max(
                np.maximum(nominal_active[1] - bootstrap_low[1],
                           bootstrap_high[1] - nominal_active[1]) / median[active]
            )),
            "max_q84_endpoint_uncertainty_relative": float(np.max(
                np.maximum(nominal_active[2] - bootstrap_low[2],
                           bootstrap_high[2] - nominal_active[2]) / median[active]
            )),
        },
        "transform": transform_meta,
        "experimental_replica_source": str(replica_path),
    }
    return out, metrics


def main() -> None:
    u, u_metrics = updated_band("u")
    d, d_metrics = updated_band("d")
    display_u = u[u["kT"] <= 2.25]
    display_d = d[d["kT"] <= 2.25]

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.labelsize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })
    fig, ax = plt.subplots(figsize=(7.2, 5.4), constrained_layout=True)
    styles = {
        "u": ("#0072B2", display_u, r"$u$"),
        "d": ("#D55E00", display_d, r"$d$"),
    }
    for _, (color, data, label) in styles.items():
        ax.fill_between(
            data["kT"], data["q16"], data["q84"],
            color=color, alpha=0.15, linewidth=0,
        )
        ax.plot(data["kT"], data["central"], color=color, lw=1.55, label=label + " quark")

    ax.set_xlim(0, 2.25)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"$k_T\ (\mathrm{GeV})$")
    ax.set_ylabel(r"$f_1^q(x,k_T)$")
    ax.text(
        0.98, 0.93, r"$x=0.1\qquad Q=10\ \mathrm{GeV}$",
        transform=ax.transAxes, ha="right", va="top",
    )
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.tick_params(which="major", length=5.0, width=0.8)
    ax.tick_params(which="minor", length=2.5, width=0.6)
    ax.grid(which="major", color="0.88", linewidth=0.55, alpha=0.65)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="0.75", edgecolor="none", alpha=0.45))
    labels.append(r"provisional 68% ensemble interval")
    ax.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.66, 0.69),
        frameon=False, fontsize=10, handlelength=2.3,
    )

    TARGET.mkdir(parents=True, exist_ok=True)
    fig.savefig(TARGET / "fig6_updated_ud_central_1sigma.png", dpi=220)
    fig.savefig(TARGET / "fig6_updated_ud_central_1sigma.pdf")
    plt.close(fig)
    pd.concat((u, d), ignore_index=True).to_csv(
        TARGET / "fig6_updated_ud_central_1sigma.csv", index=False
    )
    summary = {
        "status": "isolated_updated_fig6_crossed_ensemble_pilot_not_production",
        "figure": "u and d updated crossed-ensemble pointwise medians with empirical q16-q84 bands",
        "x": 0.1,
        "Q_GeV": 10.0,
        "local_start_count": len(SEEDS),
        "experimental_replica_count_per_flavor": 50,
        "crossed_curve_count_per_flavor": len(SEEDS) * 50,
        "band_interpretation": (
            f"empirical central 68% interval under equal weights for {len(SEEDS)} stationary unanchored starts "
            "crossed with 50 conditional experimental-replica excursions; provisional, "
            "not yet a validated frequentist 1-sigma interval"
        ),
        "flavors": {"u": u_metrics, "d": d_metrics},
        "production_sources_modified": False,
    }
    (TARGET / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
