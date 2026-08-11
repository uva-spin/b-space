#!/usr/bin/env python3
"""Isolated Fig. 6 pilot crossing local-start and experimental ensembles.

The existing experimental replicas were trained conditionally around the
accepted anchored solution.  This diagnostic transfers their pointwise
excursions to each unanchored production-objective start, producing a crossed
6-start x 50-replica ensemble.  It is an approximation to, not a replacement
for, refitting every experimental replica from multiple independent starts.
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
CENTRAL_BSPACE = (
    ROOT
    / "systematics/collins_factorization_validity/plots/"
    "rowidfix_stageFT_all_qmax0p20_lam0p50_central_exactx/"
    "v22_scheme_tmd_bspace_long.csv"
)
REPLICA_LONG = (
    ROOT
    / "plots/prd_q020_figures/kspace_fixedx_q10_u_current/"
    "v23a_regularized_kspace_replica_long.csv"
)
FNP_COMPARISON = BASE / "summaries/production_fnp_stability_control/fnp_grid_comparison.csv"
PRIOR_COMPONENTS = (
    BASE / "summaries/production_fnp_stability_control/kspace_u_x0p1_Q10/"
    "u_kspace_uncertainty_components.csv"
)
TRANSFORMER = ROOT / "construct_v23a_regularized_kspace_tmd_v2.py"
TARGET = BASE / "summaries/fig6_crossed_start_replica_pilot"
SEEDS = (303, 304, 305, 306, 307, 308)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def central_start_curves() -> tuple[pd.DataFrame, dict]:
    transform = load_module("fig6_crossed_transform", TRANSFORMER)
    all_b = pd.read_csv(CENTRAL_BSPACE)
    central = all_b[
        np.isclose(all_b["x"], 0.1)
        & np.isclose(all_b["Q"], 10.0)
        & all_b["flavor"].astype(str).eq("u")
    ].sort_values("bT").copy()
    fnp_all = pd.read_csv(FNP_COMPARISON)
    fnp = fnp_all[np.isclose(fnp_all["x"], 0.1)].sort_values("bT")
    if len(central) < 3 or len(fnp) < 3:
        raise RuntimeError("required accepted or local-start curves are missing")

    b = central["bT"].to_numpy(float)
    perturbative = central["ftilde_no_np"].to_numpy(float)
    curves = []
    for seed in SEEDS:
        trial = central.copy()
        if f"fnp_s{seed}" in fnp:
            seed_fnp = fnp[["bT", f"fnp_s{seed}"]].rename(columns={f"fnp_s{seed}": "F_NP"})
        else:
            seed_fnp = pd.read_csv(BASE / f"outputs/fig6_newcenter_unanchored_s{seed}/fnp_grid.csv")
            seed_fnp = seed_fnp[np.isclose(seed_fnp["x"], 0.1)].sort_values("bT")
        trial["F_NP"] = np.interp(b, seed_fnp["bT"], seed_fnp["F_NP"])
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
    transformed, meta = transform.transform_curves(pd.concat(curves, ignore_index=True), args)
    return transformed[["seed", "kT", "value"]].copy(), meta


def main() -> None:
    starts, transform_meta = central_start_curves()
    starts["kT"] = starts["kT"].round(10)
    start_wide = starts.pivot(index="kT", columns="seed", values="value").sort_index()
    replicas = pd.read_csv(REPLICA_LONG)
    replicas = replicas[
        replicas["quantity"].eq("ftilde")
        & replicas["flavor"].astype(str).eq("u")
        & np.isclose(replicas["x"], 0.1)
        & np.isclose(replicas["Q"], 10.0)
    ]
    replicas = replicas.copy()
    replicas["kT"] = replicas["kT"].round(10)
    replica_wide = replicas.pivot(index="kT", columns="seed", values="value").sort_index()
    if len(replica_wide.columns) != 50:
        raise RuntimeError(f"expected 50 experimental replicas, found {len(replica_wide.columns)}")
    if not start_wide.index.equals(replica_wide.index):
        raise RuntimeError("start and experimental-replica kT grids differ")

    k = start_wide.index.to_numpy(float)
    start_values = start_wide.to_numpy(float).T  # start, kT
    replica_values = replica_wide.to_numpy(float).T  # replica, kT
    replica_reference = np.median(replica_values, axis=0)
    residuals = replica_values - replica_reference
    joint = start_values[:, None, :] + residuals[None, :, :]
    joint_flat = joint.reshape(-1, len(k))

    center_mean = np.mean(joint_flat, axis=0)
    center_median = np.median(joint_flat, axis=0)
    q16, q84 = np.quantile(joint_flat, [0.16, 0.84], axis=0)
    exp_q16, exp_q84 = np.quantile(replica_values, [0.16, 0.84], axis=0)
    prior = pd.read_csv(PRIOR_COMPONENTS).sort_values("kT")
    prior["kT"] = prior["kT"].round(10)
    accepted = np.interp(k, prior["kT"].to_numpy(float), prior["accepted"].to_numpy(float))

    active = (k <= 2.25) & (center_median > 0.05 * np.max(center_median[k <= 2.25]))
    scale = np.maximum(0.5 * (q84 - q16), 1.0e-12)
    medoid_scores = np.mean(
        np.square((start_values[:, active] - center_median[active]) / scale[active]), axis=1
    )
    medoid_index = int(np.argmin(medoid_scores))
    medoid_seed = SEEDS[medoid_index]
    center_medoid = start_values[medoid_index]

    # The empirical central interval is asymmetric about any chosen center.
    median_low = center_median - q16
    median_high = q84 - center_median
    medoid_low = center_medoid - q16
    medoid_high = q84 - center_medoid
    start_sigma = np.std(start_values, axis=0, ddof=1)
    exp_sigma = np.std(residuals, axis=0, ddof=1)
    gaussian_sigma = np.sqrt(start_sigma**2 + exp_sigma**2)

    def crossed_quantiles(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        crossed = (values[:, None, :] + residuals[None, :, :]).reshape(-1, len(k))
        return (
            np.quantile(crossed, 0.16, axis=0),
            np.median(crossed, axis=0),
            np.quantile(crossed, 0.84, axis=0),
        )

    early_q16, early_median, early_q84 = crossed_quantiles(start_values[:3])
    late_q16, late_median, late_q84 = crossed_quantiles(start_values[3:])

    out = pd.DataFrame({
        "kT": k,
        "frozen_accepted": accepted,
        "joint_mean": center_mean,
        "joint_median": center_median,
        "medoid": center_medoid,
        "joint_q16": q16,
        "joint_q84": q84,
        "conditional_replica_q16": exp_q16,
        "conditional_replica_q84": exp_q84,
        "median_minus_q16": median_low,
        "q84_minus_median": median_high,
        "medoid_minus_q16": medoid_low,
        "q84_minus_medoid": medoid_high,
        "start_sample_sigma": start_sigma,
        "experimental_sample_sigma": exp_sigma,
        "quadrature_random_effect_sigma": gaussian_sigma,
    })
    for index, seed in enumerate(SEEDS):
        out[f"start_{seed}"] = start_values[index]

    display = out[out["kT"] <= 2.25]
    fig, axes = plt.subplots(
        2, 1, figsize=(8.2, 7.4), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]}, constrained_layout=True,
    )
    ax = axes[0]
    ax.fill_between(
        display.kT, display.joint_q16, display.joint_q84,
        color="#D55E00", alpha=0.25, label="crossed ensemble central 68%",
    )
    ax.fill_between(
        display.kT, display.conditional_replica_q16, display.conditional_replica_q84,
        color="0.60", alpha=0.30, label="original conditional 68%",
    )
    colors = ("#0072B2", "#009E73", "#E69F00", "#56B4E9", "#CC79A7", "#8C564B")
    for seed, color in zip(SEEDS, colors):
        ax.plot(display.kT, display[f"start_{seed}"], color=color, lw=0.9, alpha=0.75,
                label=f"start {seed}")
    ax.plot(display.kT, display.joint_median, color="black", lw=2.0,
            label="crossed-ensemble median")
    ax.plot(display.kT, display.frozen_accepted, color="0.30", lw=1.2, ls="-.",
            label="frozen accepted central")
    ax.plot(display.kT, display.medoid, color="#6A3D9A", lw=1.4, ls="--",
            label=f"curve medoid (start {medoid_seed})")
    ax.set_ylim(bottom=0)
    ax.set_ylabel(r"$f_1^u(x,k_T;Q)$")
    ax.set_title(r"Fig. 6 pilot: $u$, $x=0.1$, $Q=10\,\mathrm{GeV}$")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", frameon=True, framealpha=0.93, fontsize=7.4, ncol=2)

    ax = axes[1]
    denom = display.joint_median.to_numpy(float)
    ax.plot(display.kT, 100 * display.median_minus_q16 / denom,
            color="#D55E00", lw=1.8, label="full 68% lower excursion")
    ax.plot(display.kT, 100 * display.q84_minus_median / denom,
            color="#D55E00", lw=1.8, ls="--", label="full 68% upper excursion")
    ax.plot(display.kT, 100 * display.experimental_sample_sigma / denom,
            color="0.35", lw=1.6, label="experimental sample σ")
    ax.plot(display.kT, 100 * display.quadrature_random_effect_sigma / denom,
            color="#0072B2", lw=1.4, ls=":", label="Gaussian random-effect check")
    ax.set_xlim(0, 2.25)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"$k_T\ [\mathrm{GeV}]$")
    ax.set_ylabel("relative uncertainty [%]")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=7.5, ncol=2)

    TARGET.mkdir(parents=True, exist_ok=True)
    fig.savefig(TARGET / "fig6_u_crossed_start_replica_pilot.png", dpi=220)
    fig.savefig(TARGET / "fig6_u_crossed_start_replica_pilot.pdf")
    plt.close(fig)
    out.to_csv(TARGET / "fig6_u_crossed_ensemble.csv", index=False)

    active_out = out.loc[active]
    summary = {
        "status": "isolated_crossed_start_replica_pilot_not_production",
        "statistical_interpretation": (
            "empirical central 68% interval under equal discrete weights for six local starts "
            "and 50 experimental replicas; provisional because starts did not converge and "
            "replicas were not independently refit from each start"
        ),
        "joint_ensemble_size": int(joint_flat.shape[0]),
        "local_start_count": len(SEEDS),
        "experimental_replica_count": int(replica_values.shape[0]),
        "central_estimators": {
            "recommended_realizable_curve_for_this_pilot": f"start_{medoid_seed}",
            "medoid_scores": {str(seed): float(score) for seed, score in zip(SEEDS, medoid_scores)},
            "reported_distribution_center": "pointwise median",
            "mean_median_max_relative_difference_active": float(np.max(
                np.abs(active_out["joint_mean"] - active_out["joint_median"])
                / active_out["joint_median"]
            )),
        },
        "split_start_ensemble_stability": {
            "first_three_seeds": list(SEEDS[:3]),
            "last_three_seeds": list(SEEDS[3:]),
            "max_median_center_difference_relative_active": float(np.max(
                np.abs(early_median[active] - late_median[active]) / center_median[active]
            )),
            "max_lower_quantile_difference_relative_active": float(np.max(
                np.abs(early_q16[active] - late_q16[active]) / center_median[active]
            )),
            "max_upper_quantile_difference_relative_active": float(np.max(
                np.abs(early_q84[active] - late_q84[active]) / center_median[active]
            )),
            "gate_pass": False,
            "reason": "central fits did not plateau and three-start estimates change materially when seeds are replaced",
        },
        "active_definition": "joint median > 5% of displayed peak and kT <= 2.25 GeV",
        "active_kT_max_GeV": float(active_out["kT"].max()),
        "active_region": {
            "max_new_center_shift_from_frozen_relative": float(np.max(
                np.abs(active_out["joint_median"] - active_out["frozen_accepted"])
                / active_out["frozen_accepted"]
            )),
            "median_new_center_shift_from_frozen_relative": float(np.median(
                np.abs(active_out["joint_median"] - active_out["frozen_accepted"])
                / active_out["frozen_accepted"]
            )),
            "max_lower_68pct_relative_excursion": float(np.max(
                active_out["median_minus_q16"] / active_out["joint_median"]
            )),
            "max_upper_68pct_relative_excursion": float(np.max(
                active_out["q84_minus_median"] / active_out["joint_median"]
            )),
            "median_lower_68pct_relative_excursion": float(np.median(
                active_out["median_minus_q16"] / active_out["joint_median"]
            )),
            "median_upper_68pct_relative_excursion": float(np.median(
                active_out["q84_minus_median"] / active_out["joint_median"]
            )),
            "max_start_sample_sigma_relative": float(np.max(
                active_out["start_sample_sigma"] / active_out["joint_median"]
            )),
            "max_experimental_sample_sigma_relative": float(np.max(
                active_out["experimental_sample_sigma"] / active_out["joint_median"]
            )),
        },
        "required_validation": (
            "multi-start refits of the same experimental replicas, plus enough converged "
            "central starts to demonstrate stable start weights and quantiles"
        ),
        "transform": transform_meta,
        "sources": {
            "central_bspace": str(CENTRAL_BSPACE),
            "production_start_FNPs": str(FNP_COMPARISON),
            "experimental_replica_curves": str(REPLICA_LONG),
            "frozen_accepted_curve": str(PRIOR_COMPONENTS),
        },
        "production_sources_modified": False,
    }
    (TARGET / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
