#!/usr/bin/env python3
"""Plot matched b-space/k-space diagnostics for current architecture studies."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
SYSTEMATICS = BASE.parent
ROOT = SYSTEMATICS.parent
UNITARY = (
    SYSTEMATICS
    / "high_qt_direct_production_benchmark/experimental_unitary_transition")
TARGET = BASE / "summaries/fig6_model_status_comparison"
BASELINE_K = (
    UNITARY / "summaries/fig6_updated_ud_band/"
    "fig6_updated_ud_central_1sigma.csv")
REFERENCE_B = (
    SYSTEMATICS / "collins_factorization_validity/plots/"
    "rowidfix_stageFT_all_qmax0p20_lam0p50_central_exactx/"
    "v22_scheme_tmd_bspace_long.csv")
REPLICA_B = (
    SYSTEMATICS / "collins_factorization_validity/replicas/"
    "rowidfix_stageFT_all_qmax0p20_lam0p50_lambda3_50rep/"
    "tmd_bspace_bands_exactx_50rep/v22_tmd_replica_bspace_long.csv")
FILM_B = (
    BASE / "summaries/selected_local_xslope3em4_multistart_bspace/"
    "bspace_tmd_ensemble_long.csv")
FILM_K = (
    BASE / "summaries/selected_local_xslope3em4_multistart_kspace/"
    "kspace_tmd_ensemble_long.csv")
SPLINE_B = (
    BASE / "summaries/diagnostic_best_spline_bspace/"
    "bspace_tmd_ensemble_long.csv")
SPLINE_K = (
    BASE / "summaries/diagnostic_best_spline_kspace/"
    "kspace_tmd_ensemble_long.csv")
PCA_B = (
    BASE / "summaries/diagnostic_empirical_pca_rank1_bspace/"
    "bspace_tmd_ensemble_long.csv")
PCA_K = (
    BASE / "summaries/diagnostic_empirical_pca_rank1_kspace/"
    "kspace_tmd_ensemble_long.csv")
BASELINE_START_SEEDS = tuple(range(303, 327))
ELIGIBLE_FILM = {
    "selected_local_xslope3em4_init304",
    "selected_local_xslope3em4_init305",
    "logcurv5em5_fslope4em3_xslope3em4_c2closure_b5p5_s1971_init307",
    "selected_local_xslope3em4_init312",
    "selected_local_xslope3em4_init313",
}
COLORS = {"u": "#0072B2", "d": "#D55E00"}


def quantile_frame(
        frame: pd.DataFrame, coordinate: str, member: str,
        value: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    result = {}
    for flavor in ("u", "d"):
        group = frame[frame["flavor"].astype(str).eq(flavor)]
        wide = group.pivot(
            index=coordinate, columns=member, values=value).sort_index()
        result[flavor] = (
            wide.index.to_numpy(float),
            np.quantile(wide.to_numpy(float), [0.16, 0.50, 0.84], axis=1))
    return result


def baseline_bspace() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    reference = pd.read_csv(REFERENCE_B)
    reference = reference[
        np.isclose(reference["x"], 0.1)
        & np.isclose(reference["Q"], 10.0)
        & reference["flavor"].astype(str).isin(["u", "d"])
    ]
    replicas = pd.read_csv(REPLICA_B)
    replicas = replicas[
        np.isclose(replicas["x"], 0.1)
        & np.isclose(replicas["Q"], 10.0)
        & replicas["flavor"].astype(str).isin(["u", "d"])
    ]
    result = {}
    for flavor in ("u", "d"):
        central = reference[
            reference["flavor"].astype(str).eq(flavor)].sort_values("bT")
        b = central["bT"].to_numpy(float)
        perturbative = central["ftilde_no_np"].to_numpy(float)
        starts = []
        for seed in BASELINE_START_SEEDS:
            grid = pd.read_csv(
                UNITARY / f"outputs/fig6_lbfgs_stationary_s{seed}/fnp_grid.csv")
            grid = grid[np.isclose(grid["x"], 0.1)].sort_values("bT")
            starts.append(
                perturbative * np.interp(
                    b, grid["bT"].to_numpy(float),
                    grid["F_NP"].to_numpy(float)))
        starts = np.asarray(starts)
        rep = replicas[
            replicas["flavor"].astype(str).eq(flavor)].pivot(
                index="bT", columns="seed", values="ftilde").sort_index()
        if not np.allclose(rep.index.to_numpy(float), b):
            raise RuntimeError("baseline replica and central b grids differ")
        rep_values = rep.to_numpy(float).T
        residuals = rep_values - np.median(rep_values, axis=0)
        crossed = (
            starts[:, None, :] + residuals[None, :, :]
        ).reshape(-1, len(b))
        result[flavor] = (
            b, np.quantile(crossed, [0.16, 0.50, 0.84], axis=0))
    return result


def main() -> None:
    baseline_b = baseline_bspace()
    baseline_k_frame = pd.read_csv(BASELINE_K)
    baseline_k = {
        flavor: (
            group["kT"].to_numpy(float),
            group[["q16", "central", "q84"]].to_numpy(float).T)
        for flavor, group in baseline_k_frame.groupby("flavor", sort=False)
    }

    film_b_frame = pd.read_csv(FILM_B)
    film_b_frame = film_b_frame[
        film_b_frame["run_tag"].astype(str).isin(ELIGIBLE_FILM)
        & np.isclose(film_b_frame["x"], 0.1)
        & np.isclose(film_b_frame["Q"], 10.0)
        & film_b_frame["flavor"].astype(str).isin(["u", "d"])]
    film_b = quantile_frame(film_b_frame, "bT", "run_tag", "ftilde")
    film_k_frame = pd.read_csv(FILM_K)
    film_k_frame = film_k_frame[
        film_k_frame["_replica_key"].astype(str).isin(ELIGIBLE_FILM)
        & film_k_frame["quantity"].eq("ftilde")
        & np.isclose(film_k_frame["x"], 0.1)
        & np.isclose(film_k_frame["Q"], 10.0)]
    film_k = quantile_frame(
        film_k_frame, "kT", "_replica_key", "value")

    single_cases = {}
    for name, b_path, k_path in (
        ("spline", SPLINE_B, SPLINE_K),
        ("pca", PCA_B, PCA_K),
    ):
        b_frame = pd.read_csv(b_path)
        b_frame = b_frame[
            np.isclose(b_frame["x"], 0.1)
            & np.isclose(b_frame["Q"], 10.0)
            & b_frame["flavor"].astype(str).isin(["u", "d"])]
        k_frame = pd.read_csv(k_path)
        k_frame = k_frame[
            k_frame["quantity"].eq("ftilde")
            & np.isclose(k_frame["x"], 0.1)
            & np.isclose(k_frame["Q"], 10.0)]
        single_cases[name] = {
            "b": quantile_frame(b_frame, "bT", "run_tag", "ftilde"),
            "k": quantile_frame(k_frame, "kT", "_replica_key", "value"),
        }

    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm",
        "axes.linewidth": 0.9, "xtick.direction": "in",
        "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    })
    fig, axes = plt.subplots(
        2, 4, figsize=(15.2, 7.5), sharey="row",
        constrained_layout=True)
    columns = [
        ("Previous baseline", baseline_b, baseline_k, True),
        ("Later constrained FiLM\n(rejected: broader)", film_b, film_k, True),
        ("Simple spline\n(rejected: fit)", single_cases["spline"]["b"],
         single_cases["spline"]["k"], False),
        ("Empirical PCA rank 1\n(preliminary)", single_cases["pca"]["b"],
         single_cases["pca"]["k"], False),
    ]
    for column, (title, b_data, k_data, has_band) in enumerate(columns):
        axes[0, column].set_title(title, fontsize=11)
        for row, data, coordinate, xmax in (
            (0, b_data, "b", 4.0), (1, k_data, "k", 2.25)):
            ax = axes[row, column]
            for flavor in ("u", "d"):
                grid, quantiles = data[flavor]
                selected = grid <= xmax
                if has_band:
                    ax.fill_between(
                        grid[selected], quantiles[0, selected],
                        quantiles[2, selected], color=COLORS[flavor],
                        alpha=0.20, linewidth=0)
                ax.plot(
                    grid[selected], quantiles[1, selected],
                    color=COLORS[flavor], lw=1.65,
                    label=rf"${flavor}$ quark")
            ax.set_xlim(0.0, xmax)
            ax.set_ylim(bottom=0.0)
            ax.grid(alpha=0.16)
            ax.text(
                0.97, 0.96, r"$x=0.1,\ Q=10\ \mathrm{GeV}$",
                ha="right", va="top", transform=ax.transAxes, fontsize=8.5)
            if row == 0:
                ax.axvline(2.0, color="0.35", ls=":", lw=0.9)
                ax.text(
                    2.03, 0.08, r"$b_T=2$", rotation=90,
                    transform=ax.get_xaxis_transform(), fontsize=8,
                    color="0.35")
                ax.set_xlabel(r"$b_T\ \mathrm{(GeV^{-1})}$")
            else:
                ax.set_xlabel(r"$k_T\ \mathrm{(GeV)}$")
        if column == 0:
            axes[0, column].legend(
                frameon=False, fontsize=8, loc="lower left")
    axes[0, 0].set_ylabel(r"$\widetilde f_1^q(x,b_T;Q)$")
    axes[1, 0].set_ylabel(r"$f_1^q(x,k_T;Q)$")
    axes[0, 2].text(
        0.04, 0.07, r"$\chi^2/N=4.62$", transform=axes[0, 2].transAxes,
        fontsize=9)
    axes[0, 3].text(
        0.04, 0.07, r"$\chi^2/N=0.412$; coverage untested",
        transform=axes[0, 3].transAxes, fontsize=9)
    fig.suptitle(
        "Architecture status in conjugate spaces — diagnostic, not final",
        fontsize=13)

    TARGET.mkdir(parents=True, exist_ok=True)
    png = TARGET / "fig2_fig6_architecture_status.png"
    pdf = TARGET / "fig2_fig6_architecture_status.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    summary = {
        "status": "isolated_diagnostic_not_final",
        "rows": ["b-space TMD", "regularized finite-b k-space transform"],
        "baseline": (
            "24 stationary starts crossed with 50 experimental-replica "
            "residuals; provisional q16-q84"),
        "later_film": (
            "five fit-admissible starts; descriptive q16-q84 "
            "nonuniqueness only"),
        "spline": "single rejected preflight; no band",
        "empirical_pca": "single preliminary rank-1 fit; no band",
        "production_sources_modified": False,
    }
    (TARGET / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(png)


if __name__ == "__main__":
    main()
