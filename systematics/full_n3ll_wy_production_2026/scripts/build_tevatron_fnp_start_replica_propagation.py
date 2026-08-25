#!/usr/bin/env python3
"""Build the isolated Tevatron F_NP/start/replica TMD propagation.

The perturbative DYTurbo run and the fitted F_NP ensemble are intentionally
kept as separate auditable inputs.  This script crosses the complete isolated
lambda=1 start ensemble with the available experimental-replica residuals in
both b and k space, and writes a single central curve (the empirical median)
plus pointwise q16--q84 bands.  The non-uniqueness component is not assigned a
Gaussian confidence interpretation.

The external W+Y and conventional-Y status files are required for a matched
candidate run.  ``--allow-missing-external`` is provided only for an early
propagation smoke test while the high-statistics DYTurbo jobs are running; its
output is explicitly marked incomplete.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


SYSTEMATICS = Path(__file__).resolve().parents[1].parent
ROOT = SYSTEMATICS.parent
CAMPAIGN = SYSTEMATICS / "dataset_identifiability_campaign_2026"
START_B = CAMPAIGN / "summaries/lambda1_start_expansion96_bspace/bspace_tmd_ensemble_long.csv"
START_K = CAMPAIGN / "summaries/lambda1_start_expansion96_kspace/kspace_tmd_ensemble_long.csv"
REFERENCE_B = (
    SYSTEMATICS / "collins_factorization_validity/plots/"
    "rowidfix_stageFT_all_qmax0p20_lam0p50_central_exactx/"
    "v22_scheme_tmd_bspace_long.csv"
)
REPLICA_B = (
    SYSTEMATICS / "collins_factorization_validity/replicas/"
    "rowidfix_stageFT_all_qmax0p20_lam0p50_lambda3_50rep/"
    "tmd_bspace_bands_exactx_50rep/v22_tmd_replica_bspace_long.csv"
)
REPLICA_K_ROOT = ROOT / "plots/prd_q020_figures"
EXTERNAL_ROOT = SYSTEMATICS / "full_n3ll_wy_production_2026/reports"
EXTERNAL_GRID = EXTERNAL_ROOT / "tevatron_n3ll_nnlo_wy_final_g1_1p017/tevatron_full_wy_grid.csv"
EXTERNAL_STATUS = EXTERNAL_ROOT / "tevatron_n3ll_nnlo_wy_final_g1_1p017/grid_status.json"
Y_STATUS = EXTERNAL_ROOT / "tevatron_n3ll_nnlo_wy_final_g1_1p017/conventional_y/y_grid_status.json"
TARGET = EXTERNAL_ROOT / "tevatron_fnp_start_replica_propagation"

COLORS = {
    "u": "#1f77b4", "d": "#d95f02", "s": "#2ca02c",
    "ubar": "#9467bd", "dbar": "#8c564b", "sbar": "#e377c2",
}
LABELS = {
    "u": r"$u$ quark", "d": r"$d$ quark", "s": r"$s$ quark",
    "ubar": r"$\bar u$ quark", "dbar": r"$\bar d$ quark",
    "sbar": r"$\bar s$ quark",
}
EPS = 1.0e-30


def crossed_quantiles(starts: np.ndarray, replicas: np.ndarray) -> np.ndarray:
    """Cross correlated start curves with centered replica residual curves."""
    if starts.ndim != 2 or replicas.ndim != 2 or starts.shape[1] != replicas.shape[1]:
        raise ValueError("start and replica curves do not share a coordinate grid")
    residuals = replicas - np.median(replicas, axis=0)
    crossed = (starts[:, None, :] + residuals[None, :, :]).reshape(
        -1, starts.shape[1])
    return np.quantile(crossed, [0.16, 0.50, 0.84], axis=0)


def require_members(frame: pd.DataFrame, member: str, name: str, count: int) -> None:
    if member not in frame.columns:
        raise RuntimeError(f"{name} lacks member column {member}")
    actual = int(frame[member].nunique())
    if actual != count:
        raise RuntimeError(f"{name} has {actual} members; expected {count}")


def bspace_bands() -> tuple[pd.DataFrame, dict]:
    starts = pd.read_csv(START_B)
    replicas = pd.read_csv(REPLICA_B)
    reference = pd.read_csv(REFERENCE_B)
    require_members(starts, "run_tag", "96-start b-space ensemble", 96)
    require_members(replicas, "seed", "50-replica b-space ensemble", 50)
    # The 96-start propagation was stored at Q=10 for the two diagnostic
    # flavors, but F_NP is flavor independent in this model.  Reconstruct the
    # Q=7.5 all-flavor start curves from that common F_NP and the frozen
    # perturbative reference factor at Q=7.5.  Verify the stated
    # flavor-independence rather than assuming it silently.
    fstart = starts[np.isclose(starts.x, .1) & np.isclose(starts.Q, 10.0)
                    & starts.flavor.astype(str).eq("u")]
    fstart_d = starts[np.isclose(starts.x, .1) & np.isclose(starts.Q, 10.0)
                      & starts.flavor.astype(str).eq("d")]
    fw = fstart.pivot(index="bT", columns="run_tag", values="F_NP").sort_index()
    fd = fstart_d.pivot(index="bT", columns="run_tag", values="F_NP").sort_index()
    if fw.shape != (len(fw.index), 96) or not np.allclose(fw.index, fd.index) or not np.allclose(fw, fd):
        raise RuntimeError("96-start F_NP is not a common flavor-independent curve")
    start_tags = fw.columns.astype(str).tolist()
    rows = []
    metrics = {}
    for flavor in ("u", "d", "s", "ubar", "dbar", "sbar"):
        r = replicas[np.isclose(replicas.x, .1) & np.isclose(replicas.Q, 7.5)
                     & replicas.flavor.astype(str).eq(flavor)]
        rw = r.pivot(index="bT", columns="seed", values="ftilde").sort_index()
        ref = reference[np.isclose(reference.x, .1) & np.isclose(reference.Q, 7.5)
                        & reference.flavor.astype(str).eq(flavor)]
        ref = ref.sort_values("bT")
        if rw.shape[1] != 50 or len(ref) == 0 or not np.allclose(ref.bT, rw.index):
            raise RuntimeError(f"b-space grids or member counts invalid for {flavor}")
        perturbative = ref.ftilde_no_np.to_numpy(float)
        sw = pd.DataFrame(
            fw.to_numpy(float) * perturbative[:, None],
            index=fw.index, columns=start_tags)
        if not np.allclose(sw.index, rw.index):
            raise RuntimeError(f"reconstructed Q=7.5 start grid differs for {flavor}")
        q16, q50, q84 = crossed_quantiles(sw.to_numpy(float).T, rw.to_numpy(float).T)
        b = sw.index.to_numpy(float)
        for i, coordinate in enumerate(b):
            rows.append({"flavor": flavor, "x": .1, "Q": 7.5, "bT": coordinate,
                         "q16": q16[i], "central": q50[i], "q84": q84[i]})
        active = (b <= 4.0) & (q50 > .05 * np.max(q50[b <= 4.0]))
        width = (q84 - q16) / np.maximum(np.abs(q50), EPS)
        metrics[flavor] = {
            "max_relative_full_width_active": float(np.max(width[active])),
            "median_relative_full_width_active": float(np.median(width[active])),
        }
    return pd.DataFrame(rows), metrics


def kspace_bands() -> tuple[pd.DataFrame, dict]:
    starts = pd.read_csv(START_K)
    require_members(starts, "_replica_key", "96-start k-space ensemble", 96)
    rows = []
    metrics = {}
    for flavor in ("u", "d"):
        replica_path = REPLICA_K_ROOT / f"kspace_fixedx_q10_{flavor}_current" / "v23a_regularized_kspace_replica_long.csv"
        if not replica_path.exists():
            raise FileNotFoundError(replica_path)
        replicas = pd.read_csv(replica_path)
        require_members(replicas, "seed", f"50-replica k-space ensemble ({flavor})", 50)
        s = starts[starts.quantity.eq("ftilde") & np.isclose(starts.x, .1)
                   & np.isclose(starts.Q, 10.0) & starts.flavor.astype(str).eq(flavor)]
        r = replicas[replicas.quantity.eq("ftilde") & np.isclose(replicas.x, .1)
                     & np.isclose(replicas.Q, 10.0) & replicas.flavor.astype(str).eq(flavor)]
        sw = s.pivot(index="kT", columns="_replica_key", values="value").sort_index()
        rw = r.pivot(index="kT", columns="seed", values="value").sort_index()
        if sw.shape[1] != 96 or rw.shape[1] != 50 or not np.allclose(sw.index, rw.index):
            raise RuntimeError(f"k-space grids or member counts invalid for {flavor}")
        q16, q50, q84 = crossed_quantiles(sw.to_numpy(float).T, rw.to_numpy(float).T)
        k = sw.index.to_numpy(float)
        for i, coordinate in enumerate(k):
            rows.append({"flavor": flavor, "x": .1, "Q": 10.0, "kT": coordinate,
                         "q16": q16[i], "central": q50[i], "q84": q84[i]})
        display = k <= 2.25
        active = display & (q50 > .05 * np.max(q50[display]))
        width = (q84 - q16) / np.maximum(np.abs(q50), EPS)
        metrics[flavor] = {
            "max_relative_full_width_active": float(np.max(width[active])),
            "median_relative_full_width_active": float(np.median(width[active])),
            "active_kT_max_GeV": float(k[active].max()),
        }
    return pd.DataFrame(rows), metrics


def external_gate(allow_missing: bool) -> dict:
    present = EXTERNAL_STATUS.exists() and Y_STATUS.exists() and EXTERNAL_GRID.exists()
    if not present and not allow_missing:
        raise RuntimeError("final external W+Y grid and conventional-Y status are not complete")
    result = {"present": bool(present), "grid": str(EXTERNAL_GRID),
              "grid_status": str(EXTERNAL_STATUS), "y_status": str(Y_STATUS)}
    if not present:
        return result
    grid_status = json.loads(EXTERNAL_STATUS.read_text())
    y_status = json.loads(Y_STATUS.read_text())
    grid = pd.read_csv(EXTERNAL_GRID)
    checks = grid_status.get("checks", {})
    if len(grid) != 122 or not checks.get("all_finite") or not checks.get("all_positive"):
        raise RuntimeError("external W+Y grid failed its 122-row finite/positive gate")
    if y_status.get("row_count") != 122 or not y_status.get("checks", {}).get("all_finite") \
            or not y_status.get("checks", {}).get("all_positive_reconstructed_full_wy"):
        raise RuntimeError("conventional-Y table failed its 122-row gate")
    result.update({"row_count": int(len(grid)), "grid_status_value": grid_status.get("status"),
                   "y_status_value": y_status.get("status"),
                   "all_finite": bool(checks.get("all_finite")),
                   "all_positive": bool(checks.get("all_positive"))})
    return result


def render(bands_b: pd.DataFrame, bands_k: pd.DataFrame) -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                         "axes.linewidth": .9, "xtick.direction": "in",
                         "ytick.direction": "in", "xtick.top": True,
                         "ytick.right": True})
    fig, ax = plt.subplots(figsize=(7.4, 5.0), constrained_layout=True)
    view = bands_b[bands_b.bT <= 4.0]
    for flavor in ("u", "d", "s", "ubar", "dbar", "sbar"):
        g = view[view.flavor.eq(flavor)].sort_values("bT")
        c = COLORS[flavor]
        ax.fill_between(g.bT, g.q16, g.q84, color=c, alpha=.15, linewidth=0)
        ax.plot(g.bT, g.central, color=c, lw=1.7, label=LABELS[flavor])
    ax.text(.98, .96, r"$x=0.1\qquad Q=7.5\ \mathrm{GeV}$", ha="right", va="top",
            transform=ax.transAxes, fontsize=12)
    ax.set(xlabel=r"$b_T\ [\mathrm{GeV}^{-1}]$",
           ylabel=r"$\widetilde f_1^q(x,b_T;Q)$", xlim=(0, 4))
    ax.grid(alpha=.15)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="0.55", alpha=.25, edgecolor="none"))
    labels.append("empirical 16--84% propagated envelope")
    ax.legend(handles, labels, frameon=False, fontsize=8.5, ncol=2)
    fig.savefig(TARGET / "fig2_bspace_fnp_start_replica.png", dpi=240)
    fig.savefig(TARGET / "fig2_bspace_fnp_start_replica.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 5.0), constrained_layout=True)
    view = bands_k[bands_k.kT <= 2.25]
    for flavor in ("u", "d"):
        g = view[view.flavor.eq(flavor)].sort_values("kT")
        c = COLORS[flavor]
        ax.fill_between(g.kT, g.q16, g.q84, color=c, alpha=.18, linewidth=0)
        ax.plot(g.kT, g.central, color=c, lw=1.8, label=LABELS[flavor])
    ax.text(.98, .96, r"$x=0.1\qquad Q=10\ \mathrm{GeV}$", ha="right", va="top",
            transform=ax.transAxes, fontsize=12)
    ax.set(xlabel=r"$k_T\ [\mathrm{GeV}]$", ylabel=r"$f_1^q(x,k_T;Q)$",
           xlim=(0, 2.25)); ax.set_ylim(bottom=0); ax.grid(alpha=.15)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="0.55", alpha=.25, edgecolor="none"))
    labels.append("empirical 16--84% propagated envelope")
    ax.legend(handles, labels, frameon=False, fontsize=10)
    fig.savefig(TARGET / "fig6_kspace_ud_fnp_start_replica.png", dpi=240)
    fig.savefig(TARGET / "fig6_kspace_ud_fnp_start_replica.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-missing-external", action="store_true")
    args = parser.parse_args()
    bands_b, metrics_b = bspace_bands()
    bands_k, metrics_k = kspace_bands()
    external = external_gate(args.allow_missing_external)
    TARGET.mkdir(parents=True, exist_ok=True)
    bands_b.to_csv(TARGET / "fig2_bspace_bands.csv", index=False)
    bands_k.to_csv(TARGET / "fig6_kspace_bands.csv", index=False)
    render(bands_b, bands_k)
    status = {
        "status": ("isolated_tevatron_full_fnp_start_replica_propagation_complete_not_production"
                   if external.get("present") else
                   "isolated_fnp_start_replica_propagation_ready_external_gate_pending"),
        "external_wy_gate": external,
        "start_count": 96,
        "experimental_replica_count": 50,
        "crossed_member_count_per_flavor": 4800,
        "construction": "centered experimental-replica residuals crossed with correlated optimizer-start curves",
        "central_curve": "empirical q50 of the crossed ensemble",
        "band": "pointwise empirical q16-q84; non-uniqueness is not assigned a Gaussian confidence meaning",
        "bspace_metrics": metrics_b,
        "kspace_metrics": metrics_k,
        "artifacts": {"fig2": str(TARGET / "fig2_bspace_fnp_start_replica.png"),
                       "fig6": str(TARGET / "fig6_kspace_ud_fnp_start_replica.png"),
                       "bspace_csv": str(TARGET / "fig2_bspace_bands.csv"),
                       "kspace_csv": str(TARGET / "fig6_kspace_bands.csv")},
        "formal_confidence_level_assigned": False,
        "contains_individual_start_curves": False,
        "contains_legacy_conditional_result": False,
        "frozen_baseline_unchanged": True,
        "production_outputs_modified": False,
    }
    (TARGET / "propagation_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
