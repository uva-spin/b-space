#!/usr/bin/env python3
"""Audit the small-bT hook/hump in the v23a b-space TMD grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FLAVOR_ORDER = ["u", "d", "s", "ubar", "dbar", "sbar"]


def norm_to_b0(group: pd.DataFrame, column: str) -> np.ndarray:
    values = group[column].to_numpy(dtype=float)
    b0 = float(values[0])
    if abs(b0) < 1.0e-300:
        return np.full_like(values, np.nan, dtype=float)
    return values / b0


def first_true_below(values: np.ndarray, threshold: float) -> int | None:
    idx = np.where(values <= threshold)[0]
    if len(idx) == 0:
        return None
    return int(idx[0])


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for flavor, group in df.groupby("flavor", sort=False):
        group = group.sort_values("bT").reset_index(drop=True)
        b = group["bT"].to_numpy(dtype=float)
        f = group["ftilde"].to_numpy(dtype=float)
        fnp = group["F_NP"].to_numpy(dtype=float)
        evol = group["evol_half"].to_numpy(dtype=float)
        ope = group["ope_boundary_nlo"].to_numpy(dtype=float)

        peak_idx = int(np.nanargmax(f))
        hook_min_idx = int(np.nanargmin(f[b <= 0.15])) if np.any(b <= 0.15) else 0
        mub_uncap_idx = first_true_below(group["mu_b"].to_numpy(dtype=float), float(group["Q"].iloc[0]) * (1.0 - 1.0e-10))

        rows.append(
            {
                "flavor": flavor,
                "x": float(group["x"].iloc[0]),
                "Q": float(group["Q"].iloc[0]),
                "ftilde_b0": float(f[0]),
                "ftilde_peak": float(f[peak_idx]),
                "ftilde_peak_bT": float(b[peak_idx]),
                "ftilde_peak_rel_to_b0": float(f[peak_idx] / f[0] - 1.0),
                "low_b_min_bT_le_0p15": float(b[hook_min_idx]),
                "low_b_min_rel_to_b0": float(f[hook_min_idx] / f[0] - 1.0),
                "F_NP_at_peak": float(fnp[peak_idx]),
                "F_NP_peak_rel_to_b0": float(fnp[peak_idx] / fnp[0] - 1.0),
                "evol_half_at_peak": float(evol[peak_idx]),
                "evol_half_peak_rel_to_b0": float(evol[peak_idx] / evol[0] - 1.0),
                "ope_boundary_at_peak": float(ope[peak_idx]),
                "ope_boundary_peak_rel_to_b0": float(ope[peak_idx] / ope[0] - 1.0),
                "mu_b_at_b0": float(group["mu_b"].iloc[0]),
                "mu_b_at_peak": float(group["mu_b"].iloc[peak_idx]),
                "b_pert_at_b0": float(group["b_pert"].iloc[0]),
                "b_pert_at_peak": float(group["b_pert"].iloc[peak_idx]),
                "sudakov_S_at_peak": float(group["sudakov_S_pair"].iloc[peak_idx]),
                "mu_b_first_below_Q_bT": None if mub_uncap_idx is None else float(b[mub_uncap_idx]),
            }
        )
    return pd.DataFrame(rows)


def make_plot(df: pd.DataFrame, summary: pd.DataFrame, out_png: Path, out_pdf: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 14,
            "axes.linewidth": 1.1,
            "xtick.direction": "in",
            "ytick.direction": "in",
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 7.7), dpi=320, sharex=True)
    axes = axes.ravel()

    labels = {
        "ftilde": r"$\tilde f$",
        "ftilde_no_np": r"$(C\otimes f)e^{-S/2}$",
        "ope_boundary_nlo": r"$C\otimes f$",
        "evol_half": r"$e^{-S/2}$",
        "F_NP": r"$F_{\rm NP}$",
    }
    styles = {
        "ftilde": {"color": "black", "lw": 2.0, "ls": "-"},
        "ftilde_no_np": {"color": "#1f77b4", "lw": 1.7, "ls": "--"},
        "ope_boundary_nlo": {"color": "#ff7f0e", "lw": 1.4, "ls": "-."},
        "evol_half": {"color": "#2ca02c", "lw": 1.4, "ls": ":"},
        "F_NP": {"color": "#d62728", "lw": 1.4, "ls": "-"},
    }

    for ax, flavor in zip(axes, FLAVOR_ORDER):
        group = df[df["flavor"] == flavor].sort_values("bT").reset_index(drop=True)
        if group.empty:
            ax.set_visible(False)
            continue
        focus = group[group["bT"] <= 1.2].reset_index(drop=True)
        b = focus["bT"].to_numpy(dtype=float)

        for col in ["ftilde", "ftilde_no_np", "ope_boundary_nlo", "evol_half", "F_NP"]:
            ax.plot(b, norm_to_b0(focus, col), label=labels[col], **styles[col])

        row = summary[summary["flavor"] == flavor].iloc[0]
        ax.axvline(float(row["mu_b_first_below_Q_bT"]), color="0.65", lw=1.0, ls="--")
        ax.axvline(float(row["ftilde_peak_bT"]), color="0.25", lw=1.0, ls=":")
        ax.set_title(flavor, fontsize=17, pad=4)
        ax.set_xlim(0.0, 1.2)
        ax.grid(True, color="0.88", lw=0.55)
        ax.tick_params(which="major", top=True, right=True, labelsize=13, length=5.0, width=1.0)
        ax.tick_params(which="minor", top=True, right=True, length=2.6, width=0.75)

    axes[0].legend(loc="best", fontsize=11, frameon=False, handlelength=2.4)
    for ax in axes[3:]:
        ax.set_xlabel(r"$b_T\ [{\rm GeV}^{-1}]$", fontsize=16)
    for ax in axes[::3]:
        ax.set_ylabel(r"normalized to $b_T=0$", fontsize=16)

    x = float(df["x"].iloc[0])
    q = float(df["Q"].iloc[0])
    fig.suptitle(rf"Low-$b_T$ decomposition, $x={x:g}$, $Q={q:g}$ GeV", fontsize=17, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=1.15, w_pad=0.75)
    fig.savefig(out_png, dpi=360, bbox_inches="tight", pad_inches=0.035)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="plots/v23a_tevatron_plus_lhcb7_fidacc_lowqt010_central_exactx/v22_scheme_tmd_bspace_long.csv")
    ap.add_argument("--x", type=float, default=0.1)
    ap.add_argument("--Q", type=float, default=7.5)
    ap.add_argument("--out-dir", default="plots/v23a_bspace_lowb_hook_audit")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src = Path(args.grid)
    df = pd.read_csv(src)
    df = df[(np.isclose(df["x"], args.x)) & (np.isclose(df["Q"], args.Q))].copy()
    df = df[df["flavor"].isin(FLAVOR_ORDER)].copy()
    df["flavor"] = pd.Categorical(df["flavor"], categories=FLAVOR_ORDER, ordered=True)
    df = df.sort_values(["flavor", "bT"]).reset_index(drop=True)
    if df.empty:
        raise SystemExit(f"no rows found for x={args.x}, Q={args.Q} in {src}")

    lowb = df[df["bT"] <= 1.2].copy()
    summary = build_summary(lowb)
    summary["flavor"] = pd.Categorical(summary["flavor"], categories=FLAVOR_ORDER, ordered=True)
    summary = summary.sort_values("flavor").reset_index(drop=True)

    lowb.to_csv(out_dir / "v23a_lowb_hook_components.csv", index=False)
    summary.to_csv(out_dir / "v23a_lowb_hook_summary.csv", index=False)

    with (out_dir / "v23a_lowb_hook_audit.json").open("w") as f:
        json.dump(
            {
                "grid": str(src),
                "x": float(args.x),
                "Q": float(args.Q),
                "interpretation": (
                    "The hook/hump is audited by decomposing ftilde into the NLO OPE boundary, "
                    "single-leg Sudakov evolution, and fitted F_NP. A monotone, near-unity F_NP "
                    "at low bT means the feature is perturbative-profile/OPE/evolution driven."
                ),
                "outputs": {
                    "components_csv": "v23a_lowb_hook_components.csv",
                    "summary_csv": "v23a_lowb_hook_summary.csv",
                    "plot_png": "v23a_lowb_hook_decomposition.png",
                    "plot_pdf": "v23a_lowb_hook_decomposition.pdf",
                },
            },
            f,
            indent=2,
        )

    make_plot(
        lowb,
        summary,
        out_dir / "v23a_lowb_hook_decomposition.png",
        out_dir / "v23a_lowb_hook_decomposition.pdf",
    )


if __name__ == "__main__":
    main()
