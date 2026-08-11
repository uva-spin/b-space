#!/usr/bin/env python3
"""Compare old fixed-target and new Tevatron b-space TMD bands by flavor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator


FLAVOR_TO_PID = {"u": 2, "d": 1, "s": 3, "ubar": -2, "dbar": -1, "sbar": -3}
FLAVOR_TEX = {
    "u": r"$u$",
    "d": r"$d$",
    "s": r"$s$",
    "ubar": r"$\bar u$",
    "dbar": r"$\bar d$",
    "sbar": r"$\bar s$",
}
COLORS = {
    "u": "#1f77b4",
    "d": "#ff7f0e",
    "s": "#9467bd",
    "ubar": "#2ca02c",
    "dbar": "#d62728",
    "sbar": "#8c564b",
}


def load_curve(path: Path, flavor: str, x: float, Q: float, quantity: str, b_max: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    pid = FLAVOR_TO_PID[flavor]
    cols = [f"{quantity}_median", f"{quantity}_q16", f"{quantity}_q84"]
    missing = [c for c in ["pid", "flavor", "x", "Q", "bT", *cols] if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} missing columns {missing}")
    mask = (
        pd.to_numeric(df["pid"], errors="coerce").eq(pid)
        & df["flavor"].astype(str).eq(flavor)
        & np.isclose(pd.to_numeric(df["x"], errors="coerce"), x, rtol=0, atol=1e-10)
        & np.isclose(pd.to_numeric(df["Q"], errors="coerce"), Q, rtol=0, atol=1e-10)
        & (pd.to_numeric(df["bT"], errors="coerce") <= b_max)
    )
    out = df.loc[mask, ["bT", *cols]].copy()
    out.columns = ["bT", "median", "q16", "q84"]
    out = out.apply(pd.to_numeric, errors="coerce").dropna().sort_values("bT")
    if out.empty:
        available = {
            "x": sorted(pd.to_numeric(df["x"], errors="coerce").dropna().unique().tolist()),
            "Q": sorted(pd.to_numeric(df["Q"], errors="coerce").dropna().unique().tolist()),
            "flavors": sorted(df["flavor"].dropna().astype(str).unique().tolist()),
        }
        raise SystemExit(f"No {flavor} curve at x={x:g}, Q={Q:g} in {path}\n{available}")
    return out.drop_duplicates("bT")


def ylabel(quantity: str) -> str:
    if quantity == "ftilde":
        return r"$\widetilde f_1^{q/p}(x,b_T;Q)$"
    if quantity == "x_ftilde":
        return r"$x\,\widetilde f_1^{q/p}(x,b_T;Q)$"
    return quantity


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-bands", required=True)
    ap.add_argument("--new-bands", required=True)
    ap.add_argument("--old-label", default="fixed target only")
    ap.add_argument("--new-label", default="fixed target + Tevatron")
    ap.add_argument("--quantity", default="ftilde", choices=["ftilde", "x_ftilde"])
    ap.add_argument(
        "--flavors",
        nargs="+",
        default=["u", "d", "ubar", "dbar"],
        choices=["u", "d", "s", "ubar", "dbar", "sbar"],
    )
    ap.add_argument("--x", type=float, default=0.10)
    ap.add_argument("--Q", type=float, default=7.5)
    ap.add_argument("--b-max", type=float, default=4.0)
    ap.add_argument("--uncertainty-label", default="q16--q84 experimental-replica bands for each fit")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.linewidth": 1.1,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "savefig.dpi": 300,
    })

    old_path = Path(args.old_bands)
    new_path = Path(args.new_bands)
    flavors = list(args.flavors)
    ncols = 3 if len(flavors) > 4 else 2
    nrows = int(np.ceil(len(flavors) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.95 * ncols, 3.25 * nrows), dpi=180, sharex=True)
    axes = axes.ravel()
    export = []
    diagnostics = {}

    for ax, flavor in zip(axes, flavors):
        old = load_curve(old_path, flavor, args.x, args.Q, args.quantity, args.b_max)
        new = load_curve(new_path, flavor, args.x, args.Q, args.quantity, args.b_max)
        color = COLORS[flavor]

        for frame, fit_label in [(old, "old"), (new, "new")]:
            e = frame.copy()
            e.insert(0, "fit", fit_label)
            e.insert(1, "flavor", flavor)
            e.insert(2, "pid", FLAVOR_TO_PID[flavor])
            e.insert(3, "x", args.x)
            e.insert(4, "Q", args.Q)
            export.append(e)

        ax.fill_between(old["bT"], old["q16"], old["q84"], color=color, alpha=0.13, linewidth=0)
        ax.plot(old["bT"], old["median"], color=color, linestyle="--", linewidth=2.0)
        ax.fill_between(new["bT"], new["q16"], new["q84"], color=color, alpha=0.23, linewidth=0)
        ax.plot(new["bT"], new["median"], color=color, linestyle="-", linewidth=2.25)
        ax.set_title(FLAVOR_TEX[flavor], loc="left", fontsize=15)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which="major", length=5, width=1.0, labelsize=10.5)
        ax.tick_params(which="minor", length=2.8, width=0.8)

        merged = pd.merge(
            old[["bT", "median"]].rename(columns={"median": "old_median"}),
            new[["bT", "median"]].rename(columns={"median": "new_median"}),
            on="bT",
            how="inner",
        )
        ratio = merged["new_median"].to_numpy(float) / np.maximum(np.abs(merged["old_median"].to_numpy(float)), 1e-300)
        diagnostics[flavor] = {
            "old_b0": float(old.loc[np.abs(old["bT"]).idxmin(), "median"]),
            "new_b0": float(new.loc[np.abs(new["bT"]).idxmin(), "median"]),
            "new_over_old_b0": float(new.loc[np.abs(new["bT"]).idxmin(), "median"] / old.loc[np.abs(old["bT"]).idxmin(), "median"]),
            "new_over_old_median_bgrid": float(np.nanmedian(ratio)),
            "new_over_old_min_bgrid": float(np.nanmin(ratio)),
            "new_over_old_max_bgrid": float(np.nanmax(ratio)),
        }

    for ax in axes[len(flavors):]:
        ax.axis("off")

    for ax in axes[(nrows - 1) * ncols:]:
        ax.set_xlabel(r"$b_T\;[\mathrm{GeV}^{-1}]$", fontsize=13)
    for ax in axes[:len(flavors):ncols]:
        ax.set_ylabel(ylabel(args.quantity), fontsize=13)

    handles = [
        Line2D([0], [0], color="black", linestyle="--", linewidth=2.0, label=args.old_label),
        Line2D([0], [0], color="black", linestyle="-", linewidth=2.25, label=args.new_label),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=2, frameon=False, fontsize=11.5)
    fig.suptitle(rf"$x={args.x:g},\quad Q={args.Q:g}\,\mathrm{{GeV}}$", x=0.06, y=0.995, ha="left", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")

    curves = pd.concat(export, ignore_index=True)
    curves.to_csv(out.with_suffix(".curves.csv"), index=False)
    summary = {
        "old_bands": str(old_path),
        "new_bands": str(new_path),
        "old_label": args.old_label,
        "new_label": args.new_label,
        "quantity": args.quantity,
        "x": args.x,
        "Q": args.Q,
        "b_max": args.b_max,
        "uncertainty": str(args.uncertainty_label),
        "flavor_diagnostics": diagnostics,
        "outputs": {
            "pdf": str(out),
            "png": str(out.with_suffix(".png")),
            "curves_csv": str(out.with_suffix(".curves.csv")),
            "diagnostics_json": str(out.with_suffix(".diagnostics.json")),
        },
    }
    out.with_suffix(".diagnostics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
