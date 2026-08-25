#!/usr/bin/env python3
"""Paper-style single-curve b_T-space TMDPDF plot for the d quark.

This produces the standard fixed-target DY b_T companion figure:

  ftilde / f_1 in b_T space, d quark, with q16--q84 uncertainty band.

It is intentionally narrower than the multi-page plotting utilities: it makes
one clean paper figure for one flavor/x/Q point, defaulting to

  flavor = d
  x      = 0.10
  Q      = 10 GeV
  bT     = 0..4 GeV^-1
  quantity = ftilde

Expected input band CSV formats include either:
  v23a_dataPDF_tmd_replica_bspace_bands.csv
with columns:
  ftilde_median, ftilde_q16, ftilde_q84

or the older:
  v22_tmd_replica_bspace_bands.csv
with the same quantity-specific band columns.

If --central-grid is supplied, a dashed central PDF0 curve is added from the
central b-space grid.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


FLAVOR_TO_PID = {"dbar": -1, "ubar": -2, "d": 1, "u": 2}

FLAVOR_TEX = {
    "u": "u",
    "d": "d",
    "ubar": r"\bar u",
    "dbar": r"\bar d",
}


def first_col(df: pd.DataFrame, names: Iterable[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for name in names:
        if name in df.columns:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def find_bands_csv(band_dir: Path, explicit: str | None = None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise SystemExit(f"Requested --bands-csv does not exist: {p}")
        return p

    if band_dir.is_file():
        return band_dir

    preferred = [
        "v23a_dataPDF_tmd_replica_bspace_bands.csv",
        "v22_tmd_replica_bspace_bands.csv",
        "tmd_replica_bspace_bands.csv",
        "v23a_tmd_replica_bspace_bands.csv",
    ]
    for name in preferred:
        p = band_dir / name
        if p.exists():
            return p

    candidates = sorted(band_dir.glob("*bspace*bands*.csv")) + sorted(band_dir.glob("*bands*.csv"))
    if candidates:
        return candidates[0]

    raise SystemExit(f"Could not find a b-space bands CSV in {band_dir}")


def find_long_csv(band_dir: Path, explicit: str | None = None) -> Path | None:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise SystemExit(f"Requested --long-csv does not exist: {p}")
        return p

    preferred = [
        "v23a_dataPDF_tmd_replica_bspace_long.csv",
        "v22_tmd_replica_bspace_long.csv",
        "tmd_replica_bspace_long.csv",
    ]
    for name in preferred:
        p = band_dir / name
        if p.exists():
            return p
    found = sorted(band_dir.glob("*bspace*long*.csv"))
    return found[0] if found else None


def quantity_band_columns(df: pd.DataFrame, quantity: str) -> tuple[str, str, str]:
    med = first_col(df, [f"{quantity}_median", f"{quantity}_q50", f"{quantity}_p50", "median", "q50"])
    lo = first_col(df, [f"{quantity}_q16", f"{quantity}_p16", f"{quantity}_lo68", f"{quantity}_lower68", "q16", "p16", "lo68"])
    hi = first_col(df, [f"{quantity}_q84", f"{quantity}_p84", f"{quantity}_hi68", f"{quantity}_upper68", "q84", "p84", "hi68"])
    missing = [name for name, col in [("median", med), ("q16/lo68", lo), ("q84/hi68", hi)] if col is None]
    if missing:
        raise SystemExit(
            f"Could not infer {quantity} band columns; missing {missing}.\n"
            f"Available columns: {list(df.columns)}"
        )
    return med, lo, hi


def select_curve(
    df: pd.DataFrame,
    *,
    quantity: str,
    flavor: str,
    pid: int,
    x: float,
    Q: float,
    b_max: float,
) -> pd.DataFrame:
    required = ["pid", "x", "Q", "bT"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Input bands file missing columns {missing}. Columns: {list(df.columns)}")

    med_col, lo_col, hi_col = quantity_band_columns(df, quantity)

    m = (
        (df["pid"].astype(int) == int(pid))
        & np.isclose(pd.to_numeric(df["x"], errors="coerce"), float(x), rtol=0, atol=1e-10)
        & np.isclose(pd.to_numeric(df["Q"], errors="coerce"), float(Q), rtol=0, atol=1e-10)
        & (pd.to_numeric(df["bT"], errors="coerce") <= float(b_max))
    )
    if "flavor" in df.columns:
        m &= df["flavor"].astype(str).eq(flavor)

    g = df[m].copy()
    if g.empty:
        debug = {
            "available_pids": sorted(pd.to_numeric(df["pid"], errors="coerce").dropna().unique().tolist()) if "pid" in df.columns else None,
            "available_flavors": sorted(map(str, df["flavor"].dropna().unique())) if "flavor" in df.columns else None,
            "available_x": sorted(pd.to_numeric(df["x"], errors="coerce").dropna().unique().tolist()) if "x" in df.columns else None,
            "available_Q": sorted(pd.to_numeric(df["Q"], errors="coerce").dropna().unique().tolist()) if "Q" in df.columns else None,
        }
        raise SystemExit(
            f"No matching b-space curve for quantity={quantity}, flavor={flavor}, pid={pid}, x={x}, Q={Q}.\n"
            + json.dumps(debug, indent=2)
        )

    out = pd.DataFrame({
        "bT": pd.to_numeric(g["bT"], errors="coerce"),
        "median": pd.to_numeric(g[med_col], errors="coerce"),
        "q16": pd.to_numeric(g[lo_col], errors="coerce"),
        "q84": pd.to_numeric(g[hi_col], errors="coerce"),
    }).dropna().sort_values("bT")

    return out


def add_derived_long(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "ftilde" in df.columns:
        if "x_ftilde" not in df.columns and "x" in df.columns:
            df["x_ftilde"] = pd.to_numeric(df["x"], errors="coerce") * pd.to_numeric(df["ftilde"], errors="coerce")
    return df


def load_thin_replicas(
    long_csv: Path | None,
    *,
    quantity: str,
    flavor: str,
    pid: int,
    x: float,
    Q: float,
    b_max: float,
    n: int,
    random_seed: int,
) -> list[pd.DataFrame]:
    if n <= 0 or long_csv is None or not long_csv.exists():
        return []

    df = pd.read_csv(long_csv)
    df = add_derived_long(df)
    if quantity not in df.columns:
        return []

    if {"seed", "pdf_member"}.issubset(df.columns):
        df["_replica_key"] = df["seed"].astype(str) + "|pdf" + df["pdf_member"].astype(str)
    elif "seed" in df.columns:
        df["_replica_key"] = df["seed"].astype(str)
    else:
        return []

    m = (
        (df["pid"].astype(int) == int(pid))
        & np.isclose(pd.to_numeric(df["x"], errors="coerce"), float(x), rtol=0, atol=1e-10)
        & np.isclose(pd.to_numeric(df["Q"], errors="coerce"), float(Q), rtol=0, atol=1e-10)
        & (pd.to_numeric(df["bT"], errors="coerce") <= float(b_max))
    )
    if "flavor" in df.columns:
        m &= df["flavor"].astype(str).eq(flavor)

    df = df[m].copy()
    keys = sorted(df["_replica_key"].dropna().astype(str).unique().tolist())
    if not keys:
        return []
    if len(keys) > n:
        rng = np.random.default_rng(int(random_seed))
        keys = sorted(rng.choice(keys, size=int(n), replace=False).tolist())

    curves = []
    for key in keys:
        g = df[df["_replica_key"].astype(str).eq(str(key))].copy()
        g = g[["bT", quantity]].dropna().sort_values("bT")
        curves.append(g.rename(columns={quantity: "value"}))
    return curves


def load_central_curve(
    central_grid: str | None,
    *,
    quantity: str,
    flavor: str,
    pid: int,
    x: float,
    Q: float,
    b_max: float,
) -> pd.DataFrame | None:
    if not central_grid:
        return None

    p = Path(central_grid)
    if not p.exists():
        raise SystemExit(f"Central grid does not exist: {p}")

    df = pd.read_csv(p)
    df = add_derived_long(df)
    if quantity not in df.columns:
        raise SystemExit(f"Central grid missing quantity column {quantity}. Columns: {list(df.columns)}")

    m = (
        (df["pid"].astype(int) == int(pid))
        & np.isclose(pd.to_numeric(df["x"], errors="coerce"), float(x), rtol=0, atol=1e-10)
        & np.isclose(pd.to_numeric(df["Q"], errors="coerce"), float(Q), rtol=0, atol=1e-10)
        & (pd.to_numeric(df["bT"], errors="coerce") <= float(b_max))
    )
    if "flavor" in df.columns:
        m &= df["flavor"].astype(str).eq(flavor)

    g = df[m].copy()
    if g.empty:
        raise SystemExit(f"No central curve found for {flavor}, x={x}, Q={Q}")

    return pd.DataFrame({
        "bT": pd.to_numeric(g["bT"], errors="coerce"),
        "central": pd.to_numeric(g[quantity], errors="coerce"),
    }).dropna().sort_values("bT")


def default_output(flavor: str, x: float, Q: float) -> str:
    xlab = f"{x:g}".replace(".", "p")
    qlab = f"{Q:g}".replace(".", "p")
    return f"plots/v23a_paper_bspace_TMDPDF_{flavor}_x{xlab}_Q{qlab}.pdf"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--band-dir", required=True, help="Directory containing b-space band CSVs, or the band CSV itself.")
    ap.add_argument("--bands-csv", default=None)
    ap.add_argument("--long-csv", default=None, help="Optional long replica CSV for thin diagnostic curves.")
    ap.add_argument("--central-grid", default=None)
    ap.add_argument("--quantity", default="ftilde", choices=["ftilde", "x_ftilde"])
    ap.add_argument("--flavor", default="d", choices=["u", "d", "ubar", "dbar"])
    ap.add_argument("--pid", type=int, default=None)
    ap.add_argument("--x", type=float, default=0.10)
    ap.add_argument("--Q", type=float, default=10.0)
    ap.add_argument("--b-max", type=float, default=4.0)
    ap.add_argument("--title", default="", help="Optional plot title. Empty by default for paper style.")
    ap.add_argument("--label", default="v23a FT-DY")
    ap.add_argument("--band-label", default="68% exp+PDF overlay")
    ap.add_argument("--central-label", default="central fit, PDF0")
    ap.add_argument("--thin-replicas", type=int, default=0)
    ap.add_argument("--thin-alpha", type=float, default=0.08)
    ap.add_argument("--random-seed", type=int, default=303)
    ap.add_argument("--legend-loc", default="upper right")
    ap.add_argument("--out", default=None)
    ap.add_argument("--show-grid", action="store_true")
    ap.add_argument("--show-note", default="", help="Optional small note drawn inside the plot.")
    args = ap.parse_args()

    pid = args.pid if args.pid is not None else FLAVOR_TO_PID[args.flavor]
    band_dir = Path(args.band_dir)
    bands_csv = find_bands_csv(band_dir, args.bands_csv)
    long_csv = find_long_csv(band_dir if band_dir.is_dir() else band_dir.parent, args.long_csv)

    bands = pd.read_csv(bands_csv)
    curve = select_curve(
        bands,
        quantity=args.quantity,
        flavor=args.flavor,
        pid=pid,
        x=args.x,
        Q=args.Q,
        b_max=args.b_max,
    )

    central = load_central_curve(
        args.central_grid,
        quantity=args.quantity,
        flavor=args.flavor,
        pid=pid,
        x=args.x,
        Q=args.Q,
        b_max=args.b_max,
    )

    thin_curves = load_thin_replicas(
        long_csv,
        quantity=args.quantity,
        flavor=args.flavor,
        pid=pid,
        x=args.x,
        Q=args.Q,
        b_max=args.b_max,
        n=args.thin_replicas,
        random_seed=args.random_seed,
    )

    # Diagnostics.
    eps = 1e-300
    curve["abs_halfwidth"] = 0.5 * (curve["q84"] - curve["q16"])
    curve["rel_halfwidth"] = curve["abs_halfwidth"] / np.maximum(np.abs(curve["median"]), eps)
    peak = float(np.nanmax(np.abs(curve["median"])))
    active = curve[np.abs(curve["median"]) > 0.05 * max(peak, eps)]

    diagnostics = {
        "bands_csv": str(bands_csv),
        "long_csv": str(long_csv) if long_csv else None,
        "central_grid": str(args.central_grid) if args.central_grid else None,
        "quantity": args.quantity,
        "flavor": args.flavor,
        "pid": int(pid),
        "x": float(args.x),
        "Q": float(args.Q),
        "b_max": float(args.b_max),
        "n_points": int(len(curve)),
        "peak_abs_median": peak,
        "relative_68_halfwidth_median_active": float(active["rel_halfwidth"].median()) if len(active) else None,
        "relative_68_halfwidth_p90_active": float(active["rel_halfwidth"].quantile(0.90)) if len(active) else None,
        "relative_68_halfwidth_max_active": float(active["rel_halfwidth"].max()) if len(active) else None,
        "n_thin_replicas": int(len(thin_curves)),
    }

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.linewidth": 1.25,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "legend.frameon": False,
    })

    fig, ax = plt.subplots(figsize=(7.4, 5.15), dpi=180)

    for i, g in enumerate(thin_curves):
        ax.plot(
            g["bT"],
            g["value"],
            color="0.45",
            alpha=float(args.thin_alpha),
            lw=0.7,
            label="thin replicas" if i == 0 else None,
            zorder=1,
        )

    b = curve["bT"].to_numpy(float)
    med = curve["median"].to_numpy(float)
    q16 = curve["q16"].to_numpy(float)
    q84 = curve["q84"].to_numpy(float)

    ax.fill_between(
        b,
        q16,
        q84,
        color="tab:blue",
        alpha=0.28,
        linewidth=0,
        label=args.band_label,
        zorder=2,
    )
    ax.plot(
        b,
        med,
        color="tab:blue",
        lw=2.4,
        label=f"{args.label} median",
        zorder=4,
    )

    if central is not None:
        ax.plot(
            central["bT"],
            central["central"],
            color="black",
            lw=1.8,
            ls="--",
            alpha=0.95,
            label=args.central_label,
            zorder=5,
        )

    ymax = max(float(np.nanmax(q84)), float(np.nanmax(med)))
    ymin = min(float(np.nanmin(q16)), float(np.nanmin(med)))
    if central is not None:
        ymax = max(ymax, float(np.nanmax(central["central"])))
        ymin = min(ymin, float(np.nanmin(central["central"])))
    if ymin >= 0:
        ax.set_ylim(0, ymax * 1.16 if ymax > 0 else 1.0)
    else:
        pad = 0.10 * (ymax - ymin)
        ax.set_ylim(ymin - pad, ymax + pad)

    ax.set_xlim(0, args.b_max)

    if args.title:
        ax.set_title(args.title, fontsize=23, fontweight="bold", pad=14)

    ax.set_xlabel(r"$b_T\;(\mathrm{GeV}^{-1})$", fontsize=18)
    flav_tex = FLAVOR_TEX.get(args.flavor, args.flavor)
    if args.quantity == "ftilde":
        ax.set_ylabel(rf"$\widetilde f_1^{{\,{flav_tex}}}(x,b_T;Q)$", fontsize=17)
    else:
        ax.set_ylabel(rf"$x\,\widetilde f_1^{{\,{flav_tex}}}(x,b_T;Q)$", fontsize=17)

    ax.text(
        0.03,
        0.93,
        rf"$x={args.x:g}\quad Q={args.Q:g}\,\mathrm{{GeV}}\quad {flav_tex}$-quark",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2},
    )

    if args.show_note:
        ax.text(
            0.03,
            0.04,
            args.show_note,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            alpha=0.85,
        )

    ax.tick_params(which="major", length=6, width=1.2, labelsize=13)
    ax.tick_params(which="minor", length=3, width=1.0)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    if args.show_grid:
        ax.grid(alpha=0.18)

    ax.legend(loc=args.legend_loc, fontsize=10)
    fig.tight_layout()

    out = Path(args.out or default_output(args.flavor, args.x, args.Q))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")

    curve.to_csv(out.with_suffix(".curve.csv"), index=False)
    out.with_suffix(".diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")

    print(json.dumps(diagnostics, indent=2))
    print("wrote:", out)
    print("wrote:", out.with_suffix(".png"))
    print("wrote:", out.with_suffix(".curve.csv"))
    print("wrote:", out.with_suffix(".diagnostics.json"))


if __name__ == "__main__":
    main()
