#!/usr/bin/env python3
"""Make a traditional single-panel b_T-space TMD-PDF sanity plot.

This mimics common ART/MAP-style b-space plots:
  title: "TMD PDFs"
  annotation: x = ..., Q = ... GeV, flavor
  x-axis: b [GeV^-1]
  y-axis: f_1(x,b)

The script supports the current v22/v23 b-space band format, including wide
columns like:
  ftilde_median, ftilde_q16, ftilde_q84
  F_NP_median, F_NP_q16, F_NP_q84
  b_ftilde_median, ...
and long replica files with a generic value column.

Example:
  PYTHONPATH=. python3 v23/tools/plot_traditional_bspace_tmd.py \
    --band-dir replica_pilot_v23a_lambda3_normpriors15_p2p5_E772_E288400_cached_cuda/tmd_bspace_bands_exactx_50rep \
    --central-grid plots/v23a_fixed_target_lowQ_normpriors15_p2p5_E772_E288400_central_exactx/v22_scheme_tmd_bspace_long.csv \
    --quantity ftilde \
    --flavor d \
    --x 0.10 \
    --Q 10 \
    --b-max 4 \
    --label "v23a FT-DY" \
    --out plots/v23a_traditional_TMDPDF_d_x0p10_Q10.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


FLAVOR_TO_PID = {
    "dbar": -1,
    "ubar": -2,
    "d": 1,
    "u": 2,
}


QUANTITY_ALIASES = {
    "ftilde": ["ftilde", "f_eff", "f1", "f1_eff"],
    "f_eff": ["f_eff", "ftilde", "f1", "f1_eff"],
    "x_ftilde": ["x_ftilde", "x_f_eff", "xf_eff"],
    "b_ftilde": ["b_ftilde", "b_f_eff"],
    "b_x_ftilde": ["b_x_ftilde", "b_xf_eff", "b_x_f_eff"],
    "F_NP": ["F_NP", "FNP", "fnp"],
}


def _aliases(quantity: str) -> list[str]:
    return QUANTITY_ALIASES.get(quantity, [quantity])


def _first_col(df: pd.DataFrame, names: Iterable[str]) -> str | None:
    lower_to_actual = {c.lower(): c for c in df.columns}
    for name in names:
        if name in df.columns:
            return name
        if name.lower() in lower_to_actual:
            return lower_to_actual[name.lower()]
    return None


def _candidate_columns_for_quantity(quantity: str, suffixes: Iterable[str]) -> list[str]:
    names: list[str] = []
    for q in _aliases(quantity):
        for s in suffixes:
            names.append(f"{q}_{s}")
            names.append(f"{q}{s}")
            names.append(f"{s}_{q}")
    return names


def _find_band_csv(root: Path) -> Path:
    if root.is_file():
        return root

    preferred_names = [
        "v22_tmd_replica_bspace_bands.csv",
        "v23_tmd_replica_bspace_bands.csv",
        "tmd_replica_bspace_bands.csv",
        "bspace_bands_long.csv",
        "v22_scheme_tmd_bspace_long.csv",
    ]

    for name in preferred_names:
        p = root / name
        if p.exists():
            return p

    patterns = [
        "*replica*bspace*band*.csv",
        "*bspace*band*.csv",
        "*band*long*.csv",
        "*bands*.csv",
        "*bspace*long*.csv",
        "*long*.csv",
        "*.csv",
    ]

    seen: set[Path] = set()
    scored: list[tuple[int, Path]] = []

    for pattern in patterns:
        for p in sorted(root.glob(pattern)):
            if p in seen or not p.is_file():
                continue
            seen.add(p)
            try:
                df = pd.read_csv(p, nrows=5)
            except Exception:
                continue
            cols_l = {c.lower() for c in df.columns}
            score = 0
            if "bt" in cols_l or "b" in cols_l:
                score += 5
            if "x" in cols_l:
                score += 2
            if "q" in cols_l or "qm" in cols_l:
                score += 2
            if "pid" in cols_l or "flavor" in cols_l or "flavour" in cols_l:
                score += 2
            if any(c.endswith("_median") for c in df.columns):
                score += 5
            if any(c.endswith("_q16") for c in df.columns):
                score += 3
            if "value" in cols_l:
                score += 2
            if score >= 8:
                scored.append((score, p))

    if not scored:
        listing = "\n".join(str(p) for p in sorted(root.glob("*.csv"))[:50])
        raise SystemExit(f"Could not detect a band CSV in {root}. CSV files found:\n{listing}")

    scored.sort(key=lambda x: (-x[0], str(x[1])))
    return scored[0][1]


def _infer_basic_columns(df: pd.DataFrame) -> dict[str, str | None]:
    return {
        "b": _first_col(df, ["bT", "b", "b_T", "bT_GeV_inv", "bt_gev_inv"]),
        "x": _first_col(df, ["x"]),
        "Q": _first_col(df, ["Q", "QM"]),
        "pid": _first_col(df, ["pid", "PID"]),
        "flavor": _first_col(df, ["flavor", "parton", "flavour"]),
        "quantity": _first_col(df, ["quantity", "observable"]),
    }


def _select_curve(
    df: pd.DataFrame,
    *,
    quantity: str,
    flavor: str | None,
    pid: int | None,
    x: float,
    Q: float,
) -> tuple[pd.DataFrame, dict[str, str | None]]:
    cols = _infer_basic_columns(df)
    missing = [k for k in ["b", "x", "Q"] if cols[k] is None]
    if missing:
        raise SystemExit(f"Missing required columns {missing}. Available columns: {list(df.columns)}")

    work = df.copy()

    for c in [cols["b"], cols["x"], cols["Q"]]:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    m = np.isclose(work[cols["x"]], x, rtol=0, atol=5e-8)
    m &= np.isclose(work[cols["Q"]], Q, rtol=0, atol=5e-8)

    if cols["quantity"] is not None:
        qvals = work[cols["quantity"]].astype(str)
        m &= qvals.isin(_aliases(quantity))

    chosen_pid = pid
    if chosen_pid is None and flavor is not None:
        chosen_pid = FLAVOR_TO_PID.get(flavor)

    if cols["flavor"] is not None and flavor is not None:
        m &= work[cols["flavor"]].astype(str).str.lower().eq(flavor.lower())
    elif cols["pid"] is not None and chosen_pid is not None:
        work[cols["pid"]] = pd.to_numeric(work[cols["pid"]], errors="coerce")
        m &= work[cols["pid"]].eq(chosen_pid)

    sub = work[m].copy()
    if sub.empty:
        debug = []
        if cols["quantity"] is not None:
            debug.append(f"quantities={sorted(map(str, work[cols['quantity']].dropna().unique()))[:50]}")
        if cols["flavor"] is not None:
            debug.append(f"flavors={sorted(map(str, work[cols['flavor']].dropna().unique()))[:50]}")
        if cols["pid"] is not None:
            debug.append(f"pids={sorted(pd.to_numeric(work[cols['pid']], errors='coerce').dropna().unique())[:50]}")
        debug.append(f"x_values={sorted(pd.to_numeric(work[cols['x']], errors='coerce').dropna().unique())[:50]}")
        debug.append(f"Q_values={sorted(pd.to_numeric(work[cols['Q']], errors='coerce').dropna().unique())[:50]}")
        raise SystemExit(
            "No matching curve found. "
            f"Requested quantity={quantity}, flavor={flavor}, pid={pid}, x={x}, Q={Q}.\n"
            + "\n".join(debug)
        )

    return sub, cols


def _summary_columns(df: pd.DataFrame, quantity: str) -> dict[str, str | None]:
    median_names = (
        _candidate_columns_for_quantity(quantity, ["median", "q50", "p50"])
        + ["replica_median", "median", "q50", "p50", "value_median", "band_median", "replica_q50"]
    )
    lo_names = (
        _candidate_columns_for_quantity(quantity, ["q16", "p16", "lo68", "lower68", "lower_68"])
        + ["lo68", "lower68", "q16", "p16", "value_p16", "replica_q16", "replica_lo68", "low_68", "lower_68"]
    )
    hi_names = (
        _candidate_columns_for_quantity(quantity, ["q84", "p84", "hi68", "upper68", "upper_68"])
        + ["hi68", "upper68", "q84", "p84", "value_p84", "replica_q84", "replica_hi68", "high_68", "upper_68"]
    )
    central_names = (
        _candidate_columns_for_quantity(quantity, ["central"])
        + ["central", "central_value", "frozen_central", "value_central"]
    )
    value_names = _aliases(quantity) + ["value", "replica_value"]

    return {
        "median": _first_col(df, median_names),
        "lo": _first_col(df, lo_names),
        "hi": _first_col(df, hi_names),
        "central": _first_col(df, central_names),
        "value": _first_col(df, value_names),
        "seed": _first_col(df, ["seed", "replica_seed", "replica", "replica_id"]),
    }


def _collapse_to_curve(sub: pd.DataFrame, bcol: str, quantity: str) -> pd.DataFrame:
    scols = _summary_columns(sub, quantity)
    b = pd.to_numeric(sub[bcol], errors="coerce")

    if scols["median"] and scols["lo"] and scols["hi"]:
        out = pd.DataFrame({
            "b": b,
            "median": pd.to_numeric(sub[scols["median"]], errors="coerce"),
            "lo": pd.to_numeric(sub[scols["lo"]], errors="coerce"),
            "hi": pd.to_numeric(sub[scols["hi"]], errors="coerce"),
        })
        if scols["central"]:
            out["central"] = pd.to_numeric(sub[scols["central"]], errors="coerce")
        return out.dropna(subset=["b", "median", "lo", "hi"]).sort_values("b")

    # Long replica format: use the requested quantity/value column.
    if scols["value"]:
        tmp = pd.DataFrame({
            "b": b,
            "value": pd.to_numeric(sub[scols["value"]], errors="coerce"),
        })
        if scols["central"]:
            tmp["central"] = pd.to_numeric(sub[scols["central"]], errors="coerce")
        tmp = tmp.dropna(subset=["b", "value"])
        g = tmp.groupby("b", observed=False)
        out = g["value"].quantile([0.16, 0.50, 0.84]).unstack()
        out = out.rename(columns={0.16: "lo", 0.50: "median", 0.84: "hi"}).reset_index()
        if "central" in tmp.columns:
            central = tmp.groupby("b", observed=False)["central"].median().reset_index()
            out = out.merge(central, on="b", how="left")
        return out.sort_values("b")

    raise SystemExit(
        "Could not infer value columns.\n"
        f"Requested quantity: {quantity}\n"
        f"Tried median/lo/hi columns based on aliases: {_aliases(quantity)}\n"
        f"Available columns: {list(sub.columns)}"
    )


def _load_central_curve(
    central_grid: Path | None,
    *,
    quantity: str,
    flavor: str | None,
    pid: int | None,
    x: float,
    Q: float,
) -> pd.DataFrame | None:
    if central_grid is None or not central_grid.exists():
        return None

    df = pd.read_csv(central_grid)
    sub, cols = _select_curve(df, quantity=quantity, flavor=flavor, pid=pid, x=x, Q=Q)

    value_col = None
    # Prefer exact central-grid quantity columns, not *_median.
    for q in _aliases(quantity):
        value_col = _first_col(sub, [q])
        if value_col is not None:
            break
    if value_col is None:
        # Some grids may store central value under quantity_central.
        value_col = _summary_columns(sub, quantity)["central"]
    if value_col is None:
        return None

    return pd.DataFrame({
        "b": pd.to_numeric(sub[cols["b"]], errors="coerce"),
        "central": pd.to_numeric(sub[value_col], errors="coerce"),
    }).dropna().sort_values("b")


def _sanitize_float_label(v: float) -> str:
    return (f"{v:g}").replace(".", "p").replace("-", "m")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--band-dir", required=True, help="Band directory or band CSV.")
    ap.add_argument("--central-grid", default="", help="Optional central b-space long CSV.")
    ap.add_argument("--quantity", default="ftilde", help="ftilde, x_ftilde, b_ftilde, b_x_ftilde, or F_NP.")
    ap.add_argument("--flavor", default="d", choices=["u", "d", "ubar", "dbar"])
    ap.add_argument("--pid", type=int, default=None)
    ap.add_argument("--x", type=float, default=0.10)
    ap.add_argument("--Q", type=float, default=10.0)
    ap.add_argument("--b-max", type=float, default=4.0)
    ap.add_argument("--y-max", type=float, default=None)
    ap.add_argument("--label", default="v23a FT-DY")
    ap.add_argument("--title", default="TMD PDFs")
    ap.add_argument("--out", default="")
    ap.add_argument("--no-central", action="store_true")
    ap.add_argument("--show-grid", action="store_true")
    args = ap.parse_args()

    band_csv = _find_band_csv(Path(args.band_dir))
    print(f"Using band CSV: {band_csv}", file=sys.stderr)

    df = pd.read_csv(band_csv)
    sub, cols = _select_curve(
        df,
        quantity=args.quantity,
        flavor=args.flavor,
        pid=args.pid,
        x=args.x,
        Q=args.Q,
    )
    curve = _collapse_to_curve(sub, cols["b"], args.quantity)
    curve = curve[curve["b"].between(0, args.b_max)].copy()

    if curve.empty:
        raise SystemExit("Selected curve is empty after applying b range.")

    central = None
    if not args.no_central:
        if "central" in curve.columns and curve["central"].notna().any():
            central = curve[["b", "central"]].dropna().copy()
        elif args.central_grid:
            central = _load_central_curve(
                Path(args.central_grid),
                quantity=args.quantity,
                flavor=args.flavor,
                pid=args.pid,
                x=args.x,
                Q=args.Q,
            )
            if central is not None:
                central = central[central["b"].between(0, args.b_max)].copy()

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

    fig, ax = plt.subplots(figsize=(7.2, 5.1), dpi=180)

    ax.fill_between(
        curve["b"].to_numpy(dtype=float),
        curve["lo"].to_numpy(dtype=float),
        curve["hi"].to_numpy(dtype=float),
        color="0.25",
        alpha=0.25,
        linewidth=0,
        label=f"{args.label} 68%",
    )
    ax.plot(
        curve["b"],
        curve["median"],
        color="black",
        lw=2.0,
        label=f"{args.label} median",
    )

    if central is not None and not central.empty:
        ax.plot(
            central["b"],
            central["central"],
            color="black",
            lw=1.4,
            ls="--",
            alpha=0.75,
            label="frozen central",
        )

    ax.set_xlim(0, args.b_max)
    if args.y_max is not None:
        ax.set_ylim(0, args.y_max)
    else:
        ymax = float(np.nanmax(curve["hi"].to_numpy(dtype=float)))
        if central is not None and not central.empty:
            ymax = max(ymax, float(np.nanmax(central["central"].to_numpy(dtype=float))))
        ax.set_ylim(0, ymax * 1.18 if ymax > 0 else 1.0)

    ax.set_xlabel(r"$b\;(\mathrm{GeV}^{-1})$", fontsize=18)
    if args.quantity in {"ftilde", "f_eff"}:
        ax.set_ylabel(r"$f_1(x,b)$", fontsize=18)
    elif args.quantity == "x_ftilde":
        ax.set_ylabel(r"$x f_1(x,b)$", fontsize=18)
    elif args.quantity == "F_NP":
        ax.set_ylabel(r"$F_{\rm NP}(x,b)$", fontsize=18)
    else:
        ax.set_ylabel(args.quantity, fontsize=18)

    ax.set_title(args.title, fontsize=26, fontweight="bold", pad=18)
    ax.text(
        0.97,
        0.92,
        rf"$x={args.x:g}\quad Q={args.Q:g}\,\mathrm{{GeV}}\quad {args.flavor}$-quark",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=16,
    )

    ax.tick_params(which="major", length=6, width=1.2, labelsize=14)
    ax.tick_params(which="minor", length=3, width=1.0)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    if args.show_grid:
        ax.grid(alpha=0.18, lw=0.8)

    ax.legend(loc="upper right", fontsize=12)
    fig.tight_layout()

    out = args.out
    if not out:
        xlab = _sanitize_float_label(args.x)
        qlab = _sanitize_float_label(args.Q)
        out = f"plots/traditional_bspace_tmd_{args.flavor}_x{xlab}_Q{qlab}.pdf"
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    png_path = out_path.with_suffix(".png")
    fig.savefig(png_path, bbox_inches="tight")
    print(f"wrote: {out_path}")
    print(f"wrote: {png_path}")


if __name__ == "__main__":
    main()
