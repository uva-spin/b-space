#!/usr/bin/env python3
r"""PRD-style all-flavor b_T-space TMDPDF figure.

The default figure compares the four light flavors

    u, d, ubar, dbar

at one representative fixed-target point, x=0.10 and Q=7.5 GeV.  It plots the
absolute b-space TMDPDF,

    \widetilde f_1^{q/p}(x,b_T;Q),

with one median curve and one q16--q84 (68%) exp+PDF-overlay band per flavor.

The design is intentionally conventional and compact:
  * one normal 2D axis;
  * fixed x and Q so the flavor normalizations are directly comparable;
  * color plus distinct line styles for print/grayscale robustness;
  * no thin replica curves in the main paper figure;
  * no central-PDF0 lines by default, to avoid eight overlapping curves.

Expected band-table formats include:
  v23a_dataPDF_tmd_replica_bspace_bands.csv
  v22_tmd_replica_bspace_bands.csv

with columns such as:
  pid, flavor, x, Q, bT,
  ftilde_median, ftilde_q16, ftilde_q84,
  x_ftilde_median, x_ftilde_q16, x_ftilde_q84.
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
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter


FLAVOR_TO_PID = {"sbar": -3, "dbar": -1, "ubar": -2, "d": 1, "u": 2, "s": 3}
FLAVOR_TEX = {
    "u": r"$u$",
    "d": r"$d$",
    "s": r"$s$",
    "ubar": r"$\bar u$",
    "dbar": r"$\bar d$",
    "sbar": r"$\bar s$",
}
LINESTYLES = {
    "u": "-",
    "d": "--",
    "s": (0, (5, 1.5)),
    "ubar": "-.",
    "dbar": ":",
    "sbar": (0, (1, 1.6)),
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
        "v23a_tmd_replica_bspace_bands.csv",
        "tmd_replica_bspace_bands.csv",
    ]
    for name in preferred:
        p = band_dir / name
        if p.exists():
            return p

    candidates = sorted(band_dir.glob("*bspace*bands*.csv"))
    if not candidates:
        candidates = sorted(band_dir.glob("*bands*.csv"))
    if candidates:
        return candidates[0]

    raise SystemExit(f"Could not find a b-space band CSV in {band_dir}")


def quantity_columns(df: pd.DataFrame, quantity: str) -> tuple[str, str, str]:
    med = first_col(
        df,
        [f"{quantity}_median", f"{quantity}_q50", f"{quantity}_p50", "median"],
    )
    lo = first_col(
        df,
        [f"{quantity}_q16", f"{quantity}_p16", f"{quantity}_lo68", "q16"],
    )
    hi = first_col(
        df,
        [f"{quantity}_q84", f"{quantity}_p84", f"{quantity}_hi68", "q84"],
    )
    if med is None or lo is None or hi is None:
        raise SystemExit(
            f"Could not identify {quantity} median/q16/q84 columns.\n"
            f"Available columns: {list(df.columns)}"
        )
    return med, lo, hi


def select_curve(
    df: pd.DataFrame,
    *,
    quantity: str,
    flavor: str,
    x: float,
    Q: float,
    b_max: float,
) -> pd.DataFrame:
    pid = FLAVOR_TO_PID[flavor]
    med_col, lo_col, hi_col = quantity_columns(df, quantity)

    m = (
        pd.to_numeric(df["pid"], errors="coerce").eq(pid)
        & df["flavor"].astype(str).eq(flavor)
        & np.isclose(
            pd.to_numeric(df["x"], errors="coerce"),
            float(x),
            rtol=0,
            atol=1e-10,
        )
        & np.isclose(
            pd.to_numeric(df["Q"], errors="coerce"),
            float(Q),
            rtol=0,
            atol=1e-10,
        )
        & (pd.to_numeric(df["bT"], errors="coerce") <= float(b_max))
    )

    g = df[m].copy()
    if g.empty:
        return pd.DataFrame()

    out = pd.DataFrame(
        {
            "bT": pd.to_numeric(g["bT"], errors="coerce"),
            "median": pd.to_numeric(g[med_col], errors="coerce"),
            "q16": pd.to_numeric(g[lo_col], errors="coerce"),
            "q84": pd.to_numeric(g[hi_col], errors="coerce"),
        }
    )
    return out.dropna().sort_values("bT").drop_duplicates("bT")


def available_summary(df: pd.DataFrame) -> dict:
    return {
        "flavors": sorted(df["flavor"].dropna().astype(str).unique().tolist())
        if "flavor" in df.columns
        else [],
        "pids": sorted(
            pd.to_numeric(df["pid"], errors="coerce").dropna().astype(int).unique().tolist()
        )
        if "pid" in df.columns
        else [],
        "x": sorted(pd.to_numeric(df["x"], errors="coerce").dropna().unique().tolist())
        if "x" in df.columns
        else [],
        "Q": sorted(pd.to_numeric(df["Q"], errors="coerce").dropna().unique().tolist())
        if "Q" in df.columns
        else [],
    }


def y_label(quantity: str) -> str:
    if quantity == "ftilde":
        return r"$\widetilde f_1^{\,q/p}(x,b_T;Q)$"
    if quantity == "x_ftilde":
        return r"$x\,\widetilde f_1^{\,q/p}(x,b_T;Q)$"
    return quantity


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--band-dir", required=True)
    ap.add_argument("--bands-csv", default=None)
    ap.add_argument(
        "--quantity",
        default="ftilde",
        choices=["ftilde", "x_ftilde"],
    )
    ap.add_argument(
        "--flavors",
        nargs="+",
        default=["u", "d", "ubar", "dbar"],
        choices=["u", "d", "s", "ubar", "dbar", "sbar"],
    )
    ap.add_argument("--x", type=float, default=0.10)
    ap.add_argument("--Q", type=float, default=7.5)
    ap.add_argument("--b-max", type=float, default=4.0)
    ap.add_argument("--yscale", choices=["linear", "log"], default="linear")
    ap.add_argument("--band-alpha", type=float, default=0.14)
    ap.add_argument("--line-width", type=float, default=2.25)
    ap.add_argument("--figure-width", type=float, default=7.0)
    ap.add_argument("--figure-height", type=float, default=4.8)
    ap.add_argument("--legend-ncol", type=int, default=2)
    ap.add_argument("--legend-loc", default="upper right")
    ap.add_argument(
        "--legend-mode",
        choices=["inside", "top"],
        default="inside",
        help="Place the flavor legend inside the axes or in a strip above it.",
    )
    ap.add_argument(
        "--kinematics-location",
        choices=["title", "inside", "none"],
        default="title",
        help="Place x,Q above the axes, inside the axes, or omit them.",
    )
    ap.add_argument(
        "--show-band-note",
        action="store_true",
        help="Draw the uncertainty explanation inside the panel. Off by default; use the caption for PRD.",
    )
    ap.add_argument("--top-headroom", type=float, default=1.10)
    ap.add_argument("--show-grid", action="store_true")
    ap.add_argument("--title", default="")
    ap.add_argument("--uncertainty-label", default="shaded: 68% replica band")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if not 0 < args.band_alpha < 1:
        raise SystemExit("--band-alpha must lie between 0 and 1.")

    bands_csv = find_bands_csv(Path(args.band_dir), args.bands_csv)
    df = pd.read_csv(bands_csv)

    required = ["pid", "flavor", "x", "Q", "bT"]
    missing_columns = [c for c in required if c not in df.columns]
    if missing_columns:
        raise SystemExit(
            f"Band table missing columns {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )

    curves: dict[str, pd.DataFrame] = {}
    missing_flavors: list[str] = []
    for flavor in args.flavors:
        g = select_curve(
            df,
            quantity=args.quantity,
            flavor=flavor,
            x=args.x,
            Q=args.Q,
            b_max=args.b_max,
        )
        if g.empty:
            missing_flavors.append(flavor)
        else:
            curves[flavor] = g

    if missing_flavors:
        raise SystemExit(
            "No matching curve(s) for "
            + ", ".join(missing_flavors)
            + f" at x={args.x:g}, Q={args.Q:g}.\n"
            + "Available content:\n"
            + json.dumps(available_summary(df), indent=2)
            + "\nFor the Q=7.5 PRD figure, reconstruct the b-space overlay with "
              "--pids 2 1 3 -2 -1 -3."
        )

    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.linewidth": 1.15,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "legend.frameon": False,
            "savefig.dpi": 300,
        }
    )

    fig, ax = plt.subplots(
        figsize=(args.figure_width, args.figure_height),
        dpi=180,
    )

    line_handles: list[Line2D] = []
    export_rows: list[pd.DataFrame] = []
    diagnostics: dict[str, dict] = {}

    for flavor in args.flavors:
        g = curves[flavor]
        b = g["bT"].to_numpy(float)
        med = g["median"].to_numpy(float)
        lo = g["q16"].to_numpy(float)
        hi = g["q84"].to_numpy(float)

        if args.yscale == "log" and np.any(lo <= 0):
            raise SystemExit(
                f"{flavor} q16 contains nonpositive values, so --yscale log "
                "cannot be used without altering the band."
            )

        line, = ax.plot(
            b,
            med,
            linestyle=LINESTYLES[flavor],
            linewidth=args.line_width,
            label=FLAVOR_TEX[flavor],
            zorder=3,
        )
        color = line.get_color()
        ax.fill_between(
            b,
            lo,
            hi,
            color=color,
            alpha=args.band_alpha,
            linewidth=0,
            zorder=2,
        )

        line_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linestyle=LINESTYLES[flavor],
                linewidth=args.line_width,
                label=FLAVOR_TEX[flavor],
            )
        )

        e = g.copy()
        e.insert(0, "flavor", flavor)
        e.insert(1, "pid", FLAVOR_TO_PID[flavor])
        e.insert(2, "x", float(args.x))
        e.insert(3, "Q", float(args.Q))
        export_rows.append(e)

        half = 0.5 * (hi - lo)
        peak = float(np.nanmax(np.abs(med)))
        active = np.abs(med) > 0.05 * max(peak, 1e-300)
        rel = half / np.maximum(np.abs(med), 1e-300)
        diagnostics[flavor] = {
            "pid": FLAVOR_TO_PID[flavor],
            "n_points": int(len(g)),
            "median_at_b0": float(med[np.argmin(np.abs(b))]),
            "peak_abs_median": peak,
            "relative_68_halfwidth_median_active": (
                float(np.nanmedian(rel[active])) if np.any(active) else None
            ),
            "relative_68_halfwidth_p90_active": (
                float(np.nanquantile(rel[active], 0.90)) if np.any(active) else None
            ),
        }

    ax.set_xlim(0.0, float(args.b_max))
    ax.set_yscale(args.yscale)

    if args.yscale == "linear":
        ymin = min(float(g["q16"].min()) for g in curves.values())
        ymax = max(float(g["q84"].max()) for g in curves.values())
        if ymin >= 0:
            ax.set_ylim(0.0, float(args.top_headroom) * ymax)
        else:
            pad = 0.08 * (ymax - ymin)
            ax.set_ylim(ymin - pad, ymax + pad)
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    else:
        all_positive = np.concatenate(
            [g["q16"].to_numpy(float) for g in curves.values()]
        )
        ymin = float(np.nanmin(all_positive[all_positive > 0]))
        ymax = max(float(g["q84"].max()) for g in curves.values())
        ax.set_ylim(0.8 * ymin, 1.3 * ymax)
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
        ax.yaxis.set_minor_formatter(NullFormatter())

    ax.set_xlabel(r"$b_T\;[\mathrm{GeV}^{-1}]$", fontsize=16)
    ax.set_ylabel(y_label(args.quantity), fontsize=16)
    ax.xaxis.set_minor_locator(AutoMinorLocator())

    kinematics_text = rf"$x={args.x:g},\quad Q={args.Q:g}\,\mathrm{{GeV}}$"

    if args.title:
        title_text = args.title
    elif args.kinematics_location == "title":
        title_text = kinematics_text
    else:
        title_text = ""

    if title_text:
        ax.set_title(title_text, loc="left", fontsize=14.5, pad=8)

    if args.kinematics_location == "inside":
        ax.text(
            0.035,
            0.955,
            kinematics_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12.5,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.90,
                "pad": 2.0,
            },
            zorder=10,
        )

    if args.show_band_note:
        ax.text(
            0.035,
            0.885,
        str(args.uncertainty_label),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.5,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.90,
                "pad": 1.5,
            },
            zorder=10,
        )

    if args.show_grid:
        ax.grid(alpha=0.16)

    ax.tick_params(which="major", length=6, width=1.1, labelsize=12)
    ax.tick_params(which="minor", length=3, width=0.9)

    legend_kwargs = {
        "handles": line_handles,
        "ncol": args.legend_ncol,
        "fontsize": 11.5,
        "handlelength": 2.8,
        "columnspacing": 1.45,
        "frameon": True,
        "framealpha": 0.92,
        "facecolor": "white",
        "edgecolor": "none",
        "borderpad": 0.35,
        "labelspacing": 0.45,
    }

    if args.legend_mode == "inside":
        ax.legend(loc=args.legend_loc, **legend_kwargs)
        fig.tight_layout()
    else:
        fig.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 0.985),
            **legend_kwargs,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.89))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")

    table = pd.concat(export_rows, ignore_index=True)
    table.to_csv(out.with_suffix(".curves.csv"), index=False)

    summary = {
        "bands_csv": str(bands_csv),
        "quantity": args.quantity,
        "flavors": args.flavors,
        "x": float(args.x),
        "Q": float(args.Q),
        "b_max": float(args.b_max),
        "yscale": args.yscale,
        "kinematics_location": args.kinematics_location,
        "legend_mode": args.legend_mode,
        "show_band_note": bool(args.show_band_note),
        "top_headroom": float(args.top_headroom),
        "uncertainty": str(args.uncertainty_label),
        "flavor_diagnostics": diagnostics,
        "outputs": {
            "pdf": str(out),
            "png": str(out.with_suffix(".png")),
            "curves_csv": str(out.with_suffix(".curves.csv")),
            "diagnostics_json": str(out.with_suffix(".diagnostics.json")),
        },
    }
    out.with_suffix(".diagnostics.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print(json.dumps(summary, indent=2))
    print("wrote:", out)
    print("wrote:", out.with_suffix(".png"))
    print("wrote:", out.with_suffix(".curves.csv"))
    print("wrote:", out.with_suffix(".diagnostics.json"))


if __name__ == "__main__":
    main()
