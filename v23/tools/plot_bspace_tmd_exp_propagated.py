#!/usr/bin/env python3
"""Traditional b_T-space TMDPDF plot with data-replica propagated uncertainty.

The uncertainty band shown by this script is the q16--q84 envelope of the
trained pseudo-data / experimental-replica ensemble.  In the current v23a
workflow, that is the practical propagated experimental uncertainty under the
fixed backend, fixed PDF set, fixed functional form, and fixed anchor choice.

This is NOT a full theory/PDF/scale/model uncertainty.

Typical v23a usage:

  PYTHONPATH=. python3 v23/tools/plot_bspace_tmd_exp_propagated.py \
    --band-dir replica_pilot_v23a_lambda3_normpriors15_p2p5_E772_E288400_cached_cuda/tmd_bspace_bands_exactx_50rep \
    --central-grid plots/v23a_fixed_target_lowQ_normpriors15_p2p5_E772_E288400_central_exactx/v22_scheme_tmd_bspace_long.csv \
    --quantity ftilde \
    --flavor d \
    --x 0.10 \
    --Q 10 \
    --b-max 4 \
    --label "v23a FT-DY" \
    --out plots/v23a_TMDPDF_exp_propagated_d_x0p10_Q10.pdf

It writes:
  * the requested PDF/PNG plot
  * a companion CSV with b, median, q16, q84, relative halfwidth
  * a companion JSON with numerical band diagnostics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


FLAVOR_TO_PID = {"dbar": -1, "ubar": -2, "d": 1, "u": 2}

QUANTITY_ALIASES = {
    "ftilde": ["ftilde", "f_eff", "f1", "f1_eff"],
    "f_eff": ["f_eff", "ftilde", "f1", "f1_eff"],
    "x_ftilde": ["x_ftilde", "x_f_eff", "xf_eff"],
    "b_ftilde": ["b_ftilde", "b_f_eff"],
    "b_x_ftilde": ["b_x_ftilde", "b_xf_eff", "b_x_f_eff"],
    "F_NP": ["F_NP", "FNP", "fnp"],
}


def aliases(quantity: str) -> list[str]:
    return QUANTITY_ALIASES.get(quantity, [quantity])


def first_col(df: pd.DataFrame, names: Iterable[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for name in names:
        if name in df.columns:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def qcols(quantity: str, suffixes: Iterable[str]) -> list[str]:
    out: list[str] = []
    for q in aliases(quantity):
        for s in suffixes:
            out.extend([f"{q}_{s}", f"{q}{s}", f"{s}_{q}"])
    return out


def find_band_csv(root: Path) -> Path:
    if root.is_file():
        return root

    preferred = [
        "v22_tmd_replica_bspace_bands.csv",
        "v23_tmd_replica_bspace_bands.csv",
        "tmd_replica_bspace_bands.csv",
    ]
    for name in preferred:
        p = root / name
        if p.exists():
            return p

    candidates: list[tuple[int, Path]] = []
    for p in sorted(root.glob("*.csv")):
        try:
            preview = pd.read_csv(p, nrows=3)
        except Exception:
            continue
        cols_l = {c.lower() for c in preview.columns}
        score = 0
        score += 5 if ("bt" in cols_l or "b" in cols_l) else 0
        score += 2 if "x" in cols_l else 0
        score += 2 if ("q" in cols_l or "qm" in cols_l) else 0
        score += 2 if ("pid" in cols_l or "flavor" in cols_l or "flavour" in cols_l) else 0
        score += 5 if any(c.endswith("_median") for c in preview.columns) else 0
        score += 3 if any(c.endswith("_q16") for c in preview.columns) else 0
        score += 2 if "value" in cols_l else 0
        if score >= 8:
            candidates.append((score, p))
    if not candidates:
        raise SystemExit(f"No suitable band CSV found in {root}")
    candidates.sort(key=lambda item: (-item[0], str(item[1])))
    return candidates[0][1]


def basic_cols(df: pd.DataFrame) -> dict[str, str | None]:
    return {
        "b": first_col(df, ["bT", "b", "b_T", "bT_GeV_inv"]),
        "x": first_col(df, ["x"]),
        "Q": first_col(df, ["Q", "QM"]),
        "pid": first_col(df, ["pid", "PID"]),
        "flavor": first_col(df, ["flavor", "flavour", "parton"]),
        "quantity": first_col(df, ["quantity", "observable"]),
        "seed": first_col(df, ["seed", "replica_seed", "replica", "replica_id"]),
    }


def select_curve(
    df: pd.DataFrame,
    *,
    quantity: str,
    flavor: str,
    pid: int | None,
    x: float,
    Q: float,
) -> tuple[pd.DataFrame, dict[str, str | None]]:
    cols = basic_cols(df)
    missing = [k for k in ["b", "x", "Q"] if cols[k] is None]
    if missing:
        raise SystemExit(f"Missing required columns {missing}. Available: {list(df.columns)}")

    work = df.copy()
    for c in [cols["b"], cols["x"], cols["Q"]]:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    mask = np.isclose(work[cols["x"]], x, rtol=0, atol=5e-8)
    mask &= np.isclose(work[cols["Q"]], Q, rtol=0, atol=5e-8)

    if cols["quantity"] is not None:
        mask &= work[cols["quantity"]].astype(str).isin(aliases(quantity))

    chosen_pid = pid if pid is not None else FLAVOR_TO_PID.get(flavor)
    if cols["flavor"] is not None:
        mask &= work[cols["flavor"]].astype(str).str.lower().eq(flavor.lower())
    elif cols["pid"] is not None and chosen_pid is not None:
        work[cols["pid"]] = pd.to_numeric(work[cols["pid"]], errors="coerce")
        mask &= work[cols["pid"]].eq(chosen_pid)

    sub = work[mask].copy()
    if sub.empty:
        debug = {
            "available_flavors": sorted(map(str, work[cols["flavor"]].dropna().unique())) if cols["flavor"] else None,
            "available_pids": sorted(pd.to_numeric(work[cols["pid"]], errors="coerce").dropna().unique().tolist()) if cols["pid"] else None,
            "available_x": sorted(pd.to_numeric(work[cols["x"]], errors="coerce").dropna().unique().tolist()),
            "available_Q": sorted(pd.to_numeric(work[cols["Q"]], errors="coerce").dropna().unique().tolist()),
        }
        raise SystemExit(
            f"No curve found for quantity={quantity}, flavor={flavor}, x={x}, Q={Q}.\n"
            + json.dumps(debug, indent=2)
        )
    return sub, cols


def summary_cols(df: pd.DataFrame, quantity: str) -> dict[str, str | None]:
    return {
        "median": first_col(df, qcols(quantity, ["median", "q50", "p50"]) + ["median", "q50", "p50", "value_median"]),
        "lo": first_col(df, qcols(quantity, ["q16", "p16", "lo68", "lower68", "lower_68"]) + ["q16", "p16", "lo68", "lower68", "lower_68"]),
        "hi": first_col(df, qcols(quantity, ["q84", "p84", "hi68", "upper68", "upper_68"]) + ["q84", "p84", "hi68", "upper68", "upper_68"]),
        "central": first_col(df, qcols(quantity, ["central"]) + ["central", "central_value", "frozen_central"]),
        "value": first_col(df, aliases(quantity) + ["value", "replica_value"]),
        "seed": first_col(df, ["seed", "replica_seed", "replica", "replica_id"]),
    }


def collapse_curve(sub: pd.DataFrame, b_col: str, quantity: str) -> tuple[pd.DataFrame, int | None]:
    """Return b, median, q16, q84 and optional central.

    n_replicas is None if the input file is already summarized.
    """
    sc = summary_cols(sub, quantity)

    b = pd.to_numeric(sub[b_col], errors="coerce")

    # Already summarized band file.
    if sc["median"] and sc["lo"] and sc["hi"]:
        out = pd.DataFrame({
            "b": b,
            "median": pd.to_numeric(sub[sc["median"]], errors="coerce"),
            "q16": pd.to_numeric(sub[sc["lo"]], errors="coerce"),
            "q84": pd.to_numeric(sub[sc["hi"]], errors="coerce"),
        })
        if sc["central"]:
            out["central"] = pd.to_numeric(sub[sc["central"]], errors="coerce")
        out = out.dropna(subset=["b", "median", "q16", "q84"]).sort_values("b")
        return out, None

    # Long individual-replica file.
    if sc["value"]:
        tmp = pd.DataFrame({
            "b": b,
            "value": pd.to_numeric(sub[sc["value"]], errors="coerce"),
        })
        if sc["seed"]:
            tmp["seed"] = sub[sc["seed"]]
        if sc["central"]:
            tmp["central"] = pd.to_numeric(sub[sc["central"]], errors="coerce")
        tmp = tmp.dropna(subset=["b", "value"])
        g = tmp.groupby("b", observed=False)
        out = g["value"].quantile([0.16, 0.50, 0.84]).unstack()
        out = out.rename(columns={0.16: "q16", 0.50: "median", 0.84: "q84"}).reset_index()
        if "central" in tmp.columns:
            c = tmp.groupby("b", observed=False)["central"].median().reset_index()
            out = out.merge(c, on="b", how="left")
        n_rep = int(tmp["seed"].nunique()) if "seed" in tmp.columns else None
        return out.sort_values("b"), n_rep

    raise SystemExit(
        f"Could not infer band/value columns for quantity={quantity}. "
        f"Expected e.g. {quantity}_median, {quantity}_q16, {quantity}_q84. "
        f"Available: {list(sub.columns)}"
    )


def load_central(path: str, quantity: str, flavor: str, pid: int | None, x: float, Q: float) -> pd.DataFrame | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    sub, cols = select_curve(df, quantity=quantity, flavor=flavor, pid=pid, x=x, Q=Q)
    value_col = None
    for q in aliases(quantity):
        value_col = first_col(sub, [q])
        if value_col:
            break
    if value_col is None:
        return None
    return pd.DataFrame({
        "b": pd.to_numeric(sub[cols["b"]], errors="coerce"),
        "central": pd.to_numeric(sub[value_col], errors="coerce"),
    }).dropna().sort_values("b")


def y_label(quantity: str) -> str:
    if quantity in {"ftilde", "f_eff"}:
        return r"$f_1(x,b)$"
    if quantity == "x_ftilde":
        return r"$x f_1(x,b)$"
    if quantity == "b_ftilde":
        return r"$b f_1(x,b)$"
    if quantity == "b_x_ftilde":
        return r"$b x f_1(x,b)$"
    if quantity == "F_NP":
        return r"$F_{\rm NP}(x,b)$"
    return quantity


def make_diagnostics(curve: pd.DataFrame, n_replicas: int | None, args: argparse.Namespace) -> dict[str, object]:
    eps = 1e-300
    active = curve[np.abs(curve["median"]) > eps].copy()
    active["halfwidth"] = 0.5 * (active["q84"] - active["q16"])
    active["rel_halfwidth"] = active["halfwidth"] / np.maximum(np.abs(active["median"]), eps)

    diag = {
        "quantity": args.quantity,
        "flavor": args.flavor,
        "x": args.x,
        "Q": args.Q,
        "b_max": args.b_max,
        "n_b_points": int(len(curve)),
        "n_replicas_if_long_file": n_replicas,
        "median_rel_halfwidth": float(active["rel_halfwidth"].median()),
        "p90_rel_halfwidth": float(active["rel_halfwidth"].quantile(0.90)),
        "max_rel_halfwidth": float(active["rel_halfwidth"].max()),
        "median_abs_halfwidth": float(active["halfwidth"].median()),
        "max_abs_halfwidth": float(active["halfwidth"].max()),
        "interpretation": (
            "Band is q16--q84 over the pseudo-data/data-replica TMD ensemble. "
            "It propagates the experimental data-replica uncertainty under the fixed fit setup; "
            "it does not include PDF, scale, model-form, or profile variations unless those were varied in the replicas."
        ),
    }
    return diag


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--band-dir", required=True, help="Band directory or CSV.")
    ap.add_argument("--central-grid", default="", help="Optional central-grid CSV.")
    ap.add_argument("--quantity", default="ftilde", help="ftilde, x_ftilde, b_ftilde, b_x_ftilde, F_NP.")
    ap.add_argument("--flavor", default="d", choices=["u", "d", "ubar", "dbar"])
    ap.add_argument("--pid", type=int, default=None)
    ap.add_argument("--x", type=float, default=0.10)
    ap.add_argument("--Q", type=float, default=10.0)
    ap.add_argument("--b-max", type=float, default=4.0)
    ap.add_argument("--label", default="v23a FT-DY")
    ap.add_argument("--title", default="TMD PDFs")
    ap.add_argument("--out", default="")
    ap.add_argument("--no-central", action="store_true")
    ap.add_argument("--no-inset", action="store_true")
    ap.add_argument("--band-scale", type=float, default=1.0, help="Visual-only scale for q16/q84 deviations from median.")
    ap.add_argument("--legend-outside", action="store_true")
    args = ap.parse_args()

    band_csv = find_band_csv(Path(args.band_dir))
    print(f"Using band CSV: {band_csv}")

    df = pd.read_csv(band_csv)
    sub, cols = select_curve(df, quantity=args.quantity, flavor=args.flavor, pid=args.pid, x=args.x, Q=args.Q)
    curve, n_rep = collapse_curve(sub, cols["b"], args.quantity)
    curve = curve[curve["b"].between(0, args.b_max)].copy()
    if curve.empty:
        raise SystemExit("Selected curve is empty after b cut.")

    # Optional central curve.
    central = None
    if not args.no_central:
        if "central" in curve.columns and curve["central"].notna().any():
            central = curve[["b", "central"]].dropna()
        elif args.central_grid:
            central = load_central(args.central_grid, args.quantity, args.flavor, args.pid, args.x, args.Q)
            if central is not None:
                central = central[central["b"].between(0, args.b_max)].copy()

    # Prepare plotting band. band-scale is explicitly visual-only.
    plot_curve = curve.copy()
    if args.band_scale != 1.0:
        plot_curve["q16_plot"] = plot_curve["median"] - args.band_scale * (plot_curve["median"] - plot_curve["q16"])
        plot_curve["q84_plot"] = plot_curve["median"] + args.band_scale * (plot_curve["q84"] - plot_curve["median"])
    else:
        plot_curve["q16_plot"] = plot_curve["q16"]
        plot_curve["q84_plot"] = plot_curve["q84"]

    diag = make_diagnostics(curve, n_rep, args)

    # Add computed columns to companion CSV.
    eps = 1e-300
    curve["abs_halfwidth"] = 0.5 * (curve["q84"] - curve["q16"])
    curve["rel_halfwidth"] = curve["abs_halfwidth"] / np.maximum(np.abs(curve["median"]), eps)

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

    fig, ax = plt.subplots(figsize=(7.6, 5.2), dpi=180)

    b = plot_curve["b"].to_numpy(dtype=float)
    med = plot_curve["median"].to_numpy(dtype=float)
    q16 = plot_curve["q16_plot"].to_numpy(dtype=float)
    q84 = plot_curve["q84_plot"].to_numpy(dtype=float)

    band_label = f"{args.label} exp.-replica 68%"
    if args.band_scale != 1.0:
        band_label += rf" ($\times {args.band_scale:g}$ visual)"

    ax.fill_between(b, q16, q84, color="0.55", alpha=0.38, linewidth=0, label=band_label)
    ax.plot(b, q16, color="0.45", lw=0.9)
    ax.plot(b, q84, color="0.45", lw=0.9)
    ax.plot(b, med, color="black", lw=2.0, label=f"{args.label} median")

    if central is not None and not central.empty:
        ax.plot(central["b"], central["central"], color="black", ls="--", lw=1.5, alpha=0.8, label="frozen central")

    ymax = float(np.nanmax(q84))
    if central is not None and not central.empty:
        ymax = max(ymax, float(np.nanmax(central["central"].to_numpy(dtype=float))))
    ax.set_ylim(0.0, ymax * 1.18 if ymax > 0 else 1.0)
    ax.set_xlim(0.0, args.b_max)

    ax.set_title(args.title, fontsize=26, fontweight="bold", pad=16)
    ax.set_xlabel(r"$b\;(\mathrm{GeV}^{-1})$", fontsize=18)
    ax.set_ylabel(y_label(args.quantity), fontsize=18)

    ax.text(
        0.03,
        0.93,
        rf"$x={args.x:g}\quad Q={args.Q:g}\,\mathrm{{GeV}}\quad {args.flavor}$-quark",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=15,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.80, "pad": 2},
    )

    ax.tick_params(which="major", length=6, width=1.2, labelsize=13)
    ax.tick_params(which="minor", length=3, width=1.0)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    if args.legend_outside:
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.72), fontsize=11)
    else:
        ax.legend(loc="upper right", fontsize=11)

    if not args.no_inset:
        inset = inset_axes(ax, width="38%", height="28%", loc="center right", borderpad=1.2)
        inset.plot(curve["b"], 100.0 * curve["rel_halfwidth"], color="black", lw=1.3)
        inset.set_title(r"68% halfwidth", fontsize=9)
        inset.set_xlabel(r"$b$", fontsize=8)
        inset.set_ylabel(r"%", fontsize=8)
        inset.tick_params(axis="both", labelsize=7, direction="in", top=True, right=True)
        inset.xaxis.set_minor_locator(AutoMinorLocator())
        inset.yaxis.set_minor_locator(AutoMinorLocator())
        ymax_inset = float(np.nanmax(100.0 * curve["rel_halfwidth"]))
        inset.set_ylim(0.0, ymax_inset * 1.2 if ymax_inset > 0 else 1.0)
        inset.set_xlim(0.0, args.b_max)

    fig.tight_layout()

    out = args.out
    if not out:
        xlab = f"{args.x:g}".replace(".", "p")
        qlab = f"{args.Q:g}".replace(".", "p")
        out = f"plots/exp_propagated_TMDPDF_{args.flavor}_x{xlab}_Q{qlab}.pdf"

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    png_path = out_path.with_suffix(".png")
    fig.savefig(png_path, bbox_inches="tight")

    csv_path = out_path.with_suffix(".curve.csv")
    json_path = out_path.with_suffix(".diagnostics.json")
    curve.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(diag, indent=2) + "\n")

    print(json.dumps(diag, indent=2))
    print(f"wrote: {out_path}")
    print(f"wrote: {png_path}")
    print(f"wrote: {csv_path}")
    print(f"wrote: {json_path}")


if __name__ == "__main__":
    main()
