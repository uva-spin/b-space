#!/usr/bin/env python3
"""Improved b-space TMD band plots for v23a data/PDF ensembles.

Reads the ensemble CSVs produced by construct_v23a_data_pdf_bspace_tmd_bands_v2.py:

  v23a_dataPDF_tmd_replica_bspace_long.csv
  v23a_dataPDF_tmd_replica_bspace_bands.csv

and makes multi-page PDFs with:
  * solid colored ensemble median curves;
  * same-color 68% shaded bands;
  * optional thin replica curves;
  * optional dashed central-grid curves.

The central grid should normally be the member-0 central fit grid, e.g.

  plots/v23a_fixed_target_lowQ_normpriors15_p2p5_E772_E288400_central_exactx/v22_scheme_tmd_bspace_long.csv

Examples:

  PYTHONPATH=. python3 v23/tools/plot_v23a_data_pdf_bspace_bands_improved.py \
    --band-dir replica_v23a_expPDF_overlay_lambda3_normpriors15_p2p5_50rep/tmd_bspace_bands_expPDF_overlay \
    --central-grid plots/v23a_fixed_target_lowQ_normpriors15_p2p5_E772_E288400_central_exactx/v22_scheme_tmd_bspace_long.csv \
    --out-dir replica_v23a_expPDF_overlay_lambda3_normpriors15_p2p5_50rep/tmd_bspace_bands_expPDF_overlay/improved_plots \
    --replica-thin 10

  # Traditional single-page d-quark x=0.1, Q=10 plot:
  PYTHONPATH=. python3 v23/tools/plot_v23a_data_pdf_bspace_bands_improved.py \
    --band-dir replica_v23a_expPDF_overlay_lambda3_normpriors15_p2p5_50rep/tmd_bspace_bands_expPDF_overlay \
    --central-grid plots/v23a_fixed_target_lowQ_normpriors15_p2p5_E772_E288400_central_exactx/v22_scheme_tmd_bspace_long.csv \
    --out-dir plots/v23a_traditional_expPDF_overlay \
    --quantities ftilde \
    --flavors d \
    --x-values 0.10 \
    --Q-values 10 \
    --b-max 4 \
    --central-style black \
    --replica-style gray \
    --replica-thin 12
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
from matplotlib.backends.backend_pdf import PdfPages


QUANTITY_LABELS = {
    "F_NP": r"$F_{\rm NP}(x,b_T)$",
    "ftilde": r"$f_1(x,b_T)$",
    "x_ftilde": r"$x\,f_1(x,b_T)$",
    "b_ftilde": r"$b_T\,f_1(x,b_T)$",
    "b_x_ftilde": r"$b_T\,x\,f_1(x,b_T)$",
}

QUANTITY_TITLES = {
    "F_NP": "F_NP",
    "ftilde": "ftilde",
    "x_ftilde": "x_ftilde",
    "b_ftilde": "b_ftilde",
    "b_x_ftilde": "b_x_ftilde",
}


def parse_float_tokens(tokens: list[str] | None) -> list[float] | None:
    if not tokens:
        return None
    vals: list[float] = []
    for tok in tokens:
        for piece in str(tok).replace(",", " ").split():
            vals.append(float(piece))
    return vals


def parse_int_tokens(tokens: list[str] | None) -> list[int] | None:
    if not tokens:
        return None
    vals: list[int] = []
    for tok in tokens:
        for piece in str(tok).replace(",", " ").split():
            vals.append(int(piece))
    return vals


def add_derived_quantities(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "ftilde" in df.columns:
        if "x_ftilde" not in df.columns and "x" in df.columns:
            df["x_ftilde"] = pd.to_numeric(df["x"], errors="coerce") * pd.to_numeric(df["ftilde"], errors="coerce")
        if "b_ftilde" not in df.columns and "bT" in df.columns:
            df["b_ftilde"] = pd.to_numeric(df["bT"], errors="coerce") * pd.to_numeric(df["ftilde"], errors="coerce")
        if "b_x_ftilde" not in df.columns and {"bT", "x"}.issubset(df.columns):
            df["b_x_ftilde"] = (
                pd.to_numeric(df["bT"], errors="coerce")
                * pd.to_numeric(df["x"], errors="coerce")
                * pd.to_numeric(df["ftilde"], errors="coerce")
            )
    return df


def add_derived_band_quantities(df: pd.DataFrame) -> pd.DataFrame:
    """Bands usually already have derived columns. Compute them if needed."""
    df = df.copy()
    for base in ["median", "q16", "q84"]:
        fcol = f"ftilde_{base}"
        if fcol not in df.columns:
            continue
        if f"x_ftilde_{base}" not in df.columns and "x" in df.columns:
            df[f"x_ftilde_{base}"] = pd.to_numeric(df["x"], errors="coerce") * pd.to_numeric(df[fcol], errors="coerce")
        if f"b_ftilde_{base}" not in df.columns and "bT" in df.columns:
            df[f"b_ftilde_{base}"] = pd.to_numeric(df["bT"], errors="coerce") * pd.to_numeric(df[fcol], errors="coerce")
        if f"b_x_ftilde_{base}" not in df.columns and {"bT", "x"}.issubset(df.columns):
            df[f"b_x_ftilde_{base}"] = (
                pd.to_numeric(df["bT"], errors="coerce")
                * pd.to_numeric(df["x"], errors="coerce")
                * pd.to_numeric(df[fcol], errors="coerce")
            )
    return df


def near_filter(series: pd.Series, values: Iterable[float] | None, *, atol: float = 1e-9) -> np.ndarray:
    if values is None:
        return np.ones(len(series), dtype=bool)
    arr = pd.to_numeric(series, errors="coerce").to_numpy(float)
    mask = np.zeros(len(series), dtype=bool)
    for v in values:
        mask |= np.isclose(arr, float(v), rtol=0, atol=atol)
    return mask


def filter_table(
    df: pd.DataFrame,
    *,
    x_values: list[float] | None,
    Q_values: list[float] | None,
    pids: list[int] | None,
    flavors: list[str] | None,
    b_max: float | None,
) -> pd.DataFrame:
    out = df.copy()
    if x_values is not None and "x" in out.columns:
        out = out[near_filter(out["x"], x_values)]
    if Q_values is not None and "Q" in out.columns:
        out = out[near_filter(out["Q"], Q_values)]
    if pids is not None and "pid" in out.columns:
        out = out[out["pid"].astype(int).isin([int(p) for p in pids])]
    if flavors is not None and "flavor" in out.columns:
        out = out[out["flavor"].astype(str).isin([str(f) for f in flavors])]
    if b_max is not None and "bT" in out.columns:
        out = out[pd.to_numeric(out["bT"], errors="coerce") <= float(b_max)]
    return out


def replica_key_frame(long: pd.DataFrame) -> pd.DataFrame:
    out = long.copy()
    seed = out["seed"].astype(str) if "seed" in out.columns else pd.Series(["unknown"] * len(out), index=out.index)
    pdf = out["pdf_member"].astype(str) if "pdf_member" in out.columns else pd.Series([""] * len(out), index=out.index)
    out["_replica_key"] = seed + "|pdf" + pdf
    return out


def select_replica_keys(long: pd.DataFrame, n: int, random_seed: int) -> list[str]:
    if n <= 0 or "_replica_key" not in long.columns:
        return []
    keys = sorted(long["_replica_key"].dropna().unique().tolist())
    if len(keys) <= n:
        return keys
    rng = np.random.default_rng(int(random_seed))
    pick = rng.choice(keys, size=int(n), replace=False)
    return sorted([str(k) for k in pick])


def sorted_unique_float(df: pd.DataFrame, col: str) -> list[float]:
    return sorted(float(x) for x in pd.to_numeric(df[col], errors="coerce").dropna().unique())


def sorted_unique_int(df: pd.DataFrame, col: str) -> list[int]:
    return sorted(int(x) for x in pd.to_numeric(df[col], errors="coerce").dropna().unique())


def values_for(
    df: pd.DataFrame,
    *,
    pid: int,
    flavor: str | None,
    Q: float,
    x: float,
    quantity: str,
) -> pd.DataFrame:
    g = df[
        (df["pid"].astype(int) == int(pid))
        & np.isclose(pd.to_numeric(df["Q"], errors="coerce"), float(Q), rtol=0, atol=1e-9)
        & np.isclose(pd.to_numeric(df["x"], errors="coerce"), float(x), rtol=0, atol=1e-9)
    ].copy()
    if flavor is not None and "flavor" in g.columns:
        # pid is authoritative; flavor only guards against accidental mixed labels.
        g = g[g["flavor"].astype(str) == str(flavor)]
    if quantity not in g.columns:
        return pd.DataFrame()
    g = g[["bT", quantity] + (["_replica_key"] if "_replica_key" in g.columns else [])].dropna()
    return g.sort_values("bT")


def plot_quantity(
    quantity: str,
    bands: pd.DataFrame,
    long: pd.DataFrame,
    central: pd.DataFrame | None,
    out_pdf: Path,
    args: argparse.Namespace,
) -> dict:
    required = [f"{quantity}_median", f"{quantity}_q16", f"{quantity}_q84"]
    missing = [c for c in required if c not in bands.columns]
    if missing:
        raise SystemExit(f"Quantity {quantity!r} missing required band columns: {missing}")

    replica_keys = select_replica_keys(long, int(args.replica_thin), int(args.random_seed))
    pages_written = 0
    curves_written = 0

    # Page order: Q, pid. Flavor is taken from the first matching row.
    page_keys = (
        bands[["Q", "pid", "flavor"]]
        .drop_duplicates()
        .sort_values(["Q", "pid", "flavor"])
        .itertuples(index=False, name=None)
    )

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_pdf) as pdf:
        for Q, pid, flavor in page_keys:
            page = bands[
                np.isclose(pd.to_numeric(bands["Q"], errors="coerce"), float(Q), rtol=0, atol=1e-9)
                & (bands["pid"].astype(int) == int(pid))
                & (bands["flavor"].astype(str) == str(flavor))
            ].copy()
            if page.empty:
                continue

            xs = sorted_unique_float(page, "x")
            if not xs:
                continue

            fig, ax = plt.subplots(figsize=(9.0, 5.8))
            did_band_label = False
            did_replica_label = False
            did_central_label = False

            for x in xs:
                g = page[np.isclose(pd.to_numeric(page["x"], errors="coerce"), float(x), rtol=0, atol=1e-9)].sort_values("bT")
                if g.empty:
                    continue

                b = pd.to_numeric(g["bT"], errors="coerce").to_numpy(float)
                q16 = pd.to_numeric(g[f"{quantity}_q16"], errors="coerce").to_numpy(float)
                med = pd.to_numeric(g[f"{quantity}_median"], errors="coerce").to_numpy(float)
                q84 = pd.to_numeric(g[f"{quantity}_q84"], errors="coerce").to_numpy(float)
                finite = np.isfinite(b) & np.isfinite(q16) & np.isfinite(med) & np.isfinite(q84)
                if not np.any(finite):
                    continue

                (median_line,) = ax.plot(
                    b[finite],
                    med[finite],
                    lw=float(args.median_lw),
                    label=f"x={x:g} median",
                    zorder=4,
                )
                color = median_line.get_color()
                curves_written += 1

                ax.fill_between(
                    b[finite],
                    q16[finite],
                    q84[finite],
                    color=color,
                    alpha=float(args.band_alpha),
                    linewidth=0,
                    label=args.band_label if not did_band_label else None,
                    zorder=2,
                )
                did_band_label = True

                if replica_keys:
                    for rk in replica_keys:
                        sub = long[long["_replica_key"].astype(str) == str(rk)]
                        rg = values_for(sub, pid=int(pid), flavor=str(flavor), Q=float(Q), x=float(x), quantity=quantity)
                        if rg.empty:
                            continue
                        rcolor = "0.55" if args.replica_style == "gray" else color
                        ax.plot(
                            pd.to_numeric(rg["bT"], errors="coerce").to_numpy(float),
                            pd.to_numeric(rg[quantity], errors="coerce").to_numpy(float),
                            color=rcolor,
                            alpha=float(args.replica_alpha),
                            lw=float(args.replica_lw),
                            label="thin replicas" if not did_replica_label else None,
                            zorder=1,
                        )
                        did_replica_label = True

                if central is not None:
                    cg = values_for(central, pid=int(pid), flavor=str(flavor), Q=float(Q), x=float(x), quantity=quantity)
                    if not cg.empty:
                        ccolor = "black" if args.central_style == "black" else color
                        ax.plot(
                            pd.to_numeric(cg["bT"], errors="coerce").to_numpy(float),
                            pd.to_numeric(cg[quantity], errors="coerce").to_numpy(float),
                            color=ccolor,
                            ls="--",
                            lw=float(args.central_lw),
                            alpha=float(args.central_alpha),
                            label=args.central_label if not did_central_label else None,
                            zorder=5,
                        )
                        did_central_label = True

            if curves_written == 0:
                plt.close(fig)
                continue

            q_title = QUANTITY_TITLES.get(quantity, quantity)
            ax.set_title(f"{q_title}: {flavor}, Q={float(Q):g} GeV")
            ax.set_xlabel(r"$b_T\,[{\rm GeV}^{-1}]$")
            ax.set_ylabel(QUANTITY_LABELS.get(quantity, quantity))
            if args.y_log:
                ax.set_yscale("log")
            ax.grid(True, alpha=0.28)
            ax.legend(loc=args.legend_loc, ncol=int(args.legend_ncol), fontsize=float(args.legend_fontsize))
            if args.note:
                ax.text(
                    0.02,
                    0.02,
                    args.note,
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=float(args.note_fontsize),
                    alpha=0.8,
                )
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            pages_written += 1

    return {
        "quantity": quantity,
        "pdf": str(out_pdf),
        "pages_written": int(pages_written),
        "median_curves_written": int(curves_written),
        "n_thin_replica_keys": int(len(replica_keys)),
        "thin_replica_keys": replica_keys,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--band-dir", required=True)
    ap.add_argument("--bands-csv", default=None)
    ap.add_argument("--long-csv", default=None)
    ap.add_argument("--central-grid", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument(
        "--quantities",
        nargs="+",
        default=["F_NP", "ftilde", "b_ftilde", "b_x_ftilde"],
        choices=list(QUANTITY_LABELS.keys()),
    )
    ap.add_argument("--x-values", nargs="*", default=None)
    ap.add_argument("--Q-values", nargs="*", default=None)
    ap.add_argument("--pids", nargs="*", default=None)
    ap.add_argument("--flavors", nargs="*", default=None)
    ap.add_argument("--b-max", type=float, default=None)
    ap.add_argument("--replica-thin", type=int, default=10)
    ap.add_argument("--random-seed", type=int, default=303)
    ap.add_argument("--replica-style", choices=["same-color", "gray"], default="same-color")
    ap.add_argument("--central-style", choices=["same-color", "black"], default="same-color")
    ap.add_argument("--band-label", default="68% data⊗PDF")
    ap.add_argument("--central-label", default="central fit (PDF0)")
    ap.add_argument("--note", default="")
    ap.add_argument("--replace-original-names", action="store_true")
    ap.add_argument("--y-log", action="store_true")
    ap.add_argument("--legend-loc", default="best")
    ap.add_argument("--legend-ncol", type=int, default=2)
    ap.add_argument("--legend-fontsize", type=float, default=9.0)
    ap.add_argument("--median-lw", type=float, default=2.0)
    ap.add_argument("--central-lw", type=float, default=1.7)
    ap.add_argument("--central-alpha", type=float, default=0.95)
    ap.add_argument("--replica-lw", type=float, default=0.55)
    ap.add_argument("--replica-alpha", type=float, default=0.18)
    ap.add_argument("--band-alpha", type=float, default=0.22)
    ap.add_argument("--note-fontsize", type=float, default=9.0)
    args = ap.parse_args()

    band_dir = Path(args.band_dir)
    bands_csv = Path(args.bands_csv) if args.bands_csv else band_dir / "v23a_dataPDF_tmd_replica_bspace_bands.csv"
    long_csv = Path(args.long_csv) if args.long_csv else band_dir / "v23a_dataPDF_tmd_replica_bspace_long.csv"
    if not bands_csv.exists():
        raise SystemExit(f"Missing bands CSV: {bands_csv}")
    if not long_csv.exists():
        raise SystemExit(f"Missing long replica CSV: {long_csv}")

    out_dir = Path(args.out_dir) if args.out_dir else band_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    x_values = parse_float_tokens(args.x_values)
    Q_values = parse_float_tokens(args.Q_values)
    pids = parse_int_tokens(args.pids)
    flavors = args.flavors if args.flavors else None

    bands = pd.read_csv(bands_csv)
    long = pd.read_csv(long_csv)
    bands = add_derived_band_quantities(bands)
    long = add_derived_quantities(long)
    long = replica_key_frame(long)

    bands = filter_table(
        bands,
        x_values=x_values,
        Q_values=Q_values,
        pids=pids,
        flavors=flavors,
        b_max=args.b_max,
    )
    long = filter_table(
        long,
        x_values=x_values,
        Q_values=Q_values,
        pids=pids,
        flavors=flavors,
        b_max=args.b_max,
    )

    central = None
    if args.central_grid:
        central_path = Path(args.central_grid)
        if not central_path.exists():
            raise SystemExit(f"Missing central grid: {central_path}")
        central = pd.read_csv(central_path)
        central = add_derived_quantities(central)
        central = filter_table(
            central,
            x_values=x_values,
            Q_values=Q_values,
            pids=pids,
            flavors=flavors,
            b_max=args.b_max,
        )

    outputs = []
    for q in args.quantities:
        suffix = "" if args.replace_original_names else "_improved"
        pdf_path = out_dir / f"{q}_dataPDF_bands{suffix}.pdf"
        outputs.append(plot_quantity(q, bands, long, central, pdf_path, args))

    manifest = {
        "band_dir": str(band_dir),
        "bands_csv": str(bands_csv),
        "long_csv": str(long_csv),
        "central_grid": str(args.central_grid) if args.central_grid else None,
        "out_dir": str(out_dir),
        "quantities": args.quantities,
        "filters": {
            "x_values": x_values,
            "Q_values": Q_values,
            "pids": pids,
            "flavors": flavors,
            "b_max": args.b_max,
        },
        "plot_features": {
            "median": "solid curves",
            "band": args.band_label,
            "thin_replicas": int(args.replica_thin),
            "replica_style": args.replica_style,
            "central": bool(args.central_grid),
            "central_style": args.central_style,
            "central_label": args.central_label,
        },
        "outputs": outputs,
    }
    (out_dir / "improved_plot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print("\n=== Improved v23a data/PDF b-space plots ===")
    print(json.dumps({
        "out_dir": str(out_dir),
        "outputs": outputs,
        "manifest": str(out_dir / "improved_plot_manifest.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
