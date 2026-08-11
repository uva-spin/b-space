#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter


def read_replicas(pattern: str) -> pd.DataFrame:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"No replica prediction files matched: {pattern}")

    frames = []
    for path in paths:
        df = pd.read_csv(path)
        if "row_id" not in df.columns or "pred_match_CS" not in df.columns:
            raise SystemExit(f"{path} is missing row_id or pred_match_CS")
        seed = Path(path).parent.name
        frames.append(
            pd.DataFrame({
                "seed": seed,
                "row_id": df["row_id"].astype(str),
                "pred_match_CS": pd.to_numeric(df["pred_match_CS"], errors="coerce"),
            })
        )

    return pd.concat(frames, ignore_index=True)


def aggregate_replicas(rep: pd.DataFrame) -> pd.DataFrame:
    return (
        rep.groupby("row_id", observed=False)
        .agg(
            pred_replica_q16=("pred_match_CS", lambda s: float(np.nanquantile(s, 0.16))),
            pred_replica_q50=("pred_match_CS", "median"),
            pred_replica_q84=("pred_match_CS", lambda s: float(np.nanquantile(s, 0.84))),
            pred_replica_mean=("pred_match_CS", "mean"),
            pred_replica_std=("pred_match_CS", "std"),
            n_replicas=("pred_match_CS", "count"),
        )
        .reset_index()
    )


def panel_label(g: pd.DataFrame) -> str:
    row = g.iloc[0]
    qm = float(row["QM"])
    qlo = row.get("QM_Low", np.nan)
    qhi = row.get("QM_High", np.nan)
    bits = [rf"$Q_M={qm:g}$"]
    if np.isfinite(qlo) and np.isfinite(qhi):
        bits.append(rf"$[{float(qlo):g},{float(qhi):g}]$")
    if "xF" in g.columns and np.isfinite(row.get("xF", np.nan)):
        bits.append(rf"$x_F={float(row['xF']):g}$")
    elif "y" in g.columns and np.isfinite(row.get("y", np.nan)):
        bits.append(rf"$y={float(row['y']):g}$")
    return "\n".join(bits)


def make_groups(df: pd.DataFrame, dataset: str) -> list[tuple[str, pd.DataFrame]]:
    sub = df[df["dataset"].astype(str).eq(dataset)].copy()
    if sub.empty:
        return []

    if dataset.startswith(("E288", "E605", "E772")):
        key = "QM"
        groups = []
        for value, g in sub.groupby(key, observed=False, sort=True):
            groups.append((f"QM={float(value):g}", g.sort_values("qT")))
        return groups

    groups = []
    for value, g in sub.groupby("dataset", observed=False, sort=True):
        groups.append((str(value), g.sort_values("qT")))
    return groups


def yscale_for(groups: list[tuple[str, pd.DataFrame]], requested: str) -> str:
    if requested != "auto":
        return requested
    vals = []
    for _, g in groups:
        for col in ["CS", "pred_replica_q16", "pred_replica_q84"]:
            vals.extend(pd.to_numeric(g[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().tolist())
    vals = np.asarray([v for v in vals if v > 0.0], dtype=float)
    if vals.size == 0:
        return "linear"
    return "log" if float(np.nanmax(vals) / np.nanmin(vals)) > 80.0 else "linear"


def balanced_grid(n: int, max_cols: int) -> tuple[int, int]:
    if n <= 1:
        return 1, 1
    ncols = min(int(max_cols), int(math.ceil(math.sqrt(n))))
    nrows = int(math.ceil(n / ncols))
    while ncols < int(max_cols) and nrows * ncols < n:
        ncols += 1
        nrows = int(math.ceil(n / ncols))
    return nrows, ncols


def plot_dataset(
    df: pd.DataFrame,
    dataset: str,
    out_pdf: Path,
    out_png: Path,
    *,
    yscale: str,
    max_cols: int,
) -> dict:
    groups = make_groups(df, dataset)
    if not groups:
        raise SystemExit(f"No rows for dataset {dataset}")

    scale = yscale_for(groups, yscale)
    n = len(groups)
    nrows, ncols = balanced_grid(n, int(max_cols))

    width = max(4.6, 3.35 * ncols)
    height = max(3.7, 2.8 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), dpi=180, squeeze=False)
    axes_flat = axes.ravel()

    for ax, (_, g) in zip(axes_flat, groups):
        g = g.sort_values("qT")
        x = pd.to_numeric(g["qT"], errors="coerce").to_numpy(float)
        data = pd.to_numeric(g["CS"], errors="coerce").to_numpy(float)
        sigma = pd.to_numeric(g["sigma_used"], errors="coerce").to_numpy(float)
        pred = pd.to_numeric(g["pred_match_CS"], errors="coerce").to_numpy(float)
        lo = pd.to_numeric(g["pred_replica_q16"], errors="coerce").to_numpy(float)
        hi = pd.to_numeric(g["pred_replica_q84"], errors="coerce").to_numpy(float)

        order = np.argsort(x)
        x, data, sigma, pred, lo, hi = [arr[order] for arr in [x, data, sigma, pred, lo, hi]]

        ax.fill_between(x, lo, hi, color="#1f77b4", alpha=0.22, linewidth=0, zorder=1)
        ax.plot(x, pred, color="#1f77b4", linewidth=1.9, zorder=2)
        ax.errorbar(
            x,
            data,
            yerr=sigma,
            fmt="o",
            ms=3.3,
            mfc="white",
            mec="black",
            mew=0.8,
            ecolor="black",
            elinewidth=0.8,
            capsize=1.7,
            zorder=3,
        )

        ax.text(0.06, 0.91, panel_label(g), transform=ax.transAxes, va="top", ha="left", fontsize=8.5)
        ax.set_yscale(scale)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        if scale == "log":
            ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
            ax.yaxis.set_minor_formatter(NullFormatter())
        else:
            ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which="major", direction="in", top=True, right=True, length=4.5, width=0.9, labelsize=8.5)
        ax.tick_params(which="minor", direction="in", top=True, right=True, length=2.5, width=0.7)

        ymin = np.nanmin(np.concatenate([data - sigma, lo]))
        ymax = np.nanmax(np.concatenate([data + sigma, hi]))
        if scale == "log":
            positives = np.concatenate([data[data > 0], lo[lo > 0], hi[hi > 0]])
            if positives.size:
                ax.set_ylim(0.75 * np.nanmin(positives), 1.35 * np.nanmax(positives))
        else:
            span = ymax - ymin
            pad = 0.12 * span if span > 0 else 0.1 * abs(ymax)
            ax.set_ylim(max(0.0, ymin - pad), ymax + pad)

    for ax in axes_flat[n:]:
        ax.axis("off")

    for ax in axes[-1, :]:
        if ax.has_data():
            ax.set_xlabel(r"$q_T$ [GeV]", fontsize=10.5)
    for ax in axes[:, 0]:
        if ax.has_data():
            ax.set_ylabel(r"$d\sigma$ (table units)", fontsize=10.5)

    handles = [
        Line2D([0], [0], color="#1f77b4", lw=1.9, label="central fit"),
        Line2D([0], [0], color="#1f77b4", lw=7, alpha=0.22, label="68% experimental replicas"),
        Line2D([0], [0], color="black", marker="o", linestyle="None", mfc="white", label="data"),
    ]
    fig.suptitle(dataset, x=0.02, y=0.99, ha="left", va="top", fontsize=13)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.56, 0.985), ncol=3, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    return {
        "dataset": dataset,
        "n_panels": n,
        "n_points": int(sum(len(g) for _, g in groups)),
        "yscale": scale,
        "pdf": str(out_pdf),
        "png": str(out_png),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--central-predictions", required=True)
    ap.add_argument("--replica-prediction-glob", required=True)
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--yscale", choices=["auto", "linear", "log"], default="auto")
    ap.add_argument("--max-cols", type=int, default=4)
    args = ap.parse_args()

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.linewidth": 1.0,
        "savefig.dpi": 300,
    })

    central = pd.read_csv(args.central_predictions)
    central["row_id"] = central["row_id"].astype(str)
    rep = read_replicas(args.replica_prediction_glob)
    bands = aggregate_replicas(rep)

    df = central.merge(bands, on="row_id", how="left", validate="one_to_one")
    missing = df["pred_replica_q16"].isna().sum()
    if missing:
        raise SystemExit(f"Missing replica prediction bands for {missing} central rows")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "v23a_production_cross_section_predictions_with_replica_bands.csv", index=False)

    datasets = args.datasets or list(dict.fromkeys(df["dataset"].astype(str).tolist()))
    summaries = []
    for dataset in datasets:
        pdf_path = out / f"{dataset}_cross_section_panels.pdf"
        png_path = out / f"{dataset}_cross_section_panels.png"
        summary = plot_dataset(
            df,
            dataset,
            pdf_path,
            png_path,
            yscale=args.yscale,
            max_cols=args.max_cols,
        )
        summaries.append(summary)

    summary = {
        "central_predictions": args.central_predictions,
        "replica_prediction_glob": args.replica_prediction_glob,
        "n_replica_prediction_files": int(rep["seed"].nunique()),
        "datasets": summaries,
        "band_definition": "q16/q50/q84 of pred_match_CS over fitted experimental pseudo-data replicas; central curve is the production central pred_match_CS; data errors are sigma_used.",
        "note": "PDF uncertainty is not included in these cross-section bands unless true per-PDF-member refits are supplied as replica prediction files.",
    }
    (out / "v23a_production_cross_section_panel_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame(summaries).to_csv(out / "v23a_production_cross_section_panel_summary.csv", index=False)

    print(json.dumps(summary, indent=2))
    print("wrote:", out)


if __name__ == "__main__":
    main()
