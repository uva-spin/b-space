#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter
from scipy.interpolate import PchipInterpolator

from v23.tools.plot_v23a_prd_all_cross_sections import label_for_group, make_groups


COLLIDER_DATASETS = {"CDF_RUN_1", "CDF_RUN_2", "D0_RUN_1", "LHCb_7"}


def compact_label(g: pd.DataFrame) -> str:
    text = label_for_group(g)
    replacements = {
        "E288_200": "E288, 200 GeV",
        "E288_300": "E288, 300 GeV",
        "E288_400": "E288, 400 GeV",
        "E605": "E605",
        "E772": "E772",
        "CDF_RUN_1": "CDF Run I",
        "CDF_RUN_2": "CDF Run II",
        "D0_RUN_1": "D0 Run I",
        "LHCb_7": "LHCb 7 TeV",
    }
    for raw, pretty in replacements.items():
        text = text.replace(raw, pretty)
    return text


def load_row_replica_bands(path: str | None) -> pd.DataFrame | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Missing --replica-predictions file: {p}")
    rep = pd.read_csv(p)
    required = {"row_id", "pred_match_CS_replica"}
    missing = required.difference(rep.columns)
    if missing:
        raise SystemExit(f"{p} missing required columns: {sorted(missing)}")
    bands = (
        rep.groupby("row_id", observed=False)["pred_match_CS_replica"]
        .agg(
            pred_row_replica_q16=lambda s: float(np.nanquantile(s, 0.16)),
            pred_row_replica_q50="median",
            pred_row_replica_q84=lambda s: float(np.nanquantile(s, 0.84)),
            n_row_replicas="count",
        )
        .reset_index()
    )
    return bands


def smooth_through_rows(x: np.ndarray, y: np.ndarray, n: int = 220) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = y[good]
    if len(x) < 2:
        return x, y
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    uniq, idx = np.unique(x, return_index=True)
    x = uniq
    y = y[idx]
    if len(x) < 4:
        return x, y
    xx = np.linspace(float(x.min()), float(x.max()), int(n))
    yy = PchipInterpolator(x, y, extrapolate=False)(xx)
    return xx, yy


def plot_canvas(
    central: pd.DataFrame,
    dense: pd.DataFrame,
    row_bands: pd.DataFrame | None,
    datasets: list[str],
    out: Path,
    *,
    max_cols: int,
) -> dict:
    groups = make_groups(central, datasets)
    n = len(groups)
    ncols = min(int(max_cols), 5)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.25, 9.65), dpi=420, squeeze=False)
    data_errorbar_column = "sigma_used"

    for ax, (_, g) in zip(axes.ravel(), groups):
        dataset = str(g["dataset"].iloc[0])
        g = g.sort_values("qT").copy()
        if data_errorbar_column not in g.columns:
            g[data_errorbar_column] = g["sigma_used"]

        if dataset in COLLIDER_DATASETS:
            if row_bands is not None:
                g = g.merge(row_bands, on="row_id", how="left")
            for col in ["pred_row_replica_q16", "pred_row_replica_q84"]:
                if col not in g.columns:
                    g[col] = g["pred_match_CS"]
                g[col] = pd.to_numeric(g[col], errors="coerce").fillna(pd.to_numeric(g["pred_match_CS"], errors="coerce"))

            x = pd.to_numeric(g["qT"], errors="coerce").to_numpy(float)
            y = pd.to_numeric(g["pred_match_CS"], errors="coerce").to_numpy(float)
            lo = pd.to_numeric(g["pred_row_replica_q16"], errors="coerce").to_numpy(float)
            hi = pd.to_numeric(g["pred_row_replica_q84"], errors="coerce").to_numpy(float)
            xx, yy = smooth_through_rows(x, y)
            xxlo, yylo = smooth_through_rows(x, lo)
            xxhi, yyhi = smooth_through_rows(x, hi)
            if len(xxlo) == len(xxhi) and len(xxlo) > 1 and np.allclose(xxlo, xxhi):
                ax.fill_between(xxlo, yylo, yyhi, color="#2b7bbb", alpha=0.20, linewidth=0, zorder=1)
            else:
                ax.fill_between(x, lo, hi, color="#2b7bbb", alpha=0.20, linewidth=0, zorder=1)
            ax.plot(xx, yy, color="#0f6aa8", lw=1.15, zorder=2)
            ax.plot(x, y, "s", ms=1.5, mfc="#0f6aa8", mec="#0f6aa8", mew=0.0, zorder=2.5)
            scale_vals = pd.Series(np.concatenate([g["CS"].to_numpy(float), lo, hi, y]))
        else:
            dsub = dense[dense["dataset"].astype(str).eq(dataset)].copy()
            dsub = dsub[np.isclose(dsub["QM"].astype(float), float(g["QM"].iloc[0]))]
            dsub = dsub.sort_values("qT")
            ax.fill_between(
                dsub["qT"].to_numpy(float),
                dsub["pred_smooth_replica_q16"].to_numpy(float),
                dsub["pred_smooth_replica_q84"].to_numpy(float),
                color="#2b7bbb",
                alpha=0.22,
                linewidth=0,
                zorder=1,
            )
            ax.plot(dsub["qT"], dsub["pred_smooth_CS"], color="#0f6aa8", lw=1.15, zorder=2)
            scale_vals = pd.concat(
                [g["CS"], dsub["pred_smooth_replica_q16"], dsub["pred_smooth_replica_q84"]],
                ignore_index=True,
            )

        ax.errorbar(
            g["qT"],
            g["CS"],
            yerr=g[data_errorbar_column],
            fmt="o",
            ms=1.85,
            mfc="white",
            mec="black",
            mew=0.55,
            ecolor="black",
            elinewidth=0.50,
            capsize=0.9,
            zorder=3,
        )
        vals = pd.to_numeric(scale_vals, errors="coerce")
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if len(vals) and vals.max() / max(vals.min(), 1.0e-300) > 60:
            ax.set_yscale("log")
            ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
            ax.yaxis.set_minor_formatter(NullFormatter())
        else:
            ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which="major", direction="in", top=True, right=True, length=2.8, width=0.70, labelsize=5.7, pad=1.2)
        ax.tick_params(which="minor", direction="in", top=True, right=True, length=1.5, width=0.45)
        ax.set_title(compact_label(g), loc="left", fontsize=5.6, pad=1.2, linespacing=0.88)

    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax in axes[-1, :]:
        if ax.has_data():
            ax.set_xlabel(r"$q_T$ [GeV]", fontsize=7.1, labelpad=1.8)
    for ax in axes[:, 0]:
        if ax.has_data():
            ax.set_ylabel(r"$d\sigma/dq_T$", fontsize=7.1, labelpad=1.8)

    handles = [
        Line2D([0], [0], color="#0f6aa8", lw=1.4, label="central fit"),
        Line2D([0], [0], color="#2b7bbb", lw=5.0, alpha=0.22, label="68% experimental replicas"),
        Line2D([0], [0], color="black", marker="o", linestyle="None", mfc="white", ms=3.0, label="data"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.997), ncol=3, frameon=False, fontsize=7.3, handlelength=1.8, columnspacing=1.0)
    fig.tight_layout(rect=(0.004, 0.0, 0.996, 0.975), h_pad=1.30, w_pad=0.50)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", pad_inches=0.025)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    return {
        "n_panels": n,
        "png": str(out.with_suffix(".png")),
        "pdf": str(out.with_suffix(".pdf")),
        "data_errorbar_column": data_errorbar_column,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--central-predictions", required=True)
    ap.add_argument("--smooth-predictions", required=True)
    ap.add_argument("--replica-predictions", default="")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--max-cols", type=int, default=5)
    args = ap.parse_args()

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.linewidth": 1.0,
        "savefig.dpi": 300,
    })

    central = pd.read_csv(args.central_predictions)
    dense = pd.read_csv(args.smooth_predictions)
    row_bands = load_row_replica_bands(args.replica_predictions)
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    dense.to_csv(out_prefix.with_name(out_prefix.name + "_predictions_exponly.csv"), index=False)
    canvas = plot_canvas(central, dense, row_bands, [str(d) for d in args.datasets], out_prefix, max_cols=int(args.max_cols))

    rel = {}
    for ds, g in dense.groupby("dataset"):
        c = g["pred_smooth_CS"].to_numpy(float)
        hw = (
            g["pred_smooth_replica_q84"].to_numpy(float)
            - g["pred_smooth_replica_q16"].to_numpy(float)
        ) / (2 * np.maximum(np.abs(c), 1.0e-300))
        rel[str(ds)] = {
            "median_experimental_replica_rel_halfwidth": float(np.nanmedian(hw)),
            "max_experimental_replica_rel_halfwidth": float(np.nanmax(hw)),
        }

    manifest = {
        "run": str(args.run),
        "central_predictions": str(args.central_predictions),
        "smooth_predictions": str(args.smooth_predictions),
        "replica_predictions": str(args.replica_predictions),
        "canvas": canvas,
        "relative_band_summary": rel,
        "definition": (
            "Single-canvas cross-section comparison. Fixed-target panels use smooth dense "
            "W(b) evaluations. Collider panels use the fitted row/bin convention, with a "
            "PCHIP visual guide through the validated row-bin predictions because the dense "
            "collider fiducial evaluator is not the production observable. Bands are q16/q84 "
            "experimental-data replica propagation; no collinear-PDF uncertainty is included. "
            "Point error bars use the same effective point-uncertainty column used by the accepted fit."
        ),
    }
    out_prefix.with_name(out_prefix.name + "_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
