#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter

from v23.tools import plot_v23a_smooth_cross_section_panels as smooth


def parse_members(text: str) -> list[int]:
    out: list[int] = []
    for piece in str(text).replace(",", " ").split():
        m = re.fullmatch(r"(\d+)-(\d+)", piece)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if b < a:
                raise SystemExit(f"bad PDF member range: {piece}")
            out.extend(range(a, b + 1))
        else:
            out.append(int(piece))
    out = sorted(dict.fromkeys(out))
    if not out:
        raise SystemExit("no PDF members requested")
    return out


def load_backend_and_cfg(run: Path):
    metrics = json.loads((run / "metrics.json").read_text())
    config = metrics["config"]
    backend = smooth.import_from_path(Path(config["backend_script"]), "v23_prd_xsec_backend")
    cfg = smooth.backend_cfg_from_config(backend, config)
    return config, backend, cfg


def pdf_lumi_ratios(dense: pd.DataFrame, backend, cfg, pdf_set: str, members: list[int]) -> tuple[pd.DataFrame, dict]:
    rows = []
    central_pdf = backend.LHAPDFProvider(pdf_set, 0, use_toy_pdf=False)
    central = []
    for _, row in dense.iterrows():
        central.append(float(backend.charge_weighted_lumi(row, float(row["QM"]), central_pdf, cfg)))
    central = np.asarray(central, dtype=float)
    central = np.where(np.isfinite(central) & (central > 0.0), central, np.nan)

    ratio_matrix = []
    for member in members:
        pdf = backend.LHAPDFProvider(pdf_set, int(member), use_toy_pdf=False)
        vals = []
        for _, row in dense.iterrows():
            vals.append(float(backend.charge_weighted_lumi(row, float(row["QM"]), pdf, cfg)))
        vals = np.asarray(vals, dtype=float)
        ratio = vals / central
        ratio = np.where(np.isfinite(ratio) & (ratio > 0.0), ratio, np.nan)
        ratio_matrix.append(ratio)
        rows.append({
            "pdf_member": int(member),
            "n_finite": int(np.isfinite(ratio).sum()),
            "ratio_median": float(np.nanmedian(ratio)),
            "ratio_p16": float(np.nanquantile(ratio, 0.16)),
            "ratio_p84": float(np.nanquantile(ratio, 0.84)),
        })
        print(f"  PDF luminosity overlay: member {member}", flush=True)

    arr = np.vstack(ratio_matrix)
    pred = dense["pred_smooth_CS"].to_numpy(float)
    pdf_pred = arr * pred.reshape(1, -1)
    dense = dense.copy()
    dense["pred_pdf_q16"] = np.nanquantile(pdf_pred, 0.16, axis=0)
    dense["pred_pdf_q50"] = np.nanquantile(pdf_pred, 0.50, axis=0)
    dense["pred_pdf_q84"] = np.nanquantile(pdf_pred, 0.84, axis=0)

    tmd_lo = dense["pred_smooth_replica_q16"].to_numpy(float)
    tmd_hi = dense["pred_smooth_replica_q84"].to_numpy(float)
    pdf_lo = dense["pred_pdf_q16"].to_numpy(float)
    pdf_hi = dense["pred_pdf_q84"].to_numpy(float)
    c = pred
    down = np.sqrt(np.maximum(c - tmd_lo, 0.0) ** 2 + np.maximum(c - pdf_lo, 0.0) ** 2)
    up = np.sqrt(np.maximum(tmd_hi - c, 0.0) ** 2 + np.maximum(pdf_hi - c, 0.0) ** 2)
    dense["pred_total_q16"] = c - down
    dense["pred_total_q84"] = c + up

    meta = {
        "pdf_set": pdf_set,
        "pdf_members": members,
        "pdf_overlay_definition": (
            "PDF band from member variation of the charge-weighted Born luminosity "
            "at mu=Q, multiplying the smooth central cross-section curve. This is "
            "a fast PDF overlay, not a PDF-through-refit or full W(b) recomputation."
        ),
    }
    return dense, {"members": rows, **meta}


def make_groups(df: pd.DataFrame, datasets: list[str]) -> list[tuple[str, pd.DataFrame]]:
    groups = []
    for dataset in datasets:
        sub = df[df["dataset"].astype(str).eq(dataset)].copy()
        if sub.empty:
            continue
        if dataset.startswith(("E288", "E605", "E772")):
            for qm, g in sub.groupby("QM", observed=False, sort=True):
                groups.append((f"{dataset}, Q={float(qm):g}", g.sort_values("qT")))
        else:
            groups.append((dataset, sub.sort_values("qT")))
    return groups


def label_for_group(g: pd.DataFrame) -> str:
    row = g.iloc[0]
    bits = [str(row["dataset"])]
    if np.isfinite(float(row.get("QM", np.nan))):
        bits.append(rf"$Q={float(row['QM']):g}$")
    if "QM_Low" in row and np.isfinite(float(row.get("QM_Low", np.nan))) and np.isfinite(float(row.get("QM_High", np.nan))):
        bits.append(rf"$[{float(row['QM_Low']):g},{float(row['QM_High']):g}]$")
    if "xF" in row and np.isfinite(float(row.get("xF", np.nan))):
        bits.append(rf"$x_F={float(row['xF']):g}$")
    elif "y_Low" in row and np.isfinite(float(row.get("y_Low", np.nan))) and np.isfinite(float(row.get("y_High", np.nan))):
        bits.append(rf"$y\in[{float(row['y_Low']):g},{float(row['y_High']):g}]$")
    return "\n".join(bits)


def plot_canvas(central: pd.DataFrame, dense: pd.DataFrame, datasets: list[str], out: Path, *, max_cols: int) -> dict:
    groups = make_groups(central, datasets)
    n = len(groups)
    ncols = min(max_cols, 5)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.15 * ncols, 2.45 * nrows), dpi=220, squeeze=False)

    for ax, (_, g) in zip(axes.ravel(), groups):
        dataset = str(g["dataset"].iloc[0])
        dsub = dense[dense["dataset"].astype(str).eq(dataset)].copy()
        if dataset.startswith(("E288", "E605", "E772")):
            dsub = dsub[np.isclose(dsub["QM"].astype(float), float(g["QM"].iloc[0]))]
        dsub = dsub.sort_values("qT")
        g = g.sort_values("qT")

        ax.fill_between(
            dsub["qT"].to_numpy(float),
            dsub["pred_total_q16"].to_numpy(float),
            dsub["pred_total_q84"].to_numpy(float),
            color="#2b7bbb",
            alpha=0.20,
            linewidth=0,
            zorder=1,
        )
        ax.plot(dsub["qT"], dsub["pred_smooth_CS"], color="#0f6aa8", lw=1.45, zorder=2)
        ax.errorbar(
            g["qT"],
            g["CS"],
            yerr=g["sigma_used"],
            fmt="o",
            ms=2.4,
            mfc="white",
            mec="black",
            mew=0.7,
            ecolor="black",
            elinewidth=0.65,
            capsize=1.1,
            zorder=3,
        )
        vals = pd.concat([g["CS"], dsub["pred_total_q16"], dsub["pred_total_q84"]], ignore_index=True)
        vals = pd.to_numeric(vals, errors="coerce")
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if len(vals) and vals.max() / max(vals.min(), 1.0e-300) > 60:
            ax.set_yscale("log")
            ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
            ax.yaxis.set_minor_formatter(NullFormatter())
        else:
            ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which="major", direction="in", top=True, right=True, length=3.6, width=0.8, labelsize=6.7)
        ax.tick_params(which="minor", direction="in", top=True, right=True, length=2.0, width=0.55)
        ax.text(0.06, 0.92, label_for_group(g), transform=ax.transAxes, va="top", ha="left", fontsize=6.7)

    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax in axes[-1, :]:
        if ax.has_data():
            ax.set_xlabel(r"$q_T$ [GeV]", fontsize=8.5)
    for ax in axes[:, 0]:
        if ax.has_data():
            ax.set_ylabel(r"$d\sigma$ (table units)", fontsize=8.5)

    handles = [
        Line2D([0], [0], color="#0f6aa8", lw=1.8, label="central fit"),
        Line2D([0], [0], color="#2b7bbb", lw=7, alpha=0.20, label="68% TMD+PDF"),
        Line2D([0], [0], color="black", marker="o", linestyle="None", mfc="white", label="data"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.975), h_pad=0.9, w_pad=0.7)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return {"n_panels": n, "png": str(out.with_suffix(".png")), "pdf": str(out.with_suffix(".pdf"))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--central-predictions", required=True)
    ap.add_argument("--smooth-predictions", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--pdf-members", default="1-50")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--max-cols", type=int, default=5)
    args = ap.parse_args()

    run = Path(args.run)
    config, backend, cfg = load_backend_and_cfg(run)
    central = pd.read_csv(args.central_predictions)
    dense = pd.read_csv(args.smooth_predictions)
    members = parse_members(args.pdf_members)
    dense, pdf_meta = pdf_lumi_ratios(
        dense,
        backend,
        cfg,
        str(config.get("pdf_set", "NNPDF40_nnlo_as_01180")),
        members,
    )

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    dense.to_csv(out_prefix.with_name(out_prefix.name + "_predictions_with_pdf.csv"), index=False)
    canvas = plot_canvas(central, dense, [str(d) for d in args.datasets], out_prefix, max_cols=int(args.max_cols))
    rel = {}
    for ds, g in dense.groupby("dataset"):
        c = g["pred_smooth_CS"].to_numpy(float)
        hw = (g["pred_total_q84"].to_numpy(float) - g["pred_total_q16"].to_numpy(float)) / (2 * np.maximum(np.abs(c), 1.0e-300))
        rel[str(ds)] = {
            "median_total_rel_halfwidth": float(np.nanmedian(hw)),
            "max_total_rel_halfwidth": float(np.nanmax(hw)),
        }
    manifest = {
        "run": str(run),
        "smooth_predictions": str(args.smooth_predictions),
        "canvas": canvas,
        "pdf": pdf_meta,
        "relative_band_summary": rel,
        "warning": (
            "PDF component is a luminosity-reweight overlay at mu=Q. It is suitable "
            "for a fast complete-looking PRD comparison plot, but not a substitute "
            "for true PDF-member dense W(b) recomputation and refit."
        ),
    }
    out_prefix.with_name(out_prefix.name + "_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
