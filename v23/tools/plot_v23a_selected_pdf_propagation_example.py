#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator

from v23.tools import plot_v23a_smooth_cross_section_panels as smooth


def parse_panels(items: list[str]) -> list[tuple[str, float]]:
    panels = []
    for item in items:
        if ":" not in item:
            raise SystemExit(f"panel selector must be DATASET:Q, got {item!r}")
        ds, q = item.split(":", 1)
        panels.append((ds, float(q)))
    return panels


def load_backend(run: Path):
    metrics = json.loads((run / "metrics.json").read_text())
    config = metrics["config"]
    backend = smooth.import_from_path(Path(config["backend_script"]), "v23_selected_pdf_example_backend")
    cfg = smooth.backend_cfg_from_config(backend, config)
    return config, backend, cfg


def born_lumi_audit(central: pd.DataFrame, panels: list[tuple[str, float]], backend, cfg, pdf_set: str, members: list[int]) -> pd.DataFrame:
    records = []
    for ds, q in panels:
        g = central[central["dataset"].astype(str).eq(ds) & np.isclose(pd.to_numeric(central["QM"], errors="coerce"), q)]
        if g.empty:
            raise SystemExit(f"central predictions missing selected panel {ds}:{q}")
        row = g.sort_values("qT").iloc[0]
        central_pdf = backend.LHAPDFProvider(pdf_set, 0, use_toy_pdf=False)
        central_lumi = float(backend.charge_weighted_lumi(row, float(row["QM"]), central_pdf, cfg))
        ratios = []
        for member in members:
            pdf = backend.LHAPDFProvider(pdf_set, int(member), use_toy_pdf=False)
            lumi = float(backend.charge_weighted_lumi(row, float(row["QM"]), pdf, cfg))
            ratios.append(lumi / max(central_lumi, 1.0e-300))
        ratios = np.asarray(ratios, dtype=float)
        records.append(
            {
                "dataset": ds,
                "QM": float(q),
                "x1": float(row["x1"]),
                "x2": float(row["x2"]),
                "born_lumi_ratio_q16": float(np.nanquantile(ratios, 0.16)),
                "born_lumi_ratio_q50": float(np.nanquantile(ratios, 0.50)),
                "born_lumi_ratio_q84": float(np.nanquantile(ratios, 0.84)),
                "born_lumi_rel_halfwidth": float((np.nanquantile(ratios, 0.84) - np.nanquantile(ratios, 0.16)) / 2.0),
                "born_lumi_ratio_min": float(np.nanmin(ratios)),
                "born_lumi_ratio_max": float(np.nanmax(ratios)),
            }
        )
    return pd.DataFrame(records)


def selected_truew_audit(dense: pd.DataFrame, panels: list[tuple[str, float]]) -> pd.DataFrame:
    records = []
    for ds, q in panels:
        g = dense[dense["dataset"].astype(str).eq(ds) & np.isclose(pd.to_numeric(dense["QM"], errors="coerce"), q)].copy()
        if g.empty:
            raise SystemExit(f"true-W predictions missing selected panel {ds}:{q}")
        c = g["pred_smooth_CS"].to_numpy(float)
        pdf_hw = (g["pred_pdf_truew_q84"].to_numpy(float) - g["pred_pdf_truew_q16"].to_numpy(float)) / (2.0 * np.maximum(np.abs(c), 1.0e-300))
        exp_hw = (g["pred_smooth_replica_q84"].to_numpy(float) - g["pred_smooth_replica_q16"].to_numpy(float)) / (2.0 * np.maximum(np.abs(c), 1.0e-300))
        records.append(
            {
                "dataset": ds,
                "QM": float(q),
                "truew_pdf_rel_halfwidth_median": float(np.nanmedian(pdf_hw)),
                "truew_pdf_rel_halfwidth_max": float(np.nanmax(pdf_hw)),
                "experimental_rel_halfwidth_median": float(np.nanmedian(exp_hw)),
                "experimental_rel_halfwidth_max": float(np.nanmax(exp_hw)),
            }
        )
    return pd.DataFrame(records)


def label_for_panel(ds: str, q: float, central: pd.DataFrame) -> str:
    g = central[central["dataset"].astype(str).eq(ds) & np.isclose(pd.to_numeric(central["QM"], errors="coerce"), q)]
    row = g.iloc[0]
    labels = {
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
    bits = [labels.get(str(ds), str(ds)), rf"$Q={float(q):g}$"]
    if np.isfinite(float(row.get("xF", np.nan))):
        bits.append(rf"$x_F={float(row['xF']):g}$")
    return "\n".join(bits)


def plot_example(central: pd.DataFrame, dense: pd.DataFrame, panels: list[tuple[str, float]], out: Path) -> dict:
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.8 * n, 4.05), dpi=340, squeeze=False)
    for ax, (ds, q) in zip(axes.ravel(), panels):
        g = central[central["dataset"].astype(str).eq(ds) & np.isclose(pd.to_numeric(central["QM"], errors="coerce"), q)].sort_values("qT")
        dsub = dense[dense["dataset"].astype(str).eq(ds) & np.isclose(pd.to_numeric(dense["QM"], errors="coerce"), q)].sort_values("qT")
        ax.fill_between(
            dsub["qT"].to_numpy(float),
            dsub["pred_pdf_truew_q16"].to_numpy(float),
            dsub["pred_pdf_truew_q84"].to_numpy(float),
            color="#d95f02",
            alpha=0.18,
            linewidth=0,
            zorder=1,
        )
        ax.fill_between(
            dsub["qT"].to_numpy(float),
            dsub["pred_smooth_replica_q16"].to_numpy(float),
            dsub["pred_smooth_replica_q84"].to_numpy(float),
            color="#2b7bbb",
            alpha=0.24,
            linewidth=0,
            zorder=2,
        )
        ax.plot(dsub["qT"], dsub["pred_smooth_CS"], color="#0f6aa8", lw=2.25, zorder=3)
        ax.errorbar(
            g["qT"],
            g["CS"],
            yerr=g["sigma_used"],
            fmt="o",
            ms=3.8,
            mfc="white",
            mec="black",
            mew=0.9,
            ecolor="black",
            elinewidth=0.9,
            capsize=1.8,
            zorder=4,
        )
        ax.text(0.055, 0.93, label_for_panel(ds, q, central), transform=ax.transAxes, va="top", ha="left", fontsize=12.0, linespacing=0.95)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which="major", direction="in", top=True, right=True, length=5.0, width=1.0, labelsize=12.0, pad=3.0)
        ax.tick_params(which="minor", direction="in", top=True, right=True, length=2.7, width=0.75)
        ax.set_xlabel(r"$q_T$ [GeV]", fontsize=15)
    axes[0, 0].set_ylabel(r"$d\sigma/dq_T$", fontsize=15)
    handles = [
        Line2D([0], [0], color="#0f6aa8", lw=2.25, label="central fit"),
        Line2D([0], [0], color="#2b7bbb", lw=8.0, alpha=0.24, label="68% experimental replicas"),
        Line2D([0], [0], color="#d95f02", lw=8.0, alpha=0.18, label="68% PDF through W"),
        Line2D([0], [0], color="black", marker="o", linestyle="None", mfc="white", ms=4.4, label="data"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=4, frameon=False, fontsize=12.0, handlelength=1.8, columnspacing=1.1)
    fig.tight_layout(rect=(0, 0, 1, 0.875), w_pad=1.0)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", pad_inches=0.035)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    return {"png": str(out.with_suffix(".png")), "pdf": str(out.with_suffix(".pdf")), "n_panels": int(n)}


def parse_members(text: str) -> list[int]:
    members = []
    for piece in str(text).replace(",", " ").split():
        m = re.fullmatch(r"(\d+)-(\d+)", piece)
        if m:
            members.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            members.append(int(piece))
    return sorted(dict.fromkeys(members))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--central-predictions", required=True)
    ap.add_argument("--truew-predictions", required=True)
    ap.add_argument("--panels", nargs="+", required=True)
    ap.add_argument("--pdf-members", default="1-50")
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.linewidth": 1.0,
        "savefig.dpi": 300,
    })

    run = Path(args.run)
    central = pd.read_csv(args.central_predictions)
    truew = pd.read_csv(args.truew_predictions)
    panels = parse_panels([str(p) for p in args.panels])
    members = parse_members(str(args.pdf_members))
    config, backend, cfg = load_backend(run)
    pdf_set = str(config.get("pdf_set", "NNPDF40_nnlo_as_01180"))

    born = born_lumi_audit(central, panels, backend, cfg, pdf_set, members)
    truew_audit = selected_truew_audit(truew, panels)
    audit = pd.merge(born, truew_audit, on=["dataset", "QM"], how="inner")

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out_prefix.with_name(out_prefix.name + "_audit.csv"), index=False)
    plot = plot_example(central, truew, panels, out_prefix)
    manifest = {
        "run": str(run),
        "central_predictions": str(args.central_predictions),
        "truew_predictions": str(args.truew_predictions),
        "pdf_set": pdf_set,
        "pdf_members": members,
        "panels": [{"dataset": ds, "QM": q} for ds, q in panels],
        "plot": plot,
        "audit_csv": str(out_prefix.with_name(out_prefix.name + "_audit.csv")),
        "definition": (
            "Selected-panel example only. Blue band is experimental-data replica propagation. "
            "Orange band is collinear-PDF propagation from recomputing dense W(b) kernels for "
            "the listed PDF members at fixed central fitted F_NP and central dataset normalizations."
        ),
    }
    out_prefix.with_name(out_prefix.name + "_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
