#!/usr/bin/env python3
"""Render PRD Fig. 5 with lambda=1 TMD/experimental and true-W PDF bands."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIG4_DENSE = HERE / "fig4_lambda1_dense_predictions.csv"
CENTRAL = ROOT / "systematics/collins_factorization_validity/outputs/rowidfix_stageFT_all_qmax0p20_lam0p50_central_s303/predictions.csv"
TRUEW_DIR = ROOT / "plots/prd_q020_figures/true_pdf_w_subset_highx_q020"
TRUEW = TRUEW_DIR / "v23a_true_pdf_w_predictions.csv"
PANELS = (("E288_200", 8.5), ("E288_400", 12.5), ("E605", 15.75))


def panel_label(dataset: str, q: float, central: pd.DataFrame) -> str:
    names = {"E288_200": "E288, 200 GeV", "E288_400": "E288, 400 GeV", "E605": "E605"}
    row = central[central.dataset.eq(dataset) & np.isclose(central.QM, q)].iloc[0]
    return names[dataset] + "\n" + rf"$Q={q:g}$" + "\n" + rf"$x_F={float(row.xF):g}$"


def main() -> None:
    dense = pd.read_csv(FIG4_DENSE, low_memory=False)
    old = pd.read_csv(TRUEW, low_memory=False)
    central = pd.read_csv(CENTRAL, low_memory=False)
    member_predictions = []
    for member in range(1, 51):
        with np.load(TRUEW_DIR / f"member_predictions/member_{member:04d}.npz") as saved:
            member_predictions.append(saved["pred"].astype(float))
    members = np.asarray(member_predictions)

    plot_rows = []
    offset = 0
    for dataset, q in PANELS:
        new = dense[dense.dataset.eq(dataset) & np.isclose(dense.QM, q)].sort_values("qT").copy()
        prior = old[old.dataset.eq(dataset) & np.isclose(old.QM, q)].sort_values("qT").copy()
        if len(new) != 100 or len(prior) != 100 or not np.allclose(new.qT, prior.qT):
            raise RuntimeError(f"grid mismatch for {dataset}:{q}")
        old_central = prior.pred_smooth_CS.to_numpy(float)
        ratios = members[:, offset:offset+100] / np.maximum(old_central[None, :], 1e-300)
        ratio_q16, ratio_q50, ratio_q84 = np.quantile(ratios, [.16, .50, .84], axis=0)
        median = new["median"].to_numpy(float)
        for i, (_, row) in enumerate(new.iterrows()):
            plot_rows.append({
                "dataset": dataset, "QM": q, "qT": float(row.qT),
                "combined_q16": float(row.q16), "combined_median": float(row["median"]),
                "combined_q84": float(row.q84),
                "pdf_q16": median[i]*ratio_q16[i],
                "pdf_q50": median[i]*ratio_q50[i],
                "pdf_q84": median[i]*ratio_q84[i],
            })
        offset += 100
    out = pd.DataFrame(plot_rows)
    out.to_csv(HERE / "fig5_lambda1_selected_pdf_comparison.csv", index=False)

    plt.rcParams.update({"font.family":"serif", "mathtext.fontset":"cm",
                         "font.size":14, "axes.linewidth":1.15,
                         "xtick.direction":"in", "ytick.direction":"in"})
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.35), dpi=320)
    for ax, (dataset, q) in zip(axes, PANELS):
        rows = out[out.dataset.eq(dataset) & np.isclose(out.QM, q)].sort_values("qT")
        data = central[central.dataset.eq(dataset) & np.isclose(central.QM, q)].sort_values("qT")
        ax.fill_between(rows.qT, rows.pdf_q16, rows.pdf_q84,
                        color="#d95f02", alpha=.20, lw=0, zorder=1)
        ax.fill_between(rows.qT, rows.combined_q16, rows.combined_q84,
                        color="#2b7bbb", alpha=.27, lw=0, zorder=2)
        ax.plot(rows.qT, rows.combined_median, color="#0f6aa8", lw=2.6, zorder=3)
        ax.errorbar(data.qT, data.CS, yerr=data.sigma_used, fmt="o", ms=4.2,
                    mfc="white", mec="black", mew=1.0, ecolor="black",
                    elinewidth=1.0, capsize=2.0, zorder=4)
        ax.text(.05,.94,panel_label(dataset,q,central),transform=ax.transAxes,
                va="top",ha="left",fontsize=14.5,linespacing=.95)
        ax.xaxis.set_minor_locator(AutoMinorLocator()); ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which="major",top=True,right=True,labelsize=14,length=6,width=1.1)
        ax.tick_params(which="minor",top=True,right=True,length=3.2,width=.85)
        ax.set_xlabel(r"$q_T\ [\mathrm{GeV}]$",fontsize=17)
    axes[0].set_ylabel(r"$d\sigma/dq_T$",fontsize=17)
    handles=[
        Line2D([0],[0],color="#0f6aa8",lw=2.6,label=r"$\lambda=1$ ensemble median"),
        Line2D([0],[0],color="#2b7bbb",lw=8,alpha=.27,label="combined TMD/experimental 68% interval"),
        Line2D([0],[0],color="#d95f02",lw=8,alpha=.20,label=r"true-$W$ PDF 68% interval"),
        Line2D([0],[0],color="black",marker="o",ls="None",mfc="white",ms=4.8,label="data"),
    ]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.5,1.015),
               ncol=4,frameon=False,fontsize=13,handlelength=1.9,columnspacing=1.1)
    fig.tight_layout(rect=(0,0,1,.88),w_pad=1.1)
    fig.savefig(HERE/"fig5_lambda1_selected_pdf_comparison.pdf",bbox_inches="tight",pad_inches=.04)
    fig.savefig(HERE/"fig5_lambda1_selected_pdf_comparison.png",dpi=300,bbox_inches="tight",pad_inches=.04)
    plt.close(fig)
    manifest={"status":"complete","panels":[{"dataset":d,"Q":q} for d,q in PANELS],
              "blue_band":"lambda=1 24-start x 50-experimental-replica combined central 68% interval",
              "orange_band":"central 68% relative true-W variation of NNPDF40 members 1-50, multiplicatively recentered on the lambda=1 ensemble median",
              "pdf_tmd_correlation_included":False,"production_sources_modified":False}
    (HERE/"fig5_lambda1_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")


if __name__ == "__main__": main()
