#!/usr/bin/env python3
"""Build PRD Fig. 4 for the lambda=1 24x50 operational ensemble (CPU only)."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CAMPAIGN = ROOT / "systematics/dataset_identifiability_campaign_2026"
SOURCE = ROOT / "systematics/collins_factorization_validity/outputs/rowidfix_stageFT_all_qmax0p20_lam0p50_central_s303"
SMOOTH_DIR = ROOT / "plots/prd_q020_figures/smooth_cross_sections_q020"
CACHE = SMOOTH_DIR / "v23a_smooth_cross_section_dense_kernel_cache.npz"
OLD_DENSE = SMOOTH_DIR / "v23a_smooth_cross_section_predictions.csv"
OLD_REPLICAS = ROOT / (
    "systematics/collins_factorization_validity/replicas/"
    "rowidfix_stageFT_all_qmax0p20_lam0p50_lambda3_50rep"
)
TRAINER = ROOT / "v21_smoothedA_tail_candidate/train_bt_dnn_v21_smoothedA_tail.py"
SMOOTH_SCRIPT = ROOT / "v23/tools/plot_v23a_smooth_cross_section_panels.py"
REGISTRY = CAMPAIGN / "summaries/champion_registry/empirical_reference_lambda1_b0p1_2p0_full24.json"
DATASETS = ("E288_200", "E288_300", "E288_400", "E605", "E772",
            "CDF_RUN_1", "CDF_RUN_2", "D0_RUN_1", "LHCb_7")
COLLIDERS = {"CDF_RUN_1", "CDF_RUN_2", "D0_RUN_1", "LHCb_7"}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def norms(run: Path, frame: pd.DataFrame, candidate: bool) -> np.ndarray:
    table = pd.read_csv(run / "dataset_norms.csv")
    column = "control_norm" if candidate else "norm_scale"
    mapping = dict(zip(table.dataset.astype(str), table[column].astype(float)))
    return frame.dataset.astype(str).map(mapping).fillna(1.0).to_numpy(float)


def pchip(x, y, n=220):
    x, y = np.asarray(x, float), np.asarray(y, float)
    order = np.argsort(x); x, y = x[order], y[order]
    x, first = np.unique(x, return_index=True); y = y[first]
    if len(x) < 4:
        return x, y
    xx = np.linspace(x.min(), x.max(), n)
    return xx, PchipInterpolator(x, y, extrapolate=False)(xx)


def groups(frame: pd.DataFrame):
    result = []
    for dataset in DATASETS:
        sub = frame[frame.dataset.astype(str).eq(dataset)]
        if dataset.startswith(("E288", "E605", "E772")):
            for _, group in sub.groupby("QM", sort=True):
                result.append(group.sort_values("qT"))
        else:
            result.append(sub.sort_values("qT"))
    return result


def label(group):
    row = group.iloc[0]; dataset = str(row.dataset)
    names = {"E288_200":"E288, 200 GeV", "E288_300":"E288, 300 GeV",
             "E288_400":"E288, 400 GeV", "E605":"E605", "E772":"E772",
             "CDF_RUN_1":"CDF Run I", "CDF_RUN_2":"CDF Run II",
             "D0_RUN_1":"D0 Run I", "LHCb_7":"LHCb 7 TeV"}
    if dataset in COLLIDERS:
        return names[dataset] + "\n" + rf"$Q={float(row.QM):g}$"
    text = names[dataset] + "\n" + rf"$Q={float(row.QM):g}$"
    if np.isfinite(float(row.get("QM_Low", np.nan))):
        text += "\n" + rf"$[{float(row.QM_Low):g},{float(row.QM_High):g}]$"
    if np.isfinite(float(row.get("xF", np.nan))):
        text += "\n" + rf"$x_F={float(row.xF):g}$"
    return text


def render(central, dense, row_bands):
    panel_groups = groups(central)
    fig, axes = plt.subplots(7, 5, figsize=(8.2, 10.8), dpi=400, squeeze=False)
    for ax, group in zip(axes.ravel(), panel_groups):
        dataset = str(group.dataset.iloc[0]); group = group.copy()
        if dataset in COLLIDERS:
            rb = row_bands[row_bands.dataset.eq(dataset)]
            group = group.merge(rb[["row_id","q16","median","q84"]], on="row_id", how="left")
            x = group.qT.to_numpy(float)
            xx, yy = pchip(x, group["median"])
            xl, yl = pchip(x, group.q16); xh, yh = pchip(x, group.q84)
            ax.fill_between(xl, yl, yh, color="#2b7bbb", alpha=.22, lw=0)
            ax.plot(xx, yy, color="#0f6aa8", lw=1.45)
            scale = np.r_[group.CS, group.q16, group.q84]
        else:
            sub = dense[dense.dataset.eq(dataset) & np.isclose(dense.QM, float(group.QM.iloc[0]))].sort_values("qT")
            ax.fill_between(sub.qT, sub.q16, sub.q84, color="#2b7bbb", alpha=.22, lw=0)
            ax.plot(sub.qT, sub["median"], color="#0f6aa8", lw=1.45)
            scale = np.r_[group.CS, sub.q16, sub.q84]
        ax.errorbar(group.qT, group.CS, yerr=group.sigma_used, fmt="o", ms=2.2,
                    mfc="white", mec="black", mew=.65, ecolor="black",
                    elinewidth=.62, capsize=1.0, zorder=3)
        positive = scale[np.isfinite(scale) & (scale > 0)]
        if len(positive) and positive.max()/positive.min() > 60:
            ax.set_yscale("log"); ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2,10)*.1)); ax.yaxis.set_minor_formatter(NullFormatter())
        else:
            ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which="major", direction="in", top=True, right=True,
                       length=3.4, width=.8, labelsize=7.2, pad=1.4)
        ax.tick_params(which="minor", direction="in", top=True, right=True,
                       length=1.8, width=.55)
        ax.set_title(label(group), loc="left", fontsize=7.0, pad=1.5, linespacing=.9)
    for ax in axes.ravel()[len(panel_groups):]: ax.axis("off")
    for ax in axes[-1]:
        if ax.has_data(): ax.set_xlabel(r"$q_T$ [GeV]", fontsize=9.0, labelpad=2)
    for ax in axes[:,0]:
        if ax.has_data(): ax.set_ylabel(r"$d\sigma/dq_T$", fontsize=9.0, labelpad=2)
    handles = [
        Line2D([0],[0],color="#0f6aa8",lw=1.7,label=r"$\lambda=1$ ensemble median"),
        Line2D([0],[0],color="#2b7bbb",lw=6,alpha=.22,label=r"combined central 68% interval"),
        Line2D([0],[0],color="black",marker="o",ls="None",mfc="white",ms=3.4,label="data"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5,.998),
               ncol=3, frameon=False, fontsize=9.4, handlelength=2.0, columnspacing=1.2)
    fig.tight_layout(rect=(.005,0,.995,.975), h_pad=1.38, w_pad=.58)
    fig.savefig(HERE/"fig4_lambda1_all_cross_sections.pdf", bbox_inches="tight", pad_inches=.03)
    fig.savefig(HERE/"fig4_lambda1_all_cross_sections.png", dpi=320, bbox_inches="tight", pad_inches=.03)
    plt.close(fig)


def main():
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    smooth = load_module("fig4_smooth", SMOOTH_SCRIPT)
    trainer = load_module("fig4_trainer", TRAINER)
    config = json.loads((SOURCE/"metrics.json").read_text())["config"]
    central = pd.read_csv(SOURCE/"predictions.csv")
    dense = pd.read_csv(CACHE.with_suffix(".rows.csv"))
    arrays = np.load(CACHE); b, kernel = arrays["b_grid"], arrays["kernel"]
    device, dtype = torch.device("cpu"), torch.float32
    scale = pd.to_numeric(dense.smooth_prefactor_scale, errors="coerce").fillna(1).to_numpy(float)
    registry = json.loads(REGISTRY.read_text())

    start_dense=[]; start_rows=[]
    for i, tag in enumerate(registry["endpoint_tags"], 1):
        run=CAMPAIGN/"outputs"/tag
        raw=smooth.state_prediction(trainer,config,dense,b,kernel,run/"model_state.pt",device,dtype)
        start_dense.append(raw*norms(run,dense,True)*scale)
        p=pd.read_csv(run/"accepted_predictions.csv").set_index("row_id")
        start_rows.append(p.loc[central.row_id,"control_prediction"].to_numpy(float))
        print(f"start {i}/24",flush=True)
    start_dense=np.asarray(start_dense); start_rows=np.asarray(start_rows)

    old_central_norm=norms(SOURCE,dense,False)
    replica_dense=[]
    replica_runs=sorted((OLD_REPLICAS/"outputs").glob("*/model_state.pt"))
    for i,state in enumerate(replica_runs,1):
        raw=smooth.state_prediction(trainer,config,dense,b,kernel,state,device,dtype)
        replica_dense.append(raw*old_central_norm*scale)
        print(f"replica {i}/50",flush=True)
    replica_dense=np.asarray(replica_dense)
    old_dense=pd.read_csv(OLD_DENSE)
    old_rep_med=old_dense.pred_smooth_replica_q50.to_numpy(float)
    dense_cross=(start_dense[:,None,:]+(replica_dense-old_rep_med)[None,:,:]).reshape(1200,len(dense))
    dense[["q16","median","q84"]]=np.quantile(dense_cross,[.16,.5,.84],axis=0).T

    rep_rows=pd.read_csv(OLD_REPLICAS/"audit_basic/v22_replica_predictions_long.csv")
    wide=rep_rows.pivot(index="seed",columns="row_id",values="pred_match_CS_replica")
    rep_arr=wide.loc[:,central.row_id].to_numpy(float)
    rep_res=rep_arr-np.median(rep_arr,axis=0)
    row_cross=(start_rows[:,None,:]+rep_res[None,:,:]).reshape(1200,len(central))
    q=np.quantile(row_cross,[.16,.5,.84],axis=0)
    row_bands=pd.DataFrame({"dataset":central.dataset,"row_id":central.row_id,
                            "q16":q[0],"median":q[1],"q84":q[2]})
    dense.to_csv(HERE/"fig4_lambda1_dense_predictions.csv",index=False)
    row_bands.to_csv(HERE/"fig4_lambda1_row_predictions.csv",index=False)
    render(central,dense,row_bands)
    summary={"status":"complete","start_count":24,"experimental_replica_count":50,
             "combined_member_count":1200,
             "combination":"Cartesian crossing of lambda=1 start predictions with centered conditional experimental-replica prediction residuals",
             "fixed_target":"dense frozen W(b) kernel evaluation","colliders":"stored fitted row/bin predictions with PCHIP visual guides",
             "production_sources_modified":False}
    (HERE/"fig4_lambda1_manifest.json").write_text(json.dumps(summary,indent=2)+"\n")


if __name__ == "__main__": main()
