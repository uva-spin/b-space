#!/usr/bin/env python3
"""Build a valence-quark x-kT surface with combined lambda=1 uncertainty."""

from __future__ import annotations

import importlib.util
import argparse
import json
from pathlib import Path
import sys

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.special import j0
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CAMPAIGN = ROOT / "systematics/dataset_identifiability_campaign_2026"
REGISTRY = CAMPAIGN / "summaries/champion_registry/empirical_reference_lambda1_b0p1_2p0_full24.json"
REFERENCE = ROOT / (
    "systematics/collins_factorization_validity/plots/"
    "rowidfix_stageFT_all_qmax0p20_lam0p50_central_exactx/"
    "v22_scheme_tmd_bspace_long.csv"
)
REPLICAS = ROOT / (
    "systematics/collins_factorization_validity/replicas/"
    "rowidfix_stageFT_all_qmax0p20_lam0p50_lambda3_50rep/"
    "tmd_bspace_bands_exactx_50rep/v22_tmd_replica_bspace_long.csv"
)
TRAINER = ROOT / "v21_smoothedA_tail_candidate/train_bt_dnn_v21_smoothedA_tail.py"
TRANSFORM = ROOT / "workflows/v23a/construction/construct_v23a_regularized_kspace_tmd_v2.py"
X_EXACT = np.array([0.1, 0.2, 0.3, 0.5])


def module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod); return mod


def model_for_state(trainer, state_path: Path):
    model = trainer.FilmNPFactor(
        width=48, cond_width=32, n_blocks=3, a0=.05, min_a=0.,
        a_mode="positive", exponent_clip=40., shape_mode="monotone",
        a_smooth_sigma=.45, a_tail_amp=.08, a_tail_b0=3.5,
        a_tail_width=.25, dtype=torch.float32)
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    state = {key.removeprefix("np_factor."): value for key, value in state.items()
             if key.startswith("np_factor.")}
    model.load_state_dict(state, strict=True); model.eval(); return model


def transform_curves(curves: np.ndarray, b_in: np.ndarray, transform):
    b = np.linspace(0., 24., 6001); k = np.linspace(0., 3., 181)
    window = transform.taper_window(b, .92)
    weights = b * window * transform.trapezoid_weights_uniform(b) / (2*np.pi)
    J = j0(np.outer(k, b))
    result=[]
    for curve in curves:
        extended=transform.extend_curve(b_in,curve,b,"expb2",None,1e-300)
        result.append(J @ (extended*weights))
    return k,np.asarray(result)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--flavor",choices=("u","d"),default="u")
    args=parser.parse_args(); flavor=args.flavor
    pid={"u":2,"d":1}[flavor]; figure={"u":7,"d":8}[flavor]
    trainer=module("fig7_trainer",TRAINER); transform=module("fig7_transform",TRANSFORM)
    ref=pd.read_csv(REFERENCE)
    ref=ref[(ref.flavor.astype(str)==flavor) & np.isclose(ref.Q,7.5) & ref.x.isin(X_EXACT)].copy()
    first=ref[np.isclose(ref.x,X_EXACT[0])].sort_values("bT")
    b=first.bT.to_numpy(float)
    perturbative=np.asarray([
        ref[np.isclose(ref.x,x)].sort_values("bT").ftilde_no_np.to_numpy(float)
        for x in X_EXACT])

    registry=json.loads(REGISTRY.read_text()); starts=[]
    x_tensor=torch.tensor(X_EXACT,dtype=torch.float32)
    b_tensor=torch.tensor(b,dtype=torch.float32)
    for i,tag in enumerate(registry["endpoint_tags"],1):
        model=model_for_state(trainer,CAMPAIGN/"outputs"/tag/"model_state.pt")
        with torch.no_grad(): fnp=model(x_tensor,b_tensor).numpy()
        starts.extend([(x*perturbative[ix]*fnp[ix]) for ix,x in enumerate(X_EXACT)])
        print(f"start {i}/24",flush=True)
    starts=np.asarray(starts).reshape(24,4,len(b))

    rep=pd.read_csv(REPLICAS,usecols=["seed","flavor","x","Q","bT","x_ftilde"])
    rep=rep[(rep.flavor.astype(str)==flavor) & np.isclose(rep.Q,7.5) & rep.x.isin(X_EXACT)]
    replicas=[]
    for seed in sorted(rep.seed.unique()):
        replicas.append([rep[(rep.seed==seed)&np.isclose(rep.x,x)].sort_values("bT").x_ftilde.to_numpy(float)
                         for x in X_EXACT])
    replicas=np.asarray(replicas)
    if starts.shape != (24,4,321) or replicas.shape != (50,4,321):
        raise RuntimeError((starts.shape,replicas.shape))

    exact=[]
    for ix,x in enumerate(X_EXACT):
        k,start_k=transform_curves(starts[:,ix,:],b,transform)
        _,rep_k=transform_curves(replicas[:,ix,:],b,transform)
        crossed=(start_k[:,None,:]+(rep_k-np.median(rep_k,axis=0))[None,:,:]).reshape(1200,len(k))
        q16,med,q84=np.quantile(crossed,[.16,.5,.84],axis=0)
        exact.append((q16,med,q84))
    exact=np.asarray(exact)

    log_exact=np.log10(X_EXACT); log_grid=np.linspace(log_exact.min(),log_exact.max(),41)
    x_grid=10**log_grid
    surfaces=np.empty((3,len(x_grid),len(k)))
    for iq in range(3):
        for ik in range(len(k)):
            surfaces[iq,:,ik]=PchipInterpolator(log_exact,exact[:,iq,ik])(log_grid)
    q16,med,q84=surfaces
    # Separate shape-preserving interpolants need not preserve cross-quantile
    # ordering between exact x ridges.  Keep the interpolated median and expand
    # the descriptive envelope only where necessary to contain it.
    qlo=np.minimum(np.minimum(q16,q84),med)
    qhi=np.maximum(np.maximum(q16,q84),med)
    peak=np.max(np.abs(med),axis=1,keepdims=True); floor=np.maximum(.05*peak,1e-300)
    rel=100*.5*(qhi-qlo)/np.maximum(np.abs(med),floor)
    active=np.abs(med)>=floor

    records=[]
    for ix,x in enumerate(x_grid):
        for ik,kv in enumerate(k):
            records.append({"quantity":"x_ftilde","flavor":flavor,"pid":pid,"Q":7.5,
                            "x":x,"kT":kv,"median":med[ix,ik],"q16":qlo[ix,ik],
                            "q84":qhi[ix,ik],"relative_68_halfwidth_percent":rel[ix,ik],
                            "uncertainty_active":bool(active[ix,ik])})
    stem=f"fig{figure}_lambda1_{flavor}_combined_surface"
    pd.DataFrame(records).to_csv(HERE/f"{stem}.csv",index=False)

    active_values=rel[active]
    vmax=float(np.ceil(np.quantile(active_values,.99)/5)*5)
    vmax=max(5.,vmax)
    cmap=matplotlib.colormaps.get_cmap("magma"); norm=colors.Normalize(0,vmax,clip=True)
    K,LX=np.meshgrid(k,log_grid); face=cmap(norm(rel)); face[...,-1]=.96
    plt.rcParams.update({"font.family":"serif","mathtext.fontset":"cm","font.size":13})
    fig=plt.figure(figsize=(7.5,5.8)); ax=fig.add_axes([.02,.06,.70,.86],projection="3d",computed_zorder=False)
    ax.plot_surface(K,LX,med,facecolors=face,linewidth=.08,edgecolor=(1,1,1,.16),
                    antialiased=True,shade=True,rcount=len(x_grid),ccount=len(k))
    lift=.008*max(float(med.max()-med.min()),1e-12)
    for xv in X_EXACT:
        ix=int(np.argmin(abs(x_grid-xv)))
        ax.plot(k,np.full_like(k,log_grid[ix]),med[ix]+lift,color=".22",lw=1.9,alpha=.9,zorder=20)
    zmin=float(med.min()); zmax=float(med.max()); zspan=max(zmax-zmin,1e-12)
    ax.set_xlim(0,3); ax.set_ylim(log_grid.max(),log_grid.min())
    ax.set_zlim(min(0.,zmin)-.03*zspan,zmax+.08*zspan)
    ax.set_xlabel(r"$k_T\ [\mathrm{GeV}]$",labelpad=11,fontsize=16)
    ax.set_ylabel(r"$x$",labelpad=14,fontsize=16)
    ax.set_zlabel(rf"$x f_1^{{{flavor}}}(x,k_T;Q)\ [\mathrm{{GeV}}^{{-2}}]$",labelpad=2,fontsize=13)
    ax.set_yticks(np.log10(X_EXACT)); ax.set_yticklabels([rf"${x:g}$" for x in X_EXACT],fontsize=12)
    ax.tick_params(axis="x",labelsize=12); ax.tick_params(axis="z",labelsize=11)
    ax.set_box_aspect((1.35,1,.78)); ax.view_init(elev=24,azim=-54); ax.grid(False)
    for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
        axis.pane.set_facecolor((1,1,1,0)); axis.pane.set_edgecolor((.7,.7,.7,.75))
    fig.suptitle(rf"${flavor}$, $Q=7.5\ \mathrm{{GeV}}$",fontsize=21,y=.965)
    cax=fig.add_axes([.805,.24,.027,.50]); sm=cm.ScalarMappable(norm=norm,cmap=cmap); sm.set_array([])
    cbar=fig.colorbar(sm,cax=cax); ticks=np.linspace(0,vmax,5); cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{v:g}%" for v in ticks]); cbar.ax.tick_params(labelsize=11)
    cbar.set_label("Combined relative 68% half-width",fontsize=13,labelpad=11)
    fig.savefig(HERE/f"{stem}.pdf",bbox_inches="tight")
    fig.savefig(HERE/f"{stem}.png",dpi=300,bbox_inches="tight")
    plt.close(fig)
    manifest={"status":"complete","flavor":flavor,"Q_GeV":7.5,"x_exact":X_EXACT.tolist(),
              "start_count":24,"experimental_replica_count":50,"combined_members_per_x":1200,
              "transform":{"tail_mode":"expb2","b_max":24,"n_b":6001,"k_max":3,"n_k":181,"taper_start_fraction":.92},
              "color":"combined relative central-68% half-width with 5%-of-peak denominator floor",
              "color_vmax_percent":vmax,"color_values_above_vmax_saturated":False,
              "active_p90_percent":float(np.quantile(active_values,.9)),
              "production_sources_modified":False}
    (HERE/f"fig{figure}_lambda1_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")


if __name__=="__main__": main()
