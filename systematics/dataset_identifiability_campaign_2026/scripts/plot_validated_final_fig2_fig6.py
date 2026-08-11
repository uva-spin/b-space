#!/usr/bin/env python3
"""Render validated or explicitly non-promotable diagnostic Fig. 2/Fig. 6."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from nested_interaction_and_final_envelope_validation import (
    final_promotion_gate,
    validated_final_directional_envelope,
)


BASE=Path(__file__).resolve().parents[1]
SOURCE=BASE/"summaries/final_combined_tmd_ensemble"
AUDIT=BASE/"summaries/final_combined_ensemble_stability/summary.json"
FINAL_ENVELOPE=(
    BASE/"summaries/lambda600_final_directional_envelope/summary.json"
)
TARGET=BASE/"summaries/final_fig2_fig6"
COLORS={"u":"#1f77b4","d":"#d95f02","s":"#2ca02c","ubar":"#9467bd","dbar":"#8c564b","sbar":"#e377c2"}
LABELS={"u":r"$u$ quark","d":r"$d$ quark","s":r"$s$ quark","ubar":r"$\bar u$ quark","dbar":r"$\bar d$ quark","sbar":r"$\bar s$ quark"}
VALIDATED_STEMS={
    "fnp":"updated_fnp_bspace_product_plus_directional_envelope",
    "fig2":"updated_fig2_bspace_product_plus_directional_envelope",
    "fig6":"updated_fig6_kspace_ud_product_plus_directional_envelope",
}
DIAGNOSTIC_STEMS={
    "fnp":"diagnostic_failed_fnp_bspace_product_plus_directional_envelope",
    "fig2":"diagnostic_failed_fig2_bspace_product_plus_directional_envelope",
    "fig6":"diagnostic_failed_fig6_kspace_ud_product_plus_directional_envelope",
}
BAND_LABEL=(
    "experimental replicas + optimizer-start non-uniqueness\n"
    "+ residual convergence/interaction"
)


def explicit_bool(value,label):
    if isinstance(value,(bool,np.bool_)):
        return bool(value)
    if isinstance(value,str) and value.strip().lower() in {"true","false"}:
        return value.strip().lower()=="true"
    raise RuntimeError(f"{label} is not an explicit boolean")


def fsync_directory(path: Path) -> None:
    descriptor=os.open(path,os.O_RDONLY|getattr(os,"O_DIRECTORY",0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save(fig,stem):
    """Publish a complete PNG/PDF pair without exposing partial bytes."""
    staged=[]
    try:
        for suffix,format_name,dpi in ((".png","png",240),(".pdf","pdf",None)):
            target=TARGET/f"{stem}{suffix}"
            temporary=target.with_name(f".{target.name}.tmp.{os.getpid()}")
            staged.append((temporary,target))
            kwargs={"format":format_name}
            if dpi is not None:
                kwargs["dpi"]=dpi
            fig.savefig(temporary,**kwargs)
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
        for temporary,target in staged:
            os.replace(temporary,target)
        fsync_directory(TARGET)
    finally:
        for temporary,_ in staged:
            temporary.unlink(missing_ok=True)
        plt.close(fig)


def remove_opposite_status_artifacts(diagnostic: bool) -> None:
    """Remove only exact stale figure names from the opposite outcome class."""
    opposite=VALIDATED_STEMS if diagnostic else DIAGNOSTIC_STEMS
    for stem in opposite.values():
        for suffix in (".png",".pdf"):
            (TARGET/f"{stem}{suffix}").unlink(missing_ok=True)
    fsync_directory(TARGET)


def atomic_json(path: Path,payload: dict) -> None:
    temporary=path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w",encoding="utf-8") as stream:
            json.dump(payload,stream,indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary,path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def interval_metadata(diagnostic: bool) -> dict:
    uncertainty=(
        "diagnostic empirical product band plus residual convergence/"
        "interaction envelope; the product combines experimental replicas and "
        "optimizer-start nonuniqueness; scientific gates failed"
        if diagnostic else
        "empirical product band plus residual convergence/interaction envelope; "
        "the product combines experimental replicas and optimizer-start "
        "nonuniqueness")
    return {
        "uncertainty":uncertainty,
        "formal_confidence_level_assigned":False,
        "one_sigma_claimed":False,
        "probability_semantics":(
            "the experimental-replica marginal has its conventional conditional "
            "replica interpretation; optimizer-start nonuniqueness has no calibrated "
            "probability law; checkpoint motion and the 2x3 interaction design "
            "are directional stresses, so the final envelope is not a confidence "
            "interval or standard-deviation interval"),
    }


def main()->None:
    audit=json.loads(AUDIT.read_text())
    final_envelope,final_envelope_hash=validated_final_directional_envelope(
        FINAL_ENVELOPE, stability_path=AUDIT)
    endpoint_pass=final_promotion_gate(final_envelope)
    diagnostic_gate=explicit_bool(
        final_envelope.get("diagnostic_figure_gate_pass"),
        "final envelope diagnostic_figure_gate_pass")
    stationarity_pass=explicit_bool(
        audit.get("candidate_stationarity_gate_pass"),
        "candidate_stationarity_gate_pass")
    if not diagnostic_gate:
        raise RuntimeError("ensemble lacks complete finite 24x50 diagnostic evidence")
    if endpoint_pass and not stationarity_pass:
        raise RuntimeError("stationarity failure cannot produce validated figures")
    diagnostic=not endpoint_pass
    if diagnostic:
        failure_reasons=[str(value) for value in final_envelope.get(
            "scientific_failure_reasons",[]) if str(value)]
        if not failure_reasons:
            raise RuntimeError("diagnostic figure lacks an explicit failed gate")
        diagnostic_title=(
            "DIAGNOSTIC ONLY — stationarity failed; not promotable"
            if not stationarity_pass else
            "DIAGNOSTIC ONLY — final promotion gate failed; not promotable")
        central_label="diagnostic trained central endpoint"
        band_label="diagnostic: "+BAND_LABEL
    else:
        failure_reasons=[]
        diagnostic_title=None
        central_label="trained central endpoint"
        band_label=BAND_LABEL
    fnp=pd.read_csv(Path(final_envelope["artifacts"]["fnp_final_envelope"]))
    b=pd.read_csv(Path(final_envelope["artifacts"]["fig2_bspace_final_envelope"]))
    k=pd.read_csv(Path(final_envelope["artifacts"]["fig6_kspace_final_envelope"]))
    TARGET.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"serif","mathtext.fontset":"cm","axes.linewidth":.9,
                         "xtick.direction":"in","ytick.direction":"in","xtick.top":True,"ytick.right":True})

    fig,ax=plt.subplots(figsize=(7.4,5.0),constrained_layout=True)
    g=fnp[(np.isclose(fnp.x,.1))&(fnp.bT<=4)].sort_values("bT")
    if g.empty: raise RuntimeError("final FNP diagnostic is missing x=0.1")
    ax.fill_between(g.bT,g.final_envelope_low,g.final_envelope_high,
                    color="#4c72b0",alpha=.22,linewidth=0)
    ax.plot(g.bT,g["trained_central"],color="#1f4e79",lw=1.8,
            label=central_label)
    if diagnostic:
        ax.set_title(diagnostic_title,color="#9b2226",fontsize=11,fontweight="bold")
    ax.text(.98,.96,r"$x=0.1$",ha="right",va="top",transform=ax.transAxes,fontsize=12)
    ax.set(xlabel=r"$b_T\ [\mathrm{GeV}^{-1}]$",ylabel=r"$F_{\rm NP}(x,b_T)$",xlim=(0,4))
    ax.grid(alpha=.15)
    ax.legend(handles=[ax.lines[0],Patch(facecolor="#4c72b0",alpha=.22,edgecolor="none")],
              labels=[central_label,band_label],
              frameon=False,fontsize=10)
    save(fig,DIAGNOSTIC_STEMS["fnp"] if diagnostic else VALIDATED_STEMS["fnp"])

    fig,ax=plt.subplots(figsize=(7.4,5.0),constrained_layout=True)
    view=b[np.isclose(b.Q,7.5)&(b.bT<=4)]
    for flavor in ("u","d","s","ubar","dbar","sbar"):
        g=view[view.flavor.astype(str)==flavor].sort_values("bT"); c=COLORS[flavor]
        if g.empty: raise RuntimeError(f"Fig. 2 missing {flavor}")
        ax.fill_between(g.bT,g.final_envelope_low,g.final_envelope_high,
                        color=c,alpha=.18,linewidth=0)
        ax.plot(g.bT,g["trained_central"],color=c,lw=1.7,
                label=LABELS[flavor])
    if diagnostic:
        ax.set_title(diagnostic_title,color="#9b2226",fontsize=11,fontweight="bold")
    ax.text(.98,.96,r"$x=0.1\qquad Q=7.5\ \mathrm{GeV}$",ha="right",va="top",transform=ax.transAxes,fontsize=12)
    ax.set(xlabel=r"$b_T\ [\mathrm{GeV}^{-1}]$",ylabel=r"$\widetilde f_1^q(x,b_T;Q)$",xlim=(0,4))
    ax.grid(alpha=.15)
    h,l=ax.get_legend_handles_labels(); h.append(Patch(facecolor=".55",alpha=.25,edgecolor="none")); l.append(band_label)
    # Put the seven-entry legend in the open upper-right interior below the
    # kinematic annotation.  At this bT range every flavor curve has already
    # fallen below that region, avoiding both annotation and curve overlap.
    ax.legend(h,l,loc="center right",bbox_to_anchor=(.98,.66),
              frameon=False,fontsize=8.2,ncol=2,columnspacing=1.1,
              handlelength=2.6)
    save(fig,DIAGNOSTIC_STEMS["fig2"] if diagnostic else VALIDATED_STEMS["fig2"])

    fig,ax=plt.subplots(figsize=(7.4,5.0),constrained_layout=True)
    view=k[np.isclose(k.Q,10)&(k.kT<=2.25)]
    for flavor in ("u","d"):
        g=view[view.flavor.astype(str)==flavor].sort_values("kT"); c=COLORS[flavor]
        if g.empty: raise RuntimeError(f"Fig. 6 missing {flavor}")
        ax.fill_between(g.kT,g.final_envelope_low,g.final_envelope_high,
                        color=c,alpha=.20,linewidth=0)
        ax.plot(g.kT,g["trained_central"],color=c,lw=1.8,
                label=LABELS[flavor])
    if diagnostic:
        ax.set_title(diagnostic_title,color="#9b2226",fontsize=11,fontweight="bold")
    ax.text(.98,.96,r"$x=0.1\qquad Q=10\ \mathrm{GeV}$",ha="right",va="top",transform=ax.transAxes,fontsize=12)
    ax.set(xlabel=r"$k_T\ (\mathrm{GeV})$",ylabel=r"$f_1^q(x,k_T;Q)$",xlim=(0,2.25))
    ax.set_ylim(bottom=0); ax.grid(alpha=.15)
    h,l=ax.get_legend_handles_labels(); h.append(Patch(facecolor=".55",alpha=.25,edgecolor="none")); l.append(band_label)
    # Match the original Fig. 6 visual balance: a compact legend centered in
    # the open region above the falling u/d curves, away from the x,Q label.
    ax.legend(h,l,loc="center",bbox_to_anchor=(.62,.59),
              frameon=False,fontsize=10)
    save(fig,DIAGNOSTIC_STEMS["fig6"] if diagnostic else VALIDATED_STEMS["fig6"])
    # All three current-status pairs now exist.  Only at this point retire the
    # exact filenames belonging to the opposite outcome class; summary.json is
    # the final atomic commit marker for this rendering generation.
    remove_opposite_status_artifacts(diagnostic)
    stems=DIAGNOSTIC_STEMS if diagnostic else VALIDATED_STEMS
    fnp_name=f"{stems['fnp']}.pdf"
    fig2_name=f"{stems['fig2']}.pdf"
    fig6_name=f"{stems['fig6']}.pdf"
    summary={"status":("diagnostic_figures_not_promotable" if diagnostic
                       else "final_validated_figures"),
             "fnp_diagnostic":fnp_name,
             "figure_2":fig2_name,
             "figure_6":fig6_name,"updated_only":True,
             "contains_individual_seed_curves":False,"contains_legacy_conditional_result":False,
             "central_line":"the separately trained terminal lambda600 central endpoint, propagated directly in b space and through its exact paired expb2 finite-b transform",
             "ensemble_median_role":"retained in source tables as a distribution diagnostic; not used as the plotted central model",
             "displayed_band":(
                 "empirical product of experimental replicas and optimizer-start "
                 "nonuniqueness plus residual convergence/interaction envelope"),
             **interval_metadata(diagnostic),
             "diagnostic_only":diagnostic,
             "promotion_eligible":endpoint_pass,
             "endpoint_gate_pass":endpoint_pass,
             "candidate_stationarity_gate_pass":stationarity_pass,
             "scientific_failure_reasons":failure_reasons,
             "validated_promotion_artifacts_written":not diagnostic,
             "source_audit":str(AUDIT),
             "source_final_directional_envelope":str(FINAL_ENVELOPE),
             "source_final_directional_envelope_sha256":final_envelope_hash,
             "production_sources_modified":False}
    atomic_json(TARGET/"summary.json",summary)
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
