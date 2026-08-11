#!/usr/bin/env python3
"""Audit sampling robustness and improvement of the combined ensemble.

Exact 24x50 coverage receives comparison metrics even after a scientific
stationarity failure. Such metrics are diagnostic: stationarity remains an
independent mandatory conjunct of the endpoint/promotion gate.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fixed_challenger_protocol import (
    EXPECTED_FNP_REFERENCE_SHA256,
    FNP_REFERENCE as FIXED_FNP_REFERENCE,
    PROTOCOL as FIXED_PROTOCOL,
    fixed_implementation_binding,
    require_fixed_implementation_binding,
    validate_fixed_challenger_protocol,
)


BASE=Path(__file__).resolve().parents[1]
SOURCE=BASE/"summaries/final_combined_tmd_ensemble"
SELECTED=BASE/"summaries/replica_robust_reference_full24/summary.json"
REPLICAS=BASE/"summaries/selected_reference_central_replicas/summary.json"
TARGET=BASE/"summaries/final_combined_ensemble_stability"
STATE_CHAIN_AUDIT=BASE/"summaries/lambda600_state_chain_audit/summary.json"
START_CHAIN_AUDIT=BASE/"summaries/lambda600_start_chain_audit/summary.json"
CHAMPION=BASE/"summaries/champion_registry/empirical_reference_lambda1_b0p1_2p0_full24.json"
HARMONIZED_CONTROL=BASE/"summaries/harmonized_lambda1_logfnp_24x50_comparator"
BOOTSTRAPS=300
SPLITS=200
FLAVORS=("u","d")
DISPLAY_K_MAX=2.25
ACTIVE_PEAK_FRACTION=0.05
STORED_QUANTILE_RTOL=5e-13
STORED_QUANTILE_ATOL=5e-15
LOCKED_INCUMBENT_WIDTHS={
    "u":0.11772613918747582,
    "d":0.12490071924111977,
}


def explicit_bool(value, label: str)->bool:
    if isinstance(value,(bool,np.bool_)):
        return bool(value)
    if isinstance(value,str) and value.strip().lower() in {"true","false"}:
        return value.strip().lower()=="true"
    raise RuntimeError(f"{label} is not an explicit boolean")


def sha256(path: Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1<<20),b""):
            digest.update(block)
    return digest.hexdigest()


def validate_band_table(frame: pd.DataFrame, coordinate: str,
                        required_flavors: set[str]) -> dict:
    required={"component","Q","flavor",coordinate,"q16","median","q84"}
    missing=required-set(frame.columns)
    if missing: raise RuntimeError(f"{coordinate} bands missing columns: {sorted(missing)}")
    if set(frame.component.astype(str))!={"experimental","nonuniqueness","combined"}:
        raise RuntimeError(f"{coordinate} bands have incomplete components")
    numeric=frame[["Q",coordinate,"q16","median","q84"]].to_numpy(float)
    if not np.all(np.isfinite(numeric)):
        raise RuntimeError(f"{coordinate} bands contain non-finite values")
    if np.any(frame.q16.to_numpy(float)>frame["median"].to_numpy(float)) or np.any(
            frame["median"].to_numpy(float)>frame.q84.to_numpy(float)):
        raise RuntimeError(f"{coordinate} bands violate q16<=median<=q84")
    key=["component","Q","flavor",coordinate]
    if "quantity" in frame.columns: key.insert(3,"quantity")
    if frame.duplicated(key).any():
        raise RuntimeError(f"{coordinate} bands contain duplicate grid keys")
    combined=frame[frame.component.astype(str)=="combined"]
    observed=set(combined.flavor.astype(str))
    if not required_flavors.issubset(observed):
        raise RuntimeError(f"{coordinate} bands are missing required flavors")
    group_key=["Q","flavor"] + (["quantity"] if "quantity" in frame.columns else [])
    grid_counts=combined.groupby(group_key)[coordinate].nunique()
    if len(grid_counts)==0 or int(grid_counts.min())<3 or grid_counts.nunique()!=1:
        raise RuntimeError(f"{coordinate} bands have incomplete/inconsistent grids")
    return {"row_count":int(len(frame)),"combined_group_count":int(len(grid_counts)),
            "grid_points_per_group":int(grid_counts.iloc[0])}


def validate_fnp_band_table(frame: pd.DataFrame) -> dict:
    required={"component","x","bT","q16","median","q84"}
    missing=required-set(frame.columns)
    if missing: raise RuntimeError(f"FNP bands missing columns: {sorted(missing)}")
    if set(frame.component.astype(str))!={"experimental","nonuniqueness","combined"}:
        raise RuntimeError("FNP bands have incomplete components")
    numeric=frame[["x","bT","q16","median","q84"]].to_numpy(float)
    if not np.all(np.isfinite(numeric)) or np.any(frame[["q16","median","q84"]].to_numpy(float)<=0):
        raise RuntimeError("FNP bands contain non-finite or non-positive values")
    if np.any(frame.q16.to_numpy(float)>frame["median"].to_numpy(float)) or np.any(
            frame["median"].to_numpy(float)>frame.q84.to_numpy(float)):
        raise RuntimeError("FNP bands violate q16<=median<=q84")
    if frame.duplicated(["component","x","bT"]).any():
        raise RuntimeError("FNP bands contain duplicate grid keys")
    counts=frame.groupby(["component","x"]).bT.nunique()
    if len(counts)!=3 or int(counts.min())<3 or counts.nunique()!=1:
        raise RuntimeError("FNP bands have incomplete/inconsistent grids")
    return {"row_count":int(len(frame)),"grid_points_per_component":int(counts.iloc[0])}


def flavor_local_width_statistic(
        sample: np.ndarray, kvals: np.ndarray,
        incumbent_active: np.ndarray) -> dict:
    """Recompute the exact flavor-local final-width statistic for one sample.

    Each resample supplies its own q16/median/q84 and therefore its own
    candidate active mask.  The incumbent active mask remains fixed.  This is
    the same statistic used by the final incumbent-replacement comparison, not
    a proxy based on motion of individual quantile endpoints.
    """
    values=np.asarray(sample,dtype=float)
    kvals=np.asarray(kvals,dtype=float)
    incumbent_active=np.asarray(incumbent_active,dtype=bool)
    if (values.ndim!=3 or values.shape[0]<1 or values.shape[2]!=len(kvals)
            or incumbent_active.shape!=values.shape[1:]
            or not np.all(np.isfinite(values))
            or not np.all(np.isfinite(kvals))):
        raise RuntimeError("invalid sample for flavor-local width statistic")
    quantiles=np.quantile(values,[.16,.50,.84],axis=0)
    if (not np.all(np.isfinite(quantiles))
            or np.any(quantiles[0]>quantiles[1])
            or np.any(quantiles[1]>quantiles[2])):
        raise RuntimeError("invalid empirical quantiles for width statistic")
    display=kvals<=DISPLAY_K_MAX
    if not np.any(display):
        raise RuntimeError("width-statistic grid has no displayed kT points")
    candidate_active=np.zeros_like(incumbent_active)
    union_active=np.zeros_like(incumbent_active)
    relative_width=np.empty_like(quantiles[1])
    maximum=np.empty(values.shape[1],dtype=float)
    for fi in range(values.shape[1]):
        median=quantiles[1,fi]
        peak=float(np.max(median[display]))
        if not np.isfinite(peak) or peak<=0.0:
            raise RuntimeError("sample has no positive displayed central peak")
        candidate_active[fi]=display&(
            median>ACTIVE_PEAK_FRACTION*peak)
        union_active[fi]=display&(
            candidate_active[fi]|incumbent_active[fi])
        if not np.any(candidate_active[fi]) or not np.any(union_active[fi]):
            raise RuntimeError("sample has an empty flavor-local active mask")
        relative_width[fi]=(quantiles[2,fi]-quantiles[0,fi]) / np.maximum(
            median,1e-30)
        if (not np.all(np.isfinite(relative_width[fi,union_active[fi]]))
                or np.any(relative_width[fi,union_active[fi]]<0.0)):
            raise RuntimeError("sample has invalid relative full widths")
        maximum[fi]=float(np.max(relative_width[fi,union_active[fi]]))
    return {
        "quantiles":quantiles,
        "candidate_active":candidate_active,
        "union_active":union_active,
        "relative_full_width":relative_width,
        "max_relative_full_width":maximum,
    }


def absolute_width_statistic_deviation(
        sample: np.ndarray, full_width: np.ndarray, kvals: np.ndarray,
        incumbent_active: np.ndarray) -> np.ndarray:
    """Absolute resample-to-full movement of the actual final statistic."""
    observed=flavor_local_width_statistic(
        sample,kvals,incumbent_active)["max_relative_full_width"]
    expected=np.asarray(full_width,dtype=float)
    if expected.shape!=observed.shape or not np.all(np.isfinite(expected)):
        raise RuntimeError("invalid full width supplied to resampling audit")
    return np.abs(observed-expected)


def absolute_width_statistic_difference(
        first: np.ndarray, second: np.ndarray, kvals: np.ndarray,
        incumbent_active: np.ndarray) -> np.ndarray:
    """Absolute exact-statistic disagreement between complementary halves."""
    first_width=flavor_local_width_statistic(
        first,kvals,incumbent_active)["max_relative_full_width"]
    second_width=flavor_local_width_statistic(
        second,kvals,incumbent_active)["max_relative_full_width"]
    return np.abs(first_width-second_width)


def direct_resampling_allowance(
        bootstrap: np.ndarray, start_split: np.ndarray,
        replica_split: np.ndarray, joint_split: np.ndarray) -> dict:
    """Take p95 direct-statistic movements with no endpoint-motion factors."""
    arrays=[np.asarray(value,dtype=float) for value in (
        bootstrap,start_split,replica_split,joint_split)]
    if (any(value.ndim!=2 or value.shape[0]<1 for value in arrays)
            or len({value.shape[1] for value in arrays})!=1
            or not all(np.all(np.isfinite(value)) and np.all(value>=0.0)
                       for value in arrays)):
        raise RuntimeError("invalid direct width-statistic resampling arrays")
    p95=[np.quantile(value,.95,axis=0) for value in arrays]
    return {
        "bootstrap_p95":p95[0],
        "start_split_p95":p95[1],
        "replica_split_p95":p95[2],
        "joint_split_p95":p95[3],
        "allowance":np.maximum.reduce(p95),
    }


def validate_stored_combined_kspace_quantiles(
        bands: pd.DataFrame, flavors: list[str], kvals: np.ndarray,
        recomputed: tuple[np.ndarray,np.ndarray,np.ndarray]) -> dict:
    """Require stored renderer inputs to equal long-table quantiles pointwise."""
    frame=bands.copy()
    mask=(frame.component.astype(str).eq("combined")
          &frame.flavor.astype(str).isin(flavors)
          &np.isclose(pd.to_numeric(frame.Q,errors="coerce"),10.0,
                      rtol=0.0,atol=1e-12))
    if "quantity" in frame.columns:
        mask&=frame.quantity.astype(str).eq("ftilde")
    frame=frame[mask]
    if (len(frame)!=len(flavors)*len(kvals)
            or frame.duplicated(["flavor","kT"]).any()):
        raise RuntimeError(
            "stored combined kT band lacks exact renderer-row coverage")
    expected=np.asarray(recomputed,dtype=float)
    if expected.shape!=(3,len(flavors),len(kvals)):
        raise RuntimeError("recomputed combined kT quantiles have invalid shape")
    max_abs=0.0
    max_scaled=0.0
    for fi,flavor in enumerate(flavors):
        group=frame[frame.flavor.astype(str).eq(flavor)].sort_values("kT")
        observed_k=group.kT.to_numpy(float)
        if (len(group)!=len(kvals)
                or not np.allclose(observed_k,kvals,rtol=0.0,atol=1e-14)):
            raise RuntimeError(
                f"stored combined kT grid differs from long data for {flavor}")
        observed=group[["q16","median","q84"]].to_numpy(float).T
        target=expected[:,fi]
        if not np.allclose(
                observed,target,rtol=STORED_QUANTILE_RTOL,
                atol=STORED_QUANTILE_ATOL,equal_nan=False):
            difference=np.abs(observed-target)
            scale=np.maximum(np.abs(target),STORED_QUANTILE_ATOL)
            index=np.unravel_index(int(np.argmax(difference/scale)),
                                   difference.shape)
            raise RuntimeError(
                "stored combined kT quantiles disagree with long data at "
                f"flavor={flavor}, quantile={('q16','median','q84')[index[0]]}, "
                f"kT={kvals[index[1]]:.17g}")
        difference=np.abs(observed-target)
        scale=np.maximum(np.abs(target),STORED_QUANTILE_ATOL)
        max_abs=max(max_abs,float(np.max(difference)))
        max_scaled=max(max_scaled,float(np.max(difference/scale)))
    return {
        "status":"pass",
        "compared_row_count":int(len(frame)),
        "compared_value_count":int(3*len(frame)),
        "relative_tolerance":STORED_QUANTILE_RTOL,
        "absolute_tolerance":STORED_QUANTILE_ATOL,
        "max_absolute_difference":max_abs,
        "max_scaled_difference":max_scaled,
    }


def beats_locked_width(candidate_width: float, allowance: float,
                       locked_width: float) -> bool:
    """Return the fail-closed, flavor-local incumbent replacement decision."""
    values=np.asarray([candidate_width,allowance,locked_width],dtype=float)
    if (not np.all(np.isfinite(values)) or candidate_width<0.0
            or allowance<0.0 or locked_width<=0.0):
        raise RuntimeError("invalid robust-width comparison")
    return bool(candidate_width+allowance<locked_width)


def main()->None:
    _, fixed_protocol_hash=validate_fixed_challenger_protocol()
    source=json.loads((SOURCE/"summary.json").read_text())
    state_chain=json.loads(STATE_CHAIN_AUDIT.read_text())
    require_fixed_implementation_binding(source, "combined-ensemble summary")
    require_fixed_implementation_binding(state_chain, "state-chain audit")
    state_chain_hash=sha256(STATE_CHAIN_AUDIT)
    start_chain_hash=sha256(START_CHAIN_AUDIT)
    prescription=state_chain.get("selected_prescription",{})
    state_chain_gate=(
        state_chain.get("status")=="pass"
        and int(state_chain.get("lambda300_source_count",0))==24
        and int(state_chain.get("start_chain_count",0))==24
        and int(state_chain.get("central_chain_count",0))==1
        and int(state_chain.get("experimental_replica_chain_count",0))==50
        and Path(state_chain.get(
            "terminal_start_ancestry_audit","")).resolve()
            == START_CHAIN_AUDIT.resolve()
        and state_chain.get("terminal_start_ancestry_audit_sha256")
            == start_chain_hash
        and Path(state_chain.get("fixed_challenger_protocol","")).resolve()
            == FIXED_PROTOCOL.resolve()
        and state_chain.get("fixed_challenger_protocol_sha256")
            == fixed_protocol_hash
        and Path(state_chain.get("fixed_fnp_reference","")).resolve()
            == FIXED_FNP_REFERENCE.resolve()
        and state_chain.get("fixed_fnp_reference_sha256")
            == EXPECTED_FNP_REFERENCE_SHA256
        and np.isclose(float(prescription.get("reference_strength",-1)),600.0,
                       rtol=0.0,atol=1e-12)
        and np.isclose(float(prescription.get(
            "fit_quality_barrier_strength",-1)),100.0,
            rtol=0.0,atol=1e-12)
        and int(prescription.get("fit_quality_barrier_power",-1))==2
        and not explicit_bool(state_chain.get("production_sources_modified"),
                              "state_chain.production_sources_modified")
        and explicit_bool(source.get("state_chain_gate_pass"),
                          "source.state_chain_gate_pass")
        and Path(source.get("state_chain_audit","")).resolve()
            == STATE_CHAIN_AUDIT.resolve()
        and source.get("state_chain_audit_sha256")==state_chain_hash)
    if not state_chain_gate:
        raise RuntimeError("combined ensemble state-chain provenance is invalid")
    selected=json.loads(SELECTED.read_text()); replicas=json.loads(REPLICAS.read_text())
    champion=json.loads(CHAMPION.read_text())
    if champion.get("champion_id")!="empirical_reference_lambda1_b0p1_2p0_full24":
        raise RuntimeError("immutable lambda1 incumbent record is invalid")
    incumbent_path=Path(champion["artifacts"]["kspace_combined_bands"])
    expected_hash=champion["artifact_sha256"]["kspace_combined_bands"]
    if not incumbent_path.is_file() or sha256(incumbent_path)!=expected_hash:
        raise RuntimeError("immutable lambda1 incumbent band hash mismatch")
    registered_baseline={key:float(value) for key,value in
                        champion["combined_fig6_max_active_relative_full_width"].items()}
    if set(registered_baseline)!=set(FLAVORS) or any(
            not np.isclose(registered_baseline[key],LOCKED_INCUMBENT_WIDTHS[key],
                           rtol=0.0,atol=1e-14) for key in FLAVORS):
        raise RuntimeError("immutable lambda1 registered width thresholds changed")
    harmonized_summary_path=HARMONIZED_CONTROL/"summary.json"
    harmonized_band_path=HARMONIZED_CONTROL/"kspace_combined_bands.csv"
    if not harmonized_summary_path.is_file() or not harmonized_band_path.is_file():
        raise RuntimeError("post-processing-harmonized lambda1 control is missing")
    harmonized_summary=json.loads(harmonized_summary_path.read_text())
    if (harmonized_summary.get("status") !=
            "complete_postprocessing_harmonized_lambda1_comparator_not_production"
            or harmonized_summary.get("champion_id") != champion.get("champion_id")
            or int(harmonized_summary.get("start_count",0)) != 24
            or int(harmonized_summary.get("experimental_replica_count",0)) != 50
            or harmonized_summary.get("training_protocol_harmonized",True)
            or harmonized_summary.get("registry_modified",True)
            or harmonized_summary.get("comparison_gate_modified",True)
            or harmonized_summary.get("frozen_sources_modified",True)
            or harmonized_summary.get("production_sources_modified",True)):
        raise RuntimeError("post-processing-harmonized lambda1 control is invalid")
    harmonized_bands=pd.read_csv(harmonized_band_path)
    if source["status"]!="complete": raise RuntimeError("combined ensemble incomplete")
    start_stationarity=(selected.get("status")=="complete" and explicit_bool(
        selected.get("all_starts_fnp_plateaued_and_fit_preserved"),
        "all_starts_fnp_plateaued_and_fit_preserved"))
    central_stationarity=explicit_bool(
        replicas.get("central_fnp_plateau_pass"),"central_fnp_plateau_pass")
    replica_stationarity=explicit_bool(
        replicas.get("all_replicas_fnp_plateaued"),
        "all_replicas_fnp_plateaued")
    stationarity=(start_stationarity and central_stationarity
                  and replica_stationarity)
    if (explicit_bool(source.get("candidate_stationarity_gate_pass"),
                      "source candidate_stationarity_gate_pass") != stationarity
            or explicit_bool(source.get("diagnostic_only"),
                             "source diagnostic_only") == stationarity
            or explicit_bool(source.get("promotion_eligible"),
                             "source promotion_eligible")):
        raise RuntimeError("combined ensemble scientific-gate metadata is inconsistent")
    fnp_band_integrity=validate_fnp_band_table(pd.read_csv(SOURCE/"fnp_bands.csv"))
    b_band_integrity=validate_band_table(
        pd.read_csv(SOURCE/"bT_tmd_bands.csv"),"bT",
        {"u","d","s","ubar","dbar","sbar"})
    k_band_frame=pd.read_csv(SOURCE/"kT_tmd_bands.csv")
    k_band_integrity=validate_band_table(
        k_band_frame,"kT",{"u","d"})
    long=pd.read_csv(SOURCE/"kT_tmd_ensemble_long.csv")
    long=long[(long.component=="combined")&(long.quantity=="ftilde")]
    seeds=sorted(long.seed.unique()); reps=sorted(long.pdf_member.unique())
    if ([int(value) for value in seeds] != list(range(303,327))
            or [int(value) for value in reps] != list(range(1001,1051))):
        raise RuntimeError("combined Cartesian ensemble lacks exact 24x50 identities")
    flavors=list(FLAVORS); kvals=np.sort(long.kT.unique())
    pivot=long.pivot(index=["seed","pdf_member"],columns=["flavor","kT"],values="value")
    pivot=pivot.reindex(index=pd.MultiIndex.from_product([seeds,reps],names=["seed","pdf_member"]))
    pivot=pivot.reindex(columns=pd.MultiIndex.from_product([flavors,kvals],names=["flavor","kT"]))
    if pivot.isna().any().any(): raise RuntimeError("combined Cartesian k ensemble is incomplete")
    arr=pivot.to_numpy().reshape(len(seeds),len(reps),len(flavors),len(kvals))
    flat=arr.reshape(-1,len(flavors),len(kvals)); full=tuple(np.quantile(flat,q,axis=0) for q in (.16,.5,.84))
    stored_k_quantile_agreement=validate_stored_combined_kspace_quantiles(
        k_band_frame,flavors,kvals,full)
    incumbent_active=np.zeros((len(flavors),len(kvals)),dtype=bool)
    harmonized_active=np.zeros((len(flavors),len(kvals)),dtype=bool)
    incumbent_bands=pd.read_csv(incumbent_path)
    if "component" in incumbent_bands.columns:
        incumbent_bands=incumbent_bands[
            incumbent_bands.component.astype(str).eq("combined")]
        if "quantity" in incumbent_bands.columns:
            incumbent_bands=incumbent_bands[
                incumbent_bands.quantity.astype(str).eq("ftilde")]
        if "Q" in incumbent_bands.columns:
            incumbent_bands=incumbent_bands[np.isclose(incumbent_bands.Q,10.0)]
    if "median" in incumbent_bands.columns and "central" not in incumbent_bands.columns:
        incumbent_bands=incumbent_bands.rename(columns={"median":"central"})
    incumbent_quantiles=[]
    harmonized_quantiles=[]
    for fi in range(len(flavors)):
        old=incumbent_bands[
            incumbent_bands.flavor.astype(str).eq(flavors[fi])
        ].sort_values("kT")
        if (len(old)!=len(kvals)
                or not np.allclose(old.kT.to_numpy(float),kvals)):
            raise RuntimeError(
                f"incumbent and candidate kT grids differ for {flavors[fi]}")
        oldq=old[["q16","central","q84"]].to_numpy(float).T
        if not np.all(np.isfinite(oldq)) or np.any(oldq[0]>oldq[1]) or np.any(oldq[1]>oldq[2]):
            raise RuntimeError(f"incumbent band is invalid for {flavors[fi]}")
        incumbent_quantiles.append(oldq)
        incumbent_active[fi]=(kvals<=2.25)&(
            oldq[1]>.05*np.max(oldq[1,kvals<=2.25]))
        harmonized=harmonized_bands[
            harmonized_bands.flavor.astype(str).eq(flavors[fi])
        ].sort_values("kT")
        if (len(harmonized)!=len(kvals)
                or not np.allclose(harmonized.kT.to_numpy(float),kvals)):
            raise RuntimeError(
                f"harmonized incumbent and candidate kT grids differ for {flavors[fi]}")
        harmonized_q=harmonized[["q16","central","q84"]].to_numpy(float).T
        if (not np.all(np.isfinite(harmonized_q))
                or np.any(harmonized_q[0]>harmonized_q[1])
                or np.any(harmonized_q[1]>harmonized_q[2])):
            raise RuntimeError(f"harmonized incumbent band is invalid for {flavors[fi]}")
        harmonized_quantiles.append(harmonized_q)
        harmonized_active[fi]=(kvals<=2.25)&(
            harmonized_q[1]>.05*np.max(harmonized_q[1,kvals<=2.25]))
    incumbent_quantiles=np.asarray(incumbent_quantiles)
    harmonized_quantiles=np.asarray(harmonized_quantiles)
    full_statistic=flavor_local_width_statistic(
        flat,kvals,incumbent_active)
    candidate_active=full_statistic["candidate_active"]
    active=full_statistic["union_active"]
    full_width_by_flavor=full_statistic["max_relative_full_width"]
    full_relative_width=full_statistic["relative_full_width"]
    rng=np.random.default_rng(20260731)
    bootstrap=[]
    for _ in range(BOOTSTRAPS):
        si=rng.integers(0,len(seeds),len(seeds)); ri=rng.integers(0,len(reps),len(reps))
        sample=arr[si][:,ri].reshape(-1,len(flavors),len(kvals))
        bootstrap.append(absolute_width_statistic_deviation(
            sample,full_width_by_flavor,kvals,incumbent_active))
    start_split=[]; replica_split=[]; joint_split=[]
    for _ in range(SPLITS):
        sp=rng.permutation(len(seeds)); rp=rng.permutation(len(reps))
        sa,sb=np.array_split(sp,2); ra,rb=np.array_split(rp,2)
        start_split.append(absolute_width_statistic_difference(
            arr[sa].reshape(-1,len(flavors),len(kvals)),
            arr[sb].reshape(-1,len(flavors),len(kvals)),kvals,
            incumbent_active))
        replica_split.append(absolute_width_statistic_difference(
            arr[:,ra].reshape(-1,len(flavors),len(kvals)),
            arr[:,rb].reshape(-1,len(flavors),len(kvals)),kvals,
            incumbent_active))
        joint_split.append(absolute_width_statistic_difference(
            arr[sa][:,ra].reshape(-1,len(flavors),len(kvals)),
            arr[sb][:,rb].reshape(-1,len(flavors),len(kvals)),kvals,
            incumbent_active))
    bootstrap=np.asarray(bootstrap); start_split=np.asarray(start_split)
    replica_split=np.asarray(replica_split); joint_split=np.asarray(joint_split)
    resampling=direct_resampling_allowance(
        bootstrap,start_split,replica_split,joint_split)
    bootstrap_p95_by_flavor=resampling["bootstrap_p95"]
    start_split_p95_by_flavor=resampling["start_split_p95"]
    replica_split_p95_by_flavor=resampling["replica_split_p95"]
    joint_split_p95_by_flavor=resampling["joint_split_p95"]
    bootstrap_p95=float(np.max(bootstrap_p95_by_flavor))
    start_split_p95=float(np.max(start_split_p95_by_flavor))
    replica_split_p95=float(np.max(replica_split_p95_by_flavor))
    joint_split_p95=float(np.max(joint_split_p95_by_flavor))
    widths={}; own_active_widths={}; union_mask_baseline={}; robust_improvement={}
    union_mask_robust_diagnostic={}
    harmonized_candidate_widths={}; harmonized_control_widths={}
    resampling_allowance_by_flavor=resampling["allowance"]
    resampling_full_width_allowance=float(np.max(resampling_allowance_by_flavor))
    for fi,flavor in enumerate(flavors):
        width=full_relative_width[fi]
        widths[flavor]=float(full_width_by_flavor[fi])
        own_active_widths[flavor]=float(np.max(width[candidate_active[fi]]))
        oldq=incumbent_quantiles[fi]
        old_width=(oldq[2]-oldq[0])/np.maximum(oldq[1],1e-30)
        union_mask_baseline[flavor]=float(np.max(old_width[active[fi]]))
        # The union mask is deliberately used for the candidate width so a
        # challenger cannot hide a broader active tail.  It must nevertheless
        # beat the immutable, pre-registered lambda=1 threshold; recomputing the
        # incumbent maximum on the enlarged union could otherwise loosen the
        # gate precisely when the candidate extends farther in kT.
        robust_improvement[flavor]=beats_locked_width(
            widths[flavor],resampling_allowance_by_flavor[fi],
            registered_baseline[flavor])
        union_mask_robust_diagnostic[flavor]=beats_locked_width(
            widths[flavor],resampling_allowance_by_flavor[fi],
            union_mask_baseline[flavor])
        # Diagnostic only: remove the propagation/transform-order confound by
        # comparing against the separately constructed log-FNP lambda1 control
        # on its own union mask.  This does not weaken or replace the pinned
        # incumbent promotion gate above.
        harmonized_union=candidate_active[fi]|harmonized_active[fi]
        harmonized_q=harmonized_quantiles[fi]
        harmonized_width=(harmonized_q[2]-harmonized_q[0])/np.maximum(
            harmonized_q[1],1e-30)
        harmonized_candidate_widths[flavor]=float(np.max(width[harmonized_union]))
        harmonized_control_widths[flavor]=float(np.max(
            harmonized_width[harmonized_union]))
    coverage=(len(seeds)==24 and len(reps)==50
              and int(source.get("start_count",0))==24
              and int(source.get("experimental_replica_count",0))==50
              and int(source.get("combined_member_count",0))==1200)
    band_integrity=stored_k_quantile_agreement.get("status")=="pass"
    finite_sampling_metrics=bool(np.isfinite(
        bootstrap_p95+start_split_p95+replica_split_p95+joint_split_p95))
    diagnostic_figure_gate=(state_chain_gate and coverage and band_integrity
                            and finite_sampling_metrics)
    gate=(diagnostic_figure_gate and stationarity
          and all(robust_improvement.values()))
    failure_reasons=[]
    if not start_stationarity:
        failure_reasons.append("24-start FNP stationarity/fit-preservation gate failed")
    if not central_stationarity:
        failure_reasons.append("central FNP stationarity gate failed")
    if not replica_stationarity:
        failure_reasons.append("experimental-replica FNP stationarity/agreement gate failed")
    if not coverage:
        failure_reasons.append("exact 24x50 coverage gate failed")
    if not band_integrity:
        failure_reasons.append("band integrity gate failed")
    if not finite_sampling_metrics:
        failure_reasons.append("resampling metrics are non-finite")
    if not state_chain_gate:
        failure_reasons.append("state-chain provenance gate failed")
    for flavor in flavors:
        if not robust_improvement[flavor]:
            failure_reasons.append(
                f"{flavor} final width plus resampling allowance does not beat "
                "the immutable registered lambda1 threshold")
    TARGET.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({
        "bootstrap_max_absolute_full_width_statistic_deviation":np.max(
            bootstrap,axis=1),
        **{f"bootstrap_{flavor}_absolute_full_width_statistic_deviation":
           bootstrap[:,fi]
           for fi,flavor in enumerate(flavors)},
    }).to_csv(
        TARGET/"bootstrap_full_width_statistic_deviations.csv",index=False)
    pd.DataFrame({
        "start_split_max_absolute_full_width_statistic_difference":np.max(
            start_split,axis=1),
        "replica_split_max_absolute_full_width_statistic_difference":np.max(
            replica_split,axis=1),
        "joint_split_max_absolute_full_width_statistic_difference":np.max(
            joint_split,axis=1),
        **{f"start_split_{flavor}_absolute_full_width_statistic_difference":
           start_split[:,fi]
           for fi,flavor in enumerate(flavors)},
        **{f"replica_split_{flavor}_absolute_full_width_statistic_difference":
           replica_split[:,fi]
           for fi,flavor in enumerate(flavors)},
        **{f"joint_split_{flavor}_absolute_full_width_statistic_difference":
           joint_split[:,fi]
           for fi,flavor in enumerate(flavors)},
    }).to_csv(TARGET/"split_half_full_width_statistic_differences.csv",index=False)
    summary={
        "status":"complete","endpoint_gate_pass":gate,
        "promotion_eligible":gate,
        "diagnostic_only":not gate,
        "diagnostic_figure_gate_pass":diagnostic_figure_gate,
        "state_chain_gate_pass":state_chain_gate,
        "state_chain_audit":str(STATE_CHAIN_AUDIT),
        "state_chain_audit_sha256":state_chain_hash,
        "fixed_challenger_protocol":str(FIXED_PROTOCOL),
        "fixed_challenger_protocol_sha256":fixed_protocol_hash,
        "fixed_fnp_reference":str(FIXED_FNP_REFERENCE),
        "fixed_fnp_reference_sha256":EXPECTED_FNP_REFERENCE_SHA256,
        **fixed_implementation_binding(),
        "candidate_stationarity_gate_pass":stationarity,
        "start_stationarity_and_fit_gate_pass":start_stationarity,
        "central_stationarity_gate_pass":central_stationarity,
        "replica_stationarity_and_agreement_gate_pass":replica_stationarity,
        "scientific_failure_reasons":failure_reasons,
        "band_integrity_gate_pass":band_integrity,
        "fnp_band_integrity":fnp_band_integrity,
        "bspace_band_integrity":b_band_integrity,
        "kspace_band_integrity":k_band_integrity,
        "kspace_combined_long_band_quantile_agreement":
            stored_k_quantile_agreement,
        "coverage_gate_pass":coverage,"start_count":len(seeds),"replica_count":len(reps),
        "bootstrap_replicates":BOOTSTRAPS,"split_half_replicates":SPLITS,
        "bootstrap_p95_absolute_full_width_statistic_deviation":bootstrap_p95,
        "start_split_half_p95_absolute_full_width_statistic_difference":
            start_split_p95,
        "replica_split_half_p95_absolute_full_width_statistic_difference":
            replica_split_p95,
        "joint_split_half_p95_absolute_full_width_statistic_difference":
            joint_split_p95,
        "resampling_full_width_allowance":resampling_full_width_allowance,
        "bootstrap_p95_absolute_full_width_statistic_deviation_by_flavor":
            dict(zip(flavors,bootstrap_p95_by_flavor.tolist())),
        "start_split_half_p95_absolute_full_width_statistic_difference_by_flavor":
            dict(zip(flavors,start_split_p95_by_flavor.tolist())),
        "replica_split_half_p95_absolute_full_width_statistic_difference_by_flavor":
            dict(zip(flavors,replica_split_p95_by_flavor.tolist())),
        "joint_split_half_p95_absolute_full_width_statistic_difference_by_flavor":
            dict(zip(flavors,joint_split_p95_by_flavor.tolist())),
        "resampling_full_width_allowance_by_flavor":dict(
            zip(flavors,resampling_allowance_by_flavor.tolist())),
        "resampling_artifacts":{
            "bootstrap_full_width_statistic_deviations":str(
                TARGET/"bootstrap_full_width_statistic_deviations.csv"),
            "split_half_full_width_statistic_differences":str(
                TARGET/"split_half_full_width_statistic_differences.csv"),
        },
        "resampling_allowance_semantics":(
            "direct resampling of the exact flavor-local final statistic: every "
            "bootstrap or complementary half recomputes q16/median/q84, its own "
            "5%-of-own-peak candidate mask, the union with the fixed incumbent "
            "active mask, and the maximum relative full width. The allowance is "
            "the maximum of the bootstrap p95 absolute deviation from the raw full "
            "statistic and the start/replica/joint split p95 absolute half-to-half "
            "differences, with no endpoint-motion conversion factor. It is not "
            "added to the plotted q16--q84 band and is not a confidence-level "
            "calibration"),
        "cartesian_sampling_semantics":(
            "1200 values are the equal-weight product empirical distribution of 24 "
            "optimizer-start residual curves and 50 experimental-replica residual curves, "
            "not 1200 independent nested fits"),
        "raw_final_max_active_relative_full_width":widths,
        # Backward-compatible gating key; it is the same uninflated raw width.
        "final_max_active_relative_full_width":widths,
        "robust_adjusted_max_active_relative_full_width":{
            flavor:float(widths[flavor]+resampling_allowance_by_flavor[fi])
            for fi,flavor in enumerate(flavors)},
        "candidate_own_active_relative_full_width":own_active_widths,
        "comparison_active_mask_definition":"union, flavor by flavor, of candidate and incumbent kT<=2.25 regions whose respective central exceeds 5% of its own displayed peak",
        "comparison_active_kT_max_by_flavor":{
            flavor:float(np.max(kvals[active[fi]]))
            for fi,flavor in enumerate(flavors)},
        "comparison_champion_id":champion["champion_id"],
        # Backward-compatible gating key: this is intentionally the immutable
        # registered own-mask threshold, never a candidate-dependent union-mask
        # recomputation.
        "comparison_champion_max_active_relative_full_width":registered_baseline,
        "comparison_champion_registered_own_mask_relative_full_width":registered_baseline,
        "comparison_champion_union_mask_relative_full_width":union_mask_baseline,
        "union_mask_robust_improvement_diagnostic_by_flavor":
            union_mask_robust_diagnostic,
        "postprocessing_harmonized_lambda1_control":{
            "status":harmonized_summary["status"],
            "summary":str(harmonized_summary_path),
            "summary_sha256":sha256(harmonized_summary_path),
            "kspace_band":str(harmonized_band_path),
            "kspace_band_sha256":sha256(harmonized_band_path),
            "training_protocol_harmonized":False,
            "comparison_active_mask_definition":"union, flavor by flavor, of candidate and post-processing-harmonized lambda1 kT<=2.25 regions whose respective central exceeds 5% of its own displayed peak",
            "candidate_raw_width_on_union_mask":harmonized_candidate_widths,
            "lambda1_control_raw_width_on_union_mask":harmonized_control_widths,
            "gating":False,
            "interpretation":"diagnostic control for propagation algebra and transform order; the immutable legacy lambda1 band remains the stricter promotion gate",
        },
        "incumbent_and_candidate_training_protocols_identical":False,
        "comparison_semantics":(
            "candidate width is evaluated on the conservative union active mask and "
            "pays its flavor-specific resampling allowance, but promotion is compared "
            "against the immutable pre-registered lambda1 own-mask threshold. The "
            "lambda1 width recomputed on the union mask is secondary diagnostic evidence "
            "only. Width metrics are retained as diagnostics when stationarity fails, "
            "but cannot authorize promotion"),
        "historical_baseline_max_active_relative_full_width":champion["historical_baseline_fig6_max_active_relative_full_width"],
        "robust_improvement_rule":"for each flavor separately, the raw final candidate maximum relative q16--q84 full width on the full-sample candidate/incumbent union mask plus max(bootstrap p95 absolute direct-statistic deviation, start/replica/joint complementary-half p95 absolute direct-statistic difference) remains below the immutable registered lambda1 threshold (u=0.11772613918747582, d=0.12490071924111977); every resample or half recomputes its own candidate 5%-of-own-peak mask unioned with the fixed incumbent mask, no endpoint-motion factor is used, and recomputed incumbent union-mask widths are diagnostic only",
        "robust_improvement_gate_by_flavor":robust_improvement,
        "two_percent_partition_gate_used":False,
        "production_sources_modified":False,
    }
    (TARGET/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
