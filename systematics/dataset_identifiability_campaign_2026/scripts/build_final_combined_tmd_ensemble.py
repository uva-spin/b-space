#!/usr/bin/env python3
"""Build final b/k TMD bands from experimental and start distributions.

The positive FNP curves are combined through a Cartesian convolution of
centered log-curve residuals: central + start residual + replica residual in
log space. This preserves each member's b correlation, positivity, and the
separate empirical distributions without assigning Gaussian meaning to the
start ensemble. Pointwise q16--q84 quantiles of a stationary ensemble define
the requested operational empirical band, not a formal one-sigma confidence
interval. Exact 24x50 coverage that fails a scientific
stationarity gate is still constructed, but is marked diagnostic-only and can
never authorize promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

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


BASE = Path(__file__).resolve().parents[1]
SYSTEMATICS = BASE.parent
ROOT = SYSTEMATICS.parent
SELECTED = BASE / "summaries/replica_robust_reference_full24/summary.json"
REPLICAS = BASE / "summaries/selected_reference_central_replicas/summary.json"
BOUNDARY_CONFIRMATION = BASE / "summaries/selected_reference_boundary_confirmation/summary.json"
REFERENCE_B = SYSTEMATICS / "collins_factorization_validity/plots/rowidfix_stageFT_all_qmax0p20_lam0p50_central_exactx/v22_scheme_tmd_bspace_long.csv"
TRANSFORMER = ROOT / "construct_v23a_regularized_kspace_tmd_v2.py"
TARGET = BASE / "summaries/final_combined_tmd_ensemble"
STATE_CHAIN_SCRIPT = BASE / "scripts/audit_lambda600_state_chains.py"
STATE_CHAIN_AUDIT = BASE / "summaries/lambda600_state_chain_audit/summary.json"
START_CHAIN_AUDIT = BASE / "summaries/lambda600_start_chain_audit/summary.json"
FLAVORS_B = ("u", "d", "s", "ubar", "dbar", "sbar")
FLAVORS_K = ("u", "d")
EPS = 1e-30


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validated_state_chain_audit() -> dict:
    """Regenerate and require the exact 24+1+50 ancestry manifest."""
    _, fixed_protocol_hash = validate_fixed_challenger_protocol()
    subprocess.run([sys.executable, str(STATE_CHAIN_SCRIPT)], check=True,
                   stdout=subprocess.DEVNULL)
    audit = json.loads(STATE_CHAIN_AUDIT.read_text())
    require_fixed_implementation_binding(audit, "state-chain audit")
    start_audit_hash = sha256(START_CHAIN_AUDIT)
    prescription = audit.get("selected_prescription", {})
    if not (audit.get("status") == "pass"
            and int(audit.get("lambda300_source_count", 0)) == 24
            and int(audit.get("start_chain_count", 0)) == 24
            and int(audit.get("central_chain_count", 0)) == 1
            and int(audit.get("experimental_replica_chain_count", 0)) == 50
            and Path(audit.get(
                "terminal_start_ancestry_audit", "")).resolve()
                == START_CHAIN_AUDIT.resolve()
            and audit.get("terminal_start_ancestry_audit_sha256")
                == start_audit_hash
            and Path(audit.get("fixed_challenger_protocol", "")).resolve()
                == FIXED_PROTOCOL.resolve()
            and audit.get("fixed_challenger_protocol_sha256")
                == fixed_protocol_hash
            and Path(audit.get("fixed_fnp_reference", "")).resolve()
                == FIXED_FNP_REFERENCE.resolve()
            and audit.get("fixed_fnp_reference_sha256")
                == EXPECTED_FNP_REFERENCE_SHA256
            and np.isclose(float(prescription.get("reference_strength", -1)),
                           600.0, rtol=0.0, atol=1e-12)
            and np.isclose(float(prescription.get(
                "fit_quality_barrier_strength", -1)),
                100.0, rtol=0.0, atol=1e-12)
            and int(prescription.get("fit_quality_barrier_power", -1)) == 2
            and not explicit_bool(audit.get("production_sources_modified"),
                                  "state_chain.production_sources_modified")):
        raise RuntimeError("lambda=600 state-chain audit failed")
    return audit


def explicit_bool(value, label: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise RuntimeError(f"{label} is not an explicit boolean")


def scientific_gate_state(selected: dict, replicas: dict) -> dict:
    """Validate terminal-status semantics without conflating three gates."""
    selected_status=str(selected.get("status"))
    replica_status=str(replicas.get("status"))
    if selected_status not in {"complete","verification_failed"}:
        raise RuntimeError(
            "24-start evidence is neither complete nor an explicit scientific failure")
    if replica_status not in {
            "complete","complete_with_scientific_failures",
            "central_stationarity_failed","replica_stationarity_failed"}:
        raise RuntimeError(
            "50-replica evidence is neither complete nor an explicit scientific failure")
    start_pass=explicit_bool(
        selected.get("all_starts_fnp_plateaued_and_fit_preserved"),
        "all_starts_fnp_plateaued_and_fit_preserved")
    central_pass=explicit_bool(
        replicas.get("central_fnp_plateau_pass"),
        "central_fnp_plateau_pass")
    replica_pass=explicit_bool(
        replicas.get("all_replicas_fnp_plateaued"),
        "all_replicas_fnp_plateaued")
    if (selected_status == "complete") != start_pass:
        raise RuntimeError("24-start status and scientific gate disagree")
    overall=start_pass and central_pass and replica_pass
    if ((replica_status == "complete" and not replica_pass)
            or (replica_status == "complete_with_scientific_failures" and overall)
            or (replica_status == "central_stationarity_failed" and central_pass)
            or (replica_status == "replica_stationarity_failed" and replica_pass)):
        raise RuntimeError("replica terminal status and explicit gates disagree")
    return {
        "selected_status":selected_status,"replica_status":replica_status,
        "start_pass":start_pass,"central_pass":central_pass,
        "replica_pass":replica_pass,"overall_pass":overall,
    }


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def curve(run: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(run / "fnp_grid.csv")
    frame = frame[np.isclose(frame.x, .1)].sort_values("bT")
    b = frame.bT.to_numpy(float)
    value = frame.F_NP.to_numpy(float)
    if len(b) < 3 or len(np.unique(b)) != len(b):
        raise RuntimeError(f"invalid x=0.1 FNP grid in {run}")
    if not np.all(np.isfinite(b)) or not np.all(np.isfinite(value)):
        raise RuntimeError(f"non-finite x=0.1 FNP curve in {run}")
    return b, value


def curves_on_common_grid(runs: list[Path], expected_b: np.ndarray,
                          label: str) -> np.ndarray:
    values = []
    for run in runs:
        b, value = curve(run)
        if b.shape != expected_b.shape or not np.allclose(
                b, expected_b, rtol=0.0, atol=1e-12):
            raise RuntimeError(f"{label} FNP grid differs from central grid: {run}")
        values.append(value)
    return np.asarray(values)


def validated_status(run: Path, strength: float, bmax: float,
                     barrier_strength: float, barrier_power: int) -> dict:
    status = json.loads((run / "fit_status.json").read_text())
    reference = status["regularization"]["fnp_reference_distance"]
    if not np.isclose(float(reference["lambda"]),strength,
                      rtol=0.0,atol=1e-12):
        raise RuntimeError(f"constraint strength mismatch in {run}")
    if not np.isclose(float(reference["b_max"]),bmax,
                      rtol=0.0,atol=1e-12):
        raise RuntimeError(f"constraint bmax mismatch in {run}")
    if Path(reference["target_csv"]).resolve() != (
            BASE / "summaries/exact_baseline_fnp_median/fnp_median.csv").resolve():
        raise RuntimeError(f"constraint reference mismatch in {run}")
    if explicit_bool(status["production_state_modified"],
                     f"production_state_modified in {run}"):
        raise RuntimeError(f"endpoint reports modified production state: {run}")
    barrier = status["regularization"]["fit_quality_barrier"]
    if not np.isclose(float(barrier["lambda"]),barrier_strength,
                      rtol=0.0,atol=1e-12):
        raise RuntimeError(f"fit-quality barrier strength mismatch in {run}")
    if int(barrier["power"]) != barrier_power:
        raise RuntimeError(f"fit-quality barrier power mismatch in {run}")
    return status


def quantiles(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(np.quantile(values, value, axis=0) for value in (.16, .50, .84))


def fnp_bands(components: list[tuple[str, np.ndarray]]) -> pd.DataFrame:
    rows = []
    for component, log_curves in components:
        q16, med, q84 = quantiles(np.exp(log_curves))
        rows.append(pd.DataFrame({
            "component": component, "x": 0.1, "bT": B_GRID,
            "q16": q16, "median": med, "q84": q84,
        }))
    return pd.concat(rows, ignore_index=True)


def bspace_bands(reference: pd.DataFrame, log_curves: np.ndarray,
                 component: str) -> pd.DataFrame:
    rows=[]
    for (q, flavor), group in reference.groupby(["Q","flavor"],sort=False):
        group=group.sort_values("bT")
        b=group.bT.to_numpy(float)
        # All source FNP grids share the reference 321-node b grid; interpolate
        # only to make this invariant explicit.
        fnp=np.exp(np.asarray([np.interp(b, B_GRID, item) for item in log_curves]))
        tmd=fnp*group.ftilde_no_np.to_numpy(float)[None,:]
        q16,med,q84=quantiles(tmd)
        rows.append(pd.DataFrame({"component":component,"Q":q,"flavor":flavor,
                                  "bT":b,"q16":q16,"median":med,"q84":q84}))
    return pd.concat(rows,ignore_index=True)


def transform_component(reference: pd.DataFrame, logs: np.ndarray,
                        keys: list[tuple[int,int]], component: str,
                        transform) -> tuple[pd.DataFrame,pd.DataFrame,dict]:
    rows=[]
    selected=reference[np.isclose(reference.Q,10)&reference.flavor.astype(str).isin(FLAVORS_K)]
    for index, ((start_id,replica_id),logf) in enumerate(zip(keys,logs)):
        for (_,flavor),group in selected.groupby(["pid","flavor"],sort=False):
            group=group.sort_values("bT").copy(); b=group.bT.to_numpy(float)
            group["F_NP"]=np.exp(np.interp(b,B_GRID,logf))
            group["ftilde"]=group.ftilde_no_np*group.F_NP
            group["_replica_key"]=f"{component}|s{start_id}|r{replica_id}"
            group["seed"]=start_id; group["pdf_member"]=replica_id
            rows.append(group)
    frame=pd.concat(rows,ignore_index=True)
    settings=argparse.Namespace(
        quantities=["ftilde"],tail_mode="expb2",tail_fit_bmin=None,
        eps=1e-300,b_transform_max=24.0,n_b_transform=6001,
        k_max=4.0,n_k=401,end_taper_start_fraction=.92)
    long,meta=transform.transform_curves(frame,settings)
    long["component"]=component
    bands=transform.make_bands(long); bands["component"]=component
    return long,bands,meta


def active_width_metrics(bands: pd.DataFrame) -> dict:
    result={}
    for flavor,group in bands[bands.kT<=2.25].groupby("flavor"):
        group=group.sort_values("kT"); med=group["median"].to_numpy(float)
        active=med>.05*np.max(med)
        width=(group.q84-group.q16).to_numpy(float)/np.maximum(med,EPS)
        result[str(flavor)]={
            "max_relative_full_width":float(np.max(width[active])),
            "median_relative_full_width":float(np.median(width[active])),
            "active_kT_max":float(group.kT.to_numpy(float)[active].max()),
        }
    return result


def main() -> None:
    state_chain = validated_state_chain_audit()
    state_chain_hash = sha256(STATE_CHAIN_AUDIT)
    _, fixed_protocol_hash = validate_fixed_challenger_protocol()
    selected=json.loads(SELECTED.read_text()); replicas=json.loads(REPLICAS.read_text())
    gate_state=scientific_gate_state(selected,replicas)
    selected_status=gate_state["selected_status"]
    replica_status=gate_state["replica_status"]
    start_stationarity_pass=gate_state["start_pass"]
    central_stationarity_pass=gate_state["central_pass"]
    replica_stationarity_pass=gate_state["replica_pass"]
    candidate_stationarity_pass=gate_state["overall_pass"]
    long_start_protocol=(int(selected.get(
        "mandatory_same_objective_iterations_per_start",0))>=200_000)
    long_replica_protocol=(int(replicas.get(
        "minimum_cumulative_iterations",0))>=200_000)
    boundary=(json.loads(BOUNDARY_CONFIRMATION.read_text())
              if BOUNDARY_CONFIRMATION.exists() else {})
    legacy_replica_boundary_pass=(boundary.get("status")=="complete"
        and not boundary.get("failed_replica_seeds",[]))
    if not (legacy_replica_boundary_pass or long_replica_protocol):
        raise RuntimeError("replica boundary confirmation is incomplete")
    sensitive=selected.get("sensitive_start_boundary_confirmation",{})
    legacy_start_boundary_pass=(sensitive.get("all_selected_starts_confirmed",False)
                                and not sensitive.get("failed_seeds",[]))
    if not (legacy_start_boundary_pass or long_start_protocol):
        raise RuntimeError("near-boundary selected starts lack fresh confirmation")
    if not np.isclose(float(replicas["selected_strength"]),
                      float(selected["selected_strength"]),
                      rtol=0.0,atol=1e-12):
        raise RuntimeError("start and experimental ensembles use different constraint strengths")
    if not np.isclose(float(replicas["selected_bmax"]),
                      float(selected["selected_bmax"]),
                      rtol=0.0,atol=1e-12):
        raise RuntimeError("start and experimental ensembles use different FNP constraint domains")
    barrier_strength=float(selected.get("fit_quality_barrier_strength",0.0))
    barrier_power=int(selected.get("fit_quality_barrier_power",2))
    if (not np.isclose(float(replicas.get("fit_quality_barrier_strength",0.0)),
                       barrier_strength,rtol=0.0,atol=1e-12)
            or int(replicas.get("fit_quality_barrier_power",2)) != barrier_power):
        raise RuntimeError("start and experimental ensembles use different fit-quality barriers")
    if len(selected["endpoint_tags"]) != 24 or len(set(selected["endpoint_tags"])) != 24:
        raise RuntimeError("final ensemble requires exactly 24 unique start endpoints")
    if (len(replicas["replica_endpoint_tags"]) != 50
            or len(set(replicas["replica_endpoint_tags"])) != 50):
        raise RuntimeError("final ensemble requires exactly 50 unique replica endpoints")
    start_runs=[BASE/"outputs"/tag for tag in selected["endpoint_tags"]]
    replica_runs=[BASE/"outputs"/tag for tag in replicas["replica_endpoint_tags"]]
    central_run=BASE/"outputs"/replicas["central_endpoint_tag"]
    strength=float(selected["selected_strength"]); bmax=float(selected["selected_bmax"])
    central_status=validated_status(
        central_run,strength,bmax,barrier_strength,barrier_power)
    start_statuses=[validated_status(
        run,strength,bmax,barrier_strength,barrier_power) for run in start_runs]
    replica_statuses=[validated_status(
        run,strength,bmax,barrier_strength,barrier_power) for run in replica_runs]
    if central_status.get("replica_seed") is not None:
        raise RuntimeError("stationary central is labeled as an experimental replica")
    if any(status.get("replica_seed") is not None for status in start_statuses):
        raise RuntimeError("start ensemble contains an experimental-replica endpoint")
    status_replica_ids=[status.get("replica_seed") for status in replica_statuses]
    if any(value is None for value in status_replica_ids) or len(set(status_replica_ids)) != 50:
        raise RuntimeError("experimental ensemble lacks 50 unique replica identities")
    start_ids=[int(status["seed"]) for status in start_statuses]
    replica_ids=[int(value) for value in status_replica_ids]
    if start_ids != list(range(303,327)):
        raise RuntimeError("24-start diagnostic lacks exact seeds 303--326 in order")
    if replica_ids != list(range(1001,1051)):
        raise RuntimeError("50-replica diagnostic lacks exact identities 1001--1050 in order")
    global B_GRID
    B_GRID,central=curve(central_run)
    starts=curves_on_common_grid(start_runs,B_GRID,"start")
    reps=curves_on_common_grid(replica_runs,B_GRID,"experimental replica")
    if np.any(starts<=0) or np.any(reps<=0) or np.any(central<=0):
        raise RuntimeError("log-FNP convolution requires positive input curves")
    logc=np.log(central); logs=np.log(starts); logr=np.log(reps)
    start_res=logs-np.median(logs,axis=0)
    replica_res=logr-np.median(logr,axis=0)
    start_centering_residual=float(np.max(np.abs(np.median(start_res,axis=0))))
    replica_centering_residual=float(np.max(np.abs(np.median(replica_res,axis=0))))
    combined=(logc[None,None,:]+start_res[:,None,:]+replica_res[None,:,:]).reshape(
        len(starts)*len(reps),-1)
    start_only=logc[None,:]+start_res
    replica_only=logc[None,:]+replica_res

    combined_keys=[(s,r) for s in start_ids for r in replica_ids]
    start_keys=[(s,0) for s in start_ids]; replica_keys=[(0,r) for r in replica_ids]
    reference=pd.read_csv(REFERENCE_B)
    reference=reference[np.isclose(reference.x,.1)&(
        (np.isclose(reference.Q,7.5)&reference.flavor.astype(str).isin(FLAVORS_B))
        |(np.isclose(reference.Q,10)&reference.flavor.astype(str).isin(FLAVORS_K)))].copy()
    TARGET.mkdir(parents=True,exist_ok=True)
    fnp_bands([
        ("nonuniqueness", start_only),
        ("experimental", replica_only),
        ("combined", combined),
    ]).to_csv(TARGET/"fnp_bands.csv", index=False)
    b_bands=pd.concat([
        bspace_bands(reference,start_only,"nonuniqueness"),
        bspace_bands(reference,replica_only,"experimental"),
        bspace_bands(reference,combined,"combined"),
    ],ignore_index=True)
    b_bands.to_csv(TARGET/"bT_tmd_bands.csv",index=False)

    transform=load_module("final_combined_transform",TRANSFORMER)
    k_longs=[]; k_bands=[]; transform_metadata={}
    for logs_,keys_,name in [(start_only,start_keys,"nonuniqueness"),
                             (replica_only,replica_keys,"experimental"),
                             (combined,combined_keys,"combined")]:
        long,bands,meta=transform_component(reference,logs_,keys_,name,transform)
        k_longs.append(long); k_bands.append(bands)
        transform_metadata[name]=meta
    k_long=pd.concat(k_longs,ignore_index=True); k_bands=pd.concat(k_bands,ignore_index=True)
    k_long.to_csv(TARGET/"kT_tmd_ensemble_long.csv",index=False)
    k_bands.to_csv(TARGET/"kT_tmd_bands.csv",index=False)
    metrics={name:active_width_metrics(k_bands[k_bands.component==name])
             for name in ("experimental","nonuniqueness","combined")}
    failure_reasons=[]
    if not start_stationarity_pass:
        failure_reasons.append("24-start FNP stationarity/fit-preservation gate failed")
    if not central_stationarity_pass:
        failure_reasons.append("central FNP stationarity gate failed")
    if not replica_stationarity_pass:
        failure_reasons.append("experimental-replica FNP stationarity/agreement gate failed")
    summary={
        "status":"complete","selected_strength":selected["selected_strength"],
        "selected_bmax":selected["selected_bmax"],
        "fit_quality_barrier_strength":barrier_strength,
        "fit_quality_barrier_power":barrier_power,
        "start_count":len(starts),"experimental_replica_count":len(reps),
        "combined_member_count":len(combined),
        "source_start_status":selected_status,
        "source_replica_status":replica_status,
        "start_stationarity_and_fit_gate_pass":start_stationarity_pass,
        "central_stationarity_gate_pass":central_stationarity_pass,
        "replica_stationarity_and_agreement_gate_pass":replica_stationarity_pass,
        "candidate_stationarity_gate_pass":candidate_stationarity_pass,
        "state_chain_gate_pass":True,
        "state_chain_audit":str(STATE_CHAIN_AUDIT),
        "state_chain_audit_sha256":state_chain_hash,
        "state_chain_checkpoint_count":state_chain[
            "total_continuation_checkpoint_count"],
        "fixed_challenger_protocol":str(FIXED_PROTOCOL),
        "fixed_challenger_protocol_sha256":fixed_protocol_hash,
        "fixed_fnp_reference":str(FIXED_FNP_REFERENCE),
        "fixed_fnp_reference_sha256":EXPECTED_FNP_REFERENCE_SHA256,
        **fixed_implementation_binding(),
        "scientific_failure_reasons":failure_reasons,
        "diagnostic_only":not candidate_stationarity_pass,
        "promotion_eligible":False,
        "promotion_eligibility_semantics":(
            "ensemble construction never authorizes promotion; a failed stationarity "
            "gate is permanently non-promotable and a passing candidate still requires "
            "the independent final stability and completion audits"),
        "combination_rule":"Cartesian convolution of centered log-FNP start and experimental-replica residual curves about the declared full-reference central",
        "hierarchical_transfer_assumption":(
            "the empirically measured centered 24-start residual distribution is applied "
            "to each experimental replica as an operational separability assumption; "
            "the 1200 Cartesian members are derived products of 24 and 50 marginal "
            "samples, not 1200 independently fitted nested start-by-replica models"
            if candidate_stationarity_pass else
            "the same 24x50 propagation is evaluated only as a diagnostic scale; one or "
            "more FNP stationarity/agreement gates failed, so its q16--q84 envelope is "
            "not a validated uncertainty interval and cannot be promoted"),
        "start_log_residual_pointwise_median_abs_max":start_centering_residual,
        "experimental_log_residual_pointwise_median_abs_max":replica_centering_residual,
        "central_counted_once":True,
        "declared_reference_central_endpoint":central_run.name,
        "joint_nested_start_by_replica_refits_performed":False,
        "independent_sampling_axis_counts":{"optimizer_starts":len(starts),
                                            "experimental_replicas":len(reps)},
        "available_same_replica_cross_optimizer_check_count":int(
            replicas.get("available_cross_optimizer_comparison_count",0)),
        "central_line_rule":"pointwise median of the propagated 24x50 TMD ensemble after b-space construction or finite-b transform; this displayed median is a central curve, not necessarily the separately trained central endpoint or any single realizable model member",
        "interval":"pointwise empirical q16--q84 of the operational hierarchical ensemble",
        "interval_probability_semantics":(
            "only the conditional experimental-replica marginal has the usual "
            "replica interpretation; optimizer-start nonuniqueness has no calibrated "
            "probability measure, so the combined interval is neither a formal 68% "
            "confidence interval nor a one-sigma error band"),
        "transform_settings":{"implementation":str(TRANSFORMER),"tail_mode":"expb2",
                              "b_transform_max":24.0,"n_b_transform":6001,
                              "k_max":4.0,"n_k":401,
                              "end_taper_start_fraction":0.92},
        "transform_metadata_by_component":transform_metadata,
        "metrics":metrics,"contains_legacy_conditional_result":False,
        "production_sources_modified":False,
    }
    (TARGET/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
