#!/usr/bin/env python3
"""Verify the replica-robust full-reference strength on all 24 starts.

Each case begins from its already stationary lambda=300 endpoint, rather than
from another seed or from a stronger-prior fit. Every case receives forty
sequential same-objective float64 LBFGS continuation blocks, each capped at
5,000 iterations, before it can qualify. This 200k label is requested optimizer
capacity, not a claim that PyTorch LBFGS executes every allowed iteration. Ten
genuinely post-anchor 5k-capacity blocks must then have <=2% FNP drift both
block-to-block and relative to the anchor on x=0.1, 0.1<=bT<=4. Fit cost is assessed from the
*unpenalized* chi2 against the lambda=300 source fit; sqrt(2N) for N=329 is the
fit-preservation scale, with the older +3.29 threshold retained only as a
reported sensitivity diagnostic.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from checkpoint_launch_ancestry import (
    build_continuation_command,
    build_launch_receipt,
    classify_launch_ancestry,
    exclusive_checkpoint_launch,
    prepare_launch_receipt,
    receipt_path,
)
from fixed_challenger_protocol import (
    EXPECTED_FNP_REFERENCE_SHA256,
    FNP_REFERENCE as FIXED_FNP_REFERENCE,
    PROTOCOL as FIXED_PROTOCOL,
    fixed_implementation_binding,
    validate_fixed_challenger_protocol,
)


BASE = Path(__file__).resolve().parents[1]
SYSTEMATICS = BASE.parent
ROOT = SYSTEMATICS.parent
UNITARY = SYSTEMATICS / "high_qt_direct_production_benchmark/experimental_unitary_transition"
RUNNER = UNITARY / "scripts/run_production_fnp_stability_control.py"
SOURCE = SYSTEMATICS / "collins_factorization_validity/outputs/rowidfix_stageFT_all_qmax0p20_lam0p50_central_s303"
W_GRID = ROOT / "outputs/v23a_tevatron_plus_lhcb7_fidacc_merged_cache/backend_cache/wpert_v23a_tevatron_plus_lhcb7_fidacc_b160.csv"
REFERENCE = BASE / "summaries/exact_baseline_fnp_median/fnp_median.csv"
BRACKET = BASE / "summaries/full_reference_replica_strength_bracket/summary.json"
STAGED = BASE / "summaries/lambda675_fit_quality_barrier_stress/summary.json"
MINIMUM_SEARCH = BASE / "summaries/minimum_fitbar_constraint_search/summary.json"
SOURCE_ENSEMBLE = BASE / "summaries/selected_reference_method_full24/summary.json"
TARGET = BASE / "summaries/replica_robust_reference_full24"
LAUNCH_RECEIPTS = BASE / "summaries/checkpoint_launch_receipts"
PYTHON = Path("/home/dustin/miniforge3/envs/pdf-fit/bin/python")
SEEDS = tuple(range(303, 327))
BMAX = 4.0
FNP_DRIFT_GATE = 0.02
SENSITIVITY = (0.0025, 0.005, 0.01, 0.02)
REQUIRED_CONSECUTIVE = 10
# Every promoted start must see the same long unchanged-objective exposure as
# the hard-replica discriminator.  The former early two-block exit (often at
# 15k) is invalidated by observed reversals after 150k.  Forty blocks establish
# the mandatory anchor and twenty more leave room for a 50k post-anchor quiet
# window plus a full replacement window after a late sensitivity event.
MANDATORY_CHUNKS = 40
MAX_CHUNKS = 60
N_DATA = 329
CHI2_NATURAL_SCALE = float(np.sqrt(2 * N_DATA))
REQUIRED_ENDPOINT_FILES = (
    "fit_status.json",
    "model_state.pt",
    "dataset_norms.csv",
    "fnp_grid.csv",
)


def token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def atomic_write_text(path: Path, content: str) -> None:
    """Atomically publish a manifest without exposing a partial JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: dict) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2) + "\n")


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    """Atomically publish a progress ledger reconstructed from checkpoints."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            frame.to_csv(stream, index=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def exact_number(observed, expected: float) -> bool:
    try:
        value = float(observed)
    except (TypeError, ValueError):
        return False
    return (np.isfinite(value)
            and np.isclose(value, float(expected),
                           rtol=0.0, atol=1.0e-12))


def exact_bool(value, label: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise RuntimeError(f"{label} is not an explicit boolean")


def terminal_capacity_metadata(
        records: list[dict], failed: list[int]) -> dict:
    """Prove that every nonpassing start exhausted the declared 300k horizon."""
    terminal_by_seed = {}
    for seed in SEEDS:
        capacities = [int(row["cumulative_lbfgs_iterations"])
                      for row in records if int(row["seed"]) == int(seed)]
        if not capacities:
            raise RuntimeError(f"missing continuation ledger for seed {seed}")
        terminal_by_seed[int(seed)] = max(capacities)
    failed_by_seed = {
        int(seed): terminal_by_seed[int(seed)] for seed in failed
    }
    exhausted = all(
        capacity == MAX_CHUNKS * 5000
        for capacity in failed_by_seed.values())
    return {
        "terminal_requested_capacity_by_seed": terminal_by_seed,
        "failed_terminal_requested_capacity_by_seed": failed_by_seed,
        "full_requested_capacity_per_nonpassing_start": MAX_CHUNKS * 5000,
        "failed_starts_exhausted_full_requested_horizon": exhausted,
    }


def require_cached_endpoint(
        run: Path, *, seed: int, strength: float, barrier_strength: float,
        barrier_power: int, expected_barrier_ceiling: float | None,
        expected_initial_state: Path, expected_initial_norms: Path,
        expected_launch_receipt: dict,
        allow_legacy_without_launch_receipt: bool) -> tuple[dict, dict]:
    """Fail closed before reusing a same-objective continuation checkpoint."""
    missing = [name for name in REQUIRED_ENDPOINT_FILES
               if not (run / name).is_file()
               or (run / name).stat().st_size == 0]
    if missing:
        raise RuntimeError(
            f"cached endpoint {run.name} lacks required files: {missing}")
    status = json.loads((run / "fit_status.json").read_text())
    reference = status["regularization"]["fnp_reference_distance"]
    barrier = status["regularization"]["fit_quality_barrier"]
    nonzero_regularizers = {
        name for name, spec in status["regularization"].items()
        if isinstance(spec, dict) and "lambda" in spec
        and not exact_number(spec["lambda"], 0.0)
    }
    expected_nonzero = {"fnp_reference_distance"}
    if barrier_strength > 0:
        expected_nonzero.add("fit_quality_barrier")
    complexity = status.get("model_complexity", {})
    profile = status.get("point_profile", {})
    initial_state = status.get("initial_state")
    if not (
            exact_bool(status.get("convergence_gate_pass"),
                       "convergence_gate_pass")
            and not exact_bool(status.get("production_state_modified"),
                               "production_state_modified")
            and Path(status["source_production"]).resolve() == SOURCE.resolve()
            and Path(status["w_grid"]).resolve() == W_GRID.resolve()
            and int(status.get("seed", -1)) == seed
            and status.get("replica_seed") is None
            and initial_state is not None
            and Path(initial_state).resolve() == expected_initial_state.resolve()
            and exact_number(status.get("initial_relative_parameter_perturbation"), 0.0)
            and int(status.get("row_count", -1)) == N_DATA
            and int(status.get("max_epochs", -1)) == 0
            and int(status.get("lbfgs", {}).get("max_iter", -1)) == 5000
            and exact_number(reference.get("lambda"), strength)
            and Path(reference.get("target_csv", "")).resolve()
                == REFERENCE.resolve()
            and exact_number(reference.get("b_min"), 0.1)
            and exact_number(reference.get("b_max"), BMAX)
            and exact_number(barrier.get("lambda"), barrier_strength)
            and int(barrier.get("power", -1)) == barrier_power
            and (barrier_strength <= 0
                 or (expected_barrier_ceiling is not None
                     and exact_number(barrier.get("ceiling_total_chi2"),
                                      expected_barrier_ceiling)))
            and nonzero_regularizers == expected_nonzero
            and exact_number(status["regularization"][
                "likelihood_weight"]["value"], 1.0)
            and not exact_bool(profile.get("enabled"),
                               "point_profile.enabled")
            and exact_number(profile.get("lambda_per_row"), 0.0)
            and status.get("model_constraint", {}).get("kind") == "none"
            and int(complexity.get("np_width", -1)) == 48
            and int(complexity.get("np_cond_width", -1)) == 32
            and int(complexity.get("np_blocks", -1)) == 3
            and complexity.get("global_spline_nx") is None
            and complexity.get("global_spline_nb") is None
            and int(complexity.get("distill_accepted_steps", -1)) == 0
            and int(complexity.get("distill_prediction_steps", -1)) == 0):
        raise RuntimeError(
            f"cached endpoint objective/provenance mismatch: {run.name}")
    # Checkpoints created by the already-running pre-receipt controller cannot
    # acquire retrospective launch proof. They are explicitly classified for a
    # terminal current-byte seal. Every newly launched repair block is required
    # to carry the prospective immutable receipt.
    ancestry = classify_launch_ancestry(
        LAUNCH_RECEIPTS, run, run.name, expected_launch_receipt,
        allow_legacy_without_receipt=allow_legacy_without_launch_receipt)
    if (Path(ancestry["parent_state_path"]).resolve()
            != expected_initial_state.resolve()
            or Path(ancestry["parent_norms_path"]).resolve()
            != expected_initial_norms.resolve()):
        raise RuntimeError(
            f"cached endpoint paired restart ancestry mismatch: {run.name}")
    return status, ancestry


def vector(run: Path) -> np.ndarray:
    frame = pd.read_csv(run / "fnp_grid.csv")
    mask = np.isclose(frame.x, .1) & (frame.bT >= .1) & (frame.bT <= BMAX)
    return frame.loc[mask, "F_NP"].to_numpy()


def load_extent_tools():
    path = BASE / "scripts/scan_reference_distance_extent.py"
    spec = importlib.util.spec_from_file_location("reference_extent_tools", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    _, fixed_protocol_hash = validate_fixed_challenger_protocol()
    parser = argparse.ArgumentParser()
    parser.add_argument("--strength", type=float)
    parser.add_argument("--fit-quality-barrier-strength", type=float)
    parser.add_argument("--fit-quality-barrier-power", type=int, default=2)
    parser.add_argument("--strongest-failing-strength", type=float)
    args = parser.parse_args()
    bracket = json.loads(BRACKET.read_text())
    if bracket["status"] != "complete":
        raise RuntimeError("replica-stress strength bracket is incomplete")
    staged = json.loads(STAGED.read_text()) if STAGED.exists() else None
    if args.strength is not None:
        strength = float(args.strength)
        barrier_strength = float(args.fit_quality_barrier_strength or 0.0)
        barrier_power = int(args.fit_quality_barrier_power)
    elif staged is not None and staged.get("status") == "complete":
        strength = float(staged["reference_strength"])
        barrier_strength = float(staged["fit_quality_barrier_strength"])
        barrier_power = int(staged["fit_quality_barrier_power"])
    else:
        strength = float(bracket["weakest_tested_passing_strength"])
        barrier_strength = 0.0
        barrier_power = 2
    source_summary = json.loads(SOURCE_ENSEMBLE.read_text())
    if not source_summary["all_starts_fnp_plateaued_and_fit_preserved"]:
        raise RuntimeError("lambda=300 source ensemble is incomplete")
    sources = [BASE / "outputs" / tag for tag in source_summary["endpoint_tags"]]
    if len(sources) != len(SEEDS):
        raise RuntimeError("expected exactly 24 source endpoints")

    TARGET.mkdir(parents=True, exist_ok=True)
    # Do not leave a superseded completed generation visible while a stronger
    # selected strength is being verified.  The final summary below replaces
    # this manifest only after every start has reached a terminal endpoint.
    atomic_write_json(TARGET / "summary.json", {
        "status": "in_progress",
        "fixed_challenger_protocol": str(FIXED_PROTOCOL),
        "fixed_challenger_protocol_sha256": fixed_protocol_hash,
        "fixed_fnp_reference": str(FIXED_FNP_REFERENCE),
        "fixed_fnp_reference_sha256": EXPECTED_FNP_REFERENCE_SHA256,
        **fixed_implementation_binding(),
        "selected_strength": strength,
        "selected_bmax": BMAX,
        "expected_member_count": len(SEEDS),
        "completed_member_count": 0,
        "production_sources_modified": False,
    })
    # Any summary derived from an older selected strength is no longer a
    # terminal result once this generation begins. Replace only the summary
    # manifests; immutable endpoint data remain untouched for provenance.
    for relative in (
        "replica_robust_constraint_scale/summary.json",
        "selected_reference_start_boundary_confirmation/summary.json",
        "selected_reference_start_sensitivity_confirmation/summary.json",
        "selected_reference_central_replicas/summary.json",
        "selected_reference_boundary_confirmation/summary.json",
        "final_combined_tmd_ensemble/summary.json",
        "final_combined_ensemble_stability/summary.json",
        "lambda600_start_chain_audit/summary.json",
        "lambda600_state_chain_audit/summary.json",
        "lambda600_postfit_tail_transform_audit/summary.json",
        "lambda600_nested_start_replica_interaction/summary.json",
        "lambda600_final_directional_envelope/summary.json",
        "final_fig2_fig6/summary.json",
        "lambda600_vs_lambda1_diagnostic/summary.json",
        "lambda600_terminal_evidence/summary.json",
        "lambda600_like_for_like_completion/summary.json",
        "campaign_completion_audit/summary.json",
        "final_study_report/summary.json",
    ):
        downstream = BASE / "summaries" / relative
        if downstream.exists():
            atomic_write_json(downstream, {
                "status": "superseded_by_in_progress_verification",
                "selected_strength": strength,
                "upstream": str(TARGET / "summary.json"),
                "production_sources_modified": False,
            })
    # Snapshot the exact no-receipt cache population at updated-controller
    # entry. Only these checkpoints may use the disclosed legacy path; any
    # child appearing later must have been launched under a prospective receipt.
    barrier_tag = (f"_fitbar_p{barrier_power}_mu{token(barrier_strength)}"
                   if barrier_strength > 0 else "")
    legacy_pre_receipt_admission_tags = set()
    for seed in SEEDS:
        for chunk in range(1, MAX_CHUNKS + 1):
            cumulative = 5000 * chunk
            tag = (
                f"fullref_replica_robust_lam{token(strength)}{barrier_tag}"
                f"_b4_s{seed}_polish64_{cumulative}")
            run = BASE / "outputs" / tag
            if ((run / "fit_status.json").is_file()
                    and not receipt_path(LAUNCH_RECEIPTS, tag).exists()):
                legacy_pre_receipt_admission_tags.add(tag)

    records: list[dict] = []
    endpoints: list[Path] = []
    failed: list[int] = []
    for seed, initial in zip(SEEDS, sources, strict=True):
        initial_status = json.loads((initial / "fit_status.json").read_text())
        source_chi2 = float(initial_status["final"]["unpenalized_total_chi2"])
        previous = initial
        consecutive = 0
        passed = False
        sensitivity_active = False
        fresh_after_sensitivity = 0
        stationarity_anchor = None
        stationarity_anchor_iterations = None
        for chunk in range(1, MAX_CHUNKS + 1):
            cumulative = 5000 * chunk
            barrier_tag = (f"_fitbar_p{barrier_power}_mu{token(barrier_strength)}"
                           if barrier_strength > 0 else "")
            tag = (f"fullref_replica_robust_lam{token(strength)}{barrier_tag}_b4_s{seed}_"
                   f"polish64_{cumulative}")
            target = BASE / "outputs" / tag
            parent_state = previous / "model_state.pt"
            parent_norms = previous / "dataset_norms.csv"
            barrier_ceiling = (
                source_chi2 + CHI2_NATURAL_SCALE
                if barrier_strength > 0 else None)
            command = build_continuation_command(
                python=PYTHON, runner=RUNNER, seed=seed,
                source_production=SOURCE, w_grid=W_GRID,
                output_root=BASE / "outputs", child_tag=tag,
                parent_state=parent_state, parent_norms=parent_norms,
                reference_strength=strength, reference_csv=REFERENCE,
                reference_bmin=0.1, reference_bmax=BMAX,
                barrier_strength=barrier_strength,
                barrier_power=barrier_power,
                barrier_ceiling=barrier_ceiling, replica_seed=None)
            expected_receipt = build_launch_receipt(
                receipt_root=LAUNCH_RECEIPTS, child_output=target,
                child_tag=tag, parent_state=parent_state,
                parent_norms=parent_norms, fit_seed=seed,
                replica_seed=None, command=command,
                reference_strength=strength, reference_bmin=0.1,
                reference_bmax=BMAX, barrier_strength=barrier_strength,
                barrier_power=barrier_power,
                barrier_ceiling=barrier_ceiling)
            with exclusive_checkpoint_launch(LAUNCH_RECEIPTS, tag):
                if not (target / "fit_status.json").exists():
                    prepare_launch_receipt(
                        LAUNCH_RECEIPTS, target, tag, expected_receipt)
                    with (BASE / "logs" / f"{tag}.log").open("w") as stream:
                        validate_fixed_challenger_protocol()
                        subprocess.run(
                            command, stdout=stream,
                            stderr=subprocess.STDOUT, check=True)
                status, launch_ancestry = require_cached_endpoint(
                    target, seed=seed, strength=strength,
                    barrier_strength=barrier_strength,
                    barrier_power=barrier_power,
                    expected_barrier_ceiling=barrier_ceiling,
                    expected_initial_state=parent_state,
                    expected_initial_norms=parent_norms,
                    expected_launch_receipt=expected_receipt,
                    allow_legacy_without_launch_receipt=(
                        tag in legacy_pre_receipt_admission_tags))
            drift = float(np.max(np.abs(vector(target) - vector(previous)) /
                                 np.maximum(vector(previous), .05)))
            if chunk == MANDATORY_CHUNKS:
                stationarity_anchor = vector(target)
                stationarity_anchor_iterations = cumulative
            eligible = chunk > MANDATORY_CHUNKS
            tested_anchor_iterations = stationarity_anchor_iterations
            window_drift = (float(np.max(
                np.abs(vector(target) - stationarity_anchor) /
                np.maximum(stationarity_anchor, .05)))
                if eligible and stationarity_anchor is not None else None)
            window_quiet = (window_drift is not None
                            and window_drift <= FNP_DRIFT_GATE)
            # A qualifying block must be quiet locally *and* remain inside the
            # anchor envelope. This prevents a sequence of small same-direction
            # steps from accumulating into a moving endpoint.
            quiet = eligible and drift <= FNP_DRIFT_GATE and window_quiet
            if quiet:
                consecutive += 1
            elif eligible:
                # A late descent is allowed to establish a new candidate
                # endpoint.  It must then remain quiet for a wholly new 50k
                # window; do not reject it merely for differing from the
                # arbitrary 200k checkpoint.
                stationarity_anchor = vector(target)
                stationarity_anchor_iterations = cumulative
                consecutive = 0
                sensitivity_active = False
                fresh_after_sensitivity = 0
            else:
                consecutive = 0
            unpenalized = float(status["final"]["unpenalized_total_chi2"])
            fit_delta = unpenalized - source_chi2
            triggered_now = (quiet and drift >= SENSITIVITY[-2]
                             and not sensitivity_active)
            if triggered_now:
                sensitivity_active = True
                fresh_after_sensitivity = 0
            elif sensitivity_active:
                fresh_after_sensitivity = fresh_after_sensitivity + 1 if quiet else 0
                passed = (fresh_after_sensitivity >= REQUIRED_CONSECUTIVE
                          and fit_delta <= CHI2_NATURAL_SCALE and window_quiet)
            else:
                passed = (consecutive >= REQUIRED_CONSECUTIVE
                          and fit_delta <= CHI2_NATURAL_SCALE and window_quiet)
            records.append({
                "strength": strength, "seed": seed,
                "cumulative_lbfgs_iterations": cumulative,
                "requested_lbfgs_max_iterations_this_block": 5000,
                "executed_lbfgs_closure_evaluations_this_block": int(
                    status["lbfgs"]["closure_evaluations"]),
                "initial_state_path": launch_ancestry["parent_state_path"],
                "initial_state_sha256": launch_ancestry[
                    "parent_state_sha256"],
                "initial_norms_path": launch_ancestry["parent_norms_path"],
                "initial_norms_sha256": launch_ancestry[
                    "parent_norms_sha256"],
                "launch_ancestry_kind": launch_ancestry["kind"],
                "launch_receipt_path": launch_ancestry["path"],
                "launch_receipt_sha256": launch_ancestry["sha256"],
                "launch_argv_sha256": launch_ancestry["argv_sha256"],
                "fnp_drift_from_previous_chunk": drift,
                "eligible_post_mandatory_confirmation": eligible,
                "post_mandatory_window_fnp_drift": window_drift,
                "stationarity_window_anchor_iterations": tested_anchor_iterations,
                "next_stationarity_window_anchor_iterations":
                    stationarity_anchor_iterations,
                "passes_post_mandatory_window_drift_2pct": window_quiet,
                "passes_drift_0p25pct": chunk > 1 and drift <= SENSITIVITY[0],
                "passes_drift_0p5pct": chunk > 1 and drift <= SENSITIVITY[1],
                "passes_drift_1pct": chunk > 1 and drift <= SENSITIVITY[2],
                "passes_drift_2pct": chunk > 1 and drift <= SENSITIVITY[3],
                "consecutive_quiet_blocks": consecutive,
                "sensitivity_confirmation_triggered": sensitivity_active,
                "fresh_quiet_blocks_after_sensitivity_trigger": fresh_after_sensitivity,
                "source_unpenalized_total_chi2": source_chi2,
                "unpenalized_total_chi2": unpenalized,
                "unpenalized_chi2_delta": fit_delta,
                "passes_natural_chi2_scale": fit_delta <= CHI2_NATURAL_SCALE,
                "passes_legacy_delta3p29_sensitivity": fit_delta <= 3.29,
                "stationarity_and_fit_pass": passed, "tag": tag,
            })
            atomic_write_csv(TARGET / "runs.csv", pd.DataFrame(records))
            previous = target
            if passed:
                break
            # A start that can no longer accumulate the required ten quiet
            # blocks inside the declared 300k window is already known to be
            # non-promotable, but it is *not* yet a complete endpoint for the
            # requested non-uniqueness distribution.  Continue every such
            # scientific failure through MAX_CHUNKS so failure cannot truncate
            # the evidence collection or bias the terminal start ensemble.
        endpoints.append(previous)
        if not passed:
            failed.append(seed)
        # Persist coverage progress without claiming completion. This also
        # makes interruption/resumption state unambiguous to external audits.
        validate_fixed_challenger_protocol()
        atomic_write_json(TARGET / "summary.json", {
            "status": "in_progress",
            "fixed_challenger_protocol": str(FIXED_PROTOCOL),
            "fixed_challenger_protocol_sha256": fixed_protocol_hash,
            "fixed_fnp_reference": str(FIXED_FNP_REFERENCE),
            "fixed_fnp_reference_sha256": EXPECTED_FNP_REFERENCE_SHA256,
            **fixed_implementation_binding(),
            "selected_strength": strength,
            "selected_bmax": BMAX,
            "expected_member_count": len(SEEDS),
            "completed_member_count": len(endpoints),
            "failed_seeds_so_far": failed,
            "endpoint_tags_so_far": [run.name for run in endpoints],
            "production_sources_modified": False,
        })

    curves = np.asarray([vector(run) for run in endpoints])
    median = np.median(curves, axis=0)
    q16, q84 = np.quantile(curves, [.16, .84], axis=0)
    scale = np.maximum(median, .05)
    extent_tools = load_extent_tools()
    k_bands, _ = extent_tools.project_kspace(endpoints)
    atomic_write_csv(TARGET / "kspace_nonuniqueness_bands.csv", k_bands)
    k_widths = {}
    for flavor, group in k_bands[k_bands.kT <= 2.25].groupby("flavor"):
        group = group.sort_values("kT")
        center = group["median"].to_numpy(float)
        active_k = center > .05 * np.max(center)
        width = ((group.q84 - group.q16).to_numpy(float) /
                 np.maximum(center, 1e-300))
        k_widths[str(flavor)] = float(np.max(width[active_k]))
    fit_deltas = [row["unpenalized_chi2_delta"] for row in records
                  if row["tag"] in {run.name for run in endpoints}]
    capacity_metadata = terminal_capacity_metadata(records, failed)
    launch_receipt_count = sum(
        row["launch_ancestry_kind"] == "launch_time_content_receipt"
        for row in records)
    legacy_launch_count = len(records) - launch_receipt_count
    legacy_used_tags = sorted({
        str(row["tag"]) for row in records
        if row["launch_ancestry_kind"] != "launch_time_content_receipt"
    })
    fnp_full_range = float(np.max((curves.max(0) - curves.min(0)) / scale))
    fnp_central68_width = float(np.max((q84 - q16) / scale))
    minimum = (json.loads(MINIMUM_SEARCH.read_text())
               if MINIMUM_SEARCH.exists() else None)
    if args.strongest_failing_strength is not None:
        strongest_failing = args.strongest_failing_strength
    elif (minimum is not None and minimum.get("status") == "complete"
          and np.isclose(float(minimum["selected_weakest_surviving_strength"]),
                         strength)):
        strongest_failing = minimum[
            "strongest_rejected_strength_below_selection"]
    elif barrier_strength > 0:
        strongest_failing = staged.get("strongest_tested_failing_strength", 637.5)
    else:
        strongest_failing = bracket["strongest_tested_failing_strength"]
    validate_fixed_challenger_protocol()
    summary = {
        "status": "complete" if not failed else "verification_failed",
        "fixed_challenger_protocol": str(FIXED_PROTOCOL),
        "fixed_challenger_protocol_sha256": fixed_protocol_hash,
        "fixed_fnp_reference": str(FIXED_FNP_REFERENCE),
        "fixed_fnp_reference_sha256": EXPECTED_FNP_REFERENCE_SHA256,
        **fixed_implementation_binding(),
        "selected_strength": strength, "selected_bmax": BMAX,
        "fit_quality_barrier_strength": barrier_strength,
        "fit_quality_barrier_power": barrier_power,
        "staged_prescription": barrier_strength > 0,
        "selection_fail_pass_bracket": {
            "strongest_failing": strongest_failing,
            "weakest_passing": strength,
        },
        "member_count": len(SEEDS),
        "all_starts_fnp_plateaued_and_fit_preserved": not failed,
        "failed_seeds": failed,
        **capacity_metadata,
        "readiness_fnp_drift_gate": FNP_DRIFT_GATE,
        "required_consecutive_quiet_blocks": REQUIRED_CONSECUTIVE,
        # Backward-compatible legacy key: its value is requested LBFGS
        # capacity, not measured executed optimizer iterations.
        "mandatory_same_objective_iterations_per_start": MANDATORY_CHUNKS * 5000,
        "mandatory_same_objective_continuation_blocks_per_start": MANDATORY_CHUNKS,
        "mandatory_same_objective_requested_lbfgs_capacity_per_start": MANDATORY_CHUNKS * 5000,
        "iteration_accounting": "cumulative labels are requested LBFGS max-iteration capacity; each runs.csv row records actual closure evaluations separately",
        "fit_gate": f"unpenalized chi2 delta <= sqrt(2*{N_DATA})",
        "fit_gate_value": CHI2_NATURAL_SCALE,
        "max_endpoint_unpenalized_chi2_delta": float(max(fit_deltas)),
        "fnp_width_metric_definition": "maximum over x=0.1 and 0.1<=bT<=4 of pointwise width divided by max(pointwise median,0.05); no active-mask points are omitted",
        "max_endpoint_fnp_full_range_selected_domain_floor_normalized": fnp_full_range,
        "max_endpoint_fnp_central68_width_selected_domain_floor_normalized": fnp_central68_width,
        # Backward-compatible aliases; the definition above is authoritative.
        "max_endpoint_fnp_full_range_active": fnp_full_range,
        "max_endpoint_fnp_central68_width_active": fnp_central68_width,
        "candidate_nonuniqueness_fig6_widths": k_widths,
        "endpoint_tags": [run.name for run in endpoints],
        "launch_time_content_receipt_checkpoint_count": launch_receipt_count,
        "legacy_pre_receipt_checkpoint_count": legacy_launch_count,
        "legacy_pre_receipt_admission_tag_count": len(
            legacy_pre_receipt_admission_tags),
        "legacy_pre_receipt_admission_tags": sorted(
            legacy_pre_receipt_admission_tags),
        "legacy_pre_receipt_used_tags": legacy_used_tags,
        "legacy_checkpoint_ancestry_semantics": (
            "pre-receipt checkpoints retain path-level fit_status ancestry and "
            "must be frozen by the terminal precentral current-byte seal; that "
            "seal does not retroactively prove the bytes used at launch"),
        "production_sources_modified": False,
    }
    atomic_write_json(TARGET / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
