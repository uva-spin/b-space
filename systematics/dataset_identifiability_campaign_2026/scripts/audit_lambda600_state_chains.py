#!/usr/bin/env python3
"""Audit every state transition used by the final lambda=600 ensemble.

This is a provenance audit, not another scientific selection gate.  It proves
that the 24 independent-start endpoints are uninterrupted continuations of the
ordered lambda=300 sources, that the declared central endpoint descends from
its declared start initializer, and that all 50 pseudo-data replicas descend
from that central endpoint.  Every 5k checkpoint must use the exact selected
objective and contain the complete restart/evaluation artifact set.

The fit directories are read-only inputs.  The only output is a deterministic
campaign-local manifest whose hash is bound by the downstream completion
audit.
"""

from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from checkpoint_launch_ancestry import (
    build_continuation_command,
    build_launch_receipt,
    canonical_json_sha256,
    classify_launch_ancestry,
    immutable_create_or_validate_json,
    sha256 as receipt_sha256,
)
from fixed_challenger_protocol import (
    EXPECTED_FNP_REFERENCE_SHA256,
    FNP_REFERENCE as FIXED_FNP_REFERENCE,
    EXPECTED_IMPLEMENTATION_SHA256,
    IMPLEMENTATION_FILES as FIXED_IMPLEMENTATION_FILES,
    PROTOCOL as FIXED_PROTOCOL,
    fixed_implementation_binding,
    require_fixed_implementation_binding,
    validate_fixed_challenger_protocol,
)


BASE = Path(__file__).resolve().parents[1]
SYSTEMATICS = BASE.parent
ROOT = SYSTEMATICS.parent
SUM = BASE / "summaries"
OUT = BASE / "outputs"
SOURCE_PRODUCTION = (
    SYSTEMATICS
    / "collins_factorization_validity/outputs/"
      "rowidfix_stageFT_all_qmax0p20_lam0p50_central_s303"
)
UNITARY = (
    SYSTEMATICS
    / "high_qt_direct_production_benchmark/experimental_unitary_transition"
)
RUNNER = UNITARY / "scripts/run_production_fnp_stability_control.py"
PYTHON = Path("/home/dustin/miniforge3/envs/pdf-fit/bin/python")
W_GRID = (
    ROOT
    / "outputs/v23a_tevatron_plus_lhcb7_fidacc_merged_cache/"
      "backend_cache/wpert_v23a_tevatron_plus_lhcb7_fidacc_b160.csv"
)
REFERENCE = SUM / "exact_baseline_fnp_median/fnp_median.csv"
EVEN_SOURCE_REFERENCE = SUM / "crossfit_reference_evenref/fnp_median.csv"
ODD_SOURCE_REFERENCE = SUM / "crossfit_reference_oddref/fnp_median.csv"
START_SOURCE_ENSEMBLE = SUM / "selected_reference_method_full24/summary.json"
STARTS = SUM / "replica_robust_reference_full24/summary.json"
START_LEDGER = SUM / "replica_robust_reference_full24/runs.csv"
REPLICAS = SUM / "selected_reference_central_replicas/summary.json"
REPLICA_LEDGER = SUM / "selected_reference_central_replicas/runs.csv"
TARGET = SUM / "lambda600_state_chain_audit"
START_ONLY_TARGET = SUM / "lambda600_start_chain_audit"
START_SEAL_ROOT = START_ONLY_TARGET / "current_byte_seals"
LAUNCH_RECEIPTS = SUM / "checkpoint_launch_receipts"
SEEDS = tuple(range(303, 327))
REPLICA_SEEDS = tuple(range(1001, 1051))
REQUIRED_ENDPOINT_FILES = (
    "fit_status.json",
    "model_state.pt",
    "dataset_norms.csv",
    "fnp_grid.csv",
    "accepted_predictions.csv",
)
N_DATA = 329
STRENGTH = 600.0
BMAX = 4.0
BARRIER_STRENGTH = 100.0
BARRIER_POWER = 2
CENTRAL_REQUESTED_CAPACITY = 300_000
MINIMUM_CUMULATIVE_ITERATIONS = 200_000
MAX_CHECKPOINTS = 60
FNP_DRIFT_GATE = 0.02
FNP_DRIFT_SENSITIVITY = (0.0025, 0.005, 0.01, 0.02)
REQUIRED_CONSECUTIVE_BLOCKS = 10
DIRECT_LONG_EVIDENCE = (
    SUM / "fitbar_candidate_lam600p0_long_horizon/summary.json")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def exact_number(observed: object, expected: float) -> bool:
    try:
        value = float(observed)
    except (TypeError, ValueError):
        return False
    return (np.isfinite(value)
            and np.isclose(value, float(expected), rtol=0.0, atol=1.0e-12))


def explicit_bool(value: object, label: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise RuntimeError(f"{label} is not an explicit boolean")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(
            path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def endpoint_evidence_digest(run: Path) -> str:
    """Hash the complete required artifact set into one stable run digest."""
    digest = hashlib.sha256()
    for name in REQUIRED_ENDPOINT_FILES:
        path = run / name
        require(path.is_file() and path.stat().st_size > 0,
                f"endpoint {run.name} lacks required artifact: {name}")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def stable_text_digest(values: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


@lru_cache(maxsize=None)
def expected_fit_target_evidence(
        replica_seed: int | None) -> tuple[tuple[tuple[str, str], ...],
                                           tuple[str, ...], str]:
    """Recreate one pseudo-data identity exactly as the isolated runner does."""
    accepted = pd.read_csv(SOURCE_PRODUCTION / "predictions.csv")
    target = accepted.target_used.to_numpy(float).copy()
    if replica_seed is not None:
        rng = np.random.default_rng(replica_seed)
        for _, subset in accepted.groupby("dataset", sort=False):
            indices = subset.index.to_numpy()
            norm_rel = float(subset["norm_rel_used"].iloc[0])
            target[indices] *= 1.0 + rng.normal() * norm_rel
        target += (rng.normal(size=len(accepted))
                   * accepted.sigma_uncorr.to_numpy(float))
    # The runner writes with pandas' default float formatter. Comparing those
    # exact tokens avoids false failures from a second CSV float parse while
    # still binding the full deterministic pseudo-data realization.
    tokens = tuple(pd.DataFrame({"fit_target": target}).to_csv(
        index=False).splitlines()[1:])
    identities = tuple(zip(
        accepted.dataset.astype(str), accepted.row_id.astype(str), strict=True))
    digest = stable_text_digest(tuple(
        f"{dataset}\0{row_id}\0{token}"
        for (dataset, row_id), token in zip(identities, tokens, strict=True)))
    return identities, tokens, digest


def validate_fit_target(run: Path, replica_seed: int | None) -> str:
    """Require the artifact contents, not merely its declared replica seed."""
    path = run / "accepted_predictions.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames is not None
                and {"dataset", "row_id", "fit_target"}.issubset(
                    reader.fieldnames),
                f"accepted-prediction schema is incomplete: {run.name}")
        rows = list(reader)
    observed_identities = tuple(
        (str(row["dataset"]), str(row["row_id"])) for row in rows)
    observed_tokens = tuple(str(row["fit_target"]) for row in rows)
    expected_identities, expected_tokens, expected_digest = (
        expected_fit_target_evidence(replica_seed))
    require(observed_identities == expected_identities,
            f"fit-target row identity/order mismatch: {run.name}")
    require(observed_tokens == expected_tokens,
            f"fit-target realization mismatch: {run.name}")
    observed_digest = stable_text_digest(tuple(
        f"{dataset}\0{row_id}\0{token}"
        for (dataset, row_id), token in zip(
            observed_identities, observed_tokens, strict=True)))
    require(observed_digest == expected_digest,
            f"fit-target digest mismatch: {run.name}")
    return observed_digest


def fnp_curve(run: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(run / "fnp_grid.csv")
    require({"x", "bT", "F_NP"}.issubset(frame.columns),
            f"FNP grid schema is incomplete: {run.name}")
    frame = frame[np.isclose(frame.x, 0.1) & frame.bT.between(
        0.1, BMAX)].sort_values("bT")
    b = frame.bT.to_numpy(float)
    values = frame.F_NP.to_numpy(float)
    require(len(b) >= 3 and len(np.unique(b)) == len(b)
            and np.all(np.diff(b) > 0.0)
            and np.all(np.isfinite(b)) and np.all(np.isfinite(values))
            and np.all(values > 0.0),
            f"invalid x=0.1 FNP grid: {run.name}")
    return b, values


def float_array_digest(values: np.ndarray) -> str:
    canonical = np.ascontiguousarray(values, dtype="<f8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def central_initializer_selection(start_tags: list[str]) -> dict:
    """Independently recompute the declared 24-member median-nearest medoid."""
    require(len(start_tags) == len(SEEDS)
            and len(set(start_tags)) == len(SEEDS),
            "central selection requires exactly 24 unique start tags")
    grids = []
    curves = []
    for tag in start_tags:
        b, values = fnp_curve(OUT / tag)
        grids.append(b)
        curves.append(values)
    reference_grid = grids[0]
    require(all(grid.shape == reference_grid.shape and np.array_equal(
                    grid, reference_grid) for grid in grids[1:]),
            "24-start FNP curves do not share one exact ordered b grid")
    array = np.asarray(curves, dtype=float)
    pointwise_median = np.median(array, axis=0)
    distances = np.mean((array - pointwise_median[None, :]) ** 2, axis=1)
    require(np.all(np.isfinite(distances)),
            "central initializer distances are non-finite")
    selected_index = int(np.argmin(distances))
    return {
        "rule": (
            "minimum unweighted mean squared direct-FNP distance to the "
            "pointwise 24-start median"),
        "x": 0.1,
        "b_min": 0.1,
        "b_max": BMAX,
        "space": "direct_F_NP",
        "member_count": len(start_tags),
        "ordered_start_tags": start_tags,
        "ordered_squared_distances": [float(value) for value in distances],
        "selected_index": selected_index,
        "selected_tag": start_tags[selected_index],
        "b_grid_sha256": float_array_digest(reference_grid),
        "pointwise_median_sha256": float_array_digest(pointwise_median),
    }


def load_status(run: Path) -> dict:
    endpoint_evidence_digest(run)
    return json.loads((run / "fit_status.json").read_text())


def validate_endpoint(
        run: Path, *, strength: float, expected_reference: Path,
        barrier_strength: float, barrier_power: int, seed: int,
        replica_seed: int | None, expected_barrier_ceiling: float | None,
        expected_initial_state: Path | None,
        continuation_checkpoint: bool) -> dict:
    """Validate one endpoint's exact objective, identity, and ancestry."""
    status = load_status(run)
    reference = status["regularization"]["fnp_reference_distance"]
    barrier = status["regularization"]["fit_quality_barrier"]
    nonzero_regularizers = {
        name for name, spec in status["regularization"].items()
        if isinstance(spec, dict) and "lambda" in spec
        and not exact_number(spec["lambda"], 0.0)
    }
    expected_nonzero = {"fnp_reference_distance"}
    if barrier_strength > 0.0:
        expected_nonzero.add("fit_quality_barrier")
    observed_replica = status.get("replica_seed")
    replica_matches = (
        replica_seed is None and observed_replica is None
    ) or (
        replica_seed is not None and observed_replica is not None
        and int(observed_replica) == int(replica_seed)
    )
    complexity = status.get("model_complexity", {})
    profile = status.get("point_profile", {})
    require(
        explicit_bool(status.get("convergence_gate_pass"),
                      f"{run.name}.convergence_gate_pass")
        and not explicit_bool(status.get("production_state_modified"),
                              f"{run.name}.production_state_modified")
        and Path(status["source_production"]).resolve()
            == SOURCE_PRODUCTION.resolve()
        and Path(status["w_grid"]).resolve() == W_GRID.resolve()
        and int(status.get("seed", -1)) == seed
        and replica_matches
        and int(status.get("row_count", -1)) == N_DATA
        and exact_number(reference.get("lambda"), strength)
        and Path(reference.get("target_csv", "")).resolve()
            == expected_reference.resolve()
        and exact_number(reference.get("b_min"), 0.1)
        and exact_number(reference.get("b_max"), BMAX)
        and exact_number(barrier.get("lambda"), barrier_strength)
        and int(barrier.get("power", -1)) == barrier_power
        and (barrier_strength <= 0.0
             or (expected_barrier_ceiling is not None
                 and exact_number(barrier.get("ceiling_total_chi2"),
                                  expected_barrier_ceiling)))
        and nonzero_regularizers == expected_nonzero
        and exact_number(status["regularization"]
                         ["likelihood_weight"]["value"], 1.0)
        and not explicit_bool(profile.get("enabled"),
                              f"{run.name}.point_profile.enabled")
        and exact_number(profile.get("lambda_per_row"), 0.0)
        and status.get("model_constraint", {}).get("kind") == "none"
        and int(complexity.get("np_width", -1)) == 48
        and int(complexity.get("np_cond_width", -1)) == 32
        and int(complexity.get("np_blocks", -1)) == 3
        and complexity.get("global_spline_nx") is None
        and complexity.get("global_spline_nb") is None
        and int(complexity.get("distill_accepted_steps", -1)) == 0
        and int(complexity.get("distill_prediction_steps", -1)) == 0,
        f"endpoint objective/identity mismatch: {run.name}",
    )
    if expected_initial_state is not None:
        observed_initial = status.get("initial_state")
        require(observed_initial is not None
                and Path(observed_initial).resolve()
                    == expected_initial_state.resolve(),
                f"endpoint state ancestry mismatch: {run.name}")
    if continuation_checkpoint:
        require(exact_number(
                    status.get("initial_relative_parameter_perturbation"), 0.0)
                and int(status.get("max_epochs", -1)) == 0
                and int(status.get("lbfgs", {}).get("max_iter", -1)) == 5000,
                f"continuation protocol mismatch: {run.name}")
    status["_validated_fit_target_sha256"] = validate_fit_target(
        run, replica_seed)
    return status


def validate_lambda300_sources(source_summary: dict) -> tuple[list[Path], list[dict]]:
    tags = [str(tag) for tag in source_summary.get("endpoint_tags", [])]
    require(source_summary.get("status") == "complete"
            and exact_number(source_summary.get("selected_strength"), 300.0)
            and exact_number(source_summary.get("selected_bmax"), BMAX)
            and int(source_summary.get("member_count", -1)) == len(SEEDS)
            and explicit_bool(source_summary.get(
                "all_starts_fnp_plateaued_and_fit_preserved"),
                "lambda300.all_starts_fnp_plateaued_and_fit_preserved")
            and not explicit_bool(source_summary.get("production_sources_modified"),
                                  "lambda300.production_sources_modified")
            and len(tags) == len(SEEDS) and len(set(tags)) == len(SEEDS),
            "lambda=300 ordered source ensemble is invalid")
    runs = []
    records = []
    for seed, tag in zip(SEEDS, tags, strict=True):
        run = OUT / tag
        expected_reference = (
            EVEN_SOURCE_REFERENCE if seed % 2 else ODD_SOURCE_REFERENCE)
        status = validate_endpoint(
            run, strength=300.0, expected_reference=expected_reference,
            barrier_strength=0.0, barrier_power=2, seed=seed,
            replica_seed=None, expected_barrier_ceiling=None,
            expected_initial_state=None, continuation_checkpoint=False)
        runs.append(run)
        records.append({
            "seed": seed,
            "tag": tag,
            "reference": str(expected_reference),
            "unpenalized_total_chi2": float(
                status["final"]["unpenalized_total_chi2"]),
            "fit_target_sha256": status[
                "_validated_fit_target_sha256"],
            "endpoint_evidence_sha256": endpoint_evidence_digest(run),
        })
    return runs, records


def validate_chain(
        group: pd.DataFrame, *, origin: Path, terminal_tag: str,
        seed: int, replica_seed: int | None, barrier_ceiling: float,
        label: str, require_recorded_restart_inputs: bool = False,
        allow_legacy_without_launch_receipt: bool = False) -> dict:
    """Validate a contiguous ordered 5k continuation chain."""
    require(len(group) > 0, f"{label} has no ledger rows")
    group = group.copy()
    group["cumulative_lbfgs_iterations"] = pd.to_numeric(
        group["cumulative_lbfgs_iterations"], errors="raise").astype(int)
    group = group.sort_values("cumulative_lbfgs_iterations")
    require(not group.duplicated("cumulative_lbfgs_iterations").any(),
            f"{label} has duplicate iteration checkpoints")
    require(not group.tag.astype(str).duplicated().any(),
            f"{label} has duplicate checkpoint tags")
    observed_iterations = group.cumulative_lbfgs_iterations.to_numpy(int)
    expected_iterations = np.arange(5000, observed_iterations[-1] + 1, 5000)
    require(np.array_equal(observed_iterations, expected_iterations),
            f"{label} is not a contiguous 5k checkpoint chain")
    require(str(group.iloc[-1].tag) == terminal_tag,
            f"{label} terminal tag does not match its last ledger checkpoint")

    previous_state = origin / "model_state.pt"
    previous_norms = origin / "dataset_norms.csv"
    chain_digest = hashlib.sha256()
    checkpoint_tags = []
    fit_target_digests = []
    checkpoint_ancestry = []
    for row in group.itertuples(index=False):
        tag = str(row.tag)
        run = OUT / tag
        status = validate_endpoint(
            run, strength=STRENGTH, expected_reference=REFERENCE,
            barrier_strength=BARRIER_STRENGTH,
            barrier_power=BARRIER_POWER, seed=seed,
            replica_seed=replica_seed,
            expected_barrier_ceiling=barrier_ceiling,
            expected_initial_state=previous_state,
            continuation_checkpoint=True)
        command = build_continuation_command(
            python=PYTHON, runner=RUNNER, seed=seed,
            source_production=SOURCE_PRODUCTION, w_grid=W_GRID,
            output_root=OUT, child_tag=tag,
            parent_state=previous_state, parent_norms=previous_norms,
            reference_strength=STRENGTH, reference_csv=REFERENCE,
            reference_bmin=0.1, reference_bmax=BMAX,
            barrier_strength=BARRIER_STRENGTH,
            barrier_power=BARRIER_POWER,
            barrier_ceiling=barrier_ceiling, replica_seed=replica_seed)
        expected_receipt = build_launch_receipt(
            receipt_root=LAUNCH_RECEIPTS, child_output=run,
            child_tag=tag, parent_state=previous_state,
            parent_norms=previous_norms, fit_seed=seed,
            replica_seed=replica_seed, command=command,
            reference_strength=STRENGTH, reference_bmin=0.1,
            reference_bmax=BMAX, barrier_strength=BARRIER_STRENGTH,
            barrier_power=BARRIER_POWER, barrier_ceiling=barrier_ceiling)
        launch_ancestry = classify_launch_ancestry(
            LAUNCH_RECEIPTS, run, tag, expected_receipt,
            allow_legacy_without_receipt=allow_legacy_without_launch_receipt)
        if require_recorded_restart_inputs:
            required_restart_columns = (
                "initial_state_path", "initial_state_sha256",
                "initial_norms_path", "initial_norms_sha256",
                "launch_ancestry_kind", "launch_receipt_path",
                "launch_receipt_sha256", "launch_argv_sha256",
            )
            require(all(hasattr(row, name) for name in required_restart_columns),
                    f"{label} lacks recorded restart-input ancestry")
            require(Path(str(row.initial_state_path)).resolve()
                        == previous_state.resolve()
                    and str(row.initial_state_sha256) == sha256(previous_state)
                    and Path(str(row.initial_norms_path)).resolve()
                        == previous_norms.resolve()
                    and str(row.initial_norms_sha256) == sha256(previous_norms),
                    f"{label} restart-input content ancestry mismatch at {tag}")
            require(str(row.launch_ancestry_kind) == launch_ancestry["kind"]
                    and str(row.launch_receipt_path)
                        == str(launch_ancestry["path"])
                    and str(row.launch_receipt_sha256)
                        == str(launch_ancestry["sha256"])
                    and str(row.launch_argv_sha256)
                        == str(launch_ancestry["argv_sha256"]),
                    f"{label} launch-receipt ledger binding mismatch at {tag}")
        evidence = endpoint_evidence_digest(run)
        chain_digest.update(tag.encode("utf-8"))
        chain_digest.update(b"\0")
        chain_digest.update(evidence.encode("ascii"))
        chain_digest.update(b"\0")
        chain_digest.update(canonical_json_sha256(
            launch_ancestry).encode("ascii"))
        chain_digest.update(b"\0")
        checkpoint_tags.append(tag)
        fit_target_digests.append(status[
            "_validated_fit_target_sha256"])
        checkpoint_ancestry.append({
            "tag": tag,
            "parent_state_path": launch_ancestry["parent_state_path"],
            "parent_state_sha256": launch_ancestry["parent_state_sha256"],
            "parent_norms_path": launch_ancestry["parent_norms_path"],
            "parent_norms_sha256": launch_ancestry["parent_norms_sha256"],
            "launch_ancestry_kind": launch_ancestry["kind"],
            "launch_receipt_path": launch_ancestry["path"],
            "launch_receipt_sha256": launch_ancestry["sha256"],
            "launch_argv_sha256": launch_ancestry["argv_sha256"],
            "child_endpoint_evidence_sha256": evidence,
        })
        previous_state = run / "model_state.pt"
        previous_norms = run / "dataset_norms.csv"
    require(len(set(fit_target_digests)) == 1,
            f"{label} changes pseudo-data targets within its continuation chain")
    return {
        "label": label,
        "seed": seed,
        "replica_seed": replica_seed,
        "origin_tag": origin.name,
        "terminal_tag": terminal_tag,
        "checkpoint_count": len(checkpoint_tags),
        "first_checkpoint_tag": checkpoint_tags[0],
        "terminal_cumulative_requested_capacity": int(
            observed_iterations[-1]),
        "fit_target_sha256": fit_target_digests[0],
        "restart_state_and_norm_content_ancestry_recorded":
            require_recorded_restart_inputs,
        "launch_time_receipt_checkpoint_count": sum(
            row["launch_ancestry_kind"] == "launch_time_content_receipt"
            for row in checkpoint_ancestry),
        "legacy_pre_receipt_checkpoint_count": sum(
            row["launch_ancestry_kind"] != "launch_time_content_receipt"
            for row in checkpoint_ancestry),
        "checkpoint_ancestry": checkpoint_ancestry,
        "checkpoint_evidence_sha256": chain_digest.hexdigest(),
    }


def _close(observed: object, expected: float, label: str) -> None:
    try:
        value = float(observed)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{label} is not numeric") from error
    require(np.isfinite(value) and np.isclose(
                value, float(expected), rtol=2.0e-11, atol=2.0e-13),
            f"{label} differs from independently recomputed value")


def _optional_int_matches(observed: object, expected: int | None) -> bool:
    if expected is None:
        return pd.isna(observed)
    try:
        value = float(observed)
    except (TypeError, ValueError):
        return False
    return np.isfinite(value) and int(value) == expected and value == int(value)


def recompute_adaptive_chain_gates(
        group: pd.DataFrame, *, origin: Path, replica_seed: int | None,
        fit_quality_ceiling: float, label: str) -> dict:
    """Recompute the locked 2% adaptive gate from endpoint artifacts.

    No declared drift, stationarity, fit, or acceptance Boolean is trusted.
    The implementation mirrors the registered supervisor's state machine but
    derives every quantity from the parent/child FNP grids and fit statuses.
    """
    ordered = group.copy()
    ordered["cumulative_lbfgs_iterations"] = pd.to_numeric(
        ordered["cumulative_lbfgs_iterations"], errors="raise").astype(int)
    ordered = ordered.sort_values("cumulative_lbfgs_iterations")
    required_columns = {
        "minimum_required_iterations", "fnp_drift_from_previous_chunk",
        "eligible_post_mandatory_confirmation",
        "post_mandatory_window_fnp_drift",
        "stationarity_window_anchor_iterations",
        "next_stationarity_window_anchor_iterations",
        "passes_post_mandatory_window_drift_2pct",
        "passes_drift_0p25pct", "passes_drift_0p5pct",
        "passes_drift_1pct", "passes_drift_2pct",
        "consecutive_quiet_blocks", "sensitivity_confirmation_triggered",
        "fresh_quiet_blocks_after_sensitivity_trigger",
        "unpenalized_total_chi2", "data_chi2",
        "replica_chi2_sanity_ceiling", "replica_chi2_sanity_pass",
        "fit_quality_ceiling_total_chi2", "fit_quality_gate_pass",
        "stationarity_gate_pass", "endpoint_acceptance_gate_pass",
        "fnp_plateau_gate_pass", "full_horizon_required",
        "requested_lbfgs_max_iterations_this_block",
        "executed_lbfgs_closure_evaluations_this_block",
    }
    require(required_columns.issubset(ordered.columns),
            f"{label} adaptive-gate ledger schema is incomplete")
    minimum_values = pd.to_numeric(
        ordered.minimum_required_iterations, errors="raise").astype(int).unique()
    require(len(minimum_values) == 1,
            f"{label} changes its minimum-required horizon")
    minimum_required = int(minimum_values[0])

    previous = origin
    consecutive = 0
    sensitivity_active = False
    fresh_after_sensitivity = 0
    anchor: np.ndarray | None = None
    anchor_iterations: int | None = None
    computed_rows = []
    replica_ceiling = N_DATA + 5.0 * math.sqrt(2.0 * N_DATA)
    for row in ordered.itertuples(index=False):
        cumulative = int(row.cumulative_lbfgs_iterations)
        run = OUT / str(row.tag)
        previous_b, previous_values = fnp_curve(previous)
        current_b, current_values = fnp_curve(run)
        require(previous_b.shape == current_b.shape
                and np.array_equal(previous_b, current_b),
                f"{label} changes FNP grid at {run.name}")
        drift = float(np.max(
            np.abs(current_values - previous_values)
            / np.maximum(previous_values, 0.05)))
        if cumulative == MINIMUM_CUMULATIVE_ITERATIONS:
            anchor = current_values.copy()
            anchor_iterations = cumulative
        eligible = cumulative > MINIMUM_CUMULATIVE_ITERATIONS
        tested_anchor_iterations = anchor_iterations
        window_drift = (
            float(np.max(np.abs(current_values - anchor)
                         / np.maximum(anchor, 0.05)))
            if eligible and anchor is not None else None)
        window_quiet = (
            window_drift is not None and window_drift <= FNP_DRIFT_GATE)
        quiet = eligible and drift <= FNP_DRIFT_GATE and window_quiet
        if not eligible:
            consecutive = 0
        elif not quiet:
            anchor = current_values.copy()
            anchor_iterations = cumulative
            consecutive = 0
            sensitivity_active = False
            fresh_after_sensitivity = 0
        else:
            consecutive += 1
        triggered_now = (
            quiet and drift >= FNP_DRIFT_SENSITIVITY[-2]
            and not sensitivity_active)
        if triggered_now:
            sensitivity_active = True
            fresh_after_sensitivity = 0
        elif sensitivity_active:
            fresh_after_sensitivity = fresh_after_sensitivity + 1 if quiet else 0
        confirmation = (
            fresh_after_sensitivity >= REQUIRED_CONSECUTIVE_BLOCKS
            if sensitivity_active else
            consecutive >= REQUIRED_CONSECUTIVE_BLOCKS)
        stationarity = bool(
            cumulative >= max(
                MINIMUM_CUMULATIVE_ITERATIONS
                + 5000 * REQUIRED_CONSECUTIVE_BLOCKS,
                minimum_required)
            and confirmation and window_quiet)
        status = load_status(run)
        unpenalized = float(status["final"]["unpenalized_total_chi2"])
        data_chi2 = float(status["final"]["data_chi2"])
        fit_pass = unpenalized <= fit_quality_ceiling
        replica_sanity = (
            replica_seed is None or unpenalized <= replica_ceiling)
        accepted = stationarity and fit_pass and replica_sanity

        _close(row.fnp_drift_from_previous_chunk, drift,
               f"{label} adjacent drift at {cumulative}")
        require(explicit_bool(row.eligible_post_mandatory_confirmation,
                              f"{label}.eligible") == eligible,
                f"{label} eligible flag differs at {cumulative}")
        if window_drift is None:
            require(pd.isna(row.post_mandatory_window_fnp_drift),
                    f"{label} has a spurious window drift at {cumulative}")
        else:
            _close(row.post_mandatory_window_fnp_drift, window_drift,
                   f"{label} window drift at {cumulative}")
        require(_optional_int_matches(
                    row.stationarity_window_anchor_iterations,
                    tested_anchor_iterations)
                and _optional_int_matches(
                    row.next_stationarity_window_anchor_iterations,
                    anchor_iterations),
                f"{label} anchor/reset history differs at {cumulative}")
        expected_bools = {
            "passes_post_mandatory_window_drift_2pct": window_quiet,
            "passes_drift_0p25pct": drift <= FNP_DRIFT_SENSITIVITY[0],
            "passes_drift_0p5pct": drift <= FNP_DRIFT_SENSITIVITY[1],
            "passes_drift_1pct": drift <= FNP_DRIFT_SENSITIVITY[2],
            "passes_drift_2pct": drift <= FNP_DRIFT_SENSITIVITY[3],
            "sensitivity_confirmation_triggered": sensitivity_active,
            "replica_chi2_sanity_pass": replica_sanity,
            "fit_quality_gate_pass": fit_pass,
            "stationarity_gate_pass": stationarity,
            "endpoint_acceptance_gate_pass": accepted,
            "fnp_plateau_gate_pass": accepted,
            "full_horizon_required": replica_seed is None,
        }
        for field, expected in expected_bools.items():
            require(explicit_bool(getattr(row, field),
                                  f"{label}.{field}") == expected,
                    f"{label} {field} differs at {cumulative}")
        require(int(row.consecutive_quiet_blocks) == consecutive
                and int(row.fresh_quiet_blocks_after_sensitivity_trigger)
                    == fresh_after_sensitivity,
                f"{label} confirmation counters differ at {cumulative}")
        _close(row.unpenalized_total_chi2, unpenalized,
               f"{label} unpenalized chi2 at {cumulative}")
        _close(row.data_chi2, data_chi2,
               f"{label} data chi2 at {cumulative}")
        _close(row.fit_quality_ceiling_total_chi2, fit_quality_ceiling,
               f"{label} fit ceiling at {cumulative}")
        if replica_seed is None:
            require(pd.isna(row.replica_chi2_sanity_ceiling),
                    f"{label} central has a replica chi2 ceiling")
        else:
            _close(row.replica_chi2_sanity_ceiling, replica_ceiling,
                   f"{label} replica chi2 ceiling at {cumulative}")
        require(int(row.requested_lbfgs_max_iterations_this_block) == 5000
                and int(row.executed_lbfgs_closure_evaluations_this_block)
                    == int(status["lbfgs"]["closure_evaluations"]),
                f"{label} LBFGS accounting differs at {cumulative}")
        computed_rows.append({
            "cumulative_requested_capacity": cumulative,
            "stationarity_gate_pass": stationarity,
            "fit_quality_gate_pass": fit_pass,
            "replica_chi2_sanity_pass": replica_sanity,
            "endpoint_acceptance_gate_pass": accepted,
        })
        previous = run

    terminal = computed_rows[-1]
    passing_capacities = [
        row["cumulative_requested_capacity"] for row in computed_rows
        if row["endpoint_acceptance_gate_pass"]]
    if replica_seed is None:
        require(terminal["cumulative_requested_capacity"]
                    == CENTRAL_REQUESTED_CAPACITY
                and len(computed_rows) == MAX_CHECKPOINTS,
                "central did not exhaust the exact 300k horizon")
    elif passing_capacities:
        require(passing_capacities[0]
                    == terminal["cumulative_requested_capacity"],
                f"{label} continued after its first accepted endpoint")
    else:
        require(terminal["cumulative_requested_capacity"]
                    == CENTRAL_REQUESTED_CAPACITY,
                f"{label} nonpassing endpoint truncated before 300k")
    return {
        "minimum_required_iterations": minimum_required,
        "checkpoint_count": len(computed_rows),
        "terminal_requested_capacity": terminal[
            "cumulative_requested_capacity"],
        "terminal_stationarity_gate_pass": terminal[
            "stationarity_gate_pass"],
        "terminal_fit_quality_gate_pass": terminal[
            "fit_quality_gate_pass"],
        "terminal_replica_chi2_sanity_pass": terminal[
            "replica_chi2_sanity_pass"],
        "terminal_endpoint_acceptance_gate_pass": terminal[
            "endpoint_acceptance_gate_pass"],
        "first_acceptance_requested_capacity": (
            passing_capacities[0] if passing_capacities else None),
    }


def _validate_start_groups(
        source_runs: list[Path], ledger: pd.DataFrame,
        terminal_by_seed: dict[int, str] | None) -> list[dict]:
    require({"strength", "seed", "cumulative_lbfgs_iterations", "tag"}
            .issubset(ledger.columns), "start ledger schema is incomplete")
    selected = ledger[np.isclose(pd.to_numeric(
        ledger.strength, errors="coerce"), STRENGTH)].copy()
    observed_seeds = sorted(pd.to_numeric(
        selected.seed, errors="raise").astype(int).unique().tolist())
    expected_seeds = (list(SEEDS) if terminal_by_seed is not None
                      else list(SEEDS[:len(observed_seeds)]))
    require(observed_seeds == expected_seeds,
            "start ledger contains missing, extra, or unordered seed identities")
    records = []
    for seed, source in zip(SEEDS, source_runs, strict=True):
        if seed not in observed_seeds:
            continue
        group = selected[pd.to_numeric(
            selected.seed, errors="raise").astype(int).eq(seed)]
        terminal = (terminal_by_seed[seed] if terminal_by_seed is not None
                    else str(group.sort_values(
                        "cumulative_lbfgs_iterations").iloc[-1].tag))
        source_status = load_status(source)
        ceiling = (float(source_status["final"]["unpenalized_total_chi2"])
                   + math.sqrt(2.0 * int(source_status["row_count"])))
        records.append(validate_chain(
            group, origin=source, terminal_tag=terminal, seed=seed,
            replica_seed=None, barrier_ceiling=ceiling,
            label=f"start_s{seed}",
            allow_legacy_without_launch_receipt=True))
    require(sum(item["checkpoint_count"] for item in records) == len(selected),
            "start ledger contains rows outside the audited chains")
    return records


def audit_current_start_prefixes() -> dict:
    """Callable read-only helper for auditing an interrupted/in-progress run."""
    _, fixed_protocol_hash = validate_fixed_challenger_protocol()
    source_summary = json.loads(START_SOURCE_ENSEMBLE.read_text())
    source_runs, source_records = validate_lambda300_sources(source_summary)
    ledger = pd.read_csv(START_LEDGER)
    chains = _validate_start_groups(source_runs, ledger, terminal_by_seed=None)
    return {
        "status": "pass_partial_start_prefixes",
        "fixed_challenger_protocol": str(FIXED_PROTOCOL),
        "fixed_challenger_protocol_sha256": fixed_protocol_hash,
        "fixed_fnp_reference": str(FIXED_FNP_REFERENCE),
        "fixed_fnp_reference_sha256": EXPECTED_FNP_REFERENCE_SHA256,
        **fixed_implementation_binding(),
        "source_count": len(source_records),
        "audited_start_prefix_count": len(chains),
        "audited_checkpoint_count": sum(
            item["checkpoint_count"] for item in chains),
        "start_chains": chains,
    }


def validate_start_chains(starts: dict, source_runs: list[Path]) -> list[dict]:
    failed_seeds = sorted(int(value) for value in starts.get("failed_seeds", []))
    failed_capacities = {
        int(seed): int(value) for seed, value in starts.get(
            "failed_terminal_requested_capacity_by_seed", {}).items()
    }
    require(str(starts.get("status")) in {"complete", "verification_failed"}
            and exact_number(starts.get("selected_strength"), STRENGTH)
            and exact_number(starts.get("selected_bmax"), BMAX)
            and exact_number(starts.get("fit_quality_barrier_strength"),
                             BARRIER_STRENGTH)
            and int(starts.get("fit_quality_barrier_power", -1))
                == BARRIER_POWER
            and int(starts.get("member_count", -1)) == len(SEEDS)
            and int(starts.get(
                "mandatory_same_objective_iterations_per_start", -1))
                >= 200_000
            and explicit_bool(starts.get(
                "failed_starts_exhausted_full_requested_horizon"),
                "starts.failed_starts_exhausted_full_requested_horizon")
            and int(starts.get(
                "full_requested_capacity_per_nonpassing_start", -1)) == 300_000
            and set(failed_capacities) == set(failed_seeds)
            and all(value == 300_000 for value in failed_capacities.values())
            and not explicit_bool(starts.get("production_sources_modified"),
                                  "starts.production_sources_modified"),
            "lambda=600 start summary is not an exact terminal generation")
    tags = [str(tag) for tag in starts.get("endpoint_tags", [])]
    require(len(tags) == len(SEEDS) and len(set(tags)) == len(SEEDS),
            "lambda=600 start endpoints do not have exact 24-member coverage")
    terminal_by_seed = dict(zip(SEEDS, tags, strict=True))
    chains = _validate_start_groups(
        source_runs, pd.read_csv(START_LEDGER), terminal_by_seed)
    require(all(item["terminal_cumulative_requested_capacity"] >= 200_000
                for item in chains),
            "one or more terminal starts lack the mandatory 200k capacity")
    chain_by_seed = {int(item["seed"]): item for item in chains}
    require(all(
        int(chain_by_seed[seed]["terminal_cumulative_requested_capacity"])
            == 300_000
        for seed in failed_seeds),
        "one or more failed starts did not exhaust the full 300k horizon")
    observed_legacy_tags = {
        checkpoint["tag"]
        for chain in chains
        for checkpoint in chain["checkpoint_ancestry"]
        if checkpoint["launch_ancestry_kind"]
            != "launch_time_content_receipt"
    }
    declared_legacy_tags = {
        str(value) for value in starts.get(
            "legacy_pre_receipt_used_tags", [])
    }
    admitted_legacy_tags = {
        str(value) for value in starts.get(
            "legacy_pre_receipt_admission_tags", [])
    }
    require(len(admitted_legacy_tags) == int(starts.get(
                "legacy_pre_receipt_admission_tag_count", -1))
            and declared_legacy_tags == observed_legacy_tags
            and declared_legacy_tags.issubset(admitted_legacy_tags),
            "legacy pre-receipt admission snapshot differs from audited chains")
    return chains


def seal_terminal_start_current_bytes(start_chains: list[dict]) -> dict:
    """Freeze the terminal legacy graph without overstating historical proof.

    The controller already running when launch receipts were introduced cannot
    supply retrospective hashes of the bytes it consumed.  This generation-
    addressed immutable seal records the entire current parent/child graph at
    the precentral boundary.  It detects every later byte change but is
    explicitly not represented as launch-time ancestry for legacy checkpoints.
    """
    input_sha = {
        str(START_SOURCE_ENSEMBLE): sha256(START_SOURCE_ENSEMBLE),
        str(STARTS): sha256(STARTS),
        str(START_LEDGER): sha256(START_LEDGER),
    }
    generation = canonical_json_sha256(input_sha)
    checkpoints = [
        item
        for chain in start_chains
        for item in chain["checkpoint_ancestry"]
    ]
    launch_count = sum(
        item["launch_ancestry_kind"] == "launch_time_content_receipt"
        for item in checkpoints)
    legacy_count = len(checkpoints) - launch_count
    payload = {
        "status": (
            "sealed_current_bytes_with_disclosed_legacy_limit"
            if legacy_count else "sealed_launch_time_receipt_graph"),
        "generation_sha256": generation,
        "input_sha256": input_sha,
        "checkpoint_count": len(checkpoints),
        "launch_time_receipt_checkpoint_count": launch_count,
        "legacy_pre_receipt_checkpoint_count": legacy_count,
        "checkpoint_ancestry": checkpoints,
        "assurance": (
            "Launch-time receipts prove exact parent state/norm bytes for "
            "receipt-bearing checkpoints. Legacy checkpoints created by the "
            "already-running pre-receipt controller have only fit_status path "
            "ancestry plus this immutable precentral seal of current bytes. "
            "The seal detects later mutation but does not retroactively prove "
            "which historical parent bytes the legacy child consumed."),
        "production_sources_modified": False,
    }
    path = START_SEAL_ROOT / f"{generation}.json"
    immutable_create_or_validate_json(path, payload)
    observed = json.loads(path.read_text())
    require(observed == payload,
            "terminal start current-byte seal changed during validation")
    return {
        "status": payload["status"],
        "path": str(path.resolve()),
        "sha256": receipt_sha256(path),
        "generation_sha256": generation,
        "checkpoint_count": len(checkpoints),
        "launch_time_receipt_checkpoint_count": launch_count,
        "legacy_pre_receipt_checkpoint_count": legacy_count,
        "historical_ancestry_limitation": payload["assurance"],
    }


def build_terminal_start_audit() -> dict:
    """Build deterministic terminal 24-start ancestry evidence."""
    _, fixed_protocol_hash = validate_fixed_challenger_protocol()
    source_summary = json.loads(START_SOURCE_ENSEMBLE.read_text())
    starts = json.loads(STARTS.read_text())
    start_status = str(starts.get("status"))
    start_scientific_gate = explicit_bool(starts.get(
        "all_starts_fnp_plateaued_and_fit_preserved"),
        "starts.all_starts_fnp_plateaued_and_fit_preserved")
    failed_seeds = sorted(int(value) for value in starts.get(
        "failed_seeds", []))
    require((start_status == "complete") == start_scientific_gate
            and (not failed_seeds if start_scientific_gate else bool(failed_seeds))
            and len(failed_seeds) == len(set(failed_seeds))
            and all(seed in SEEDS for seed in failed_seeds),
            "terminal start status/scientific-failure flags are inconsistent")
    source_runs, source_records = validate_lambda300_sources(source_summary)
    start_chains = validate_start_chains(starts, source_runs)
    current_byte_seal = seal_terminal_start_current_bytes(start_chains)
    selection = central_initializer_selection([
        str(tag) for tag in starts["endpoint_tags"]])
    summary = {
        "status": "pass",
        "fixed_challenger_protocol": str(FIXED_PROTOCOL),
        "fixed_challenger_protocol_sha256": fixed_protocol_hash,
        "fixed_fnp_reference": str(FIXED_FNP_REFERENCE),
        "fixed_fnp_reference_sha256": EXPECTED_FNP_REFERENCE_SHA256,
        **fixed_implementation_binding(),
        "audit_scope": (
            "terminal exact ordered lambda300-to-lambda600 ancestry before "
            "central/replica fitting"),
        "selected_prescription": {
            "reference_strength": STRENGTH,
            "reference_b_min": 0.1,
            "reference_b_max": BMAX,
            "fit_quality_barrier_strength": BARRIER_STRENGTH,
            "fit_quality_barrier_power": BARRIER_POWER,
            "reference": str(REFERENCE),
        },
        "source_terminal_status": source_summary["status"],
        "start_terminal_status": start_status,
        "start_scientific_gate_pass": start_scientific_gate,
        "failed_start_seeds": failed_seeds,
        "failed_starts_exhausted_full_requested_horizon": True,
        "full_requested_capacity_per_nonpassing_start": 300_000,
        "required_endpoint_files": list(REQUIRED_ENDPOINT_FILES),
        "lambda300_source_count": len(source_records),
        "start_chain_count": len(start_chains),
        "total_start_continuation_checkpoint_count": sum(
            item["checkpoint_count"] for item in start_chains),
        "lambda300_sources": source_records,
        "start_chains": start_chains,
        "terminal_start_current_byte_seal": current_byte_seal,
        "exact_launch_time_ancestry_proven_for_all_start_checkpoints": (
            current_byte_seal["legacy_pre_receipt_checkpoint_count"] == 0),
        "legacy_start_ancestry_limitation": current_byte_seal[
            "historical_ancestry_limitation"],
        "central_initializer_selection": selection,
        "input_sha256": {
            str(FIXED_PROTOCOL): fixed_protocol_hash,
            str(FIXED_FNP_REFERENCE): EXPECTED_FNP_REFERENCE_SHA256,
            **{
                str(FIXED_IMPLEMENTATION_FILES[role]): digest
                for role, digest in EXPECTED_IMPLEMENTATION_SHA256.items()
            },
            str(START_SOURCE_ENSEMBLE): sha256(START_SOURCE_ENSEMBLE),
            str(STARTS): sha256(STARTS),
            str(START_LEDGER): sha256(START_LEDGER),
        },
        "production_sources_modified": False,
    }
    atomic_write_json(START_ONLY_TARGET / "summary.json", summary)
    return summary


def validate_declared_terminal_gate_flags(
        replicas: dict, *, expected_status: str,
        expected_bools: dict[str, bool],
        expected_seed_lists: dict[str, list[int]]) -> None:
    """Pure fail-closed check used after artifact-level gate recomputation."""
    require(str(replicas.get("status")) == expected_status,
            "terminal central/replica status differs from recomputation")
    for key, expected in expected_bools.items():
        require(explicit_bool(replicas.get(key), f"replicas.{key}") == expected,
                f"terminal central/replica Boolean differs: {key}")
    for key, expected in expected_seed_lists.items():
        try:
            observed = [int(value) for value in replicas.get(key, [])]
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                f"terminal central/replica seed list is invalid: {key}") from error
        require(observed == expected,
                f"terminal central/replica seed list differs: {key}")


def validate_declared_central_replica_gates(
        replicas: dict, central_gate: dict, replica_gates: list[dict],
        replica_tags: list[str]) -> None:
    """Require every terminal summary gate to equal artifact recomputation."""
    gates_by_seed = {
        int(item["replica_seed"]): item for item in replica_gates}
    require(set(gates_by_seed) == set(REPLICA_SEEDS),
            "independent replica-gate recomputation lacks exact coverage")
    acceptance_failed = [
        seed for seed in REPLICA_SEEDS
        if not gates_by_seed[seed]["terminal_endpoint_acceptance_gate_pass"]]
    stationarity_failed = [
        seed for seed in REPLICA_SEEDS
        if not gates_by_seed[seed]["terminal_stationarity_gate_pass"]]
    fit_failed = [
        seed for seed in REPLICA_SEEDS
        if not gates_by_seed[seed]["terminal_fit_quality_gate_pass"]]
    sanity_failed = [
        seed for seed in REPLICA_SEEDS
        if not gates_by_seed[seed]["terminal_replica_chi2_sanity_pass"]]
    failed_capacities = {
        seed: int(gates_by_seed[seed]["terminal_requested_capacity"])
        for seed in acceptance_failed
    }
    declared_failed_capacities = {
        int(seed): int(value) for seed, value in replicas.get(
            "failed_replica_terminal_requested_capacity_by_seed", {}).items()
    }
    require(declared_failed_capacities == failed_capacities
            and int(replicas.get(
                "full_requested_capacity_per_nonpassing_replica", -1))
                == CENTRAL_REQUESTED_CAPACITY
            and explicit_bool(replicas.get(
                "failed_replicas_exhausted_full_requested_horizon"),
                "replicas.failed_replicas_exhausted_full_requested_horizon")
                == all(value == CENTRAL_REQUESTED_CAPACITY
                       for value in failed_capacities.values()),
            "failed-replica full-horizon declarations differ from endpoints")

    direct = json.loads(DIRECT_LONG_EVIDENCE.read_text())
    require(direct.get("status") == "candidate_survives_discriminator"
            and exact_number(direct.get("tested_reference_strength"), STRENGTH)
            and exact_number(direct.get(
                "fit_quality_barrier_strength"), BARRIER_STRENGTH)
            and int(direct.get("fit_quality_barrier_power", -1)) == BARRIER_POWER,
            "independent lambda600 stress evidence is not exact")
    stress_tag = str(direct.get("endpoint_tag", ""))
    stress_seed = int(direct.get("replica_seed", -1))
    require(stress_seed in REPLICA_SEEDS and stress_tag,
            "independent lambda600 stress identity is invalid")
    stress_run = OUT / stress_tag
    stress_status = load_status(stress_run)
    expected_replica_ceiling = N_DATA + 5.0 * math.sqrt(2.0 * N_DATA)
    validate_endpoint(
        stress_run, strength=STRENGTH, expected_reference=REFERENCE,
        barrier_strength=BARRIER_STRENGTH, barrier_power=BARRIER_POWER,
        seed=int(stress_status["seed"]), replica_seed=stress_seed,
        expected_barrier_ceiling=expected_replica_ceiling,
        expected_initial_state=None, continuation_checkpoint=True)
    primary_run = OUT / replica_tags[stress_seed - REPLICA_SEEDS[0]]
    _, primary_values = fnp_curve(primary_run)
    _, stress_values = fnp_curve(stress_run)
    denominator = np.maximum(0.5 * (primary_values + stress_values), 0.05)
    disagreement = float(np.max(
        np.abs(primary_values - stress_values) / denominator))
    starts = json.loads(STARTS.read_text())
    threshold = float(starts[
        "max_endpoint_fnp_full_range_selected_domain_floor_normalized"])
    agreement = disagreement <= threshold
    declared_checks = replicas.get("cross_optimizer_checks")
    require(isinstance(declared_checks, list) and len(declared_checks) == 1,
            "cross-optimizer check coverage is not the exact stress case")
    declared = declared_checks[0]
    require(int(declared.get("replica_seed", -1)) == stress_seed
            and str(declared.get("primary_endpoint_tag")) == primary_run.name
            and str(declared.get("independent_stress_endpoint_tag")) == stress_tag
            and explicit_bool(declared.get("agreement_pass"),
                              "cross_optimizer.agreement_pass") == agreement,
            "declared cross-optimizer identity/gate differs")
    _close(declared.get("max_fnp_symmetric_relative_difference"), disagreement,
           "cross-optimizer FNP disagreement")
    _close(declared.get("allowed_full24_start_range"), threshold,
           "cross-optimizer allowed start range")
    cross_failed = [] if agreement else [stress_seed]
    cross_pass = agreement

    start_gate = explicit_bool(starts.get(
        "all_starts_fnp_plateaued_and_fit_preserved"),
        "starts.all_starts_fnp_plateaued_and_fit_preserved")
    central_acceptance = bool(central_gate[
        "terminal_endpoint_acceptance_gate_pass"])
    stationarity_gate = bool(
        start_gate and central_gate["terminal_stationarity_gate_pass"]
        and not stationarity_failed)
    downstream_fit_gate = bool(
        central_gate["terminal_fit_quality_gate_pass"]
        and not fit_failed and not sanity_failed)
    coverage = bool(
        len(replica_tags) == len(REPLICA_SEEDS)
        and len(set(replica_tags)) == len(REPLICA_SEEDS)
        and central_gate["terminal_requested_capacity"]
            == CENTRAL_REQUESTED_CAPACITY)
    promotion = bool(
        coverage and start_gate and central_acceptance
        and not acceptance_failed and cross_pass)
    failed = sorted(set(acceptance_failed) | set(cross_failed))
    expected_status = (
        "complete" if promotion else "complete_with_scientific_failures")

    validate_declared_terminal_gate_flags(
        replicas, expected_status=expected_status,
        expected_bools={
            "central_fnp_plateau_pass": central_acceptance,
            "central_stationarity_gate_pass": bool(
                central_gate["terminal_stationarity_gate_pass"]),
            "central_acceptance_gate_pass": central_acceptance,
            "central_fit_quality_gate_pass": bool(
                central_gate["terminal_fit_quality_gate_pass"]),
            "all_replica_endpoint_acceptance_gates_pass":
                not acceptance_failed,
            "all_replicas_stationarity_gate_pass": not stationarity_failed,
            "all_replica_fit_quality_gates_pass": not fit_failed,
            "all_replica_chi2_sanity_gates_pass": not sanity_failed,
            "all_replicas_fnp_plateaued": not failed,
            "cross_optimizer_coverage_complete": True,
            "all_stress_replicas_cross_optimizer_consistent": cross_pass,
            "coverage_complete": coverage,
            "stationarity_gate_pass": stationarity_gate,
            "downstream_fit_quality_gate_pass": downstream_fit_gate,
            "promotion_eligible": promotion,
        },
        expected_seed_lists={
            "replica_acceptance_failed_seeds": acceptance_failed,
            "replica_stationarity_failed_seeds": stationarity_failed,
            "replica_fit_quality_failed_seeds": fit_failed,
            "replica_chi2_sanity_failed_seeds": sanity_failed,
            "cross_optimizer_failed_replica_seeds": cross_failed,
            "failed_replica_seeds": failed,
        })

    require(str(replicas.get("status")) == expected_status
            and explicit_bool(replicas.get("central_fnp_plateau_pass"),
                              "replicas.central_fnp_plateau_pass")
                == central_acceptance
            and explicit_bool(replicas.get("central_stationarity_gate_pass"),
                              "replicas.central_stationarity_gate_pass")
                == central_gate["terminal_stationarity_gate_pass"]
            and explicit_bool(replicas.get("central_acceptance_gate_pass"),
                              "replicas.central_acceptance_gate_pass")
                == central_acceptance
            and explicit_bool(replicas.get("central_fit_quality_gate_pass"),
                              "replicas.central_fit_quality_gate_pass")
                == central_gate["terminal_fit_quality_gate_pass"]
            and [int(value) for value in replicas.get(
                "replica_acceptance_failed_seeds", [])] == acceptance_failed
            and [int(value) for value in replicas.get(
                "replica_stationarity_failed_seeds", [])] == stationarity_failed
            and [int(value) for value in replicas.get(
                "replica_fit_quality_failed_seeds", [])] == fit_failed
            and [int(value) for value in replicas.get(
                "replica_chi2_sanity_failed_seeds", [])] == sanity_failed
            and [int(value) for value in replicas.get(
                "cross_optimizer_failed_replica_seeds", [])] == cross_failed
            and [int(value) for value in replicas.get(
                "failed_replica_seeds", [])] == failed,
            "terminal central/replica failure lists or gates differ from artifacts")
    require(explicit_bool(replicas.get(
                "all_replica_endpoint_acceptance_gates_pass"),
                "replicas.all_replica_endpoint_acceptance_gates_pass")
                == (not acceptance_failed)
            and explicit_bool(replicas.get(
                "all_replicas_stationarity_gate_pass"),
                "replicas.all_replicas_stationarity_gate_pass")
                == (not stationarity_failed)
            and explicit_bool(replicas.get(
                "all_replica_fit_quality_gates_pass"),
                "replicas.all_replica_fit_quality_gates_pass")
                == (not fit_failed)
            and explicit_bool(replicas.get(
                "all_replica_chi2_sanity_gates_pass"),
                "replicas.all_replica_chi2_sanity_gates_pass")
                == (not sanity_failed)
            and explicit_bool(replicas.get("all_replicas_fnp_plateaued"),
                              "replicas.all_replicas_fnp_plateaued")
                == (not failed)
            and int(replicas.get(
                "available_cross_optimizer_comparison_count", -1)) == 1
            and int(replicas.get(
                "completed_cross_optimizer_comparison_count", -1)) == 1
            and explicit_bool(replicas.get(
                "cross_optimizer_coverage_complete"),
                "replicas.cross_optimizer_coverage_complete")
            and explicit_bool(replicas.get(
                "all_stress_replicas_cross_optimizer_consistent"),
                "replicas.all_stress_replicas_cross_optimizer_consistent")
                == cross_pass
            and explicit_bool(replicas.get("coverage_complete"),
                              "replicas.coverage_complete") == coverage
            and explicit_bool(replicas.get("stationarity_gate_pass"),
                              "replicas.stationarity_gate_pass")
                == stationarity_gate
            and explicit_bool(replicas.get("downstream_fit_quality_gate_pass"),
                              "replicas.downstream_fit_quality_gate_pass")
                == downstream_fit_gate
            and explicit_bool(replicas.get("promotion_eligible"),
                              "replicas.promotion_eligible") == promotion,
            "terminal aggregate central/replica gates differ from recomputation")


def validate_central_and_replica_chains(replicas: dict) -> tuple[dict, list[dict]]:
    require_fixed_implementation_binding(
        replicas, "central/replica terminal summary")
    require(str(replicas.get("status")) in {
                "complete", "complete_with_scientific_failures"}
            and exact_number(replicas.get("selected_strength"), STRENGTH)
            and exact_number(replicas.get("selected_bmax"), BMAX)
            and exact_number(replicas.get("fit_quality_barrier_strength"),
                             BARRIER_STRENGTH)
            and int(replicas.get("fit_quality_barrier_power", -1))
                == BARRIER_POWER
            and int(replicas.get("completed_replica_count", -1))
                == len(REPLICA_SEEDS)
            and explicit_bool(replicas.get("replica_coverage_complete"),
                              "replicas.replica_coverage_complete")
            and explicit_bool(replicas.get(
                "central_full_horizon_complete"),
                "replicas.central_full_horizon_complete")
            and int(replicas.get(
                "central_full_horizon_requested_capacity", -1))
                == CENTRAL_REQUESTED_CAPACITY
            and int(replicas.get(
                "central_terminal_requested_capacity", -1))
                == CENTRAL_REQUESTED_CAPACITY
            and explicit_bool(replicas.get(
                "restart_state_and_norm_content_ancestry_recorded"),
                "replicas.restart_state_and_norm_content_ancestry_recorded")
            and explicit_bool(replicas.get(
                "launch_time_state_and_norm_content_receipts_required"),
                "replicas.launch_time_state_and_norm_content_receipts_required")
            and Path(replicas.get("launch_receipt_root", "")).resolve()
                == LAUNCH_RECEIPTS.resolve()
            and explicit_bool(replicas.get(
                "fit_target_realizations_content_validated"),
                "replicas.fit_target_realizations_content_validated")
            and not explicit_bool(replicas.get("production_sources_modified"),
                                  "replicas.production_sources_modified"),
            "central/replica summary is not an exact terminal generation")
    initializer_tag = str(replicas.get("central_initializer_tag", ""))
    central_tag = str(replicas.get("central_endpoint_tag", ""))
    start_tags = [str(value) for value in json.loads(
        STARTS.read_text()).get("endpoint_tags", [])]
    initializer_selection = central_initializer_selection(start_tags)
    require(initializer_tag == initializer_selection["selected_tag"]
            and central_tag,
            "declared central initializer is not the recomputed 24-start medoid")
    declared_selection = replicas.get("central_initializer_selection")
    require(isinstance(declared_selection, dict)
            and declared_selection == initializer_selection,
            "declared central initializer-selection evidence is missing or changed")
    require(initializer_tag in set(start_tags),
            "declared central initializer is outside the 24 start endpoints")
    initializer = OUT / initializer_tag
    initializer_status = load_status(initializer)
    central_ceiling = (
        float(initializer_status["final"]["unpenalized_total_chi2"])
        + math.sqrt(2.0 * int(initializer_status["row_count"])))

    ledger = pd.read_csv(REPLICA_LEDGER)
    required_columns = {
        "kind", "replica_seed", "fit_seed",
        "cumulative_lbfgs_iterations", "tag",
    }
    require(required_columns.issubset(ledger.columns),
            "central/replica ledger schema is incomplete")
    kinds = set(ledger.kind.astype(str))
    require(kinds == {"central", "experimental_replica"},
            "central/replica ledger contains unexpected chain kinds")
    central_group = ledger[ledger.kind.astype(str).eq("central")]
    require(len(central_group) > 0
            and pd.to_numeric(central_group.replica_seed,
                              errors="coerce").isna().all(),
            "central ledger identity is invalid")
    central_fit_seeds = pd.to_numeric(
        central_group.fit_seed, errors="raise").astype(int).unique()
    require(len(central_fit_seeds) == 1
            and int(central_fit_seeds[0])
                == int(initializer_status["seed"]),
            "central ledger fit seed differs from its initializer")
    central_chain = validate_chain(
        central_group, origin=initializer, terminal_tag=central_tag,
        seed=int(central_fit_seeds[0]), replica_seed=None,
        barrier_ceiling=central_ceiling, label="central",
        require_recorded_restart_inputs=True)
    central_gate = recompute_adaptive_chain_gates(
        central_group, origin=initializer, replica_seed=None,
        fit_quality_ceiling=central_ceiling, label="central")
    require(central_chain["terminal_cumulative_requested_capacity"]
                == CENTRAL_REQUESTED_CAPACITY,
            "central continuation did not terminate at exactly 300k capacity")
    require(central_chain["legacy_pre_receipt_checkpoint_count"] == 0,
            "central chain contains a checkpoint without launch-time ancestry")

    tags = [str(tag) for tag in replicas.get("replica_endpoint_tags", [])]
    require(len(tags) == len(REPLICA_SEEDS)
            and len(set(tags)) == len(REPLICA_SEEDS),
            "replica endpoint list lacks exact 50-member coverage")
    replica_rows = ledger[
        ledger.kind.astype(str).eq("experimental_replica")].copy()
    observed_replica_ids = sorted(pd.to_numeric(
        replica_rows.replica_seed, errors="raise").astype(int).unique())
    require(observed_replica_ids == list(REPLICA_SEEDS),
            "replica ledger lacks exact r1001-r1050 identity coverage")
    central = OUT / central_tag
    expected_replica_ceiling = N_DATA + 5.0 * math.sqrt(2.0 * N_DATA)
    replica_chains = []
    replica_gates = []
    for index, (replica_seed, tag) in enumerate(zip(
            REPLICA_SEEDS, tags, strict=True)):
        group = replica_rows[pd.to_numeric(
            replica_rows.replica_seed, errors="raise").astype(int).eq(
                replica_seed)]
        fit_seeds = pd.to_numeric(
            group.fit_seed, errors="raise").astype(int).unique()
        expected_fit_seed = 2001 + index
        require(len(fit_seeds) == 1
                and int(fit_seeds[0]) == expected_fit_seed,
                f"replica r{replica_seed} ledger fit seed is invalid")
        replica_chains.append(validate_chain(
            group, origin=central, terminal_tag=tag,
            seed=expected_fit_seed, replica_seed=replica_seed,
            barrier_ceiling=expected_replica_ceiling,
            label=f"replica_r{replica_seed}",
            require_recorded_restart_inputs=True))
        replica_gates.append({
            "replica_seed": replica_seed,
            **recompute_adaptive_chain_gates(
                group, origin=central, replica_seed=replica_seed,
                fit_quality_ceiling=expected_replica_ceiling,
                label=f"replica_r{replica_seed}"),
        })
    require(all(item["legacy_pre_receipt_checkpoint_count"] == 0
                for item in replica_chains),
            "one or more replica chains lack launch-time ancestry receipts")
    require(len(central_group) + sum(
                item["checkpoint_count"] for item in replica_chains)
            == len(ledger),
            "central/replica ledger contains rows outside audited chains")
    central_chain["initializer_selection"] = initializer_selection
    central_chain["independently_recomputed_terminal_gates"] = central_gate
    for chain, gates in zip(replica_chains, replica_gates, strict=True):
        chain["independently_recomputed_terminal_gates"] = gates
    validate_declared_central_replica_gates(
        replicas, central_gate, replica_gates, tags)
    return central_chain, replica_chains


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--starts-only", action="store_true",
        help=(
            "publish the terminal 24-start ancestry/medoid manifest without "
            "requiring central or replica outputs"))
    args = parser.parse_args()
    terminal_start_audit = build_terminal_start_audit()
    if args.starts_only:
        print(json.dumps(terminal_start_audit, indent=2))
        return

    source_summary = json.loads(START_SOURCE_ENSEMBLE.read_text())
    starts = json.loads(STARTS.read_text())
    replicas = json.loads(REPLICAS.read_text())
    source_runs, source_records = validate_lambda300_sources(source_summary)
    start_chains = validate_start_chains(starts, source_runs)
    central_chain, replica_chains = validate_central_and_replica_chains(replicas)
    total_checkpoints = (
        sum(item["checkpoint_count"] for item in start_chains)
        + central_chain["checkpoint_count"]
        + sum(item["checkpoint_count"] for item in replica_chains)
    )
    summary = {
        "status": "pass",
        "fixed_challenger_protocol": str(FIXED_PROTOCOL),
        "fixed_challenger_protocol_sha256":
            terminal_start_audit["fixed_challenger_protocol_sha256"],
        "fixed_fnp_reference": str(FIXED_FNP_REFERENCE),
        "fixed_fnp_reference_sha256": EXPECTED_FNP_REFERENCE_SHA256,
        **fixed_implementation_binding(),
        "audit_scope": (
            "exact ordered lambda300-to-lambda600 start ancestry, declared "
            "start-to-central ancestry, and central-to-50-replica ancestry"
        ),
        "selected_prescription": {
            "reference_strength": STRENGTH,
            "reference_b_min": 0.1,
            "reference_b_max": BMAX,
            "fit_quality_barrier_strength": BARRIER_STRENGTH,
            "fit_quality_barrier_power": BARRIER_POWER,
            "reference": str(REFERENCE),
        },
        "required_endpoint_files": list(REQUIRED_ENDPOINT_FILES),
        "lambda300_source_count": len(source_records),
        "start_chain_count": len(start_chains),
        "central_chain_count": 1,
        "experimental_replica_chain_count": len(replica_chains),
        "total_continuation_checkpoint_count": total_checkpoints,
        "terminal_start_ancestry_audit": str(
            START_ONLY_TARGET / "summary.json"),
        "terminal_start_ancestry_audit_sha256": sha256(
            START_ONLY_TARGET / "summary.json"),
        "central_initializer_selection": terminal_start_audit[
            "central_initializer_selection"],
        "lambda300_sources": source_records,
        "start_chains": start_chains,
        "central_chain": central_chain,
        "replica_chains": replica_chains,
        "input_ledger_sha256": {
            str(START_LEDGER): sha256(START_LEDGER),
            str(REPLICA_LEDGER): sha256(REPLICA_LEDGER),
        },
        "input_protocol_sha256": {
            str(FIXED_PROTOCOL):
                terminal_start_audit["fixed_challenger_protocol_sha256"],
            str(FIXED_FNP_REFERENCE): EXPECTED_FNP_REFERENCE_SHA256,
            **{
                str(FIXED_IMPLEMENTATION_FILES[role]): digest
                for role, digest in EXPECTED_IMPLEMENTATION_SHA256.items()
            },
        },
        "production_sources_modified": False,
    }
    atomic_write_json(TARGET / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
