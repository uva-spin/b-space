#!/usr/bin/env python3
"""Validate the centered-log-FNP product model with nested refits.

This is an isolated, cache-resumable validation of the final lambda=600
uncertainty construction.  It is deliberately downstream of the exact 24
central-data starts, the selected 300k-capacity central trajectory, all 50
experimental-replica trajectories, and the final combined-ensemble audit.

Two noncentral start endpoints spanning the leading start log-FNP residual
direction are crossed with three pseudo-data replicas spanning the leading
experimental log-FNP residual direction.  The resulting six fits use the exact
locked lambda=600, bmax=4, fit-barrier mu=100/power=2, float64 objective and
exhaust sixty 5k-capacity LBFGS continuation blocks.  Their nonadditive
interaction is

    I_ij(b) = log F_actual,ij(b)
              - [log C(b) + s_i(b) + r_j(b)],

where s_i and r_j are the same pointwise-median-centered whole-curve
residuals used by ``build_final_combined_tmd_ensemble.py``.

The 2x3 design cannot calibrate a probability distribution for interactions.
It instead supplies a directional observed-interaction envelope.  Materiality
has no free numerical tolerance: the envelope is material exactly when it
changes the pre-registered, flavor-local incumbent replacement decision after
the already-declared finite-ensemble allowance is paid.

No frozen or existing fit is modified.  All new endpoints, logs, ledgers, and
summaries use the ``nestedint_`` namespace below the campaign directory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from fixed_challenger_protocol import (
    EXPECTED_FNP_REFERENCE_SHA256,
    FNP_REFERENCE as FIXED_FNP_REFERENCE,
    LOCKED_INCUMBENT_WIDTHS,
    PROTOCOL as FIXED_PROTOCOL,
    fixed_implementation_binding,
    require_fixed_implementation_binding,
    validate_fixed_challenger_protocol,
)
from postfit_tail_transform_validation import validated_postfit_tail_audit
import supervise_selected_reference_central_replicas as replica_supervisor


BASE = Path(__file__).resolve().parents[1]
SYSTEMATICS = BASE.parent
ROOT = SYSTEMATICS.parent
UNITARY = SYSTEMATICS / "high_qt_direct_production_benchmark/experimental_unitary_transition"
RUNNER = UNITARY / "scripts/run_production_fnp_stability_control.py"
SOURCE_PRODUCTION = (
    SYSTEMATICS / "collins_factorization_validity/outputs/"
    "rowidfix_stageFT_all_qmax0p20_lam0p50_central_s303"
)
W_GRID = (
    ROOT / "outputs/v23a_tevatron_plus_lhcb7_fidacc_merged_cache/"
    "backend_cache/wpert_v23a_tevatron_plus_lhcb7_fidacc_b160.csv"
)
REFERENCE_B = (
    SYSTEMATICS / "collins_factorization_validity/plots/"
    "rowidfix_stageFT_all_qmax0p20_lam0p50_central_exactx/"
    "v22_scheme_tmd_bspace_long.csv"
)
TRANSFORMER = ROOT / "construct_v23a_regularized_kspace_tmd_v2.py"
FROZEN_MANIFEST = BASE / "manifests/input_files.json"
SELECTED = BASE / "summaries/replica_robust_reference_full24/summary.json"
REPLICAS = BASE / "summaries/selected_reference_central_replicas/summary.json"
STATE_CHAIN_AUDIT = BASE / "summaries/lambda600_state_chain_audit/summary.json"
FINAL_ENSEMBLE = BASE / "summaries/final_combined_tmd_ensemble"
FINAL_AUDIT = BASE / "summaries/final_combined_ensemble_stability/summary.json"
POSTFIT_TAIL_AUDIT = (
    BASE / "summaries/lambda600_postfit_tail_transform_audit/summary.json"
)
INCUMBENT = (
    BASE / "summaries/champion_registry/"
    "empirical_reference_lambda1_b0p1_2p0_full24.json"
)
OUTPUTS = BASE / "outputs"
LOGS = BASE / "logs"
TARGET = BASE / "summaries/lambda600_nested_start_replica_interaction"
PYTHON = Path("/home/dustin/miniforge3/envs/pdf-fit/bin/python")

STRENGTH = 600.0
BMAX = 4.0
BMIN = 0.1
BARRIER_STRENGTH = 100.0
BARRIER_POWER = 2
MINIMUM_CAPACITY = 200_000
MAX_CHUNKS = 60
BLOCK_CAPACITY = 5_000
REQUIRED_QUIET_BLOCKS = 10
DRIFT_GATE = 0.02
SENSITIVITY_TRIGGER = 0.01
EPS = 1.0e-30
FLAVORS = ("u", "d")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def float_array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(values, dtype="<f8").tobytes()
    ).hexdigest()


def explicit_bool(value: object, label: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise RuntimeError(f"{label} is not an explicit boolean")


def exact_number(value: object, expected: float, label: str) -> float:
    try:
        observed = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{label} is not numeric") from error
    if not (math.isfinite(observed) and math.isclose(
            observed, expected, rel_tol=0.0, abs_tol=1.0e-12)):
        raise RuntimeError(f"{label}={observed!r}, expected {expected!r}")
    return observed


def atomic_write_text(path: Path, content: str) -> None:
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


def read_json(path: Path, label: str) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"{label} is missing: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} is not a JSON object: {path}")
    return payload


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_registered_frozen_inputs(paths: Iterable[Path]) -> dict[str, dict]:
    manifest = read_json(FROZEN_MANIFEST, "frozen-input manifest")
    registered = manifest.get("files", {})
    checked: dict[str, dict] = {}
    for path in paths:
        resolved = str(path.resolve())
        if resolved not in registered:
            raise RuntimeError(f"required frozen input is unregistered: {path}")
        expected = registered[resolved]
        observed_hash = sha256(path)
        observed_bytes = path.stat().st_size
        if (observed_hash != expected["sha256"]
                or observed_bytes != int(expected["bytes"])):
            raise RuntimeError(f"registered frozen input changed: {path}")
        checked[resolved] = {
            "sha256": observed_hash,
            "bytes": observed_bytes,
            "registered_immutable_input": True,
        }
    return checked


def require_float64_state(path: Path) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"model state is not a nonempty mapping: {path}")
    floating = [value for value in state.values()
                if isinstance(value, torch.Tensor) and value.is_floating_point()]
    if not floating or any(value.dtype != torch.float64 for value in floating):
        raise RuntimeError(f"model state is not entirely float64: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "floating_tensor_count": len(floating),
        "dtype": "torch.float64",
    }


def fnp_curve(run: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(run / "fnp_grid.csv")
    if not {"x", "bT", "F_NP"}.issubset(frame.columns):
        raise RuntimeError(f"FNP grid schema is incomplete: {run.name}")
    frame = frame[np.isclose(frame["x"], 0.1)].sort_values("bT")
    b = frame["bT"].to_numpy(float)
    value = frame["F_NP"].to_numpy(float)
    if (len(b) < 3 or len(np.unique(b)) != len(b)
            or not np.all(np.diff(b) > 0.0)
            or not np.all(np.isfinite(b))
            or not np.all(np.isfinite(value))
            or np.any(value <= 0.0)):
        raise RuntimeError(f"invalid full x=0.1 FNP grid: {run.name}")
    return b, value


def curves_on_common_grid(runs: list[Path], label: str) -> tuple[np.ndarray, np.ndarray]:
    grid: np.ndarray | None = None
    values: list[np.ndarray] = []
    for run in runs:
        b, curve = fnp_curve(run)
        if grid is None:
            grid = b
        elif b.shape != grid.shape or not np.array_equal(b, grid):
            raise RuntimeError(f"{label} grid differs in {run.name}")
        values.append(curve)
    if grid is None:
        raise RuntimeError(f"{label} has no curves")
    return grid, np.asarray(values, dtype=float)


def oriented_leading_scores(log_values: np.ndarray) -> dict:
    """Return deterministic PC1 scores about the pointwise log median."""
    values = np.asarray(log_values, dtype=float)
    if (values.ndim != 2 or values.shape[0] < 3 or values.shape[1] < 3
            or not np.all(np.isfinite(values))):
        raise RuntimeError("leading-direction input is invalid")
    center = np.median(values, axis=0)
    residual = values - center[None, :]
    _, singular, directions = np.linalg.svd(residual, full_matrices=False)
    if not np.isfinite(singular[0]) or singular[0] <= 0.0:
        raise RuntimeError("leading residual direction has exactly zero variation")
    direction = directions[0].copy()
    pivot = int(np.argmax(np.abs(direction)))
    if direction[pivot] < 0.0:
        direction *= -1.0
    scores = residual @ direction
    total = float(np.sum(singular ** 2))
    fraction = float(singular[0] ** 2 / total) if total > 0.0 else 0.0
    return {
        "center": center,
        "residuals": residual,
        "direction": direction,
        "scores": scores,
        "leading_variance_fraction": fraction,
    }


def _ordered_indices(scores: np.ndarray, identities: list[int], allowed: set[int]) -> list[int]:
    return sorted(allowed, key=lambda index: (float(scores[index]), identities[index]))


def select_start_extremes(log_values: np.ndarray, seeds: list[int],
                          excluded_index: int) -> tuple[list[int], dict]:
    if len(seeds) != len(log_values) or len(set(seeds)) != len(seeds):
        raise RuntimeError("start identities do not match the curve matrix")
    if excluded_index not in range(len(seeds)):
        raise RuntimeError("excluded central-initializer index is invalid")
    leading = oriented_leading_scores(log_values)
    allowed = set(range(len(seeds))) - {excluded_index}
    ordered = _ordered_indices(leading["scores"], seeds, allowed)
    chosen = [ordered[0], ordered[-1]]
    if (chosen[0] == chosen[1]
            or not float(leading["scores"][chosen[0]])
                < float(leading["scores"][chosen[1]])):
        raise RuntimeError("two distinct noncentral start extremes were not found")
    metadata = {
        "rule": (
            "minimum and maximum projection on deterministic PC1 of the 24 "
            "pointwise-log-median-centered full-grid start curves, excluding "
            "the declared central initializer"
        ),
        "leading_variance_fraction": leading["leading_variance_fraction"],
        "direction_sha256": float_array_sha256(leading["direction"]),
        "scores_by_seed": {
            str(seed): float(score)
            for seed, score in zip(seeds, leading["scores"], strict=True)
        },
        "excluded_seed": seeds[excluded_index],
        "selected_seeds_low_high": [seeds[index] for index in chosen],
    }
    return chosen, metadata


def select_replica_span(log_values: np.ndarray, identities: list[int]) -> tuple[list[int], dict]:
    if len(identities) != len(log_values) or len(set(identities)) != len(identities):
        raise RuntimeError("replica identities do not match the curve matrix")
    leading = oriented_leading_scores(log_values)
    all_indices = set(range(len(identities)))
    ordered = _ordered_indices(leading["scores"], identities, all_indices)
    low, high = ordered[0], ordered[-1]
    remaining = all_indices - {low, high}
    score_median = float(np.median(leading["scores"]))
    middle = min(
        remaining,
        key=lambda index: (
            abs(float(leading["scores"][index]) - score_median),
            identities[index],
        ),
    )
    chosen = [low, middle, high]
    if not float(leading["scores"][low]) < float(leading["scores"][high]):
        raise RuntimeError("replica residual direction has no distinct span")
    metadata = {
        "rule": (
            "minimum, median-nearest, and maximum projection on deterministic "
            "PC1 of the 50 pointwise-log-median-centered full-grid primary "
            "experimental-replica curves"
        ),
        "leading_variance_fraction": leading["leading_variance_fraction"],
        "direction_sha256": float_array_sha256(leading["direction"]),
        "score_median": score_median,
        "scores_by_replica": {
            str(identity): float(score)
            for identity, score in zip(
                identities, leading["scores"], strict=True)
        },
        "selected_replicas_low_middle_high": [
            identities[index] for index in chosen
        ],
    }
    return chosen, metadata


def additive_log_prediction(log_central: np.ndarray,
                            all_start_logs: np.ndarray,
                            all_replica_logs: np.ndarray,
                            start_index: int, replica_index: int) -> np.ndarray:
    start_center = np.median(all_start_logs, axis=0)
    replica_center = np.median(all_replica_logs, axis=0)
    return (log_central
            + all_start_logs[start_index] - start_center
            + all_replica_logs[replica_index] - replica_center)


def new_tracker() -> dict:
    return {
        "anchor": None,
        "anchor_capacity": None,
        "consecutive": 0,
        "sensitivity_active": False,
        "fresh_after_sensitivity": 0,
    }


def update_stationarity_tracker(tracker: dict, previous: np.ndarray,
                                current: np.ndarray, cumulative: int) -> dict:
    """Apply the fixed 200k-anchor/ten-block rule to one curve domain."""
    previous = np.asarray(previous, dtype=float)
    current = np.asarray(current, dtype=float)
    drift = float(np.max(np.abs(current - previous) / np.maximum(previous, 0.05)))
    if cumulative == MINIMUM_CAPACITY:
        tracker["anchor"] = current.copy()
        tracker["anchor_capacity"] = cumulative
    eligible = cumulative > MINIMUM_CAPACITY
    tested_anchor = tracker["anchor_capacity"]
    window_drift = (
        float(np.max(
            np.abs(current - tracker["anchor"])
            / np.maximum(tracker["anchor"], 0.05)
        ))
        if eligible and tracker["anchor"] is not None else None
    )
    quiet = bool(
        eligible and drift <= DRIFT_GATE
        and window_drift is not None and window_drift <= DRIFT_GATE
    )
    if not eligible:
        tracker["consecutive"] = 0
    elif not quiet:
        tracker["anchor"] = current.copy()
        tracker["anchor_capacity"] = cumulative
        tracker["consecutive"] = 0
        tracker["sensitivity_active"] = False
        tracker["fresh_after_sensitivity"] = 0
    else:
        tracker["consecutive"] += 1
    triggered = (
        quiet and drift >= SENSITIVITY_TRIGGER
        and not tracker["sensitivity_active"]
    )
    if triggered:
        tracker["sensitivity_active"] = True
        tracker["fresh_after_sensitivity"] = 0
    elif tracker["sensitivity_active"]:
        tracker["fresh_after_sensitivity"] = (
            tracker["fresh_after_sensitivity"] + 1 if quiet else 0
        )
    confirmation_count = (
        tracker["fresh_after_sensitivity"]
        if tracker["sensitivity_active"] else tracker["consecutive"]
    )
    stationarity = bool(
        cumulative >= MINIMUM_CAPACITY + REQUIRED_QUIET_BLOCKS * BLOCK_CAPACITY
        and confirmation_count >= REQUIRED_QUIET_BLOCKS
        and window_drift is not None and window_drift <= DRIFT_GATE
    )
    return {
        "drift": drift,
        "eligible": eligible,
        "window_drift": window_drift,
        "tested_anchor_capacity": tested_anchor,
        "next_anchor_capacity": tracker["anchor_capacity"],
        "quiet": quiet,
        "consecutive": tracker["consecutive"],
        "sensitivity_active": tracker["sensitivity_active"],
        "fresh_after_sensitivity": tracker["fresh_after_sensitivity"],
        "stationarity_pass": stationarity,
    }


def interaction_tag(start_seed: int, replica_seed: int, cumulative: int) -> str:
    return (
        f"nestedint_fullref_lam600_fitbar_p2_mu100_b4_"
        f"s{start_seed}_r{replica_seed}_polish64_{cumulative}"
    )


def validate_preflight() -> dict:
    _, protocol_hash = validate_fixed_challenger_protocol()
    selected = read_json(SELECTED, "exact24 summary")
    replicas = read_json(REPLICAS, "central/full50 summary")
    state_chain = read_json(STATE_CHAIN_AUDIT, "lambda600 state-chain audit")
    final_summary = read_json(FINAL_ENSEMBLE / "summary.json", "final ensemble summary")
    final_audit = read_json(FINAL_AUDIT, "final ensemble audit")
    postfit_tail, postfit_tail_hash = validated_postfit_tail_audit(
        POSTFIT_TAIL_AUDIT, FINAL_AUDIT
    )
    incumbent = read_json(INCUMBENT, "immutable incumbent")

    start_tags = [str(value) for value in selected.get("endpoint_tags", [])]
    replica_tags = [str(value) for value in replicas.get("replica_endpoint_tags", [])]
    if (selected.get("status") not in {"complete", "verification_failed"}
            or int(selected.get("member_count", -1)) != 24
            or len(start_tags) != 24 or len(set(start_tags)) != 24):
        raise RuntimeError("exact24 terminal evidence is incomplete")
    if (replicas.get("status") not in {
            "complete", "complete_with_scientific_failures",
            "central_stationarity_failed", "replica_stationarity_failed"}
            or int(replicas.get("completed_replica_count", -1)) != 50
            or len(replica_tags) != 50 or len(set(replica_tags)) != 50
            or not replicas.get("central_endpoint_tag")):
        raise RuntimeError("selected central/full50 terminal evidence is incomplete")
    for label, payload in (("exact24", selected), ("full50", replicas)):
        exact_number(payload.get("selected_strength"), STRENGTH,
                     f"{label} strength")
        exact_number(payload.get("selected_bmax"), BMAX, f"{label} bmax")
        exact_number(payload.get("fit_quality_barrier_strength"),
                     BARRIER_STRENGTH, f"{label} barrier strength")
        exact_number(payload.get("fit_quality_barrier_power"),
                     BARRIER_POWER, f"{label} barrier power")
    if (not explicit_bool(replicas.get("central_full_horizon_complete"),
                          "central full horizon")
            or int(replicas.get("central_terminal_requested_capacity", -1))
                != MAX_CHUNKS * BLOCK_CAPACITY):
        raise RuntimeError("selected central did not complete its full horizon")

    require_fixed_implementation_binding(state_chain, "state-chain audit")
    prescription = state_chain.get("selected_prescription", {})
    if (state_chain.get("status") != "pass"
            or int(state_chain.get("start_chain_count", -1)) != 24
            or int(state_chain.get("central_chain_count", -1)) != 1
            or int(state_chain.get("experimental_replica_chain_count", -1)) != 50
            or Path(state_chain.get("fixed_challenger_protocol", "")).resolve()
                != FIXED_PROTOCOL.resolve()
            or state_chain.get("fixed_challenger_protocol_sha256") != protocol_hash
            or Path(state_chain.get("fixed_fnp_reference", "")).resolve()
                != FIXED_FNP_REFERENCE.resolve()
            or state_chain.get("fixed_fnp_reference_sha256")
                != EXPECTED_FNP_REFERENCE_SHA256):
        raise RuntimeError("lambda600 state-chain evidence is incomplete or stale")
    audited_start_tags = [str(item.get("terminal_tag"))
                          for item in state_chain.get("start_chains", [])]
    audited_replica_tags = [str(item.get("terminal_tag"))
                            for item in state_chain.get("replica_chains", [])]
    audited_central_tag = str(
        state_chain.get("central_chain", {}).get("terminal_tag", ""))
    if (audited_start_tags != start_tags
            or audited_replica_tags != replica_tags
            or audited_central_tag != str(replicas["central_endpoint_tag"])):
        raise RuntimeError(
            "state-chain terminal tags differ from the selected evidence")
    exact_number(prescription.get("reference_strength"), STRENGTH,
                 "state-chain strength")
    exact_number(prescription.get("reference_b_max"), BMAX,
                 "state-chain bmax")
    exact_number(prescription.get("fit_quality_barrier_strength"),
                 BARRIER_STRENGTH, "state-chain barrier strength")
    exact_number(prescription.get("fit_quality_barrier_power"),
                 BARRIER_POWER, "state-chain barrier power")

    if (final_summary.get("status") != "complete"
            or int(final_summary.get("start_count", -1)) != 24
            or int(final_summary.get("experimental_replica_count", -1)) != 50
            or int(final_summary.get("combined_member_count", -1)) != 1200
            or final_audit.get("status") != "complete"
            or int(final_audit.get("start_count", -1)) != 24
            or int(final_audit.get("replica_count", -1)) != 50):
        raise RuntimeError("final 24x50 ensemble/audit is incomplete")
    if incumbent.get("champion_id") != "empirical_reference_lambda1_b0p1_2p0_full24":
        raise RuntimeError("immutable incumbent identity changed")
    for flavor in FLAVORS:
        exact_number(
            incumbent["combined_fig6_max_active_relative_full_width"][flavor],
            LOCKED_INCUMBENT_WIDTHS[flavor],
            f"locked incumbent {flavor} width",
        )

    frozen = validate_registered_frozen_inputs((W_GRID, REFERENCE_B, TRANSFORMER))
    return {
        "protocol_sha256": protocol_hash,
        "selected": selected,
        "replicas": replicas,
        "state_chain": state_chain,
        "final_summary": final_summary,
        "final_audit": final_audit,
        "postfit_tail": postfit_tail,
        "postfit_tail_hash": postfit_tail_hash,
        "incumbent": incumbent,
        "start_tags": start_tags,
        "replica_tags": replica_tags,
        "frozen_inputs": frozen,
    }


def select_design(evidence: dict) -> dict:
    start_runs = [OUTPUTS / tag for tag in evidence["start_tags"]]
    replica_runs = [OUTPUTS / tag for tag in evidence["replica_tags"]]
    central_run = OUTPUTS / evidence["replicas"]["central_endpoint_tag"]
    b, starts = curves_on_common_grid(start_runs, "start")
    rb, replicas = curves_on_common_grid(replica_runs, "primary replica")
    cb, central = fnp_curve(central_run)
    if not (np.array_equal(b, rb) and np.array_equal(b, cb)):
        raise RuntimeError("start, central, and replica full FNP grids differ")

    start_statuses = [read_json(run / "fit_status.json", run.name)
                      for run in start_runs]
    replica_statuses = [read_json(run / "fit_status.json", run.name)
                        for run in replica_runs]
    start_seeds = [int(status["seed"]) for status in start_statuses]
    replica_ids = [int(status["replica_seed"]) for status in replica_statuses]
    if start_seeds != list(range(303, 327)):
        raise RuntimeError("start identities are not exact seeds 303--326")
    if replica_ids != list(range(1001, 1051)):
        raise RuntimeError("replica identities are not exact seeds 1001--1050")
    for run in start_runs:
        require_float64_state(run / "model_state.pt")
    for index, (run, replica_id) in enumerate(
            zip(replica_runs, replica_ids, strict=True)):
        replica_supervisor.require_endpoint_objective(
            run.name,
            strength=STRENGTH,
            bmax=BMAX,
            barrier_strength=BARRIER_STRENGTH,
            barrier_power=BARRIER_POWER,
            replica_seed=replica_id,
            fit_seed=2001 + index,
        )
        require_float64_state(run / "model_state.pt")
    central_status = read_json(
        central_run / "fit_status.json", central_run.name)
    initializer = OUTPUTS / evidence["replicas"]["central_initializer_tag"]
    initializer_status = read_json(
        initializer / "fit_status.json", initializer.name)
    central_ceiling = (
        float(initializer_status["final"]["unpenalized_total_chi2"])
        + math.sqrt(2.0 * int(initializer_status["row_count"]))
    )
    replica_supervisor.require_endpoint_objective(
        central_run.name,
        strength=STRENGTH,
        bmax=BMAX,
        barrier_strength=BARRIER_STRENGTH,
        barrier_power=BARRIER_POWER,
        replica_seed=None,
        fit_seed=int(central_status["seed"]),
        expected_barrier_ceiling=central_ceiling,
    )
    require_float64_state(central_run / "model_state.pt")
    initializer_tag = str(evidence["replicas"]["central_initializer_tag"])
    try:
        excluded = evidence["start_tags"].index(initializer_tag)
    except ValueError as error:
        raise RuntimeError("central initializer is outside exact24") from error

    start_indices, start_selection = select_start_extremes(
        np.log(starts), start_seeds, excluded)
    replica_indices, replica_selection = select_replica_span(
        np.log(replicas), replica_ids)
    pairs = [
        {
            "start_index": start_index,
            "start_seed": start_seeds[start_index],
            "start_tag": start_runs[start_index].name,
            "replica_index": replica_index,
            "replica_seed": replica_ids[replica_index],
            "primary_replica_tag": replica_runs[replica_index].name,
        }
        for start_index in start_indices
        for replica_index in replica_indices
    ]
    if len(pairs) != 6 or len({
            (row["start_seed"], row["replica_seed"]) for row in pairs}) != 6:
        raise RuntimeError("nested design is not an exact 2x3 Cartesian set")
    return {
        "b_grid": b,
        "starts": starts,
        "replicas": replicas,
        "central": central,
        "start_runs": start_runs,
        "replica_runs": replica_runs,
        "central_run": central_run,
        "start_seeds": start_seeds,
        "replica_ids": replica_ids,
        "start_selection": start_selection,
        "replica_selection": replica_selection,
        "pairs": pairs,
    }


def objective_slice(b: np.ndarray, values: np.ndarray) -> np.ndarray:
    mask = (b >= BMIN - 1.0e-12) & (b <= BMAX + 1.0e-12)
    if int(np.sum(mask)) < 3:
        raise RuntimeError("objective-domain FNP grid is incomplete")
    return values[mask]


def run_nested_pair(pair: dict, b_grid: np.ndarray,
                    records: list[dict], execute: bool) -> tuple[Path | None, bool]:
    start = OUTPUTS / pair["start_tag"]
    previous = start
    initial_status = read_json(start / "fit_status.json", start.name)
    row_count = int(initial_status["row_count"])
    fit_quality_ceiling = row_count + 5.0 * math.sqrt(2.0 * row_count)
    objective_tracker = new_tracker()
    full_tracker = new_tracker()
    terminal_pass = False
    if not execute:
        return None, False

    for chunk in range(1, MAX_CHUNKS + 1):
        cumulative = chunk * BLOCK_CAPACITY
        tag = interaction_tag(
            pair["start_seed"], pair["replica_seed"], cumulative)
        target = OUTPUTS / tag
        initial_state = previous / "model_state.pt"
        initial_norms = previous / "dataset_norms.csv"
        initial_state_hash = sha256(initial_state)
        initial_norms_hash = sha256(initial_norms)
        require_float64_state(initial_state)
        if not (target / "fit_status.json").exists():
            command = [
                str(PYTHON), str(RUNNER),
                "--seed", str(pair["start_seed"]),
                "--source-production", str(SOURCE_PRODUCTION),
                "--w-grid", str(W_GRID),
                "--output-root", str(OUTPUTS),
                "--tag", tag,
                "--initial-state", str(initial_state),
                "--initial-norms", str(initial_norms),
                "--max-epochs", "0", "--min-epochs", "0",
                "--plateau-patience", "0",
                "--lbfgs-max-iter", str(BLOCK_CAPACITY),
                "--float64",
                "--lambda-fnp-reference-distance", str(STRENGTH),
                "--fnp-reference-distance-csv", str(FIXED_FNP_REFERENCE),
                "--fnp-reference-distance-bmin", str(BMIN),
                "--fnp-reference-distance-bmax", str(BMAX),
                "--fit-quality-ceiling-total-chi2", str(fit_quality_ceiling),
                "--lambda-fit-quality-barrier", str(BARRIER_STRENGTH),
                "--fit-quality-barrier-power", str(BARRIER_POWER),
                "--replica-seed", str(pair["replica_seed"]),
            ]
            LOGS.mkdir(parents=True, exist_ok=True)
            validate_fixed_challenger_protocol()
            with (LOGS / f"{tag}.log").open("w", encoding="utf-8") as stream:
                subprocess.run(
                    command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        if sha256(initial_state) != initial_state_hash:
            raise RuntimeError(f"initializer state changed while fitting {tag}")
        if sha256(initial_norms) != initial_norms_hash:
            raise RuntimeError(f"initializer norms changed while fitting {tag}")

        status = replica_supervisor.require_endpoint_objective(
            tag,
            strength=STRENGTH,
            bmax=BMAX,
            barrier_strength=BARRIER_STRENGTH,
            barrier_power=BARRIER_POWER,
            replica_seed=pair["replica_seed"],
            fit_seed=pair["start_seed"],
            expected_initial_state=initial_state,
        )
        precision = require_float64_state(target / "model_state.pt")
        old_b, old_values = fnp_curve(previous)
        new_b, new_values = fnp_curve(target)
        if not (np.array_equal(old_b, b_grid)
                and np.array_equal(new_b, b_grid)):
            raise RuntimeError(f"nested trajectory FNP grid changed: {tag}")
        objective = update_stationarity_tracker(
            objective_tracker,
            objective_slice(b_grid, old_values),
            objective_slice(b_grid, new_values),
            cumulative,
        )
        full = update_stationarity_tracker(
            full_tracker, old_values, new_values, cumulative)
        unpenalized = float(status["final"]["unpenalized_total_chi2"])
        replica_ceiling = row_count + 5.0 * math.sqrt(2.0 * row_count)
        fit_pass = unpenalized <= fit_quality_ceiling
        replica_chi2_pass = unpenalized <= replica_ceiling
        terminal_pass = bool(
            objective["stationarity_pass"] and full["stationarity_pass"]
            and fit_pass and replica_chi2_pass
        )
        records.append({
            "start_seed": pair["start_seed"],
            "replica_seed": pair["replica_seed"],
            "primary_replica_tag": pair["primary_replica_tag"],
            "initial_state_path": str(initial_state.resolve()),
            "initial_state_sha256": initial_state_hash,
            "initial_norms_path": str(initial_norms.resolve()),
            "initial_norms_sha256": initial_norms_hash,
            "fit_target_sha256": status["_validated_fit_target_sha256"],
            "cumulative_requested_lbfgs_capacity": cumulative,
            "requested_lbfgs_max_iterations_this_block": BLOCK_CAPACITY,
            "executed_lbfgs_closure_evaluations_this_block": int(
                status["lbfgs"]["closure_evaluations"]),
            "objective_domain_drift_from_previous": objective["drift"],
            "objective_domain_window_drift": objective["window_drift"],
            "objective_domain_window_anchor_capacity": objective[
                "tested_anchor_capacity"],
            "objective_domain_next_anchor_capacity": objective[
                "next_anchor_capacity"],
            "objective_domain_consecutive_quiet_blocks": objective["consecutive"],
            "objective_domain_stationarity_pass": objective["stationarity_pass"],
            "full_grid_drift_from_previous": full["drift"],
            "full_grid_window_drift": full["window_drift"],
            "full_grid_window_anchor_capacity": full["tested_anchor_capacity"],
            "full_grid_next_anchor_capacity": full["next_anchor_capacity"],
            "full_grid_consecutive_quiet_blocks": full["consecutive"],
            "full_grid_stationarity_pass": full["stationarity_pass"],
            "unpenalized_total_chi2": unpenalized,
            "fit_quality_ceiling_total_chi2": fit_quality_ceiling,
            "fit_quality_gate_pass": fit_pass,
            "replica_chi2_sanity_ceiling": replica_ceiling,
            "replica_chi2_sanity_pass": replica_chi2_pass,
            "float64_state_pass": precision["dtype"] == "torch.float64",
            "terminal_interaction_endpoint_acceptance_pass": terminal_pass,
            "tag": tag,
        })
        atomic_write_csv(TARGET / "runs.csv", pd.DataFrame(records))
        previous = target
    return previous, terminal_pass


def transform_logs(log_curves: np.ndarray, keys: list[tuple[int, int]],
                   component: str, b_grid: np.ndarray) -> tuple[pd.DataFrame, dict]:
    if len(log_curves) != len(keys):
        raise RuntimeError("transform curve/key counts differ")
    reference = pd.read_csv(REFERENCE_B)
    reference = reference[
        np.isclose(reference["x"], 0.1)
        & np.isclose(reference["Q"], 10.0)
        & reference["flavor"].astype(str).isin(FLAVORS)
    ].copy()
    if reference.groupby("flavor").size().to_dict() != {"d": 321, "u": 321}:
        raise RuntimeError("frozen u/d Q10 reference grid is incomplete")
    source_b = np.sort(reference["bT"].unique())
    if (source_b.shape != np.asarray(b_grid).shape
            or not np.array_equal(source_b, np.asarray(b_grid, dtype=float))
            or log_curves.shape[1] != len(source_b)):
        raise RuntimeError(
            "interaction FNP grid does not exactly match the frozen TMD grid")
    rows: list[pd.DataFrame] = []
    for (start_seed, replica_seed), log_curve in zip(keys, log_curves, strict=True):
        for (_, flavor), group in reference.groupby(["pid", "flavor"], sort=False):
            group = group.sort_values("bT").copy()
            b = group["bT"].to_numpy(float)
            if b.shape != source_b.shape or not np.array_equal(b, source_b):
                raise RuntimeError(f"reference b grid differs for {flavor}")
            group["F_NP"] = np.exp(log_curve)
            group["ftilde"] = (
                group["ftilde_no_np"].to_numpy(float)
                * group["F_NP"].to_numpy(float)
            )
            group["_replica_key"] = (
                f"{component}|s{start_seed}|r{replica_seed}")
            group["seed"] = start_seed
            group["pdf_member"] = replica_seed
            rows.append(group)
    frame = pd.concat(rows, ignore_index=True)
    transform = load_module(
        f"nested_interaction_transform_{component}", TRANSFORMER)
    settings = argparse.Namespace(
        quantities=["ftilde"], tail_mode="expb2", tail_fit_bmin=None,
        eps=1.0e-300, b_transform_max=24.0, n_b_transform=6001,
        k_max=4.0, n_k=401, end_taper_start_fraction=0.92,
    )
    long, metadata = transform.transform_curves(frame, settings)
    long["interaction_component"] = component
    return long, metadata


def normalized_candidate_and_incumbent() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    audit = read_json(FINAL_AUDIT, "final ensemble audit")
    candidate = pd.read_csv(FINAL_ENSEMBLE / "kT_tmd_bands.csv")
    candidate = candidate[
        candidate["component"].astype(str).eq("combined")
        & candidate["quantity"].astype(str).eq("ftilde")
        & np.isclose(candidate["Q"], 10.0)
        & candidate["flavor"].astype(str).isin(FLAVORS)
    ].rename(columns={"median": "central"})
    incumbent_record = read_json(INCUMBENT, "immutable incumbent")
    incumbent_path = Path(incumbent_record["artifacts"]["kspace_combined_bands"])
    if sha256(incumbent_path) != incumbent_record["artifact_sha256"][
            "kspace_combined_bands"]:
        raise RuntimeError("immutable incumbent k-space band changed")
    incumbent = pd.read_csv(incumbent_path)
    if "component" in incumbent.columns:
        incumbent = incumbent[incumbent["component"].astype(str).eq("combined")]
    if "quantity" in incumbent.columns:
        incumbent = incumbent[incumbent["quantity"].astype(str).eq("ftilde")]
    if "Q" in incumbent.columns:
        incumbent = incumbent[np.isclose(incumbent["Q"], 10.0)]
    incumbent = incumbent.rename(columns={"median": "central"})
    return candidate, incumbent, audit


def compute_interaction_decision_impact(
        candidate: pd.DataFrame, incumbent: pd.DataFrame,
        audit: dict, transformed: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    required_components = {"actual", "additive_prediction"}
    if set(transformed["interaction_component"].astype(str)) != required_components:
        raise RuntimeError("interaction transform components are incomplete")
    actual = transformed[
        transformed["interaction_component"].eq("actual")
        & transformed["quantity"].eq("ftilde")
    ]
    predicted = transformed[
        transformed["interaction_component"].eq("additive_prediction")
        & transformed["quantity"].eq("ftilde")
    ]
    joined = actual.merge(
        predicted,
        on=["seed", "pdf_member", "pid", "flavor", "x", "Q", "kT", "quantity"],
        suffixes=("_actual", "_predicted"), validate="one_to_one",
    )
    joined["interaction_delta"] = (
        joined["value_actual"] - joined["value_predicted"])

    rows = []
    metrics: dict[str, dict] = {}
    for flavor in FLAVORS:
        cand = candidate[candidate["flavor"].astype(str).eq(flavor)].sort_values("kT")
        inc = incumbent[incumbent["flavor"].astype(str).eq(flavor)].sort_values("kT")
        if (len(cand) != 401 or len(inc) != 401
                or not np.array_equal(cand["kT"].to_numpy(float),
                                      inc["kT"].to_numpy(float))):
            raise RuntimeError(f"candidate/incumbent k grid differs for {flavor}")
        grid = cand["kT"].to_numpy(float)
        med = cand["central"].to_numpy(float)
        old_med = inc["central"].to_numpy(float)
        display = grid <= 2.25 + 1.0e-12
        candidate_active = display & (med > 0.05 * np.max(med[display]))
        incumbent_active = display & (
            old_med > 0.05 * np.max(old_med[display]))
        active = candidate_active | incumbent_active
        if not np.any(active):
            raise RuntimeError(f"empty comparison mask for {flavor}")
        deltas = joined[joined["flavor"].astype(str).eq(flavor)].pivot(
            index=["seed", "pdf_member"], columns="kT",
            values="interaction_delta",
        ).reindex(columns=grid)
        if deltas.shape != (6, 401) or deltas.isna().any().any():
            raise RuntimeError(f"interaction transform coverage differs for {flavor}")
        matrix = deltas.to_numpy(float)
        low = np.minimum(np.min(matrix, axis=0), 0.0)
        high = np.maximum(np.max(matrix, axis=0), 0.0)
        expanded_low = cand["q16"].to_numpy(float) + low
        expanded_high = cand["q84"].to_numpy(float) + high
        base_width_curve = (
            cand["q84"].to_numpy(float) - cand["q16"].to_numpy(float)
        ) / np.maximum(med, EPS)
        expanded_width_curve = (
            expanded_high - expanded_low
        ) / np.maximum(med, EPS)
        base_width = float(np.max(base_width_curve[active]))
        expanded_width = float(np.max(expanded_width_curve[active]))
        audited_width = float(
            audit["final_max_active_relative_full_width"][flavor])
        if not np.isclose(base_width, audited_width, rtol=1.0e-12,
                          atol=1.0e-14):
            raise RuntimeError(
                f"recomputed candidate width differs from final audit for {flavor}")
        registered = float(LOCKED_INCUMBENT_WIDTHS[flavor])
        allowance = float(
            audit["resampling_full_width_allowance_by_flavor"][flavor])
        if not np.isfinite(allowance) or allowance < 0.0:
            raise RuntimeError(f"invalid finite-ensemble allowance for {flavor}")
        margin_before = registered - (base_width + allowance)
        margin_after = registered - (expanded_width + allowance)
        decision_before = bool(margin_before > 0.0)
        decision_after = bool(margin_after > 0.0)
        metrics[flavor] = {
            "candidate_full_width_before_interaction": base_width,
            "observed_interaction_envelope_expanded_full_width": expanded_width,
            "interaction_full_width_increment": expanded_width - base_width,
            "finite_ensemble_allowance": allowance,
            "locked_incumbent_width": registered,
            "replacement_margin_before_interaction": margin_before,
            "replacement_margin_after_interaction": margin_after,
            "replacement_decision_before_interaction": decision_before,
            "replacement_decision_after_interaction": decision_after,
            "replacement_decision_sign_changed": decision_before != decision_after,
            "max_absolute_transformed_pair_interaction_relative_to_candidate_central":
                float(np.max(np.abs(matrix[:, active])
                             / np.maximum(med[active], EPS)[None, :])),
        }
        rows.append(pd.DataFrame({
            "flavor": flavor,
            "kT": grid,
            "candidate_q16": cand["q16"].to_numpy(float),
            "candidate_central": med,
            "candidate_q84": cand["q84"].to_numpy(float),
            "interaction_delta_low": low,
            "interaction_delta_high": high,
            "interaction_expanded_low": expanded_low,
            "interaction_expanded_high": expanded_high,
            "candidate_active_mask": candidate_active,
            "incumbent_active_mask": incumbent_active,
            "comparison_union_active_mask": active,
        }))
    envelope = pd.concat(rows, ignore_index=True)
    overall_before = all(
        metrics[flavor]["replacement_decision_before_interaction"]
        for flavor in FLAVORS)
    overall_after = all(
        metrics[flavor]["replacement_decision_after_interaction"]
        for flavor in FLAVORS)
    summary = {
        "by_flavor": metrics,
        "overall_width_replacement_decision_before_interaction": overall_before,
        "overall_width_replacement_decision_after_interaction": overall_after,
        "overall_replacement_decision_sign_changed": overall_before != overall_after,
        "materiality_rule": (
            "material exactly when the directional observed-interaction envelope "
            "changes the pre-registered all-flavor replacement decision after "
            "the existing flavor-local finite-ensemble allowance; no separate "
            "numerical interaction tolerance is introduced"
        ),
    }
    return envelope, summary


def write_progress(evidence: dict, design: dict, completed: list[dict],
                   *, status: str, execute: bool) -> None:
    atomic_write_json(TARGET / "summary.json", {
        "status": status,
        "fixed_challenger_protocol": str(FIXED_PROTOCOL),
        "fixed_challenger_protocol_sha256": evidence["protocol_sha256"],
        "fixed_fnp_reference": str(FIXED_FNP_REFERENCE),
        "fixed_fnp_reference_sha256": EXPECTED_FNP_REFERENCE_SHA256,
        **fixed_implementation_binding(),
        "design": {
            "start_selection": design["start_selection"],
            "replica_selection": design["replica_selection"],
            "pairs": design["pairs"],
        },
        "execute_requested": execute,
        "completed_pair_count": len(completed),
        "completed_pairs": completed,
        "required_pair_count": 6,
        "production_sources_modified": False,
    })


def raw_terminal_endpoint_evidence(
    endpoint: Path, start_seed: int, replica_seed: int
) -> dict:
    """Bind a nested terminal endpoint to every raw scientific state file."""
    roles = {
        "fit_status": endpoint / "fit_status.json",
        "fnp_grid": endpoint / "fnp_grid.csv",
        "model_state": endpoint / "model_state.pt",
    }
    norms = endpoint / "dataset_norms.csv"
    if norms.exists():
        roles["dataset_norms"] = norms
    if any(not path.is_file() or path.stat().st_size == 0 for path in roles.values()):
        raise RuntimeError(f"nested terminal raw endpoint is incomplete: {endpoint}")
    return {
        "start_seed": int(start_seed),
        "replica_seed": int(replica_seed),
        "endpoint_tag": endpoint.name,
        "raw_artifacts": {
            role: str(path.resolve()) for role, path in roles.items()
        },
        "raw_artifact_sha256": {
            role: sha256(path) for role, path in roles.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan-only", action="store_true",
        help="validate/select the deterministic 2x3 design without launching fits",
    )
    args = parser.parse_args()
    execute = not args.plan_only
    evidence = validate_preflight()
    design = select_design(evidence)
    TARGET.mkdir(parents=True, exist_ok=True)
    write_progress(
        evidence, design, [], status=(
            "planned_not_executed" if not execute else "in_progress"),
        execute=execute,
    )
    if not execute:
        print((TARGET / "summary.json").read_text(), end="")
        return

    frozen_before = validate_registered_frozen_inputs(
        (W_GRID, REFERENCE_B, TRANSFORMER))
    records: list[dict] = []
    completed: list[dict] = []
    endpoints: list[Path] = []
    acceptance: list[bool] = []
    for pair in design["pairs"]:
        endpoint, passed = run_nested_pair(
            pair, design["b_grid"], records, execute=True)
        if endpoint is None:
            raise RuntimeError("executed nested pair returned no endpoint")
        endpoints.append(endpoint)
        acceptance.append(passed)
        endpoint_evidence = raw_terminal_endpoint_evidence(
            endpoint, pair["start_seed"], pair["replica_seed"]
        )
        completed.append({
            "start_seed": pair["start_seed"],
            "replica_seed": pair["replica_seed"],
            "endpoint_tag": endpoint.name,
            "terminal_acceptance_pass": passed,
            "raw_artifacts": endpoint_evidence["raw_artifacts"],
            "raw_artifact_sha256": endpoint_evidence["raw_artifact_sha256"],
        })
        write_progress(
            evidence, design, completed, status="in_progress", execute=True)
    frozen_after = validate_registered_frozen_inputs(
        (W_GRID, REFERENCE_B, TRANSFORMER))
    if frozen_after != frozen_before:
        raise RuntimeError("registered frozen input evidence changed during validation")

    _, actual_values = curves_on_common_grid(endpoints, "nested actual")
    all_start_logs = np.log(design["starts"])
    all_replica_logs = np.log(design["replicas"])
    log_central = np.log(design["central"])
    predicted_logs = np.asarray([
        additive_log_prediction(
            log_central, all_start_logs, all_replica_logs,
            pair["start_index"], pair["replica_index"],
        )
        for pair in design["pairs"]
    ])
    actual_logs = np.log(actual_values)
    interactions = actual_logs - predicted_logs
    keys = [(pair["start_seed"], pair["replica_seed"])
            for pair in design["pairs"]]

    interaction_rows = []
    for pair, actual, predicted, interaction in zip(
            design["pairs"], actual_logs, predicted_logs, interactions, strict=True):
        interaction_rows.append(pd.DataFrame({
            "start_seed": pair["start_seed"],
            "replica_seed": pair["replica_seed"],
            "bT": design["b_grid"],
            "actual_log_F_NP": actual,
            "additive_predicted_log_F_NP": predicted,
            "interaction_log_F_NP": interaction,
        }))
    interaction_long = pd.concat(interaction_rows, ignore_index=True)
    atomic_write_csv(TARGET / "interaction_logfnp_long.csv", interaction_long)
    log_low = np.minimum(np.min(interactions, axis=0), 0.0)
    log_high = np.maximum(np.max(interactions, axis=0), 0.0)
    atomic_write_csv(TARGET / "interaction_logfnp_directional_envelope.csv", pd.DataFrame({
        "bT": design["b_grid"],
        "interaction_log_delta_low": log_low,
        "interaction_log_delta_high": log_high,
        "max_absolute_interaction_log_delta": np.max(np.abs(interactions), axis=0),
    }))

    actual_long, actual_meta = transform_logs(
        actual_logs, keys, "actual", design["b_grid"])
    predicted_long, predicted_meta = transform_logs(
        predicted_logs, keys, "additive_prediction", design["b_grid"])
    transformed = pd.concat([actual_long, predicted_long], ignore_index=True)
    atomic_write_csv(TARGET / "interaction_kspace_actual_vs_additive_long.csv", transformed)
    candidate, incumbent, audit = normalized_candidate_and_incumbent()
    k_envelope, impact = compute_interaction_decision_impact(
        candidate, incumbent, audit, transformed)
    atomic_write_csv(TARGET / "interaction_kspace_directional_envelope.csv", k_envelope)

    artifacts = {
        "ledger": str(TARGET / "runs.csv"),
        "logfnp_interactions": str(TARGET / "interaction_logfnp_long.csv"),
        "logfnp_directional_envelope": str(
            TARGET / "interaction_logfnp_directional_envelope.csv"),
        "kspace_actual_vs_additive": str(
            TARGET / "interaction_kspace_actual_vs_additive_long.csv"),
        "kspace_directional_envelope": str(
            TARGET / "interaction_kspace_directional_envelope.csv"),
    }
    for endpoint in completed:
        prefix = (
            f"terminal_s{int(endpoint['start_seed'])}_"
            f"r{int(endpoint['replica_seed'])}"
        )
        for role, path in endpoint["raw_artifacts"].items():
            artifacts[f"{prefix}_{role}"] = path
    artifact_sha256 = {
        label: sha256(Path(path)) for label, path in artifacts.items()
    }

    all_trajectories_pass = bool(all(acceptance))
    sign_stable = not impact["overall_replacement_decision_sign_changed"]
    base_audit_gate = explicit_bool(
        audit.get("endpoint_gate_pass"), "base candidate endpoint gate")
    if not all_trajectories_pass:
        status = "nested_trajectory_stationarity_or_fit_failure"
    elif not sign_stable:
        status = "material_nonadditive_interaction_changes_decision"
    else:
        status = "complete_observed_interaction_decision_sign_stable"
    summary = {
        "status": status,
        "fixed_challenger_protocol": str(FIXED_PROTOCOL),
        "fixed_challenger_protocol_sha256": evidence["protocol_sha256"],
        "fixed_fnp_reference": str(FIXED_FNP_REFERENCE),
        "fixed_fnp_reference_sha256": EXPECTED_FNP_REFERENCE_SHA256,
        **fixed_implementation_binding(),
        "method": (
            "deterministic 2x3 nested alternate-start by pseudo-data-replica "
            "validation of the centered whole-curve log-FNP product hierarchy"
        ),
        "design": {
            "start_selection": design["start_selection"],
            "replica_selection": design["replica_selection"],
            "pairs": design["pairs"],
        },
        "trajectory_protocol": {
            "reference_strength": STRENGTH,
            "reference_b_min": BMIN,
            "reference_b_max": BMAX,
            "fit_quality_barrier_strength": BARRIER_STRENGTH,
            "fit_quality_barrier_power": BARRIER_POWER,
            "float64": True,
            "continuation_block_capacity": BLOCK_CAPACITY,
            "continuation_block_count": MAX_CHUNKS,
            "total_requested_lbfgs_capacity": MAX_CHUNKS * BLOCK_CAPACITY,
            "full_horizon_exhausted_for_every_pair": True,
            "objective_domain_stationarity_gate": DRIFT_GATE,
            "full_grid_stationarity_gate": DRIFT_GATE,
            "required_post_200k_quiet_blocks": REQUIRED_QUIET_BLOCKS,
        },
        "completed_pair_count": len(endpoints),
        "terminal_endpoints": completed,
        "all_nested_trajectories_stationary_and_fit_preserving":
            all_trajectories_pass,
        "interaction_definition": (
            "actual log(F_NP) minus [declared central log(F_NP) + "
            "pointwise-median-centered start log residual + pointwise-median-"
            "centered primary-replica log residual]"
        ),
        "full_b_grid": {
            "node_count": int(len(design["b_grid"])),
            "minimum": float(design["b_grid"][0]),
            "maximum": float(design["b_grid"][-1]),
            "sha256": float_array_sha256(design["b_grid"]),
        },
        "max_absolute_interaction_log_delta_full_grid": float(
            np.max(np.abs(interactions))),
        "decision_impact": impact,
        "observed_interaction_decision_sign_stable": sign_stable,
        "interaction_validation_gate_pass": all_trajectories_pass,
        "legacy_product_median_interaction_sign_can_gate": False,
        "final_joint_sampling_gate_authoritative": True,
        "base_candidate_endpoint_gate_pass": base_audit_gate,
        "diagnostic_base_product_decision_after_observed_interaction": bool(
            base_audit_gate and all_trajectories_pass
            and impact["overall_width_replacement_decision_after_interaction"]
        ),
        "promotion_eligible_after_observed_interaction_envelope": False,
        "decision_stage_semantics": (
            "this nested validator is an intermediate component audit and "
            "never authorizes promotion or another lambda/prior; only the "
            "subsequent jointly expanded exact lambda600-versus-lambda1 "
            "comparison can decide replacement"
        ),
        "probability_semantics": (
            "the six interactions are a deterministic stratified validation "
            "design, not six draws from a calibrated probability law; their "
            "directional envelope is a robustness stress, not a one-sigma term"
        ),
        "transform_settings": {
            "tail_mode": "expb2", "b_transform_max": 24.0,
            "n_b_transform": 6001, "k_max": 4.0, "n_k": 401,
            "end_taper_start_fraction": 0.92,
        },
        "transform_metadata": {
            "actual": actual_meta,
            "additive_prediction": predicted_meta,
        },
        "frozen_inputs_before": frozen_before,
        "frozen_inputs_after": frozen_after,
        "postfit_tail_transform_audit": str(POSTFIT_TAIL_AUDIT),
        "postfit_tail_transform_audit_sha256": evidence["postfit_tail_hash"],
        "artifacts": artifacts,
        "artifact_sha256": artifact_sha256,
        "production_sources_modified": False,
    }
    atomic_write_json(TARGET / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def guarded_main() -> None:
    """Publish an explicit non-promotable terminal marker on technical error."""
    try:
        main()
    except Exception as error:
        TARGET.mkdir(parents=True, exist_ok=True)
        atomic_write_json(TARGET / "summary.json", {
            "status": "technical_failure",
            "error_type": type(error).__name__,
            "error": str(error),
            "interaction_validation_gate_pass": False,
            "promotion_eligible_after_observed_interaction_envelope": False,
            "failure_semantics": (
                "missing, stale, nonstationary, non-float64, or otherwise "
                "invalid nested evidence fails closed and cannot support the "
                "lambda600 replacement decision"
            ),
            "production_sources_modified": False,
        })
        raise


if __name__ == "__main__":
    guarded_main()
