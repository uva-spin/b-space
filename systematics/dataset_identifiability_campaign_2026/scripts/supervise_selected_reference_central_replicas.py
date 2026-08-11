#!/usr/bin/env python3
"""Fit a full-reference central model and all 50 data replicas.

The full empirical reference is used only after reciprocal cross-validation
and the selected strength's 24-start verification.  The central initializer
is the verified endpoint nearest the pointwise-median FNP in squared distance,
avoiding an arbitrary seed choice. The central and every replica receive forty
unchanged-objective float64 continuation blocks, each capped at 5k LBFGS
iterations. The 200k label denotes requested capacity, not necessarily executed
iterations. Ten wholly subsequent 5k-capacity confirmations must have both
adjacent and anchor-window FNP movement at most 2%. Experimental replicas must
also pass the declared N+5sqrt(2N) unpenalized chi-square sanity ceiling.
Tighter drift thresholds are recorded as sensitivity diagnostics. Scientific
gate failures are retained as measured outcomes: the central exhausts the
fixed horizon, all 50 replicas are run, and every available independent-
optimizer comparison is evaluated before promotion eligibility is decided.
"""

from __future__ import annotations

import csv
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from checkpoint_launch_ancestry import (
    build_continuation_command,
    build_launch_receipt,
    exclusive_checkpoint_launch,
    prepare_launch_receipt,
    validate_launch_receipt,
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
EVEN_SOURCE_REFERENCE = (
    BASE / "summaries/crossfit_reference_evenref/fnp_median.csv")
ODD_SOURCE_REFERENCE = (
    BASE / "summaries/crossfit_reference_oddref/fnp_median.csv")
SELECTED = BASE / "summaries/replica_robust_reference_full24/summary.json"
START_SOURCE_ENSEMBLE = (
    BASE / "summaries/selected_reference_method_full24/summary.json")
BRACKET = BASE / "summaries/full_reference_replica_strength_bracket/summary.json"
STAGED_STRESS = BASE / "summaries/lambda675_fit_quality_barrier_stress/summary.json"
MINIMUM_SEARCH = BASE / "summaries/minimum_fitbar_constraint_search/summary.json"
BARRIER_SEARCH = BASE / "summaries/minimum_barrier_constraint_search/summary.json"
PYTHON = Path("/home/dustin/miniforge3/envs/pdf-fit/bin/python")
TARGET = BASE / "summaries/selected_reference_central_replicas"
LAUNCH_RECEIPTS = BASE / "summaries/checkpoint_launch_receipts"
FNP_DRIFT_GATE = 0.02
FNP_DRIFT_SENSITIVITY = (0.0025, 0.005, 0.01, 0.02)
REQUIRED_CONSECUTIVE_BLOCKS = 10
MINIMUM_CUMULATIVE_ITERATIONS = 200_000
# Observed delayed reversals invalidate the earlier short-horizon/two-block
# protocol. Keep every central and replica on the strengthened forty-block
# pre-anchor protocol, with twenty further blocks available for a wholly fresh
# ten-block confirmation window after a late reset.
MAX_CHUNKS = 60
REPLICA_SEEDS = tuple(range(1001, 1051))
REQUIRED_ENDPOINT_FILES = (
    "fit_status.json",
    "model_state.pt",
    "dataset_norms.csv",
    "fnp_grid.csv",
    "accepted_predictions.csv",
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
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
    accepted = pd.read_csv(SOURCE / "predictions.csv")
    target = accepted.target_used.to_numpy(float).copy()
    if replica_seed is not None:
        rng = np.random.default_rng(replica_seed)
        for _, subset in accepted.groupby("dataset", sort=False):
            indices = subset.index.to_numpy()
            norm_rel = float(subset["norm_rel_used"].iloc[0])
            target[indices] *= 1.0 + rng.normal() * norm_rel
        target += (rng.normal(size=len(accepted))
                   * accepted.sigma_uncorr.to_numpy(float))
    tokens = tuple(pd.DataFrame({"fit_target": target}).to_csv(
        index=False).splitlines()[1:])
    identities = tuple(zip(
        accepted.dataset.astype(str), accepted.row_id.astype(str), strict=True))
    digest = stable_text_digest(tuple(
        f"{dataset}\0{row_id}\0{token}"
        for (dataset, row_id), token in zip(identities, tokens, strict=True)))
    return identities, tokens, digest


def validate_fit_target(run: Path, replica_seed: int | None) -> str:
    with (run / "accepted_predictions.csv").open(
            newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if (reader.fieldnames is None
                or not {"dataset", "row_id", "fit_target"}.issubset(
                    reader.fieldnames)):
            raise RuntimeError(
                f"accepted-prediction schema is incomplete: {run.name}")
        rows = list(reader)
    observed_identities = tuple(
        (str(row["dataset"]), str(row["row_id"])) for row in rows)
    observed_tokens = tuple(str(row["fit_target"]) for row in rows)
    expected_identities, expected_tokens, expected_digest = (
        expected_fit_target_evidence(replica_seed))
    if (observed_identities != expected_identities
            or observed_tokens != expected_tokens):
        raise RuntimeError(f"fit-target realization mismatch: {run.name}")
    observed_digest = stable_text_digest(tuple(
        f"{dataset}\0{row_id}\0{token}"
        for (dataset, row_id), token in zip(
            observed_identities, observed_tokens, strict=True)))
    if observed_digest != expected_digest:
        raise RuntimeError(f"fit-target digest mismatch: {run.name}")
    return observed_digest


def fnp_curve(run: Path, bmax: float) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(run / "fnp_grid.csv")
    if not {"x", "bT", "F_NP"}.issubset(frame.columns):
        raise RuntimeError(f"FNP grid schema is incomplete: {run.name}")
    frame = frame[np.isclose(frame.x, .1) & frame.bT.between(
        .1, bmax)].sort_values("bT")
    b = frame.bT.to_numpy(float)
    values = frame.F_NP.to_numpy(float)
    if (len(b) < 3 or len(np.unique(b)) != len(b)
            or not np.all(np.diff(b) > 0.0)
            or not np.all(np.isfinite(b))
            or not np.all(np.isfinite(values))
            or not np.all(values > 0.0)):
        raise RuntimeError(f"invalid x=0.1 FNP grid: {run.name}")
    return b, values


def vector(run: Path, bmax: float) -> np.ndarray:
    return fnp_curve(run, bmax)[1]


def float_array_digest(values: np.ndarray) -> str:
    canonical = np.ascontiguousarray(values, dtype="<f8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def central_initializer_selection(endpoints: list[Path], bmax: float) -> dict:
    if len(endpoints) != 24 or len({run.name for run in endpoints}) != 24:
        raise RuntimeError("central selection requires 24 unique endpoints")
    curves = [fnp_curve(run, bmax) for run in endpoints]
    reference_grid = curves[0][0]
    if not all(grid.shape == reference_grid.shape and np.array_equal(
            grid, reference_grid) for grid, _ in curves[1:]):
        raise RuntimeError(
            "24-start FNP curves do not share one exact ordered b grid")
    values = np.asarray([item for _, item in curves], dtype=float)
    median = np.median(values, axis=0)
    distances = np.mean((values - median[None, :]) ** 2, axis=1)
    if not np.all(np.isfinite(distances)):
        raise RuntimeError("central initializer distances are non-finite")
    selected_index = int(np.argmin(distances))
    return {
        "rule": (
            "minimum unweighted mean squared direct-FNP distance to the "
            "pointwise 24-start median"),
        "x": 0.1,
        "b_min": 0.1,
        "b_max": bmax,
        "space": "direct_F_NP",
        "member_count": len(endpoints),
        "ordered_start_tags": [run.name for run in endpoints],
        "ordered_squared_distances": [float(value) for value in distances],
        "selected_index": selected_index,
        "selected_tag": endpoints[selected_index].name,
        "b_grid_sha256": float_array_digest(reference_grid),
        "pointwise_median_sha256": float_array_digest(median),
    }


def require_endpoint_objective(tag: str, *, strength: float, bmax: float,
                               barrier_strength: float,
                               barrier_power: int,
                               replica_seed: int | None = None,
                               fit_seed: int | None = None,
                               expected_barrier_ceiling: float | None = None,
                               expected_initial_state: Path | None = None,
                               expected_initial_norms: Path | None = None,
                               expected_launch_receipt: dict | None = None,
                               expected_reference: Path = REFERENCE) -> dict:
    """Fail closed unless an external stress endpoint uses this prescription."""
    run = BASE / "outputs" / tag
    missing = [name for name in REQUIRED_ENDPOINT_FILES
               if not (run / name).is_file()
               or (run / name).stat().st_size == 0]
    if missing:
        raise RuntimeError(
            f"cached endpoint {tag} lacks required files: {missing}")
    path = run / "fit_status.json"
    status = json.loads(path.read_text())
    reference = status["regularization"]["fnp_reference_distance"]
    barrier = status["regularization"]["fit_quality_barrier"]
    nonzero_regularizers = {
        name for name, spec in status["regularization"].items()
        if isinstance(spec, dict) and "lambda" in spec
        and not exact_number(spec["lambda"], 0.0)
    }
    observed_replica = status.get("replica_seed")
    replica_matches = (
        (replica_seed is None and observed_replica is None)
        or (replica_seed is not None and observed_replica is not None
            and int(observed_replica) == int(replica_seed)))
    expected_replica_ceiling = (int(status["row_count"])
                                + 5.0 * math.sqrt(
                                    2.0 * int(status["row_count"])))
    barrier_ceiling_matches = (
        barrier_strength <= 0
        or (expected_barrier_ceiling is not None
            and exact_number(barrier["ceiling_total_chi2"],
                             expected_barrier_ceiling))
        or (replica_seed is not None
            and exact_number(barrier["ceiling_total_chi2"],
                             expected_replica_ceiling)))
    complexity = status.get("model_complexity", {})
    profile = status.get("point_profile", {})
    expected_nonzero_regularizers = {"fnp_reference_distance"}
    if barrier_strength > 0:
        expected_nonzero_regularizers.add("fit_quality_barrier")
    if not (exact_bool(status.get("convergence_gate_pass"),
                       "convergence_gate_pass")
            and not exact_bool(status.get("production_state_modified"),
                               "production_state_modified")
            and Path(status["source_production"]).resolve() == SOURCE.resolve()
            and Path(status["w_grid"]).resolve() == W_GRID.resolve()
            and exact_number(reference["lambda"], strength)
            and Path(reference["target_csv"]).resolve() ==
                expected_reference.resolve()
            and exact_number(reference["b_min"], 0.1)
            and exact_number(reference["b_max"], bmax)
            and exact_number(barrier["lambda"], barrier_strength)
            and int(barrier["power"]) == barrier_power
            and barrier_ceiling_matches
            and nonzero_regularizers == expected_nonzero_regularizers
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
            and int(complexity.get("distill_prediction_steps", -1)) == 0
            and replica_matches
            and (fit_seed is None or int(status.get("seed", -1)) == fit_seed)
            and (expected_initial_state is None
                 or Path(status.get("initial_state", "")).resolve()
                    == expected_initial_state.resolve())):
        raise RuntimeError(
            f"stress endpoint objective/provenance mismatch: {tag}")
    if expected_initial_norms is not None:
        if expected_launch_receipt is None:
            raise RuntimeError(
                f"paired initial norms lack launch evidence: {tag}")
        receipt = validate_launch_receipt(
            LAUNCH_RECEIPTS, tag, expected_launch_receipt)
        if (Path(receipt["parent_state_path"]).resolve()
                != expected_initial_state.resolve()
                or Path(receipt["parent_norms_path"]).resolve()
                != expected_initial_norms.resolve()
                or receipt["parent_state_sha256"]
                != sha256(expected_initial_state)
                or receipt["parent_norms_sha256"]
                != sha256(expected_initial_norms)):
            raise RuntimeError(
                f"restart state/norm launch evidence mismatch: {tag}")
        status["_validated_launch_receipt"] = receipt
    status["_validated_fit_target_sha256"] = validate_fit_target(
        run, replica_seed)
    return status


def resolve_direct_long_evidence(*, strength: float, bmax: float,
                                 barrier_strength: float,
                                 barrier_power: int) -> dict | None:
    """Resolve evidence by metadata, never by an ambiguous float token."""
    matches = []
    for path in sorted((BASE / "summaries").glob(
            "fitbar_candidate_lam*_long_horizon/summary.json")):
        evidence = json.loads(path.read_text())
        if (evidence.get("status") == "candidate_survives_discriminator"
                and exact_number(evidence.get(
                    "tested_reference_strength"), strength)
                and exact_number(evidence.get(
                    "fit_quality_barrier_strength"), barrier_strength)
                and int(evidence.get("fit_quality_barrier_power", -1)) ==
                    barrier_power):
            tag = str(evidence["endpoint_tag"])
            require_endpoint_objective(
                tag, strength=strength, bmax=bmax,
                barrier_strength=barrier_strength,
                barrier_power=barrier_power,
                replica_seed=int(evidence["replica_seed"]))
            matches.append((path, evidence))
    if len(matches) > 1:
        raise RuntimeError(
            "multiple same-prescription long-horizon stress summaries: "
            + ", ".join(str(path) for path, _ in matches))
    return matches[0][1] if matches else None


def run_adaptive(*, label: str, seed: int, strength: float,
                 bmax: float,
                 barrier_strength: float, barrier_power: int,
                 initial: Path, replica_seed: int | None,
                 records: list[dict],
                 minimum_required_iterations: int = 210_000,
                 exhaust_horizon: bool = False) -> tuple[Path, bool]:
    previous = initial
    initial_status = json.loads((initial / "fit_status.json").read_text())
    initial_row_count = int(initial_status["row_count"])
    fit_quality_ceiling = (
        float(initial_status["final"]["unpenalized_total_chi2"])
        + float(np.sqrt(2.0 * initial_row_count))
        if replica_seed is None else
        initial_row_count + 5.0 * math.sqrt(2.0 * initial_row_count)
    )
    passed = False
    consecutive_quiet_blocks = 0
    sensitivity_active = False
    fresh_after_sensitivity = 0
    stationarity_anchor = None
    stationarity_anchor_iterations = None
    for chunk in range(1, MAX_CHUNKS + 1):
        cumulative = chunk * 5000
        initial_state_path = previous / "model_state.pt"
        initial_norms_path = previous / "dataset_norms.csv"
        initial_state_sha256 = sha256(initial_state_path)
        initial_norms_sha256 = sha256(initial_norms_path)
        barrier_tag = (f"_fitbar_p{barrier_power}_mu{token(barrier_strength)}"
                       if barrier_strength > 0 else "")
        tag = (f"fullref_lam{token(strength)}{barrier_tag}_b{token(bmax)}_"
               f"{label}_polish64_{cumulative}")
        target = BASE / "outputs" / tag
        barrier_ceiling = (
            fit_quality_ceiling if barrier_strength > 0 else None)
        command = build_continuation_command(
            python=PYTHON, runner=RUNNER, seed=seed,
            source_production=SOURCE, w_grid=W_GRID,
            output_root=BASE / "outputs", child_tag=tag,
            parent_state=initial_state_path, parent_norms=initial_norms_path,
            reference_strength=strength, reference_csv=REFERENCE,
            reference_bmin=0.1, reference_bmax=bmax,
            barrier_strength=barrier_strength, barrier_power=barrier_power,
            barrier_ceiling=barrier_ceiling, replica_seed=replica_seed)
        expected_receipt = build_launch_receipt(
            receipt_root=LAUNCH_RECEIPTS, child_output=target,
            child_tag=tag, parent_state=initial_state_path,
            parent_norms=initial_norms_path, fit_seed=seed,
            replica_seed=replica_seed, command=command,
            reference_strength=strength, reference_bmin=0.1,
            reference_bmax=bmax, barrier_strength=barrier_strength,
            barrier_power=barrier_power, barrier_ceiling=barrier_ceiling)
        with exclusive_checkpoint_launch(LAUNCH_RECEIPTS, tag):
            if not (target / "fit_status.json").exists():
                # The immutable receipt is durably visible before the frozen
                # runner can create any child artifact. A crash here is safe:
                # the next attempt validates the same receipt and resumes.
                prepare_launch_receipt(
                    LAUNCH_RECEIPTS, target, tag, expected_receipt)
                with (BASE / "logs" / f"{tag}.log").open("w") as stream:
                    validate_fixed_challenger_protocol()
                    subprocess.run(
                        command, stdout=stream, stderr=subprocess.STDOUT,
                        check=True)
            launch_receipt = validate_launch_receipt(
                LAUNCH_RECEIPTS, tag, expected_receipt)
            status = require_endpoint_objective(
                tag, strength=strength, bmax=bmax,
                barrier_strength=barrier_strength,
                barrier_power=barrier_power, replica_seed=replica_seed,
                fit_seed=seed,
                expected_barrier_ceiling=barrier_ceiling,
                expected_initial_state=initial_state_path,
                expected_initial_norms=initial_norms_path,
                expected_launch_receipt=expected_receipt)
        old, new = vector(previous, bmax), vector(target, bmax)
        drift = float(np.max(np.abs(new - old) / np.maximum(old, .05)))
        if cumulative == MINIMUM_CUMULATIVE_ITERATIONS:
            stationarity_anchor = new.copy()
            stationarity_anchor_iterations = cumulative
        eligible = cumulative > MINIMUM_CUMULATIVE_ITERATIONS
        tested_anchor_iterations = stationarity_anchor_iterations
        window_drift = (float(np.max(
            np.abs(new - stationarity_anchor) / np.maximum(stationarity_anchor, .05)))
            if eligible and stationarity_anchor is not None else None)
        window_quiet = (window_drift is not None
                        and window_drift <= FNP_DRIFT_GATE)
        # Require every qualifying block to be quiet both locally and relative
        # to the active 50k-window anchor; small same-direction steps may not
        # accumulate into a moving endpoint and still count as a plateau.
        quiet = eligible and drift <= FNP_DRIFT_GATE and window_quiet
        # Independent starts and stress replicas both exhibited delayed
        # reversals after apparently quiet early endpoints. Count confirmation
        # blocks only after all forty mandatory continuation blocks.
        if not eligible:
            consecutive_quiet_blocks = 0
        elif not quiet:
            # Permit a late legitimate descent, but restart the entire 50k
            # stationarity clock at its endpoint.
            stationarity_anchor = new.copy()
            stationarity_anchor_iterations = cumulative
            consecutive_quiet_blocks = 0
            sensitivity_active = False
            fresh_after_sensitivity = 0
        else:
            consecutive_quiet_blocks += 1
        row_count = int(status["row_count"])
        if row_count != initial_row_count:
            raise RuntimeError(f"row count changed within adaptive fit: {label}")
        replica_chi2_ceiling = row_count + 5.0 * math.sqrt(2.0 * row_count)
        replica_chi2_pass = (replica_seed is None or
                             float(status["final"]["unpenalized_total_chi2"])
                             <= replica_chi2_ceiling)
        fit_quality_pass = (
            float(status["final"]["unpenalized_total_chi2"])
            <= fit_quality_ceiling
        )
        triggered_now = (quiet and drift >= FNP_DRIFT_SENSITIVITY[-2]
                         and not sensitivity_active)
        if triggered_now:
            sensitivity_active = True
            fresh_after_sensitivity = 0
        elif sensitivity_active:
            fresh_after_sensitivity = fresh_after_sensitivity + 1 if quiet else 0
        confirmation_pass = (fresh_after_sensitivity >= REQUIRED_CONSECUTIVE_BLOCKS
                             if sensitivity_active else
                             consecutive_quiet_blocks >= REQUIRED_CONSECUTIVE_BLOCKS)
        stationarity_pass = (
            cumulative >= max(
                MINIMUM_CUMULATIVE_ITERATIONS
                + 5000 * REQUIRED_CONSECUTIVE_BLOCKS,
                minimum_required_iterations)
            and confirmation_pass and window_quiet)
        passed = (stationarity_pass and replica_chi2_pass
                  and fit_quality_pass)
        records.append({
            "kind": "central" if replica_seed is None else "experimental_replica",
            "replica_seed": replica_seed,
            "fit_seed": seed,
            "initial_state_path": str(initial_state_path.resolve()),
            "initial_state_sha256": initial_state_sha256,
            "initial_norms_path": str(initial_norms_path.resolve()),
            "initial_norms_sha256": initial_norms_sha256,
            "launch_ancestry_kind": launch_receipt["kind"],
            "launch_receipt_path": launch_receipt["path"],
            "launch_receipt_sha256": launch_receipt["sha256"],
            "launch_argv_sha256": launch_receipt["argv_sha256"],
            "fit_target_sha256": status[
                "_validated_fit_target_sha256"],
            "cumulative_lbfgs_iterations": cumulative,
            "requested_lbfgs_max_iterations_this_block": 5000,
            "executed_lbfgs_closure_evaluations_this_block": int(
                status["lbfgs"]["closure_evaluations"]),
            "minimum_required_iterations": minimum_required_iterations,
            "fnp_drift_from_previous_chunk": drift,
            "eligible_post_mandatory_confirmation": eligible,
            "post_mandatory_window_fnp_drift": window_drift,
            "stationarity_window_anchor_iterations": tested_anchor_iterations,
            "next_stationarity_window_anchor_iterations":
                stationarity_anchor_iterations,
            "passes_post_mandatory_window_drift_2pct": window_quiet,
            "passes_drift_0p25pct": drift <= FNP_DRIFT_SENSITIVITY[0],
            "passes_drift_0p5pct": drift <= FNP_DRIFT_SENSITIVITY[1],
            "passes_drift_1pct": drift <= FNP_DRIFT_SENSITIVITY[2],
            "passes_drift_2pct": drift <= FNP_DRIFT_SENSITIVITY[3],
            "consecutive_quiet_blocks": consecutive_quiet_blocks,
            "sensitivity_confirmation_triggered": sensitivity_active,
            "fresh_quiet_blocks_after_sensitivity_trigger": fresh_after_sensitivity,
            "unpenalized_total_chi2": status["final"]["unpenalized_total_chi2"],
            "data_chi2": status["final"]["data_chi2"],
            "replica_chi2_sanity_ceiling": (
                replica_chi2_ceiling if replica_seed is not None else None),
            "replica_chi2_sanity_pass": replica_chi2_pass,
            "fit_quality_ceiling_total_chi2": fit_quality_ceiling,
            "fit_quality_gate_pass": fit_quality_pass,
            "stationarity_gate_pass": stationarity_pass,
            "endpoint_acceptance_gate_pass": passed,
            "fnp_plateau_gate_pass": passed,
            "full_horizon_required": exhaust_horizon,
            "tag": tag,
        })
        atomic_write_csv(TARGET / "runs.csv", pd.DataFrame(records))
        previous = target
        if passed and not exhaust_horizon:
            break
        # A nonpassing replica that can no longer accumulate ten quiet blocks
        # is scientifically ineligible, but truncating its endpoint would bias
        # the experimental distribution used by the requested diagnostic.
        # Preserve every failed trajectory through the complete 300k cap.
    return previous, passed


def main() -> None:
    _, fixed_protocol_hash = validate_fixed_challenger_protocol()
    selected = json.loads(SELECTED.read_text())
    selected_status = str(selected.get("status"))
    if selected_status not in {"complete", "verification_failed"}:
        raise RuntimeError(
            "selected-strength summary is not an exact terminal generation")
    start_scientific_gate_pass = exact_bool(selected.get(
        "all_starts_fnp_plateaued_and_fit_preserved"),
        "all_starts_fnp_plateaued_and_fit_preserved")
    failed_start_seeds = sorted(int(value) for value in selected.get(
        "failed_seeds", []))
    if ((selected_status == "complete") != start_scientific_gate_pass
            or (start_scientific_gate_pass and failed_start_seeds)
            or (not start_scientific_gate_pass and not failed_start_seeds)
            or any(seed not in range(303, 327)
                   for seed in failed_start_seeds)
            or len(failed_start_seeds) != len(set(failed_start_seeds))):
        raise RuntimeError("selected-strength terminal gate flags are inconsistent")
    if int(selected.get(
            "mandatory_same_objective_iterations_per_start", 0)) < 200_000:
        raise RuntimeError(
            "selected-strength terminal generation lacks the strengthened horizon")
    if exact_bool(selected.get("production_sources_modified"),
                  "selected.production_sources_modified"):
        raise RuntimeError("selected-strength summary reports production mutation")
    strength = float(selected["selected_strength"])
    bmax = float(selected["selected_bmax"])
    barrier_strength = float(selected.get("fit_quality_barrier_strength", 0.0))
    barrier_power = int(selected.get("fit_quality_barrier_power", 2))
    if (not exact_number(strength, 600.0)
            or not exact_number(bmax, 4.0)
            or not exact_number(barrier_strength, 100.0)
            or barrier_power != 2):
        raise RuntimeError("selected endpoints are not the fixed lambda600 prescription")
    endpoint_tags = list(selected["endpoint_tags"])
    if (int(selected.get("member_count", -1)) != 24
            or len(endpoint_tags) != 24 or len(set(endpoint_tags)) != 24):
        raise RuntimeError("selected-strength verification must provide 24 unique endpoints")
    endpoints = [BASE / "outputs" / tag for tag in endpoint_tags]
    missing = [run.name for run in endpoints
               if not (run / "fit_status.json").exists() or not (run / "fnp_grid.csv").exists()]
    if missing:
        raise RuntimeError(f"selected-strength endpoints are incomplete: {missing}")
    source_ensemble = json.loads(START_SOURCE_ENSEMBLE.read_text())
    source_tags = [str(value) for value in source_ensemble.get(
        "endpoint_tags", [])]
    if (source_ensemble.get("status") != "complete"
            or not exact_number(source_ensemble.get("selected_strength"), 300.0)
            or not exact_number(source_ensemble.get("selected_bmax"), 4.0)
            or int(source_ensemble.get("member_count", -1)) != 24
            or not exact_bool(source_ensemble.get(
                "all_starts_fnp_plateaued_and_fit_preserved"),
                "lambda300.all_starts_fnp_plateaued_and_fit_preserved")
            or exact_bool(source_ensemble.get("production_sources_modified"),
                          "lambda300.production_sources_modified")
            or len(source_tags) != 24 or len(set(source_tags)) != 24):
        raise RuntimeError("lambda300 start-source ensemble is incomplete")
    for seed, endpoint, source_tag in zip(
            range(303, 327), endpoints, source_tags, strict=True):
        expected_source_reference = (
            EVEN_SOURCE_REFERENCE if seed % 2 else ODD_SOURCE_REFERENCE)
        source_status = require_endpoint_objective(
            source_tag, strength=300.0, bmax=4.0,
            barrier_strength=0.0, barrier_power=2,
            replica_seed=None, fit_seed=seed,
            expected_reference=expected_source_reference)
        expected_ceiling = (
            float(source_status["final"]["unpenalized_total_chi2"])
            + math.sqrt(2.0 * int(source_status["row_count"])))
        require_endpoint_objective(
            endpoint.name, strength=strength, bmax=bmax,
            barrier_strength=barrier_strength,
            barrier_power=barrier_power, replica_seed=None,
            fit_seed=seed, expected_barrier_ceiling=expected_ceiling)
    prevalidated_direct_evidence = None
    if strength < 675.0 and barrier_strength > 0:
        prevalidated_direct_evidence = resolve_direct_long_evidence(
            strength=strength, bmax=bmax,
            barrier_strength=barrier_strength,
            barrier_power=barrier_power)
        if prevalidated_direct_evidence is None:
            raise RuntimeError(
                "no exact same-model/objective stress evidence exists before "
                "central continuation")
    initializer_selection = central_initializer_selection(endpoints, bmax)
    median_nearest = BASE / "outputs" / initializer_selection["selected_tag"]
    median_nearest_status = json.loads((median_nearest / "fit_status.json").read_text())

    TARGET.mkdir(parents=True, exist_ok=True)
    def write_progress(*, central_tag: str | None,
                       central_pass: bool | None,
                       replica_endpoints: list[str],
                       failed: list[int]) -> None:
        """Publish coverage without exposing an older generation as final."""
        validate_fixed_challenger_protocol()
        atomic_write_json(TARGET / "summary.json", {
            "status": "in_progress",
            "fixed_challenger_protocol": str(FIXED_PROTOCOL),
            "fixed_challenger_protocol_sha256": fixed_protocol_hash,
            "fixed_fnp_reference": str(FIXED_FNP_REFERENCE),
            "fixed_fnp_reference_sha256": EXPECTED_FNP_REFERENCE_SHA256,
            **fixed_implementation_binding(),
            "selected_strength": strength,
            "selected_bmax": bmax,
            "start_verification_status": selected_status,
            "start_coverage_complete": True,
            "all_starts_fnp_plateaued_and_fit_preserved":
                start_scientific_gate_pass,
            "failed_start_seeds": failed_start_seeds,
            "fnp_drift_domain": {"x": 0.1, "b_min": 0.1, "b_max": bmax},
            "reference": str(REFERENCE),
            "central_initializer_rule": (
                "FNP pointwise-median-nearest member of 24 verified full-reference endpoints initialized "
                "from the reciprocal-cross-fit selection ensemble"),
            "central_initializer_selection": initializer_selection,
            "central_initializer_tag": median_nearest.name,
            "central_endpoint_tag": central_tag,
            "central_fnp_plateau_pass": central_pass,
            "central_full_horizon_requested_capacity": MAX_CHUNKS * 5000,
            "minimum_cumulative_iterations": MINIMUM_CUMULATIVE_ITERATIONS,
            "mandatory_continuation_block_count": (
                MINIMUM_CUMULATIVE_ITERATIONS // 5000),
            "mandatory_requested_lbfgs_capacity": MINIMUM_CUMULATIVE_ITERATIONS,
            "iteration_accounting": (
                "cumulative labels are requested LBFGS max-iteration capacity; "
                "each runs.csv row records actual closure evaluations separately"),
            "restart_state_and_norm_content_ancestry_recorded": True,
            "launch_time_state_and_norm_content_receipts_required": True,
            "fit_target_realizations_content_validated": True,
            "expected_replica_count": len(REPLICA_SEEDS),
            "completed_replica_count": len(replica_endpoints),
            "failed_replica_seeds_so_far": failed,
            "replica_endpoint_tags_so_far": replica_endpoints,
            "coverage_complete": False,
            "promotion_eligible": False,
            "production_sources_modified": False,
        })

    write_progress(central_tag=None, central_pass=None,
                   replica_endpoints=[], failed=[])
    records: list[dict] = []
    central, central_pass = run_adaptive(
        label="central", seed=int(median_nearest_status["seed"]), strength=strength, bmax=bmax,
        barrier_strength=barrier_strength, barrier_power=barrier_power,
        initial=median_nearest, replica_seed=None, records=records,
        exhaust_horizon=True)
    central_terminal_record = next(
        row for row in reversed(records) if row["kind"] == "central")
    if int(central_terminal_record["cumulative_lbfgs_iterations"]) != (
            MAX_CHUNKS * 5000):
        raise RuntimeError("central did not exhaust its fixed continuation horizon")
    write_progress(central_tag=central.name, central_pass=central_pass,
                   replica_endpoints=[], failed=[])

    replica_endpoints: list[str] = []
    replica_acceptance_failed: list[int] = []
    bracket = json.loads(BRACKET.read_text())
    staged_stress = (json.loads(STAGED_STRESS.read_text())
                     if STAGED_STRESS.exists() else None)
    minimum_search = (json.loads(MINIMUM_SEARCH.read_text())
                      if MINIMUM_SEARCH.exists() else None)
    barrier_search = (json.loads(BARRIER_SEARCH.read_text())
                      if BARRIER_SEARCH.exists() else None)
    lower_evidence = None
    direct_long_evidence = (prevalidated_direct_evidence
                            if prevalidated_direct_evidence is not None else
                            resolve_direct_long_evidence(
                                strength=strength, bmax=bmax,
                                barrier_strength=barrier_strength,
                                barrier_power=barrier_power))
    if (strength < 675.0 and barrier_strength > 0
            and direct_long_evidence is not None):
        lower_evidence = direct_long_evidence
        stress_tags = [lower_evidence["endpoint_tag"]]
        bracket_ledger = None
    elif (minimum_search is not None and minimum_search.get("status") == "complete"
            and np.isclose(float(minimum_search["selected_weakest_surviving_strength"]),
                           strength)
            and strength < 675.0):
        if (barrier_search is not None and barrier_search.get("status") == "complete"
                and np.isclose(float(barrier_search[
                    "selected_weakest_surviving_barrier_strength"]),
                               barrier_strength)):
            selected_barrier = float(
                barrier_search["selected_weakest_surviving_barrier_strength"])
            barrier_record = next(
                row for row in barrier_search["trials"]
                if np.isclose(float(row["barrier_strength"]), selected_barrier)
            )
            lower_evidence = json.loads(Path(barrier_record["evidence"]).read_text())
        else:
            selected_record = next(
                row for row in minimum_search["tested_candidates"]
                if np.isclose(float(row["strength"]), strength)
            )
            lower_evidence = json.loads(Path(selected_record["evidence"]).read_text())
        stress_tags = [lower_evidence["endpoint_tag"]]
        bracket_ledger = None
    elif (staged_stress is not None and staged_stress.get("status") == "complete"
          and np.isclose(strength, 675.0)):
        stress_tags = staged_stress["endpoint_tags"]
        bracket_ledger = pd.read_csv(
            BASE / "summaries/lambda675_fit_quality_barrier_stress/runs.csv")
    elif barrier_strength > 0 and strength < 675.0:
        # Never substitute an older no-barrier or different-lambda bracket
        # merely because a same-prescription evidence filename was misspelled.
        raise RuntimeError(
            "no exact same-prescription long-horizon stress evidence exists "
            f"for lambda={strength}, barrier={barrier_strength}, "
            f"power={barrier_power}, bmax={bmax}")
    else:
        stress_tags = bracket.get("selected_endpoint_tags", [])
        bracket_ledger = pd.read_csv(
            BASE / "summaries/full_reference_replica_strength_bracket/runs.csv")
    stress_required_iterations: dict[int, int] = {}
    historical_stress_ledger = None
    if lower_evidence is not None:
        stress_required_iterations[int(lower_evidence["replica_seed"])] = min(
            int(lower_evidence["mandatory_same_objective_iterations"]) + 10_000,
            MAX_CHUNKS * 5000,
        )
    elif (staged_stress is not None and staged_stress.get("status") == "complete"
          and np.isclose(strength, 675.0)):
        historical_stress_ledger = pd.read_csv(
            BASE / "summaries/full_reference_replica_strength_bracket/runs.csv")
    for stress_tag in ([] if lower_evidence is not None else stress_tags):
        matches = bracket_ledger[bracket_ledger["tag"].eq(stress_tag)]
        if len(matches) != 1:
            raise RuntimeError(f"bracket endpoint lacks one ledger row: {stress_tag}")
        row = matches.iloc[0]
        if "total_reference_iterations" in row.index and pd.notna(
                row["total_reference_iterations"]):
            iterations_column = "total_reference_iterations"
            multiplier = 1
        else:
            iterations_column = ("cumulative_lbfgs_iterations"
                                 if "cumulative_lbfgs_iterations" in row.index
                                 else "barrier_block")
            multiplier = (1 if iterations_column ==
                          "cumulative_lbfgs_iterations" else 5000)
        replica_seed = int(row["replica_seed"])
        required = int(row[iterations_column]) * multiplier
        # The staged barrier blocks continue trajectories from the earlier
        # lambda=675 stress audit.  Their local block counter therefore does
        # not represent the delayed-instability exposure that selected these
        # replicas.  Preserve the total historical-plus-staged exposure when
        # rebuilding each pseudo-data replica from the common central model.
        if (historical_stress_ledger is not None
                and iterations_column != "total_reference_iterations"):
            historical = historical_stress_ledger[
                np.isclose(historical_stress_ledger["strength"], strength)
                & historical_stress_ledger["replica_seed"].eq(replica_seed)
            ]
            if len(historical):
                # ``required`` is the staged barrier continuation counted
                # from a historical lambda=675 endpoint.  A fresh trajectory
                # from the common central must reproduce the total exposure,
                # not merely the larger of the two local counters.
                required += int(
                    historical["cumulative_lbfgs_iterations"].max())
        stress_required_iterations[replica_seed] = min(
            required, MAX_CHUNKS * 5000)
    # Validate all external evidence before spending any time on the central
    # replica ensemble.  A later cross-check must never discover that the
    # supposedly independent trajectory used a different objective.
    for stress_tag in stress_tags:
        stress_status = json.loads(
            (BASE / "outputs" / stress_tag / "fit_status.json").read_text())
        require_endpoint_objective(
            stress_tag, strength=strength, bmax=bmax,
            barrier_strength=barrier_strength, barrier_power=barrier_power,
            replica_seed=int(stress_status["replica_seed"]))
    for index, replica_seed in enumerate(REPLICA_SEEDS):
        endpoint, passed = run_adaptive(
            label=f"replica_r{replica_seed}", seed=2001 + index,
            strength=strength, bmax=bmax, initial=central, replica_seed=replica_seed,
            barrier_strength=barrier_strength, barrier_power=barrier_power,
            records=records,
            minimum_required_iterations=stress_required_iterations.get(
                replica_seed, 50_000))
        replica_endpoints.append(endpoint.name)
        if not passed:
            replica_acceptance_failed.append(replica_seed)
        write_progress(
            central_tag=central.name, central_pass=central_pass,
            replica_endpoints=replica_endpoints,
            failed=replica_acceptance_failed)

    replica_coverage_complete = (
        len(replica_endpoints) == len(REPLICA_SEEDS)
        and len(set(replica_endpoints)) == len(REPLICA_SEEDS))
    if not replica_coverage_complete:
        raise RuntimeError("full 50-replica endpoint coverage was not produced")

    # A stationary endpoint is not sufficient if another optimizer trajectory
    # for the *same pseudo-data replica* settles on a materially different FNP.
    # The independent stress fits supply that second trajectory.  Require their
    # disagreement to fit inside the full 24-start residual-nonuniqueness scale
    # already accepted for this prescription; otherwise the experimental
    # ensemble would hide an additional optimizer-start ambiguity that is not
    # represented by its nominal 24x50 hierarchy.
    cross_optimizer_threshold = float(
        selected["max_endpoint_fnp_full_range_selected_domain_floor_normalized"]
    )
    primary_by_replica = {
        replica_seed: BASE / "outputs" / tag
        for replica_seed, tag in zip(REPLICA_SEEDS, replica_endpoints)
    }
    cross_optimizer_checks = []
    cross_optimizer_failed: list[int] = []
    if stress_tags:
        for stress_tag in stress_tags:
            stress_run = BASE / "outputs" / stress_tag
            stress_status = json.loads((stress_run / "fit_status.json").read_text())
            replica_seed = int(stress_status["replica_seed"])
            primary_run = primary_by_replica.get(replica_seed)
            if primary_run is None:
                raise RuntimeError(
                    f"stress replica {replica_seed} is absent from the full ensemble"
                )
            primary_vector = vector(primary_run, bmax)
            stress_vector = vector(stress_run, bmax)
            denominator = np.maximum(0.5 * (primary_vector + stress_vector), 0.05)
            disagreement = float(np.max(
                np.abs(primary_vector - stress_vector) / denominator
            ))
            agreement_pass = disagreement <= cross_optimizer_threshold
            cross_optimizer_checks.append({
                "replica_seed": replica_seed,
                "primary_endpoint_tag": primary_run.name,
                "independent_stress_endpoint_tag": stress_run.name,
                "max_fnp_symmetric_relative_difference": disagreement,
                "allowed_full24_start_range": cross_optimizer_threshold,
                "agreement_pass": agreement_pass,
            })
            if (not agreement_pass
                    and replica_seed not in cross_optimizer_failed):
                cross_optimizer_failed.append(replica_seed)

    replica_terminal_records = {}
    for replica_seed, endpoint_tag in zip(
            REPLICA_SEEDS, replica_endpoints, strict=True):
        terminal = next(
            row for row in reversed(records)
            if row["kind"] == "experimental_replica"
            and int(row["replica_seed"]) == replica_seed
            and row["tag"] == endpoint_tag)
        replica_terminal_records[replica_seed] = terminal
    derived_acceptance_failed = [
        seed for seed, row in replica_terminal_records.items()
        if not bool(row["endpoint_acceptance_gate_pass"])]
    if derived_acceptance_failed != replica_acceptance_failed:
        raise RuntimeError("replica terminal acceptance flags are inconsistent")
    replica_stationarity_failed = [
        seed for seed, row in replica_terminal_records.items()
        if not bool(row["stationarity_gate_pass"])]
    replica_fit_quality_failed = [
        seed for seed, row in replica_terminal_records.items()
        if not bool(row["fit_quality_gate_pass"])]
    replica_chi2_sanity_failed = [
        seed for seed, row in replica_terminal_records.items()
        if not bool(row["replica_chi2_sanity_pass"])]
    failed_replica_terminal_capacities = {
        int(seed): int(replica_terminal_records[seed][
            "cumulative_lbfgs_iterations"])
        for seed in replica_acceptance_failed
    }
    failed_replicas_exhausted_full_requested_horizon = all(
        value == MAX_CHUNKS * 5000
        for value in failed_replica_terminal_capacities.values())
    if not failed_replicas_exhausted_full_requested_horizon:
        raise RuntimeError(
            "one or more nonpassing replicas did not exhaust the 300k cap")
    central_stationarity_pass = bool(
        central_terminal_record["stationarity_gate_pass"])
    central_acceptance_pass = bool(
        central_terminal_record["endpoint_acceptance_gate_pass"])
    if central_acceptance_pass != central_pass:
        raise RuntimeError("central terminal acceptance flag is inconsistent")
    cross_optimizer_coverage_complete = (
        len(cross_optimizer_checks) == len(stress_tags))
    cross_optimizer_gate_pass = (
        cross_optimizer_coverage_complete and not cross_optimizer_failed)
    coverage_complete = (
        len(endpoint_tags) == 24
        and int(central_terminal_record["cumulative_lbfgs_iterations"])
            == MAX_CHUNKS * 5000
        and replica_coverage_complete
        and cross_optimizer_coverage_complete)
    stationarity_gate_pass = (
        start_scientific_gate_pass and central_stationarity_pass
        and not replica_stationarity_failed)
    downstream_fit_quality_gate_pass = (
        bool(central_terminal_record["fit_quality_gate_pass"])
        and not replica_fit_quality_failed
        and not replica_chi2_sanity_failed)
    promotion_eligible = (
        coverage_complete and start_scientific_gate_pass
        and central_acceptance_pass and not replica_acceptance_failed
        and cross_optimizer_gate_pass)
    failed = sorted(set(replica_acceptance_failed)
                    | set(cross_optimizer_failed))
    validate_fixed_challenger_protocol()
    summary = {
        "status": ("complete" if promotion_eligible
                   else "complete_with_scientific_failures"),
        "fixed_challenger_protocol": str(FIXED_PROTOCOL),
        "fixed_challenger_protocol_sha256": fixed_protocol_hash,
        "fixed_fnp_reference": str(FIXED_FNP_REFERENCE),
        "fixed_fnp_reference_sha256": EXPECTED_FNP_REFERENCE_SHA256,
        **fixed_implementation_binding(),
        "selected_strength": strength,
        "fit_quality_barrier_strength": barrier_strength,
        "fit_quality_barrier_power": barrier_power,
        "staged_prescription": barrier_strength > 0,
        "selected_bmax": bmax,
        "start_verification_status": selected_status,
        "start_coverage_complete": True,
        "all_starts_fnp_plateaued_and_fit_preserved":
            start_scientific_gate_pass,
        "failed_start_seeds": failed_start_seeds,
        "lambda300_source_order_seed_objective_validated": True,
        "lambda300_source_endpoint_tags": source_tags,
        "fnp_drift_domain": {"x": 0.1, "b_min": 0.1, "b_max": bmax},
        "reference": str(REFERENCE),
        "central_initializer_rule": "FNP pointwise-median-nearest member of 24 verified full-reference endpoints initialized from the reciprocal-cross-fit selection ensemble",
        "central_initializer_selection": initializer_selection,
        "central_initializer_tag": median_nearest.name,
        "central_endpoint_tag": central.name,
        "central_fnp_plateau_pass": central_pass,
        "central_stationarity_gate_pass": central_stationarity_pass,
        "central_acceptance_gate_pass": central_acceptance_pass,
        "central_full_horizon_complete": True,
        "central_full_horizon_requested_capacity": MAX_CHUNKS * 5000,
        "central_terminal_requested_capacity": int(
            central_terminal_record["cumulative_lbfgs_iterations"]),
        "central_fit_quality_ceiling_total_chi2": central_terminal_record[
            "fit_quality_ceiling_total_chi2"],
        "central_fit_quality_gate_pass": central_terminal_record[
            "fit_quality_gate_pass"],
        "central_unpenalized_total_chi2": central_terminal_record[
            "unpenalized_total_chi2"],
        "minimum_cumulative_iterations": MINIMUM_CUMULATIVE_ITERATIONS,
        "mandatory_continuation_block_count": (
            MINIMUM_CUMULATIVE_ITERATIONS // 5000),
        "mandatory_requested_lbfgs_capacity": MINIMUM_CUMULATIVE_ITERATIONS,
        "iteration_accounting": "cumulative labels are requested LBFGS max-iteration capacity; each runs.csv row records actual closure evaluations separately",
        "restart_state_and_norm_content_ancestry_recorded": True,
        "launch_time_state_and_norm_content_receipts_required": True,
        "launch_receipt_root": str(LAUNCH_RECEIPTS.resolve()),
        "fit_target_realizations_content_validated": True,
        "replica_count": len(REPLICA_SEEDS),
        "completed_replica_count": len(replica_endpoints),
        "replica_coverage_complete": replica_coverage_complete,
        "early_stopped_after_definitive_failure": False,
        # Backward-compatible aggregate consumed by the final-band builder:
        # a replica ensemble is not accepted if either its primary endpoint or
        # an available independent-optimizer comparison fails.
        "all_replicas_fnp_plateaued": not failed,
        "all_replica_endpoint_acceptance_gates_pass":
            not replica_acceptance_failed,
        "all_replicas_stationarity_gate_pass":
            not replica_stationarity_failed,
        "all_replica_fit_quality_gates_pass":
            not replica_fit_quality_failed,
        "all_replica_chi2_sanity_gates_pass":
            not replica_chi2_sanity_failed,
        "replica_acceptance_failed_seeds": replica_acceptance_failed,
        "failed_replica_terminal_requested_capacity_by_seed":
            failed_replica_terminal_capacities,
        "full_requested_capacity_per_nonpassing_replica": MAX_CHUNKS * 5000,
        "failed_replicas_exhausted_full_requested_horizon":
            failed_replicas_exhausted_full_requested_horizon,
        "replica_stationarity_failed_seeds": replica_stationarity_failed,
        "replica_fit_quality_failed_seeds": replica_fit_quality_failed,
        "replica_chi2_sanity_failed_seeds": replica_chi2_sanity_failed,
        "failed_replica_seeds": failed,
        "replica_endpoint_tags": replica_endpoints,
        "fnp_drift_gate": FNP_DRIFT_GATE,
        "required_consecutive_quiet_blocks": REQUIRED_CONSECUTIVE_BLOCKS,
        "minimum_cumulative_iterations": MINIMUM_CUMULATIVE_ITERATIONS,
        "stress_replica_required_iterations": stress_required_iterations,
        "drift_sensitivity_thresholds": list(FNP_DRIFT_SENSITIVITY),
        "cross_optimizer_agreement_rule": (
            "independent stress and full-ensemble endpoints for each bracket "
            "replica must differ by no more than the selected prescription's "
            "full 24-start FNP range on x=0.1, 0.1<=bT<=bmax"
        ),
        "cross_optimizer_agreement_threshold": cross_optimizer_threshold,
        "cross_optimizer_checks": cross_optimizer_checks,
        "available_cross_optimizer_comparison_count": len(stress_tags),
        "completed_cross_optimizer_comparison_count": len(
            cross_optimizer_checks),
        "cross_optimizer_coverage_complete":
            cross_optimizer_coverage_complete,
        "cross_optimizer_failed_replica_seeds": cross_optimizer_failed,
        "all_stress_replicas_cross_optimizer_consistent":
            cross_optimizer_gate_pass,
        "coverage_complete": coverage_complete,
        "stationarity_gate_pass": stationarity_gate_pass,
        "downstream_fit_quality_gate_pass":
            downstream_fit_quality_gate_pass,
        "promotion_eligible": promotion_eligible,
        "scientific_failure_semantics": (
            [] if promotion_eligible else [
                label for label, failed_gate in (
                    ("24-start stationarity/fit gate",
                     not start_scientific_gate_pass),
                    ("central stationarity/fit gate",
                     not central_acceptance_pass),
                    ("replica stationarity/fit gate",
                     bool(replica_acceptance_failed)),
                    ("cross-optimizer agreement gate",
                     not cross_optimizer_gate_pass),
                ) if failed_gate
            ]),
        "production_sources_modified": False,
    }
    atomic_write_json(TARGET / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
