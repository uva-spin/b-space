#!/usr/bin/env python3
"""Audit terminal lambda=600 convergence outside the trained FNP gate.

The fixed challenger objective checks stationarity only through bT=4, while
the final finite-b transform consumes each FNP curve through bT=8 and is most
tail-sensitive at low kT.  This read-only post-fit audit therefore compares
the exact terminal 24-start/central/50-replica ensemble with the formal
stationarity-window checkpoint from every passing chain on the full bT grid;
failed/nonstationary chains instead use the fixed formal 200k checkpoint so a
late reset cannot conceal long-horizon motion.  It also transforms the paired
terminal/earlier ensembles with the frozen Fig. 6
convention and evaluates exp(b^2), exp(b), and taper continuations for the
terminal lambda=600 and post-processing-harmonized lambda=1 ensembles.

No new percentage tolerance is introduced.  The terminal/earlier expb2
incumbent comparison and its prior allowance are retained as a diagnostic,
but cannot accept or reject the challenger.  Their exact union envelope is
recorded as convergence evidence.  It is not charged the finite-ensemble
allowance measured for the distinct
  product-median-normalized statistic.  The final directional-envelope stage
  resamples the exact terminal/anchor member arrays with the trained central
  denominator and is solely authoritative for incumbent replacement.

The locked expb2 incumbent comparison remains primary.  The raw sign of the
mode-matched lambda=600 versus harmonized-lambda=1 comparison must additionally
remain favorable for expb2, expb, and taper; alternate modes cannot loosen the
locked threshold.  Mode-specific sampling margins are explicitly deferred
rather than borrowing the expb2 margin without evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.special import j0

from fixed_challenger_protocol import (
    fixed_implementation_binding,
    require_fixed_implementation_binding,
    validate_fixed_challenger_protocol,
)


BASE = Path(__file__).resolve().parents[1]
SYSTEMATICS = BASE.parent
ROOT = SYSTEMATICS.parent
OUTPUTS = BASE / "outputs"
SUMMARIES = BASE / "summaries"
SELECTED = SUMMARIES / "replica_robust_reference_full24/summary.json"
START_LEDGER = SUMMARIES / "replica_robust_reference_full24/runs.csv"
REPLICAS = SUMMARIES / "selected_reference_central_replicas/summary.json"
REPLICA_LEDGER = SUMMARIES / "selected_reference_central_replicas/runs.csv"
FINAL = SUMMARIES / "final_combined_tmd_ensemble"
STABILITY = SUMMARIES / "final_combined_ensemble_stability/summary.json"
HARMONIZED = SUMMARIES / "harmonized_lambda1_logfnp_24x50_comparator"
CHAMPION = (
    SUMMARIES / "champion_registry/empirical_reference_lambda1_b0p1_2p0_full24.json"
)
REFERENCE_B = (
    SYSTEMATICS / "collins_factorization_validity/plots/"
    "rowidfix_stageFT_all_qmax0p20_lam0p50_central_exactx/"
    "v22_scheme_tmd_bspace_long.csv"
)
TRANSFORMER = ROOT / "workflows/v23a/construction/construct_v23a_regularized_kspace_tmd_v2.py"
HARMONIZED_BUILDER = BASE / "scripts/build_harmonized_lambda1_logfnp_comparator.py"
TARGET = SUMMARIES / "lambda600_postfit_tail_transform_audit"
TAIL_MODES = ("expb2", "expb", "taper")
FLAVORS = ("u", "d")
LOCKED_INCUMBENT_WIDTHS = {
    "u": 0.11772613918747582,
    "d": 0.12490071924111977,
}
EPS = 1.0e-30
CHECKPOINT_ARRAY_SCHEMA = "lambda600_exact_expb2_checkpoint_arrays_v1"
CHECKPOINT_QUANTILE_RTOL = 5.0e-13
CHECKPOINT_QUANTILE_ATOL = 5.0e-15


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def explicit_bool(value, label: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise RuntimeError(f"{label} is not an explicit boolean")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_curve(run: Path) -> tuple[np.ndarray, np.ndarray]:
    path = run / "fnp_grid.csv"
    frame = pd.read_csv(path)
    frame = frame[np.isclose(frame["x"], 0.1)].sort_values("bT")
    b = frame["bT"].to_numpy(float)
    values = frame["F_NP"].to_numpy(float)
    if (len(b) != 321 or len(np.unique(b)) != len(b)
            or not np.all(np.isfinite(b))
            or not np.all(np.isfinite(values))
            or np.any(values <= 0.0)
            or b[-1] < 8.0 - 1.0e-12):
        raise RuntimeError(f"invalid full-grid positive FNP curve: {run}")
    return b, values


def _integer(value, label: str) -> int:
    number = float(value)
    if not np.isfinite(number) or not np.isclose(number, round(number)):
        raise RuntimeError(f"{label} is not a finite integer")
    return int(round(number))


def checkpoint_pair(
    ledger: pd.DataFrame,
    terminal_tag: str,
    *,
    family: str,
    scientifically_nonstationary: bool = False,
) -> dict:
    """Resolve a terminal tag and its tested stationarity-window anchor."""
    rows = ledger[ledger["tag"].astype(str).eq(str(terminal_tag))]
    if len(rows) != 1:
        raise RuntimeError(f"terminal ledger row is not unique: {terminal_tag}")
    terminal = rows.iloc[0]
    terminal_iterations = _integer(
        terminal["cumulative_lbfgs_iterations"], f"{terminal_tag} terminal iterations"
    )
    if scientifically_nonstationary:
        anchor_iterations = 200_000
        anchor_selection_rule = "fixed_formal_200k_for_nonstationary_chain"
    else:
        anchor_iterations = _integer(
            terminal["stationarity_window_anchor_iterations"],
            f"{terminal_tag} stationarity anchor",
        )
        anchor_selection_rule = "validated_terminal_stationarity_window_anchor"
    if anchor_iterations >= terminal_iterations:
        raise RuntimeError(f"stationarity anchor is not earlier than {terminal_tag}")

    candidates = ledger[
        np.isclose(
            pd.to_numeric(ledger["cumulative_lbfgs_iterations"], errors="coerce"),
            anchor_iterations,
        )
    ]
    identity: str
    if family == "start":
        seed = _integer(terminal["seed"], f"{terminal_tag} seed")
        candidates = candidates[
            pd.to_numeric(candidates["seed"], errors="coerce").eq(seed)
        ]
        identity = str(seed)
    elif family == "central":
        candidates = candidates[candidates["kind"].astype(str).eq("central")]
        fit_seed = _integer(terminal["fit_seed"], f"{terminal_tag} fit seed")
        candidates = candidates[
            pd.to_numeric(candidates["fit_seed"], errors="coerce").eq(fit_seed)
        ]
        identity = "central"
    elif family == "experimental_replica":
        replica_seed = _integer(
            terminal["replica_seed"], f"{terminal_tag} replica seed"
        )
        candidates = candidates[
            candidates["kind"].astype(str).eq("experimental_replica")
            & pd.to_numeric(candidates["replica_seed"], errors="coerce").eq(
                replica_seed
            )
        ]
        identity = str(replica_seed)
    else:
        raise ValueError(f"unknown chain family: {family}")
    if len(candidates) != 1:
        raise RuntimeError(
            f"stationarity anchor checkpoint is not unique for {terminal_tag}"
        )
    anchor_tag = str(candidates.iloc[0]["tag"])
    anchor_run = OUTPUTS / anchor_tag
    terminal_run = OUTPUTS / terminal_tag
    for run in (anchor_run, terminal_run):
        if not (run / "fnp_grid.csv").is_file() or not (
            run / "fit_status.json"
        ).is_file():
            raise RuntimeError(f"checkpoint output is incomplete: {run}")
    return {
        "family": family,
        "identity": identity,
        "scientifically_nonstationary": bool(scientifically_nonstationary),
        "anchor_selection_rule": anchor_selection_rule,
        "anchor_iterations": anchor_iterations,
        "terminal_iterations": terminal_iterations,
        "anchor_tag": anchor_tag,
        "terminal_tag": terminal_tag,
        "anchor_run": anchor_run,
        "terminal_run": terminal_run,
    }


def resolve_checkpoint_pairs(selected: dict, replicas: dict) -> list[dict]:
    start_ledger = pd.read_csv(START_LEDGER)
    replica_ledger = pd.read_csv(REPLICA_LEDGER)
    failed_starts = {int(value) for value in selected.get("failed_seeds", [])}
    failed_replicas = {
        int(value)
        for value in replicas.get(
            "replica_stationarity_failed_seeds",
            replicas.get("failed_replica_seeds", []),
        )
    }
    central_nonstationary = not explicit_bool(
        replicas.get(
            "central_stationarity_gate_pass",
            replicas.get("central_fnp_plateau_pass"),
        ),
        "central stationarity gate",
    )
    pairs = [
        checkpoint_pair(
            start_ledger, tag, family="start",
            scientifically_nonstationary=(
                int(start_ledger[
                    start_ledger["tag"].astype(str).eq(str(tag))
                ].iloc[0]["seed"]) in failed_starts
            ),
        )
        for tag in selected["endpoint_tags"]
    ]
    pairs.append(
        checkpoint_pair(
            replica_ledger, replicas["central_endpoint_tag"], family="central",
            scientifically_nonstationary=central_nonstationary,
        )
    )
    pairs.extend(
        checkpoint_pair(
            replica_ledger, tag, family="experimental_replica",
            scientifically_nonstationary=(
                int(replica_ledger[
                    replica_ledger["tag"].astype(str).eq(str(tag))
                ].iloc[0]["replica_seed"]) in failed_replicas
            ),
        )
        for tag in replicas["replica_endpoint_tags"]
    )
    start_ids = sorted(
        int(pair["identity"]) for pair in pairs if pair["family"] == "start"
    )
    replica_ids = sorted(
        int(pair["identity"])
        for pair in pairs
        if pair["family"] == "experimental_replica"
    )
    if start_ids != list(range(303, 327)):
        raise RuntimeError("checkpoint audit lacks exact start seeds 303--326")
    if replica_ids != list(range(1001, 1051)):
        raise RuntimeError("checkpoint audit lacks exact replica seeds 1001--1050")
    if sum(pair["family"] == "central" for pair in pairs) != 1 or len(pairs) != 75:
        raise RuntimeError("checkpoint audit lacks exact 24+1+50 chain coverage")
    return pairs


def ordered_pair_curves(
    pairs: list[dict], checkpoint: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    if checkpoint not in {"anchor", "terminal"}:
        raise ValueError(checkpoint)
    ordered = (
        sorted(
            [pair for pair in pairs if pair["family"] == "start"],
            key=lambda item: int(item["identity"]),
        )
        + [next(pair for pair in pairs if pair["family"] == "central")]
        + sorted(
            [
                pair
                for pair in pairs
                if pair["family"] == "experimental_replica"
            ],
            key=lambda item: int(item["identity"]),
        )
    )
    grid: np.ndarray | None = None
    values: list[np.ndarray] = []
    for pair in ordered:
        b, curve = load_curve(pair[f"{checkpoint}_run"])
        if grid is None:
            grid = b
        elif grid.shape != b.shape or not np.allclose(
            grid, b, rtol=0.0, atol=1.0e-12
        ):
            raise RuntimeError("checkpoint FNP grids are not identical")
        values.append(curve)
    if grid is None:
        raise RuntimeError("no checkpoint curves")
    array = np.asarray(values)
    return grid, array[:24], array[24], array[25:], ordered


def combined_log_members(
    central: np.ndarray, starts: np.ndarray, replicas: np.ndarray
) -> tuple[np.ndarray, dict]:
    if starts.shape[0] != 24 or replicas.shape[0] != 50:
        raise RuntimeError("hierarchy is not exactly 24x50")
    logc = np.log(central)
    start_logs = np.log(starts)
    replica_logs = np.log(replicas)
    start_residual = start_logs - np.median(start_logs, axis=0)
    replica_residual = replica_logs - np.median(replica_logs, axis=0)
    combined = (
        logc[None, None, :]
        + start_residual[:, None, :]
        + replica_residual[None, :, :]
    ).reshape(1200, central.size)
    return combined, {
        "start_log_residual_pointwise_median_abs_max": float(
            np.max(np.abs(np.median(start_residual, axis=0)))
        ),
        "experimental_log_residual_pointwise_median_abs_max": float(
            np.max(np.abs(np.median(replica_residual, axis=0)))
        ),
    }


def chain_movement_rows(
    b_grid: np.ndarray,
    anchor_curves: np.ndarray,
    terminal_curves: np.ndarray,
    ordered_pairs: list[dict],
) -> pd.DataFrame:
    regions = {
        "formal_b0p1_4": (b_grid >= 0.1) & (b_grid <= 4.0),
        "unconstrained_b4_8": (b_grid > 4.0) & (b_grid <= 8.0),
        "tailfit_b6_8": (b_grid >= 6.0) & (b_grid <= 8.0),
        "full_b0p1_8": (b_grid >= 0.1) & (b_grid <= 8.0),
    }
    rows = []
    for pair, anchor, terminal in zip(ordered_pairs, anchor_curves, terminal_curves):
        relative = np.abs(terminal - anchor) / np.maximum(np.abs(anchor), 0.05)
        row = {
            key: value
            for key, value in pair.items()
            if key not in {"anchor_run", "terminal_run"}
        }
        row["anchor_fnp_grid_sha256"] = sha256(
            pair["anchor_run"] / "fnp_grid.csv"
        )
        row["terminal_fnp_grid_sha256"] = sha256(
            pair["terminal_run"] / "fnp_grid.csv"
        )
        for name, mask in regions.items():
            row[f"max_relative_fnp_movement_{name}"] = float(
                np.max(relative[mask])
            )
        rows.append(row)
    return pd.DataFrame(rows)


def quantile_frame(
    b_grid: np.ndarray,
    logs: np.ndarray,
    declared_central_log: np.ndarray,
    checkpoint: str,
) -> pd.DataFrame:
    values = np.exp(logs)
    q16, median, q84 = np.quantile(values, (0.16, 0.50, 0.84), axis=0)
    return pd.DataFrame(
        {
            "checkpoint": checkpoint,
            "x": 0.1,
            "bT": b_grid,
            "q16": q16,
            "median": median,
            "q84": q84,
            "declared_trained_central": np.exp(declared_central_log),
        }
    )


class ExactTransformEngine:
    """Batch the frozen per-curve extension with the exact transform kernel."""

    def __init__(self, transformer):
        self.transformer = transformer
        self.b_grid = np.linspace(0.0, 24.0, 6001)
        self.k_grid = np.linspace(0.0, 4.0, 401)
        window = transformer.taper_window(self.b_grid, 0.92)
        trap = transformer.trapezoid_weights_uniform(self.b_grid)
        quadrature = self.b_grid * window * trap / (2.0 * np.pi)
        self.kernel = j0(np.outer(self.k_grid, self.b_grid)) * quadrature[None, :]

    def transform(
        self,
        b_in: np.ndarray,
        curves: np.ndarray,
        mode: str,
        *,
        batch_size: int = 96,
    ) -> np.ndarray:
        if mode not in TAIL_MODES:
            raise ValueError(mode)
        curves = np.atleast_2d(np.asarray(curves, dtype=float))
        result = np.empty((curves.shape[0], len(self.k_grid)), dtype=float)
        for start in range(0, curves.shape[0], batch_size):
            stop = min(start + batch_size, curves.shape[0])
            extended = np.asarray(
                [
                    self.transformer.extend_curve(
                        b_in,
                        curve,
                        self.b_grid,
                        tail_mode=mode,
                        tail_fit_bmin=None,
                        eps=1.0e-300,
                    )
                    for curve in curves[start:stop]
                ]
            )
            result[start:stop] = extended @ self.kernel.T
        return result


def reference_table() -> pd.DataFrame:
    frame = pd.read_csv(REFERENCE_B)
    return frame[
        np.isclose(frame["x"], 0.1)
        & (
            (
                np.isclose(frame["Q"], 7.5)
                & frame["flavor"].astype(str).isin(
                    ("u", "d", "s", "ubar", "dbar", "sbar")
                )
            )
            | (
                np.isclose(frame["Q"], 10.0)
                & frame["flavor"].astype(str).isin(FLAVORS)
            )
        )
    ].copy()


def bspace_terminal_outputs(
    reference: pd.DataFrame,
    b_grid: np.ndarray,
    terminal_logs: np.ndarray,
    terminal_central_log: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    band_rows = []
    central_rows = []
    for (q, flavor), group in reference.groupby(["Q", "flavor"], sort=False):
        group = group.sort_values("bT")
        b = group["bT"].to_numpy(float)
        perturbative = group["ftilde_no_np"].to_numpy(float)
        members = np.exp(
            np.asarray([np.interp(b, b_grid, values) for values in terminal_logs])
        ) * perturbative[None, :]
        q16, median, q84 = np.quantile(members, (0.16, 0.50, 0.84), axis=0)
        declared = np.exp(np.interp(b, b_grid, terminal_central_log)) * perturbative
        band_rows.append(
            pd.DataFrame(
                {
                    "Q": q,
                    "flavor": flavor,
                    "bT": b,
                    "q16": q16,
                    "median": median,
                    "q84": q84,
                    "declared_trained_central": declared,
                }
            )
        )
        central_rows.append(
            pd.DataFrame(
                {
                    "Q": q,
                    "flavor": flavor,
                    "bT": b,
                    "declared_trained_central": declared,
                }
            )
        )
    return pd.concat(band_rows, ignore_index=True), pd.concat(
        central_rows, ignore_index=True
    )


def transform_bands(
    engine: ExactTransformEngine,
    reference: pd.DataFrame,
    b_grid: np.ndarray,
    logs: np.ndarray,
    central_log: np.ndarray,
    *,
    model: str,
    checkpoint: str,
    mode: str,
) -> pd.DataFrame:
    rows = []
    selected = reference[
        np.isclose(reference["Q"], 10.0)
        & reference["flavor"].astype(str).isin(FLAVORS)
    ]
    for flavor in FLAVORS:
        group = selected[selected["flavor"].astype(str).eq(flavor)].sort_values(
            "bT"
        )
        b = group["bT"].to_numpy(float)
        perturbative = group["ftilde_no_np"].to_numpy(float)
        members = np.exp(
            np.asarray([np.interp(b, b_grid, values) for values in logs])
        ) * perturbative[None, :]
        print(
            f"transform {model} {checkpoint} {mode} {flavor}: {len(members)} members",
            flush=True,
        )
        transformed = engine.transform(b, members, mode)
        q16, median, q84 = np.quantile(
            transformed, (0.16, 0.50, 0.84), axis=0
        )
        central_b = np.exp(np.interp(b, b_grid, central_log)) * perturbative
        declared = engine.transform(b, central_b[None, :], mode)[0]
        rows.append(
            pd.DataFrame(
                {
                    "model": model,
                    "checkpoint": checkpoint,
                    "tail_mode": mode,
                    "flavor": flavor,
                    "x": 0.1,
                    "Q": 10.0,
                    "kT": engine.k_grid,
                    "q16": q16,
                    "median": median,
                    "q84": q84,
                    "declared_central": declared,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def transform_checkpoint_member_arrays(
    engine: ExactTransformEngine,
    reference: pd.DataFrame,
    b_grid: np.ndarray,
    logs: np.ndarray,
    central_log: np.ndarray,
    *,
    checkpoint: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact expb2 members in flavor/start/replica/k ordering."""
    logs = np.asarray(logs, dtype=float)
    central_log = np.asarray(central_log, dtype=float)
    if logs.shape != (1200, len(b_grid)) or central_log.shape != (len(b_grid),):
        raise RuntimeError(f"invalid {checkpoint} member hierarchy for exact transform")
    selected = reference[
        np.isclose(reference["Q"], 10.0)
        & reference["flavor"].astype(str).isin(FLAVORS)
    ]
    values = np.empty((len(FLAVORS), 24, 50, len(engine.k_grid)), dtype=float)
    central = np.empty((len(FLAVORS), len(engine.k_grid)), dtype=float)
    for flavor_index, flavor in enumerate(FLAVORS):
        group = selected[selected["flavor"].astype(str).eq(flavor)].sort_values(
            "bT"
        )
        b = group["bT"].to_numpy(float)
        perturbative = group["ftilde_no_np"].to_numpy(float)
        members_b = np.exp(
            np.asarray([np.interp(b, b_grid, curve) for curve in logs])
        ) * perturbative[None, :]
        print(
            f"transform lambda600 {checkpoint} expb2 {flavor}: "
            f"{len(members_b)} exact members",
            flush=True,
        )
        transformed = engine.transform(b, members_b, "expb2")
        values[flavor_index] = transformed.reshape(24, 50, len(engine.k_grid))
        central_b = np.exp(np.interp(b, b_grid, central_log)) * perturbative
        central[flavor_index] = engine.transform(
            b, central_b[None, :], "expb2"
        )[0]
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(central)):
        raise RuntimeError(f"non-finite exact {checkpoint} expb2 member transform")
    return values, central


def checkpoint_arrays_to_bands(
    engine: ExactTransformEngine,
    values: np.ndarray,
    central: np.ndarray,
    *,
    checkpoint: str,
) -> pd.DataFrame:
    """Declare checkpoint bands directly from the exact stored member arrays."""
    values = np.asarray(values, dtype=float)
    central = np.asarray(central, dtype=float)
    expected = (len(FLAVORS), 24, 50, len(engine.k_grid))
    if values.shape != expected or central.shape != (len(FLAVORS), len(engine.k_grid)):
        raise RuntimeError(f"invalid exact {checkpoint} checkpoint array shape")
    rows = []
    for flavor_index, flavor in enumerate(FLAVORS):
        q16, median, q84 = np.quantile(
            values[flavor_index].reshape(1200, len(engine.k_grid)),
            (0.16, 0.50, 0.84),
            axis=0,
        )
        rows.append(pd.DataFrame({
            "model": "lambda600",
            "checkpoint": checkpoint,
            "tail_mode": "expb2",
            "flavor": flavor,
            "x": 0.1,
            "Q": 10.0,
            "kT": engine.k_grid,
            "q16": q16,
            "median": median,
            "q84": q84,
            "declared_central": central[flavor_index],
        }))
    return pd.concat(rows, ignore_index=True)


def require_band_identity(
    observed: pd.DataFrame, expected: pd.DataFrame, label: str
) -> None:
    """Require two u/d checkpoint band declarations to agree pointwise."""
    for flavor in FLAVORS:
        left = observed[observed["flavor"].astype(str).eq(flavor)].sort_values("kT")
        right = expected[expected["flavor"].astype(str).eq(flavor)].sort_values("kT")
        if (len(left) != 401 or len(right) != 401
                or not np.allclose(
                    left["kT"].to_numpy(float), right["kT"].to_numpy(float),
                    rtol=0.0, atol=1.0e-14,
                )
                or not np.allclose(
                    left[["q16", "median", "q84", "declared_central"]].to_numpy(float),
                    right[["q16", "median", "q84", "declared_central"]].to_numpy(float),
                    rtol=CHECKPOINT_QUANTILE_RTOL,
                    atol=CHECKPOINT_QUANTILE_ATOL,
                )):
            raise RuntimeError(f"{label} differs from exact arrays for {flavor}")


def source_terminal_expb2(
    engine: ExactTransformEngine,
    reference: pd.DataFrame,
    b_grid: np.ndarray,
    central_log: np.ndarray,
) -> pd.DataFrame:
    source = pd.read_csv(FINAL / "kT_tmd_bands.csv")
    source = source[source["component"].astype(str).eq("combined")]
    if "quantity" in source.columns:
        source = source[source["quantity"].astype(str).eq("ftilde")]
    source = source[
        np.isclose(source["Q"], 10.0)
        & source["flavor"].astype(str).isin(FLAVORS)
    ].copy()
    if source.groupby("flavor").size().to_dict() != {"d": 401, "u": 401}:
        raise RuntimeError("terminal expb2 source lacks exact u/d grids")
    rows = []
    for flavor in FLAVORS:
        group = source[source["flavor"].astype(str).eq(flavor)].sort_values("kT")
        ref = reference[
            np.isclose(reference["Q"], 10.0)
            & reference["flavor"].astype(str).eq(flavor)
        ].sort_values("bT")
        b = ref["bT"].to_numpy(float)
        central_b = (
            np.exp(np.interp(b, b_grid, central_log))
            * ref["ftilde_no_np"].to_numpy(float)
        )
        declared = engine.transform(b, central_b[None, :], "expb2")[0]
        rows.append(
            pd.DataFrame(
                {
                    "model": "lambda600",
                    "checkpoint": "terminal",
                    "tail_mode": "expb2",
                    "flavor": flavor,
                    "x": 0.1,
                    "Q": 10.0,
                    "kT": group["kT"].to_numpy(float),
                    "q16": group["q16"].to_numpy(float),
                    "median": group["median"].to_numpy(float),
                    "q84": group["q84"].to_numpy(float),
                    "declared_central": declared,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def harmonized_lambda1_members() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    module = load_module("postfit_harmonized_lambda1", HARMONIZED_BUILDER)
    input_manifest = module.validate_pinned_input_manifest()
    pinned, tags, _ = module.validate_pinned_incumbent()
    module.validate_registered_frozen_sources()
    b_grid, starts, _, _ = module.load_start_curves(tags)
    replica_logs, _, replica_metadata = module.load_replica_curves(b_grid)
    start_logs = np.log(starts)
    center = np.median(start_logs, axis=0)
    start_residuals = start_logs - center
    replica_residuals = replica_logs - np.median(replica_logs, axis=0)
    combined = (
        center[None, None, :]
        + start_residuals[:, None, :]
        + replica_residuals[None, :, :]
    ).reshape(1200, len(b_grid))
    return b_grid, combined, center, {
        "champion_id": pinned["champion_id"],
        "pinned_input_manifest_sha256": sha256(module.PINNED_INPUT_MANIFEST),
        "conditional_replica_metadata": replica_metadata,
        "training_protocol_harmonized": False,
        "postprocessing_only": True,
        "input_file_count": int(input_manifest["file_count"]),
    }


def harmonized_source_expb2(
    engine: ExactTransformEngine,
    reference: pd.DataFrame,
    b_grid: np.ndarray,
    central_log: np.ndarray,
) -> pd.DataFrame:
    source = pd.read_csv(HARMONIZED / "kspace_combined_bands.csv")
    if source.groupby("flavor").size().to_dict() != {"d": 401, "u": 401}:
        raise RuntimeError("harmonized lambda1 expb2 source lacks exact u/d grids")
    rows = []
    for flavor in FLAVORS:
        group = source[source["flavor"].astype(str).eq(flavor)].sort_values("kT")
        ref = reference[
            np.isclose(reference["Q"], 10.0)
            & reference["flavor"].astype(str).eq(flavor)
        ].sort_values("bT")
        b = ref["bT"].to_numpy(float)
        central_b = (
            np.exp(np.interp(b, b_grid, central_log))
            * ref["ftilde_no_np"].to_numpy(float)
        )
        declared = engine.transform(b, central_b[None, :], "expb2")[0]
        rows.append(
            pd.DataFrame(
                {
                    "model": "lambda1_harmonized",
                    "checkpoint": "historical_terminal",
                    "tail_mode": "expb2",
                    "flavor": flavor,
                    "x": 0.1,
                    "Q": 10.0,
                    "kT": group["kT"].to_numpy(float),
                    "q16": group["q16"].to_numpy(float),
                    "median": group["central"].to_numpy(float),
                    "q84": group["q84"].to_numpy(float),
                    "declared_central": declared,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def active(central: np.ndarray, k: np.ndarray) -> np.ndarray:
    displayed = k <= 2.25
    peak = float(np.max(central[displayed]))
    mask = displayed & (central > 0.05 * peak)
    if not np.any(mask):
        raise RuntimeError("empty positive active kT region")
    return mask


def band_arrays(frame: pd.DataFrame, flavor: str) -> tuple[np.ndarray, np.ndarray]:
    group = frame[frame["flavor"].astype(str).eq(flavor)].sort_values("kT")
    if len(group) != 401:
        raise RuntimeError(f"unexpected transformed grid for {flavor}")
    return group["kT"].to_numpy(float), group[
        ["q16", "median", "q84", "declared_central"]
    ].to_numpy(float).T


def incumbent_bands() -> pd.DataFrame:
    champion = json.loads(CHAMPION.read_text())
    source = Path(champion["artifacts"]["kspace_combined_bands"])
    if sha256(source) != champion["artifact_sha256"]["kspace_combined_bands"]:
        raise RuntimeError("immutable incumbent kspace band hash mismatch")
    frame = pd.read_csv(source)
    if "component" in frame.columns:
        frame = frame[frame["component"].astype(str).eq("combined")]
    if "quantity" in frame.columns:
        frame = frame[frame["quantity"].astype(str).eq("ftilde")]
    if "Q" in frame.columns:
        frame = frame[np.isclose(frame["Q"], 10.0)]
    if "central" not in frame.columns and "median" in frame.columns:
        frame = frame.rename(columns={"median": "central"})
    return frame


def checkpoint_outcome_metrics(
    terminal: pd.DataFrame,
    anchor: pd.DataFrame,
    incumbent: pd.DataFrame,
    allowance_by_flavor: dict[str, float],
    source_verdict_by_flavor: dict[str, bool],
) -> tuple[dict, bool]:
    """Evaluate terminal/anchor outcome stability without a new percent gate."""
    result: dict[str, dict] = {}
    all_pass = True
    for flavor in FLAVORS:
        k, terminal_q = band_arrays(terminal, flavor)
        anchor_k, anchor_q = band_arrays(anchor, flavor)
        old = incumbent[incumbent["flavor"].astype(str).eq(flavor)].sort_values(
            "kT"
        )
        if len(old) != len(k) or not np.allclose(
            anchor_k, k, rtol=0.0, atol=1.0e-12
        ) or not np.allclose(old["kT"].to_numpy(float), k, rtol=0.0, atol=1e-12):
            raise RuntimeError(f"checkpoint/incumbent kT grids differ for {flavor}")
        old_center = old["central"].to_numpy(float)
        terminal_active = active(terminal_q[1], k)
        anchor_active = active(anchor_q[1], k)
        incumbent_active = active(old_center, k)
        terminal_mask = terminal_active | incumbent_active
        anchor_mask = anchor_active | incumbent_active
        convergence_mask = terminal_active | anchor_active | incumbent_active

        terminal_width_curve = (terminal_q[2] - terminal_q[0]) / np.maximum(
            np.abs(terminal_q[1]), EPS
        )
        anchor_width_curve = (anchor_q[2] - anchor_q[0]) / np.maximum(
            np.abs(anchor_q[1]), EPS
        )
        terminal_width = float(np.max(terminal_width_curve[terminal_mask]))
        anchor_width = float(np.max(anchor_width_curve[anchor_mask]))
        allowance = float(allowance_by_flavor[flavor])
        locked = float(LOCKED_INCUMBENT_WIDTHS[flavor])
        terminal_verdict = bool(terminal_width + allowance < locked)
        anchor_verdict = bool(anchor_width + allowance < locked)
        source_verdict = explicit_bool(
            source_verdict_by_flavor[flavor], f"source verdict {flavor}"
        )

        scale = np.maximum(np.abs(terminal_q[1]), EPS)
        endpoint_movements = {
            "q16": float(
                np.max(np.abs(terminal_q[0] - anchor_q[0])[convergence_mask]
                       / scale[convergence_mask])
            ),
            "median": float(
                np.max(np.abs(terminal_q[1] - anchor_q[1])[convergence_mask]
                       / scale[convergence_mask])
            ),
            "q84": float(
                np.max(np.abs(terminal_q[2] - anchor_q[2])[convergence_mask]
                       / scale[convergence_mask])
            ),
        }
        width_movement = float(
            np.max(
                np.abs(terminal_width_curve - anchor_width_curve)[convergence_mask]
            )
        )
        envelope_low = np.minimum(anchor_q[0], terminal_q[0])
        envelope_high = np.maximum(anchor_q[2], terminal_q[2])
        envelope_width_curve = (envelope_high - envelope_low) / scale
        checkpoint_envelope_width = float(
            np.max(envelope_width_curve[convergence_mask])
        )
        tail_envelope_increment = max(0.0, checkpoint_envelope_width - terminal_width)
        charged_width = checkpoint_envelope_width + allowance
        verdict_stable = terminal_verdict == anchor_verdict
        source_match = terminal_verdict == source_verdict
        flavor_pass = bool(
            verdict_stable and source_match and charged_width < locked
        )
        all_pass = all_pass and flavor_pass
        result[flavor] = {
            "terminal_raw_full_width_on_terminal_incumbent_union": terminal_width,
            "anchor_raw_full_width_on_anchor_incumbent_union": anchor_width,
            "finite_ensemble_full_width_allowance": allowance,
            "terminal_anchor_exact_union_envelope_full_width": checkpoint_envelope_width,
            "tail_envelope_increment_beyond_terminal_width": tail_envelope_increment,
            "exact_union_envelope_plus_finite_ensemble_allowance": charged_width,
            "immutable_lambda1_width": locked,
            "terminal_incumbent_replacement_verdict": terminal_verdict,
            "anchor_incumbent_replacement_verdict": anchor_verdict,
            "source_stability_verdict": source_verdict,
            "terminal_matches_source_stability_verdict": source_match,
            "incumbent_replacement_verdict_unchanged_across_checkpoint": verdict_stable,
            "max_relative_checkpoint_movement_on_common_union_active_mask": endpoint_movements,
            "max_relative_full_width_movement_on_common_union_active_mask": width_movement,
            "promotion_validation_gate_pass": flavor_pass,
        }
    return result, bool(all_pass)


def transform_robustness_metrics(all_bands: pd.DataFrame) -> dict:
    """Require the candidate/control decision sign to survive all tail modes."""
    metrics: dict[str, dict] = {}
    decision_gate = True
    for mode in TAIL_MODES:
        candidate = all_bands[
            (all_bands["model"] == "lambda600")
            & (all_bands["checkpoint"] == "terminal")
            & (all_bands["tail_mode"] == mode)
        ]
        control = all_bands[
            (all_bands["model"] == "lambda1_harmonized")
            & (all_bands["tail_mode"] == mode)
        ]
        metrics[mode] = {}
        for flavor in FLAVORS:
            k, cq = band_arrays(candidate, flavor)
            lk, lq = band_arrays(control, flavor)
            if not np.allclose(k, lk, rtol=0.0, atol=1e-12):
                raise RuntimeError(f"tail-mode grids differ for {mode} {flavor}")
            union = active(cq[1], k) | active(lq[1], k)
            cwidth = (cq[2] - cq[0]) / np.maximum(np.abs(cq[1]), EPS)
            lwidth = (lq[2] - lq[0]) / np.maximum(np.abs(lq[1]), EPS)
            candidate_width = float(np.max(cwidth[union]))
            control_width = float(np.max(lwidth[union]))
            # The expb2 finite-ensemble allowance has not been proved
            # conservative for a different continuation transform.  Require
            # the raw sign here and explicitly defer any sampling-margin claim
            # until mode-specific 24x50 resampling is available.
            favorable = bool(candidate_width < control_width)
            decision_gate = decision_gate and favorable
            metrics[mode][flavor] = {
                "lambda600_raw_full_width_on_mode_matched_union": candidate_width,
                "harmonized_lambda1_raw_full_width_on_mode_matched_union": control_width,
                "lambda600_raw_width_is_smaller": favorable,
                "mode_specific_sampling_margin_computed": False,
                "expb2_sampling_margin_reused_for_this_mode": False,
                "raw_decision_sign_gate_pass": favorable,
            }
    expb2 = all_bands[all_bands["tail_mode"] == "expb2"]
    mode_changes: dict[str, dict] = {}
    for model in ("lambda600", "lambda1_harmonized"):
        reference = expb2[
            (expb2["model"] == model)
            & (
                (expb2["checkpoint"] == "terminal")
                if model == "lambda600"
                else np.ones(len(expb2), dtype=bool)
            )
        ]
        mode_changes[model] = {}
        for mode in TAIL_MODES:
            candidate = all_bands[
                (all_bands["model"] == model)
                & (all_bands["tail_mode"] == mode)
            ]
            if model == "lambda600":
                candidate = candidate[candidate["checkpoint"] == "terminal"]
            mode_changes[model][mode] = {}
            for flavor in FLAVORS:
                k, rq = band_arrays(reference, flavor)
                ck, cq = band_arrays(candidate, flavor)
                if not np.allclose(k, ck, rtol=0.0, atol=1e-12):
                    raise RuntimeError("tail-mode comparison grids differ")
                union = active(rq[1], k) | active(cq[1], k)
                scale = np.maximum(np.abs(rq[1]), EPS)
                mode_changes[model][mode][flavor] = {
                    "max_relative_ensemble_median_change_vs_expb2": float(
                        np.max(np.abs(cq[1] - rq[1])[union] / scale[union])
                    ),
                    "max_relative_declared_central_change_vs_expb2": float(
                        np.max(np.abs(cq[3] - rq[3])[union] / scale[union])
                    ),
                    "max_relative_q16_or_q84_change_vs_expb2": float(
                        max(
                            np.max(np.abs(cq[0] - rq[0])[union] / scale[union]),
                            np.max(np.abs(cq[2] - rq[2])[union] / scale[union]),
                        )
                    ),
                    "gating": False,
                }
    return {
        "mode_matched_lambda600_vs_harmonized_lambda1": metrics,
        "within_model_change_relative_to_expb2": mode_changes,
        "decision_robustness_gate_pass": bool(decision_gate),
        "gating": True,
        "gate_scope": "raw mode-matched width sign only",
        "sampling_margin_robustness_claimed": False,
        "sampling_margin_robustness_deferred_reason": (
            "a finite-ensemble allowance must be resampled from each mode's own "
            "transformed 24x50 arrays; the expb2 allowance is not silently reused"
        ),
        "primary_locked_width_gate_remains_expb2": True,
        "interpretation": (
            "the immutable registered expb2 widths remain the primary gate; "
            "promotion additionally requires the favorable raw lambda600 versus "
            "harmonized-lambda1 decision sign to survive every identical "
            "finite-b continuation mode"
        ),
    }


def containment(frame: pd.DataFrame, coordinate: str, limit: float) -> dict:
    result: dict[str, dict] = {}
    for keys, group in frame.groupby(
        [column for column in ("Q", "flavor") if column in frame.columns],
        sort=False,
    ):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        label = "|".join(str(value) for value in key_tuple)
        group = group.sort_values(coordinate)
        group = group[group[coordinate] <= limit]
        q16 = group["q16"].to_numpy(float)
        central_column = (
            "declared_trained_central"
            if "declared_trained_central" in group.columns
            else "declared_central"
        )
        central = group[central_column].to_numpy(float)
        q84 = group["q84"].to_numpy(float)
        scale = np.maximum.reduce([np.abs(q16), np.abs(central), np.abs(q84),
                                   np.ones_like(central)])
        tolerance = 64.0 * np.finfo(float).eps * scale
        inside = (central >= q16 - tolerance) & (central <= q84 + tolerance)
        result[label] = {
            "displayed_point_count": int(len(group)),
            "declared_central_inside_q16_q84_at_every_displayed_point": bool(
                np.all(inside)
            ),
            "outside_point_count": int(np.count_nonzero(~inside)),
            "numeric_slack_rule": "64*float64_epsilon*max(abs(q16),abs(central),abs(q84),1)",
        }
    return result


def all_contained(payload: dict) -> bool:
    return bool(payload) and all(
        explicit_bool(
            value["declared_central_inside_q16_q84_at_every_displayed_point"],
            "central containment",
        )
        for value in payload.values()
    )


def regional_fnp_quantile_movement(
    b_grid: np.ndarray, anchor: pd.DataFrame, terminal: pd.DataFrame
) -> dict:
    a = anchor.sort_values("bT")[["q16", "median", "q84"]].to_numpy(float).T
    t = terminal.sort_values("bT")[["q16", "median", "q84"]].to_numpy(float).T
    scale = np.maximum(np.abs(t[1]), 0.05)
    regions = {
        "formal_b0p1_4": (b_grid >= 0.1) & (b_grid <= 4.0),
        "unconstrained_b4_8": (b_grid > 4.0) & (b_grid <= 8.0),
        "tailfit_b6_8": (b_grid >= 6.0) & (b_grid <= 8.0),
        "full_b0p1_8": (b_grid >= 0.1) & (b_grid <= 8.0),
    }
    output = {}
    for region, mask in regions.items():
        output[region] = {
            name: float(np.max(np.abs(t[index] - a[index])[mask] / scale[mask]))
            for index, name in enumerate(("q16", "median", "q84"))
        }
    return output


def main() -> None:
    validate_fixed_challenger_protocol()
    selected = json.loads(SELECTED.read_text())
    replicas = json.loads(REPLICAS.read_text())
    final_summary = json.loads((FINAL / "summary.json").read_text())
    stability = json.loads(STABILITY.read_text())
    require_fixed_implementation_binding(final_summary, "final combined ensemble")
    require_fixed_implementation_binding(stability, "final stability audit")
    if final_summary.get("status") != "complete" or stability.get("status") != "complete":
        raise RuntimeError("final 24x50 builder and stability audit must complete first")
    if int(final_summary.get("start_count", 0)) != 24 or int(
        final_summary.get("experimental_replica_count", 0)
    ) != 50 or int(final_summary.get("combined_member_count", 0)) != 1200:
        raise RuntimeError("final combined ensemble lacks exact 24x50 coverage")
    allowance_by_flavor = {
        flavor: float(stability["resampling_full_width_allowance_by_flavor"][flavor])
        for flavor in FLAVORS
    }
    source_verdict = {
        flavor: explicit_bool(
            stability["robust_improvement_gate_by_flavor"][flavor],
            f"stability robust verdict {flavor}",
        )
        for flavor in FLAVORS
    }
    registered = {
        flavor: float(
            stability["comparison_champion_registered_own_mask_relative_full_width"][
                flavor
            ]
        )
        for flavor in FLAVORS
    }
    if any(
        not np.isclose(
            registered[flavor], LOCKED_INCUMBENT_WIDTHS[flavor], rtol=0.0, atol=1e-14
        )
        for flavor in FLAVORS
    ):
        raise RuntimeError("stability audit no longer uses immutable lambda1 widths")

    pairs = resolve_checkpoint_pairs(selected, replicas)
    b_anchor, starts_anchor, central_anchor, reps_anchor, ordered_anchor = (
        ordered_pair_curves(pairs, "anchor")
    )
    b_terminal, starts_terminal, central_terminal, reps_terminal, ordered_terminal = (
        ordered_pair_curves(pairs, "terminal")
    )
    if not np.allclose(b_anchor, b_terminal, rtol=0.0, atol=1e-12):
        raise RuntimeError("anchor and terminal FNP grids differ")
    if [item["identity"] for item in ordered_anchor] != [
        item["identity"] for item in ordered_terminal
    ]:
        raise RuntimeError("anchor and terminal chain orders differ")
    anchor_all = np.vstack([starts_anchor, central_anchor[None, :], reps_anchor])
    terminal_all = np.vstack(
        [starts_terminal, central_terminal[None, :], reps_terminal]
    )
    chain_movements = chain_movement_rows(
        b_terminal, anchor_all, terminal_all, ordered_terminal
    )

    anchor_logs, anchor_hierarchy = combined_log_members(
        central_anchor, starts_anchor, reps_anchor
    )
    terminal_logs, terminal_hierarchy = combined_log_members(
        central_terminal, starts_terminal, reps_terminal
    )
    anchor_fnp = quantile_frame(
        b_terminal, anchor_logs, np.log(central_anchor), "stationarity_anchor"
    )
    terminal_fnp = quantile_frame(
        b_terminal, terminal_logs, np.log(central_terminal), "terminal"
    )
    fnp_bands = pd.concat([anchor_fnp, terminal_fnp], ignore_index=True)

    reference = reference_table()
    bspace_bands, central_bspace = bspace_terminal_outputs(
        reference, b_terminal, terminal_logs, np.log(central_terminal)
    )
    transformer = load_module("postfit_exact_transform", TRANSFORMER)
    engine = ExactTransformEngine(transformer)
    terminal_values_expb2, terminal_central_expb2 = (
        transform_checkpoint_member_arrays(
            engine, reference, b_terminal, terminal_logs,
            np.log(central_terminal), checkpoint="terminal",
        )
    )
    anchor_values_expb2, anchor_central_expb2 = (
        transform_checkpoint_member_arrays(
            engine, reference, b_terminal, anchor_logs,
            np.log(central_anchor), checkpoint="stationarity_anchor",
        )
    )
    terminal_expb2 = checkpoint_arrays_to_bands(
        engine, terminal_values_expb2, terminal_central_expb2,
        checkpoint="terminal",
    )
    anchor_expb2 = checkpoint_arrays_to_bands(
        engine, anchor_values_expb2, anchor_central_expb2,
        checkpoint="stationarity_anchor",
    )
    terminal_source_expb2 = source_terminal_expb2(
        engine, reference, b_terminal, np.log(central_terminal)
    )
    require_band_identity(
        terminal_source_expb2, terminal_expb2,
        "stored terminal combined expb2 band",
    )

    terminal_modes = [terminal_expb2]
    for mode in ("expb", "taper"):
        terminal_modes.append(
            transform_bands(
                engine,
                reference,
                b_terminal,
                terminal_logs,
                np.log(central_terminal),
                model="lambda600",
                checkpoint="terminal",
                mode=mode,
            )
        )

    lambda1_b, lambda1_logs, lambda1_center, lambda1_provenance = (
        harmonized_lambda1_members()
    )
    lambda1_modes = [
        harmonized_source_expb2(
            engine, reference, lambda1_b, lambda1_center
        )
    ]
    for mode in ("expb", "taper"):
        lambda1_modes.append(
            transform_bands(
                engine,
                reference,
                lambda1_b,
                lambda1_logs,
                lambda1_center,
                model="lambda1_harmonized",
                checkpoint="historical_terminal",
                mode=mode,
            )
        )
    tailmode_bands = pd.concat(terminal_modes + lambda1_modes, ignore_index=True)
    checkpoint_bands = pd.concat([anchor_expb2, terminal_expb2], ignore_index=True)

    outcome, outcome_gate = checkpoint_outcome_metrics(
        terminal_expb2,
        anchor_expb2,
        incumbent_bands(),
        allowance_by_flavor,
        source_verdict,
    )
    robustness = transform_robustness_metrics(tailmode_bands)
    fnp_containment = containment(
        terminal_fnp.rename(
            columns={"declared_trained_central": "declared_central"}
        ).assign(Q=0.0, flavor="FNP"),
        "bT",
        8.0,
    )
    bspace_containment = containment(bspace_bands, "bT", 4.0)
    kspace_containment = containment(terminal_expb2, "kT", 2.25)
    central_containment_gate = bool(
        all_contained(fnp_containment)
        and all_contained(bspace_containment)
        and all_contained(kspace_containment)
    )
    transform_decision_gate = explicit_bool(
        robustness.get("decision_robustness_gate_pass"),
        "transform decision robustness gate",
    )
    stationarity = explicit_bool(
        final_summary.get("candidate_stationarity_gate_pass"),
        "final candidate stationarity",
    )
    evidence_complete = bool(
        len(chain_movements) == 75
        and len(anchor_logs) == 1200
        and len(terminal_logs) == 1200
        and np.all(np.isfinite(chain_movements.filter(like="max_relative").to_numpy()))
        and np.all(
            np.isfinite(
                tailmode_bands[
                    ["q16", "median", "q84", "declared_central"]
                ].to_numpy(float)
            )
        )
    )
    # The product-median-normalized checkpoint comparison remains useful as a
    # diagnostic, but its sampling allowance is not calibrated for the final
    # trained-central-normalized joint envelope.  Only the final-envelope
    # resampling audit can make the incumbent-replacement decision.
    promotion_gate = bool(
        evidence_complete
        and stationarity
        and transform_decision_gate
        and central_containment_gate
    )
    failure_reasons = []
    if not stationarity:
        failure_reasons.append("source FNP stationarity/fit-preservation gate failed")
    if not central_containment_gate:
        failure_reasons.append(
            "declared trained central is not contained by q16--q84 through the full "
            "FNP b<=8 audit or at every displayed Fig. 2/Fig. 6 point"
        )
    if not transform_decision_gate:
        failure_reasons.append(
            "raw lambda600 width does not remain narrower than the "
            "mode-matched harmonized lambda1 control for every flavor/tail mode"
        )
    if not evidence_complete:
        failure_reasons.append("post-fit full-tail evidence is incomplete or non-finite")

    TARGET.mkdir(parents=True, exist_ok=True)
    atomic_csv(TARGET / "endpoint_checkpoint_pairs_and_fnp_movement.csv", chain_movements)
    atomic_csv(TARGET / "fnp_checkpoint_bands.csv", fnp_bands)
    atomic_csv(TARGET / "terminal_bspace_bands_with_trained_central.csv", bspace_bands)
    atomic_csv(TARGET / "terminal_trained_central_bspace.csv", central_bspace)
    atomic_csv(TARGET / "kspace_checkpoint_expb2_bands.csv", checkpoint_bands)
    atomic_csv(TARGET / "kspace_tailmode_bands.csv", tailmode_bands)
    checkpoint_array_path = TARGET / "exact_checkpoint_expb2_members.npz"
    atomic_npz(
        checkpoint_array_path,
        schema=np.asarray(CHECKPOINT_ARRAY_SCHEMA),
        checkpoints=np.asarray(("terminal", "stationarity_anchor")),
        flavors=np.asarray(FLAVORS),
        start_seeds=np.arange(303, 327, dtype=np.int64),
        replica_seeds=np.arange(1001, 1051, dtype=np.int64),
        kT=engine.k_grid.astype(np.float64),
        terminal_values=terminal_values_expb2.astype(np.float64),
        stationarity_anchor_values=anchor_values_expb2.astype(np.float64),
        terminal_declared_central=terminal_central_expb2.astype(np.float64),
        stationarity_anchor_declared_central=anchor_central_expb2.astype(np.float64),
    )

    artifacts = {
        "endpoint_pairs": str(
            TARGET / "endpoint_checkpoint_pairs_and_fnp_movement.csv"
        ),
        "fnp_checkpoint_bands": str(TARGET / "fnp_checkpoint_bands.csv"),
        "terminal_bspace_bands_with_trained_central": str(
            TARGET / "terminal_bspace_bands_with_trained_central.csv"
        ),
        "terminal_trained_central_bspace": str(
            TARGET / "terminal_trained_central_bspace.csv"
        ),
        "kspace_checkpoint_expb2_bands": str(
            TARGET / "kspace_checkpoint_expb2_bands.csv"
        ),
        "kspace_tailmode_bands": str(TARGET / "kspace_tailmode_bands.csv"),
        "exact_checkpoint_expb2_members": str(checkpoint_array_path),
    }
    artifact_sha256 = {
        label: sha256(Path(path)) for label, path in artifacts.items()
    }
    summary = {
        "status": "complete_postfit_full_tail_transform_validation",
        "promotion_validation_gate_pass": promotion_gate,
        "promotion_eligible": False,
        "diagnostic_only": not promotion_gate,
        "diagnostic_figure_gate_pass": evidence_complete,
        "coverage_gate_pass": evidence_complete,
        "candidate_stationarity_gate_pass": stationarity,
        "central_line_containment_gate_pass": central_containment_gate,
        "tail_checkpoint_outcome_diagnostic_pass": outcome_gate,
        "obsolete_product_median_allowance_can_gate_postfit": False,
        "final_joint_sampling_gate_authoritative": True,
        "transform_decision_robustness_gate_pass": transform_decision_gate,
        "scientific_failure_reasons": failure_reasons,
        "chain_counts": {
            "optimizer_starts": 24,
            "trained_central": 1,
            "experimental_replicas": 50,
            "total": 75,
        },
        "combined_member_count": 1200,
        "formal_training_stationarity_domain": "x=0.1, 0.1<=bT<=4 GeV^-1",
        "postfit_audit_domain": "x=0.1, 0.1<=bT<=8 GeV^-1",
        "earlier_checkpoint_definition": (
            "passing chains use the exact stationarity_window_anchor_iterations "
            "checkpoint tested by their terminal ledger row; any scientifically "
            "nonstationary chain uses the fixed formal 200k checkpoint so a late "
            "reset cannot hide long-horizon movement"
        ),
        "failed_chain_anchor_policy": {
            "selection": "fixed_formal_200k_for_nonstationary_chain",
            "fixed_requested_lbfgs_capacity": 200_000,
            "late_reset_anchor_allowed_for_nonstationary_chain": False,
            "passing_chain_selection":
                "validated_terminal_stationarity_window_anchor",
            "nonstationary_chain_count": int(sum(
                bool(pair["scientifically_nonstationary"]) for pair in pairs
            )),
        },
        "fnp_checkpoint_quantile_movement": regional_fnp_quantile_movement(
            b_terminal, anchor_fnp, terminal_fnp
        ),
        "same_expb2_incumbent_replacement": {
            "gating": False,
            "flavor_metrics": outcome,
            "rule": (
                "legacy diagnostic only: terminal and earlier checkpoint bands "
                "are compared using the product-median-normalized statistic and "
                "its matching allowance. This result cannot accept or reject the "
                "postfit audit or final challenger. The final stage instead "
                "resamples both exact correlated checkpoint arrays, adds the "
                "interaction envelope, divides by the fixed trained central, and "
                "uses the fixed trained-central/incumbent union mask"
            ),
            "new_arbitrary_percent_tolerance_used": False,
        },
        "exact_checkpoint_transform_arrays": {
            "schema": CHECKPOINT_ARRAY_SCHEMA,
            "shape": [2, 24, 50, 401],
            "axis_order": ["flavor", "start_seed", "replica_seed", "kT"],
            "checkpoints": ["terminal", "stationarity_anchor"],
            "flavors": list(FLAVORS),
            "start_seed_range": [303, 326],
            "replica_seed_range": [1001, 1050],
            "quantiles_declared_from_exact_arrays": True,
        },
        "transform_robustness": robustness,
        "tail_modes": list(TAIL_MODES),
        "locked_gating_tail_mode": "expb2",
        "alternate_tail_modes_gate_promotion": True,
        "alternate_tail_modes_can_replace_or_loosen_locked_expb2_width_gate": False,
        "transform_settings": {
            "implementation": str(TRANSFORMER),
            "tail_fit_bmin": None,
            "b_transform_max": 24.0,
            "n_b_transform": 6001,
            "k_max": 4.0,
            "n_k": 401,
            "end_taper_start_fraction": 0.92,
            "convention": "f(k)=1/(2*pi) int db b J0(kb) ftilde(b)",
        },
        "declared_central_line": (
            "the separately trained terminal lambda600 central endpoint (forced "
            "to the 300k requested-capacity horizon), propagated in b space and "
            "through the exact paired finite-b transform; ensemble medians are "
            "retained as diagnostics but are not the plotted central model"
        ),
        "central_containment": {
            "FNP_bT_le_8": fnp_containment,
            "Fig2_bT_le_4": bspace_containment,
            "Fig6_kT_le_2p25": kspace_containment,
        },
        "hierarchy": {
            "combination_rule": (
                "trained central plus pointwise-median-centered whole-curve "
                "24-start and 50-replica residuals in log(F_NP)"
            ),
            "terminal": terminal_hierarchy,
            "stationarity_anchor": anchor_hierarchy,
            "cartesian_members_are_independent_nested_fits": False,
        },
        "harmonized_lambda1": lambda1_provenance,
        "input_provenance": {
            "selected_summary": str(SELECTED),
            "selected_summary_sha256": sha256(SELECTED),
            "start_ledger": str(START_LEDGER),
            "start_ledger_sha256": sha256(START_LEDGER),
            "replica_summary": str(REPLICAS),
            "replica_summary_sha256": sha256(REPLICAS),
            "replica_ledger": str(REPLICA_LEDGER),
            "replica_ledger_sha256": sha256(REPLICA_LEDGER),
            "final_combined_summary": str(FINAL / "summary.json"),
            "final_combined_summary_sha256": sha256(FINAL / "summary.json"),
            "stability_summary": str(STABILITY),
            "stability_summary_sha256": sha256(STABILITY),
            "reference_bspace": str(REFERENCE_B),
            "reference_bspace_sha256": sha256(REFERENCE_B),
            "transformer": str(TRANSFORMER),
            "transformer_sha256": sha256(TRANSFORMER),
        },
        "artifacts": artifacts,
        "artifact_sha256": artifact_sha256,
        **fixed_implementation_binding(),
        "fixed_challenger_protocol_modified": False,
        "frozen_sources_modified": False,
        "production_sources_modified": False,
    }
    atomic_json(TARGET / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
