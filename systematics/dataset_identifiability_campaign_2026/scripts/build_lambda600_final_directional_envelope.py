#!/usr/bin/env python3
"""Build the final product+convergence+interaction directional envelope.

This is a read-only post-processing step.  The displayed and gating interval
is deliberately not a confidence interval: it is the empirical 24x50 product
band enlarged first by terminal-versus-stationarity-anchor motion and then by
the observed 2x3 nested nonadditive interaction directions.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fixed_challenger_protocol import (
    fixed_implementation_binding,
    require_fixed_implementation_binding,
)
from nested_interaction_and_final_envelope_validation import (
    explicit_bool,
    validated_nested_interaction,
)
from postfit_tail_transform_validation import (
    validated_exact_checkpoint_transform_arrays,
    validated_postfit_tail_audit,
)


BASE = Path(__file__).resolve().parents[1]
SYSTEMATICS = BASE.parent
SUM = BASE / "summaries"
FINAL_PRODUCT = SUM / "final_combined_tmd_ensemble/summary.json"
STABILITY = SUM / "final_combined_ensemble_stability/summary.json"
POSTFIT = SUM / "lambda600_postfit_tail_transform_audit/summary.json"
NESTED = SUM / "lambda600_nested_start_replica_interaction/summary.json"
INCUMBENT = (
    SUM / "champion_registry/empirical_reference_lambda1_b0p1_2p0_full24.json"
)
REFERENCE_B = (
    SYSTEMATICS / "collins_factorization_validity/plots/"
    "rowidfix_stageFT_all_qmax0p20_lam0p50_central_exactx/"
    "v22_scheme_tmd_bspace_long.csv"
)
TARGET = SUM / "lambda600_final_directional_envelope"
FLAVORS_K = ("u", "d")
FLAVORS_B = ("u", "d", "s", "ubar", "dbar", "sbar")
LOCKED_WIDTHS = {"u": 0.11772613918747582, "d": 0.12490071924111977}
EPS = 1.0e-30
BOOTSTRAPS = 300
SPLITS = 200
RESAMPLING_SEED = 20260806


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


def active(central: np.ndarray, coordinate: np.ndarray, limit: float) -> np.ndarray:
    displayed = coordinate <= limit + 1.0e-12
    peak = float(np.max(central[displayed]))
    result = displayed & (central > 0.05 * peak)
    if not np.any(result):
        raise RuntimeError("empty positive active region")
    return result


def incumbent_bands() -> pd.DataFrame:
    record = json.loads(INCUMBENT.read_text())
    if record.get("champion_id") != "empirical_reference_lambda1_b0p1_2p0_full24":
        raise RuntimeError("immutable incumbent identity changed")
    path = Path(record["artifacts"]["kspace_combined_bands"])
    if sha256(path) != record["artifact_sha256"]["kspace_combined_bands"]:
        raise RuntimeError("immutable incumbent k-space band changed")
    frame = pd.read_csv(path)
    if "component" in frame.columns:
        frame = frame[frame["component"].astype(str).eq("combined")]
    if "quantity" in frame.columns:
        frame = frame[frame["quantity"].astype(str).eq("ftilde")]
    if "Q" in frame.columns:
        frame = frame[np.isclose(frame["Q"], 10.0)]
    if "central" not in frame.columns and "median" in frame.columns:
        frame = frame.rename(columns={"median": "central"})
    return frame


def final_joint_width_statistic(
    terminal: np.ndarray,
    anchor: np.ndarray,
    interaction_low: np.ndarray,
    interaction_high: np.ndarray,
    trained_central: np.ndarray,
    fixed_union_mask: np.ndarray,
) -> np.ndarray:
    """Evaluate exactly the final trained-central-normalized width statistic."""
    terminal = np.asarray(terminal, dtype=float)
    anchor = np.asarray(anchor, dtype=float)
    interaction_low = np.asarray(interaction_low, dtype=float)
    interaction_high = np.asarray(interaction_high, dtype=float)
    trained_central = np.asarray(trained_central, dtype=float)
    fixed_union_mask = np.asarray(fixed_union_mask, dtype=bool)
    if (terminal.ndim != 4 or anchor.shape != terminal.shape
            or terminal.shape[0] != len(FLAVORS_K)
            or terminal.shape[1] < 1 or terminal.shape[2] < 1
            or interaction_low.shape != terminal.shape[::3]
            or interaction_high.shape != interaction_low.shape
            or trained_central.shape != interaction_low.shape
            or fixed_union_mask.shape != interaction_low.shape
            or not np.all(np.isfinite(terminal))
            or not np.all(np.isfinite(anchor))
            or not np.all(np.isfinite(interaction_low))
            or not np.all(np.isfinite(interaction_high))
            or not np.all(np.isfinite(trained_central))
            or np.any(interaction_low > 0.0)
            or np.any(interaction_high < 0.0)
            or not np.all(np.any(fixed_union_mask, axis=1))):
        raise RuntimeError("invalid inputs for exact final joint-width statistic")
    flat_terminal = terminal.reshape(
        terminal.shape[0], terminal.shape[1] * terminal.shape[2], terminal.shape[3]
    )
    flat_anchor = anchor.reshape(
        anchor.shape[0], anchor.shape[1] * anchor.shape[2], anchor.shape[3]
    )
    terminal_q16, terminal_q84 = np.quantile(
        flat_terminal, (0.16, 0.84), axis=1
    )
    anchor_q16, anchor_q84 = np.quantile(
        flat_anchor, (0.16, 0.84), axis=1
    )
    low = np.minimum(terminal_q16, anchor_q16) + interaction_low
    high = np.maximum(terminal_q84, anchor_q84) + interaction_high
    relative = (high - low) / np.maximum(np.abs(trained_central), EPS)
    if (not np.all(np.isfinite(relative[fixed_union_mask]))
            or np.any(relative[fixed_union_mask] < 0.0)):
        raise RuntimeError("exact final joint width is non-finite or negative")
    return np.asarray([
        np.max(relative[index, fixed_union_mask[index]])
        for index in range(len(FLAVORS_K))
    ])


def exact_final_statistic_resampling(
    kspace: pd.DataFrame,
    checkpoint_arrays: dict[str, np.ndarray],
    incumbent: pd.DataFrame,
    *,
    bootstraps: int = BOOTSTRAPS,
    splits: int = SPLITS,
    seed: int = RESAMPLING_SEED,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Bootstrap/split the two correlated checkpoints using the final statistic."""
    if bootstraps < 1 or splits < 1:
        raise RuntimeError("final-statistic resampling counts must be positive")
    k = np.asarray(checkpoint_arrays["kT"], dtype=float)
    terminal = np.asarray(checkpoint_arrays["terminal_values"], dtype=float)
    anchor = np.asarray(
        checkpoint_arrays["stationarity_anchor_values"], dtype=float
    )
    trained = np.asarray(
        checkpoint_arrays["terminal_declared_central"], dtype=float
    )
    if (checkpoint_arrays["flavors"].tolist() != list(FLAVORS_K)
            or terminal.shape != (2, 24, 50, 401)
            or anchor.shape != terminal.shape or trained.shape != (2, 401)):
        raise RuntimeError("exact checkpoint arrays have invalid final-statistic shape")

    interaction_low = np.empty((2, 401), dtype=float)
    interaction_high = np.empty((2, 401), dtype=float)
    fixed_mask = np.empty((2, 401), dtype=bool)
    declared_final_width = np.empty(2, dtype=float)
    for flavor_index, flavor in enumerate(FLAVORS_K):
        group = kspace[kspace["flavor"].astype(str).eq(flavor)].sort_values("kT")
        old = incumbent[incumbent["flavor"].astype(str).eq(flavor)].sort_values("kT")
        if (len(group) != 401 or len(old) != 401
                or not np.array_equal(group["kT"].to_numpy(float), k)
                or not np.array_equal(old["kT"].to_numpy(float), k)
                or not np.allclose(group["trained_central"].to_numpy(float),
                                   trained[flavor_index], rtol=5e-13, atol=5e-15)):
            raise RuntimeError(f"final-statistic geometry differs for {flavor}")
        own = active(trained[flavor_index], k, 2.25)
        incumbent_active = active(old["central"].to_numpy(float), k, 2.25)
        fixed_mask[flavor_index] = own | incumbent_active
        interaction_low[flavor_index] = group[
            "interaction_delta_low"
        ].to_numpy(float)
        interaction_high[flavor_index] = group[
            "interaction_delta_high"
        ].to_numpy(float)
        relative = (
            group["final_envelope_high"].to_numpy(float)
            - group["final_envelope_low"].to_numpy(float)
        ) / np.maximum(np.abs(trained[flavor_index]), EPS)
        declared_final_width[flavor_index] = np.max(
            relative[fixed_mask[flavor_index]]
        )

    full = final_joint_width_statistic(
        terminal, anchor, interaction_low, interaction_high, trained, fixed_mask
    )
    if not np.allclose(full, declared_final_width, rtol=5e-13, atol=5e-15):
        raise RuntimeError("exact arrays do not reproduce the declared final width")

    rng = np.random.default_rng(seed)
    bootstrap = np.empty((bootstraps, 2), dtype=float)
    for index in range(bootstraps):
        starts = rng.integers(0, terminal.shape[1], terminal.shape[1])
        replicas = rng.integers(0, terminal.shape[2], terminal.shape[2])
        statistic = final_joint_width_statistic(
            terminal[:, starts][:, :, replicas],
            anchor[:, starts][:, :, replicas],
            interaction_low, interaction_high, trained, fixed_mask,
        )
        bootstrap[index] = np.abs(statistic - full)

    start_split = np.empty((splits, 2), dtype=float)
    replica_split = np.empty((splits, 2), dtype=float)
    joint_split = np.empty((splits, 2), dtype=float)
    for index in range(splits):
        start_order = rng.permutation(terminal.shape[1])
        replica_order = rng.permutation(terminal.shape[2])
        start_a, start_b = np.array_split(start_order, 2)
        replica_a, replica_b = np.array_split(replica_order, 2)

        def statistic(start_indices: np.ndarray, replica_indices: np.ndarray) -> np.ndarray:
            return final_joint_width_statistic(
                terminal[:, start_indices][:, :, replica_indices],
                anchor[:, start_indices][:, :, replica_indices],
                interaction_low, interaction_high, trained, fixed_mask,
            )

        start_split[index] = np.abs(
            statistic(start_a, np.arange(50))
            - statistic(start_b, np.arange(50))
        )
        replica_split[index] = np.abs(
            statistic(np.arange(24), replica_a)
            - statistic(np.arange(24), replica_b)
        )
        joint_split[index] = np.abs(
            statistic(start_a, replica_a) - statistic(start_b, replica_b)
        )

    p95 = {
        "bootstrap": np.quantile(bootstrap, 0.95, axis=0),
        "start_split": np.quantile(start_split, 0.95, axis=0),
        "replica_split": np.quantile(replica_split, 0.95, axis=0),
        "joint_split": np.quantile(joint_split, 0.95, axis=0),
    }
    allowance = np.maximum.reduce(list(p95.values()))
    summary = {
        "bootstrap_replicates": int(bootstraps),
        "split_half_replicates": int(splits),
        "rng_seed": int(seed),
        "full_exact_final_statistic_by_flavor": dict(zip(FLAVORS_K, full.tolist())),
        "bootstrap_p95_absolute_deviation_by_flavor": dict(
            zip(FLAVORS_K, p95["bootstrap"].tolist())
        ),
        "start_split_p95_absolute_difference_by_flavor": dict(
            zip(FLAVORS_K, p95["start_split"].tolist())
        ),
        "replica_split_p95_absolute_difference_by_flavor": dict(
            zip(FLAVORS_K, p95["replica_split"].tolist())
        ),
        "joint_split_p95_absolute_difference_by_flavor": dict(
            zip(FLAVORS_K, p95["joint_split"].tolist())
        ),
        "allowance_by_flavor": dict(zip(FLAVORS_K, allowance.tolist())),
        "statistic": (
            "max over the fixed trained-central/incumbent union mask of "
            "[max(q84_terminal,q84_anchor)+interaction_high-"
            "min(q16_terminal,q16_anchor)-interaction_low] divided by the "
            "absolute fixed trained-300k central"
        ),
        "correlation_preserved": (
            "terminal and stationarity-anchor arrays use identical resampled "
            "start/replica identities in every bootstrap and split"
        ),
        "interaction_resampled": False,
        "interaction_treatment": (
            "the deterministic nested directional interaction endpoints are fixed "
            "while finite 24x50 checkpoint sampling is resampled"
        ),
    }
    bootstrap_frame = pd.DataFrame({
        **{f"{flavor}_absolute_final_statistic_deviation": bootstrap[:, index]
           for index, flavor in enumerate(FLAVORS_K)},
        "max_absolute_final_statistic_deviation": np.max(bootstrap, axis=1),
    })
    split_frame = pd.DataFrame({
        **{f"start_split_{flavor}_absolute_final_statistic_difference":
           start_split[:, index] for index, flavor in enumerate(FLAVORS_K)},
        **{f"replica_split_{flavor}_absolute_final_statistic_difference":
           replica_split[:, index] for index, flavor in enumerate(FLAVORS_K)},
        **{f"joint_split_{flavor}_absolute_final_statistic_difference":
           joint_split[:, index] for index, flavor in enumerate(FLAVORS_K)},
    })
    return summary, bootstrap_frame, split_frame


def build_fnp(postfit: dict, nested: dict) -> pd.DataFrame:
    product = pd.read_csv(Path(postfit["artifacts"]["fnp_checkpoint_bands"]))
    terminal = product[product["checkpoint"].astype(str).eq("terminal")].sort_values("bT")
    anchor = product[
        product["checkpoint"].astype(str).eq("stationarity_anchor")
    ].sort_values("bT")
    interaction = pd.read_csv(
        Path(nested["artifacts"]["logfnp_directional_envelope"])
    ).sort_values("bT")
    if (len(terminal) != 321 or len(anchor) != 321 or len(interaction) != 321
            or not np.array_equal(terminal["bT"].to_numpy(float),
                                  anchor["bT"].to_numpy(float))
            or not np.array_equal(terminal["bT"].to_numpy(float),
                                  interaction["bT"].to_numpy(float))):
        raise RuntimeError("FNP product/checkpoint/interaction grids differ")
    convergence_low = np.minimum(
        terminal["q16"].to_numpy(float), anchor["q16"].to_numpy(float)
    )
    convergence_high = np.maximum(
        terminal["q84"].to_numpy(float), anchor["q84"].to_numpy(float)
    )
    log_low = interaction["interaction_log_delta_low"].to_numpy(float)
    log_high = interaction["interaction_log_delta_high"].to_numpy(float)
    final_low = convergence_low * np.exp(log_low)
    final_high = convergence_high * np.exp(log_high)
    central = terminal["declared_trained_central"].to_numpy(float)
    if (not np.all(np.isfinite(final_low)) or not np.all(np.isfinite(final_high))
            or np.any(final_low <= 0.0) or np.any(final_low > final_high)):
        raise RuntimeError("full FNP envelope is non-finite, non-positive, or inverted")
    return pd.DataFrame({
        "x": 0.1,
        "bT": terminal["bT"].to_numpy(float),
        "terminal_product_q16": terminal["q16"].to_numpy(float),
        "terminal_product_median": terminal["median"].to_numpy(float),
        "terminal_product_q84": terminal["q84"].to_numpy(float),
        "anchor_product_q16": anchor["q16"].to_numpy(float),
        "anchor_product_median": anchor["median"].to_numpy(float),
        "anchor_product_q84": anchor["q84"].to_numpy(float),
        "convergence_envelope_low": convergence_low,
        "convergence_envelope_high": convergence_high,
        "interaction_log_delta_low": log_low,
        "interaction_log_delta_high": log_high,
        "final_envelope_low": final_low,
        "trained_central": central,
        "final_envelope_high": final_high,
    })


def build_bspace(fnp: pd.DataFrame) -> pd.DataFrame:
    reference = pd.read_csv(REFERENCE_B)
    reference = reference[
        np.isclose(reference["x"], 0.1)
        & np.isclose(reference["Q"], 7.5)
        & reference["flavor"].astype(str).isin(FLAVORS_B)
    ].copy()
    rows = []
    for flavor in FLAVORS_B:
        group = reference[reference["flavor"].astype(str).eq(flavor)].sort_values("bT")
        b = group["bT"].to_numpy(float)
        perturbative = group["ftilde_no_np"].to_numpy(float)
        if (len(group) != 321 or np.any(perturbative <= 0.0)
                or not np.all(np.isfinite(perturbative))):
            raise RuntimeError(
                f"positive deterministic Fig.2 perturbative scaling is unavailable for {flavor}"
            )
        payload = {column: np.interp(b, fnp["bT"], fnp[column])
                   for column in (
                       "terminal_product_q16", "terminal_product_median",
                       "terminal_product_q84", "convergence_envelope_low",
                       "convergence_envelope_high", "final_envelope_low",
                       "trained_central", "final_envelope_high")}
        rows.append(pd.DataFrame({
            "x": 0.1, "Q": 7.5, "flavor": flavor, "bT": b,
            **{key: value * perturbative for key, value in payload.items()},
        }))
    return pd.concat(rows, ignore_index=True)


def build_kspace(postfit: dict, nested: dict) -> pd.DataFrame:
    product = pd.read_csv(
        Path(postfit["artifacts"]["kspace_checkpoint_expb2_bands"])
    )
    product = product[
        product["model"].astype(str).eq("lambda600")
        & product["tail_mode"].astype(str).eq("expb2")
    ]
    interaction = pd.read_csv(
        Path(nested["artifacts"]["kspace_directional_envelope"])
    )
    rows = []
    for flavor in FLAVORS_K:
        terminal = product[
            product["flavor"].astype(str).eq(flavor)
            & product["checkpoint"].astype(str).eq("terminal")
        ].sort_values("kT")
        anchor = product[
            product["flavor"].astype(str).eq(flavor)
            & product["checkpoint"].astype(str).eq("stationarity_anchor")
        ].sort_values("kT")
        delta = interaction[
            interaction["flavor"].astype(str).eq(flavor)
        ].sort_values("kT")
        grid = terminal["kT"].to_numpy(float)
        if (len(terminal) != 401 or len(anchor) != 401 or len(delta) != 401
                or not np.array_equal(grid, anchor["kT"].to_numpy(float))
                or not np.array_equal(grid, delta["kT"].to_numpy(float))):
            raise RuntimeError(f"k-space grids differ for {flavor}")
        convergence_low = np.minimum(
            terminal["q16"].to_numpy(float), anchor["q16"].to_numpy(float)
        )
        convergence_high = np.maximum(
            terminal["q84"].to_numpy(float), anchor["q84"].to_numpy(float)
        )
        delta_low = delta["interaction_delta_low"].to_numpy(float)
        delta_high = delta["interaction_delta_high"].to_numpy(float)
        final_low = convergence_low + delta_low
        final_high = convergence_high + delta_high
        central = terminal["declared_central"].to_numpy(float)
        rows.append(pd.DataFrame({
            "x": 0.1, "Q": 10.0, "flavor": flavor, "kT": grid,
            "terminal_product_q16": terminal["q16"].to_numpy(float),
            "terminal_product_median": terminal["median"].to_numpy(float),
            "terminal_product_q84": terminal["q84"].to_numpy(float),
            "anchor_product_q16": anchor["q16"].to_numpy(float),
            "anchor_product_median": anchor["median"].to_numpy(float),
            "anchor_product_q84": anchor["q84"].to_numpy(float),
            "convergence_envelope_low": convergence_low,
            "convergence_envelope_high": convergence_high,
            "interaction_delta_low": delta_low,
            "interaction_delta_high": delta_high,
            "final_envelope_low": final_low,
            "trained_central": central,
            "final_envelope_high": final_high,
        }))
    return pd.concat(rows, ignore_index=True)


def width_metrics(
    kspace: pd.DataFrame, stability: dict, final_resampling: dict
) -> tuple[pd.DataFrame, dict, bool]:
    incumbent = incumbent_bands()
    rows = []
    result = {}
    overall = True
    for flavor in FLAVORS_K:
        group = kspace[kspace["flavor"].astype(str).eq(flavor)].sort_values("kT")
        old = incumbent[incumbent["flavor"].astype(str).eq(flavor)].sort_values("kT")
        grid = group["kT"].to_numpy(float)
        if len(old) != len(group) or not np.array_equal(
                old["kT"].to_numpy(float), grid):
            raise RuntimeError(f"final/incumbent k grid differs for {flavor}")
        central = group["trained_central"].to_numpy(float)
        own = active(central, grid, 2.25)
        old_active = active(old["central"].to_numpy(float), grid, 2.25)
        union = own | old_active
        scale = np.maximum(np.abs(central), EPS)
        product_width = (
            group["terminal_product_q84"].to_numpy(float)
            - group["terminal_product_q16"].to_numpy(float)
        ) / scale
        convergence_width = (
            group["convergence_envelope_high"].to_numpy(float)
            - group["convergence_envelope_low"].to_numpy(float)
        ) / scale
        final_width = (
            group["final_envelope_high"].to_numpy(float)
            - group["final_envelope_low"].to_numpy(float)
        ) / scale
        product_max = float(np.max(product_width[union]))
        convergence_max = float(np.max(convergence_width[union]))
        final_max = float(np.max(final_width[union]))
        allowance = float(final_resampling["allowance_by_flavor"][flavor])
        prior_allowance = float(
            stability["resampling_full_width_allowance_by_flavor"][flavor]
        )
        resampled_full = float(
            final_resampling["full_exact_final_statistic_by_flavor"][flavor]
        )
        locked = float(LOCKED_WIDTHS[flavor])
        if (not np.isfinite(allowance) or allowance < 0.0
                or not np.isfinite(prior_allowance) or prior_allowance < 0.0
                or not np.isclose(resampled_full, final_max, rtol=5e-13, atol=5e-15)):
            raise RuntimeError(f"invalid exact final-statistic sampling evidence for {flavor}")
        adjusted = final_max + allowance
        passed = bool(adjusted < locked)
        overall = overall and passed
        result[flavor] = {
            "terminal_product_raw_full_width": product_max,
            "terminal_anchor_convergence_raw_full_width": convergence_max,
            "joint_convergence_interaction_raw_full_width": final_max,
            "convergence_full_width_increment": convergence_max - product_max,
            "interaction_full_width_increment_beyond_convergence": final_max - convergence_max,
            "corrected_finite_sampling_full_width_margin": allowance,
            "final_statistic_finite_sampling_full_width_margin": allowance,
            "prior_product_median_statistic_sampling_margin_diagnostic":
                prior_allowance,
            "joint_raw_width_plus_corrected_sampling_margin": adjusted,
            "joint_raw_width_plus_final_statistic_sampling_margin": adjusted,
            "immutable_lambda1_width": locked,
            "replacement_gate_pass": passed,
            "active_kT_max": float(grid[union].max()),
            "width_denominator": "absolute trained 300k central endpoint",
            "sampling_allowance_statistic_matches_width_statistic": True,
        }
        rows.append(pd.DataFrame({
            "flavor": flavor, "kT": grid,
            "candidate_active_mask": own,
            "incumbent_active_mask": old_active,
            "comparison_union_active_mask": union,
            "terminal_product_relative_full_width": product_width,
            "terminal_anchor_convergence_relative_full_width": convergence_width,
            "joint_convergence_interaction_relative_full_width": final_width,
        }))
    return pd.concat(rows, ignore_index=True), result, bool(overall)


def containment(frame: pd.DataFrame, coordinate: str, limit: float) -> bool:
    view = frame[frame[coordinate] <= limit + 1.0e-12]
    low = view["final_envelope_low"].to_numpy(float)
    central = view["trained_central"].to_numpy(float)
    high = view["final_envelope_high"].to_numpy(float)
    scale = np.maximum.reduce([np.abs(low), np.abs(central), np.abs(high),
                               np.ones_like(low)])
    tolerance = 64.0 * np.finfo(float).eps * scale
    return bool(np.all((central >= low - tolerance) & (central <= high + tolerance)))


def main() -> None:
    postfit, postfit_hash = validated_postfit_tail_audit(POSTFIT, STABILITY)
    nested, nested_hash = validated_nested_interaction(NESTED, POSTFIT, STABILITY)
    stability = json.loads(STABILITY.read_text())
    product_summary = json.loads(FINAL_PRODUCT.read_text())
    require_fixed_implementation_binding(stability, "final stability audit")
    require_fixed_implementation_binding(product_summary, "final product summary")
    if product_summary.get("status") != "complete" or int(
            product_summary.get("combined_member_count", -1)) != 1200:
        raise RuntimeError("final product hierarchy is incomplete")
    fnp = build_fnp(postfit, nested)
    bspace = build_bspace(fnp)
    kspace = build_kspace(postfit, nested)
    checkpoint_arrays = validated_exact_checkpoint_transform_arrays(postfit)
    final_resampling, bootstrap_frame, split_frame = (
        exact_final_statistic_resampling(
            kspace, checkpoint_arrays, incumbent_bands()
        )
    )
    widths_frame, widths, width_gate = width_metrics(
        kspace, stability, final_resampling
    )
    containment_gate = bool(
        containment(fnp, "bT", 8.0)
        and containment(bspace, "bT", 4.0)
        and containment(kspace, "kT", 2.25)
    )
    base_gate = bool(
        explicit_bool(
            stability.get("diagnostic_figure_gate_pass"),
            "base diagnostic evidence gate",
        )
        and explicit_bool(
            stability.get("candidate_stationarity_gate_pass"),
            "base candidate stationarity gate",
        )
    )
    prior_product_median_gate = explicit_bool(
        stability.get("endpoint_gate_pass"), "prior product-median endpoint gate"
    )
    postfit_gate = explicit_bool(
        postfit.get("promotion_validation_gate_pass"), "postfit promotion gate"
    )
    nested_gate = explicit_bool(
        nested.get("interaction_validation_gate_pass"), "nested interaction gate"
    )
    promotion_gate = bool(
        base_gate and postfit_gate and nested_gate and width_gate and containment_gate
    )
    reasons = [
        str(value) for value in postfit.get("scientific_failure_reasons", [])
        if str(value)
    ]
    if not base_gate:
        reasons.append(
            "base 24x50 evidence completeness or FNP stationarity gate failed"
        )
    if not nested_gate:
        reasons.append("nested start-by-replica interaction validation gate failed")
    if not width_gate:
        reasons.append(
            "joint convergence/interaction envelope plus exact final-statistic "
            "sampling margin "
            "does not beat immutable lambda1 for both u and d"
        )
    if not containment_gate:
        reasons.append("trained central is not contained by the final directional envelope")
    reasons = list(dict.fromkeys(reasons))

    TARGET.mkdir(parents=True, exist_ok=True)
    artifact_paths = {
        "fnp_final_envelope": TARGET / "fnp_final_directional_envelope.csv",
        "fig2_bspace_final_envelope": TARGET / "fig2_bspace_final_directional_envelope.csv",
        "fig6_kspace_final_envelope": TARGET / "fig6_kspace_final_directional_envelope.csv",
        "fig6_component_width_curves": TARGET / "fig6_component_width_curves.csv",
        "final_statistic_bootstrap_deviations":
            TARGET / "final_statistic_bootstrap_deviations.csv",
        "final_statistic_split_differences":
            TARGET / "final_statistic_split_differences.csv",
    }
    atomic_csv(artifact_paths["fnp_final_envelope"], fnp)
    atomic_csv(artifact_paths["fig2_bspace_final_envelope"], bspace)
    atomic_csv(artifact_paths["fig6_kspace_final_envelope"], kspace)
    atomic_csv(artifact_paths["fig6_component_width_curves"], widths_frame)
    atomic_csv(
        artifact_paths["final_statistic_bootstrap_deviations"], bootstrap_frame
    )
    atomic_csv(
        artifact_paths["final_statistic_split_differences"], split_frame
    )
    artifacts = {key: str(path) for key, path in artifact_paths.items()}
    hashes = {key: sha256(path) for key, path in artifact_paths.items()}
    summary = {
        "status": "complete_final_directional_envelope",
        "promotion_validation_gate_pass": promotion_gate,
        "promotion_eligible": False,
        "diagnostic_only": not promotion_gate,
        "diagnostic_figure_gate_pass": True,
        "combined_product_member_count": 1200,
        "base_product_stability_gate_pass": base_gate,
        "prior_product_median_endpoint_gate_diagnostic": prior_product_median_gate,
        "prior_product_median_sampling_gate_authoritative": False,
        "postfit_tail_convergence_gate_pass": postfit_gate,
        "nested_interaction_validation_gate_pass": nested_gate,
        "joint_width_replacement_gate_pass": width_gate,
        "final_joint_sampling_gate_authoritative": True,
        "trained_central_containment_gate_pass": containment_gate,
        "scientific_failure_reasons": reasons,
        "envelope_rule_kspace": (
            "low=min(terminal product q16, stationarity-anchor product q16)+"
            "observed nested interaction_delta_low; high=max(terminal product q84, "
            "stationarity-anchor product q84)+observed nested interaction_delta_high"
        ),
        "envelope_rule_fnp_bspace": (
            "the analogous full-grid log-FNP directional interaction is applied "
            "multiplicatively to min/max terminal-anchor FNP endpoints; positive "
            "frozen perturbative factors then propagate that common FNP envelope "
            "to all six Fig.2 flavors"
        ),
        "bspace_interaction_limitation": (
            "the full-b interaction envelope is observed only in the deterministic "
            "2x3 stratified nested design. It is defensible as a directional stress "
            "because FNP is common to all flavors and the Fig.2 perturbative factors "
            "are positive, but it is neither exhaustive nor probabilistically calibrated"
        ),
        "central_line": (
            "separately trained terminal lambda600 central endpoint and its exact "
            "paired expb2 transform; ensemble medians remain component diagnostics"
        ),
        "interval": (
            "empirical product band plus residual convergence/interaction envelope"
        ),
        "formal_confidence_level_assigned": False,
        "one_sigma_claimed": False,
        "probability_semantics": (
            "only the conditional experimental-replica marginal has its conventional "
            "replica interpretation; optimizer starts, checkpoint motion, and the 2x3 "
            "interaction stress have no calibrated probability law"
        ),
        "width_metrics_by_flavor": widths,
        "final_statistic_resampling": final_resampling,
        "final_statistic_resampling_artifacts": {
            "bootstrap_deviations": artifacts[
                "final_statistic_bootstrap_deviations"
            ],
            "split_differences": artifacts[
                "final_statistic_split_differences"
            ],
        },
        "prior_product_median_sampling_allowance_diagnostic_by_flavor": {
            flavor: float(
                stability["resampling_full_width_allowance_by_flavor"][flavor]
            ) for flavor in FLAVORS_K
        },
        "input_provenance": {
            "final_product_summary": str(FINAL_PRODUCT),
            "final_product_summary_sha256": sha256(FINAL_PRODUCT),
            "stability_summary": str(STABILITY),
            "stability_summary_sha256": sha256(STABILITY),
            "postfit_tail_transform_summary": str(POSTFIT),
            "postfit_tail_transform_summary_sha256": postfit_hash,
            "nested_interaction_summary": str(NESTED),
            "nested_interaction_summary_sha256": nested_hash,
            "reference_bspace": str(REFERENCE_B),
            "reference_bspace_sha256": sha256(REFERENCE_B),
        },
        "artifacts": artifacts,
        "artifact_sha256": hashes,
        **fixed_implementation_binding(),
        "frozen_sources_modified": False,
        "production_sources_modified": False,
    }
    atomic_json(TARGET / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
