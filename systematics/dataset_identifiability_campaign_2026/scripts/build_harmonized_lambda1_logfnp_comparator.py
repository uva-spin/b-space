#!/usr/bin/env python3
"""Build an isolated post-processing-harmonized lambda=1 comparator.

This script does not retrain any model and does not alter the campaign
champion registry.  It takes the pinned 24-start lambda=1 ensemble, combines
it with the preserved 50 conditional experimental-replica curves through the
same centered whole-curve log-FNP hierarchy used by the lambda=600 final-band
builder, and applies the same regularized finite-b transform to every 24x50
member.

Only post-processing is harmonized.  The lambda=1 starts and conditional
replicas were not produced with the strengthened lambda=600 continuation and
replica-training protocol, which is recorded as a hard interpretation caveat.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
SYSTEMATICS = BASE.parent
ROOT = SYSTEMATICS.parent
PINNED = (
    BASE / "summaries/champion_registry/"
    "empirical_reference_lambda1_b0p1_2p0_full24.json"
)
CURRENT = BASE / "summaries/champion_registry/current.json"
EXPECTED_CHAMPION_ID = "empirical_reference_lambda1_b0p1_2p0_full24"
EXACT_REFERENCE = BASE / "summaries/exact_baseline_fnp_median/fnp_median.csv"
REPLICA_ROOT = (
    SYSTEMATICS / "collins_factorization_validity/replicas/"
    "rowidfix_stageFT_all_qmax0p20_lam0p50_lambda3_50rep"
)
REPLICA_B = (
    REPLICA_ROOT / "tmd_bspace_bands_exactx_50rep/"
    "v22_tmd_replica_bspace_long.csv"
)
REPLICA_MANIFEST = (
    REPLICA_ROOT / "tmd_bspace_bands_exactx_50rep/"
    "v22_tmd_replica_band_manifest.json"
)
REPLICA_AUDIT = (
    REPLICA_ROOT / "audit_convergence_q95/lambda3_ensemble_q95_summary.json"
)
REFERENCE_B = (
    SYSTEMATICS / "collins_factorization_validity/plots/"
    "rowidfix_stageFT_all_qmax0p20_lam0p50_central_exactx/"
    "v22_scheme_tmd_bspace_long.csv"
)
TRANSFORMER = ROOT / "workflows/v23a/construction/construct_v23a_regularized_kspace_tmd_v2.py"
FROZEN_MANIFEST = BASE / "manifests/input_files.json"
PINNED_INPUT_MANIFEST = BASE / "manifests/harmonized_lambda1_inputs.json"
SOURCE_PRODUCTION = (
    SYSTEMATICS / "collins_factorization_validity/outputs/"
    "rowidfix_stageFT_all_qmax0p20_lam0p50_central_s303"
)
W_GRID = (
    ROOT / "outputs/v23a_tevatron_plus_lhcb7_fidacc_merged_cache/"
    "backend_cache/wpert_v23a_tevatron_plus_lhcb7_fidacc_b160.csv"
)
TARGET = BASE / "summaries/harmonized_lambda1_logfnp_24x50_comparator"
FLAVORS = ("u", "d")
EPS = 1.0e-30


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_pinned_input_manifest() -> dict:
    manifest = json.loads(PINNED_INPUT_MANIFEST.read_text())
    files = manifest.get("files", {})
    if (manifest.get("status") != "pinned_read_only_input_manifest"
            or manifest.get("champion_id") != EXPECTED_CHAMPION_ID
            or int(manifest.get("file_count", -1)) != len(files)
            or len(files) != 55
            or manifest.get("production_sources_modified", True)):
        raise RuntimeError("harmonized lambda=1 input manifest is invalid")
    for path_text, expected in files.items():
        path = Path(path_text)
        if (not path.is_file()
                or path.stat().st_size != int(expected["bytes"])
                or sha256(path) != expected["sha256"]):
            raise RuntimeError(f"pinned harmonized input changed: {path}")
    return manifest


def validate_pinned_incumbent() -> tuple[dict, list[str], dict[str, str]]:
    pinned = json.loads(PINNED.read_text())
    current = json.loads(CURRENT.read_text())
    for label, record in (("pinned", pinned), ("current", current)):
        if record.get("champion_id") != EXPECTED_CHAMPION_ID:
            raise RuntimeError(
                f"{label} champion is not the expected lambda=1 incumbent"
            )
    if int(pinned.get("start_count", -1)) != 24:
        raise RuntimeError("pinned lambda=1 incumbent does not declare 24 starts")
    if int(pinned.get("experimental_replica_count", -1)) != 50:
        raise RuntimeError("pinned lambda=1 incumbent does not declare 50 replicas")
    if int(pinned.get("combined_member_count_per_flavor", -1)) != 1200:
        raise RuntimeError("pinned lambda=1 incumbent does not declare 1200 members")
    tags = [str(item) for item in pinned.get("endpoint_tags", [])]
    if len(tags) != 24 or len(set(tags)) != 24:
        raise RuntimeError("pinned lambda=1 incumbent lacks 24 unique endpoints")

    observed_hashes: dict[str, str] = {}
    for key, expected in pinned.get("artifact_sha256", {}).items():
        artifact = Path(pinned["artifacts"][key])
        if not artifact.is_file():
            raise RuntimeError(f"pinned incumbent artifact is missing: {artifact}")
        observed = sha256(artifact)
        if observed != expected:
            raise RuntimeError(f"pinned incumbent artifact changed: {artifact}")
        observed_hashes[key] = observed
    if set(observed_hashes) != set(pinned.get("artifact_sha256", {})):
        raise RuntimeError("pinned incumbent artifact/hash mapping is incomplete")
    return pinned, tags, observed_hashes


def validate_registered_frozen_sources() -> dict[str, dict[str, object]]:
    registered = json.loads(FROZEN_MANIFEST.read_text())["files"]
    checked: dict[str, dict[str, object]] = {}
    for path in (REFERENCE_B, TRANSFORMER):
        key = str(path.resolve())
        if key not in registered:
            raise RuntimeError(f"required frozen source is not registered: {path}")
        expected = registered[key]
        observed_hash = sha256(path)
        observed_bytes = path.stat().st_size
        if observed_hash != expected["sha256"] or observed_bytes != expected["bytes"]:
            raise RuntimeError(f"registered frozen source changed: {path}")
        checked[key] = {
            "sha256": observed_hash,
            "bytes": observed_bytes,
            "registered_immutable_input": True,
        }
    return checked


def load_start_curves(tags: list[str]) -> tuple[np.ndarray, np.ndarray, list[int], list[dict]]:
    grid: np.ndarray | None = None
    curves: list[np.ndarray] = []
    seeds: list[int] = []
    provenance: list[dict] = []
    for tag in tags:
        run = BASE / "outputs" / tag
        status_path = run / "fit_status.json"
        curve_path = run / "fnp_grid.csv"
        if not status_path.is_file() or not curve_path.is_file():
            raise RuntimeError(f"lambda=1 endpoint is incomplete: {run}")
        status = json.loads(status_path.read_text())
        reference = status["regularization"]["fnp_reference_distance"]
        nonzero_regularizers = {
            name for name, spec in status["regularization"].items()
            if isinstance(spec, dict) and "lambda" in spec
            and not np.isclose(float(spec["lambda"]), 0.0)
        }
        complexity = status.get("model_complexity", {})
        profile = status.get("point_profile", {})
        if not np.isclose(float(reference["lambda"]), 1.0):
            raise RuntimeError(f"lambda mismatch in {run}")
        if not np.isclose(float(reference["b_min"]), 0.1):
            raise RuntimeError(f"reference bmin mismatch in {run}")
        if not np.isclose(float(reference["b_max"]), 2.0):
            raise RuntimeError(f"reference bmax mismatch in {run}")
        if Path(reference["target_csv"]).resolve() != EXACT_REFERENCE.resolve():
            raise RuntimeError(f"reference target mismatch in {run}")
        if (not status.get("convergence_gate_pass", False)
                or Path(status["source_production"]).resolve() !=
                    SOURCE_PRODUCTION.resolve()
                or Path(status["w_grid"]).resolve() != W_GRID.resolve()
                or nonzero_regularizers != {"fnp_reference_distance"}
                or not np.isclose(float(status["regularization"]
                    ["likelihood_weight"]["value"]), 1.0)
                or profile.get("enabled", True)
                or not np.isclose(float(profile.get("lambda_per_row", np.nan)), 0.0)
                or status.get("model_constraint", {}).get("kind") != "none"
                or int(complexity.get("np_width", -1)) != 48
                or int(complexity.get("np_cond_width", -1)) != 32
                or int(complexity.get("np_blocks", -1)) != 3
                or complexity.get("global_spline_nx") is not None
                or complexity.get("global_spline_nb") is not None
                or int(complexity.get("distill_accepted_steps", -1)) != 0
                or int(complexity.get("distill_prediction_steps", -1)) != 0):
            raise RuntimeError(f"lambda=1 objective/model provenance mismatch in {run}")
        if status.get("replica_seed") is not None:
            raise RuntimeError(f"start endpoint is labeled as a replica: {run}")
        if bool(status.get("production_state_modified")):
            raise RuntimeError(f"endpoint reports production mutation: {run}")
        seed = int(status["seed"])
        if seed in seeds:
            raise RuntimeError(f"duplicate start seed {seed}")

        frame = pd.read_csv(curve_path)
        frame = frame[np.isclose(frame["x"], 0.1)].sort_values("bT")
        b = frame["bT"].to_numpy(float)
        value = frame["F_NP"].to_numpy(float)
        if len(b) != 321 or len(np.unique(b)) != len(b):
            raise RuntimeError(f"invalid lambda=1 FNP grid in {run}")
        if not np.all(np.isfinite(b)) or not np.all(np.isfinite(value)):
            raise RuntimeError(f"non-finite lambda=1 FNP curve in {run}")
        if np.any(value <= 0.0):
            raise RuntimeError(f"non-positive lambda=1 FNP curve in {run}")
        if grid is None:
            grid = b
        elif grid.shape != b.shape or not np.allclose(
            grid, b, rtol=0.0, atol=1.0e-12
        ):
            raise RuntimeError(f"lambda=1 start grid differs in {run}")
        curves.append(value)
        seeds.append(seed)
        provenance.append({
            "tag": tag,
            "seed": seed,
            "fit_status_sha256": sha256(status_path),
            "fnp_grid_sha256": sha256(curve_path),
            "production_state_modified": False,
        })
    if grid is None or len(curves) != 24 or len(seeds) != 24:
        raise RuntimeError("lambda=1 start coverage is not exactly 24")
    if sorted(seeds) != list(range(303, 327)):
        raise RuntimeError("lambda=1 start seeds are not exactly 303--326")
    return grid, np.asarray(curves), seeds, provenance


def load_replica_curves(start_grid: np.ndarray) -> tuple[np.ndarray, list[int], dict]:
    manifest = json.loads(REPLICA_MANIFEST.read_text())
    audit = json.loads(REPLICA_AUDIT.read_text())
    expected_seeds = [int(value) for value in manifest.get("seeds", [])]
    if int(manifest.get("n_replicas", -1)) != 50:
        raise RuntimeError("frozen replica manifest does not declare 50 replicas")
    if len(expected_seeds) != 50 or len(set(expected_seeds)) != 50:
        raise RuntimeError("frozen replica manifest lacks 50 unique seeds")
    if int(audit.get("n_replicas", -1)) != 50:
        raise RuntimeError("frozen replica audit does not cover 50 replicas")
    if not bool(audit.get("V22_LAMBDA3_BSPACE_ENSEMBLE_Q95_PASS")):
        raise RuntimeError("frozen conditional replica audit did not pass")

    selected_chunks = []
    for chunk in pd.read_csv(
        REPLICA_B,
        usecols=["seed", "pid", "flavor", "x", "Q", "bT", "F_NP"],
        chunksize=100_000,
    ):
        keep = (
            np.isclose(chunk["x"], 0.1)
            & np.isclose(chunk["Q"], 10.0)
            & chunk["flavor"].astype(str).eq("u")
        )
        selected_chunks.append(chunk.loc[keep, ["seed", "bT", "F_NP"]])
    frame = pd.concat(selected_chunks, ignore_index=True)
    observed_seeds = sorted(int(value) for value in frame["seed"].unique())
    if observed_seeds != sorted(expected_seeds):
        raise RuntimeError("frozen replica data and manifest seeds differ")

    input_grid: np.ndarray | None = None
    values = []
    for seed in expected_seeds:
        group = frame[frame["seed"].eq(seed)].sort_values("bT")
        b = group["bT"].to_numpy(float)
        value = group["F_NP"].to_numpy(float)
        if len(b) != 321 or len(np.unique(b)) != len(b):
            raise RuntimeError(f"invalid frozen replica FNP grid for seed {seed}")
        if not np.all(np.isfinite(b)) or not np.all(np.isfinite(value)):
            raise RuntimeError(f"non-finite frozen replica FNP for seed {seed}")
        if np.any(value <= 0.0):
            raise RuntimeError(f"non-positive frozen replica FNP for seed {seed}")
        if input_grid is None:
            input_grid = b
        elif input_grid.shape != b.shape or not np.allclose(
            input_grid, b, rtol=0.0, atol=1.0e-12
        ):
            raise RuntimeError(f"frozen replica grids differ at seed {seed}")
        values.append(value)
    if input_grid is None:
        raise RuntimeError("frozen replica FNP grid is absent")
    if input_grid[0] > start_grid[0] or input_grid[-1] < start_grid[-1]:
        raise RuntimeError("frozen replica grid does not cover lambda=1 start grid")
    logs = np.log(np.asarray(values))
    harmonized_logs = np.asarray([
        np.interp(start_grid, input_grid, item) for item in logs
    ])
    grid_identical = bool(
        input_grid.shape == start_grid.shape
        and np.allclose(input_grid, start_grid, rtol=0.0, atol=1.0e-12)
    )
    grid_max_difference = (
        float(np.max(np.abs(input_grid - start_grid)))
        if input_grid.shape == start_grid.shape else None
    )
    metadata = {
        "manifest": str(REPLICA_MANIFEST),
        "manifest_sha256": sha256(REPLICA_MANIFEST),
        "audit": str(REPLICA_AUDIT),
        "audit_sha256": sha256(REPLICA_AUDIT),
        "source_grid_node_count": int(len(input_grid)),
        "source_grid_min": float(input_grid[0]),
        "source_grid_max": float(input_grid[-1]),
        "source_grid_identical_across_50_replicas": True,
        "source_grid_identical_to_lambda1_start_grid": grid_identical,
        "source_vs_start_grid_max_abs_difference": grid_max_difference,
        "harmonization": (
            "linear interpolation of log(F_NP) onto the common 321-node "
            "lambda=1 start grid; replica domain fully covers that grid"
        ),
    }
    return harmonized_logs, expected_seeds, metadata


def quantiles(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(np.quantile(values, q, axis=0) for q in (0.16, 0.50, 0.84))


def active_metrics(bands: pd.DataFrame) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for flavor in FLAVORS:
        group = bands[bands["flavor"].astype(str).eq(flavor)].sort_values("kT")
        group = group[group["kT"] <= 2.25]
        central = group["central"].to_numpy(float)
        if len(central) != 226 or not np.all(np.isfinite(central)):
            raise RuntimeError(f"unexpected transformed grid for {flavor}")
        active = central > 0.05 * np.max(central)
        if not np.any(active):
            raise RuntimeError(f"no active k-space region for {flavor}")
        width = (
            group["q84"].to_numpy(float) - group["q16"].to_numpy(float)
        ) / np.maximum(central, EPS)
        result[flavor] = {
            "max_active_relative_full_width": float(np.max(width[active])),
            "median_active_relative_full_width": float(np.median(width[active])),
            "active_kT_max_GeV": float(group["kT"].to_numpy(float)[active].max()),
            "kT0_central": float(central[0]),
            "kT0_relative_full_width": float(width[0]),
        }
    return result


def main() -> None:
    pinned_input_manifest = validate_pinned_input_manifest()
    pinned, tags, pinned_hashes = validate_pinned_incumbent()
    frozen_registered = validate_registered_frozen_sources()
    replica_hash_before = sha256(REPLICA_B)
    replica_bytes_before = REPLICA_B.stat().st_size

    b_grid, starts, start_seeds, start_provenance = load_start_curves(tags)
    replica_logs, replica_seeds, replica_grid_metadata = load_replica_curves(b_grid)
    if starts.shape != (24, 321) or replica_logs.shape != (50, 321):
        raise RuntimeError("input coverage does not form a 24x50 hierarchy")

    start_logs = np.log(starts)
    log_center = np.median(start_logs, axis=0)
    start_residuals = start_logs - log_center
    replica_median = np.median(replica_logs, axis=0)
    replica_residuals = replica_logs - replica_median
    combined_logs = (
        log_center[None, None, :]
        + start_residuals[:, None, :]
        + replica_residuals[None, :, :]
    ).reshape(1200, 321)
    if not np.all(np.isfinite(combined_logs)):
        raise RuntimeError("non-finite value in harmonized log-FNP hierarchy")
    combined_fnp = np.exp(combined_logs)
    if np.any(combined_fnp <= 0.0) or not np.all(np.isfinite(combined_fnp)):
        raise RuntimeError("harmonized hierarchy does not preserve FNP positivity")

    TARGET.mkdir(parents=True, exist_ok=True)
    q16, median, q84 = quantiles(combined_fnp)
    pd.DataFrame({
        "x": 0.1,
        "bT": b_grid,
        "q16": q16,
        "central": median,
        "q84": q84,
    }).to_csv(TARGET / "fnp_combined_bands.csv", index=False)

    reference = pd.read_csv(REFERENCE_B)
    reference = reference[
        np.isclose(reference["x"], 0.1)
        & np.isclose(reference["Q"], 10.0)
        & reference["flavor"].astype(str).isin(FLAVORS)
    ].copy()
    if reference.groupby("flavor").size().to_dict() != {"d": 321, "u": 321}:
        raise RuntimeError("frozen perturbative reference lacks the exact u/d grid")

    member_keys = [
        (start_seed, replica_seed)
        for start_seed in start_seeds
        for replica_seed in replica_seeds
    ]
    if len(member_keys) != 1200 or len(set(member_keys)) != 1200:
        raise RuntimeError("combined member keys are not exactly 24x50")

    b_band_rows = []
    transform_rows = []
    for flavor in FLAVORS:
        group = reference[reference["flavor"].astype(str).eq(flavor)].sort_values("bT")
        b_reference = group["bT"].to_numpy(float)
        perturbative = group["ftilde_no_np"].to_numpy(float)
        fnp_reference = np.exp(np.asarray([
            np.interp(b_reference, b_grid, item) for item in combined_logs
        ]))
        tmd = fnp_reference * perturbative[None, :]
        bq16, bmed, bq84 = quantiles(tmd)
        b_band_rows.append(pd.DataFrame({
            "flavor": flavor,
            "bT": b_reference,
            "q16": bq16,
            "central": bmed,
            "q84": bq84,
        }))
        pid = int(group["pid"].iloc[0])
        for index, ((start_seed, replica_seed), values) in enumerate(
            zip(member_keys, tmd)
        ):
            transform_rows.append(pd.DataFrame({
                "_replica_key": f"s{start_seed}|r{replica_seed}",
                "seed": start_seed,
                "pdf_member": replica_seed,
                "pid": pid,
                "flavor": flavor,
                "x": 0.1,
                "Q": 10.0,
                "bT": b_reference,
                "ftilde": values,
            }))
    b_bands = pd.concat(b_band_rows, ignore_index=True)
    b_bands.to_csv(TARGET / "bspace_combined_bands.csv", index=False)

    transform_input = pd.concat(transform_rows, ignore_index=True)
    transform = load_module("harmonized_lambda1_transform", TRANSFORMER)
    settings = argparse.Namespace(
        quantities=["ftilde"],
        tail_mode="expb2",
        tail_fit_bmin=None,
        eps=1.0e-300,
        b_transform_max=24.0,
        n_b_transform=6001,
        k_max=4.0,
        n_k=401,
        end_taper_start_fraction=0.92,
    )
    k_long, transform_metadata = transform.transform_curves(transform_input, settings)
    expected_k_rows = 1200 * len(FLAVORS) * 401
    if len(k_long) != expected_k_rows:
        raise RuntimeError(
            f"transform returned {len(k_long)} rows, expected {expected_k_rows}"
        )
    if k_long["_replica_key"].nunique() != 1200:
        raise RuntimeError("transformed ensemble lost 24x50 member coverage")
    if k_long.groupby(["_replica_key", "flavor"]).size().nunique() != 1:
        raise RuntimeError("transformed members do not share an identical k grid")
    if int(k_long.groupby(["_replica_key", "flavor"]).size().iloc[0]) != 401:
        raise RuntimeError("transformed member k-grid length is not 401")
    k_long.to_csv(TARGET / "kspace_combined_ensemble_long.csv", index=False)
    transformed_bands = transform.make_bands(k_long)
    k_bands = transformed_bands.rename(columns={"median": "central"})[
        ["flavor", "kT", "q16", "central", "q84"]
    ].sort_values(["flavor", "kT"])
    if k_bands.groupby("flavor").size().to_dict() != {"d": 401, "u": 401}:
        raise RuntimeError("final k-space bands lack the exact u/d 401-node grid")
    k_bands.to_csv(TARGET / "kspace_combined_bands.csv", index=False)

    metrics = active_metrics(k_bands)
    legacy_widths = pinned["combined_fig6_max_active_relative_full_width"]
    replica_hash_after = sha256(REPLICA_B)
    replica_bytes_after = REPLICA_B.stat().st_size
    if replica_hash_after != replica_hash_before or replica_bytes_after != replica_bytes_before:
        raise RuntimeError("frozen replica source changed during read-only construction")

    provenance = {
        "pinned_input_manifest": str(PINNED_INPUT_MANIFEST),
        "pinned_input_manifest_sha256": sha256(PINNED_INPUT_MANIFEST),
        "pinned_input_file_count": int(pinned_input_manifest["file_count"]),
        "pinned_incumbent": str(PINNED),
        "pinned_incumbent_sha256": sha256(PINNED),
        "pinned_artifact_sha256_revalidated": pinned_hashes,
        "start_endpoints": start_provenance,
        "conditional_replica_source": {
            "path": str(REPLICA_B),
            "sha256_before": replica_hash_before,
            "sha256_after": replica_hash_after,
            "bytes_before": replica_bytes_before,
            "bytes_after": replica_bytes_after,
            "read_only_invariant_pass": True,
        },
        "registered_frozen_sources": frozen_registered,
    }
    (TARGET / "input_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )

    summary = {
        "status": "complete_postprocessing_harmonized_lambda1_comparator_not_production",
        "champion_id": EXPECTED_CHAMPION_ID,
        "purpose": (
            "remove additive-TMD residual crossing and transform-order differences "
            "from the lambda=600 versus lambda=1 final-band comparison"
        ),
        "start_count": 24,
        "experimental_replica_count": 50,
        "combined_member_count_per_flavor": 1200,
        "flavors": list(FLAVORS),
        "x": 0.1,
        "Q_GeV": 10.0,
        "central_definition": (
            "pointwise median of the 24 lambda=1 curves in log(F_NP); this "
            "makes the start-only hierarchical members exactly the pinned 24 starts"
        ),
        "combination_rule": (
            "Cartesian convolution of centered whole-curve lambda=1 start and "
            "conditional experimental-replica residuals in log(F_NP)"
        ),
        "central_counted_once": True,
        "start_log_residual_pointwise_median_abs_max": float(
            np.max(np.abs(np.median(start_residuals, axis=0)))
        ),
        "experimental_log_residual_pointwise_median_abs_max": float(
            np.max(np.abs(np.median(replica_residuals, axis=0)))
        ),
        "positivity_pass": True,
        "grid_validation": {
            "lambda1_start_grid_identical_across_24": True,
            "lambda1_start_grid_node_count": int(len(b_grid)),
            "lambda1_start_grid_min": float(b_grid[0]),
            "lambda1_start_grid_max": float(b_grid[-1]),
            **replica_grid_metadata,
            "combined_grid_identical_across_1200_members": True,
        },
        "transform_settings": {
            "implementation": str(TRANSFORMER),
            "tail_mode": "expb2",
            "tail_fit_bmin": None,
            "b_transform_max": 24.0,
            "n_b_transform": 6001,
            "k_max": 4.0,
            "n_k": 401,
            "end_taper_start_fraction": 0.92,
        },
        "transform_metadata": transform_metadata,
        "metrics": metrics,
        "legacy_additive_incumbent_max_active_relative_full_width": {
            flavor: float(legacy_widths[flavor]) for flavor in FLAVORS
        },
        "training_protocol_harmonized": False,
        "hard_caveat": (
            "Only post-processing is harmonized. The 24 lambda=1 starts are "
            "short-horizon historical-polish endpoints, there is no separately "
            "trained and stationarity-certified lambda=1 central, and the 50 "
            "conditional replicas were not refit with the lambda=1 objective or "
            "the strengthened lambda=600 continuation protocol. This comparator "
            "must not be described as training- or objective-like-for-like."
        ),
        "registry_modified": False,
        "comparison_gate_modified": False,
        "frozen_sources_modified": False,
        "production_sources_modified": False,
        "artifacts": {
            "pinned_input_manifest": str(PINNED_INPUT_MANIFEST),
            "fnp_bands": str(TARGET / "fnp_combined_bands.csv"),
            "bspace_bands": str(TARGET / "bspace_combined_bands.csv"),
            "kspace_ensemble": str(TARGET / "kspace_combined_ensemble_long.csv"),
            "kspace_bands": str(TARGET / "kspace_combined_bands.csv"),
            "input_provenance": str(TARGET / "input_provenance.json"),
        },
    }
    (TARGET / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
