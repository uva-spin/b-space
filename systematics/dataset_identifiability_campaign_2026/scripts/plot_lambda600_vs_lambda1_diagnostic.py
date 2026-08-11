#!/usr/bin/env python3
"""Render an isolated, provenance-checked lambda=600 versus lambda=1 diagnostic.

This script is deliberately downstream of the complete 24-start x 50-replica
lambda=600 construction and its stability audit.  It neither authorizes a new
fit nor modifies the candidate figures, the champion registry, or any frozen
production input.

The lambda=1 curves use the post-processing-harmonized 24x50 comparator.  Its
training protocol is *not* like-for-like with lambda=600, so it is diagnostic
only; the immutable legacy lambda=1 Fig. 6 widths remain the promotion gate.

The harmonized comparator stores b-space TMD bands only for u,d at Q=10 GeV.
For the Fig. 2 comparison at Q=7.5 GeV, the common F_NP quantiles on the frozen
reference b grid are recovered independently from the Q=10 u and d tables by
division by their positive perturbative factors.  The two reconstructions must
agree to CSV roundoff.  Those common quantiles are then multiplied by the same
frozen Q=7.5 perturbative factors for all six flavors.  Positive deterministic
scaling commutes exactly with pointwise quantiles, so this is the same Fig. 2
band that a direct rebuild from the pinned log-F_NP ensemble would produce.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from nested_interaction_and_final_envelope_validation import (
    final_promotion_gate,
    validated_final_directional_envelope,
)


BASE = Path(__file__).resolve().parents[1]
SYSTEMATICS = BASE.parent
EXPECTED_CHAMPION_ID = "empirical_reference_lambda1_b0p1_2p0_full24"
DEFAULT_CANDIDATE = BASE / "summaries/final_combined_tmd_ensemble"
DEFAULT_CANDIDATE_AUDIT = (
    BASE / "summaries/final_combined_ensemble_stability/summary.json"
)
DEFAULT_POSTFIT_TAIL_AUDIT = (
    BASE / "summaries/lambda600_postfit_tail_transform_audit/summary.json"
)
DEFAULT_NESTED_INTERACTION = (
    BASE / "summaries/lambda600_nested_start_replica_interaction/summary.json"
)
DEFAULT_FINAL_ENVELOPE = (
    BASE / "summaries/lambda600_final_directional_envelope/summary.json"
)
DEFAULT_INCUMBENT = BASE / "summaries/harmonized_lambda1_logfnp_24x50_comparator"
DEFAULT_PINNED_RECORD = (
    BASE / "summaries/champion_registry/"
    "empirical_reference_lambda1_b0p1_2p0_full24.json"
)
DEFAULT_PIN_MANIFEST = BASE / "manifests/harmonized_lambda1_inputs.json"
DEFAULT_REFERENCE = (
    SYSTEMATICS / "collins_factorization_validity/plots/"
    "rowidfix_stageFT_all_qmax0p20_lam0p50_central_exactx/"
    "v22_scheme_tmd_bspace_long.csv"
)
DEFAULT_TARGET = BASE / "summaries/lambda600_vs_lambda1_diagnostic"

FLAVORS_B = ("u", "d", "s", "ubar", "dbar", "sbar")
FLAVORS_K = ("u", "d")
FLAVOR_LABELS = {
    "u": r"$u$ quark",
    "d": r"$d$ quark",
    "s": r"$s$ quark",
    "ubar": r"$\bar u$ quark",
    "dbar": r"$\bar d$ quark",
    "sbar": r"$\bar s$ quark",
    "F_NP": r"$F_{\rm NP}$",
}
CANDIDATE_COLOR = "#D55E00"
INCUMBENT_COLOR = "#0072B2"
EPS = 1.0e-30


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def explicit_bool(value: object, label: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise RuntimeError(f"{label} is not an explicit boolean")


def exact_number(value: object, expected: float, label: str) -> None:
    try:
        observed = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} is not numeric") from exc
    if not np.isfinite(observed) or not np.isclose(
        observed, expected, rtol=0.0, atol=1.0e-12
    ):
        raise RuntimeError(f"{label}={observed!r}, expected {expected!r}")


def read_json(path: Path, label: str) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"{label} is missing: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} is not a JSON object: {path}")
    return value


def atomic_write_json(path: Path, payload: dict) -> None:
    """Publish the terminal manifest only after bytes reach stable storage."""
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def validate_output_isolation(target: Path, protected: Iterable[Path]) -> None:
    resolved = target.resolve()
    for item in protected:
        protected_path = item.resolve()
        if resolved == protected_path or protected_path in resolved.parents:
            raise RuntimeError(
                f"diagnostic target overlaps a protected input directory: {target}"
            )
    registry = (BASE / "summaries/champion_registry").resolve()
    if resolved == registry or registry in resolved.parents:
        raise RuntimeError("diagnostic output may not be written into champion_registry")


def validate_pin_manifest(path: Path) -> tuple[dict, dict[str, str]]:
    manifest = read_json(path, "lambda=1 pinned-input manifest")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise RuntimeError("pinned-input manifest has no file mapping")
    if (
        manifest.get("status") != "pinned_read_only_input_manifest"
        or manifest.get("champion_id") != EXPECTED_CHAMPION_ID
        or int(manifest.get("file_count", -1)) != 55
        or len(files) != 55
        or explicit_bool(
            manifest.get("production_sources_modified"),
            "pin manifest production_sources_modified",
        )
    ):
        raise RuntimeError("lambda=1 pinned-input manifest metadata is invalid")
    observed: dict[str, str] = {}
    for path_text, expected in files.items():
        source = Path(path_text)
        if not source.is_file():
            raise RuntimeError(f"pinned lambda=1 input is missing: {source}")
        if source.stat().st_size != int(expected["bytes"]):
            raise RuntimeError(f"pinned lambda=1 input byte count changed: {source}")
        digest = sha256(source)
        if digest != str(expected["sha256"]):
            raise RuntimeError(f"pinned lambda=1 input hash changed: {source}")
        observed[str(source.resolve())] = digest
    return manifest, observed


def validate_pinned_record(
    path: Path, manifest: dict
) -> tuple[dict, dict[str, str]]:
    record = read_json(path, "immutable lambda=1 champion record")
    files = {str(Path(item).resolve()) for item in manifest["files"]}
    if str(path.resolve()) not in files:
        raise RuntimeError("immutable champion record is not covered by the pin manifest")
    if (
        record.get("champion_id") != EXPECTED_CHAMPION_ID
        or int(record.get("start_count", -1)) != 24
        or int(record.get("experimental_replica_count", -1)) != 50
        or int(record.get("combined_member_count_per_flavor", -1)) != 1200
        or explicit_bool(
            record.get("production_sources_modified"),
            "pinned record production_sources_modified",
        )
    ):
        raise RuntimeError("immutable lambda=1 champion record is invalid")
    widths = record.get("combined_fig6_max_active_relative_full_width", {})
    if set(widths) != set(FLAVORS_K):
        raise RuntimeError("immutable lambda=1 record lacks exact u,d gating widths")
    for flavor in FLAVORS_K:
        value = float(widths[flavor])
        if not np.isfinite(value) or value <= 0.0:
            raise RuntimeError(f"invalid immutable {flavor} gating width")
    artifacts = record.get("artifacts", {})
    expected_hashes = record.get("artifact_sha256", {})
    if not artifacts or set(artifacts) != set(expected_hashes):
        raise RuntimeError("immutable champion artifact/hash mapping is incomplete")
    observed: dict[str, str] = {}
    for key, expected in expected_hashes.items():
        artifact = Path(artifacts[key])
        if not artifact.is_file():
            raise RuntimeError(f"immutable champion artifact is missing: {artifact}")
        digest = sha256(artifact)
        if digest != expected:
            raise RuntimeError(f"immutable champion artifact changed: {artifact}")
        observed[key] = digest
    return record, observed


def validate_harmonized_metadata(
    root: Path,
    pin_manifest_path: Path,
    pinned_record_path: Path,
    pin_manifest_hash: str,
    pinned_record_hash: str,
    pinned_artifact_hashes: dict[str, str],
) -> tuple[dict, dict]:
    summary_path = root / "summary.json"
    provenance_path = root / "input_provenance.json"
    summary = read_json(summary_path, "harmonized lambda=1 summary")
    provenance = read_json(provenance_path, "harmonized lambda=1 provenance")
    if (
        summary.get("status")
        != "complete_postprocessing_harmonized_lambda1_comparator_not_production"
        or summary.get("champion_id") != EXPECTED_CHAMPION_ID
        or int(summary.get("start_count", -1)) != 24
        or int(summary.get("experimental_replica_count", -1)) != 50
        or int(summary.get("combined_member_count_per_flavor", -1)) != 1200
        or explicit_bool(
            summary.get("training_protocol_harmonized"),
            "harmonized training_protocol_harmonized",
        )
        or explicit_bool(summary.get("registry_modified"), "harmonized registry_modified")
        or explicit_bool(
            summary.get("comparison_gate_modified"),
            "harmonized comparison_gate_modified",
        )
        or explicit_bool(
            summary.get("frozen_sources_modified"),
            "harmonized frozen_sources_modified",
        )
        or explicit_bool(
            summary.get("production_sources_modified"),
            "harmonized production_sources_modified",
        )
    ):
        raise RuntimeError("post-processing-harmonized lambda=1 metadata is invalid")
    exact_number(summary.get("x"), 0.1, "harmonized x")
    exact_number(summary.get("Q_GeV"), 10.0, "harmonized stored b/k Q")
    if set(map(str, summary.get("flavors", []))) != set(FLAVORS_K):
        raise RuntimeError("harmonized comparator does not declare exact u,d coverage")
    transform = summary.get("transform_settings", {})
    required_transform = {
        "tail_mode": "expb2",
        "b_transform_max": 24.0,
        "n_b_transform": 6001,
        "k_max": 4.0,
        "n_k": 401,
        "end_taper_start_fraction": 0.92,
    }
    for key, expected in required_transform.items():
        if isinstance(expected, str):
            if transform.get(key) != expected:
                raise RuntimeError(f"harmonized transform setting {key} changed")
        else:
            exact_number(transform.get(key), expected, f"harmonized transform {key}")

    expected_artifacts = {
        "pinned_input_manifest": pin_manifest_path,
        "fnp_bands": root / "fnp_combined_bands.csv",
        "bspace_bands": root / "bspace_combined_bands.csv",
        "kspace_bands": root / "kspace_combined_bands.csv",
        "input_provenance": provenance_path,
    }
    declared = summary.get("artifacts", {})
    for key, expected in expected_artifacts.items():
        if Path(declared.get(key, "")).resolve() != expected.resolve():
            raise RuntimeError(f"harmonized artifact path mismatch for {key}")
    if (
        Path(provenance.get("pinned_input_manifest", "")).resolve()
        != pin_manifest_path.resolve()
        or provenance.get("pinned_input_manifest_sha256") != pin_manifest_hash
        or int(provenance.get("pinned_input_file_count", -1)) != 55
        or Path(provenance.get("pinned_incumbent", "")).resolve()
        != pinned_record_path.resolve()
        or provenance.get("pinned_incumbent_sha256") != pinned_record_hash
        or provenance.get("pinned_artifact_sha256_revalidated")
        != pinned_artifact_hashes
    ):
        raise RuntimeError("harmonized provenance is not bound to the pinned inputs")
    return summary, provenance


def validate_candidate_metadata(root: Path, audit_path: Path) -> tuple[dict, dict]:
    summary = read_json(root / "summary.json", "lambda=600 combined summary")
    audit = read_json(audit_path, "lambda=600 combined stability audit")
    if summary.get("status") != "complete":
        raise RuntimeError("lambda=600 24x50 combined ensemble is incomplete")
    exact_number(summary.get("selected_strength"), 600.0, "candidate strength")
    exact_number(summary.get("selected_bmax"), 4.0, "candidate bmax")
    exact_number(
        summary.get("fit_quality_barrier_strength"),
        100.0,
        "candidate fit-quality barrier strength",
    )
    if int(summary.get("fit_quality_barrier_power", -1)) != 2:
        raise RuntimeError("candidate fit-quality barrier power is not 2")
    if (
        int(summary.get("start_count", -1)) != 24
        or int(summary.get("experimental_replica_count", -1)) != 50
        or int(summary.get("combined_member_count", -1)) != 1200
        or not explicit_bool(summary.get("state_chain_gate_pass"), "candidate state chain")
        or explicit_bool(
            summary.get("production_sources_modified"),
            "candidate production_sources_modified",
        )
    ):
        raise RuntimeError("lambda=600 combined metadata lacks exact 24x50 evidence")
    stationarity = explicit_bool(
        summary.get("candidate_stationarity_gate_pass"),
        "candidate summary stationarity",
    )
    if explicit_bool(summary.get("diagnostic_only"), "candidate summary diagnostic_only") == stationarity:
        raise RuntimeError("candidate summary diagnostic/stationarity semantics disagree")

    if audit.get("status") != "complete":
        raise RuntimeError("lambda=600 final stability audit is incomplete")
    required_true = (
        "diagnostic_figure_gate_pass",
        "state_chain_gate_pass",
        "band_integrity_gate_pass",
        "coverage_gate_pass",
    )
    for key in required_true:
        if not explicit_bool(audit.get(key), f"candidate audit {key}"):
            raise RuntimeError(f"candidate audit gate failed: {key}")
    if (
        int(audit.get("start_count", -1)) != 24
        or int(audit.get("replica_count", -1)) != 50
        or explicit_bool(
            audit.get("production_sources_modified"),
            "candidate audit production_sources_modified",
        )
    ):
        raise RuntimeError("candidate audit lacks exact read-only 24x50 coverage")
    audit_stationarity = explicit_bool(
        audit.get("candidate_stationarity_gate_pass"),
        "candidate audit stationarity",
    )
    endpoint = explicit_bool(audit.get("endpoint_gate_pass"), "candidate endpoint gate")
    promotion = explicit_bool(audit.get("promotion_eligible"), "candidate promotion_eligible")
    diagnostic = explicit_bool(audit.get("diagnostic_only"), "candidate audit diagnostic_only")
    if (
        audit_stationarity != stationarity
        or endpoint != promotion
        or diagnostic == endpoint
        or (endpoint and not stationarity)
    ):
        raise RuntimeError("candidate summary/audit scientific-gate semantics disagree")
    return summary, audit


def normalize_quantiles(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    result = frame.copy()
    if "median" in result.columns and "central" not in result.columns:
        result = result.rename(columns={"median": "central"})
    required = {"q16", "central", "q84"}
    missing = required - set(result.columns)
    if missing:
        raise RuntimeError(f"{label} lacks columns {sorted(missing)}")
    values = result[["q16", "central", "q84"]].to_numpy(float)
    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"{label} contains non-finite quantiles")
    if np.any(values[:, 0] > values[:, 1]) or np.any(values[:, 1] > values[:, 2]):
        raise RuntimeError(f"{label} violates q16 <= central <= q84")
    return result


def normalize_directional_interval(
    frame: pd.DataFrame,
    label: str,
    *,
    require_central_containment: bool,
) -> pd.DataFrame:
    """Normalize a final envelope without hiding a failed containment gate.

    A promotable result must contain its trained central at every point.  For a
    rejected result, central containment is itself scientific evidence that the
    diagnostic comparison must be able to display.  Such a diagnostic still
    requires finite, ordered lower/upper endpoints; only the placement of the
    separately trained central relative to those endpoints is relaxed.
    """
    result = frame.rename(columns={
        "final_envelope_low": "q16",
        "trained_central": "central",
        "final_envelope_high": "q84",
    }).copy()
    required = {"q16", "central", "q84"}
    missing = required - set(result.columns)
    if missing:
        raise RuntimeError(f"{label} lacks columns {sorted(missing)}")
    values = result[["q16", "central", "q84"]].to_numpy(float)
    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"{label} contains non-finite interval values")
    if np.any(values[:, 0] > values[:, 2]):
        raise RuntimeError(f"{label} has an inverted directional interval")
    if require_central_containment and (
        np.any(values[:, 0] > values[:, 1])
        or np.any(values[:, 1] > values[:, 2])
    ):
        raise RuntimeError(
            f"{label} promotable interval does not contain its trained central"
        )
    return result


def validate_candidate_tables(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fnp = normalize_quantiles(pd.read_csv(root / "fnp_bands.csv"), "candidate FNP")
    bspace = normalize_quantiles(
        pd.read_csv(root / "bT_tmd_bands.csv"), "candidate b-space TMD"
    )
    kspace = normalize_quantiles(
        pd.read_csv(root / "kT_tmd_bands.csv"), "candidate k-space TMD"
    )
    if set(fnp.get("component", pd.Series(dtype=str)).astype(str)) != {
        "experimental",
        "nonuniqueness",
        "combined",
    }:
        raise RuntimeError("candidate FNP components are incomplete")
    if set(bspace.get("component", pd.Series(dtype=str)).astype(str)) != {
        "experimental",
        "nonuniqueness",
        "combined",
    }:
        raise RuntimeError("candidate b-space components are incomplete")
    if set(kspace.get("component", pd.Series(dtype=str)).astype(str)) != {
        "experimental",
        "nonuniqueness",
        "combined",
    }:
        raise RuntimeError("candidate k-space components are incomplete")
    if fnp.duplicated(["component", "x", "bT"]).any():
        raise RuntimeError("candidate FNP table has duplicate keys")
    if bspace.duplicated(["component", "Q", "flavor", "bT"]).any():
        raise RuntimeError("candidate b-space table has duplicate keys")
    kkeys = ["component", "Q", "flavor", "kT"]
    if "quantity" in kspace.columns:
        kkeys.insert(3, "quantity")
    if kspace.duplicated(kkeys).any():
        raise RuntimeError("candidate k-space table has duplicate keys")
    # The regularized finite-b transform is intentionally retained to kT=4
    # and can have a small numerical negative lobe outside the displayed
    # kT<=2.25 region.  Positivity is therefore required for FNP/b space, not
    # for every stored k-space tail node.
    for label, frame in (("FNP", fnp), ("b-space", bspace)):
        if np.any(frame[["q16", "central", "q84"]].to_numpy(float) <= 0.0):
            raise RuntimeError(f"candidate {label} band is not strictly positive")
    return fnp, bspace, kspace


def validate_final_envelope_tables(summary: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Normalize the jointly expanded final envelope for comparison plots."""
    artifacts = summary["artifacts"]
    require_central_containment = final_promotion_gate(summary)
    fnp = pd.read_csv(Path(artifacts["fnp_final_envelope"]))
    bspace = pd.read_csv(Path(artifacts["fig2_bspace_final_envelope"]))
    kspace = pd.read_csv(Path(artifacts["fig6_kspace_final_envelope"]))
    frames = []
    for label, frame, keys in (
        ("FNP", fnp, ["x", "bT"]),
        ("b-space", bspace, ["Q", "flavor", "bT"]),
        ("k-space", kspace, ["Q", "flavor", "kT"]),
    ):
        normalized = normalize_directional_interval(
            frame,
            f"candidate final {label} envelope",
            require_central_containment=require_central_containment,
        )
        normalized["component"] = "combined"
        if normalized.duplicated(keys).any():
            raise RuntimeError(f"candidate final {label} envelope has duplicate keys")
        frames.append(normalized)
    fnp, bspace, kspace = frames
    if len(fnp) != 321 or bspace.groupby("flavor").size().to_dict() != {
            flavor: 321 for flavor in FLAVORS_B} or kspace.groupby(
                "flavor").size().to_dict() != {"d": 401, "u": 401}:
        raise RuntimeError("candidate final directional-envelope coverage is incomplete")
    return fnp, bspace, kspace


def validate_harmonized_tables(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fnp = normalize_quantiles(
        pd.read_csv(root / "fnp_combined_bands.csv"), "harmonized FNP"
    )
    bspace = normalize_quantiles(
        pd.read_csv(root / "bspace_combined_bands.csv"), "harmonized b-space TMD"
    )
    kspace = normalize_quantiles(
        pd.read_csv(root / "kspace_combined_bands.csv"), "harmonized k-space TMD"
    )
    if len(fnp) != 321 or fnp.duplicated(["x", "bT"]).any():
        raise RuntimeError("harmonized FNP table is not the exact 321-node band")
    if (
        len(bspace) != 642
        or set(bspace["flavor"].astype(str)) != set(FLAVORS_K)
        or bspace.duplicated(["flavor", "bT"]).any()
    ):
        raise RuntimeError("harmonized b-space table lacks exact 2x321 coverage")
    if (
        len(kspace) != 802
        or set(kspace["flavor"].astype(str)) != set(FLAVORS_K)
        or kspace.duplicated(["flavor", "kT"]).any()
    ):
        raise RuntimeError("harmonized k-space table lacks exact 2x401 coverage")
    for label, frame in (("FNP", fnp), ("b-space", bspace)):
        if np.any(frame[["q16", "central", "q84"]].to_numpy(float) <= 0.0):
            raise RuntimeError(f"harmonized {label} band is not strictly positive")
    return fnp, bspace, kspace


def exact_grid(first: pd.DataFrame, second: pd.DataFrame, coordinate: str, label: str) -> None:
    a = first.sort_values(coordinate)[coordinate].to_numpy(float)
    b = second.sort_values(coordinate)[coordinate].to_numpy(float)
    if a.shape != b.shape or not np.allclose(a, b, rtol=0.0, atol=1.0e-12):
        raise RuntimeError(f"candidate and incumbent {label} grids differ")


def method_frame(
    frame: pd.DataFrame,
    method: str,
    method_label: str,
    coordinate: str,
    flavor: str,
    x: float,
    q: float | None,
    derivation: str,
) -> pd.DataFrame:
    result = frame.sort_values(coordinate)[
        [coordinate, "q16", "central", "q84"]
    ].copy()
    result.insert(0, "method", method)
    result.insert(1, "method_label", method_label)
    result.insert(2, "flavor", flavor)
    result.insert(3, "x", x)
    if q is not None:
        result.insert(4, "Q_GeV", q)
    result["relative_full_width"] = (
        result["q84"].to_numpy(float) - result["q16"].to_numpy(float)
    ) / np.maximum(np.abs(result["central"].to_numpy(float)), EPS)
    result["derivation"] = derivation
    return result


def annotate_masks(
    frame: pd.DataFrame, coordinate: str, all_domain_active: bool = False
) -> pd.DataFrame:
    result = frame.copy()
    result["own_active_mask"] = False
    result["comparison_union_active_mask"] = False
    for flavor, flavor_frame in result.groupby("flavor", sort=False):
        masks: dict[str, np.ndarray] = {}
        grids: np.ndarray | None = None
        indices: dict[str, pd.Index] = {}
        for method, group in flavor_frame.groupby("method", sort=False):
            group = group.sort_values(coordinate)
            grid = group[coordinate].to_numpy(float)
            if grids is None:
                grids = grid
            elif grid.shape != grids.shape or not np.allclose(
                grid, grids, rtol=0.0, atol=1.0e-12
            ):
                raise RuntimeError(f"method grids differ for {flavor}")
            central = group["central"].to_numpy(float)
            masks[str(method)] = (
                np.ones(len(group), dtype=bool)
                if all_domain_active
                else central > 0.05 * np.max(central)
            )
            indices[str(method)] = group.index
        if set(masks) != {"lambda600_candidate", "lambda1_harmonized"}:
            raise RuntimeError(f"comparison lacks both methods for {flavor}")
        union = masks["lambda600_candidate"] | masks["lambda1_harmonized"]
        for method, mask in masks.items():
            result.loc[indices[method], "own_active_mask"] = mask
            result.loc[indices[method], "comparison_union_active_mask"] = union
    return result


def width_metrics(frame: pd.DataFrame, coordinate: str, domain: str) -> list[dict]:
    rows: list[dict] = []
    for (flavor, method), group in frame.groupby(["flavor", "method"], sort=False):
        group = group.sort_values(coordinate)
        width = group["relative_full_width"].to_numpy(float)
        own = group["own_active_mask"].to_numpy(bool)
        union = group["comparison_union_active_mask"].to_numpy(bool)
        grid = group[coordinate].to_numpy(float)
        if not np.any(own) or not np.any(union):
            raise RuntimeError(f"empty width mask for {domain}/{flavor}/{method}")
        union_indices = np.flatnonzero(union)
        union_max_local = int(np.argmax(width[union]))
        union_max_index = int(union_indices[union_max_local])
        rows.append(
            {
                "domain": domain,
                "flavor": str(flavor),
                "method": str(method),
                "coordinate": coordinate,
                "coordinate_min": float(grid.min()),
                "coordinate_max": float(grid.max()),
                "grid_point_count": int(len(grid)),
                "own_active_point_count": int(np.sum(own)),
                "union_active_point_count": int(np.sum(union)),
                "max_relative_full_width_all": float(np.max(width)),
                "median_relative_full_width_all": float(np.median(width)),
                "max_relative_full_width_own_active": float(np.max(width[own])),
                "median_relative_full_width_own_active": float(np.median(width[own])),
                "max_relative_full_width_union_active": float(np.max(width[union])),
                "median_relative_full_width_union_active": float(np.median(width[union])),
                "coordinate_of_union_active_max_width": float(grid[union_max_index]),
            }
        )
    return rows


def reconstruct_incumbent_fig2(
    incumbent_bspace: pd.DataFrame, reference: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    q10 = reference[
        np.isclose(reference["x"], 0.1)
        & np.isclose(reference["Q"], 10.0)
        & reference["flavor"].astype(str).isin(FLAVORS_K)
        & (reference["bT"] <= 4.0 + 1.0e-12)
    ].copy()
    recovered: dict[str, np.ndarray] = {}
    common_grid: np.ndarray | None = None
    for flavor in FLAVORS_K:
        band = incumbent_bspace[
            incumbent_bspace["flavor"].astype(str).eq(flavor)
            & (incumbent_bspace["bT"] <= 4.0 + 1.0e-12)
        ].sort_values("bT")
        ref = q10[q10["flavor"].astype(str).eq(flavor)].sort_values("bT")
        exact_grid(band, ref, "bT", f"lambda=1 Q=10 {flavor} reconstruction")
        factor = ref["ftilde_no_np"].to_numpy(float)
        if np.any(factor <= 0.0) or not np.all(np.isfinite(factor)):
            raise RuntimeError("frozen Q=10 perturbative TMD is not strictly positive")
        grid = band["bT"].to_numpy(float)
        if common_grid is None:
            common_grid = grid
        recovered[flavor] = band[["q16", "central", "q84"]].to_numpy(float) / factor[:, None]
    if common_grid is None:
        raise RuntimeError("lambda=1 b-space reconstruction grid is absent")
    scale = np.maximum(np.abs(recovered["u"]), EPS)
    relative_disagreement = np.abs(recovered["u"] - recovered["d"]) / scale
    max_disagreement = float(np.max(relative_disagreement))
    if max_disagreement > 2.0e-12:
        raise RuntimeError(
            "u/d recovery of the harmonized FNP quantiles disagrees beyond CSV roundoff: "
            f"{max_disagreement:.3e}"
        )
    common_fnp = recovered["u"]

    q75 = reference[
        np.isclose(reference["x"], 0.1)
        & np.isclose(reference["Q"], 7.5)
        & reference["flavor"].astype(str).isin(FLAVORS_B)
        & (reference["bT"] <= 4.0 + 1.0e-12)
    ].copy()
    rows: list[pd.DataFrame] = []
    for flavor in FLAVORS_B:
        ref = q75[q75["flavor"].astype(str).eq(flavor)].sort_values("bT")
        grid = ref["bT"].to_numpy(float)
        if grid.shape != common_grid.shape or not np.allclose(
            grid, common_grid, rtol=0.0, atol=1.0e-12
        ):
            raise RuntimeError(f"frozen Q=7.5 grid differs for {flavor}")
        factor = ref["ftilde_no_np"].to_numpy(float)
        if np.any(factor <= 0.0) or not np.all(np.isfinite(factor)):
            raise RuntimeError(f"frozen Q=7.5 perturbative factor is invalid for {flavor}")
        values = common_fnp * factor[:, None]
        rows.append(
            pd.DataFrame(
                {
                    "flavor": flavor,
                    "bT": common_grid,
                    "q16": values[:, 0],
                    "central": values[:, 1],
                    "q84": values[:, 2],
                }
            )
        )
    metadata = {
        "status": "exact_positive_scaling_reconstruction_from_pinned_harmonized_logFNP",
        "stored_lambda1_bspace_Q_GeV": 10.0,
        "reconstructed_fig2_Q_GeV": 7.5,
        "flavors": list(FLAVORS_B),
        "u_d_recovered_fnp_max_relative_disagreement": max_disagreement,
        "roundoff_gate": 2.0e-12,
        "mathematical_basis": (
            "Every ensemble member is multiplied by the same strictly positive "
            "flavor/Q-dependent frozen ftilde_no_np factor, so q16, median, and "
            "q84 scale by that factor exactly."
        ),
        "q10_table_was_not_relabelled_as_q7p5": True,
    }
    return pd.concat(rows, ignore_index=True), metadata


def comparison_pair(
    candidate: pd.DataFrame,
    incumbent: pd.DataFrame,
    coordinate: str,
    flavor: str,
    x: float,
    q: float | None,
    candidate_label: str,
    candidate_derivation: str,
    incumbent_derivation: str,
) -> pd.DataFrame:
    exact_grid(candidate, incumbent, coordinate, flavor)
    return pd.concat(
        [
            method_frame(
                candidate,
                "lambda600_candidate",
                candidate_label,
                coordinate,
                flavor,
                x,
                q,
                candidate_derivation,
            ),
            method_frame(
                incumbent,
                "lambda1_harmonized",
                r"$\lambda=1$ incumbent (post-processing matched)",
                coordinate,
                flavor,
                x,
                q,
                incumbent_derivation,
            ),
        ],
        ignore_index=True,
    )


def plot_pair(
    curve_ax,
    width_ax,
    frame: pd.DataFrame,
    coordinate: str,
    title: str,
    ylabel: str,
) -> None:
    styles = {
        "lambda600_candidate": (CANDIDATE_COLOR, "-", 2.0, 0.20),
        "lambda1_harmonized": (INCUMBENT_COLOR, "--", 1.8, 0.15),
    }
    for method in ("lambda1_harmonized", "lambda600_candidate"):
        group = frame[frame["method"].eq(method)].sort_values(coordinate)
        color, linestyle, linewidth, alpha = styles[method]
        x = group[coordinate].to_numpy(float)
        curve_ax.fill_between(
            x,
            group["q16"].to_numpy(float),
            group["q84"].to_numpy(float),
            color=color,
            alpha=alpha,
            linewidth=0,
        )
        curve_ax.plot(
            x,
            group["central"].to_numpy(float),
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
        )
        width_ax.plot(
            x,
            100.0 * group["relative_full_width"].to_numpy(float),
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
        )
    curve_ax.set_title(title, fontsize=10.5)
    curve_ax.set_ylabel(ylabel)
    width_ax.set_ylabel("full width [%]")
    for axis in (curve_ax, width_ax):
        axis.grid(alpha=0.16)
        axis.set_xlim(float(frame[coordinate].min()), float(frame[coordinate].max()))


def legend_handles(candidate_label: str) -> list:
    return [
        Line2D(
            [0], [0], color=CANDIDATE_COLOR, lw=2.0, linestyle="-", label=candidate_label
        ),
        Patch(
            facecolor=CANDIDATE_COLOR,
            alpha=0.20,
            edgecolor="none",
            label=(r"$\lambda=600$ empirical product + residual "
                   "convergence/interaction envelope"),
        ),
        Line2D(
            [0],
            [0],
            color=INCUMBENT_COLOR,
            lw=1.8,
            linestyle="--",
            label=r"$\lambda=1$ combined ensemble median",
        ),
        Patch(
            facecolor=INCUMBENT_COLOR,
            alpha=0.15,
            edgecolor="none",
            label=r"$\lambda=1$ q16--q84",
        ),
    ]


def save_figure(fig, target: Path, stem: str) -> tuple[Path, Path]:
    png = target / f"{stem}.png"
    pdf = target / f"{stem}.pdf"
    fig.savefig(
        png,
        dpi=240,
        metadata={"Software": "plot_lambda600_vs_lambda1_diagnostic.py"},
    )
    fig.savefig(
        pdf,
        metadata={
            "Creator": "plot_lambda600_vs_lambda1_diagnostic.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)
    return png, pdf


def render_fnp(
    frame: pd.DataFrame, target: Path, candidate_label: str, diagnostic: bool
) -> tuple[Path, Path]:
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.7, 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
        constrained_layout=False,
    )
    plot_pair(
        axes[0], axes[1], frame, "bT", r"$x=0.1$", r"$F_{\rm NP}(x,b_T)$"
    )
    axes[1].set_xlabel(r"$b_T\ [\mathrm{GeV}^{-1}]$")
    title = r"$F_{\rm NP}$: $\lambda=600$ candidate versus $\lambda=1$ incumbent"
    if diagnostic:
        title += "\nDIAGNOSTIC ONLY — candidate final gate failed"
    fig.suptitle(title, color="#9b2226" if diagnostic else "black", fontsize=12)
    fig.legend(
        handles=legend_handles(candidate_label),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=2,
        frameon=False,
        fontsize=8.5,
    )
    fig.text(
        0.5,
        0.003,
        r"$\lambda=1$: post-processing matched only; training is not identical and legacy Fig. 6 widths remain gating.",
        ha="center",
        fontsize=7.5,
    )
    fig.tight_layout(rect=(0.0, 0.045, 1.0, 0.84))
    return save_figure(fig, target, "lambda600_vs_lambda1_fnp_x0p1")


def render_fig2(
    frame: pd.DataFrame, target: Path, candidate_label: str, diagnostic: bool
) -> tuple[Path, Path]:
    fig, axes = plt.subplots(3, 4, figsize=(13.0, 9.5), constrained_layout=False)
    for index, flavor in enumerate(FLAVORS_B):
        row, pair = divmod(index, 2)
        curve_ax, width_ax = axes[row, 2 * pair], axes[row, 2 * pair + 1]
        plot_pair(
            curve_ax,
            width_ax,
            frame[frame["flavor"].eq(flavor)],
            "bT",
            FLAVOR_LABELS[flavor],
            r"$\widetilde f_1^q$",
        )
        curve_ax.set_xlabel(r"$b_T\ [\mathrm{GeV}^{-1}]$")
        width_ax.set_xlabel(r"$b_T\ [\mathrm{GeV}^{-1}]$")
    title = r"Fig. 2-like TMD comparison: $x=0.1$, $Q=7.5\ \mathrm{GeV}$"
    if diagnostic:
        title += "\nDIAGNOSTIC ONLY — candidate final gate failed"
    fig.suptitle(title, color="#9b2226" if diagnostic else "black", fontsize=12)
    fig.legend(
        handles=legend_handles(candidate_label),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=4,
        frameon=False,
        fontsize=8.3,
    )
    fig.text(
        0.5,
        0.003,
        "The lambda=1 Q=7.5 bands are reconstructed by exact positive scaling of the pinned harmonized log-FNP ensemble; the stored Q=10 table is not relabelled.",
        ha="center",
        fontsize=7.2,
    )
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.86))
    return save_figure(fig, target, "lambda600_vs_lambda1_fig2_bT_Q7p5")


def render_fig6(
    frame: pd.DataFrame, target: Path, candidate_label: str, diagnostic: bool
) -> tuple[Path, Path]:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), constrained_layout=False)
    for row, flavor in enumerate(FLAVORS_K):
        plot_pair(
            axes[row, 0],
            axes[row, 1],
            frame[frame["flavor"].eq(flavor)],
            "kT",
            FLAVOR_LABELS[flavor],
            r"$f_1^q(x,k_T;Q)$",
        )
        axes[row, 0].set_ylim(bottom=0.0)
        axes[row, 0].set_xlabel(r"$k_T\ [\mathrm{GeV}]$")
        axes[row, 1].set_xlabel(r"$k_T\ [\mathrm{GeV}]$")
    title = r"Fig. 6 comparison: $x=0.1$, $Q=10\ \mathrm{GeV}$"
    if diagnostic:
        title += "\nDIAGNOSTIC ONLY — candidate final gate failed"
    fig.suptitle(title, color="#9b2226" if diagnostic else "black", fontsize=12)
    fig.legend(
        handles=legend_handles(candidate_label),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=2,
        frameon=False,
        fontsize=8.5,
    )
    fig.text(
        0.5,
        0.003,
        r"Harmonized $\lambda=1$ is a propagation diagnostic, not training-like-for-like; immutable legacy $\lambda=1$ widths remain the promotion gate.",
        ha="center",
        fontsize=7.3,
    )
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.84))
    return save_figure(fig, target, "lambda600_vs_lambda1_fig6_kT_Q10")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, float_format="%.17g", lineterminator="\n")


def nested_metrics(rows: list[dict]) -> dict:
    result: dict = {}
    for row in rows:
        domain = str(row["domain"])
        flavor = str(row["flavor"])
        method = str(row["method"])
        payload = {key: value for key, value in row.items() if key not in {"domain", "flavor", "method"}}
        result.setdefault(domain, {}).setdefault(flavor, {})[method] = payload
    return result


def run(args: argparse.Namespace) -> dict:
    candidate_root = Path(args.candidate_dir)
    candidate_audit_path = Path(args.candidate_audit)
    postfit_tail_audit_path = Path(args.postfit_tail_audit)
    nested_interaction_path = Path(args.nested_interaction)
    final_envelope_path = Path(args.final_envelope)
    incumbent_root = Path(args.incumbent_dir)
    pinned_record_path = Path(args.pinned_record)
    pin_manifest_path = Path(args.pin_manifest)
    reference_path = Path(args.reference_bspace)
    target = Path(args.target)
    validate_output_isolation(
        target,
        (
            candidate_root,
            incumbent_root,
        ),
    )
    for source_parent in (
        pinned_record_path.parent,
        pin_manifest_path.parent,
        reference_path.parent,
    ):
        if target.resolve() == source_parent.resolve():
            raise RuntimeError(
                f"diagnostic target is a protected input directory: {target}"
            )

    pin_manifest, _ = validate_pin_manifest(pin_manifest_path)
    pinned_record, pinned_artifact_hashes = validate_pinned_record(
        pinned_record_path, pin_manifest
    )
    if str(reference_path.resolve()) not in {
        str(Path(item).resolve()) for item in pin_manifest["files"]
    }:
        raise RuntimeError("frozen perturbative reference is not covered by the pin manifest")
    pin_manifest_hash = sha256(pin_manifest_path)
    pinned_record_hash = sha256(pinned_record_path)
    harmonized_summary, _ = validate_harmonized_metadata(
        incumbent_root,
        pin_manifest_path,
        pinned_record_path,
        pin_manifest_hash,
        pinned_record_hash,
        pinned_artifact_hashes,
    )
    candidate_summary, candidate_audit = validate_candidate_metadata(
        candidate_root, candidate_audit_path
    )
    final_envelope, final_envelope_hash = validated_final_directional_envelope(
        final_envelope_path,
        nested_interaction_path,
        postfit_tail_audit_path,
        candidate_audit_path,
    )
    candidate_fnp, candidate_b, candidate_k = validate_final_envelope_tables(
        final_envelope
    )
    incumbent_fnp, incumbent_b, incumbent_k = validate_harmonized_tables(incumbent_root)

    reference = pd.read_csv(
        reference_path, usecols=["x", "Q", "flavor", "bT", "ftilde_no_np"]
    )
    reference_numeric = reference[["x", "Q", "bT", "ftilde_no_np"]].to_numpy(float)
    if not np.all(np.isfinite(reference_numeric)):
        raise RuntimeError("frozen perturbative reference contains non-finite values")

    endpoint_pass = final_promotion_gate(final_envelope)
    stationarity_pass = explicit_bool(
        candidate_audit.get("candidate_stationarity_gate_pass"),
        "candidate stationarity gate",
    )
    diagnostic = not endpoint_pass
    candidate_label = (
        r"$\lambda=600$ trained central + final envelope"
        if endpoint_pass
        else r"$\lambda=600$ trained central + final envelope (diagnostic only)"
    )

    candidate_fnp_view = candidate_fnp[
        candidate_fnp["component"].astype(str).eq("combined")
        & np.isclose(candidate_fnp["x"], 0.1)
        & (candidate_fnp["bT"] <= 4.0 + 1.0e-12)
    ].sort_values("bT")
    incumbent_fnp_view = incumbent_fnp[
        np.isclose(incumbent_fnp["x"], 0.1)
        & (incumbent_fnp["bT"] <= 4.0 + 1.0e-12)
    ].sort_values("bT")
    if len(candidate_fnp_view) < 3 or len(incumbent_fnp_view) < 3:
        raise RuntimeError("FNP comparison lacks x=0.1, bT<=4 coverage")
    fnp_comparison = comparison_pair(
        candidate_fnp_view,
        incumbent_fnp_view,
        "bT",
        "F_NP",
        0.1,
        None,
        candidate_label,
        "lambda=600 complete 24x50 centered-log-FNP hierarchy",
        "lambda=1 pinned 24x50 post-processing-harmonized centered-log-FNP hierarchy",
    )
    fnp_comparison = annotate_masks(fnp_comparison, "bT", all_domain_active=True)

    incumbent_fig2, reconstruction = reconstruct_incumbent_fig2(incumbent_b, reference)
    fig2_parts: list[pd.DataFrame] = []
    for flavor in FLAVORS_B:
        candidate_group = candidate_b[
            candidate_b["component"].astype(str).eq("combined")
            & np.isclose(candidate_b["Q"], 7.5)
            & candidate_b["flavor"].astype(str).eq(flavor)
            & (candidate_b["bT"] <= 4.0 + 1.0e-12)
        ].sort_values("bT")
        incumbent_group = incumbent_fig2[
            incumbent_fig2["flavor"].astype(str).eq(flavor)
        ].sort_values("bT")
        if len(candidate_group) < 3:
            raise RuntimeError(f"candidate Fig. 2 lacks Q=7.5 {flavor}")
        fig2_parts.append(
            comparison_pair(
                candidate_group,
                incumbent_group,
                "bT",
                flavor,
                0.1,
                7.5,
                candidate_label,
                "lambda=600 combined b-space TMD band",
                "exact positive-scaling reconstruction from pinned harmonized log-FNP ensemble and frozen Q=7.5 perturbative factor",
            )
        )
    fig2_comparison = annotate_masks(pd.concat(fig2_parts, ignore_index=True), "bT")

    fig6_parts: list[pd.DataFrame] = []
    for flavor in FLAVORS_K:
        candidate_group = candidate_k[
            candidate_k["component"].astype(str).eq("combined")
            & candidate_k["flavor"].astype(str).eq(flavor)
            & np.isclose(candidate_k["Q"], 10.0)
            & (candidate_k["kT"] <= 2.25 + 1.0e-12)
        ].copy()
        if "quantity" in candidate_group.columns:
            candidate_group = candidate_group[
                candidate_group["quantity"].astype(str).eq("ftilde")
            ]
        candidate_group = candidate_group.sort_values("kT")
        incumbent_group = incumbent_k[
            incumbent_k["flavor"].astype(str).eq(flavor)
            & (incumbent_k["kT"] <= 2.25 + 1.0e-12)
        ].sort_values("kT")
        if len(candidate_group) != 226 or len(incumbent_group) != 226:
            raise RuntimeError(f"Fig. 6 lacks exact 226-node displayed grid for {flavor}")
        fig6_parts.append(
            comparison_pair(
                candidate_group,
                incumbent_group,
                "kT",
                flavor,
                0.1,
                10.0,
                candidate_label,
                "lambda=600 combined regularized finite-b transform",
                "lambda=1 post-processing-harmonized regularized finite-b transform",
            )
        )
    fig6_comparison = annotate_masks(pd.concat(fig6_parts, ignore_index=True), "kT")

    metric_rows = (
        width_metrics(fnp_comparison, "bT", "F_NP_x0p1_bT_le_4")
        + width_metrics(fig2_comparison, "bT", "Fig2_bT_x0p1_Q7p5")
        + width_metrics(fig6_comparison, "kT", "Fig6_kT_x0p1_Q10")
    )
    metrics = pd.DataFrame(metric_rows)

    target.mkdir(parents=True, exist_ok=True)
    fnp_csv = target / "fnp_comparison.csv"
    fig2_csv = target / "fig2_bT_comparison.csv"
    fig6_csv = target / "fig6_kT_comparison.csv"
    metrics_csv = target / "relative_full_width_metrics.csv"
    write_csv(fnp_comparison, fnp_csv)
    write_csv(fig2_comparison, fig2_csv)
    write_csv(fig6_comparison, fig6_csv)
    write_csv(metrics, metrics_csv)

    rc = {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.linewidth": 0.9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
    }
    with plt.rc_context(rc):
        fnp_png, fnp_pdf = render_fnp(fnp_comparison, target, candidate_label, diagnostic)
        fig2_png, fig2_pdf = render_fig2(
            fig2_comparison, target, candidate_label, diagnostic
        )
        fig6_png, fig6_pdf = render_fig6(
            fig6_comparison, target, candidate_label, diagnostic
        )

    artifacts = {
        "fnp_png": str(fnp_png),
        "fnp_pdf": str(fnp_pdf),
        "fig2_png": str(fig2_png),
        "fig2_pdf": str(fig2_pdf),
        "fig6_png": str(fig6_png),
        "fig6_pdf": str(fig6_pdf),
        "fnp_comparison_csv": str(fnp_csv),
        "fig2_comparison_csv": str(fig2_csv),
        "fig6_comparison_csv": str(fig6_csv),
        "relative_full_width_metrics_csv": str(metrics_csv),
    }
    artifact_hashes = {key: sha256(Path(path)) for key, path in artifacts.items()}
    input_paths = {
        "candidate_summary": candidate_root / "summary.json",
        "candidate_audit": candidate_audit_path,
        "candidate_postfit_tail_transform_audit": postfit_tail_audit_path,
        "candidate_nested_interaction_validation": nested_interaction_path,
        "candidate_final_directional_envelope": final_envelope_path,
        "candidate_final_fnp_envelope": Path(
            final_envelope["artifacts"]["fnp_final_envelope"]),
        "candidate_final_fig2_bspace_envelope": Path(
            final_envelope["artifacts"]["fig2_bspace_final_envelope"]),
        "candidate_final_fig6_kspace_envelope": Path(
            final_envelope["artifacts"]["fig6_kspace_final_envelope"]),
        "candidate_fnp_bands": candidate_root / "fnp_bands.csv",
        "candidate_bspace_bands": candidate_root / "bT_tmd_bands.csv",
        "candidate_kspace_bands": candidate_root / "kT_tmd_bands.csv",
        "candidate_kspace_ensemble_long":
            candidate_root / "kT_tmd_ensemble_long.csv",
        "candidate_bootstrap_width_statistic_deviations":
            candidate_audit_path.parent /
            "bootstrap_full_width_statistic_deviations.csv",
        "candidate_split_half_width_statistic_differences":
            candidate_audit_path.parent /
            "split_half_full_width_statistic_differences.csv",
        "harmonized_summary": incumbent_root / "summary.json",
        "harmonized_provenance": incumbent_root / "input_provenance.json",
        "harmonized_fnp_bands": incumbent_root / "fnp_combined_bands.csv",
        "harmonized_bspace_bands": incumbent_root / "bspace_combined_bands.csv",
        "harmonized_kspace_bands": incumbent_root / "kspace_combined_bands.csv",
        "pinned_input_manifest": pin_manifest_path,
        "pinned_incumbent_record": pinned_record_path,
        "frozen_perturbative_reference": reference_path,
    }
    input_hashes = {key: sha256(path) for key, path in input_paths.items()}
    legacy_widths = {
        flavor: float(
            pinned_record["combined_fig6_max_active_relative_full_width"][flavor]
        )
        for flavor in FLAVORS_K
    }
    audit_locked = candidate_audit.get(
        "comparison_champion_max_active_relative_full_width", {})
    audit_union = candidate_audit.get(
        "comparison_champion_union_mask_relative_full_width", {})
    if set(audit_locked) != set(FLAVORS_K) or set(audit_union) != set(FLAVORS_K):
        raise RuntimeError("candidate audit lacks locked and union-mask lambda1 widths")
    for flavor in FLAVORS_K:
        exact_number(audit_locked[flavor], legacy_widths[flavor],
                     f"candidate audit locked lambda1 {flavor} width")
        if not np.isfinite(float(audit_union[flavor])) or float(audit_union[flavor]) <= 0.0:
            raise RuntimeError(f"invalid candidate-audit union-mask {flavor} width")
    audit_harmonized = candidate_audit.get(
        "postprocessing_harmonized_lambda1_control", {}
    )
    if audit_harmonized:
        if (
            explicit_bool(
                audit_harmonized.get("training_protocol_harmonized"),
                "candidate audit harmonized training flag",
            )
            or explicit_bool(
                audit_harmonized.get("gating"),
                "candidate audit harmonized gating flag",
            )
        ):
            raise RuntimeError("candidate audit incorrectly treats harmonized lambda=1 as gating")
        if audit_harmonized.get("kspace_band_sha256") != input_hashes[
            "harmonized_kspace_bands"
        ]:
            raise RuntimeError("candidate audit and diagnostic use different harmonized k bands")

    status = (
        "complete_validated_candidate_comparison"
        if endpoint_pass
        else "complete_diagnostic_scientific_failure_comparison"
    )
    summary = {
        "status": status,
        "comparison_champion_id": EXPECTED_CHAMPION_ID,
        "candidate_prescription": {
            "reference_strength": 600.0,
            "reference_bmax_GeV_inverse": 4.0,
            "fit_quality_barrier_strength": 100.0,
            "fit_quality_barrier_power": 2,
            "start_count": 24,
            "experimental_replica_count": 50,
            "combined_member_count": 1200,
        },
        "candidate_endpoint_gate_pass": endpoint_pass,
        "candidate_base_stability_endpoint_gate_pass": explicit_bool(
            candidate_audit.get("endpoint_gate_pass"),
            "candidate base stability endpoint gate",
        ),
        "candidate_final_directional_envelope_gate_pass": endpoint_pass,
        "candidate_final_directional_envelope_sha256": final_envelope_hash,
        "candidate_stationarity_gate_pass": stationarity_pass,
        "diagnostic_only": diagnostic,
        "scientific_failure_reasons": [
            str(value) for value in final_envelope.get(
                "scientific_failure_reasons", []) if str(value)
        ],
        "immutable_incumbent_hashes_validated": True,
        "pinned_input_file_count": 55,
        "training_protocol_harmonized": False,
        "incumbent_and_candidate_training_protocols_identical": False,
        "legacy_lambda1_fig6_widths_remain_gating": True,
        "legacy_lambda1_fig6_max_active_relative_full_width": legacy_widths,
        "lambda1_union_mask_relative_full_width_diagnostic": {
            flavor: float(audit_union[flavor]) for flavor in FLAVORS_K
        },
        "harmonized_lambda1_role": (
            "non-gating diagnostic control for matched centered-log-FNP propagation "
            "and finite-b transform order"
        ),
        "hard_caveat": harmonized_summary.get("hard_caveat"),
        "fig2_lambda1_reconstruction": reconstruction,
        "active_width_definition": (
            "relative full width=(q84-q16)/abs(central); FNP metrics use the full "
            "displayed bT<=4 domain, while Fig. 2 and Fig. 6 comparison metrics use "
            "the union of each method's central-above-5%-of-own-displayed-peak mask"
        ),
        "central_curve_semantics": (
            "the lambda600 curve is the separately trained terminal 300k central "
            "endpoint and its exact paired transform; the harmonized lambda1 curve "
            "remains its declared post-processing ensemble central"
        ),
        "interval_probability_semantics": (
            "the lambda600 band is its empirical product band plus residual "
            "convergence/interaction envelope; only the "
            "conditional experimental-replica marginal has a conventional replica "
            "interpretation, and no confidence-level or standard-deviation meaning "
            "is assigned to the full envelope"
        ),
        "candidate_final_width_metrics_by_flavor": final_envelope.get(
            "width_metrics_by_flavor", {}
        ),
        "relative_full_width_metrics": nested_metrics(metric_rows),
        "candidate_audit_resampling_allowance_by_flavor": candidate_audit.get(
            "resampling_full_width_allowance_by_flavor", {}
        ),
        "candidate_audit_robust_improvement_gate_by_flavor": candidate_audit.get(
            "robust_improvement_gate_by_flavor", {}
        ),
        "artifacts": artifacts,
        "artifact_sha256": artifact_hashes,
        "input_sha256": input_hashes,
        "registry_modified": False,
        "candidate_or_canonical_figures_modified": False,
        "frozen_sources_modified": False,
        "production_sources_modified": False,
    }
    summary_path = target / "summary.json"
    atomic_write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--candidate-audit", type=Path, default=DEFAULT_CANDIDATE_AUDIT)
    parser.add_argument("--postfit-tail-audit", type=Path,
                        default=DEFAULT_POSTFIT_TAIL_AUDIT)
    parser.add_argument("--nested-interaction", type=Path,
                        default=DEFAULT_NESTED_INTERACTION)
    parser.add_argument("--final-envelope", type=Path,
                        default=DEFAULT_FINAL_ENVELOPE)
    parser.add_argument("--incumbent-dir", type=Path, default=DEFAULT_INCUMBENT)
    parser.add_argument("--pinned-record", type=Path, default=DEFAULT_PINNED_RECORD)
    parser.add_argument("--pin-manifest", type=Path, default=DEFAULT_PIN_MANIFEST)
    parser.add_argument("--reference-bspace", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
