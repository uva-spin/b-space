#!/usr/bin/env python3
"""Fail-closed validators for nested interactions and the final envelope."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fixed_challenger_protocol import require_fixed_implementation_binding
from postfit_tail_transform_validation import validated_postfit_tail_audit


BASE = Path(__file__).resolve().parents[1]
SUM = BASE / "summaries"
NESTED = SUM / "lambda600_nested_start_replica_interaction/summary.json"
FINAL_ENVELOPE = SUM / "lambda600_final_directional_envelope/summary.json"
STABILITY = SUM / "final_combined_ensemble_stability/summary.json"
POSTFIT = SUM / "lambda600_postfit_tail_transform_audit/summary.json"
FINAL_PRODUCT = SUM / "final_combined_tmd_ensemble/summary.json"
NESTED_TERMINAL_STATUSES = {
    "complete_observed_interaction_decision_sign_stable",
    "material_nonadditive_interaction_changes_decision",
    "nested_trajectory_stationarity_or_fit_failure",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def explicit_bool(value, label: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise RuntimeError(f"{label} is not an explicit boolean")


def reconstructed_final_promotion_gate(payload: dict) -> bool:
    """Reconstruct the final gate as the exact five-component conjunction."""
    return bool(
        explicit_bool(
            payload.get("base_product_stability_gate_pass"),
            "final envelope base_product_stability_gate_pass",
        )
        and explicit_bool(
            payload.get("postfit_tail_convergence_gate_pass"),
            "final envelope postfit_tail_convergence_gate_pass",
        )
        and explicit_bool(
            payload.get("nested_interaction_validation_gate_pass"),
            "final envelope nested_interaction_validation_gate_pass",
        )
        and explicit_bool(
            payload.get("joint_width_replacement_gate_pass"),
            "final envelope joint_width_replacement_gate_pass",
        )
        and explicit_bool(
            payload.get("trained_central_containment_gate_pass"),
            "final envelope trained_central_containment_gate_pass",
        )
    )


def reconstructed_nested_interaction_gate(payload: dict) -> bool:
    """Interaction evidence is eligible iff all six raw trajectories pass."""
    return explicit_bool(
        payload.get("all_nested_trajectories_stationary_and_fit_preserving"),
        "all_nested_trajectories_stationary_and_fit_preserving",
    )


def _load(path: Path, label: str) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"{label} is missing: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} is not a JSON object")
    return payload


def _validate_artifacts(payload: dict, label: str) -> dict[str, Path]:
    artifacts = payload.get("artifacts")
    hashes = payload.get("artifact_sha256")
    if (not isinstance(artifacts, dict) or not artifacts
            or not isinstance(hashes, dict) or set(hashes) != set(artifacts)):
        raise RuntimeError(f"{label} artifact/hash coverage is incomplete")
    resolved: dict[str, Path] = {}
    for role, text in artifacts.items():
        path = Path(text)
        if (not path.is_file() or path.stat().st_size == 0
                or not hashes.get(role) or sha256(path) != hashes[role]):
            raise RuntimeError(f"{label} artifact changed ({role}): {path}")
        resolved[str(role)] = path
    return resolved


def validate_nested_raw_endpoints(
    endpoints: list[dict], pairs: list[dict], artifacts: dict[str, Path]
) -> None:
    """Require exact 2x3 coverage and immutable raw terminal endpoint files."""
    design_pairs = {
        (int(item["start_seed"]), int(item["replica_seed"])) for item in pairs
    }
    endpoint_pairs = {
        (int(item["start_seed"]), int(item["replica_seed"])) for item in endpoints
    }
    if design_pairs != endpoint_pairs or len(design_pairs) != 6:
        raise RuntimeError("nested terminal endpoint/design pair coverage differs")
    required_raw_roles = {"fit_status", "fnp_grid", "model_state"}
    for endpoint in endpoints:
        start_seed = int(endpoint["start_seed"])
        replica_seed = int(endpoint["replica_seed"])
        endpoint_tag = str(endpoint.get("endpoint_tag", ""))
        expected_tag = (
            "nestedint_fullref_lam600_fitbar_p2_mu100_b4_"
            f"s{start_seed}_r{replica_seed}_polish64_300000"
        )
        raw = endpoint.get("raw_artifacts", {})
        raw_hashes = endpoint.get("raw_artifact_sha256", {})
        if (endpoint_tag != expected_tag or not isinstance(raw, dict)
                or not isinstance(raw_hashes, dict)
                or set(raw) != set(raw_hashes)
                or not required_raw_roles.issubset(raw)):
            raise RuntimeError("nested raw terminal endpoint declaration is invalid")
        status_path = Path(raw["fit_status"])
        endpoint_dir = status_path.parent
        expected_names = {
            "fit_status": "fit_status.json",
            "fnp_grid": "fnp_grid.csv",
            "model_state": "model_state.pt",
            "dataset_norms": "dataset_norms.csv",
        }
        if (endpoint_dir.name != endpoint_tag
                or (endpoint_dir / "dataset_norms.csv").is_file()
                    != ("dataset_norms" in raw)):
            raise RuntimeError("nested raw endpoint directory/norm coverage differs")
        prefix = f"terminal_s{start_seed}_r{replica_seed}"
        for role, path_text in raw.items():
            path = Path(path_text)
            artifact_role = f"{prefix}_{role}"
            if (role not in expected_names or path.parent != endpoint_dir
                    or path.name != expected_names[role]
                    or not path.is_file() or path.stat().st_size == 0
                    or sha256(path) != raw_hashes[role]
                    or artifacts.get(artifact_role) != path):
                raise RuntimeError(
                    f"nested raw terminal endpoint changed: {endpoint_tag} {role}"
                )


def validated_nested_interaction(
    nested_path: Path | None = None,
    postfit_path: Path | None = None,
    stability_path: Path | None = None,
) -> tuple[dict, str]:
    nested_path = NESTED if nested_path is None else Path(nested_path)
    postfit_path = POSTFIT if postfit_path is None else Path(postfit_path)
    stability_path = STABILITY if stability_path is None else Path(stability_path)
    _, postfit_hash = validated_postfit_tail_audit(postfit_path, stability_path)
    payload = _load(nested_path, "nested-interaction summary")
    require_fixed_implementation_binding(payload, "nested-interaction summary")
    endpoints = payload.get("terminal_endpoints", [])
    pairs = payload.get("design", {}).get("pairs", [])
    trajectory = payload.get("trajectory_protocol", {})
    if (
        payload.get("status") not in NESTED_TERMINAL_STATUSES
        or int(payload.get("completed_pair_count", -1)) != 6
        or len(endpoints) != 6
        or len({(int(item["start_seed"]), int(item["replica_seed"]))
                for item in endpoints}) != 6
        or len(pairs) != 6
        or not explicit_bool(
            trajectory.get("full_horizon_exhausted_for_every_pair"),
            "nested full_horizon_exhausted_for_every_pair",
        )
        or int(trajectory.get("total_requested_lbfgs_capacity", -1)) != 300_000
        or Path(payload.get("postfit_tail_transform_audit", "")).resolve()
            != postfit_path.resolve()
        or payload.get("postfit_tail_transform_audit_sha256") != postfit_hash
        or explicit_bool(
            payload.get("production_sources_modified"),
            "nested production_sources_modified",
        )
    ):
        raise RuntimeError("nested-interaction terminal evidence is invalid")
    artifacts = _validate_artifacts(payload, "nested-interaction")
    validate_nested_raw_endpoints(endpoints, pairs, artifacts)
    log_envelope = pd.read_csv(artifacts["logfnp_directional_envelope"])
    required_log = {
        "bT", "interaction_log_delta_low", "interaction_log_delta_high",
    }
    if (not required_log.issubset(log_envelope.columns)
            or len(log_envelope) != 321
            or float(log_envelope["bT"].max()) < 8.0 - 1e-12
            or not np.all(np.isfinite(log_envelope[list(required_log)].to_numpy(float)))
            or np.any(log_envelope["interaction_log_delta_low"].to_numpy(float) > 0.0)
            or np.any(log_envelope["interaction_log_delta_high"].to_numpy(float) < 0.0)):
        raise RuntimeError("nested full-grid log-FNP envelope is invalid")
    k_envelope = pd.read_csv(artifacts["kspace_directional_envelope"])
    required_k = {
        "flavor", "kT", "interaction_delta_low", "interaction_delta_high",
    }
    if (not required_k.issubset(k_envelope.columns)
            or k_envelope.groupby("flavor").size().to_dict() != {"d": 401, "u": 401}
            or not np.all(np.isfinite(
                k_envelope[["kT", "interaction_delta_low",
                            "interaction_delta_high"]].to_numpy(float)))
            or np.any(k_envelope["interaction_delta_low"].to_numpy(float) > 0.0)
            or np.any(k_envelope["interaction_delta_high"].to_numpy(float) < 0.0)):
        raise RuntimeError("nested k-space directional envelope is invalid")
    gate = explicit_bool(
        payload.get("interaction_validation_gate_pass"),
        "interaction_validation_gate_pass",
    )
    all_stationary = explicit_bool(
        payload.get("all_nested_trajectories_stationary_and_fit_preserving"),
        "all_nested_trajectories_stationary_and_fit_preserving",
    )
    sign_stable = explicit_bool(
        payload.get("observed_interaction_decision_sign_stable"),
        "observed_interaction_decision_sign_stable",
    )
    if (gate != reconstructed_nested_interaction_gate(payload)
            or explicit_bool(
                payload.get("legacy_product_median_interaction_sign_can_gate"),
                "legacy_product_median_interaction_sign_can_gate",
            )
            or not explicit_bool(
                payload.get("final_joint_sampling_gate_authoritative"),
                "nested final_joint_sampling_gate_authoritative",
            )):
        raise RuntimeError("nested-interaction gate semantics disagree")
    return payload, sha256(nested_path)


def validated_final_directional_envelope(
    envelope_path: Path | None = None,
    nested_path: Path | None = None,
    postfit_path: Path | None = None,
    stability_path: Path | None = None,
) -> tuple[dict, str]:
    envelope_path = FINAL_ENVELOPE if envelope_path is None else Path(envelope_path)
    nested_path = NESTED if nested_path is None else Path(nested_path)
    postfit_path = POSTFIT if postfit_path is None else Path(postfit_path)
    stability_path = STABILITY if stability_path is None else Path(stability_path)
    _, nested_hash = validated_nested_interaction(
        nested_path, postfit_path, stability_path
    )
    _, postfit_hash = validated_postfit_tail_audit(postfit_path, stability_path)
    payload = _load(envelope_path, "final directional-envelope summary")
    require_fixed_implementation_binding(payload, "final directional-envelope summary")
    provenance = payload.get("input_provenance", {})
    if (
        payload.get("status") != "complete_final_directional_envelope"
        or int(payload.get("combined_product_member_count", -1)) != 1200
        or not explicit_bool(
            payload.get("diagnostic_figure_gate_pass"),
            "final envelope diagnostic_figure_gate_pass",
        )
        or Path(provenance.get("nested_interaction_summary", "")).resolve()
            != nested_path.resolve()
        or provenance.get("nested_interaction_summary_sha256") != nested_hash
        or Path(provenance.get("postfit_tail_transform_summary", "")).resolve()
            != postfit_path.resolve()
        or provenance.get("postfit_tail_transform_summary_sha256") != postfit_hash
        or Path(provenance.get("stability_summary", "")).resolve()
            != stability_path.resolve()
        or provenance.get("stability_summary_sha256") != sha256(stability_path)
        or provenance.get("final_product_summary_sha256") != sha256(FINAL_PRODUCT)
        or explicit_bool(
            payload.get("production_sources_modified"),
            "final envelope production_sources_modified",
        )
        or explicit_bool(
            payload.get("formal_confidence_level_assigned"),
            "final envelope formal_confidence_level_assigned",
        )
        or explicit_bool(
            payload.get("one_sigma_claimed"),
            "final envelope one_sigma_claimed",
        )
    ):
        raise RuntimeError("final directional-envelope evidence is invalid")
    artifacts = _validate_artifacts(payload, "final directional-envelope")
    base_gate = explicit_bool(
        payload.get("base_product_stability_gate_pass"),
        "final envelope base_product_stability_gate_pass",
    )
    postfit_gate = explicit_bool(
        payload.get("postfit_tail_convergence_gate_pass"),
        "final envelope postfit_tail_convergence_gate_pass",
    )
    nested_gate = explicit_bool(
        payload.get("nested_interaction_validation_gate_pass"),
        "final envelope nested_interaction_validation_gate_pass",
    )
    width_gate = explicit_bool(
        payload.get("joint_width_replacement_gate_pass"),
        "final envelope joint_width_replacement_gate_pass",
    )
    containment_gate = explicit_bool(
        payload.get("trained_central_containment_gate_pass"),
        "final envelope trained_central_containment_gate_pass",
    )
    promotion = explicit_bool(
        payload.get("promotion_validation_gate_pass"),
        "final envelope promotion_validation_gate_pass",
    )
    diagnostic_only = explicit_bool(
        payload.get("diagnostic_only"), "final envelope diagnostic_only"
    )
    expected_promotion = reconstructed_final_promotion_gate(payload)
    if (promotion != expected_promotion or diagnostic_only != (not promotion)
            or not explicit_bool(
                payload.get("final_joint_sampling_gate_authoritative"),
                "final_joint_sampling_gate_authoritative",
            )
            or explicit_bool(
                payload.get("prior_product_median_sampling_gate_authoritative"),
                "prior_product_median_sampling_gate_authoritative",
            )):
        raise RuntimeError("final directional-envelope promotion semantics disagree")

    resampling = payload.get("final_statistic_resampling", {})
    metrics = payload.get("width_metrics_by_flavor", {})
    if (int(resampling.get("bootstrap_replicates", -1)) != 300
            or int(resampling.get("split_half_replicates", -1)) != 200
            or set(metrics) != {"u", "d"}
            or set(resampling.get("allowance_by_flavor", {})) != {"u", "d"}
            or set(resampling.get("full_exact_final_statistic_by_flavor", {}))
                != {"u", "d"}):
        raise RuntimeError("final-statistic resampling declaration is incomplete")
    bootstrap = pd.read_csv(artifacts["final_statistic_bootstrap_deviations"])
    split = pd.read_csv(artifacts["final_statistic_split_differences"])
    bootstrap_paths = payload.get("final_statistic_resampling_artifacts", {})
    if (Path(bootstrap_paths.get("bootstrap_deviations", "")).resolve()
            != artifacts["final_statistic_bootstrap_deviations"].resolve()
            or Path(bootstrap_paths.get("split_differences", "")).resolve()
            != artifacts["final_statistic_split_differences"].resolve()
            or len(bootstrap) != 300 or len(split) != 200):
        raise RuntimeError("final-statistic resampling artifacts are incomplete")
    reconstructed_width_gate = True
    for flavor in ("u", "d"):
        item = metrics[flavor]
        raw = float(item["joint_convergence_interaction_raw_full_width"])
        exact_full = float(
            resampling["full_exact_final_statistic_by_flavor"][flavor]
        )
        allowance = float(resampling["allowance_by_flavor"][flavor])
        declared_allowance = float(
            item["final_statistic_finite_sampling_full_width_margin"]
        )
        compatibility_allowance = float(
            item["corrected_finite_sampling_full_width_margin"]
        )
        adjusted = float(
            item["joint_raw_width_plus_final_statistic_sampling_margin"]
        )
        compatibility_adjusted = float(
            item["joint_raw_width_plus_corrected_sampling_margin"]
        )
        locked = float(item["immutable_lambda1_width"])
        declared_pass = explicit_bool(
            item.get("replacement_gate_pass"), f"{flavor} replacement_gate_pass"
        )
        resampling_columns = {
            "bootstrap": bootstrap[
                f"{flavor}_absolute_final_statistic_deviation"
            ].to_numpy(float),
            "start_split": split[
                f"start_split_{flavor}_absolute_final_statistic_difference"
            ].to_numpy(float),
            "replica_split": split[
                f"replica_split_{flavor}_absolute_final_statistic_difference"
            ].to_numpy(float),
            "joint_split": split[
                f"joint_split_{flavor}_absolute_final_statistic_difference"
            ].to_numpy(float),
        }
        if any(
            not np.all(np.isfinite(array)) or np.any(array < 0.0)
            for array in resampling_columns.values()
        ):
            raise RuntimeError(
                f"final-statistic resampling artifact is invalid for {flavor}"
            )
        direct_p95 = {
            label: float(np.quantile(array, 0.95))
            for label, array in resampling_columns.items()
        }
        declared_p95 = {
            "bootstrap": float(
                resampling["bootstrap_p95_absolute_deviation_by_flavor"][flavor]
            ),
            "start_split": float(
                resampling["start_split_p95_absolute_difference_by_flavor"][flavor]
            ),
            "replica_split": float(
                resampling["replica_split_p95_absolute_difference_by_flavor"][flavor]
            ),
            "joint_split": float(
                resampling["joint_split_p95_absolute_difference_by_flavor"][flavor]
            ),
        }
        direct_allowance = max(direct_p95.values())
        expected_pass = bool(raw + allowance < locked)
        values = np.asarray([
            raw, exact_full, allowance, declared_allowance,
            compatibility_allowance, adjusted, compatibility_adjusted, locked,
        ])
        if (not np.all(np.isfinite(values)) or allowance < 0.0 or locked <= 0.0
                or not np.isclose(raw, exact_full, rtol=5e-13, atol=5e-15)
                or not np.isclose(allowance, declared_allowance,
                                  rtol=0.0, atol=1e-15)
                or not np.isclose(allowance, compatibility_allowance,
                                  rtol=0.0, atol=1e-15)
                or any(
                    not np.isclose(direct_p95[label], declared_p95[label],
                                   rtol=5e-15, atol=5e-15)
                    for label in direct_p95
                )
                or not np.isclose(allowance, direct_allowance,
                                  rtol=5e-15, atol=5e-15)
                or not np.isclose(adjusted, raw + allowance,
                                  rtol=5e-15, atol=5e-15)
                or not np.isclose(compatibility_adjusted, adjusted,
                                  rtol=0.0, atol=1e-15)
                or declared_pass != expected_pass
                or not explicit_bool(
                    item.get("sampling_allowance_statistic_matches_width_statistic"),
                    f"{flavor} sampling statistic match",
                )):
            raise RuntimeError(
                f"final-statistic sampling/width evidence differs for {flavor}"
            )
        reconstructed_width_gate = reconstructed_width_gate and expected_pass
    if width_gate != reconstructed_width_gate:
        raise RuntimeError("final joint-width gate does not match flavor decisions")
    return payload, sha256(envelope_path)


def final_promotion_gate(payload: dict) -> bool:
    return explicit_bool(
        payload.get("promotion_validation_gate_pass"),
        "final envelope promotion_validation_gate_pass",
    )
