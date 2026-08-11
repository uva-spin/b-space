#!/usr/bin/env python3
"""Shared fail-closed validation for the lambda=600 post-fit tail audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fixed_challenger_protocol import require_fixed_implementation_binding


BASE = Path(__file__).resolve().parents[1]
AUDIT = BASE / "summaries/lambda600_postfit_tail_transform_audit/summary.json"
STABILITY = BASE / "summaries/final_combined_ensemble_stability/summary.json"
EXPECTED_MODES = ("expb2", "expb", "taper")
EXACT_ARRAY_SCHEMA = "lambda600_exact_expb2_checkpoint_arrays_v1"
EXACT_ARRAY_KEYS = {
    "schema", "checkpoints", "flavors", "start_seeds", "replica_seeds",
    "kT", "terminal_values", "stationarity_anchor_values",
    "terminal_declared_central", "stationarity_anchor_declared_central",
}
QUANTILE_RTOL = 5.0e-13
QUANTILE_ATOL = 5.0e-15


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


def validated_exact_checkpoint_transform_arrays(payload: dict) -> dict[str, np.ndarray]:
    """Validate and load the hash-bound correlated terminal/anchor arrays."""
    artifacts = payload.get("artifacts", {})
    path = Path(artifacts.get("exact_checkpoint_expb2_members", ""))
    bands_path = Path(artifacts.get("kspace_checkpoint_expb2_bands", ""))
    metadata = payload.get("exact_checkpoint_transform_arrays", {})
    if (not path.is_file() or not bands_path.is_file()
            or metadata.get("schema") != EXACT_ARRAY_SCHEMA
            or metadata.get("shape") != [2, 24, 50, 401]
            or metadata.get("axis_order")
                != ["flavor", "start_seed", "replica_seed", "kT"]
            or metadata.get("checkpoints")
                != ["terminal", "stationarity_anchor"]
            or metadata.get("flavors") != ["u", "d"]
            or metadata.get("start_seed_range") != [303, 326]
            or metadata.get("replica_seed_range") != [1001, 1050]
            or not explicit_bool(
                metadata.get("quantiles_declared_from_exact_arrays"),
                "quantiles_declared_from_exact_arrays",
            )):
        raise RuntimeError("exact checkpoint transform metadata is invalid")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != EXACT_ARRAY_KEYS:
            raise RuntimeError("exact checkpoint transform array keys differ")
        arrays = {key: np.asarray(archive[key]).copy() for key in archive.files}
    if (str(arrays["schema"].item()) != EXACT_ARRAY_SCHEMA
            or arrays["checkpoints"].tolist()
                != ["terminal", "stationarity_anchor"]
            or arrays["flavors"].tolist() != ["u", "d"]
            or arrays["start_seeds"].dtype != np.int64
            or arrays["replica_seeds"].dtype != np.int64
            or not np.array_equal(arrays["start_seeds"], np.arange(303, 327))
            or not np.array_equal(arrays["replica_seeds"], np.arange(1001, 1051))
            or arrays["kT"].shape != (401,)
            or arrays["kT"].dtype != np.float64
            or not np.all(np.isfinite(arrays["kT"]))
            or not np.all(np.diff(arrays["kT"]) > 0.0)
            or arrays["terminal_values"].shape != (2, 24, 50, 401)
            or arrays["stationarity_anchor_values"].shape != (2, 24, 50, 401)
            or arrays["terminal_declared_central"].shape != (2, 401)
            or arrays["stationarity_anchor_declared_central"].shape != (2, 401)):
        raise RuntimeError("exact checkpoint transform identities or shapes differ")
    numeric_keys = (
        "terminal_values", "stationarity_anchor_values",
        "terminal_declared_central", "stationarity_anchor_declared_central",
    )
    if any(
        arrays[key].dtype != np.float64 or not np.all(np.isfinite(arrays[key]))
        for key in numeric_keys
    ):
        raise RuntimeError("exact checkpoint transform arrays are not finite float64")

    bands = pd.read_csv(bands_path)
    bands = bands[
        bands["model"].astype(str).eq("lambda600")
        & bands["tail_mode"].astype(str).eq("expb2")
    ]
    if (bands.groupby(["checkpoint", "flavor"]).size().to_dict()
            != {("stationarity_anchor", "d"): 401,
                ("stationarity_anchor", "u"): 401,
                ("terminal", "d"): 401, ("terminal", "u"): 401}):
        raise RuntimeError("declared checkpoint expb2 bands lack exact coverage")
    for checkpoint, value_key, central_key in (
        ("terminal", "terminal_values", "terminal_declared_central"),
        ("stationarity_anchor", "stationarity_anchor_values",
         "stationarity_anchor_declared_central"),
    ):
        for flavor_index, flavor in enumerate(("u", "d")):
            group = bands[
                bands["checkpoint"].astype(str).eq(checkpoint)
                & bands["flavor"].astype(str).eq(flavor)
            ].sort_values("kT")
            if not np.allclose(
                group["kT"].to_numpy(float), arrays["kT"],
                rtol=0.0, atol=1.0e-14,
            ):
                raise RuntimeError(
                    f"declared/exact checkpoint grid differs for {checkpoint} {flavor}"
                )
            members = arrays[value_key][flavor_index].reshape(1200, 401)
            quantiles = np.quantile(members, (0.16, 0.50, 0.84), axis=0)
            observed = group[["q16", "median", "q84"]].to_numpy(float).T
            if (not np.allclose(observed, quantiles, rtol=QUANTILE_RTOL,
                                atol=QUANTILE_ATOL)
                    or not np.allclose(
                        group["declared_central"].to_numpy(float),
                        arrays[central_key][flavor_index],
                        rtol=QUANTILE_RTOL, atol=QUANTILE_ATOL,
                    )):
                raise RuntimeError(
                    f"declared checkpoint bands differ from exact arrays for "
                    f"{checkpoint} {flavor}"
                )
    return arrays


def reconstructed_postfit_promotion_gate(payload: dict) -> bool:
    """Reconstruct the postfit gate exclusively from its declared components."""
    return bool(
        explicit_bool(payload.get("coverage_gate_pass"), "coverage_gate_pass")
        and explicit_bool(
            payload.get("candidate_stationarity_gate_pass"),
            "candidate_stationarity_gate_pass",
        )
        and explicit_bool(
            payload.get("central_line_containment_gate_pass"),
            "central_line_containment_gate_pass",
        )
        and explicit_bool(
            payload.get("transform_decision_robustness_gate_pass"),
            "transform_decision_robustness_gate_pass",
        )
    )


def validated_postfit_tail_audit(
    audit_path: Path | None = None,
    stability_path: Path | None = None,
) -> tuple[dict, str]:
    audit_path = AUDIT if audit_path is None else Path(audit_path)
    stability_path = STABILITY if stability_path is None else Path(stability_path)
    if not audit_path.is_file():
        raise RuntimeError("post-fit full-tail/transform audit is missing")
    payload = json.loads(audit_path.read_text())
    require_fixed_implementation_binding(payload, "post-fit tail/transform audit")
    provenance = payload.get("input_provenance", {})
    artifacts = payload.get("artifacts", {})
    artifact_hashes = payload.get("artifact_sha256", {})
    if (
        payload.get("status")
        != "complete_postfit_full_tail_transform_validation"
        or int(payload.get("combined_member_count", 0)) != 1200
        or payload.get("tail_modes") != list(EXPECTED_MODES)
        or payload.get("locked_gating_tail_mode") != "expb2"
        or not explicit_bool(
            payload.get("alternate_tail_modes_gate_promotion"),
            "alternate_tail_modes_gate_promotion",
        )
        or explicit_bool(
            payload.get(
                "alternate_tail_modes_can_replace_or_loosen_locked_expb2_width_gate"
            ),
            "alternate_tail_modes_can_replace_or_loosen_locked_expb2_width_gate",
        )
        or not explicit_bool(
            payload.get("diagnostic_figure_gate_pass"),
            "diagnostic_figure_gate_pass",
        )
        or not explicit_bool(payload.get("coverage_gate_pass"), "coverage_gate_pass")
        or Path(provenance.get("stability_summary", "")).resolve()
        != stability_path.resolve()
        or provenance.get("stability_summary_sha256") != sha256(stability_path)
        or explicit_bool(
            payload.get("fixed_challenger_protocol_modified"),
            "fixed_challenger_protocol_modified",
        )
        or explicit_bool(payload.get("frozen_sources_modified"), "frozen_sources_modified")
        or explicit_bool(
            payload.get("production_sources_modified"),
            "production_sources_modified",
        )
    ):
        raise RuntimeError("post-fit full-tail/transform audit is invalid")
    if set(artifact_hashes) != set(artifacts) or not artifacts:
        raise RuntimeError("post-fit tail artifact/hash coverage is incomplete")
    for label, path_text in artifacts.items():
        path = Path(path_text)
        if (not path.is_file() or not artifact_hashes.get(label)
                or sha256(path) != artifact_hashes[label]):
            raise RuntimeError(
                f"post-fit tail artifact is missing or changed ({label}): {path}"
            )
    pairs = pd.read_csv(Path(artifacts["endpoint_pairs"]))
    required_pair_columns = {
        "family", "identity", "scientifically_nonstationary",
        "anchor_selection_rule", "anchor_iterations", "terminal_iterations",
    }
    failed_policy = payload.get("failed_chain_anchor_policy", {})
    failed_mask = pairs["scientifically_nonstationary"].astype(str).str.lower().map(
        {"true": True, "false": False}
    ) if "scientifically_nonstationary" in pairs else pd.Series(dtype=bool)
    if (not required_pair_columns.issubset(pairs.columns) or len(pairs) != 75
            or failed_mask.isna().any()
            or failed_policy.get("selection")
                != "fixed_formal_200k_for_nonstationary_chain"
            or int(failed_policy.get("fixed_requested_lbfgs_capacity", -1))
                != 200_000
            or explicit_bool(
                failed_policy.get("late_reset_anchor_allowed_for_nonstationary_chain"),
                "late_reset_anchor_allowed_for_nonstationary_chain",
            )
            or failed_policy.get("passing_chain_selection")
                != "validated_terminal_stationarity_window_anchor"
            or int(failed_policy.get("nonstationary_chain_count", -1))
                != int(failed_mask.sum())
            or np.any(
                pairs.loc[failed_mask, "anchor_iterations"].to_numpy(float)
                != 200_000
            )
            or not pairs.loc[failed_mask, "anchor_selection_rule"].astype(str).eq(
                "fixed_formal_200k_for_nonstationary_chain"
            ).all()
            or not pairs.loc[~failed_mask, "anchor_selection_rule"].astype(str).eq(
                "validated_terminal_stationarity_window_anchor"
            ).all()
            or np.any(
                pairs["anchor_iterations"].to_numpy(float)
                >= pairs["terminal_iterations"].to_numpy(float)
            )):
        raise RuntimeError("post-fit nonstationary-chain anchor policy is invalid")
    promotion = explicit_bool(
        payload.get("promotion_validation_gate_pass"),
        "promotion_validation_gate_pass",
    )
    expected_promotion = reconstructed_postfit_promotion_gate(payload)
    diagnostic_only = explicit_bool(payload.get("diagnostic_only"), "diagnostic_only")
    if (promotion != expected_promotion or diagnostic_only != (not promotion)
            or explicit_bool(
                payload.get("obsolete_product_median_allowance_can_gate_postfit"),
                "obsolete_product_median_allowance_can_gate_postfit",
            )
            or not explicit_bool(
                payload.get("final_joint_sampling_gate_authoritative"),
                "final_joint_sampling_gate_authoritative",
            )
            or explicit_bool(
                payload.get("same_expb2_incumbent_replacement", {}).get("gating"),
                "same_expb2_incumbent_replacement.gating",
            )):
        raise RuntimeError("post-fit tail audit promotion semantics disagree")
    validated_exact_checkpoint_transform_arrays(payload)
    return payload, sha256(audit_path)


def combined_endpoint_gate(stability: dict, tail_audit: dict) -> bool:
    return bool(
        explicit_bool(
            stability.get("diagnostic_figure_gate_pass"),
            "diagnostic_figure_gate_pass",
        )
        and explicit_bool(
            stability.get("candidate_stationarity_gate_pass"),
            "candidate_stationarity_gate_pass",
        )
        and explicit_bool(
            tail_audit.get("promotion_validation_gate_pass"),
            "promotion_validation_gate_pass",
        )
    )
