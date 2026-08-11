#!/usr/bin/env python3
"""Validate the pre-registered lambda600 fixed-challenger protocol.

The protocol was frozen before the complete challenger evidence existed.  Its
digest is therefore part of the scientific provenance, not a value inferred
from the eventual result.  This helper is read-only and deliberately pins the
exact protocol, empirical objective reference, and external fit implementation
bytes so downstream code fails closed if their content or interpretation
changes.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
PROTOCOL = BASE / "manifests/lambda600_fixed_challenger_protocol.json"
EXPECTED_SHA256 = (
    "cfadc53c2d5d277fb711715606977c632e9792852ea5669e8278f7b0ce4ae371"
)
FNP_REFERENCE = BASE / "summaries/exact_baseline_fnp_median/fnp_median.csv"
EXPECTED_FNP_REFERENCE_SHA256 = (
    "596265ecdbce156b549883d86d92799aa1902a5e5c7440cae95a9fb97505ec04"
)
PROJECT_ROOT = BASE.parents[1]
IMPLEMENTATION_FILES = {
    "fit_runner": (
        BASE.parent
        / "high_qt_direct_production_benchmark/experimental_unitary_transition/"
          "scripts/run_production_fnp_stability_control.py"
    ),
    "film_trainer": (
        PROJECT_ROOT / "v21_smoothedA_tail_candidate/"
        "train_bt_dnn_v21_smoothedA_tail.py"
    ),
    "fnp_refit": (
        BASE.parent
        / "high_qt_direct_production_benchmark/experimental_unitary_transition/"
          "scripts/run_differentiable_fnp_refit.py"
    ),
}
EXPECTED_IMPLEMENTATION_SHA256 = {
    "fit_runner": (
        "2e8093c90fccf76db1541c0ed45d3c545b87522356d5433cfb15893887e00c02"
    ),
    "film_trainer": (
        "79944619e3507bbbdef706c327ac2e108ab287ad52152f3a645ffaa2b365bae4"
    ),
    "fnp_refit": (
        "7c186242058f799cda1fc402b9068c261379a4fda77ddb5c6eb836159e1834f3"
    ),
}
INCUMBENT_ID = "empirical_reference_lambda1_b0p1_2p0_full24"
INCUMBENT_RECORD = (
    "summaries/champion_registry/"
    "empirical_reference_lambda1_b0p1_2p0_full24.json"
)
LOCKED_INCUMBENT_WIDTHS = {
    "u": 0.11772613918747582,
    "d": 0.12490071924111977,
}
START_SEEDS = list(range(303, 327))
REPLICA_SEEDS = list(range(1001, 1051))
REQUIRED_ORDER = [
    "finish and provenance-audit all 24 candidate starts",
    "train the selected candidate central through its full requested horizon",
    "finish and provenance-audit all 50 candidate experimental replicas",
    "construct the 1200-member centered-log-FNP start-by-replica ensemble",
    "render candidate Fig. 2 and Fig. 6 plus explicit candidate-versus-incumbent comparisons",
    "evaluate width improvement separately for u and d with the finite-ensemble allowance",
    "promote or reject the candidate",
    "only then decide whether another controlled trial is scientifically warranted",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_number(value: object, expected: float) -> bool:
    try:
        observed = float(value)
    except (TypeError, ValueError):
        return False
    return (math.isfinite(observed)
            and math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12))


def explicit_bool(value: object, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise RuntimeError(f"{label} is not an explicit boolean")


def fixed_implementation_binding() -> dict:
    """Return the canonical path/hash fields carried by pipeline evidence."""
    return {
        "fixed_implementation_files": {
            role: str(path.resolve())
            for role, path in IMPLEMENTATION_FILES.items()
        },
        "fixed_implementation_sha256": dict(EXPECTED_IMPLEMENTATION_SHA256),
    }


def require_fixed_implementation_binding(
        payload: dict, label: str = "evidence") -> None:
    """Require an evidence object to carry the exact implementation binding."""
    paths = payload.get("fixed_implementation_files")
    hashes = payload.get("fixed_implementation_sha256")
    if (not isinstance(paths, dict) or set(paths) != set(IMPLEMENTATION_FILES)
            or not isinstance(hashes, dict)
            or hashes != EXPECTED_IMPLEMENTATION_SHA256):
        raise RuntimeError(f"{label} lacks the fixed implementation hashes")
    for role, canonical in IMPLEMENTATION_FILES.items():
        try:
            observed = Path(paths[role]).resolve()
        except (TypeError, ValueError, OSError) as error:
            raise RuntimeError(
                f"{label} has an invalid implementation path for {role}"
            ) from error
        if observed != canonical.resolve():
            raise RuntimeError(
                f"{label} implementation path changed for {role}: {observed}")


def validate_fixed_challenger_protocol(
        path: Path = PROTOCOL,
        *, expected_sha256: str = EXPECTED_SHA256,
        fnp_reference: Path = FNP_REFERENCE,
        expected_fnp_reference_sha256: str = EXPECTED_FNP_REFERENCE_SHA256,
        implementation_files: dict[str, Path] | None = None,
        expected_implementation_sha256: dict[str, str] | None = None,
        require_canonical_paths: bool = True) -> tuple[dict, str]:
    """Return the validated manifest and digest or raise fail-closed.

    The empirical FNP reference and the three external Python implementations
    are decision-critical inputs to the objective. They existed before the
    fixed protocol was registered, so their pre-result digests are pinned here
    without rewriting the already locked JSON manifest.
    """
    path = Path(path)
    fnp_reference = Path(fnp_reference)
    implementation_files = (
        dict(IMPLEMENTATION_FILES) if implementation_files is None
        else {str(role): Path(value)
              for role, value in implementation_files.items()}
    )
    expected_implementation_sha256 = (
        dict(EXPECTED_IMPLEMENTATION_SHA256)
        if expected_implementation_sha256 is None
        else dict(expected_implementation_sha256)
    )
    if require_canonical_paths and path.resolve() != PROTOCOL.resolve():
        raise RuntimeError(
            f"fixed-challenger protocol path changed: {path}")
    if (require_canonical_paths
            and fnp_reference.resolve() != FNP_REFERENCE.resolve()):
        raise RuntimeError(
            f"fixed empirical FNP reference path changed: {fnp_reference}")
    if (set(implementation_files) != set(IMPLEMENTATION_FILES)
            or set(expected_implementation_sha256)
                != set(EXPECTED_IMPLEMENTATION_SHA256)):
        raise RuntimeError("fixed implementation roles changed")
    if require_canonical_paths:
        for role, canonical in IMPLEMENTATION_FILES.items():
            if implementation_files[role].resolve() != canonical.resolve():
                raise RuntimeError(
                    f"fixed implementation path changed for {role}: "
                    f"{implementation_files[role]}")
        if expected_implementation_sha256 != EXPECTED_IMPLEMENTATION_SHA256:
            raise RuntimeError("fixed implementation SHA256 registry changed")
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"fixed-challenger protocol is missing: {path}")
    observed_sha256 = sha256(path)
    if observed_sha256 != expected_sha256:
        raise RuntimeError(
            "fixed-challenger protocol SHA256 changed: "
            f"{observed_sha256} != {expected_sha256}")
    if not fnp_reference.is_file() or fnp_reference.stat().st_size == 0:
        raise RuntimeError(
            f"fixed empirical FNP reference is missing: {fnp_reference}")
    observed_reference_sha256 = sha256(fnp_reference)
    if observed_reference_sha256 != expected_fnp_reference_sha256:
        raise RuntimeError(
            "fixed empirical FNP reference SHA256 changed: "
            f"{observed_reference_sha256} != {expected_fnp_reference_sha256}")
    for role, implementation_path in implementation_files.items():
        if (not implementation_path.is_file()
                or implementation_path.stat().st_size == 0):
            raise RuntimeError(
                f"fixed implementation is missing for {role}: "
                f"{implementation_path}")
        observed_implementation_sha256 = sha256(implementation_path)
        if (observed_implementation_sha256
                != expected_implementation_sha256[role]):
            raise RuntimeError(
                f"fixed implementation SHA256 changed for {role}: "
                f"{observed_implementation_sha256} != "
                f"{expected_implementation_sha256[role]}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError("fixed-challenger protocol is not a JSON object")
    candidate = payload.get("candidate")
    incumbent = payload.get("incumbent")
    if not isinstance(candidate, dict) or not isinstance(incumbent, dict):
        raise RuntimeError("fixed-challenger protocol lacks candidate/incumbent objects")
    widths = incumbent.get("strict_max_active_relative_full_width")
    valid = (
        payload.get("status") == "locked_before_complete_candidate_evidence"
        and candidate.get("model") == "FiLM production-objective diagnostic"
        and exact_number(candidate.get("reference_distance_lambda"), 600.0)
        and exact_number(candidate.get(
            "reference_distance_b_min_GeV_inverse"), 0.1)
        and exact_number(candidate.get(
            "reference_distance_b_max_GeV_inverse"), 4.0)
        and exact_number(candidate.get("fit_quality_barrier_strength"), 100.0)
        and int(candidate.get("fit_quality_barrier_power", -1)) == 2
        and candidate.get("start_seeds") == START_SEEDS
        and int(candidate.get("required_start_count", -1)) == len(START_SEEDS)
        and int(candidate.get("central_requested_capacity_horizon", -1))
            == 300_000
        and candidate.get("experimental_replica_seeds") == REPLICA_SEEDS
        and int(candidate.get("required_experimental_replica_count", -1))
            == len(REPLICA_SEEDS)
        and incumbent.get("champion_id") == INCUMBENT_ID
        and incumbent.get("record") == INCUMBENT_RECORD
        and isinstance(widths, dict)
        and set(widths) == set(LOCKED_INCUMBENT_WIDTHS)
        and all(exact_number(widths[key], expected)
                for key, expected in LOCKED_INCUMBENT_WIDTHS.items())
        and incumbent.get("status_during_challenge") == "unchanged incumbent"
        and payload.get("required_order") == REQUIRED_ORDER
        and isinstance(payload.get("completion_semantics"), str)
        and bool(payload.get("completion_semantics", "").strip())
        and isinstance(payload.get("promotion_semantics"), str)
        and bool(payload.get("promotion_semantics", "").strip())
        and isinstance(payload.get("comparison_caveat"), str)
        and bool(payload.get("comparison_caveat", "").strip())
        and not explicit_bool(payload.get("other_constraint_authorized_before_completion"),
                              "other_constraint_authorized_before_completion")
        and not explicit_bool(payload.get("production_sources_modified"),
                              "production_sources_modified")
    )
    if not valid:
        raise RuntimeError("fixed-challenger protocol schema/value validation failed")
    return payload, observed_sha256


if __name__ == "__main__":
    manifest, digest = validate_fixed_challenger_protocol()
    print(json.dumps({
        "status": "pass",
        "manifest": str(PROTOCOL),
        "manifest_sha256": digest,
        "fixed_fnp_reference": str(FNP_REFERENCE),
        "fixed_fnp_reference_sha256": EXPECTED_FNP_REFERENCE_SHA256,
        **fixed_implementation_binding(),
        "required_start_count": manifest["candidate"]["required_start_count"],
        "required_experimental_replica_count": manifest["candidate"][
            "required_experimental_replica_count"],
        "production_sources_modified": False,
    }, indent=2))
