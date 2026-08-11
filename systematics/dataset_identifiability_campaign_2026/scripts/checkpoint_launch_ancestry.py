#!/usr/bin/env python3
"""Durable launch-time ancestry receipts for cache-backed fit checkpoints.

The frozen fit runner records the parent model-state path but not the paired
normalization path or either input's content hash.  This campaign-local helper
therefore writes an immutable receipt *before* an optimizer child is launched.
Cached reuse is allowed only when the receipt still matches the exact current
parent bytes and the exact command that would be launched now.

Receipts live outside fit output directories so the external runner cannot
replace them.  A per-child advisory lock serializes the check/launch/publish
transaction used by campaign supervisors.
"""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path

from fixed_challenger_protocol import (
    EXPECTED_FNP_REFERENCE_SHA256,
    FNP_REFERENCE as FIXED_FNP_REFERENCE,
    PROTOCOL as FIXED_PROTOCOL,
    fixed_implementation_binding,
    validate_fixed_challenger_protocol,
)


SCHEMA_VERSION = 1
RECEIPT_STATUS = "prepared_before_optimizer_launch"


@lru_cache(maxsize=1)
def _fixed_receipt_binding() -> tuple[str, dict]:
    _, protocol_hash = validate_fixed_challenger_protocol()
    return protocol_hash, fixed_implementation_binding()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8")


def canonical_json_sha256(payload: object) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def receipt_path(receipt_root: Path, child_tag: str) -> Path:
    if not child_tag or Path(child_tag).name != child_tag:
        raise RuntimeError(f"invalid checkpoint child tag: {child_tag!r}")
    return Path(receipt_root) / f"{child_tag}.json"


def build_continuation_command(
        *, python: Path, runner: Path, seed: int,
        source_production: Path, w_grid: Path, output_root: Path,
        child_tag: str, parent_state: Path, parent_norms: Path,
        reference_strength: float, reference_csv: Path,
        reference_bmin: float, reference_bmax: float,
        barrier_strength: float, barrier_power: int,
        barrier_ceiling: float | None,
        replica_seed: int | None) -> list[str]:
    """Construct the one canonical frozen-runner continuation command."""
    command = [
        str(Path(python)), str(Path(runner)), "--seed", str(int(seed)),
        "--source-production", str(Path(source_production)),
        "--w-grid", str(Path(w_grid)),
        "--output-root", str(Path(output_root)), "--tag", str(child_tag),
        "--initial-state", str(Path(parent_state)),
        "--initial-norms", str(Path(parent_norms)),
        "--max-epochs", "0", "--min-epochs", "0",
        "--plateau-patience", "0", "--lbfgs-max-iter", "5000",
        "--float64",
        "--lambda-fnp-reference-distance", str(float(reference_strength)),
        "--fnp-reference-distance-csv", str(Path(reference_csv)),
        "--fnp-reference-distance-bmin", f"{float(reference_bmin):.2f}",
        "--fnp-reference-distance-bmax", str(float(reference_bmax)),
    ]
    if float(barrier_strength) > 0.0:
        if barrier_ceiling is None:
            raise RuntimeError("a positive fit-quality barrier needs a ceiling")
        command.extend([
            "--fit-quality-ceiling-total-chi2", str(float(barrier_ceiling)),
            "--lambda-fit-quality-barrier", str(float(barrier_strength)),
            "--fit-quality-barrier-power", str(int(barrier_power)),
        ])
    elif barrier_ceiling is not None:
        raise RuntimeError("a zero fit-quality barrier cannot have a ceiling")
    if replica_seed is not None:
        command.extend(["--replica-seed", str(int(replica_seed))])
    return command


def build_launch_receipt(
        *, receipt_root: Path, child_output: Path, child_tag: str,
        parent_state: Path, parent_norms: Path, fit_seed: int,
        replica_seed: int | None, command: list[str],
        reference_strength: float, reference_bmin: float,
        reference_bmax: float, barrier_strength: float,
        barrier_power: int, barrier_ceiling: float | None) -> dict:
    """Build expected evidence from current parent bytes without writing it."""
    protocol_hash, binding = _fixed_receipt_binding()
    parent_state = Path(parent_state)
    parent_norms = Path(parent_norms)
    for label, path in (("parent state", parent_state),
                        ("parent norms", parent_norms)):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"{label} is missing or empty: {path}")
    output = Path(child_output).resolve()
    if output.name != child_tag:
        raise RuntimeError("child output directory and child tag disagree")
    command = [str(value) for value in command]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": RECEIPT_STATUS,
        "receipt_path": str(receipt_path(receipt_root, child_tag).resolve()),
        "child_tag": child_tag,
        "child_output_directory": str(output),
        "fit_seed": int(fit_seed),
        "replica_seed": (None if replica_seed is None else int(replica_seed)),
        "parent_state": {
            "path": str(parent_state.resolve()),
            "sha256": sha256(parent_state),
        },
        "parent_norms": {
            "path": str(parent_norms.resolve()),
            "sha256": sha256(parent_norms),
        },
        "continuation_protocol": {
            "float64": True,
            "max_epochs": 0,
            "min_epochs": 0,
            "plateau_patience": 0,
            "lbfgs_max_iter": 5000,
            "reference_strength": float(reference_strength),
            "reference_b_min": float(reference_bmin),
            "reference_b_max": float(reference_bmax),
            "reference_csv": str(FIXED_FNP_REFERENCE.resolve()),
            "reference_csv_sha256": EXPECTED_FNP_REFERENCE_SHA256,
            "fit_quality_barrier_strength": float(barrier_strength),
            "fit_quality_barrier_power": int(barrier_power),
            "fit_quality_barrier_ceiling_total_chi2": (
                None if barrier_ceiling is None else float(barrier_ceiling)),
        },
        "fixed_challenger_protocol": str(FIXED_PROTOCOL.resolve()),
        "fixed_challenger_protocol_sha256": protocol_hash,
        **binding,
        "argv": command,
        "argv_sha256": canonical_json_sha256(command),
    }
    return payload


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(
        path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def immutable_create_or_validate_json(path: Path, payload: dict) -> None:
    """Atomically create an immutable JSON generation, or require equality."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    expected_text = json.dumps(payload, indent=2) + "\n"
    if path.exists():
        try:
            observed = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"immutable JSON is unreadable: {path}") from error
        if observed != payload:
            raise RuntimeError(f"immutable JSON content mismatch: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(expected_text)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            try:
                observed = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as error:
                raise RuntimeError(
                    f"concurrent immutable JSON is unreadable: {path}") from error
            if observed != payload:
                raise RuntimeError(
                    f"concurrent immutable JSON content mismatch: {path}")
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def validate_launch_receipt(
        receipt_root: Path, child_tag: str, expected: dict) -> dict:
    path = receipt_path(receipt_root, child_tag)
    try:
        observed = json.loads(path.read_text())
    except FileNotFoundError as error:
        raise RuntimeError(f"checkpoint launch receipt is missing: {path}") from error
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"checkpoint launch receipt is unreadable: {path}") from error
    if observed != expected:
        raise RuntimeError(
            f"checkpoint launch receipt content/parent mismatch: {path}")
    return {
        "kind": "launch_time_content_receipt",
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "parent_state_path": expected["parent_state"]["path"],
        "parent_state_sha256": expected["parent_state"]["sha256"],
        "parent_norms_path": expected["parent_norms"]["path"],
        "parent_norms_sha256": expected["parent_norms"]["sha256"],
        "argv_sha256": expected["argv_sha256"],
    }


def prepare_launch_receipt(
        receipt_root: Path, child_output: Path, child_tag: str,
        expected: dict) -> dict:
    """Publish before launch; reject a pre-existing child without a receipt."""
    path = receipt_path(receipt_root, child_tag)
    output = Path(child_output)
    if not path.exists() and output.exists() and any(output.iterdir()):
        raise RuntimeError(
            "checkpoint artifacts exist without a launch-time receipt: "
            f"{output}")
    immutable_create_or_validate_json(path, expected)
    return validate_launch_receipt(receipt_root, child_tag, expected)


def classify_launch_ancestry(
        receipt_root: Path, child_output: Path, child_tag: str,
        expected: dict, *, allow_legacy_without_receipt: bool) -> dict:
    """Validate a receipt or explicitly classify a pre-receipt legacy child."""
    path = receipt_path(receipt_root, child_tag)
    if path.exists():
        return validate_launch_receipt(receipt_root, child_tag, expected)
    if not allow_legacy_without_receipt:
        raise RuntimeError(f"checkpoint launch receipt is missing: {path}")
    output = Path(child_output)
    if not output.exists() or not any(output.iterdir()):
        raise RuntimeError(
            f"cannot classify an absent checkpoint as legacy: {output}")
    return {
        "kind": "legacy_pre_receipt_path_only_requires_contemporaneous_seal",
        "path": None,
        "sha256": None,
        "parent_state_path": expected["parent_state"]["path"],
        "parent_state_sha256": expected["parent_state"]["sha256"],
        "parent_norms_path": expected["parent_norms"]["path"],
        "parent_norms_sha256": expected["parent_norms"]["sha256"],
        "argv_sha256": expected["argv_sha256"],
    }


@contextmanager
def exclusive_checkpoint_launch(receipt_root: Path, child_tag: str):
    """Serialize cache validation and launch for one deterministic child tag."""
    root = Path(receipt_root)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / f".{child_tag}.launch.lock"
    stream = lock_path.open("a+")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()
