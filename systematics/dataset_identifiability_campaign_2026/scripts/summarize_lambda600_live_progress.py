#!/usr/bin/env python3
"""Publish a read-only, explicitly provisional lambda=600 progress snapshot.

The controller may append ``runs.csv`` while this observer is running.  A
snapshot is therefore accepted only when all core input bytes are unchanged
across collection.  This script writes exactly one campaign-local artifact:
``summaries/lambda600_live_progress/summary.json``.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Callable

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
OUTPUTS = BASE / "outputs"
FULL24 = BASE / "summaries/replica_robust_reference_full24"
FULL24_SUMMARY = FULL24 / "summary.json"
RUNS = FULL24 / "runs.csv"
SOURCES = BASE / "summaries/selected_reference_method_full24/summary.json"
PROTOCOL = BASE / "manifests/lambda600_fixed_challenger_protocol.json"
TARGET = BASE / "summaries/lambda600_live_progress/summary.json"
REGIONS = (1.0, 2.0, 3.0, 4.0)
READ_ATTEMPTS = 5
READ_RETRY_SECONDS = 0.20
SEED_PATTERN = re.compile(r"(?:^|_)s(\d+)(?:_|$)")


class TransientSnapshotError(RuntimeError):
    """A live evidence file changed or was temporarily incomplete."""


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def truth(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return bool(value)


def parse_json(raw: bytes, label: str) -> dict:
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TransientSnapshotError(f"transient invalid {label}") from error
    if not isinstance(payload, dict):
        raise TransientSnapshotError(f"{label} is not a JSON object")
    return payload


def read_core_bytes(paths: tuple[Path, ...]) -> dict[Path, bytes]:
    try:
        return {path: path.read_bytes() for path in paths}
    except OSError as error:
        raise TransientSnapshotError(
            f"could not read live evidence: {type(error).__name__}: {error}") from error


def seed_from_tag(tag: object) -> int | None:
    match = SEED_PATTERN.search(str(tag))
    return int(match.group(1)) if match else None


def curve(outputs: Path, tag: str) -> tuple[np.ndarray, np.ndarray]:
    path = outputs / tag / "fnp_grid.csv"
    try:
        frame = pd.read_csv(BytesIO(path.read_bytes()))
        selected = frame.loc[
            np.isclose(frame["x"], 0.1) & frame["bT"].between(0.1, 4.0)
        ].sort_values("bT")
    except (OSError, KeyError, pd.errors.ParserError,
            pd.errors.EmptyDataError) as error:
        raise TransientSnapshotError(
            f"checkpoint curve is temporarily unreadable: {path}") from error
    if len(selected) < 2:
        raise TransientSnapshotError(f"checkpoint curve is incomplete: {path}")
    b = selected["bT"].to_numpy(float)
    values = selected["F_NP"].to_numpy(float)
    if not np.all(np.isfinite(b)) or not np.all(np.isfinite(values)):
        raise TransientSnapshotError(f"checkpoint curve is nonfinite: {path}")
    return b, values


def regional_width(b: np.ndarray, values: np.ndarray) -> dict[str, float]:
    result = {}
    for limit in REGIONS:
        mask = b <= limit
        if not np.any(mask):
            raise TransientSnapshotError(
                f"curve has no samples at bT <= {limit:g}")
        result[f"bT_le_{limit:g}"] = float(np.max(values[mask]))
    return result


def terminal_tags(manifest: dict) -> list[str]:
    tags = manifest.get("endpoint_tags")
    if tags is None:
        tags = manifest.get("endpoint_tags_so_far", [])
    if not isinstance(tags, list) or any(not str(tag) for tag in tags):
        raise TransientSnapshotError("full24 endpoint tag list is invalid")
    return [str(tag) for tag in tags]


def derive_progress(manifest: dict, runs: pd.DataFrame,
                    expected_seeds: list[int]) -> dict:
    if "seed" not in runs or "cumulative_lbfgs_iterations" not in runs:
        raise TransientSnapshotError("runs.csv lacks progress columns")
    try:
        started = sorted(set(runs["seed"].dropna().astype(int)))
        # In-progress manifests expose completed_member_count, while the
        # terminal full24 verifier intentionally uses member_count for its
        # exact-24 coverage.  Treat both schemas explicitly; otherwise a
        # valid terminal verification_failed manifest would look inconsistent
        # to this non-decisional observer forever.
        if "completed_member_count" in manifest:
            completed_count = int(manifest["completed_member_count"])
        elif str(manifest.get("status")) in {"complete", "verification_failed"}:
            completed_count = int(manifest.get("member_count", 0))
        else:
            completed_count = 0
    except (TypeError, ValueError) as error:
        raise TransientSnapshotError("invalid live member counts") from error
    completed = sorted({seed for seed in map(seed_from_tag, terminal_tags(manifest))
                        if seed is not None})
    expected = set(expected_seeds)
    if (not set(started).issubset(expected)
            or not set(completed).issubset(expected)
            or completed_count != len(completed)
            or not set(completed).issubset(started)):
        raise TransientSnapshotError(
            "full24 summary and runs.csv are transiently inconsistent")
    incomplete_started = sorted(set(started) - set(completed))
    unstarted = sorted(expected - set(started))
    return {
        "required_member_count": len(expected_seeds),
        "started_member_count": len(started),
        "started_seeds": started,
        "completed_terminal_member_count": completed_count,
        "completed_terminal_seeds": completed,
        "active_or_incomplete_seeds": incomplete_started,
        "active_seed": incomplete_started[0]
            if len(incomplete_started) == 1 else None,
        "unstarted_seeds": unstarted,
    }


def build_payload(raw: dict[Path, bytes], *, outputs: Path,
                  now: datetime | None = None) -> dict:
    protocol = parse_json(raw[PROTOCOL], "locked protocol")
    manifest = parse_json(raw[FULL24_SUMMARY], "authoritative full24 summary")
    source_summary = parse_json(raw[SOURCES], "source-start summary")
    try:
        runs = pd.read_csv(BytesIO(raw[RUNS]))
    except (pd.errors.ParserError, pd.errors.EmptyDataError,
            UnicodeDecodeError) as error:
        raise TransientSnapshotError("transient invalid runs.csv") from error
    expected_seeds = [int(seed) for seed in protocol["candidate"]["start_seeds"]]
    progress = derive_progress(manifest, runs, expected_seeds)
    source_tags = source_summary.get("endpoint_tags")
    if not isinstance(source_tags, list) or len(source_tags) != len(expected_seeds):
        raise TransientSnapshotError("source-start endpoint coverage is incomplete")
    source_by_seed = dict(zip(expected_seeds, map(str, source_tags), strict=True))
    latest_rows = runs.sort_values("cumulative_lbfgs_iterations").groupby(
        "seed", as_index=False).tail(1).sort_values("seed")
    members: list[dict] = []
    latest_curves: list[np.ndarray] = []
    common_b: np.ndarray | None = None
    for row in latest_rows.itertuples(index=False):
        seed = int(row.seed)
        b, source = curve(outputs, source_by_seed[seed])
        latest_b, latest = curve(outputs, str(row.tag))
        if (b.shape != latest_b.shape or not np.allclose(
                b, latest_b, rtol=0.0, atol=1e-12)):
            raise TransientSnapshotError("source/latest curve grids differ")
        if common_b is not None and (b.shape != common_b.shape or not np.allclose(
                b, common_b, rtol=0.0, atol=1e-12)):
            raise TransientSnapshotError("member curve grids differ")
        common_b = b
        displacement = np.abs(latest - source) / np.maximum(source, 0.05)
        members.append({
            "seed": seed,
            "latest_tag": str(row.tag),
            "requested_cumulative_lbfgs_capacity": int(
                row.cumulative_lbfgs_iterations),
            "latest_block_fnp_drift": float(row.fnp_drift_from_previous_chunk),
            "eligible_post_200k_anchor": truth(
                row.eligible_post_mandatory_confirmation),
            "tested_stationarity_window_anchor_iterations": (
                None if not hasattr(row, "stationarity_window_anchor_iterations")
                or pd.isna(row.stationarity_window_anchor_iterations)
                else int(row.stationarity_window_anchor_iterations)),
            "active_stationarity_window_anchor_iterations": (
                None if not hasattr(row, "next_stationarity_window_anchor_iterations")
                or pd.isna(row.next_stationarity_window_anchor_iterations)
                else int(row.next_stationarity_window_anchor_iterations)),
            "fnp_drift_from_tested_stationarity_anchor": (
                None if pd.isna(row.post_mandatory_window_fnp_drift)
                else float(row.post_mandatory_window_fnp_drift)),
            "consecutive_quiet_post_anchor_blocks": int(
                row.consecutive_quiet_blocks),
            "required_consecutive_quiet_post_anchor_blocks": 10,
            "sensitivity_confirmation_triggered": truth(
                row.sensitivity_confirmation_triggered),
            "wholly_subsequent_quiet_blocks_after_sensitivity_trigger": int(
                row.fresh_quiet_blocks_after_sensitivity_trigger),
            "effective_confirmation_progress": (
                f"{int(row.fresh_quiet_blocks_after_sensitivity_trigger)}/10"
                if truth(row.sensitivity_confirmation_triggered)
                else f"{int(row.consecutive_quiet_blocks)}/10"),
            "unpenalized_total_chi2": float(row.unpenalized_total_chi2),
            "stationarity_and_fit_pass": truth(row.stationarity_and_fit_pass),
            "cumulative_displacement_from_source": regional_width(
                b, displacement),
        })
        latest_curves.append(latest)
    provisional_ranges = None
    if len(latest_curves) >= 2 and common_b is not None:
        curves = np.asarray(latest_curves)
        center = np.median(curves, axis=0)
        relative_range = ((curves.max(axis=0) - curves.min(axis=0)) /
                          np.maximum(center, 0.05))
        provisional_ranges = regional_width(common_b, relative_range)
    instant = (datetime.now().astimezone() if now is None
               else now.astimezone())
    payload = {
        "status": "provisional_live_diagnostic_not_candidate_result",
        "generated_at_utc": instant.astimezone(timezone.utc).isoformat(),
        "generated_at_local": instant.isoformat(),
        "local_timezone": str(instant.tzinfo),
        "locked_protocol": str(PROTOCOL),
        "locked_protocol_sha256": sha256_bytes(raw[PROTOCOL]),
        "authoritative_full24_summary": str(FULL24_SUMMARY),
        "authoritative_full24_summary_sha256": sha256_bytes(
            raw[FULL24_SUMMARY]),
        "authoritative_full24_status": manifest.get("status"),
        **progress,
        "members_with_checkpoints": members,
        "provisional_latest_checkpoint_fnp_full_range": provisional_ranges,
        "provisional": True,
        "promotable": False,
        "warning": (
            "PROVISIONAL AND NON-PROMOTABLE: live checkpoints are an incomplete "
            "distribution and must never be used for promotion, rejection, or "
            "a final uncertainty band. Use the terminal audited full24 summary."
        ),
        "read_only_fit_evidence": True,
        "production_sources_modified": False,
    }
    return payload


def collect_consistent_payload(
        *, outputs: Path = OUTPUTS, attempts: int = READ_ATTEMPTS,
        retry_seconds: float = READ_RETRY_SECONDS,
        sleeper: Callable[[float], None] = time.sleep,
        now: datetime | None = None) -> dict:
    paths = (PROTOCOL, FULL24_SUMMARY, RUNS, SOURCES)
    errors: list[str] = []
    for attempt in range(1, attempts + 1):
        try:
            before = read_core_bytes(paths)
            payload = build_payload(before, outputs=outputs, now=now)
            after = read_core_bytes(paths)
            if before != after:
                raise TransientSnapshotError(
                    "live evidence changed while the snapshot was collected")
            payload["consistent_read_attempt"] = attempt
            payload["consistent_read_max_attempts"] = attempts
            return payload
        except (TransientSnapshotError, OSError, KeyError, TypeError,
                ValueError, AttributeError) as error:
            errors.append(f"attempt {attempt}: {type(error).__name__}: {error}")
            if attempt < attempts:
                sleeper(retry_seconds)
    raise RuntimeError(
        "could not obtain a consistent live lambda600 snapshot after "
        f"{attempts} attempts; " + " | ".join(errors))


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(
            path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def main() -> None:
    payload = collect_consistent_payload()
    atomic_write_json(TARGET, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
