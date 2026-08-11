#!/usr/bin/env python3
"""Finish the complete lambda600-vs-fixed-lambda1-incumbent comparison."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

from alternate_lambda_authorization import (
    FINAL_REPORT_MARKDOWN,
    FINAL_REPORT_SUMMARY,
    validate_complete_lambda600_comparison,
    validated_prepublication_final_report,
)
from fixed_challenger_protocol import (
    EXPECTED_FNP_REFERENCE_SHA256,
    FNP_REFERENCE as FIXED_FNP_REFERENCE,
    PROTOCOL as FIXED_PROTOCOL,
    fixed_implementation_binding,
    require_fixed_implementation_binding,
    validate_fixed_challenger_protocol,
)
from postfit_tail_transform_validation import (
    validated_postfit_tail_audit,
)
from nested_interaction_and_final_envelope_validation import (
    final_promotion_gate,
    validated_final_directional_envelope,
    validated_nested_interaction,
)
from promote_validated_final_champion import validated_published_promotion


BASE = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/dustin/miniforge3/envs/pdf-fit/bin/python")
SCRIPTS = BASE / "scripts"
SUM = BASE / "summaries"
TARGET = SUM / "lambda600_like_for_like_decision"
INCUMBENT = "champion_registry/empirical_reference_lambda1_b0p1_2p0_full24.json"
LOCK_PATH = TARGET / "continuation.lock"
MAX_RECOVERY_ATTEMPTS = 3
RECOVERY_BACKOFF_SECONDS = (0, 10, 30)
PROCESS_POLL_SECONDS = 30
FULL24_SCRIPT = "verify_replica_robust_reference_full24.py"
FULL24_ARGS = (
    "--strength", "600",
    "--fit-quality-barrier-strength", "100",
    "--fit-quality-barrier-power", "2",
    "--strongest-failing-strength", "562.5",
)
FULL24_TERMINAL_STATUSES = {"complete", "verification_failed"}
REPLICA_TERMINAL_STATUSES = {
    "complete", "complete_with_scientific_failures",
    "central_stationarity_failed", "replica_stationarity_failed",
}
FINAL_SCIENTIFIC_STATUSES = {
    "candidate_promoted_as_new_study_champion",
    "candidate_rejected",
}
TERMINAL_EVIDENCE = SUM / "lambda600_terminal_evidence/summary.json"
START_CHAIN_AUDIT = SUM / "lambda600_start_chain_audit/summary.json"
START_SEAL_ROOT = SUM / "lambda600_start_chain_audit/current_byte_seals"
LAUNCH_RECEIPTS = SUM / "checkpoint_launch_receipts"
TERMINAL_ENDPOINT_FILES = (
    "fit_status.json", "model_state.pt", "dataset_norms.csv", "fnp_grid.csv",
)


def running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    stat = Path(f"/proc/{pid}/stat")
    return not stat.exists() or stat.read_text().split()[2] != "Z"


def load(relative: str) -> dict:
    return json.loads((SUM / relative).read_text())


def load_optional(relative: str) -> tuple[dict | None, str | None]:
    path = SUM / relative
    try:
        payload = json.loads(path.read_text())
    except (FileNotFoundError, OSError, json.JSONDecodeError) as error:
        return None, f"{type(error).__name__}: {error}"
    if not isinstance(payload, dict):
        return None, "summary root is not a JSON object"
    return payload, None


def endpoint_files_ready(tags: list[str], outputs_root: Path) -> bool:
    root = Path(outputs_root)
    try:
        return all(
            tag and Path(tag).name == tag
            and all((root / tag / name).is_file()
                    and (root / tag / name).stat().st_size > 0
                    for name in TERMINAL_ENDPOINT_FILES)
            for tag in tags
        )
    except OSError:
        return False


def full24_terminal_coverage_ready(
        payload: dict, *, outputs_root: Path | None = None) -> bool:
    """Return true only for an exact 24-member terminal start manifest."""
    tags = [str(value) for value in payload.get("endpoint_tags", [])]
    try:
        count = int(payload.get("member_count", len(tags)))
    except (TypeError, ValueError):
        return False
    root = BASE / "outputs" if outputs_root is None else Path(outputs_root)
    return bool(
        count == 24 and len(tags) == 24 and len(set(tags)) == 24
        and endpoint_files_ready(tags, root)
    )


def replica_terminal_coverage_ready(
        payload: dict, *, outputs_root: Path | None = None) -> bool:
    """Return true only after the selected central and all 50 replicas exist.

    Older supervisors could publish an explicit scientific-failure status before
    exhausting the requested replica ensemble.  Such a status is diagnostic,
    not terminal for this fixed comparison, and must trigger deterministic
    cache replay instead of trapping the restarting controller on the same
    partial manifest.
    """
    tags = [str(value) for value in payload.get("replica_endpoint_tags", [])]
    central_tag = str(payload.get("central_endpoint_tag", ""))
    try:
        count = int(payload.get("completed_replica_count", len(tags)))
        central_requested = int(payload.get(
            "central_full_horizon_requested_capacity", -1))
        central_terminal = int(payload.get(
            "central_terminal_requested_capacity", -1))
        central_complete = exact_bool(
            payload.get("central_full_horizon_complete"),
            "central_full_horizon_complete")
    except (TypeError, ValueError, RuntimeError):
        return False
    root = BASE / "outputs" if outputs_root is None else Path(outputs_root)
    return bool(
        central_tag and Path(central_tag).name == central_tag
        and central_complete
        and central_requested == 300_000 and central_terminal == 300_000
        and count == 50 and len(tags) == 50 and len(set(tags)) == 50
        and endpoint_files_ready([central_tag, *tags], root)
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def exclusive_controller_lock():
    """Prevent two self-healing continuations from launching the same work."""
    TARGET.mkdir(parents=True, exist_ok=True)
    stream = LOCK_PATH.open("a+")
    try:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"another lambda600 continuation holds {LOCK_PATH}") from error
        stream.seek(0)
        stream.truncate()
        stream.write(f"pid={os.getpid()}\n")
        stream.flush()
        yield
    finally:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()


def exact_bool(value, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise RuntimeError(f"{label} is not an explicit boolean")


def exact_number(value, expected: float, label: str) -> None:
    if (value is None or not math.isfinite(float(value))
            or not math.isclose(float(value), expected,
                                rel_tol=0.0, abs_tol=1.0e-12)):
        raise RuntimeError(f"{label} does not match the lambda600 prescription")


def require_failed_start_horizon_exhaustion(payload: dict) -> dict[int, int]:
    """Require every nonpassing start to reach the locked 300k capacity."""
    try:
        failed = [int(value) for value in payload.get("failed_seeds", [])]
        capacities = {
            int(seed): int(value) for seed, value in payload.get(
                "failed_terminal_requested_capacity_by_seed", {}).items()
        }
        exhausted = exact_bool(
            payload.get("failed_starts_exhausted_full_requested_horizon"),
            "failed_starts_exhausted_full_requested_horizon",
        )
        requested = int(payload.get(
            "full_requested_capacity_per_nonpassing_start", -1))
    except (TypeError, ValueError, RuntimeError) as error:
        raise RuntimeError(
            "full24 lacks valid failed-start horizon metadata") from error
    if (len(failed) != len(set(failed))
            or not set(failed).issubset(set(range(303, 327)))
            or set(capacities) != set(failed)
            or not exhausted or requested != 300_000
            or any(value != 300_000 for value in capacities.values())):
        raise RuntimeError(
            "every nonpassing lambda600 start must exhaust exactly 300000 "
            "requested LBFGS capacity before the central stage")
    return capacities


def require_full24_launch_ancestry_metadata(payload: dict) -> dict:
    """Require the updated verifier's explicit legacy/receipt accounting.

    The controller that was already live when launch-time receipts were added
    cannot retroactively prove the parent bytes it consumed.  Its deterministic
    cache may still be used only after the updated verifier has snapshotted the
    exact pre-receipt population.  Missing metadata therefore means "repair and
    re-audit", never "assume the old path-only evidence is sufficient".
    """
    try:
        admitted = [str(value) for value in payload[
            "legacy_pre_receipt_admission_tags"]]
        used = [str(value) for value in payload[
            "legacy_pre_receipt_used_tags"]]
        admitted_count = int(payload[
            "legacy_pre_receipt_admission_tag_count"])
        legacy_count = int(payload["legacy_pre_receipt_checkpoint_count"])
        receipt_count = int(payload[
            "launch_time_content_receipt_checkpoint_count"])
        semantics = str(payload["legacy_checkpoint_ancestry_semantics"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            "full24 lacks updated launch-ancestry generation metadata") from error
    if (len(admitted) != len(set(admitted))
            or len(used) != len(set(used))
            or admitted_count != len(admitted)
            or not set(used).issubset(set(admitted))
            or legacy_count != len(used)
            or receipt_count < 0 or legacy_count < 0
            or "does not retroactively prove" not in semantics):
        raise RuntimeError(
            "full24 launch-ancestry generation metadata is inconsistent")
    return {
        "launch_time_content_receipt_checkpoint_count": receipt_count,
        "legacy_pre_receipt_checkpoint_count": legacy_count,
        "legacy_pre_receipt_admission_tags": admitted,
        "legacy_pre_receipt_used_tags": used,
        "legacy_checkpoint_ancestry_semantics": semantics,
    }


def option_value(command: tuple[str, ...], option: str) -> str | None:
    """Return the final explicit value for a command-line option."""
    values = [command[index + 1] for index, token in enumerate(command[:-1])
              if token == option]
    return values[-1] if values else None


def command_has_script(command: tuple[str, ...], script: str) -> bool:
    return any(Path(token).name == script for token in command)


def command_number_is(command: tuple[str, ...], option: str,
                      expected: float) -> bool:
    value = option_value(command, option)
    try:
        return (value is not None and math.isfinite(float(value))
                and math.isclose(float(value), expected,
                                 rel_tol=0.0, abs_tol=1.0e-12))
    except ValueError:
        return False


def fixed_lambda600_optimizer(command: tuple[str, ...],
                              tag_prefixes: tuple[str, ...]) -> bool:
    """Identify only optimizer children using this exact fixed objective."""
    tag = option_value(command, "--tag")
    return (
        command_has_script(command, "run_production_fnp_stability_control.py")
        and tag is not None
        and any(tag.startswith(prefix) for prefix in tag_prefixes)
        and command_number_is(
            command, "--lambda-fnp-reference-distance", 600.0)
        and command_number_is(command, "--fnp-reference-distance-bmax", 4.0)
        and command_number_is(
            command, "--lambda-fit-quality-barrier", 100.0)
        and command_number_is(command, "--fit-quality-barrier-power", 2.0)
    )


def process_matches_stage(stage: str, command: tuple[str, ...]) -> bool:
    if stage == "full24":
        controller = (
            command_has_script(command, FULL24_SCRIPT)
            and command_number_is(command, "--strength", 600.0)
            and command_number_is(
                command, "--fit-quality-barrier-strength", 100.0)
            and command_number_is(
                command, "--fit-quality-barrier-power", 2.0)
            and command_number_is(
                command, "--strongest-failing-strength", 562.5)
        )
        optimizer = fixed_lambda600_optimizer(command, (
            "fullref_replica_robust_lam600_fitbar_p2_mu100_b4_s",
        ))
        return controller or optimizer
    if stage == "central_replicas":
        controller = command_has_script(
            command, "supervise_selected_reference_central_replicas.py")
        optimizer = fixed_lambda600_optimizer(command, (
            "fullref_lam600_fitbar_p2_mu100_b4_central_polish64_",
            "fullref_lam600_fitbar_p2_mu100_b4_replica_r",
        ))
        return controller or optimizer
    raise ValueError(f"unknown recovery stage: {stage}")


def live_process_commands() -> list[tuple[int, tuple[str, ...]]]:
    """Read live process argv without invoking shell process listings."""
    found: list[tuple[int, tuple[str, ...]]] = []
    self_pid = os.getpid()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == self_pid:
            continue
        try:
            stat_tail = (entry / "stat").read_text().rpartition(")")[2].split()
            if not stat_tail or stat_tail[0] == "Z":
                continue
            raw = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, ProcessLookupError, PermissionError,
                OSError):
            continue
        command = tuple(
            token.decode(errors="surrogateescape")
            for token in raw.split(b"\0") if token)
        if command:
            found.append((pid, command))
    return found


def matching_stage_processes(stage: str) -> list[dict]:
    return [
        {"pid": pid, "command": list(command)}
        for pid, command in live_process_commands()
        if process_matches_stage(stage, command)
    ]


def wait_until_stage_quiet(
        stage: str, *,
        process_finder=matching_stage_processes,
        sleeper=time.sleep,
        poll_seconds: float = PROCESS_POLL_SECONDS) -> list[dict]:
    """Wait for matching controllers and optimizer orphans to disappear."""
    observed: dict[int, dict] = {}
    while True:
        matches = process_finder(stage)
        if not matches:
            return [observed[pid] for pid in sorted(observed)]
        for match in matches:
            observed[int(match["pid"])] = match
        sleeper(poll_seconds)


def run(script: str, *, args: tuple[str, ...] = (),
        check: bool = True) -> int:
    return subprocess.run(
        [str(PYTHON), str(SCRIPTS / script), *args], check=check).returncode


def prepare_final_study_report(
        status: str, *, runner=run,
        validator=validated_prepublication_final_report) -> dict:
    """Write and validate the report before the terminal atomic write.

    This is safe to repeat after a crash: the report writer uses atomic,
    deterministic outputs and does not modify the champion registry or frozen
    inputs.
    """
    returncode = runner(
        "write_final_study_report.py",
        args=("--decision-status", status), check=False)
    if returncode != 0:
        raise RuntimeError(
            f"final study report writer exited with status {returncode}")
    summary, summary_hash = validator(status)
    return {
        "status": "pass",
        "summary": str(FINAL_REPORT_SUMMARY),
        "summary_sha256": summary_hash,
        "report": str(FINAL_REPORT_MARKDOWN),
        "report_sha256": summary["artifact_sha256"]["final_report"],
        "outcome": status,
    }


def recover_terminal_summary(
        *, relative: str, terminal_statuses: set[str], stage: str,
        script: str, script_args: tuple[str, ...] = (),
        max_attempts: int = MAX_RECOVERY_ATTEMPTS,
        backoffs: tuple[float, ...] = RECOVERY_BACKOFF_SECONDS,
        summary_loader=load_optional,
        quiescence_waiter=wait_until_stage_quiet,
        runner=run,
        sleeper=time.sleep,
        terminal_evidence_ready=None,
) -> tuple[dict | None, str | None, list[dict]]:
    """Resume a cache-backed controller without duplicating live work.

    A coverage-complete terminal scientific manifest always wins, independent
    of the controller return code. Missing, corrupt, explicitly nonterminal, or
    prematurely scientific summaries are retried with the exact same command.
    """
    history: list[dict] = []
    payload: dict | None = None
    summary_error: str | None = None

    def terminal(payload: dict | None) -> bool:
        if payload is None or str(payload.get("status")) not in terminal_statuses:
            return False
        return (terminal_evidence_ready is None
                or bool(terminal_evidence_ready(payload)))

    for attempt in range(1, max_attempts + 1):
        observed_before = quiescence_waiter(stage)
        payload, summary_error = summary_loader(relative)
        status = (str(payload.get("status"))
                  if payload is not None else None)
        if terminal(payload):
            return payload, summary_error, history

        delay = backoffs[min(attempt - 1, len(backoffs) - 1)]
        if delay > 0:
            sleeper(delay)

        # A replacement controller may have appeared during the backoff.
        # Wait for it, then honor any terminal summary it produced instead of
        # racing it with a duplicate launch.
        observed_during_backoff = quiescence_waiter(stage)
        payload, summary_error = summary_loader(relative)
        status = (str(payload.get("status"))
                  if payload is not None else None)
        if terminal(payload):
            return payload, summary_error, history

        returncode: int | None = None
        launch_error: str | None = None
        try:
            returncode = runner(
                script, args=script_args, check=False)
        except OSError as error:
            launch_error = f"{type(error).__name__}: {error}"

        observed_after = quiescence_waiter(stage)
        payload, summary_error = summary_loader(relative)
        status = (str(payload.get("status"))
                  if payload is not None else None)
        history.append({
            "attempt": attempt,
            "script": script,
            "args": list(script_args),
            "returncode": returncode,
            "launch_error": launch_error,
            "summary_status_after_attempt": status,
            "summary_error_after_attempt": summary_error,
            "matching_processes_observed_before": observed_before,
            "matching_processes_observed_during_backoff":
                observed_during_backoff,
            "matching_processes_observed_after": observed_after,
        })
        if terminal(payload):
            return payload, summary_error, history
    return payload, summary_error, history


def atomic_write_json(path: Path, payload: dict) -> None:
    """Durably publish one complete JSON object for external classifiers."""
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


def write(status: str, stage: str, **details) -> None:
    _, protocol_hash = validate_fixed_challenger_protocol()
    TARGET.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "stage": stage,
        "candidate": {
            "reference_strength": 600.0,
            "reference_bmax": 4.0,
            "fit_quality_barrier_strength": 100.0,
            "fit_quality_barrier_power": 2,
        },
        "fixed_challenger_protocol": str(FIXED_PROTOCOL),
        "fixed_challenger_protocol_sha256": protocol_hash,
        "fixed_fnp_reference": str(FIXED_FNP_REFERENCE),
        "fixed_fnp_reference_sha256": EXPECTED_FNP_REFERENCE_SHA256,
        **details,
        **fixed_implementation_binding(),
        "production_sources_modified": False,
    }
    atomic_write_json(TARGET / "summary.json", payload)


def terminal_decision_already_published() -> bool:
    """Make successful or scientific-rejection completion idempotent."""
    try:
        payload = json.loads((TARGET / "summary.json").read_text())
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return False
    terminal = (
        isinstance(payload, dict)
        and payload.get("stage") == "complete_like_for_like_comparison"
        and payload.get("status") in FINAL_SCIENTIFIC_STATUSES
    )
    if not terminal:
        return False
    try:
        validate_complete_lambda600_comparison()
    except Exception:
        return False
    return True


def require_restartable_controller_exit(
        payload: dict | None = None, *, validator=None) -> None:
    """Allow only a complete scientific decision to exit successfully.

    Technical failures are durably published first by ``continue_like_for_like``.
    Raising here makes the process exit nonzero so the service's
    ``Restart=on-failure`` policy can resume the same cache-backed stage.
    """
    if payload is None:
        payload = json.loads((TARGET / "summary.json").read_text())
    status = str(payload.get("status"))
    stage = str(payload.get("stage"))
    if (status in FINAL_SCIENTIFIC_STATUSES
            and stage == "complete_like_for_like_comparison"):
        check = (validate_complete_lambda600_comparison
                 if validator is None else validator)
        validated, _ = check()
        if validated != payload:
            raise RuntimeError(
                "terminal decision changed during successful-exit validation")
        return
    if status == "technical_failure":
        raise RuntimeError(
            f"retryable lambda600 technical failure at stage {stage}")
    raise RuntimeError(
        "lambda600 continuation returned without a complete scientific "
        f"decision: status={status}, stage={stage}")


def continue_like_for_like(wait_pid: int) -> None:
    write("in_progress", "full24_long_horizon")
    while running(wait_pid):
        time.sleep(30)

    champion, incumbent_error = load_optional(INCUMBENT)
    if (champion is None or champion.get("champion_id")
            != "empirical_reference_lambda1_b0p1_2p0_full24"):
        write("technical_failure", "incumbent_record_invalid",
              incumbent_error=incumbent_error, incumbent_record=champion)
        return
    incumbent = champion["combined_fig6_max_active_relative_full_width"]

    full24_relative = "replica_robust_reference_full24/summary.json"
    # Always enter through the quiescence check, even when the first read
    # appears terminal. A replacement controller may still be live and own a
    # newer deterministic-cache generation.
    starts, full24_summary_error, full24_attempt_history = (
        recover_terminal_summary(
            relative=full24_relative,
            terminal_statuses=FULL24_TERMINAL_STATUSES,
            stage="full24", script=FULL24_SCRIPT,
            script_args=FULL24_ARGS,
            terminal_evidence_ready=full24_terminal_coverage_ready))
    if starts is None:
        write("technical_failure", "full24_recovery_exhausted",
              full24_summary_error=full24_summary_error,
              full24_recovery_attempts=full24_attempt_history,
              incumbent_fig6_widths=incumbent,
              failure_semantics=(
                  "three exact fixed-prescription cache-backed recovery "
                  "attempts did not produce a readable terminal manifest"))
        return
    start_status = starts.get("status")
    if str(start_status) not in FULL24_TERMINAL_STATUSES:
        write("technical_failure", "full24_recovery_exhausted",
              full24=starts, incumbent_fig6_widths=incumbent,
              full24_summary_error=full24_summary_error,
              full24_recovery_attempts=full24_attempt_history,
              failure_semantics=(
                  "three exact fixed-prescription cache-backed recovery "
                  "attempts ended without an explicit complete or "
                  "verification_failed manifest"))
        return
    try:
        require_failed_start_horizon_exhaustion(starts)
        require_full24_launch_ancestry_metadata(starts)
    except RuntimeError as legacy_generation_error:
        # Controllers started before the horizon-exhaustion correction may
        # have published scientific failures at255k merely because fewer than
        # ten blocks remained.  That is incomplete sampling, not a terminal
        # result. Controllers started before launch receipts also lack an
        # explicit snapshot of the only cache entries eligible for the
        # disclosed legacy path. Resume the exact same cache-backed fixed
        # challenger once with the corrected verifier and then re-read the
        # authoritative evidence.
        observed_before = wait_until_stage_quiet("full24")
        repair_returncode = None
        repair_launch_error = None
        try:
            repair_returncode = run(
                FULL24_SCRIPT, args=FULL24_ARGS, check=False)
        except OSError as error:
            repair_launch_error = f"{type(error).__name__}: {error}"
        observed_after = wait_until_stage_quiet("full24")
        starts, full24_summary_error = load_optional(full24_relative)
        full24_attempt_history.append({
            "attempt": "full24_horizon_and_ancestry_generation_repair",
            "script": FULL24_SCRIPT,
            "args": list(FULL24_ARGS),
            "returncode": repair_returncode,
            "launch_error": repair_launch_error,
            "legacy_validation_error": str(legacy_generation_error),
            "summary_status_after_attempt": (
                starts.get("status") if isinstance(starts, dict) else None),
            "summary_error_after_attempt": full24_summary_error,
            "matching_processes_observed_before": observed_before,
            "matching_processes_observed_after": observed_after,
        })
        if starts is None:
            write("technical_failure", "full24_horizon_repair_failed",
                  full24_summary_error=full24_summary_error,
                  full24_recovery_attempts=full24_attempt_history,
                  incumbent_fig6_widths=incumbent,
                  failure_semantics=(
                      "the corrected fixed-prescription verifier did not "
                      "publish a readable post-repair terminal manifest"))
            return
        start_status = starts.get("status")
        if str(start_status) not in FULL24_TERMINAL_STATUSES:
            write("technical_failure", "full24_horizon_repair_failed",
                  full24=starts,
                  full24_summary_error=full24_summary_error,
                  full24_recovery_attempts=full24_attempt_history,
                  incumbent_fig6_widths=incumbent,
                  failure_semantics=(
                      "the corrected fixed-prescription verifier did not "
                      "reach a terminal post-repair manifest"))
            return
    try:
        failed_start_capacities = require_failed_start_horizon_exhaustion(starts)
        start_launch_ancestry = require_full24_launch_ancestry_metadata(starts)
    except RuntimeError as error:
        write("technical_failure", "full24_terminal_evidence_incomplete",
              full24=starts,
              validation_error=f"{type(error).__name__}: {error}",
              full24_recovery_attempts=full24_attempt_history,
              incumbent_fig6_widths=incumbent,
              failure_semantics=(
                  "lambda600 may not advance until every nonpassing start "
                  "has a terminal endpoint at exactly300k requested capacity "
                  "and the updated verifier has classified every cached "
                  "checkpoint as prospective-receipt or explicitly admitted "
                  "pre-receipt legacy evidence"))
        return
    start_tags = [str(value) for value in starts.get("endpoint_tags", [])]
    try:
        start_count = int(starts.get("member_count", len(start_tags)))
    except (TypeError, ValueError) as error:
        write("technical_failure", "full24_manifest_invalid",
              full24=starts, full24_recovery_attempts=full24_attempt_history,
              validation_error=f"{type(error).__name__}: {error}",
              incumbent_fig6_widths=incumbent)
        return
    if (start_count != 24 or len(start_tags) != 24
            or len(set(start_tags)) != 24):
        write("technical_failure", "full24_coverage_incomplete",
              full24=starts, incumbent_fig6_widths=incumbent,
              full24_recovery_attempts=full24_attempt_history,
              failure_semantics=(
                  "lambda600 may not advance until all 24 requested starts "
                  "have unique terminal endpoints"))
        return
    try:
        start_scientific_gate_pass = exact_bool(starts.get(
            "all_starts_fnp_plateaued_and_fit_preserved"),
            "all_starts_fnp_plateaued_and_fit_preserved")
        if ((start_status == "complete") != start_scientific_gate_pass):
            raise RuntimeError(
                "full24 status and scientific gate are inconsistent")
        for key, expected in (
            ("selected_strength", 600.0), ("selected_bmax", 4.0),
            ("fit_quality_barrier_strength", 100.0),
            ("fit_quality_barrier_power", 2.0),
        ):
            exact_number(starts.get(key), expected, f"full24 {key}")
        nonuniqueness = starts["candidate_nonuniqueness_fig6_widths"]
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        write("technical_failure", "full24_manifest_invalid",
              full24=starts, full24_recovery_attempts=full24_attempt_history,
              validation_error=f"{type(error).__name__}: {error}",
              incumbent_fig6_widths=incumbent,
              failure_semantics=(
                  "the terminal scientific status was honored and was not "
                  "retried, but its fixed-prescription manifest failed "
                  "closed validation"))
        return
    # Do not promote or reject from the start-only width.  The declared
    # incumbent comparison uses the complete 24-start x central/50-replica
    # candidate hierarchy, including its resampling-stability allowance. The
    # fixed lambda=1 incumbent was built with the earlier optimizer/replica
    # protocol, so the final audit compares both output bands on one common
    # conservative active mask rather than claiming protocol identity.
    start_audit_returncode = run(
        "audit_lambda600_state_chains.py", args=("--starts-only",),
        check=False)
    if start_audit_returncode != 0:
        write("technical_failure", "precentral_start_ancestry_audit_failed",
              returncode=start_audit_returncode, full24=starts,
              full24_recovery_attempts=full24_attempt_history,
              incumbent_fig6_widths=incumbent,
              failure_semantics=(
                  "the exact 24-start state chains must pass their terminal "
                  "ancestry/content audit before central training begins"))
        return
    try:
        _, fixed_protocol_hash = validate_fixed_challenger_protocol()
        start_audit = json.loads(START_CHAIN_AUDIT.read_text())
        require_fixed_implementation_binding(
            start_audit, "terminal start ancestry manifest")
        start_audit_prescription = start_audit["selected_prescription"]
        start_selection = start_audit["central_initializer_selection"]
        start_seal = start_audit["terminal_start_current_byte_seal"]
        start_seal_path = Path(start_seal["path"])
        launch_receipt_count = int(
            start_seal["launch_time_receipt_checkpoint_count"])
        legacy_checkpoint_count = int(
            start_seal["legacy_pre_receipt_checkpoint_count"])
        sealed_checkpoint_count = int(start_seal["checkpoint_count"])
        exact_number(start_audit_prescription.get(
            "reference_strength"), 600.0,
            "start audit reference_strength")
        exact_number(start_audit_prescription.get(
            "reference_b_max"), 4.0,
            "start audit reference_b_max")
        exact_number(start_audit_prescription.get(
            "fit_quality_barrier_strength"), 100.0,
            "start audit fit_quality_barrier_strength")
        if not (start_audit.get("status") == "pass"
                and int(start_audit.get("lambda300_source_count", -1)) == 24
                and int(start_audit.get("start_chain_count", -1)) == 24
                and int(start_audit_prescription.get(
                    "fit_quality_barrier_power", -1)) == 2
                and int(start_selection.get("member_count", -1)) == 24
                and start_selection.get("ordered_start_tags") == start_tags
                and start_selection.get("selected_tag") in start_tags
                and start_seal.get("status") in {
                    "sealed_current_bytes_with_disclosed_legacy_limit",
                    "sealed_launch_time_receipt_graph",
                }
                and start_seal_path.parent.resolve()
                    == START_SEAL_ROOT.resolve()
                and start_seal_path.is_file()
                and start_seal.get("sha256") == sha256(start_seal_path)
                and launch_receipt_count + legacy_checkpoint_count
                    == sealed_checkpoint_count
                and sealed_checkpoint_count > 0
                and isinstance(start_seal.get(
                    "historical_ancestry_limitation"), str)
                and bool(start_seal.get(
                    "historical_ancestry_limitation", "").strip())
                and exact_bool(start_audit.get(
                    "exact_launch_time_ancestry_proven_for_all_start_checkpoints"),
                    "exact_launch_time_ancestry_proven_for_all_start_checkpoints")
                    == (legacy_checkpoint_count == 0)
                and start_audit.get("legacy_start_ancestry_limitation")
                    == start_seal.get("historical_ancestry_limitation")
                and Path(start_audit.get(
                    "fixed_challenger_protocol", "")).resolve()
                    == FIXED_PROTOCOL.resolve()
                and start_audit.get("fixed_challenger_protocol_sha256")
                    == fixed_protocol_hash
                and Path(start_audit.get(
                    "fixed_fnp_reference", "")).resolve()
                    == FIXED_FNP_REFERENCE.resolve()
                and start_audit.get("fixed_fnp_reference_sha256")
                    == EXPECTED_FNP_REFERENCE_SHA256
                and start_audit.get("production_sources_modified") is False):
            raise RuntimeError("terminal start ancestry manifest is invalid")
    except (FileNotFoundError, OSError, json.JSONDecodeError, KeyError,
            TypeError, ValueError, RuntimeError) as error:
        write("technical_failure", "precentral_start_ancestry_audit_invalid",
              full24=starts,
              validation_error=f"{type(error).__name__}: {error}",
              full24_recovery_attempts=full24_attempt_history,
              incumbent_fig6_widths=incumbent)
        return
    start_chain_evidence = {
        "status": "pass",
        "manifest": str(START_CHAIN_AUDIT),
        "manifest_sha256": sha256(START_CHAIN_AUDIT),
        "central_initializer_selection": start_selection,
        "terminal_start_current_byte_seal": start_seal,
        "failed_terminal_requested_capacity_by_seed":
            failed_start_capacities,
    }

    write("in_progress", "central_plus_50_replicas", full24=starts,
          terminal_start_ancestry=start_chain_evidence,
          full24_recovery_attempts=full24_attempt_history,
          start_scientific_gate_pass=start_scientific_gate_pass,
          candidate_start_only_fig6_widths=nonuniqueness,
          incumbent_fig6_widths=incumbent)

    replicas, replica_summary_error, replica_attempt_history = (
        recover_terminal_summary(
            relative="selected_reference_central_replicas/summary.json",
            terminal_statuses=REPLICA_TERMINAL_STATUSES,
            stage="central_replicas",
            script="supervise_selected_reference_central_replicas.py",
            terminal_evidence_ready=replica_terminal_coverage_ready))
    if replicas is None:
        write("technical_failure", "central_replica_recovery_exhausted",
              full24=starts,
              terminal_start_ancestry=start_chain_evidence,
              full24_recovery_attempts=full24_attempt_history,
              replica_summary_error=replica_summary_error,
              replica_recovery_attempts=replica_attempt_history,
              incumbent_fig6_widths=incumbent,
              failure_semantics=(
                  "three exact fixed-prescription cache-backed supervisor "
                  "attempts did not produce a readable terminal manifest"))
        return
    replica_status = replicas.get("status")
    replica_tags = [str(value) for value in replicas.get(
        "replica_endpoint_tags", [])]
    try:
        completed_replicas = int(replicas.get(
            "completed_replica_count", len(replica_tags)))
    except (TypeError, ValueError) as error:
        write("technical_failure", "central_replica_manifest_invalid",
              full24=starts, replicas=replicas,
              terminal_start_ancestry=start_chain_evidence,
              full24_recovery_attempts=full24_attempt_history,
              replica_recovery_attempts=replica_attempt_history,
              validation_error=f"{type(error).__name__}: {error}",
              incumbent_fig6_widths=incumbent)
        return
    if str(replica_status) not in REPLICA_TERMINAL_STATUSES:
        write("technical_failure", "central_replica_recovery_exhausted",
              full24=starts, replicas=replicas,
              terminal_start_ancestry=start_chain_evidence,
              full24_recovery_attempts=full24_attempt_history,
              replica_summary_error=replica_summary_error,
              replica_recovery_attempts=replica_attempt_history,
              incumbent_fig6_widths=incumbent,
              failure_semantics=(
                  "three exact fixed-prescription cache-backed supervisor "
                  "attempts ended without an explicit scientific terminal "
                  "manifest"))
        return
    if (completed_replicas != 50 or len(replica_tags) != 50
            or len(set(replica_tags)) != 50):
        write("technical_failure", "central_replica_coverage_incomplete",
              full24=starts, replicas=replicas,
              terminal_start_ancestry=start_chain_evidence,
              full24_recovery_attempts=full24_attempt_history,
              replica_recovery_attempts=replica_attempt_history,
              incumbent_fig6_widths=incumbent,
              failure_semantics=(
                  "the terminal scientific status was honored and was not "
                  "retried; however, "
                  "lambda600 may not advance to a scientific decision until "
                  "all 50 requested replicas have unique terminal endpoints"))
        return
    try:
        replica_scientific_gate_pass = exact_bool(
            replicas.get("all_replicas_fnp_plateaued"),
            "all_replicas_fnp_plateaued")
        central_scientific_gate_pass = exact_bool(
            replicas.get("central_fnp_plateau_pass"),
            "central_fnp_plateau_pass")
        for key, expected in (
            ("selected_strength", 600.0), ("selected_bmax", 4.0),
            ("fit_quality_barrier_strength", 100.0),
            ("fit_quality_barrier_power", 2.0),
        ):
            exact_number(replicas.get(key), expected, f"replica {key}")
        if not exact_bool(replicas.get("central_full_horizon_complete"),
                          "central_full_horizon_complete"):
            raise RuntimeError("central full-horizon flag is false")
        exact_number(replicas.get(
            "central_full_horizon_requested_capacity"), 300000.0,
            "central_full_horizon_requested_capacity")
        exact_number(replicas.get(
            "central_terminal_requested_capacity"), 300000.0,
            "central_terminal_requested_capacity")
        if not exact_bool(replicas.get(
                "restart_state_and_norm_content_ancestry_recorded"),
                "restart_state_and_norm_content_ancestry_recorded"):
            raise RuntimeError("restart state/norm ancestry was not recorded")
        if (not exact_bool(replicas.get(
                "launch_time_state_and_norm_content_receipts_required"),
                "launch_time_state_and_norm_content_receipts_required")
                or Path(replicas.get("launch_receipt_root", "")).resolve()
                    != LAUNCH_RECEIPTS.resolve()):
            raise RuntimeError(
                "central/replica launch-time receipts were not required")
        if not exact_bool(replicas.get(
                "fit_target_realizations_content_validated"),
                "fit_target_realizations_content_validated"):
            raise RuntimeError("fit-target realizations were not validated")
        if (replicas.get("central_initializer_tag")
                != start_selection.get("selected_tag")
                or replicas.get("central_initializer_selection")
                != start_selection):
            raise RuntimeError(
                "central initializer differs from the audited 24-start medoid")
        acceptance_failed = [int(value) for value in replicas.get(
            "replica_acceptance_failed_seeds", [])]
        failed_capacities = {
            int(seed): int(value) for seed, value in replicas.get(
                "failed_replica_terminal_requested_capacity_by_seed", {}).items()
        }
        if (set(failed_capacities) != set(acceptance_failed)
                or any(value != 300000
                       for value in failed_capacities.values())
                or int(replicas.get(
                    "full_requested_capacity_per_nonpassing_replica", -1))
                    != 300000
                or not exact_bool(replicas.get(
                    "failed_replicas_exhausted_full_requested_horizon"),
                    "failed_replicas_exhausted_full_requested_horizon")):
            raise RuntimeError(
                "nonpassing replicas did not exhaust the exact 300k cap")
    except (TypeError, ValueError, RuntimeError) as error:
        write("technical_failure", "central_replica_manifest_invalid",
              full24=starts, replicas=replicas,
              terminal_start_ancestry=start_chain_evidence,
              full24_recovery_attempts=full24_attempt_history,
              replica_recovery_attempts=replica_attempt_history,
              validation_error=f"{type(error).__name__}: {error}",
              incumbent_fig6_widths=incumbent,
              failure_semantics=(
                  "the terminal scientific status was honored and was not "
                  "retried, but its fixed-prescription manifest failed "
                  "closed validation"))
        return

    write("in_progress", "combined_24x50_figures", full24=starts,
          replicas=replicas,
          terminal_start_ancestry=start_chain_evidence,
          full24_recovery_attempts=full24_attempt_history,
          replica_recovery_attempts=replica_attempt_history,
          start_scientific_gate_pass=start_scientific_gate_pass,
          central_scientific_gate_pass=central_scientific_gate_pass,
          replica_scientific_gate_pass=replica_scientific_gate_pass,
          incumbent_fig6_widths=incumbent)
    for script in (
        "build_final_combined_tmd_ensemble.py",
        "audit_final_combined_ensemble.py",
        "audit_lambda600_postfit_tail_transform.py",
        "validate_lambda600_nested_interactions.py",
        "build_lambda600_final_directional_envelope.py",
    ):
        returncode = run(script, check=False)
        if returncode != 0:
            write("technical_failure", "combined_build_or_audit_failed",
                  failed_script=script, returncode=returncode,
                  full24=starts, replicas=replicas,
                  terminal_start_ancestry=start_chain_evidence,
                  full24_recovery_attempts=full24_attempt_history,
                  replica_recovery_attempts=replica_attempt_history,
                  incumbent_fig6_widths=incumbent)
            return
    stability = load("final_combined_ensemble_stability/summary.json")
    tail_audit, tail_audit_hash = validated_postfit_tail_audit()
    nested_interaction, nested_interaction_hash = validated_nested_interaction()
    final_envelope, final_envelope_hash = validated_final_directional_envelope()
    endpoint_gate_pass = final_promotion_gate(final_envelope)
    # Figures are evidence, not a reward for promotion.  Render the complete
    # lambda600 diagnostic even when stationarity, coverage, robustness, or
    # final-width gates reject it.  The figure renderer records whether the
    # result is promotable; the promotion path below remains fail closed.
    plot_returncode = run("plot_validated_final_fig2_fig6.py", check=False)
    if plot_returncode != 0:
        write("technical_failure", "figure_render_failed",
              returncode=plot_returncode, full24=starts,
              replicas=replicas, stability=stability,
              postfit_tail_transform_audit=tail_audit,
              postfit_tail_transform_audit_sha256=tail_audit_hash,
              nested_interaction_validation=nested_interaction,
              nested_interaction_validation_sha256=nested_interaction_hash,
              final_directional_envelope=final_envelope,
              final_directional_envelope_sha256=final_envelope_hash,
              terminal_start_ancestry=start_chain_evidence,
              full24_recovery_attempts=full24_attempt_history,
              replica_recovery_attempts=replica_attempt_history,
              incumbent_fig6_widths=incumbent)
        return
    figures = load("final_fig2_fig6/summary.json")
    write("in_progress", "explicit_lambda600_vs_lambda1_comparison",
          full24=starts, replicas=replicas, stability=stability,
          postfit_tail_transform_audit=tail_audit,
          postfit_tail_transform_audit_sha256=tail_audit_hash,
          nested_interaction_validation=nested_interaction,
          nested_interaction_validation_sha256=nested_interaction_hash,
          final_directional_envelope=final_envelope,
          final_directional_envelope_sha256=final_envelope_hash,
          figures=figures,
          terminal_start_ancestry=start_chain_evidence,
          full24_recovery_attempts=full24_attempt_history,
          replica_recovery_attempts=replica_attempt_history,
          incumbent_fig6_widths=incumbent)
    comparison_returncode = run(
        "plot_lambda600_vs_lambda1_diagnostic.py", check=False)
    if comparison_returncode != 0:
        write("technical_failure", "explicit_lambda1_comparison_failed",
              returncode=comparison_returncode, full24=starts,
              replicas=replicas, stability=stability, figures=figures,
              postfit_tail_transform_audit=tail_audit,
              postfit_tail_transform_audit_sha256=tail_audit_hash,
              nested_interaction_validation=nested_interaction,
              nested_interaction_validation_sha256=nested_interaction_hash,
              final_directional_envelope=final_envelope,
              final_directional_envelope_sha256=final_envelope_hash,
              terminal_start_ancestry=start_chain_evidence,
              full24_recovery_attempts=full24_attempt_history,
              replica_recovery_attempts=replica_attempt_history,
              incumbent_fig6_widths=incumbent)
        return
    comparison = load("lambda600_vs_lambda1_diagnostic/summary.json")
    evidence_returncode = run(
        "audit_lambda600_terminal_evidence.py", check=False)
    if evidence_returncode != 0:
        write("technical_failure", "terminal_evidence_binding_failed",
              returncode=evidence_returncode, full24=starts,
              replicas=replicas, stability=stability, figures=figures,
              postfit_tail_transform_audit=tail_audit,
              postfit_tail_transform_audit_sha256=tail_audit_hash,
              nested_interaction_validation=nested_interaction,
              nested_interaction_validation_sha256=nested_interaction_hash,
              final_directional_envelope=final_envelope,
              final_directional_envelope_sha256=final_envelope_hash,
              comparison=comparison,
              terminal_start_ancestry=start_chain_evidence,
              full24_recovery_attempts=full24_attempt_history,
              replica_recovery_attempts=replica_attempt_history,
              incumbent_fig6_widths=incumbent)
        return
    terminal_evidence_summary = json.loads(TERMINAL_EVIDENCE.read_text())
    try:
        terminal_endpoint_gate = exact_bool(
            terminal_evidence_summary.get("candidate_endpoint_gate_pass"),
            "terminal evidence candidate_endpoint_gate_pass")
    except RuntimeError:
        terminal_endpoint_gate = None
    if (terminal_evidence_summary.get("status") != "pass"
            or terminal_endpoint_gate != endpoint_gate_pass):
        write("technical_failure", "terminal_evidence_binding_failed",
              full24=starts, replicas=replicas, stability=stability,
              figures=figures, comparison=comparison,
              postfit_tail_transform_audit=tail_audit,
              postfit_tail_transform_audit_sha256=tail_audit_hash,
              nested_interaction_validation=nested_interaction,
              nested_interaction_validation_sha256=nested_interaction_hash,
              final_directional_envelope=final_envelope,
              final_directional_envelope_sha256=final_envelope_hash,
              terminal_start_ancestry=start_chain_evidence,
              terminal_evidence_summary=terminal_evidence_summary,
              full24_recovery_attempts=full24_attempt_history,
              replica_recovery_attempts=replica_attempt_history,
              incumbent_fig6_widths=incumbent)
        return
    terminal_evidence = {
        "status": "pass",
        "manifest": str(TERMINAL_EVIDENCE),
        "manifest_sha256": sha256(TERMINAL_EVIDENCE),
    }
    upstream_scientific_gate_pass = (
        start_scientific_gate_pass
        and central_scientific_gate_pass and replica_scientific_gate_pass
    )
    if endpoint_gate_pass and not upstream_scientific_gate_pass:
        write("technical_failure", "terminal_promotion_gate_inconsistent",
              full24=starts, replicas=replicas, stability=stability,
              figures=figures, comparison=comparison,
              postfit_tail_transform_audit=tail_audit,
              postfit_tail_transform_audit_sha256=tail_audit_hash,
              nested_interaction_validation=nested_interaction,
              nested_interaction_validation_sha256=nested_interaction_hash,
              final_directional_envelope=final_envelope,
              final_directional_envelope_sha256=final_envelope_hash,
              terminal_start_ancestry=start_chain_evidence,
              terminal_evidence=terminal_evidence,
              start_scientific_gate_pass=start_scientific_gate_pass,
              central_scientific_gate_pass=central_scientific_gate_pass,
              replica_scientific_gate_pass=replica_scientific_gate_pass,
              final_endpoint_gate_pass=endpoint_gate_pass,
              full24_recovery_attempts=full24_attempt_history,
              replica_recovery_attempts=replica_attempt_history,
              incumbent_fig6_widths=incumbent)
        return
    promotion_gate_pass = endpoint_gate_pass
    post_promotion_validation = None
    if promotion_gate_pass:
        write("in_progress", "protocol_native_completion_audit",
              full24=starts, replicas=replicas, stability=stability,
              figures=figures, comparison=comparison,
              postfit_tail_transform_audit=tail_audit,
              postfit_tail_transform_audit_sha256=tail_audit_hash,
              nested_interaction_validation=nested_interaction,
              nested_interaction_validation_sha256=nested_interaction_hash,
              final_directional_envelope=final_envelope,
              final_directional_envelope_sha256=final_envelope_hash,
              terminal_evidence=terminal_evidence,
              terminal_start_ancestry=start_chain_evidence,
              full24_recovery_attempts=full24_attempt_history,
              replica_recovery_attempts=replica_attempt_history,
              incumbent_fig6_widths=incumbent)
        try:
            # A previous controller may have crashed after the atomic registry
            # commit but before publishing the terminal decision.  In that
            # state the completion audit intentionally no longer accepts the
            # now-replaced current champion, so recognize the fully validated
            # commit and resume directly from it.
            post_promotion_validation = validated_published_promotion()
        except Exception:
            current_registry, _ = load_optional("champion_registry/current.json")
            current_id = (current_registry.get("champion_id")
                          if isinstance(current_registry, dict) else None)
            if current_id == "empirical_reference_lambda1_b0p1_2p0_full24":
                audit_returncode = run(
                    "audit_like_for_like_completion.py", check=False)
                if audit_returncode != 0:
                    write("technical_failure", "protocol_native_completion_audit",
                          full24=starts, replicas=replicas, stability=stability,
                          figures=figures, comparison=comparison,
                          postfit_tail_transform_audit=tail_audit,
                          postfit_tail_transform_audit_sha256=tail_audit_hash,
                          nested_interaction_validation=nested_interaction,
                          nested_interaction_validation_sha256=nested_interaction_hash,
                          final_directional_envelope=final_envelope,
                          final_directional_envelope_sha256=final_envelope_hash,
                          terminal_evidence=terminal_evidence,
                          terminal_start_ancestry=start_chain_evidence,
                          full24_recovery_attempts=full24_attempt_history,
                          replica_recovery_attempts=replica_attempt_history,
                          incumbent_fig6_widths=incumbent)
                    return
            promotion_returncode = run(
                "promote_validated_final_champion.py", check=False)
            if promotion_returncode != 0:
                write("technical_failure", "promotion_failed",
                      returncode=promotion_returncode, full24=starts,
                      replicas=replicas, stability=stability, figures=figures,
                      comparison=comparison,
                      postfit_tail_transform_audit=tail_audit,
                      postfit_tail_transform_audit_sha256=tail_audit_hash,
                      nested_interaction_validation=nested_interaction,
                      nested_interaction_validation_sha256=nested_interaction_hash,
                      final_directional_envelope=final_envelope,
                      final_directional_envelope_sha256=final_envelope_hash,
                      terminal_evidence=terminal_evidence,
                      terminal_start_ancestry=start_chain_evidence,
                      full24_recovery_attempts=full24_attempt_history,
                      replica_recovery_attempts=replica_attempt_history,
                      incumbent_fig6_widths=incumbent)
                return
        try:
            post_promotion_validation = validated_published_promotion()
        except Exception as error:
            write("technical_failure", "post_promotion_validation_failed",
                  validation_error=f"{type(error).__name__}: {error}",
                  full24=starts, replicas=replicas, stability=stability,
                  figures=figures, comparison=comparison,
                  postfit_tail_transform_audit=tail_audit,
                  postfit_tail_transform_audit_sha256=tail_audit_hash,
                  nested_interaction_validation=nested_interaction,
                  nested_interaction_validation_sha256=nested_interaction_hash,
                  final_directional_envelope=final_envelope,
                  final_directional_envelope_sha256=final_envelope_hash,
                  terminal_evidence=terminal_evidence,
                  terminal_start_ancestry=start_chain_evidence,
                  full24_recovery_attempts=full24_attempt_history,
                  replica_recovery_attempts=replica_attempt_history,
                  incumbent_fig6_widths=incumbent)
            return
        status = "candidate_promoted_as_new_study_champion"
    else:
        status = "candidate_rejected"
    try:
        final_study_report = prepare_final_study_report(status)
    except Exception as error:
        write("technical_failure", "final_study_report_failed",
              report_error=f"{type(error).__name__}: {error}",
              attempted_terminal_status=status,
              full24=starts, replicas=replicas, stability=stability,
              figures=figures, comparison=comparison,
              terminal_evidence=terminal_evidence,
              postfit_tail_transform_audit=tail_audit,
              postfit_tail_transform_audit_sha256=tail_audit_hash,
              nested_interaction_validation=nested_interaction,
              nested_interaction_validation_sha256=nested_interaction_hash,
              final_directional_envelope=final_envelope,
              final_directional_envelope_sha256=final_envelope_hash,
              terminal_start_ancestry=start_chain_evidence,
              full24_recovery_attempts=full24_attempt_history,
              replica_recovery_attempts=replica_attempt_history,
              start_scientific_gate_pass=start_scientific_gate_pass,
              central_scientific_gate_pass=central_scientific_gate_pass,
              replica_scientific_gate_pass=replica_scientific_gate_pass,
              promotion_gate_pass=promotion_gate_pass,
              post_promotion_validation=post_promotion_validation,
              incumbent_fig6_widths=incumbent)
        return
    write(status, "complete_like_for_like_comparison", full24=starts,
          replicas=replicas, stability=stability, figures=figures,
          comparison=comparison, terminal_evidence=terminal_evidence,
          final_study_report=final_study_report,
          postfit_tail_transform_audit=tail_audit,
          postfit_tail_transform_audit_sha256=tail_audit_hash,
          nested_interaction_validation=nested_interaction,
          nested_interaction_validation_sha256=nested_interaction_hash,
          final_directional_envelope=final_envelope,
          final_directional_envelope_sha256=final_envelope_hash,
          terminal_start_ancestry=start_chain_evidence,
          full24_recovery_attempts=full24_attempt_history,
          replica_recovery_attempts=replica_attempt_history,
          start_scientific_gate_pass=start_scientific_gate_pass,
          central_scientific_gate_pass=central_scientific_gate_pass,
          replica_scientific_gate_pass=replica_scientific_gate_pass,
          promotion_gate_pass=promotion_gate_pass,
          post_promotion_validation=post_promotion_validation,
          incumbent_fig6_widths=incumbent)
    try:
        validate_complete_lambda600_comparison()
    except Exception as error:
        write("technical_failure", "terminal_decision_validation_failed",
              validation_error=f"{type(error).__name__}: {error}",
              attempted_terminal_status=status,
              incumbent_fig6_widths=incumbent)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, required=True)
    args = parser.parse_args()
    # The pre-result prescription is an immutable input to every transition,
    # including a cache-backed restart of this controller.
    validate_fixed_challenger_protocol()
    # The lock is held across the watched job, all cache-backed recovery, and
    # the final comparison so a second continuation cannot launch duplicates.
    # A lock loser exits without touching the decision manifest.
    with exclusive_controller_lock():
        if terminal_decision_already_published():
            return
        continue_like_for_like(args.wait_pid)
        require_restartable_controller_exit()


if __name__ == "__main__":
    main()
