#!/usr/bin/env python3
"""Run the isolated lambda=300,bmax=4 exact-24 full-horizon gate.

The existing lambda=300 cross-fit endpoints are initial states only.  Every
start is continued with the identical objective to 300,000 requested LBFGS
capacity before the stationarity gate is evaluated.  This controller never
touches frozen production files and is safe to restart: terminal checkpoint
artifacts are reused only when their objective/seed/reference provenance
matches the locked candidate.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
SYSTEMATICS = BASE.parent
ROOT = SYSTEMATICS.parent
UNITARY = SYSTEMATICS / "high_qt_direct_production_benchmark/experimental_unitary_transition"
RUNNER = UNITARY / "scripts/run_production_fnp_stability_control.py"
SOURCE_PRODUCTION = SYSTEMATICS / "collins_factorization_validity/outputs/rowidfix_stageFT_all_qmax0p20_lam0p50_central_s303"
W_GRID = ROOT / "outputs/v23a_tevatron_plus_lhcb7_fidacc_merged_cache/backend_cache/wpert_v23a_tevatron_plus_lhcb7_fidacc_b160.csv"
PILOT_SUMMARY = BASE / "summaries/selected_reference_method_full24/summary.json"
TARGET = BASE / "summaries/lambda300_candidate_full24"
OUT = BASE / "outputs"
LOGS = BASE / "logs"
PYTHON = Path("/home/dustin/miniforge3/envs/pdf-fit/bin/python")
SEEDS = tuple(range(303, 327))
STRENGTH = 300.0
BMAX = 4.0
BMIN = 0.10
FINAL_CAPACITY = 300_000
BLOCK = 5_000
ANCHOR = 200_000
QUIET_REQUIRED = 10
DRIFT_GATE = 0.02
FIT_DELTA_GATE = float(math.sqrt(2.0 * 329.0))


def fold(seed: int) -> str:
    return "evenref" if seed % 2 else "oddref"


def reference(seed: int) -> Path:
    return BASE / "summaries" / f"crossfit_reference_{fold(seed)}" / "fnp_median.csv"


def read_status(path: Path) -> dict:
    return json.loads((path / "fit_status.json").read_text())


def fnp_vector(path: Path) -> np.ndarray:
    frame = pd.read_csv(path / "fnp_grid.csv")
    frame = frame[np.isclose(frame["x"], 0.1) & (frame["bT"] <= BMAX)]
    if len(frame) < 3:
        raise RuntimeError(f"incomplete FNP grid: {path}")
    return frame.sort_values("bT")["F_NP"].to_numpy(float)


def tag(seed: int, cumulative: int) -> str:
    return f"lambda300_candidate_full24_s{seed}_polish64_{cumulative}"


def valid_endpoint(path: Path, seed: int, initial: Path) -> bool:
    required = ("fit_status.json", "model_state.pt", "dataset_norms.csv", "fnp_grid.csv", "accepted_predictions.csv")
    if not all((path / item).is_file() and (path / item).stat().st_size > 0 for item in required):
        return False
    status = read_status(path)
    reg = status.get("regularization", {}).get("fnp_reference_distance", {})
    return (
        int(status.get("seed", -1)) == seed
        and int(status.get("max_epochs", -1)) == 0
        and int(status.get("lbfgs", {}).get("max_iter", -1)) == BLOCK
        and float(reg.get("lambda", -1.0)) == STRENGTH
        and float(reg.get("b_min", -1.0)) == BMIN
        and float(reg.get("b_max", -1.0)) == BMAX
        and Path(status.get("initial_state", "")).resolve() == (initial / "model_state.pt").resolve()
        and not bool(status.get("production_state_modified", True))
    )


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n")
    temp.replace(path)


def publish(records: list[dict], endpoints: dict[int, str], failed: list[int], status: str) -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(TARGET / "runs.csv", index=False)
    atomic_json(TARGET / "summary.json", {
        "status": status,
        "candidate": {"reference_strength": STRENGTH, "reference_bmax": BMAX,
                       "fit_quality_barrier_strength": 0.0, "fit_quality_barrier_power": 2},
        "start_count": len(SEEDS),
        "completed_terminal_count": len(endpoints),
        "failed_seeds_so_far": sorted(set(failed)),
        "expected_final_capacity": FINAL_CAPACITY,
        "stationarity_anchor_capacity": ANCHOR,
        "quiet_blocks_required_after_anchor": QUIET_REQUIRED,
        "fnp_drift_gate": DRIFT_GATE,
        "fit_delta_gate": FIT_DELTA_GATE,
        "endpoint_tags": [endpoints[s] for s in sorted(endpoints)],
        "pilot_initializers": str(PILOT_SUMMARY),
        "production_sources_modified": False,
    })


def main() -> None:
    pilot = json.loads(PILOT_SUMMARY.read_text())
    if pilot.get("selected_strength") != STRENGTH or pilot.get("selected_bmax") != BMAX:
        raise RuntimeError("pilot selection is not the locked lambda=300,bmax=4 candidate")
    initial_by_seed: dict[int, Path] = {}
    for endpoint in pilot["endpoint_tags"]:
        path = OUT / endpoint
        status = read_status(path)
        initial_by_seed[int(status["seed"])] = path
    if set(initial_by_seed) != set(SEEDS):
        raise RuntimeError("pilot initializers do not cover exactly seeds 303..326")

    TARGET.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    endpoints: dict[int, str] = {}
    failed: list[int] = []
    publish(records, endpoints, failed, "in_progress")

    for seed in SEEDS:
        initial = initial_by_seed[seed]
        previous = initial
        anchor_vec = None
        quiet = 0
        seed_records: list[dict] = []
        for cumulative in range(BLOCK, FINAL_CAPACITY + BLOCK, BLOCK):
            target = OUT / tag(seed, cumulative)
            if target.exists() and not valid_endpoint(target, seed, previous):
                raise RuntimeError(f"provenance mismatch in cached endpoint {target}")
            if not target.exists():
                command = [
                    str(PYTHON), str(RUNNER), "--seed", str(seed),
                    "--source-production", str(SOURCE_PRODUCTION), "--w-grid", str(W_GRID),
                    "--output-root", str(OUT), "--tag", target.name,
                    "--initial-state", str(previous / "model_state.pt"),
                    "--initial-norms", str(previous / "dataset_norms.csv"),
                    "--max-epochs", "0", "--min-epochs", "0", "--plateau-patience", "0",
                    "--lbfgs-max-iter", str(BLOCK), "--float64", "--initial-perturbation", "0",
                    "--lambda-fnp-reference-distance", str(STRENGTH),
                    "--fnp-reference-distance-csv", str(reference(seed)),
                    "--fnp-reference-distance-bmin", str(BMIN),
                    "--fnp-reference-distance-bmax", str(BMAX),
                ]
                LOGS.mkdir(parents=True, exist_ok=True)
                with (LOGS / f"{target.name}.log").open("w") as stream:
                    subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
            if not valid_endpoint(target, seed, previous):
                raise RuntimeError(f"invalid endpoint after run: {target}")
            status = read_status(target)
            current = fnp_vector(target)
            old = fnp_vector(previous)
            drift = float(np.max(np.abs(current - old) / np.maximum(old, 0.05)))
            if cumulative == ANCHOR:
                anchor_vec = current.copy()
            drift_anchor = None if anchor_vec is None else float(np.max(np.abs(current - anchor_vec) / np.maximum(anchor_vec, 0.05)))
            fit_delta = float(status.get("final", {}).get("unpenalized_total_chi2", np.inf)) - float(read_status(initial).get("final", {}).get("unpenalized_total_chi2", np.inf))
            quiet_block = cumulative > ANCHOR and drift <= DRIFT_GATE and (drift_anchor is None or drift_anchor <= DRIFT_GATE)
            quiet = quiet + 1 if quiet_block else 0
            row = {"seed": seed, "cumulative_lbfgs_capacity": cumulative,
                   "tag": target.name, "fnp_drift_from_previous": drift,
                   "fnp_drift_from_anchor": drift_anchor, "unpenalized_chi2_delta": fit_delta,
                   "quiet_blocks_after_anchor": quiet, "fit_quality_pass": fit_delta <= FIT_DELTA_GATE,
                   "quiet_block_pass": quiet_block}
            records.append(row); seed_records.append(row)
            endpoints[seed] = target.name
            publish(records, endpoints, failed, "in_progress")
            previous = target

        final_row = seed_records[-1]
        passed = bool(final_row["fit_quality_pass"] and final_row["quiet_blocks_after_anchor"] >= QUIET_REQUIRED)
        if not passed:
            failed.append(seed)
        publish(records, endpoints, failed, "in_progress")

    publish(records, endpoints, failed, "complete" if not failed else "verification_failed")
    print(json.dumps(json.loads((TARGET / "summary.json").read_text()), indent=2))


if __name__ == "__main__":
    main()
