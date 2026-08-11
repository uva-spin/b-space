#!/usr/bin/env python3
"""Find the weakest one-sided fit-quality polish for the lambda=675 boundary.

The reference-distance strength is held fixed at the weakest value that made
the augmented replica stress ensemble functionally stationary.  Replica 1041
missed the preregistered total-chi2 ceiling by only 0.353 at its 200k endpoint.
This isolated ladder therefore varies only a one-sided total-chi2 barrier and
requires two consecutive 5k blocks to preserve both the fit gate and the 2%
selected-domain FNP stationarity gate.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
SYSTEMATICS = BASE.parent
ROOT = SYSTEMATICS.parent
PYTHON = Path("/home/dustin/miniforge3/envs/pdf-fit/bin/python")
RUNNER = SYSTEMATICS / "high_qt_direct_production_benchmark/experimental_unitary_transition/scripts/run_production_fnp_stability_control.py"
SOURCE = SYSTEMATICS / "collins_factorization_validity/outputs/rowidfix_stageFT_all_qmax0p20_lam0p50_central_s303"
W_GRID = ROOT / "outputs/v23a_tevatron_plus_lhcb7_fidacc_merged_cache/backend_cache/wpert_v23a_tevatron_plus_lhcb7_fidacc_b160.csv"
REFERENCE = BASE / "summaries/exact_baseline_fnp_median/fnp_median.csv"
INITIAL = BASE / "outputs/fullref_replica_stress_lam675_r1041_polish64_200000"
POWER = int(os.environ.get("FIT_BARRIER_POWER", "2"))
if POWER not in (1, 2):
    raise ValueError("FIT_BARRIER_POWER must be 1 or 2")
SUFFIX = "" if POWER == 2 else "_linear"
TARGET = BASE / f"summaries/lambda675_fit_quality_barrier_rescue{SUFFIX}"
LOGS = BASE / f"logs/lambda675_fit_quality_barrier_rescue{SUFFIX}"
STRENGTHS = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
FNP_GATE = 0.02


def token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def vector(run: Path) -> np.ndarray:
    frame = pd.read_csv(run / "fnp_grid.csv")
    mask = np.isclose(frame.x, 0.1) & (frame.bT >= 0.1) & (frame.bT <= 4.0)
    return frame.loc[mask, "F_NP"].to_numpy(float)


def run_block(mu: float, block: int, previous: Path, ceiling: float) -> Path:
    power_tag = "" if POWER == 2 else "_p1"
    tag = f"fullref_lam675_r1041_fitbar{power_tag}_mu{token(mu)}_confirm{block}_polish64_5000"
    target = BASE / "outputs" / tag
    if not (target / "fit_status.json").exists():
        command = [
            str(PYTHON), str(RUNNER), "--seed", "5041",
            "--source-production", str(SOURCE), "--w-grid", str(W_GRID),
            "--output-root", str(BASE / "outputs"), "--tag", tag,
            "--initial-state", str(previous / "model_state.pt"),
            "--initial-norms", str(previous / "dataset_norms.csv"),
            "--max-epochs", "0", "--min-epochs", "0", "--plateau-patience", "0",
            "--lbfgs-max-iter", "5000", "--float64",
            "--lambda-fnp-reference-distance", "675",
            "--fnp-reference-distance-csv", str(REFERENCE),
            "--fnp-reference-distance-bmin", "0.10",
            "--fnp-reference-distance-bmax", "4.0",
            "--fit-quality-ceiling-total-chi2", str(ceiling),
            "--lambda-fit-quality-barrier", str(mu),
            "--fit-quality-barrier-power", str(POWER),
            "--replica-seed", "1041",
        ]
        with (LOGS / f"{tag}.log").open("w") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
    return target


def measure(mu: float, block: int, previous: Path, current: Path,
            ceiling: float) -> dict:
    status = json.loads((current / "fit_status.json").read_text())
    drift = float(np.max(np.abs(vector(current) - vector(previous)) /
                         np.maximum(np.abs(vector(previous)), 0.05)))
    chi2 = float(status["final"]["unpenalized_total_chi2"])
    return {
        "barrier_strength": mu,
        "confirmation_block": block,
        "tag": current.name,
        "unpenalized_total_chi2": chi2,
        "fit_gate_pass": chi2 <= ceiling,
        "fnp_drift": drift,
        "fnp_stationarity_pass": drift <= FNP_GATE,
    }


def main() -> None:
    if not (INITIAL / "fit_status.json").exists():
        raise RuntimeError("lambda675 replica1041 200k endpoint is required")
    TARGET.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    initial_status = json.loads((INITIAL / "fit_status.json").read_text())
    row_count = int(initial_status["row_count"])
    ceiling = row_count + 5.0 * math.sqrt(2.0 * row_count)
    records = []
    selected = None
    selected_endpoint = None
    for mu in STRENGTHS:
        first = run_block(mu, 1, INITIAL, ceiling)
        row1 = measure(mu, 1, INITIAL, first, ceiling)
        records.append(row1)
        pd.DataFrame(records).to_csv(TARGET / "runs.csv", index=False)
        if not (row1["fit_gate_pass"] and row1["fnp_stationarity_pass"]):
            continue
        second = run_block(mu, 2, first, ceiling)
        row2 = measure(mu, 2, first, second, ceiling)
        records.append(row2)
        pd.DataFrame(records).to_csv(TARGET / "runs.csv", index=False)
        if row2["fit_gate_pass"] and row2["fnp_stationarity_pass"]:
            selected = mu
            selected_endpoint = second.name
            break
    summary = {
        "status": "complete" if selected is not None else "no_barrier_passed",
        "reference_strength_held_fixed": 675.0,
        "fit_quality_barrier_power": POWER,
        "replica_seed": 1041,
        "initial_endpoint": INITIAL.name,
        "fit_quality_ceiling_total_chi2": ceiling,
        "fnp_stationarity_gate": FNP_GATE,
        "selection_rule": "weakest one-sided quadratic barrier passing fit quality and FNP stationarity in two consecutive 5k blocks",
        "selected_weakest_barrier_strength": selected,
        "selected_endpoint": selected_endpoint,
        "trials": records,
        "next_step": "verify selected staged prescription on replica1049 and the augmented stress ensemble" if selected is not None else "test a constrained optimizer rather than increasing reference strength",
        "production_sources_modified": False,
    }
    (TARGET / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
