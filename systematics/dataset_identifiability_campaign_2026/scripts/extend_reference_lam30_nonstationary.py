#!/usr/bin/env python3
"""Extend lambda-30 starts that missed the initial FNP-stationarity gate.

This is intentionally a second-stage supervisor: it requires the complete
24-start first sweep, then continues only failed starts with the identical
objective in 5k float64 LBFGS blocks.  No incomplete endpoint is accepted.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
SYSTEMATICS = BASE.parent
ROOT = SYSTEMATICS.parent
UNITARY = SYSTEMATICS / "high_qt_direct_production_benchmark/experimental_unitary_transition"
RUNNER = UNITARY / "scripts/run_production_fnp_stability_control.py"
SOURCE = SYSTEMATICS / "collins_factorization_validity/outputs/rowidfix_stageFT_all_qmax0p20_lam0p50_central_s303"
W_GRID = ROOT / "outputs/v23a_tevatron_plus_lhcb7_fidacc_merged_cache/backend_cache/wpert_v23a_tevatron_plus_lhcb7_fidacc_b160.csv"
PYTHON = Path("/home/dustin/miniforge3/envs/pdf-fit/bin/python")
SEEDS = tuple(range(303, 327))
INITIAL_TARGET = BASE / "summaries/crossfit_reference_lam30_full24_stationary"
TARGET = BASE / "summaries/crossfit_reference_lam30_full24_extended"
FNP_DRIFT_GATE = 0.005
MAX_CUMULATIVE_ITERATIONS = 40000


def fold(seed: int) -> str:
    return "evenref" if seed % 2 else "oddref"


def vector(run: Path) -> np.ndarray:
    frame = pd.read_csv(run / "fnp_grid.csv")
    mask = np.isclose(frame.x, .1) & (frame.bT <= 2)
    return frame.loc[mask, "F_NP"].to_numpy()


def main() -> None:
    initial_summary = INITIAL_TARGET / "summary.json"
    if not initial_summary.exists():
        raise RuntimeError("the complete first-stage 24-start summary is required")
    initial = pd.read_csv(INITIAL_TARGET / "runs.csv")
    if set(initial.seed.unique()) != set(SEEDS):
        raise RuntimeError("the first-stage sweep does not contain all 24 seeds")

    TARGET.mkdir(parents=True, exist_ok=True)
    records = initial.to_dict("records")
    endpoint_tags: list[str] = []
    for seed in SEEDS:
        seed_rows = [row for row in records if int(row["seed"]) == seed]
        passed_rows = [row for row in seed_rows if bool(row["fnp_plateau_gate_pass"])]
        if passed_rows:
            endpoint_tags.append(str(passed_rows[-1]["tag"]))
            continue

        previous_row = max(seed_rows, key=lambda row: int(row["cumulative_lbfgs_iterations"]))
        cumulative = int(previous_row["cumulative_lbfgs_iterations"])
        previous = BASE / "outputs" / str(previous_row["tag"])
        label = fold(seed)
        reference = BASE / "summaries" / f"crossfit_reference_{label}" / "fnp_median.csv"
        while cumulative < MAX_CUMULATIVE_ITERATIONS:
            cumulative += 5000
            tag = f"crossfit_{label}_reference_lam30p0_s{seed}_polish64_{cumulative}"
            target = BASE / "outputs" / tag
            if not (target / "fit_status.json").exists():
                command = [
                    str(PYTHON), str(RUNNER), "--seed", str(seed),
                    "--source-production", str(SOURCE), "--w-grid", str(W_GRID),
                    "--output-root", str(BASE / "outputs"), "--tag", tag,
                    "--initial-state", str(previous / "model_state.pt"),
                    "--initial-norms", str(previous / "dataset_norms.csv"),
                    "--max-epochs", "0", "--min-epochs", "0", "--plateau-patience", "0",
                    "--lbfgs-max-iter", "5000", "--float64",
                    "--lambda-fnp-reference-distance", "30",
                    "--fnp-reference-distance-csv", str(reference),
                    "--fnp-reference-distance-bmin", "0.10",
                    "--fnp-reference-distance-bmax", "2.0",
                ]
                with (BASE / "logs" / f"{tag}.log").open("w") as stream:
                    subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
            status = json.loads((target / "fit_status.json").read_text())
            old, new = vector(previous), vector(target)
            drift = float(np.max(np.abs(new - old) / np.maximum(old, .05)))
            passed = drift <= FNP_DRIFT_GATE
            records.append({
                "seed": seed, "reference_fold": label,
                "cumulative_lbfgs_iterations": cumulative,
                "fnp_drift_from_previous_chunk": drift,
                "unpenalized_total_chi2": status["final"]["unpenalized_total_chi2"],
                "objective_per_row": status["final"]["objective_per_row"],
                "fnp_gradient_l2_per_row_objective": status["final"]["fnp_gradient_l2_per_row_objective"],
                "fnp_plateau_gate_pass": passed, "tag": tag,
            })
            pd.DataFrame(records).to_csv(TARGET / "runs.csv", index=False)
            previous = target
            if passed:
                endpoint_tags.append(tag)
                break
        else:
            endpoint_tags.append(previous.name)

    curves = np.asarray([vector(BASE / "outputs" / tag) for tag in endpoint_tags])
    med = np.median(curves, axis=0)
    q16, q84 = np.quantile(curves, [.16, .84], axis=0)
    scale = np.maximum(med, .05)
    all_pass = all(any(int(row["seed"]) == seed and bool(row["fnp_plateau_gate_pass"])
                       for row in records) for seed in SEEDS)
    summary = {
        "status": "complete" if all_pass else "stationarity_ceiling_reached",
        "strength": 30.0, "member_count": len(SEEDS),
        "fnp_drift_gate": FNP_DRIFT_GATE,
        "max_cumulative_lbfgs_iterations": MAX_CUMULATIVE_ITERATIONS,
        "all_starts_fnp_plateaued": all_pass,
        "max_endpoint_fnp_full_range_x0p1_bT_le_2": float(np.max((curves.max(0) - curves.min(0)) / scale)),
        "max_endpoint_fnp_central68_width_x0p1_bT_le_2": float(np.max((q84 - q16) / scale)),
        "endpoint_tags": endpoint_tags,
        "production_sources_modified": False,
    }
    (TARGET / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
