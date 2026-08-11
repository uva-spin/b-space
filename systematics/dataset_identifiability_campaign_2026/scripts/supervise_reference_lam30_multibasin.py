#!/usr/bin/env python3
"""Test the weakest FNP-stationary reference strength across six basins."""

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
SEEDS = (318, 311, 313, 308, 320, 314)
MAX_CHUNKS = 3
FNP_DRIFT_GATE = 0.005
TARGET = BASE / "summaries/crossfit_reference_lam30_multibasin"


def fold(seed: int) -> str:
    return "evenref" if seed % 2 else "oddref"


def fnp_vector(run: Path) -> np.ndarray:
    frame = pd.read_csv(run / "fnp_grid.csv")
    mask = np.isclose(frame.x, .1) & (frame.bT <= 2)
    return frame.loc[mask, "F_NP"].to_numpy()


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    records, endpoints = [], []
    for seed in SEEDS:
        label = fold(seed)
        reference = BASE / "summaries" / f"crossfit_reference_{label}" / "fnp_median.csv"
        previous = UNITARY / "outputs" / f"fig6_lbfgs_stationary_s{seed}"
        for chunk in range(1, MAX_CHUNKS + 1):
            tag = f"crossfit_{label}_reference_lam30p0_s{seed}_polish64_{chunk * 5000}"
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
            old, new = fnp_vector(previous), fnp_vector(target)
            drift = float(np.max(np.abs(new - old) / np.maximum(old, .05)))
            passed = chunk > 1 and drift <= FNP_DRIFT_GATE
            records.append({
                "seed": seed, "reference_fold": label,
                "cumulative_lbfgs_iterations": chunk * 5000,
                "fnp_drift_from_previous_chunk": drift,
                "unpenalized_total_chi2": status["final"]["unpenalized_total_chi2"],
                "objective_per_row": status["final"]["objective_per_row"],
                "fnp_gradient_l2_per_row_objective": status["final"]["fnp_gradient_l2_per_row_objective"],
                "fnp_plateau_gate_pass": passed, "tag": tag,
            })
            pd.DataFrame(records).to_csv(TARGET / "runs.csv", index=False)
            if passed:
                endpoints.append(target)
                break
            previous = target
        else:
            endpoints.append(previous)
    curves = np.asarray([fnp_vector(run) for run in endpoints])
    med = np.median(curves, axis=0)
    q16, q84 = np.quantile(curves, [.16, .84], axis=0)
    scale = np.maximum(med, .05)
    summary = {
        "status": "complete",
        "strength": 30.0,
        "representative_seed_count": len(SEEDS),
        "all_starts_fnp_plateaued": all(any(r["seed"] == s and r["fnp_plateau_gate_pass"] for r in records) for s in SEEDS),
        "max_endpoint_fnp_full_range_x0p1_bT_le_2": float(np.max((curves.max(0) - curves.min(0)) / scale)),
        "max_endpoint_fnp_central68_width_x0p1_bT_le_2": float(np.max((q84 - q16) / scale)),
        "endpoint_tags": [run.name for run in endpoints],
        "production_sources_modified": False,
    }
    (TARGET / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
