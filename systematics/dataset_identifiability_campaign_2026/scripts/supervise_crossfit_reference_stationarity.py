#!/usr/bin/env python3
"""Continue selected cross-fit fits until same-objective FNP drift plateaus."""

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
MAX_CHUNKS = 4
FNP_DRIFT_GATE = 0.005
OBJECTIVE_CHANGE_GATE = 1.0e-5
TARGET = BASE / "summaries/crossfit_reference_distance_lam1p5_stationarity"


def fold(seed: int) -> str:
    return "evenref" if seed % 2 else "oddref"


def fnp_drift(first: Path, second: Path) -> float:
    a, b = pd.read_csv(first / "fnp_grid.csv"), pd.read_csv(second / "fnp_grid.csv")
    mask = np.isclose(a.x, .1) & (a.bT <= 2.0)
    av, bv = a.loc[mask, "F_NP"].to_numpy(), b.loc[mask, "F_NP"].to_numpy()
    return float(np.max(np.abs(bv - av) / np.maximum(av, .05)))


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in SEEDS:
        label = fold(seed)
        base_tag = (f"exactbaseline_matched_reference_distance_{label}_"
                    f"b0p1_2p0_lam2e00_s{seed}")
        previous = BASE / "outputs" / f"{base_tag}_polish64_5000"
        for chunk in range(2, MAX_CHUNKS + 1):
            tag = f"{base_tag}_polish64_{chunk * 5000}"
            target = BASE / "outputs" / tag
            if not (target / "fit_status.json").exists():
                reference = BASE / "summaries" / f"crossfit_reference_{label}" / "fnp_median.csv"
                command = [
                    str(PYTHON), str(RUNNER), "--seed", str(seed),
                    "--source-production", str(SOURCE), "--w-grid", str(W_GRID),
                    "--output-root", str(BASE / "outputs"), "--tag", tag,
                    "--initial-state", str(previous / "model_state.pt"),
                    "--initial-norms", str(previous / "dataset_norms.csv"),
                    "--max-epochs", "0", "--min-epochs", "0", "--plateau-patience", "0",
                    "--lbfgs-max-iter", "5000", "--float64",
                    "--lambda-fnp-reference-distance", "1.5",
                    "--fnp-reference-distance-csv", str(reference),
                    "--fnp-reference-distance-bmin", "0.10",
                    "--fnp-reference-distance-bmax", "2.0",
                ]
                with (BASE / "logs" / f"{tag}.log").open("w") as stream:
                    subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
            before = json.loads((previous / "fit_status.json").read_text())
            after = json.loads((target / "fit_status.json").read_text())
            drift = fnp_drift(previous, target)
            objective_change = abs(after["final"]["objective_per_row"] - before["final"]["objective_per_row"])
            record = {
                "seed": seed, "cumulative_lbfgs_iterations": chunk * 5000,
                "fnp_drift_from_previous_chunk": drift,
                "objective_change_per_row": objective_change,
                "objective_per_row": after["final"]["objective_per_row"],
                "unpenalized_total_chi2": after["final"]["unpenalized_total_chi2"],
                "fnp_gradient_l2_per_row_objective": after["final"]["fnp_gradient_l2_per_row_objective"],
                "plateau_gate_pass": drift <= FNP_DRIFT_GATE and objective_change <= OBJECTIVE_CHANGE_GATE,
                "tag": tag,
            }
            records.append(record)
            pd.DataFrame(records).to_csv(TARGET / "continuations.csv", index=False)
            if record["plateau_gate_pass"]:
                break
            previous = target
    summary = {
        "status": "complete",
        "fnp_drift_gate": FNP_DRIFT_GATE,
        "objective_change_per_row_gate": OBJECTIVE_CHANGE_GATE,
        "max_cumulative_lbfgs_iterations": MAX_CHUNKS * 5000,
        "all_representative_starts_plateaued": all(any(r["seed"] == s and r["plateau_gate_pass"] for r in records) for s in SEEDS),
        "production_sources_modified": False,
    }
    (TARGET / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
