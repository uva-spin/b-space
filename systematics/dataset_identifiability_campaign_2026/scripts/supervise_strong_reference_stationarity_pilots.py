#!/usr/bin/env python3
"""Calibrate reference strength using long float64 FNP-stationarity pilots."""

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
SEED = 318
STRENGTHS = (5.0, 10.0, 30.0)
MAX_CHUNKS = 3
FNP_DRIFT_GATE = 0.005
TARGET = BASE / "summaries/strong_reference_stationarity_pilots"


def token(value: float) -> str:
    return str(value).replace(".", "p")


def drift(a: Path, b: Path) -> float:
    x, y = pd.read_csv(a / "fnp_grid.csv"), pd.read_csv(b / "fnp_grid.csv")
    mask = np.isclose(x.x, .1) & (x.bT <= 2)
    xv, yv = x.loc[mask, "F_NP"].to_numpy(), y.loc[mask, "F_NP"].to_numpy()
    return float(np.max(np.abs(yv - xv) / np.maximum(xv, .05)))


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    records = []
    reference = BASE / "summaries/crossfit_reference_oddref/fnp_median.csv"
    historical = UNITARY / "outputs" / f"fig6_lbfgs_stationary_s{SEED}"
    for strength in STRENGTHS:
        previous = historical
        for chunk in range(1, MAX_CHUNKS + 1):
            tag = (f"crossfit_oddref_reference_lam{token(strength)}_"
                   f"s{SEED}_polish64_{chunk * 5000}")
            target = BASE / "outputs" / tag
            if not (target / "fit_status.json").exists():
                command = [
                    str(PYTHON), str(RUNNER), "--seed", str(SEED),
                    "--source-production", str(SOURCE), "--w-grid", str(W_GRID),
                    "--output-root", str(BASE / "outputs"), "--tag", tag,
                    "--initial-state", str(previous / "model_state.pt"),
                    "--initial-norms", str(previous / "dataset_norms.csv"),
                    "--max-epochs", "0", "--min-epochs", "0", "--plateau-patience", "0",
                    "--lbfgs-max-iter", "5000", "--float64",
                    "--lambda-fnp-reference-distance", str(strength),
                    "--fnp-reference-distance-csv", str(reference),
                    "--fnp-reference-distance-bmin", "0.10",
                    "--fnp-reference-distance-bmax", "2.0",
                ]
                with (BASE / "logs" / f"{tag}.log").open("w") as stream:
                    subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
            status = json.loads((target / "fit_status.json").read_text())
            d = drift(previous, target)
            records.append({
                "strength": strength, "cumulative_lbfgs_iterations": chunk * 5000,
                "fnp_drift_from_previous_chunk": d,
                "objective_per_row": status["final"]["objective_per_row"],
                "unpenalized_total_chi2": status["final"]["unpenalized_total_chi2"],
                "fnp_gradient_l2_per_row_objective": status["final"]["fnp_gradient_l2_per_row_objective"],
                "fnp_plateau_gate_pass": chunk > 1 and d <= FNP_DRIFT_GATE,
                "tag": tag,
            })
            pd.DataFrame(records).to_csv(TARGET / "runs.csv", index=False)
            if chunk > 1 and d <= FNP_DRIFT_GATE:
                break
            previous = target
    summary = {
        "status": "complete",
        "seed": SEED,
        "strengths": list(STRENGTHS),
        "fnp_drift_gate": FNP_DRIFT_GATE,
        "selection_rule": "weakest strength reaching <=0.5% x=0.1,bT<=2 FNP drift between 5000-step float64 chunks without material fit degradation",
        "production_sources_modified": False,
    }
    (TARGET / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
