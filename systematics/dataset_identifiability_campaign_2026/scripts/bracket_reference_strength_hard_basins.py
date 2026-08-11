#!/usr/bin/env python3
"""Bracket the weakest stationary reference strength on empirical hard basins.

The hard basins are chosen prospectively from the completed lambda-30 sweep:
all starts requiring at least 15k LBFGS iterations.  Strength 20 is tested
first.  If it passes every hard basin, strength 15 is tested; otherwise 25 is
tested.  Known lambda-10 failure supplies the lower external boundary.  Every
candidate starts from the historical unregularized endpoint and receives the
same adaptive 5k float64 continuation, avoiding warm-start bias across
strengths.
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
LAM30 = BASE / "summaries/crossfit_reference_lam30_full24_extended"
TARGET = BASE / "summaries/reference_strength_hard_basin_bracket"
FNP_DRIFT_GATE = 0.005
FIT_DELTA_GATE = 3.29
MAX_CHUNKS = 8


def token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def fold(seed: int) -> str:
    return "evenref" if seed % 2 else "oddref"


def vector(run: Path) -> np.ndarray:
    frame = pd.read_csv(run / "fnp_grid.csv")
    mask = np.isclose(frame.x, .1) & (frame.bT <= 2)
    return frame.loc[mask, "F_NP"].to_numpy()


def run_strength(strength: float, seeds: list[int], records: list[dict]) -> bool:
    all_pass = True
    for seed in seeds:
        label = fold(seed)
        reference = BASE / "summaries" / f"crossfit_reference_{label}" / "fnp_median.csv"
        historical = UNITARY / "outputs" / f"fig6_lbfgs_stationary_s{seed}"
        source_chi2 = float(json.loads((historical / "fit_status.json").read_text())["final"]["total_chi2"])
        previous = historical
        passed = False
        for chunk in range(1, MAX_CHUNKS + 1):
            cumulative = chunk * 5000
            tag = (f"crossfit_{label}_reference_lam{token(strength)}_hard_"
                   f"s{seed}_polish64_{cumulative}")
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
                    "--lambda-fnp-reference-distance", str(strength),
                    "--fnp-reference-distance-csv", str(reference),
                    "--fnp-reference-distance-bmin", "0.10",
                    "--fnp-reference-distance-bmax", "2.0",
                ]
                with (BASE / "logs" / f"{tag}.log").open("w") as stream:
                    subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
            status = json.loads((target / "fit_status.json").read_text())
            old, new = vector(previous), vector(target)
            drift = float(np.max(np.abs(new - old) / np.maximum(old, .05)))
            unpenalized = float(status["final"]["unpenalized_total_chi2"])
            fit_delta = unpenalized - source_chi2
            passed = chunk > 1 and drift <= FNP_DRIFT_GATE and fit_delta <= FIT_DELTA_GATE
            records.append({
                "strength": strength, "seed": seed, "reference_fold": label,
                "cumulative_lbfgs_iterations": cumulative,
                "fnp_drift_from_previous_chunk": drift,
                "source_total_chi2": source_chi2,
                "unpenalized_total_chi2": unpenalized,
                "unpenalized_chi2_delta": fit_delta,
                "fnp_plateau_and_fit_gate_pass": passed, "tag": tag,
            })
            pd.DataFrame(records).to_csv(TARGET / "runs.csv", index=False)
            previous = target
            if passed:
                break
        all_pass &= passed
    return all_pass


def main() -> None:
    summary_path = LAM30 / "summary.json"
    if not summary_path.exists():
        raise RuntimeError("completed extended lambda-30 ensemble is required")
    lam30_summary = json.loads(summary_path.read_text())
    if not lam30_summary["all_starts_fnp_plateaued"]:
        raise RuntimeError("lambda 30 is not yet a stationary upper boundary")
    runs = pd.read_csv(LAM30 / "runs.csv")
    max_iterations = runs.groupby("seed").cumulative_lbfgs_iterations.max()
    hard_seeds = sorted(int(seed) for seed, value in max_iterations.items() if value >= 15000)
    if 318 not in hard_seeds:
        hard_seeds.append(318)  # preserves continuity with the lambda-10 failure calibration
        hard_seeds.sort()

    TARGET.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    pass20 = run_strength(20.0, hard_seeds, records)
    second = 15.0 if pass20 else 25.0
    pass_second = run_strength(second, hard_seeds, records)
    outcomes = {20.0: pass20, second: pass_second, 30.0: True, 10.0: False}
    passing = sorted(value for value, passed in outcomes.items() if passed)
    selected = passing[0]
    failing_below = max(value for value, passed in outcomes.items()
                        if not passed and value < selected)
    summary = {
        "status": "complete",
        "hard_seed_definition": "lambda30 required >=15000 cumulative LBFGS iterations, plus seed318 calibration basin",
        "hard_seeds": hard_seeds,
        "fnp_drift_gate": FNP_DRIFT_GATE,
        "fit_delta_gate_raw_chi2": FIT_DELTA_GATE,
        "candidate_all_hard_basins_pass": {str(key): value for key, value in sorted(outcomes.items())},
        "weakest_tested_passing_strength": selected,
        "strongest_tested_failing_strength_below_selection": failing_below,
        "selection_bracket": [failing_below, selected],
        "next_step": "verify selected strength on all 24 starts before replica promotion",
        "production_sources_modified": False,
    }
    (TARGET / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
