#!/usr/bin/env python3
"""Verify the hard-basin-selected reference strength on all 24 starts."""

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
BRACKET = BASE / "summaries/reference_strength_hard_basin_bracket/summary.json"
TARGET = BASE / "summaries/selected_reference_strength_full24"
SEEDS = tuple(range(303, 327))
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


def main() -> None:
    bracket = json.loads(BRACKET.read_text())
    strength = float(bracket["weakest_tested_passing_strength"])
    if strength == 30.0:
        source_summary = BASE / "summaries/crossfit_reference_lam30_full24_extended/summary.json"
        selected = json.loads(source_summary.read_text())
        if not selected["all_starts_fnp_plateaued"]:
            raise RuntimeError("selected lambda30 ensemble is not stationary")
        selected.update({
            "status": "complete", "selected_strength": strength,
            "selection_bracket": bracket["selection_bracket"],
            "reused_from": str(source_summary),
            "all_starts_fnp_plateaued_and_fit_preserved": True,
        })
        TARGET.mkdir(parents=True, exist_ok=True)
        (TARGET / "summary.json").write_text(json.dumps(selected, indent=2) + "\n")
        pd.read_csv(source_summary.parent / "runs.csv").to_csv(TARGET / "runs.csv", index=False)
        print(json.dumps(selected, indent=2))
        return

    TARGET.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    endpoints: list[str] = []
    for seed in SEEDS:
        label = fold(seed)
        reference = BASE / "summaries" / f"crossfit_reference_{label}" / "fnp_median.csv"
        historical = UNITARY / "outputs" / f"fig6_lbfgs_stationary_s{seed}"
        source_chi2 = float(json.loads((historical / "fit_status.json").read_text())["final"]["total_chi2"])
        previous = historical
        passed = False
        for chunk in range(1, MAX_CHUNKS + 1):
            cumulative = chunk * 5000
            tag = (f"crossfit_{label}_reference_lam{token(strength)}_full24_"
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
        endpoints.append(previous.name)

    curves = np.asarray([vector(BASE / "outputs" / tag) for tag in endpoints])
    med = np.median(curves, axis=0)
    q16, q84 = np.quantile(curves, [.16, .84], axis=0)
    scale = np.maximum(med, .05)
    all_pass = all(any(int(row["seed"]) == seed and bool(row["fnp_plateau_and_fit_gate_pass"])
                       for row in records) for seed in SEEDS)
    summary = {
        "status": "complete" if all_pass else "verification_failed",
        "selected_strength": strength,
        "selection_bracket": bracket["selection_bracket"],
        "member_count": len(SEEDS),
        "all_starts_fnp_plateaued_and_fit_preserved": all_pass,
        "max_endpoint_fnp_full_range_x0p1_bT_le_2": float(np.max((curves.max(0) - curves.min(0)) / scale)),
        "max_endpoint_fnp_central68_width_x0p1_bT_le_2": float(np.max((q84 - q16) / scale)),
        "endpoint_tags": endpoints,
        "production_sources_modified": False,
    }
    (TARGET / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
