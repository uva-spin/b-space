#!/usr/bin/env python3
"""Run the selected locally stable FNP prior over 12 independent basins."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess


BASE = Path(__file__).resolve().parents[1]
SYSTEMATICS = BASE.parent
PYTHON = Path("/home/dustin/miniforge3/envs/pdf-fit/bin/python")
RUNNER = (
    SYSTEMATICS
    / "high_qt_direct_production_benchmark/experimental_unitary_transition/"
    "scripts/run_production_fnp_stability_control.py"
)
SOURCE = (
    SYSTEMATICS
    / "collins_factorization_validity/outputs/"
    "rowidfix_stageFT_E772_qmax0p20_lam0p50_central_s303"
)
W_GRID = (
    SYSTEMATICS.parent
    / "outputs/v23a_tevatron_plus_lhcb7_fidacc_merged_cache/backend_cache/"
    "wpert_v23a_tevatron_plus_lhcb7_fidacc_b160.csv"
)
SEEDS = tuple(range(303, 315))
EXISTING_307 = (
    "logcurv5em5_fslope4em3_xslope3em4_"
    "c2closure_b5p5_s1971_init307"
)


def tag(seed: int) -> str:
    return EXISTING_307 if seed == 307 else f"selected_local_xslope3em4_init{seed}"


def complete(seed: int) -> bool:
    return (BASE / "outputs" / tag(seed) / "fit_status.json").exists()


def command(seed: int) -> list[str]:
    initial = BASE / "outputs" / f"independent_datafit_D020_E772_init{seed}"
    return [
        str(PYTHON), str(RUNNER),
        "--source-production", str(SOURCE),
        "--w-grid", str(W_GRID),
        "--output-root", str(BASE / "outputs"),
        "--initial-state", str(initial / "model_state.pt"),
        "--seed", str(2100 + seed),
        "--initial-perturbation", "0",
        "--max-epochs", "20000",
        "--min-epochs", "5000",
        "--plateau-patience", "3000",
        "--learning-rate", "1e-5",
        "--lbfgs-max-iter", "15000",
        "--float64",
        "--lambda-fnp-logcurv", "5e-5",
        "--fnp-logcurv-bmin", "0.10",
        "--fnp-logcurv-bmax", "3",
        "--lambda-fnp-f-slope", "0.004",
        "--fnp-f-slope-bmin", "0.10",
        "--fnp-f-slope-bmax", "3",
        "--lambda-fnp-x-slope", "3e-4",
        "--fnp-x-slope-bmin", "0.10",
        "--fnp-x-slope-bmax", "3",
        "--closure-tail-coordinate",
        "--closure-tail-b-start", "5.5",
        "--closure-tail-b-end", "8",
        "--lambda-fnp-transform-closure", "0.001",
        "--fnp-transform-closure-bmin", "8",
        "--fnp-transform-closure-max", "0.0001",
        "--tag", tag(seed),
    ]


def main() -> None:
    log_dir = BASE / "logs/local_candidate_multistart"
    log_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = log_dir / "supervisor_status.json"
    records: list[dict] = []
    for seed in SEEDS:
        initial = (
            BASE / "outputs"
            / f"independent_datafit_D020_E772_init{seed}/model_state.pt"
        )
        if not initial.exists():
            raise FileNotFoundError(initial)
        if complete(seed):
            records.append({"source_seed": seed, "tag": tag(seed), "status": "existing"})
            ledger_path.write_text(json.dumps(records, indent=2) + "\n")
            continue
        log_path = log_dir / f"{tag(seed)}.log"
        records.append({"source_seed": seed, "tag": tag(seed), "status": "running"})
        ledger_path.write_text(json.dumps(records, indent=2) + "\n")
        with log_path.open("w") as stream:
            result = subprocess.run(
                command(seed), stdout=stream, stderr=subprocess.STDOUT,
                text=True, check=False,
            )
        records[-1]["returncode"] = result.returncode
        records[-1]["status"] = "complete" if result.returncode == 0 else "failed"
        ledger_path.write_text(json.dumps(records, indent=2) + "\n")
        if result.returncode:
            raise subprocess.CalledProcessError(result.returncode, command(seed))
    (log_dir / "COMPLETE").write_text("complete\n")


if __name__ == "__main__":
    main()
