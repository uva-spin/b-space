#!/usr/bin/env python3
"""Keep the corrected dense F-slope multistart campaign moving unattended."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time


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
OLD_STARTS = (
    SYSTEMATICS
    / "high_qt_direct_production_benchmark/experimental_unitary_transition/"
    "outputs"
)
FIT_CEILING = 119.8021
MAX_PROCESSES = 6
SEEDS = tuple(range(303, 327))


def data_tag(seed: int) -> str:
    return f"independent_datafit_D020_E772_init{seed}"


def dense_tag(seed: int) -> str:
    return f"fslope_dense_twostage_D020_E772_lam0p01_init{seed}"


def output(tag: str) -> Path:
    return BASE / "outputs" / tag


def complete(tag: str) -> bool:
    return (output(tag) / "fit_status.json").exists()


def fit_admissible(seed: int) -> bool:
    path = output(data_tag(seed)) / "fit_status.json"
    if not path.exists():
        return False
    status = json.loads(path.read_text())
    return float(status["final"]["unpenalized_total_chi2"]) <= FIT_CEILING


def externally_running(tag: str) -> bool:
    result = subprocess.run(
        ["pgrep", "-f", f"run_production_fnp_stability_control.py.*--tag {tag}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def command(seed: int, dense: bool) -> list[str]:
    tag = dense_tag(seed) if dense else data_tag(seed)
    initial = (
        output(data_tag(seed))
        if dense
        else OLD_STARTS / f"fig6_lbfgs_stationary_s{seed}"
    )
    cmd = [
        str(PYTHON), str(RUNNER),
        "--seed", str(seed),
        "--source-production", str(SOURCE),
        "--w-grid", str(W_GRID),
        "--output-root", str(BASE / "outputs"),
        "--tag", tag,
        "--initial-state", str(initial / "model_state.pt"),
        "--initial-norms", str(initial / "dataset_norms.csv"),
        "--max-epochs", "20000",
        "--min-epochs", "5000",
        "--plateau-patience", "3000",
        "--learning-rate", "1e-6" if dense else "1e-5",
        "--lbfgs-max-iter", "30000",
        "--float64",
    ]
    if dense:
        cmd += [
            "--lambda-fnp-f-slope", "0.01",
            "--fnp-f-slope-bmin", "0.10",
            "--fnp-f-slope-bmax", "3.0",
        ]
    return cmd


def main() -> None:
    running: dict[str, tuple[subprocess.Popen, object]] = {}
    ledger = BASE / "logs/supervise_dense_fslope_multistart.log"
    ledger.parent.mkdir(parents=True, exist_ok=True)

    while True:
        for tag, (process, stream) in list(running.items()):
            code = process.poll()
            if code is not None:
                stream.close()
                with ledger.open("a") as note:
                    note.write(f"finished {tag} returncode={code}\n")
                del running[tag]

        desired: list[tuple[int, bool]] = []
        for seed in SEEDS:
            tag = data_tag(seed)
            initial = OLD_STARTS / f"fig6_lbfgs_stationary_s{seed}/model_state.pt"
            if initial.exists() and not complete(tag) and not externally_running(tag):
                desired.append((seed, False))
        for seed in SEEDS:
            tag = dense_tag(seed)
            if (
                fit_admissible(seed)
                and not complete(tag)
                and not externally_running(tag)
            ):
                desired.append((seed, True))

        external_count = sum(
            externally_running(data_tag(seed)) + externally_running(dense_tag(seed))
            for seed in SEEDS
        )
        # external_count already includes children launched by this supervisor.
        capacity = max(0, MAX_PROCESSES - external_count)
        for seed, dense in desired[:capacity]:
            tag = dense_tag(seed) if dense else data_tag(seed)
            stream = (BASE / "logs" / f"{tag}.log").open("w")
            process = subprocess.Popen(
                command(seed, dense),
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
            )
            running[tag] = (process, stream)
            with ledger.open("a") as note:
                note.write(f"started {tag} pid={process.pid}\n")

        unfinished_data = any(
            (OLD_STARTS / f"fig6_lbfgs_stationary_s{s}/model_state.pt").exists()
            and not complete(data_tag(s))
            for s in SEEDS
        )
        unfinished_dense = any(
            fit_admissible(s) and not complete(dense_tag(s)) for s in SEEDS
        )
        if not running and external_count == 0 and not unfinished_data and not unfinished_dense:
            with ledger.open("a") as note:
                note.write("complete\n")
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
