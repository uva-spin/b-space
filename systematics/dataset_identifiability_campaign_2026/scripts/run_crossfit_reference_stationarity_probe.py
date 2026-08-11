#!/usr/bin/env python3
"""Long float64 same-objective polish of representative cross-fit endpoints."""

from __future__ import annotations

from pathlib import Path
import subprocess


BASE = Path(__file__).resolve().parents[1]
SYSTEMATICS = BASE.parent
ROOT = SYSTEMATICS.parent
UNITARY = SYSTEMATICS / "high_qt_direct_production_benchmark/experimental_unitary_transition"
RUNNER = UNITARY / "scripts/run_production_fnp_stability_control.py"
SOURCE = SYSTEMATICS / "collins_factorization_validity/outputs/rowidfix_stageFT_all_qmax0p20_lam0p50_central_s303"
W_GRID = ROOT / "outputs/v23a_tevatron_plus_lhcb7_fidacc_merged_cache/backend_cache/wpert_v23a_tevatron_plus_lhcb7_fidacc_b160.csv"
PYTHON = Path("/home/dustin/miniforge3/envs/pdf-fit/bin/python")
SEEDS = (318, 311, 313, 308, 320, 314)


def fold(seed: int) -> str:
    return "evenref" if seed % 2 else "oddref"


def main() -> None:
    for seed in SEEDS:
        label = fold(seed)
        source_tag = (f"exactbaseline_matched_reference_distance_{label}_"
                      f"b0p1_2p0_lam2e00_s{seed}")
        source = BASE / "outputs" / source_tag
        tag = f"{source_tag}_polish64_5000"
        target = BASE / "outputs" / tag
        if (target / "fit_status.json").exists():
            continue
        reference = BASE / "summaries" / f"crossfit_reference_{label}" / "fnp_median.csv"
        log = BASE / "logs" / f"{tag}.log"
        command = [
            str(PYTHON), str(RUNNER), "--seed", str(seed),
            "--source-production", str(SOURCE), "--w-grid", str(W_GRID),
            "--output-root", str(BASE / "outputs"), "--tag", tag,
            "--initial-state", str(source / "model_state.pt"),
            "--initial-norms", str(source / "dataset_norms.csv"),
            "--max-epochs", "0", "--min-epochs", "0", "--plateau-patience", "0",
            "--lbfgs-max-iter", "5000", "--float64",
            "--lambda-fnp-reference-distance", "1.5",
            "--fnp-reference-distance-csv", str(reference),
            "--fnp-reference-distance-bmin", "0.10",
            "--fnp-reference-distance-bmax", "2.0",
        ]
        with log.open("w") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    main()
