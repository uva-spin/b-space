#!/usr/bin/env python3
"""Repeat the historical final polish with an empirical FNP-distance prior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


BASE = Path(__file__).resolve().parents[1]
SYSTEMATICS = BASE.parent
ROOT = SYSTEMATICS.parent
UNITARY = SYSTEMATICS / "high_qt_direct_production_benchmark/experimental_unitary_transition"
RUNNER = UNITARY / "scripts/run_production_fnp_stability_control.py"
SOURCE = SYSTEMATICS / "collins_factorization_validity/outputs/rowidfix_stageFT_all_qmax0p20_lam0p50_central_s303"
W_GRID = ROOT / "outputs/v23a_tevatron_plus_lhcb7_fidacc_merged_cache/backend_cache/wpert_v23a_tevatron_plus_lhcb7_fidacc_b160.csv"
DEFAULT_REFERENCE = BASE / "summaries/exact_baseline_fnp_median/fnp_median.csv"
PYTHON = Path("/home/dustin/miniforge3/envs/pdf-fit/bin/python")


def token(value: float) -> str:
    # Preserve completed scan tags while avoiding the 1.5/2.0 collision from
    # one-significant-digit scientific formatting.
    if value == 2.0:
        return "2p0"
    return f"{value:.0e}".replace("-", "m").replace("+", "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--strength", type=float, required=True)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--reference-label", default="full24")
    args = parser.parse_args()
    old = UNITARY / "outputs" / f"fig6_lbfgs_stationary_s{args.seed}"
    old_status = json.loads((old / "fit_status.json").read_text())
    precursor = Path(old_status["initial_state"])
    if not precursor.is_absolute():
        precursor = UNITARY / precursor
    safe_label = args.reference_label.replace("/", "_")
    tag = (f"exactbaseline_matched_reference_distance_{safe_label}_"
           f"b0p1_2p0_lam{token(args.strength)}_s{args.seed}")
    target = BASE / "outputs" / tag
    if (target / "fit_status.json").exists():
        print(json.dumps({"status": "already_complete", "tag": tag}))
        return
    command = [
        str(PYTHON), str(RUNNER), "--seed", str(args.seed),
        "--source-production", str(SOURCE), "--w-grid", str(W_GRID),
        "--output-root", str(BASE / "outputs"), "--tag", tag,
        "--initial-state", str(precursor),
        "--initial-norms", str(precursor.parent / "dataset_norms.csv"),
        "--max-epochs", "0", "--min-epochs", "0",
        "--plateau-patience", "0", "--lbfgs-max-iter", "500",
        "--lambda-fnp-reference-distance", str(args.strength),
        "--fnp-reference-distance-csv", str(args.reference),
        "--fnp-reference-distance-bmin", "0.10",
        "--fnp-reference-distance-bmax", "2.0",
    ]
    log = BASE / "logs" / f"{tag}.log"
    with log.open("w") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
    print(json.dumps({"status": "complete", "tag": tag}))


if __name__ == "__main__":
    main()
