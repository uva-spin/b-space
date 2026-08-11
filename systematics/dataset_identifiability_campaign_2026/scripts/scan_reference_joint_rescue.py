#!/usr/bin/env python3
"""Rescue scan when lambda=30 fails at every physically relevant extent.

This keeps the identical cross-fitted pointwise reference-distance family and
jointly tests stronger upper-bound strengths at bmax=4 and 6.  It is not a
new architecture or tail ansatz.  Candidates are rejected hardest-basin first
and accepted only after unchanged-objective FNP stationarity and fit-quality
gates pass for every representative start.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

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
EXTENT = BASE / "summaries/reference_distance_extent_scan"
TARGET = BASE / "summaries/reference_strength_selected_extent_scan"
EXTENTS = (4.0, 6.0)
STRENGTHS = (60.0, 100.0, 300.0)
FNP_DRIFT_GATE = 0.02
FNP_DRIFT_SENSITIVITY = (0.0025, 0.005, 0.01, 0.02)
REQUIRED_CONSECUTIVE_BLOCKS = 2
FIT_DELTA_GATE = 3.29
MAX_CHUNKS = 8

sys.path.insert(0, str(BASE / "scripts"))
import scan_reference_distance_extent as extent_tools  # noqa: E402


def token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def fold(seed: int) -> str:
    return "evenref" if seed % 2 else "oddref"


def vector(run: Path, bmax: float) -> np.ndarray:
    frame = extent_tools.fnp_frame(run)
    return frame.loc[frame.bT <= bmax, "F_NP"].to_numpy()


def main() -> None:
    extent_summary = json.loads((EXTENT / "summary.json").read_text())
    if extent_summary["status"] != "no_extent_passed":
        raise RuntimeError("joint rescue is only valid after the lambda=30 extent ceiling fails")
    original_seeds = [int(seed) for seed in extent_summary["representative_seeds"]]
    # The known wandering basin goes first, followed by the seed that rejected
    # bmax=4, then the remaining preregistered representatives.
    seeds = [seed for seed in (307, 303) if seed in original_seeds]
    seeds += [seed for seed in original_seeds if seed not in seeds]
    TARGET.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    candidates: list[dict] = []

    for bmax in EXTENTS:
        for strength in STRENGTHS:
            endpoints: list[Path] = []
            passed_candidate = True
            failed_seed = None
            for seed in seeds:
                label = fold(seed)
                reference = BASE / "summaries" / f"crossfit_reference_{label}" / "fnp_median.csv"
                historical = UNITARY / "outputs" / f"fig6_lbfgs_stationary_s{seed}"
                source_chi2 = float(json.loads((historical / "fit_status.json").read_text())["final"]["total_chi2"])
                previous = historical
                passed = False
                consecutive_quiet_blocks = 0
                for chunk in range(1, MAX_CHUNKS + 1):
                    cumulative = 5000 * chunk
                    tag = (f"crossfit_{label}_reference_joint_lam{token(strength)}_"
                           f"b{token(bmax)}_s{seed}_polish64_{cumulative}")
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
                            "--fnp-reference-distance-bmax", str(bmax),
                        ]
                        with (BASE / "logs" / f"{tag}.log").open("w") as stream:
                            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
                    status = json.loads((target / "fit_status.json").read_text())
                    drift = float(np.max(np.abs(vector(target, bmax) - vector(previous, bmax)) /
                                         np.maximum(vector(previous, bmax), .05)))
                    fit_delta = float(status["final"]["unpenalized_total_chi2"]) - source_chi2
                    quiet = chunk > 1 and drift <= FNP_DRIFT_GATE and fit_delta <= FIT_DELTA_GATE
                    consecutive_quiet_blocks = consecutive_quiet_blocks + 1 if quiet else 0
                    passed = consecutive_quiet_blocks >= REQUIRED_CONSECUTIVE_BLOCKS
                    records.append({
                        "bmax": bmax, "strength": strength, "seed": seed,
                        "reference_fold": label, "cumulative_lbfgs_iterations": cumulative,
                        "fnp_drift_from_previous_chunk": drift,
                        "unpenalized_chi2_delta": fit_delta,
                        "passes_drift_0p25pct": chunk > 1 and drift <= FNP_DRIFT_SENSITIVITY[0],
                        "passes_drift_0p5pct": chunk > 1 and drift <= FNP_DRIFT_SENSITIVITY[1],
                        "passes_drift_1pct": chunk > 1 and drift <= FNP_DRIFT_SENSITIVITY[2],
                        "passes_drift_2pct": chunk > 1 and drift <= FNP_DRIFT_SENSITIVITY[3],
                        "consecutive_quiet_blocks": consecutive_quiet_blocks,
                        "fnp_plateau_and_fit_gate_pass": passed, "tag": tag,
                    })
                    pd.DataFrame(records).to_csv(TARGET / "runs.csv", index=False)
                    previous = target
                    if passed:
                        break
                endpoints.append(previous)
                if not passed:
                    passed_candidate = False
                    failed_seed = seed
                    break

            row = {
                "strength": strength, "bmax": bmax,
                "all_representative_starts_pass": passed_candidate,
                "failed_seed": failed_seed, "member_count": len(endpoints),
                "max_central68_fig6_full_width_active": None,
                "endpoint_tags": [run.name for run in endpoints],
                "joint_rescue_after_lambda30_extent_failure": True,
            }
            if passed_candidate:
                bands, width = extent_tools.project_kspace(endpoints)
                bands.to_csv(TARGET / f"kspace_joint_b{token(bmax)}_lam{token(strength)}.csv", index=False)
                row["max_central68_fig6_full_width_active"] = width
            candidates.append(row)
            pd.DataFrame(candidates).drop(columns="endpoint_tags").to_csv(TARGET / "candidates.csv", index=False)

    passing = [row for row in candidates if row["all_representative_starts_pass"]]
    if passing:
        narrowest = min(float(row["max_central68_fig6_full_width_active"]) for row in passing)
        eligible = [row for row in passing
                    if float(row["max_central68_fig6_full_width_active"]) <= 1.10 * narrowest]
        selected = min(eligible, key=lambda row: (float(row["bmax"]), float(row["strength"])))
        status = "complete"
    else:
        selected = {"bmax": None, "strength": None}
        status = "no_joint_candidate_passed"
    summary = {
        "status": status,
        "selection_rule": "shortest extent then weakest strength among passing candidates within 10% of the narrowest representative Fig6 band",
        "selected_bmax": selected["bmax"], "selected_strength": selected["strength"],
        "representative_seeds_hardest_first": seeds,
        "stationarity_rule": "two consecutive unchanged-objective 5000-iteration blocks with max FNP drift <=2%; readiness proxy calibrated below the 12% current-champion Fig6 width, not the primary promotion metric",
        "drift_sensitivity_thresholds": list(FNP_DRIFT_SENSITIVITY),
        "candidates": candidates, "production_sources_modified": False,
    }
    (TARGET / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
