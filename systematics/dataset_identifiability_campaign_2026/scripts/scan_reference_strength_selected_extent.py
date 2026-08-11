#!/usr/bin/env python3
"""Scan the weakest useful reference strength at the selected b extent."""

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
STRENGTHS = (2.5, 5.0, 10.0, 15.0, 20.0, 25.0)
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
    if extent_summary["status"] == "no_extent_passed":
        # A fixed lambda=30 can be too weak once the reference support is
        # extended.  Continue within the identical prior family by jointly
        # bracketing stronger strengths and extents rather than terminating
        # the automated scientific chain.
        import scan_reference_joint_rescue
        scan_reference_joint_rescue.main()
        return
    if extent_summary["status"] != "complete":
        raise RuntimeError("a completed extent decision is required")
    bmax = float(extent_summary["selected_bmax"])
    extent_candidate = next(
        row for row in extent_summary["candidates"] if float(row["bmax"]) == bmax)
    extent_runs = pd.read_csv(EXTENT / "runs.csv")
    selected_runs = extent_runs[np.isclose(extent_runs.bmax, bmax)]
    hardness = selected_runs.groupby("seed").cumulative_lbfgs_iterations.max().sort_values(ascending=False)
    seeds = [int(seed) for seed in hardness.index]
    TARGET.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    candidates: list[dict] = []

    for strength in STRENGTHS:
        endpoints: list[Path] = []
        candidate_pass = True
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
                cumulative = chunk * 5000
                tag = (f"crossfit_{label}_reference_lam{token(strength)}_"
                       f"extent_b{token(bmax)}_s{seed}_polish64_{cumulative}")
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
                old, new = vector(previous, bmax), vector(target, bmax)
                drift = float(np.max(np.abs(new-old)/np.maximum(old,.05)))
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
            if not passed:
                candidate_pass = False
                failed_seed = seed
                break  # hardest-first rejection avoids scientifically redundant fits
            endpoints.append(previous)

        row = {
            "strength": strength, "bmax": bmax,
            "all_representative_starts_pass": candidate_pass,
            "failed_seed": failed_seed, "member_count": len(endpoints),
            "max_central68_fig6_full_width_active": None,
            "endpoint_tags": [run.name for run in endpoints],
        }
        if candidate_pass:
            bands, width = extent_tools.project_kspace(endpoints)
            bands.to_csv(TARGET / f"kspace_lam_{token(strength)}.csv", index=False)
            row["max_central68_fig6_full_width_active"] = width
        candidates.append(row)
        pd.DataFrame(candidates).drop(columns="endpoint_tags").to_csv(
            TARGET / "candidates.csv", index=False)

    candidates.append({
        "strength": 30.0, "bmax": bmax,
        "all_representative_starts_pass": True, "failed_seed": None,
        "member_count": int(extent_candidate["member_count"]),
        "max_central68_fig6_full_width_active": float(extent_candidate["max_central68_fig6_full_width_active"]),
        "endpoint_tags": extent_candidate["endpoint_tags"],
        "reused_from_extent_scan": True,
    })
    passing = [row for row in candidates if row["all_representative_starts_pass"]]
    narrowest = min(float(row["max_central68_fig6_full_width_active"]) for row in passing)
    eligible = [row for row in passing
                if float(row["max_central68_fig6_full_width_active"]) <= 1.10*narrowest]
    selected = min(eligible, key=lambda row: float(row["strength"]))
    summary = {
        "status": "complete", "selected_bmax": bmax,
        "selection_rule": "weakest passing strength with max active central68 Fig6 full width within 10% of narrowest passing candidate",
        "selected_strength": float(selected["strength"]),
        "representative_seeds_hardest_first": seeds,
        "stationarity_rule": "two consecutive unchanged-objective 5000-iteration blocks with max FNP drift <=2%; readiness proxy calibrated below the 12% current-champion Fig6 width, not the primary promotion metric",
        "drift_sensitivity_thresholds": list(FNP_DRIFT_SENSITIVITY),
        "candidates": candidates,
        "production_sources_modified": False,
    }
    (TARGET / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
