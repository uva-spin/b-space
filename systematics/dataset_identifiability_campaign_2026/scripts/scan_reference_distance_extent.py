#!/usr/bin/env python3
"""Scan the minimum reference-distance extent needed to control Fig. 6.

Candidates use identical lambda=30 objectives and six representative basins,
including every hard basin identified by the initial bmax=2 sweep.  Fits are
independently restarted from the historical endpoints and adaptively polished.
Selection favors the shortest extent whose central-68 Fig. 6 width is within
10% of the narrowest passing candidate (a declared diminishing-return rule).
"""

from __future__ import annotations

import argparse
import importlib.util
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
REFERENCE_B = SYSTEMATICS / "collins_factorization_validity/plots/rowidfix_stageFT_all_qmax0p20_lam0p50_central_exactx/v22_scheme_tmd_bspace_long.csv"
TRANSFORMER = ROOT / "construct_v23a_regularized_kspace_tmd_v2.py"
PYTHON = Path("/home/dustin/miniforge3/envs/pdf-fit/bin/python")
LAM30 = BASE / "summaries/crossfit_reference_lam30_full24_extended"
TARGET = BASE / "summaries/reference_distance_extent_scan"
STRENGTH = 30.0
BMAX_VALUES = (2.5, 3.0, 4.0, 6.0)
FNP_DRIFT_GATE = 0.02
FNP_DRIFT_SENSITIVITY = (0.0025, 0.005, 0.01, 0.02)
REQUIRED_CONSECUTIVE_BLOCKS = 2
FIT_DELTA_GATE = 3.29
MAX_CHUNKS = 8


def token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def fold(seed: int) -> str:
    return "evenref" if seed % 2 else "oddref"


def fnp_frame(run: Path) -> pd.DataFrame:
    frame = pd.read_csv(run / "fnp_grid.csv")
    return frame[np.isclose(frame.x, .1)].sort_values("bT")


def fnp_vector(run: Path, bmax: float) -> np.ndarray:
    frame = fnp_frame(run)
    mask = frame.bT <= bmax
    return frame.loc[mask, "F_NP"].to_numpy()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def project_kspace(endpoints: list[Path]) -> tuple[pd.DataFrame, float]:
    reference = pd.read_csv(REFERENCE_B)
    reference = reference[
        np.isclose(reference.x, .1) & np.isclose(reference.Q, 10)
        & reference.flavor.astype(str).isin(["u", "d"])
    ].copy()
    rows = []
    for run in endpoints:
        fnp = fnp_frame(run)
        for (_, flavor), group in reference.groupby(["pid", "flavor"], sort=False):
            group = group.sort_values("bT").copy()
            group["F_NP"] = np.interp(group.bT, fnp.bT, fnp.F_NP)
            group["ftilde"] = group.ftilde_no_np * group.F_NP
            group["_replica_key"] = run.name
            group["seed"] = int(json.loads((run / "fit_status.json").read_text())["seed"])
            group["pdf_member"] = 0
            rows.append(group)
    curves = pd.concat(rows, ignore_index=True)
    transform = load_module("extent_scan_transform", TRANSFORMER)
    settings = argparse.Namespace(
        quantities=["ftilde"], tail_mode="expb2", tail_fit_bmin=None,
        eps=1e-300, b_transform_max=24.0, n_b_transform=6001,
        k_max=4.0, n_k=401, end_taper_start_fraction=.92)
    long, _ = transform.transform_curves(curves, settings)
    bands = transform.make_bands(long)
    maxima = []
    for flavor, group in bands.groupby("flavor"):
        group = group.sort_values("kT")
        group = group[group.kT <= 2.25]
        med = group["median"].to_numpy(float)
        peak = np.max(med)
        active = med > .05 * peak
        width = (group.q84 - group.q16).to_numpy(float) / np.maximum(med, 1e-300)
        maxima.append(float(np.max(width[active])))
    return bands, max(maxima)


def main() -> None:
    lam30 = json.loads((LAM30 / "summary.json").read_text())
    # bmax=2 is the diagnosed candidate whose unresolved tail/core motion
    # motivates this extent scan.  Its complete 24-start audit is required,
    # but requiring that inadequate candidate itself pass every stationarity
    # gate would make a scientifically informative failure block the remedy.
    if int(lam30.get("member_count", 0)) != 24:
        raise RuntimeError("complete lambda30/bmax2 24-start audit required")
    initial_runs = pd.read_csv(LAM30 / "runs.csv")
    max_iter = initial_runs.groupby("seed").cumulative_lbfgs_iterations.max()
    hard = sorted(int(seed) for seed, value in max_iter.items() if value >= 15000)
    seeds = sorted(set(hard + [303, 313, 318]))
    TARGET.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    candidate_rows = []

    for bmax in BMAX_VALUES:
        endpoints: list[Path] = []
        candidate_pass = True
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
                tag = (f"crossfit_{label}_reference_lam30_extent_b{token(bmax)}_"
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
                        "--lambda-fnp-reference-distance", "30",
                        "--fnp-reference-distance-csv", str(reference),
                        "--fnp-reference-distance-bmin", "0.10",
                        "--fnp-reference-distance-bmax", str(bmax),
                    ]
                    with (BASE / "logs" / f"{tag}.log").open("w") as stream:
                        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
                status = json.loads((target / "fit_status.json").read_text())
                old, new = fnp_vector(previous, bmax), fnp_vector(target, bmax)
                drift = float(np.max(np.abs(new - old) / np.maximum(old, .05)))
                unpenalized = float(status["final"]["unpenalized_total_chi2"])
                fit_delta = unpenalized - source_chi2
                quiet = chunk > 1 and drift <= FNP_DRIFT_GATE and fit_delta <= FIT_DELTA_GATE
                consecutive_quiet_blocks = consecutive_quiet_blocks + 1 if quiet else 0
                passed = consecutive_quiet_blocks >= REQUIRED_CONSECUTIVE_BLOCKS
                records.append({
                    "bmax": bmax, "strength": STRENGTH, "seed": seed,
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
            candidate_pass &= passed
            if not passed:
                # One ceiling-limited representative basin is sufficient to
                # reject an extent under the all-hard-starts gate.  Preserve
                # its endpoint for a diagnostic projection, then advance to
                # the next extent rather than spending GPU time on a
                # candidate that can no longer be selected.
                break

        curves = np.asarray([fnp_frame(run).F_NP.to_numpy() for run in endpoints])
        b = fnp_frame(endpoints[0]).bT.to_numpy()
        med = np.median(curves, axis=0)
        q16, q84 = np.quantile(curves, [.16, .84], axis=0)
        scale = np.maximum(med, .05)
        active_tail = (b >= 2) & (b <= 6) & (med > .05)
        bands, k_width = project_kspace(endpoints)
        bands.to_csv(TARGET / f"kspace_bmax_{token(bmax)}.csv", index=False)
        candidate_rows.append({
            "bmax": bmax, "all_hard_starts_pass": candidate_pass,
            "member_count": len(endpoints),
            "max_central68_fnp_width_b2_to_6": float(np.max((q84-q16)[active_tail]/scale[active_tail])),
            "max_central68_fig6_full_width_active": k_width,
            "endpoint_tags": [run.name for run in endpoints],
        })
        pd.DataFrame(candidate_rows).drop(columns="endpoint_tags").to_csv(
            TARGET / "candidates.csv", index=False)

    passing = [row for row in candidate_rows if row["all_hard_starts_pass"]]
    if not passing:
        selected = None
        status = "no_extent_passed"
    else:
        narrowest = min(row["max_central68_fig6_full_width_active"] for row in passing)
        eligible = [row for row in passing if row["max_central68_fig6_full_width_active"] <= 1.10*narrowest]
        selected = min(eligible, key=lambda row: row["bmax"])["bmax"]
        status = "complete"
    summary = {
        "status": status, "strength": STRENGTH,
        "representative_seeds": seeds,
        "selection_rule": "smallest passing bmax with max active central68 Fig6 full width within 10% of narrowest passing candidate",
        "selected_bmax": selected,
        "candidates": candidate_rows,
        "production_sources_modified": False,
    }
    (TARGET / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
