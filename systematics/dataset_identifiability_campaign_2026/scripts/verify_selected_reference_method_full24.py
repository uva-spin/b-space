#!/usr/bin/env python3
"""Verify the selected reference strength and extent on all 24 starts."""

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
SELECTION = BASE / "summaries/reference_strength_selected_extent_scan/summary.json"
TARGET = BASE / "summaries/selected_reference_method_full24"
CHAMPION = BASE / "summaries/champion_registry/current.json"
SEEDS = tuple(range(303, 327))
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
    frame = pd.read_csv(run / "fnp_grid.csv")
    mask = np.isclose(frame.x, .1) & (frame.bT <= bmax)
    return frame.loc[mask, "F_NP"].to_numpy()


def main() -> None:
    selection = json.loads(SELECTION.read_text())
    strength = float(selection["selected_strength"])
    bmax = float(selection["selected_bmax"])
    selected_candidate = next(
        row for row in selection["candidates"]
        if (float(row["strength"]) == strength
            and float(row["bmax"]) == bmax))
    reused: dict[int, Path] = {}
    for tag in selected_candidate["endpoint_tags"]:
        run = BASE / "outputs" / tag
        seed = int(json.loads((run / "fit_status.json").read_text())["seed"])
        reused[seed] = run

    TARGET.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    endpoints: list[str] = []
    for seed in SEEDS:
        if seed in reused:
            endpoints.append(reused[seed].name)
            records.append({
                "strength": strength, "bmax": bmax, "seed": seed,
                "reference_fold": fold(seed), "cumulative_lbfgs_iterations": None,
                "fnp_drift_from_previous_chunk": None,
                "unpenalized_chi2_delta": None,
                "fnp_plateau_and_fit_gate_pass": True,
                "tag": reused[seed].name, "reused_from_selection_scan": True,
            })
            pd.DataFrame(records).to_csv(TARGET / "runs.csv", index=False)
            continue
        label = fold(seed)
        reference = BASE / "summaries" / f"crossfit_reference_{label}" / "fnp_median.csv"
        historical = UNITARY / "outputs" / f"fig6_lbfgs_stationary_s{seed}"
        source_chi2 = float(json.loads((historical / "fit_status.json").read_text())["final"]["total_chi2"])
        previous = historical
        passed = False
        consecutive_quiet_blocks = 0
        for chunk in range(1, MAX_CHUNKS + 1):
            cumulative = chunk * 5000
            tag = (f"crossfit_{label}_reference_lam{token(strength)}_b{token(bmax)}_"
                   f"full24_s{seed}_polish64_{cumulative}")
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
            old, new = vector(previous,bmax), vector(target,bmax)
            drift = float(np.max(np.abs(new-old)/np.maximum(old,.05)))
            fit_delta = float(status["final"]["unpenalized_total_chi2"]) - source_chi2
            quiet = chunk > 1 and drift <= FNP_DRIFT_GATE and fit_delta <= FIT_DELTA_GATE
            consecutive_quiet_blocks = consecutive_quiet_blocks + 1 if quiet else 0
            passed = consecutive_quiet_blocks >= REQUIRED_CONSECUTIVE_BLOCKS
            records.append({
                "strength": strength, "bmax": bmax, "seed": seed,
                "reference_fold": label, "cumulative_lbfgs_iterations": cumulative,
                "fnp_drift_from_previous_chunk": drift,
                "unpenalized_chi2_delta": fit_delta,
                "passes_drift_0p25pct": chunk > 1 and drift <= FNP_DRIFT_SENSITIVITY[0],
                "passes_drift_0p5pct": chunk > 1 and drift <= FNP_DRIFT_SENSITIVITY[1],
                "passes_drift_1pct": chunk > 1 and drift <= FNP_DRIFT_SENSITIVITY[2],
                "passes_drift_2pct": chunk > 1 and drift <= FNP_DRIFT_SENSITIVITY[3],
                "consecutive_quiet_blocks": consecutive_quiet_blocks,
                "fnp_plateau_and_fit_gate_pass": passed,
                "tag": tag, "reused_from_selection_scan": False,
            })
            pd.DataFrame(records).to_csv(TARGET / "runs.csv", index=False)
            previous = target
            if passed:
                break
        endpoints.append(previous.name)

    curves=[]; b=None
    for tag in endpoints:
        frame=pd.read_csv(BASE/"outputs"/tag/"fnp_grid.csv")
        frame=frame[np.isclose(frame.x,.1)].sort_values("bT")
        b=frame.bT.to_numpy(); curves.append(frame.F_NP.to_numpy())
    curves=np.asarray(curves); med=np.median(curves,axis=0)
    q16,q84=np.quantile(curves,[.16,.84],axis=0); scale=np.maximum(med,.05)
    active=(b<=bmax)&(med>.05)
    all_pass = all(any(int(row["seed"])==seed and bool(row["fnp_plateau_and_fit_gate_pass"])
                       for row in records) for seed in SEEDS)
    pd.DataFrame({"bT": b, "q16": q16, "median": med, "q84": q84}).to_csv(
        TARGET / "fnp_nonuniqueness_bands.csv", index=False)
    k_bands, _ = extent_tools.project_kspace([BASE / "outputs" / tag for tag in endpoints])
    k_bands.to_csv(TARGET / "kspace_nonuniqueness_bands.csv", index=False)
    k_widths = {}
    for flavor, group in k_bands[k_bands.kT <= 2.25].groupby("flavor"):
        group = group.sort_values("kT")
        center = group["median"].to_numpy(float)
        active_k = center > .05 * np.max(center)
        width = (group.q84 - group.q16).to_numpy(float) / np.maximum(center, 1e-300)
        k_widths[str(flavor)] = float(np.max(width[active_k]))
    champion = json.loads(CHAMPION.read_text())
    champion_widths = champion["combined_fig6_max_active_relative_full_width"]
    provisional_improvement = all(
        k_widths[flavor] < float(champion_widths[flavor]) for flavor in ("u", "d"))
    verified = all_pass and provisional_improvement
    summary={
        "status":"complete" if verified else "verification_failed",
        "selected_strength":strength,"selected_bmax":bmax,
        "member_count":len(SEEDS),
        "all_starts_fnp_plateaued_and_fit_preserved":all_pass,
        "primary_promotion_metric":"complete 24-start Fig2/Fig6 nonuniqueness improvement relative to current champion, followed by combined-replica robustness audit",
        "optimizer_drift_is_readiness_proxy_not_primary_metric":True,
        "readiness_fnp_drift_gate":FNP_DRIFT_GATE,
        "required_consecutive_quiet_blocks":REQUIRED_CONSECUTIVE_BLOCKS,
        "drift_sensitivity_thresholds":list(FNP_DRIFT_SENSITIVITY),
        "current_champion_id":champion["champion_id"],
        "current_champion_combined_fig6_widths":champion_widths,
        "candidate_nonuniqueness_fig6_widths":k_widths,
        "candidate_provisionally_improves_champion":provisional_improvement,
        "max_endpoint_fnp_full_range_active":float(np.max((curves.max(0)-curves.min(0))[active]/scale[active])),
        "max_endpoint_fnp_central68_width_active":float(np.max((q84-q16)[active]/scale[active])),
        "endpoint_tags":endpoints,"production_sources_modified":False,
    }
    (TARGET/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
