#!/usr/bin/env python3
"""Require two fresh quiet blocks from every selected-strength start.

The primary 24-start verifier stops after its first two <=2% blocks.  Stress
replicas demonstrated that such an apparent plateau can reverse later, so the
non-uniqueness ensemble receives an independent confirmation beginning after
each primary endpoint.  All 24 starts, not merely threshold-adjacent members,
must establish two new quiet blocks while preserving the source-relative
unpenalized chi-square gate.
"""

from __future__ import annotations

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
REFERENCE = BASE / "summaries/exact_baseline_fnp_median/fnp_median.csv"
PRIMARY = BASE / "summaries/replica_robust_reference_full24"
SUMMARY = PRIMARY / "summary.json"
TARGET = BASE / "summaries/selected_reference_start_boundary_confirmation"
PYTHON = Path("/home/dustin/miniforge3/envs/pdf-fit/bin/python")
SEEDS = tuple(range(303, 327))
GATE = 0.02
SENSITIVITY = (0.0025, 0.005, 0.01, 0.02)
REQUIRED_FRESH_BLOCKS = 2
MINIMUM_CUMULATIVE_ITERATIONS = 40_000
MAXIMUM_CUMULATIVE_ITERATIONS = 200_000
N_DATA = 329
CHI2_GATE = float(np.sqrt(2 * N_DATA))


def token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def vector(run: Path, bmax: float) -> np.ndarray:
    frame = pd.read_csv(run / "fnp_grid.csv")
    mask = np.isclose(frame.x, .1) & (frame.bT >= .1) & (frame.bT <= bmax)
    return frame.loc[mask, "F_NP"].to_numpy(float)


def load_extent_tools():
    path = BASE / "scripts/scan_reference_distance_extent.py"
    spec = importlib.util.spec_from_file_location("confirmed_start_extent_tools", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_confirmation(status: str, strength: float, bmax: float,
                       confirmations: list[dict], failures: list[int]) -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    (TARGET / "summary.json").write_text(json.dumps({
        "status": status,
        "selected_strength": strength,
        "selected_bmax": bmax,
        "rule": "every one of 24 starts must be exposed through 40k cumulative iterations and then establish two wholly subsequent consecutive <=2% FNP-drift blocks with the source-relative unpenalized chi2 gate preserved",
        "required_fresh_quiet_blocks": REQUIRED_FRESH_BLOCKS,
        "minimum_cumulative_iterations": MINIMUM_CUMULATIVE_ITERATIONS,
        "maximum_cumulative_iterations": MAXIMUM_CUMULATIVE_ITERATIONS,
        "confirmed_count": len(confirmations),
        "confirmations": confirmations,
        "failed_seeds": failures,
        "production_sources_modified": False,
    }, indent=2) + "\n")


def main() -> None:
    summary = json.loads(SUMMARY.read_text())
    if summary["status"] != "complete" or not summary[
            "all_starts_fnp_plateaued_and_fit_preserved"]:
        raise RuntimeError("primary selected-strength full24 verification is incomplete")
    strength = float(summary["selected_strength"])
    bmax = float(summary["selected_bmax"])
    barrier_strength = float(summary.get("fit_quality_barrier_strength", 0.0))
    barrier_power = int(summary.get("fit_quality_barrier_power", 2))
    primary_tags = list(summary["endpoint_tags"])
    if len(primary_tags) != len(SEEDS) or len(set(primary_tags)) != len(SEEDS):
        raise RuntimeError("primary verification must provide 24 unique endpoints")

    records = pd.read_csv(PRIMARY / "runs.csv").to_dict("records")
    confirmations: list[dict] = []
    failures: list[int] = []
    confirmed_tags = list(primary_tags)
    write_confirmation("running", strength, bmax, confirmations, failures)

    for index, seed in enumerate(SEEDS):
        frame = pd.DataFrame(records)
        rows = frame[np.isclose(frame["seed"], seed)].sort_values(
            "cumulative_lbfgs_iterations")
        if rows.empty:
            raise RuntimeError(f"missing primary ledger rows for start {seed}")
        last = rows.iloc[-1]
        previous = BASE / "outputs" / str(last.tag)
        cumulative = int(last.cumulative_lbfgs_iterations)
        source_chi2 = float(last.source_unpenalized_total_chi2)
        consecutive = 0
        confirmed = False

        while cumulative < MAXIMUM_CUMULATIVE_ITERATIONS:
            cumulative += 5000
            barrier_tag = (f"_fitbar_p{barrier_power}_mu{token(barrier_strength)}"
                           if barrier_strength > 0 else "")
            tag = (f"fullref_replica_robust_lam{token(strength)}{barrier_tag}_b4_s{seed}_"
                   f"polish64_{cumulative}")
            target = BASE / "outputs" / tag
            if not (target / "fit_status.json").exists():
                command = [
                    str(PYTHON), str(RUNNER), "--seed", str(seed),
                    "--source-production", str(SOURCE), "--w-grid", str(W_GRID),
                    "--output-root", str(BASE / "outputs"), "--tag", tag,
                    "--initial-state", str(previous / "model_state.pt"),
                    "--initial-norms", str(previous / "dataset_norms.csv"),
                    "--max-epochs", "0", "--min-epochs", "0",
                    "--plateau-patience", "0", "--lbfgs-max-iter", "5000",
                    "--float64", "--lambda-fnp-reference-distance", str(strength),
                    "--fnp-reference-distance-csv", str(REFERENCE),
                    "--fnp-reference-distance-bmin", "0.10",
                    "--fnp-reference-distance-bmax", str(bmax),
                ]
                if barrier_strength > 0:
                    command.extend([
                        "--fit-quality-ceiling-total-chi2",
                        str(source_chi2 + CHI2_GATE),
                        "--lambda-fit-quality-barrier", str(barrier_strength),
                        "--fit-quality-barrier-power", str(barrier_power),
                    ])
                with (BASE / "logs" / f"{tag}.log").open("w") as stream:
                    subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                   check=True)
            status = json.loads((target / "fit_status.json").read_text())
            drift = float(np.max(np.abs(vector(target, bmax) - vector(previous, bmax)) /
                                 np.maximum(vector(previous, bmax), .05)))
            quiet = drift <= GATE
            # The stress ensemble showed reversals after apparently quiet
            # trajectories near 40k.  Treat 40k as exposure, not confirmation:
            # only wholly subsequent blocks contribute to the terminal pair.
            if cumulative <= MINIMUM_CUMULATIVE_ITERATIONS:
                consecutive = 0
            else:
                consecutive = consecutive + 1 if quiet else 0
            unpenalized = float(status["final"]["unpenalized_total_chi2"])
            fit_delta = unpenalized - source_chi2
            fit_pass = fit_delta <= CHI2_GATE
            confirmed = (cumulative >= MINIMUM_CUMULATIVE_ITERATIONS + 10_000
                         and consecutive >= REQUIRED_FRESH_BLOCKS and fit_pass)
            records.append({
                "strength": strength, "seed": seed,
                "cumulative_lbfgs_iterations": cumulative,
                "requested_lbfgs_max_iterations_this_block": 5000,
                "executed_lbfgs_closure_evaluations_this_block": int(
                    status["lbfgs"]["closure_evaluations"]),
                "fnp_drift_from_previous_chunk": drift,
                "passes_drift_0p25pct": drift <= SENSITIVITY[0],
                "passes_drift_0p5pct": drift <= SENSITIVITY[1],
                "passes_drift_1pct": drift <= SENSITIVITY[2],
                "passes_drift_2pct": quiet,
                "consecutive_quiet_blocks": consecutive,
                "source_unpenalized_total_chi2": source_chi2,
                "unpenalized_total_chi2": unpenalized,
                "unpenalized_chi2_delta": fit_delta,
                "passes_natural_chi2_scale": fit_pass,
                "passes_legacy_delta3p29_sensitivity": fit_delta <= 3.29,
                "stationarity_and_fit_pass": confirmed,
                "tag": tag,
            })
            pd.DataFrame(records).to_csv(PRIMARY / "runs.csv", index=False)
            previous = target
            if confirmed:
                confirmed_tags[index] = tag
                confirmations.append({
                    "seed": seed, "primary_endpoint_tag": primary_tags[index],
                    "confirmed_endpoint_tag": tag,
                    "cumulative_iterations": cumulative,
                    "additional_blocks": (cumulative - int(last.cumulative_lbfgs_iterations)) // 5000,
                    "terminal_drift": drift,
                    "unpenalized_total_chi2": unpenalized,
                    "source_relative_unpenalized_chi2_delta": fit_delta,
                })
                write_confirmation("running", strength, bmax, confirmations, failures)
                break
        if not confirmed:
            failures.append(seed)
            write_confirmation("running", strength, bmax, confirmations, failures)

    runs = [BASE / "outputs" / tag for tag in confirmed_tags]
    curves = np.asarray([vector(run, bmax) for run in runs])
    median = np.median(curves, axis=0)
    q16, q84 = np.quantile(curves, [.16, .84], axis=0)
    scale = np.maximum(median, .05)
    extent_tools = load_extent_tools()
    k_bands, _ = extent_tools.project_kspace(runs)
    k_bands.to_csv(PRIMARY / "kspace_nonuniqueness_bands.csv", index=False)
    k_widths = {}
    for flavor, group in k_bands[k_bands.kT <= 2.25].groupby("flavor"):
        group = group.sort_values("kT")
        center = group["median"].to_numpy(float)
        active = center > .05 * np.max(center)
        width = ((group.q84 - group.q16).to_numpy(float) /
                 np.maximum(center, 1e-300))
        k_widths[str(flavor)] = float(np.max(width[active]))

    endpoints = set(confirmed_tags)
    ledger = pd.DataFrame(records)
    endpoint_rows = ledger[ledger.tag.isin(endpoints)]
    summary["primary_endpoint_tags_before_fresh_confirmation"] = primary_tags
    summary["endpoint_tags"] = confirmed_tags
    summary["all_starts_fnp_plateaued_and_fit_preserved"] = not failures
    summary["status"] = "complete" if not failures else "verification_failed"
    summary["failed_seeds"] = failures
    summary["max_endpoint_unpenalized_chi2_delta"] = float(
        endpoint_rows.unpenalized_chi2_delta.max())
    summary["max_endpoint_fnp_full_range_selected_domain_floor_normalized"] = float(
        np.max((curves.max(0) - curves.min(0)) / scale))
    summary["max_endpoint_fnp_central68_width_selected_domain_floor_normalized"] = float(
        np.max((q84 - q16) / scale))
    summary["max_endpoint_fnp_full_range_active"] = summary[
        "max_endpoint_fnp_full_range_selected_domain_floor_normalized"]
    summary["max_endpoint_fnp_central68_width_active"] = summary[
        "max_endpoint_fnp_central68_width_selected_domain_floor_normalized"]
    summary["candidate_nonuniqueness_fig6_widths"] = k_widths
    summary["fresh_start_boundary_confirmation"] = {
        "all_24_starts_confirmed": not failures,
        "required_fresh_quiet_blocks": REQUIRED_FRESH_BLOCKS,
        "minimum_cumulative_iterations": MINIMUM_CUMULATIVE_ITERATIONS,
        "maximum_cumulative_iterations": MAXIMUM_CUMULATIVE_ITERATIONS,
        "iteration_accounting": "cumulative labels are requested LBFGS max-iteration capacity; each continuation row records actual closure evaluations separately",
        "confirmations": confirmations,
        "failed_seeds": failures,
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")
    final_status = "complete" if not failures else "failed"
    write_confirmation(final_status, strength, bmax, confirmations, failures)
    print((TARGET / "summary.json").read_text(), end="")
    if failures:
        raise RuntimeError(f"fresh start-boundary confirmations failed: {failures}")


if __name__ == "__main__":
    main()
