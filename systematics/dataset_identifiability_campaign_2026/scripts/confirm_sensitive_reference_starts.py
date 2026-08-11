#!/usr/bin/env python3
"""Reconfirm selected-strength starts whose terminal drift is near the gate.

The full24 confirmation exposes every member through 40k and requires two
fresh <=2% blocks.  A member ending at or above the recorded 1% sensitivity
threshold receives two *additional* quiet blocks.  One percent is only a
trigger for more evidence; the unchanged acceptance gate remains 2%.
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
SOURCE_CONFIRMATION = BASE / "summaries/selected_reference_start_boundary_confirmation/summary.json"
TARGET = BASE / "summaries/selected_reference_start_sensitivity_confirmation"
PYTHON = Path("/home/dustin/miniforge3/envs/pdf-fit/bin/python")
TRIGGER = 0.01
GATE = 0.02
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
    spec = importlib.util.spec_from_file_location("sensitive_start_extent_tools", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write(status: str, strength: float, bmax: float, selected: list[int],
          confirmations: list[dict], failures: list[int]) -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    (TARGET / "summary.json").write_text(json.dumps({
        "status": status, "selected_strength": strength, "selected_bmax": bmax,
        "selection_trigger": "confirmed terminal FNP drift >=1%",
        "selection_trigger_value": TRIGGER,
        "acceptance_gate": "two entirely new consecutive FNP drift blocks <=2% with source-relative unpenalized chi2 preserved",
        "acceptance_gate_value": GATE, "selected_seeds": selected,
        "confirmed_count": len(confirmations), "confirmations": confirmations,
        "failed_seeds": failures, "production_sources_modified": False,
    }, indent=2) + "\n")


def main() -> None:
    source_confirmation = json.loads(SOURCE_CONFIRMATION.read_text())
    summary = json.loads(SUMMARY.read_text())
    if source_confirmation["status"] != "complete" or source_confirmation["confirmed_count"] != 24:
        raise RuntimeError("all-start delayed-reversal confirmation is incomplete")
    if summary["status"] != "complete" or not summary.get(
            "fresh_start_boundary_confirmation", {}).get("all_24_starts_confirmed"):
        raise RuntimeError("confirmed full24 endpoint summary is incomplete")
    strength, bmax = float(summary["selected_strength"]), float(summary["selected_bmax"])
    barrier_strength = float(summary.get("fit_quality_barrier_strength", 0.0))
    barrier_power = int(summary.get("fit_quality_barrier_power", 2))
    selected_items = [item for item in source_confirmation["confirmations"]
                      if float(item["terminal_drift"]) >= TRIGGER]
    selected = [int(item["seed"]) for item in selected_items]
    endpoint_tags = list(summary["endpoint_tags"])
    by_seed = {int(item["seed"]): item for item in source_confirmation["confirmations"]}
    records = pd.read_csv(PRIMARY / "runs.csv").to_dict("records")
    confirmations: list[dict] = []
    failures: list[int] = []
    write("running", strength, bmax, selected, confirmations, failures)

    for seed in selected:
        item = by_seed[seed]
        previous = BASE / "outputs" / str(item["confirmed_endpoint_tag"])
        cumulative = int(item["cumulative_iterations"])
        rows = pd.DataFrame(records)
        source_chi2 = float(rows[np.isclose(rows.seed, seed)].iloc[0].source_unpenalized_total_chi2)
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
                    "--max-epochs", "0", "--min-epochs", "0", "--plateau-patience", "0",
                    "--lbfgs-max-iter", "5000", "--float64",
                    "--lambda-fnp-reference-distance", str(strength),
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
                    subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
            status = json.loads((target / "fit_status.json").read_text())
            drift = float(np.max(np.abs(vector(target, bmax) - vector(previous, bmax)) /
                                 np.maximum(vector(previous, bmax), .05)))
            quiet = drift <= GATE
            consecutive = consecutive + 1 if quiet else 0
            unpenalized = float(status["final"]["unpenalized_total_chi2"])
            fit_delta = unpenalized - source_chi2
            fit_pass = fit_delta <= CHI2_GATE
            confirmed = consecutive >= 2 and fit_pass
            records.append({
                "strength": strength, "seed": seed,
                "cumulative_lbfgs_iterations": cumulative,
                "requested_lbfgs_max_iterations_this_block": 5000,
                "executed_lbfgs_closure_evaluations_this_block": int(
                    status["lbfgs"]["closure_evaluations"]),
                "fnp_drift_from_previous_chunk": drift,
                "passes_drift_0p25pct": drift <= .0025,
                "passes_drift_0p5pct": drift <= .005,
                "passes_drift_1pct": drift <= .01,
                "passes_drift_2pct": quiet,
                "consecutive_quiet_blocks": consecutive,
                "source_unpenalized_total_chi2": source_chi2,
                "unpenalized_total_chi2": unpenalized,
                "unpenalized_chi2_delta": fit_delta,
                "passes_natural_chi2_scale": fit_pass,
                "passes_legacy_delta3p29_sensitivity": fit_delta <= 3.29,
                "stationarity_and_fit_pass": confirmed, "tag": tag,
            })
            pd.DataFrame(records).to_csv(PRIMARY / "runs.csv", index=False)
            previous = target
            if confirmed:
                index = [int(x["seed"]) for x in source_confirmation["confirmations"]].index(seed)
                endpoint_tags[index] = tag
                confirmations.append({
                    "seed": seed, "trigger_endpoint_tag": item["confirmed_endpoint_tag"],
                    "confirmed_endpoint_tag": tag, "cumulative_iterations": cumulative,
                    "terminal_drift": drift,
                    "source_relative_unpenalized_chi2_delta": fit_delta,
                })
                write("running", strength, bmax, selected, confirmations, failures)
                break
        if not confirmed:
            failures.append(seed)
            write("running", strength, bmax, selected, confirmations, failures)

    runs = [BASE / "outputs" / tag for tag in endpoint_tags]
    curves = np.asarray([vector(run, bmax) for run in runs])
    median = np.median(curves, axis=0)
    q16, q84 = np.quantile(curves, [.16, .84], axis=0)
    scale = np.maximum(median, .05)
    k_bands, _ = load_extent_tools().project_kspace(runs)
    k_bands.to_csv(PRIMARY / "kspace_nonuniqueness_bands.csv", index=False)
    k_widths = {}
    for flavor, group in k_bands[k_bands.kT <= 2.25].groupby("flavor"):
        group = group.sort_values("kT")
        center = group["median"].to_numpy(float)
        active = center > .05 * np.max(center)
        width = (group.q84 - group.q16).to_numpy(float) / np.maximum(center, 1e-300)
        k_widths[str(flavor)] = float(np.max(width[active]))
    ledger = pd.DataFrame(records)
    endpoint_rows = ledger[ledger.tag.isin(set(endpoint_tags))]
    summary["endpoint_tags"] = endpoint_tags
    summary["all_starts_fnp_plateaued_and_fit_preserved"] = not failures
    summary["status"] = "complete" if not failures else "verification_failed"
    summary["failed_seeds"] = failures
    summary["max_endpoint_unpenalized_chi2_delta"] = float(endpoint_rows.unpenalized_chi2_delta.max())
    summary["max_endpoint_fnp_full_range_selected_domain_floor_normalized"] = float(
        np.max((curves.max(0) - curves.min(0)) / scale))
    summary["max_endpoint_fnp_central68_width_selected_domain_floor_normalized"] = float(
        np.max((q84 - q16) / scale))
    summary["candidate_nonuniqueness_fig6_widths"] = k_widths
    summary["sensitive_start_boundary_confirmation"] = {
        "trigger": TRIGGER, "acceptance_gate": GATE, "selected_seeds": selected,
        "confirmations": confirmations, "failed_seeds": failures,
        "all_selected_starts_confirmed": not failures,
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")
    write("complete" if not failures else "failed", strength, bmax,
          selected, confirmations, failures)
    if failures:
        raise RuntimeError(f"sensitive start confirmations failed: {failures}")


if __name__ == "__main__":
    main()
