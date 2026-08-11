#!/usr/bin/env python3
"""Run and summarize an isolated 48-start expansion of the lambda=1 baseline.

The incumbent 24 endpoints are retained as read-only members.  The additional
24 fits are initialized from the corresponding incumbent endpoint with a
seeded 1% Gaussian parameter perturbation, while keeping the data, FiLM model,
reference-distance objective, and fit source unchanged.  This is a start-only
diagnostic: it does not alter the production package or launch replicas.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
SYSTEMATICS = BASE.parent
ROOT = SYSTEMATICS.parent
UNITARY = SYSTEMATICS / "high_qt_direct_production_benchmark/experimental_unitary_transition"
RUNNER = UNITARY / "scripts/run_production_fnp_stability_control.py"
PYTHON = Path("/home/dustin/miniforge3/envs/pdf-fit/bin/python")
SOURCE = SYSTEMATICS / "collins_factorization_validity/outputs/rowidfix_stageFT_all_qmax0p20_lam0p50_central_s303"
W_GRID = ROOT / "outputs/v23a_tevatron_plus_lhcb7_fidacc_merged_cache/backend_cache/wpert_v23a_tevatron_plus_lhcb7_fidacc_b160.csv"
REFERENCE = BASE / "summaries/exact_baseline_fnp_median/fnp_median.csv"
CHAMPION = BASE / "summaries/champion_registry/current.json"
TARGET = BASE / "summaries/lambda1_start_expansion48"
OUTPUTS = BASE / "outputs"
LOGS = BASE / "logs"
OLD_SEEDS = tuple(range(303, 327))
NEW_SEEDS = tuple(range(327, 351))
ALL_SEEDS = OLD_SEEDS + NEW_SEEDS
REFERENCE_LAMBDA = 1.0
PERTURBATION = 0.01
BLOCK_EPOCHS = 40_000
MIN_EPOCHS = 10_000
PLATEAU_PATIENCE = 5_000
# The controller advances in 40k blocks, so the 300k requested horizon is
# reached by one final block at 320k rather than silently truncating a block.
MAX_CUMULATIVE_EPOCHS = 320_000
LBFGS_MAX_ITER = 20_000
MAX_LAUNCH_RETRIES = 3
FNP_DRIFT_GATE = 0.02
REQUIRED_FNP_QUIET_BLOCKS = 2
MIN_FNP_EXPOSURE_EPOCHS = 80_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def status_path(tag: str) -> Path:
    return OUTPUTS / tag / "fit_status.json"


def old_tag(seed: int) -> str:
    return f"exactbaseline_matched_reference_distance_b0p1_2p0_lam1e00_s{seed}"


def new_tag(seed: int) -> str:
    return f"lambda1_start_expansion48_s{seed}"


def input_endpoint(seed: int) -> Path:
    endpoint = OUTPUTS / old_tag(seed)
    for name in ("model_state.pt", "dataset_norms.csv", "fnp_grid.csv", "fit_status.json"):
        if not (endpoint / name).exists():
            raise FileNotFoundError(endpoint / name)
    return endpoint


def load_transform_tools():
    path = BASE / "scripts/scan_reference_distance_extent.py"
    spec = importlib.util.spec_from_file_location("lambda1_start_transform", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def command(seed: int, initial: Path, tag: str, perturbation: float) -> list[str]:
    args = [
        str(PYTHON), str(RUNNER),
        "--seed", str(seed),
        "--source-production", str(SOURCE),
        "--w-grid", str(W_GRID),
        "--output-root", str(OUTPUTS),
        "--tag", tag,
        "--initial-state", str(initial / "model_state.pt"),
        "--initial-norms", str(initial / "dataset_norms.csv"),
        "--initial-perturbation", str(perturbation),
        "--max-epochs", str(BLOCK_EPOCHS),
        "--min-epochs", str(MIN_EPOCHS),
        "--plateau-patience", str(PLATEAU_PATIENCE),
        "--learning-rate", "2e-5",
        "--lbfgs-max-iter", str(LBFGS_MAX_ITER),
        "--lambda-fnp-reference-distance", str(REFERENCE_LAMBDA),
        "--fnp-reference-distance-csv", str(REFERENCE),
        "--fnp-reference-distance-bmin", "0.10",
        "--fnp-reference-distance-bmax", "2.0",
    ]
    if perturbation:
        args.insert(args.index("--max-epochs"), "--allow-initial-state-perturbation")
    return args


def write_protocol() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    protocol = {
        "status": "running",
        "study": "lambda1_start_expansion48",
        "incumbent": "empirical_reference_lambda1_b0p1_2p0_full24",
        "old_members": list(OLD_SEEDS),
        "new_members": list(NEW_SEEDS),
        "member_count": 48,
        "new_initialization": "corresponding incumbent endpoint plus seeded 1% Gaussian parameter perturbation",
        "objective": "unchanged lambda=1 direct-FNP reference distance on 0.1<=bT<=2.0",
        "training": {
            "block_epochs": BLOCK_EPOCHS,
            "min_epochs": MIN_EPOCHS,
            "plateau_patience": PLATEAU_PATIENCE,
            "lbfgs_max_iter": LBFGS_MAX_ITER,
            "learning_rate": 2.0e-5,
            "max_cumulative_epochs": MAX_CUMULATIVE_EPOCHS,
            "max_launch_retries": MAX_LAUNCH_RETRIES,
            "fnp_stationarity_gate": {
                "relative_drift_threshold": FNP_DRIFT_GATE,
                "required_consecutive_quiet_blocks": REQUIRED_FNP_QUIET_BLOCKS,
                "minimum_exposure_epochs": MIN_FNP_EXPOSURE_EPOCHS,
            },
        },
        "replicas_launched": False,
        "production_sources_modified": False,
        "source_inputs": {
            "source_production": str(SOURCE),
            "w_grid": str(W_GRID),
            "reference": str(REFERENCE),
            "champion_registry": str(CHAMPION),
        },
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (TARGET / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")


def latest_tag(seed: int) -> tuple[str | None, int]:
    pointer = TARGET / f"latest_s{seed}.json"
    if pointer.exists():
        record = json.loads(pointer.read_text())
        return str(record["tag"]), int(record["cumulative_epochs"])
    base = new_tag(seed)
    if status_path(base).exists():
        fit = json.loads(status_path(base).read_text())
        return base, BLOCK_EPOCHS
    return None, 0


def rebuild_records() -> list[dict]:
    """Reconstruct the ledger from immutable endpoint/status artifacts."""
    rows: list[dict] = []
    pattern = re.compile(r"^lambda1_start_expansion48_s(\d+)(?:_cont(\d+))?$")
    for path in sorted(OUTPUTS.glob("lambda1_start_expansion48_s*")):
        match = pattern.match(path.name)
        if match is None or not (path / "fit_status.json").exists():
            continue
        seed = int(match.group(1))
        if seed not in NEW_SEEDS:
            continue
        cumulative = int(match.group(2) or BLOCK_EPOCHS)
        fit = json.loads((path / "fit_status.json").read_text())
        rows.append({
            "seed": seed,
            "source_seed": OLD_SEEDS[seed - NEW_SEEDS[0]],
            "tag": path.name,
            "status": "complete",
            "initial_perturbation": PERTURBATION if cumulative == BLOCK_EPOCHS else 0.0,
            "cumulative_epochs_requested": cumulative,
            "returncode": 0,
            "stopped_on_plateau": bool(fit.get("stopped_on_plateau", False)),
            "epochs_run": int(fit.get("epochs_run", -1)),
            "lbfgs_closure_evaluations": int(fit.get("lbfgs", {}).get("closure_evaluations", -1)),
            "unpenalized_total_chi2": float(fit.get("final", {}).get("unpenalized_total_chi2", np.nan)),
        })
    return sorted(rows, key=lambda row: (int(row["seed"]), int(row["cumulative_epochs_requested"])))


def fnp_vector(run: Path) -> np.ndarray:
    frame = pd.read_csv(run / "fnp_grid.csv")
    frame = frame[np.isclose(frame.x, .1)
                  & (frame.bT >= .1) & (frame.bT <= 2.0)].sort_values("bT")
    return frame.F_NP.to_numpy(float)


def launch_new() -> None:
    """Run fresh members sequentially, extending every non-plateau fit."""
    records_path = TARGET / "runs.csv"
    records: list[dict] = rebuild_records()
    for index, seed in enumerate(NEW_SEEDS):
        source_seed = OLD_SEEDS[index]
        tag, cumulative = latest_tag(seed)
        if tag is None:
            initial = input_endpoint(source_seed)
            tag = new_tag(seed)
            perturbation = PERTURBATION
            cumulative = 0
        else:
            initial = OUTPUTS / tag
            perturbation = 0.0
        while tag is not None and cumulative < MAX_CUMULATIVE_EPOCHS:
            fit = (json.loads(status_path(tag).read_text())
                   if status_path(tag).exists() else {})
            if fit.get("stopped_on_plateau"):
                break
            # On watchdog/controller restart, honor the persisted stationarity
            # decision instead of launching a redundant confirmation block.
            pointer = TARGET / f"latest_s{seed}.json"
            if pointer.exists():
                pointer_record = json.loads(pointer.read_text())
                if pointer_record.get("stationary_fnp_gate_pass", False):
                    break
            next_cumulative = cumulative + BLOCK_EPOCHS
            run_tag = tag if cumulative == 0 else f"{new_tag(seed)}_cont{next_cumulative}"
            if cumulative == 0:
                run_tag = new_tag(seed)
            target = OUTPUTS / run_tag
            log_path = LOGS / f"{run_tag}.log"
            LOGS.mkdir(parents=True, exist_ok=True)
            row = {
                "seed": seed, "source_seed": source_seed, "tag": run_tag,
                "status": "running", "initial_perturbation": perturbation,
                "cumulative_epochs_requested": next_cumulative,
            }
            records = [existing for existing in records if existing.get("tag") != run_tag]
            records.append(row)
            pd.DataFrame(records).to_csv(records_path, index=False)
            run_command = command(seed, initial, run_tag, perturbation)
            for attempt in range(1, MAX_LAUNCH_RETRIES + 1):
                with log_path.open("a") as stream:
                    stream.write(f"\n# launch attempt {attempt}\n")
                    result = subprocess.run(run_command, stdout=stream,
                                            stderr=subprocess.STDOUT, check=False)
                if result.returncode == 0:
                    break
            row["returncode"] = result.returncode
            row["launch_attempts"] = attempt
            row["status"] = "complete" if result.returncode == 0 else "failed"
            if status_path(run_tag).exists():
                fit = json.loads(status_path(run_tag).read_text())
                drift = float("nan")
                quiet_blocks = 0
                if cumulative > 0 and (initial / "fnp_grid.csv").exists():
                    drift = float(np.max(np.abs(
                        fnp_vector(target) - fnp_vector(initial))
                        / np.maximum(fnp_vector(initial), .05)))
                    prior_rows = [item for item in records
                                  if item.get("seed") == seed
                                  and item.get("fnp_drift_from_previous_block") is not None
                                  and np.isfinite(float(item.get("fnp_drift_from_previous_block", np.nan)))]
                    previous_quiet = (int(prior_rows[-1].get("consecutive_fnp_quiet_blocks", 0))
                                      if prior_rows else 0)
                    quiet_blocks = (previous_quiet + 1
                                    if drift <= FNP_DRIFT_GATE else 0)
                stationary = bool(
                    cumulative >= MIN_FNP_EXPOSURE_EPOCHS
                    and quiet_blocks >= REQUIRED_FNP_QUIET_BLOCKS)
                row.update({
                    "stopped_on_plateau": bool(fit.get("stopped_on_plateau", False)),
                    "epochs_run": int(fit.get("epochs_run", -1)),
                    "lbfgs_closure_evaluations": int(fit.get("lbfgs", {}).get("closure_evaluations", -1)),
                    "unpenalized_total_chi2": float(fit.get("final", {}).get("unpenalized_total_chi2", np.nan)),
                    "fnp_drift_from_previous_block": drift,
                    "consecutive_fnp_quiet_blocks": quiet_blocks,
                    "stationary_fnp_gate_pass": stationary,
                })
                (TARGET / f"latest_s{seed}.json").write_text(json.dumps({
                    "tag": run_tag,
                    "cumulative_epochs": next_cumulative,
                    "stopped_on_plateau": bool(fit.get("stopped_on_plateau", False)),
                    "stationary_fnp_gate_pass": stationary,
                }, indent=2) + "\n")
            pd.DataFrame(records).to_csv(records_path, index=False)
            if result.returncode:
                raise subprocess.CalledProcessError(result.returncode, run_command)
            tag, cumulative = run_tag, next_cumulative
            initial = target
            perturbation = 0.0
            if fit.get("stopped_on_plateau") or row.get("stationary_fnp_gate_pass", False):
                break


def summarize() -> dict:
    runs = [OUTPUTS / old_tag(seed) for seed in OLD_SEEDS]
    latest = []
    for seed in NEW_SEEDS:
        tag, cumulative = latest_tag(seed)
        if tag is None:
            return {"status": "running", "completed_new_count": 0}
        latest.append(OUTPUTS / tag)
    runs += latest
    if not all(status_path(run.name).exists() for run in runs):
        return {"status": "running", "completed_new_count": sum(status_path(new_tag(s)).exists() for s in NEW_SEEDS)}
    frames = []
    fit_status = []
    for seed, run in zip(ALL_SEEDS, runs):
        frame = pd.read_csv(run / "fnp_grid.csv")
        frame = frame[np.isclose(frame.x, 0.1)].sort_values("bT").copy()
        frame["seed"] = seed
        frames.append(frame)
        fit_status.append(json.loads((run / "fit_status.json").read_text()))
    wide = pd.concat(frames, ignore_index=True).pivot(index="bT", columns="seed", values="F_NP")
    values = wide.to_numpy(float)
    median = np.median(values, axis=1)
    # Quantiles are across the 48 start members for each b_T point.
    q16, q84 = np.quantile(values, [0.16, 0.84], axis=1)
    scale = np.maximum(median, 0.05)
    active_b = median > 0.05 * np.max(median)
    bspace = pd.DataFrame({"bT": wide.index.to_numpy(float), "q16": q16, "median": median, "q84": q84,
                           "relative_full_width": (q84-q16)/scale,
                           "relative_full_range": (values.max(1)-values.min(1))/scale})
    bspace.to_csv(TARGET / "bspace_start_only_bands.csv", index=False)
    old_values = values[:, :len(OLD_SEEDS)]
    old_median = np.median(old_values, axis=1)
    old_q16, old_q84 = np.quantile(old_values, [0.16, 0.84], axis=1)
    old_scale = np.maximum(old_median, 0.05)
    old_active = old_median > 0.05 * np.max(old_median)
    bspace_old = pd.DataFrame({
        "bT": wide.index.to_numpy(float), "q16": old_q16,
        "median": old_median, "q84": old_q84,
        "relative_full_width": (old_q84-old_q16)/old_scale,
        "relative_full_range": (old_values.max(1)-old_values.min(1))/old_scale,
    })
    bspace_old.to_csv(TARGET / "bspace_24_start_only_bands.csv", index=False)
    transform = load_transform_tools()
    old_runs = runs[:len(OLD_SEEDS)]
    kspace, _ = transform.project_kspace(runs)
    kspace_old, _ = transform.project_kspace(old_runs)
    kspace.to_csv(TARGET / "kspace_start_only_bands.csv", index=False)
    kspace_old.to_csv(TARGET / "kspace_24_start_only_bands.csv", index=False)

    def widths(frame: pd.DataFrame) -> dict[str, float]:
        result: dict[str, float] = {}
        for flavor, group in frame[frame.kT <= 2.25].groupby("flavor"):
            group = group.sort_values("kT")
            center = group["median"].to_numpy(float)
            active = center > 0.05 * np.max(center)
            result[str(flavor)] = float(np.max(((group.q84-group.q16).to_numpy(float) /
                                                  np.maximum(center, 1e-300))[active]))
        return result

    k_widths = widths(kspace)
    k_widths_old = widths(kspace_old)
    terminal_drifts: dict[str, float] = {}
    stationary_flags: dict[str, bool] = {}
    for seed, run in zip(NEW_SEEDS, latest):
        match = re.match(r"^lambda1_start_expansion48_s\d+(?:_cont(\d+))?$", run.name)
        cumulative = int(match.group(1) or BLOCK_EPOCHS) if match else BLOCK_EPOCHS
        if cumulative <= BLOCK_EPOCHS:
            terminal_drifts[str(seed)] = float("nan")
            fit = json.loads((run / "fit_status.json").read_text())
            stationary_flags[str(seed)] = bool(fit.get("stopped_on_plateau", False))
            continue
        previous_tag = (f"lambda1_start_expansion48_s{seed}_cont{cumulative-BLOCK_EPOCHS}")
        previous = OUTPUTS / previous_tag
        current_frame = pd.read_csv(run / "fnp_grid.csv")
        previous_frame = pd.read_csv(previous / "fnp_grid.csv")
        current_frame = current_frame[np.isclose(current_frame.x, .1) & (current_frame.bT >= .1) & (current_frame.bT <= 2.0)].sort_values("bT")
        previous_frame = previous_frame[np.isclose(previous_frame.x, .1) & (previous_frame.bT >= .1) & (previous_frame.bT <= 2.0)].sort_values("bT")
        terminal_drifts[str(seed)] = float(np.max(
            np.abs(current_frame.F_NP.to_numpy(float) - previous_frame.F_NP.to_numpy(float))
            / np.maximum(previous_frame.F_NP.to_numpy(float), .05)))
        recent = [terminal_drifts[str(seed)]]
        if cumulative > 2 * BLOCK_EPOCHS:
            prior_tag = f"lambda1_start_expansion48_s{seed}_cont{cumulative-2*BLOCK_EPOCHS}"
            prior = OUTPUTS / prior_tag
            if prior.exists():
                prior_frame = pd.read_csv(prior / "fnp_grid.csv")
                prior_previous_tag = f"lambda1_start_expansion48_s{seed}_cont{cumulative-3*BLOCK_EPOCHS}"
                prior_previous = OUTPUTS / prior_previous_tag
                if prior_previous.exists():
                    a = prior_frame[np.isclose(prior_frame.x, .1) & (prior_frame.bT >= .1) & (prior_frame.bT <= 2.0)].sort_values("bT").F_NP.to_numpy(float)
                    bprev = pd.read_csv(prior_previous / "fnp_grid.csv")
                    b = bprev[np.isclose(bprev.x, .1) & (bprev.bT >= .1) & (bprev.bT <= 2.0)].sort_values("bT").F_NP.to_numpy(float)
                    recent.append(float(np.max(np.abs(a-b)/np.maximum(b,.05))))
        # Prefer the controller's persisted gate decision.  The endpoint
        # ledger records the required two quiet blocks even when the first
        # continuation begins at 80k and the terminal tag is 120k.
        pointer = TARGET / f"latest_s{seed}.json"
        if pointer.exists():
            stationary_flags[str(seed)] = bool(
                json.loads(pointer.read_text()).get("stationary_fnp_gate_pass", False))
        else:
            stationary_flags[str(seed)] = len(recent) >= 2 and all(
                value <= FNP_DRIFT_GATE for value in recent[-2:])
    champion = json.loads(CHAMPION.read_text())
    all_plateaued = all(bool(s.get("stopped_on_plateau", False))
                        for s in fit_status[len(OLD_SEEDS):])
    result = {
        "status": "complete" if (all_plateaued or all(stationary_flags.values())) else "horizon_reached_without_plateau",
        "study": "lambda1_start_expansion48",
        "member_count": len(ALL_SEEDS),
        "old_member_count": len(OLD_SEEDS),
        "new_member_count": len(NEW_SEEDS),
        "new_initialization": "paired incumbent endpoint plus 1% seeded Gaussian parameter perturbation",
        "all_new_starts_plateaued": all_plateaued,
        "terminal_fnp_drift_from_previous_block": terminal_drifts,
        "terminal_fnp_stationarity_flags": stationary_flags,
        "all_new_starts_pass_fnp_stationarity_gate": bool(all(stationary_flags.values())),
        "all_new_starts_pass_terminal_fnp_drift_2pct": bool(
            all(np.isfinite(value) and value <= 0.02 for value in terminal_drifts.values())),
        "new_start_terminal_tags": [run.name for run in latest],
        "max_new_unpenalized_chi2": float(max(s["final"]["unpenalized_total_chi2"] for s in fit_status[len(OLD_SEEDS):])),
        "min_new_unpenalized_chi2": float(min(s["final"]["unpenalized_total_chi2"] for s in fit_status[len(OLD_SEEDS):])),
        "bspace_max_active_relative_full_width": float(np.max(bspace.loc[active_b, "relative_full_width"])),
        "bspace_max_active_relative_full_range": float(np.max(bspace.loc[active_b, "relative_full_range"])),
        "bspace_24_start_only_max_active_relative_full_width": float(np.max(bspace_old.loc[old_active, "relative_full_width"])),
        "bspace_width_ratio_48_to_24_start_only": float(
            np.max(bspace.loc[active_b, "relative_full_width"])
            / np.max(bspace_old.loc[old_active, "relative_full_width"])),
        "kspace_max_active_relative_full_width": k_widths,
        "kspace_24_start_only_max_active_relative_full_width": k_widths_old,
        "kspace_width_ratio_48_to_24_start_only": {
            flavor: float(k_widths[flavor] / k_widths_old[flavor])
            for flavor in k_widths
        },
        "incumbent_combined_fig6_max_active_relative_full_width": champion.get("combined_fig6_max_active_relative_full_width"),
        "error_grew_relative_to_24_start_only": {
            flavor: bool(k_widths[flavor] > k_widths_old[flavor])
            for flavor in k_widths
        },
        "interpretation": "start-only q16-q84 diagnostic; no experimental replicas and no 68% confidence claim",
        "replicas_launched": False,
        "production_sources_modified": False,
        "source_hashes": {"reference": sha256(REFERENCE), "w_grid": sha256(W_GRID)},
        "endpoint_tags": [run.name for run in runs],
    }
    (TARGET / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    protocol_path = TARGET / "protocol.json"
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text())
        protocol["status"] = result["status"]
        protocol["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true", help="run missing fresh starts")
    parser.add_argument("--summarize", action="store_true", help="summarize completed 48 members")
    args = parser.parse_args()
    write_protocol()
    if args.launch:
        launch_new()
    if args.summarize or args.launch:
        print(json.dumps(summarize(), indent=2))


if __name__ == "__main__":
    main()
