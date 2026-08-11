#!/usr/bin/env python3
"""Extend the isolated lambda=1 start diagnostic from 48 to 96 members.

The original 24 accepted endpoints and the first 24 perturbed endpoints are
read-only members.  This controller adds two independent 24-member perturbation
sets, using the exact lambda=1 objective and stationarity protocol from the
48-start study.  It never launches replicas or writes production outputs.
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
TARGET = BASE / "summaries/lambda1_start_expansion96"
OUTPUTS = BASE / "outputs"
LOGS = BASE / "logs"

OLD_SEEDS = tuple(range(303, 327))
FIRST_EXPANSION_SEEDS = tuple(range(327, 351))
SECOND_EXPANSION_SEEDS = tuple(range(351, 375))
THIRD_EXPANSION_SEEDS = tuple(range(375, 399))
NEW_SEEDS = SECOND_EXPANSION_SEEDS + THIRD_EXPANSION_SEEDS
ALL_SEEDS = OLD_SEEDS + FIRST_EXPANSION_SEEDS + NEW_SEEDS

REFERENCE_LAMBDA = 1.0
PERTURBATION = 0.01
BLOCK_EPOCHS = 40_000
MIN_EPOCHS = 10_000
PLATEAU_PATIENCE = 5_000
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


def first_tag(seed: int) -> str:
    return f"lambda1_start_expansion48_s{seed}"


def first_terminal_tags() -> list[str]:
    summary = json.loads(
        (BASE / "summaries/lambda1_start_expansion48/summary.json").read_text()
    )
    return list(summary["new_start_terminal_tags"])


def new_tag(seed: int) -> str:
    return f"lambda1_start_expansion96_s{seed}"


def source_seed(seed: int) -> int:
    if seed in SECOND_EXPANSION_SEEDS:
        return OLD_SEEDS[seed - SECOND_EXPANSION_SEEDS[0]]
    if seed in THIRD_EXPANSION_SEEDS:
        return OLD_SEEDS[seed - THIRD_EXPANSION_SEEDS[0]]
    raise ValueError(seed)


def input_endpoint(seed: int) -> Path:
    endpoint = OUTPUTS / old_tag(seed)
    for name in ("model_state.pt", "dataset_norms.csv", "fnp_grid.csv", "fit_status.json"):
        if not (endpoint / name).exists():
            raise FileNotFoundError(endpoint / name)
    return endpoint


def load_transform_tools():
    path = BASE / "scripts/scan_reference_distance_extent.py"
    spec = importlib.util.spec_from_file_location("lambda1_start_transform96", path)
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
        "study": "lambda1_start_expansion96",
        "incumbent": "empirical_reference_lambda1_b0p1_2p0_full24",
        "read_only_members": list(OLD_SEEDS + FIRST_EXPANSION_SEEDS),
        "new_members": list(NEW_SEEDS),
        "member_count": 96,
        "new_initialization": "two independent paired 1% seeded Gaussian perturbation sets from each corresponding incumbent endpoint",
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
        return base, BLOCK_EPOCHS
    return None, 0


def rebuild_records() -> list[dict]:
    rows: list[dict] = []
    pattern = re.compile(r"^lambda1_start_expansion96_s(\d+)(?:_cont(\d+))?$")
    for path in sorted(OUTPUTS.glob("lambda1_start_expansion96_s*")):
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
            "source_seed": source_seed(seed),
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
    frame = frame[np.isclose(frame.x, .1) & (frame.bT >= .1) & (frame.bT <= 2.0)].sort_values("bT")
    return frame.F_NP.to_numpy(float)


def launch_new() -> None:
    records_path = TARGET / "runs.csv"
    records: list[dict] = rebuild_records()
    for seed in NEW_SEEDS:
        tag, cumulative = latest_tag(seed)
        if tag is None:
            initial = input_endpoint(source_seed(seed))
            tag = new_tag(seed)
            perturbation = PERTURBATION
            cumulative = 0
        else:
            initial = OUTPUTS / tag
            perturbation = 0.0
        while tag is not None and cumulative < MAX_CUMULATIVE_EPOCHS:
            fit = json.loads(status_path(tag).read_text()) if status_path(tag).exists() else {}
            if fit.get("stopped_on_plateau"):
                break
            pointer = TARGET / f"latest_s{seed}.json"
            if pointer.exists() and json.loads(pointer.read_text()).get("stationary_fnp_gate_pass", False):
                break
            next_cumulative = cumulative + BLOCK_EPOCHS
            run_tag = new_tag(seed) if cumulative == 0 else f"{new_tag(seed)}_cont{next_cumulative}"
            target = OUTPUTS / run_tag
            LOGS.mkdir(parents=True, exist_ok=True)
            log_path = LOGS / f"{run_tag}.log"
            row = {"seed": seed, "source_seed": source_seed(seed), "tag": run_tag, "status": "running",
                   "initial_perturbation": perturbation, "cumulative_epochs_requested": next_cumulative}
            records = [old for old in records if old.get("tag") != run_tag]
            records.append(row)
            pd.DataFrame(records).to_csv(records_path, index=False)
            run_command = command(seed, initial, run_tag, perturbation)
            for attempt in range(1, MAX_LAUNCH_RETRIES + 1):
                with log_path.open("a") as stream:
                    stream.write(f"\n# launch attempt {attempt}\n")
                    result = subprocess.run(run_command, stdout=stream, stderr=subprocess.STDOUT, check=False)
                if result.returncode == 0:
                    break
            row.update({"returncode": result.returncode, "launch_attempts": attempt,
                        "status": "complete" if result.returncode == 0 else "failed"})
            if status_path(run_tag).exists():
                fit = json.loads(status_path(run_tag).read_text())
                drift = float("nan")
                quiet_blocks = 0
                if cumulative > 0 and (initial / "fnp_grid.csv").exists():
                    drift = float(np.max(np.abs(fnp_vector(target) - fnp_vector(initial)) /
                                         np.maximum(fnp_vector(initial), .05)))
                    prior_rows = [item for item in records if item.get("seed") == seed
                                  and item.get("fnp_drift_from_previous_block") is not None
                                  and np.isfinite(float(item.get("fnp_drift_from_previous_block", np.nan)))]
                    previous_quiet = int(prior_rows[-1].get("consecutive_fnp_quiet_blocks", 0)) if prior_rows else 0
                    quiet_blocks = previous_quiet + 1 if drift <= FNP_DRIFT_GATE else 0
                stationary = bool(cumulative >= MIN_FNP_EXPOSURE_EPOCHS and quiet_blocks >= REQUIRED_FNP_QUIET_BLOCKS)
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
                    "tag": run_tag, "cumulative_epochs": next_cumulative,
                    "stopped_on_plateau": bool(fit.get("stopped_on_plateau", False)),
                    "stationary_fnp_gate_pass": stationary,
                }, indent=2) + "\n")
            pd.DataFrame(records).to_csv(records_path, index=False)
            if result.returncode:
                raise subprocess.CalledProcessError(result.returncode, run_command)
            tag, cumulative, initial, perturbation = run_tag, next_cumulative, target, 0.0
            if fit.get("stopped_on_plateau") or row.get("stationary_fnp_gate_pass", False):
                break


def project_bands(runs: list[Path], members: list[int], path: Path) -> pd.DataFrame:
    frames = []
    for member, run in zip(members, runs):
        frame = pd.read_csv(run / "fnp_grid.csv")
        frame = frame[np.isclose(frame.x, 0.1)].sort_values("bT").copy()
        frame["member"] = member
        frames.append(frame)
    wide = pd.concat(frames, ignore_index=True).pivot(index="bT", columns="member", values="F_NP")
    values = wide.to_numpy(float)
    median = np.median(values, axis=1)
    q16, q84 = np.quantile(values, [0.16, 0.84], axis=1)
    scale = np.maximum(median, .05)
    result = pd.DataFrame({"bT": wide.index.to_numpy(float), "q16": q16, "median": median, "q84": q84,
                           "relative_full_width": (q84-q16)/scale,
                           "relative_full_range": (values.max(1)-values.min(1))/scale})
    result.to_csv(path, index=False)
    return result


def summarize() -> dict:
    old_runs = [OUTPUTS / old_tag(seed) for seed in OLD_SEEDS]
    first_runs = [OUTPUTS / tag for tag in first_terminal_tags()]
    latest = []
    stationary: dict[str, bool] = {}
    for seed in NEW_SEEDS:
        tag, _ = latest_tag(seed)
        if tag is None:
            return {"status": "running", "completed_new_count": 0}
        latest.append(OUTPUTS / tag)
        pointer = TARGET / f"latest_s{seed}.json"
        stationary[str(seed)] = bool(pointer.exists() and json.loads(pointer.read_text()).get("stationary_fnp_gate_pass", False))
    if not all((run / "fnp_grid.csv").exists() for run in old_runs + first_runs + latest):
        return {"status": "running", "completed_new_count": sum((run / "fnp_grid.csv").exists() for run in latest)}
    transform = load_transform_tools()
    members = [*OLD_SEEDS, *FIRST_EXPANSION_SEEDS, *NEW_SEEDS]
    runs = [*old_runs, *first_runs, *latest]
    b24 = project_bands(old_runs, list(OLD_SEEDS), TARGET / "bspace_24_start_only_bands.csv")
    b48 = project_bands(old_runs + first_runs, list(OLD_SEEDS + FIRST_EXPANSION_SEEDS), TARGET / "bspace_48_start_only_bands.csv")
    b96 = project_bands(runs, members, TARGET / "bspace_96_start_only_bands.csv")
    kframes = {}
    for name, selected_runs, selected_members in (("24", old_runs, OLD_SEEDS), ("48", old_runs + first_runs, OLD_SEEDS + FIRST_EXPANSION_SEEDS), ("96", runs, ALL_SEEDS)):
        frame, _ = transform.project_kspace(selected_runs)
        frame.to_csv(TARGET / f"kspace_{name}_start_only_members.csv", index=False)
        kframes[name] = frame

    def widths(frame: pd.DataFrame) -> dict[str, float]:
        out = {}
        for flavor, group in frame[frame.kT <= 2.25].groupby("flavor"):
            group = group.sort_values("kT")
            center = group["median"].to_numpy(float)
            active = center > .05 * np.max(center)
            out[str(flavor)] = float(np.max(((group.q84-group.q16).to_numpy(float) / np.maximum(center, 1e-300))[active]))
        return out

    bstats = {}
    for name, band in (("24", b24), ("48", b48), ("96", b96)):
        active = band["median"] > .05 * np.max(band["median"])
        bstats[name] = {
            "max_active_relative_full_width": float(np.max(band.loc[active, "relative_full_width"])),
            "max_active_relative_full_range": float(np.max(band.loc[active, "relative_full_range"])),
        }
    kw = {name: widths(frame) for name, frame in kframes.items()}
    result = {
        "status": "complete" if all(stationary.values()) else "horizon_reached_without_full_stationarity",
        "study": "lambda1_start_expansion96",
        "member_count": 96,
        "old_member_count": 24,
        "first_expansion_member_count": 24,
        "new_member_count": 48,
        "new_seeds": list(NEW_SEEDS),
        "stationary_fnp_gate_pass_by_seed": stationary,
        "all_new_starts_pass_fnp_stationarity_gate": bool(all(stationary.values())),
        "bspace_start_only_widths": bstats,
        "kspace_start_only_widths": kw,
        "bspace_width_ratio_96_to_48": bstats["96"]["max_active_relative_full_width"] / bstats["48"]["max_active_relative_full_width"],
        "kspace_width_ratio_96_to_48": {flavor: kw["96"][flavor] / kw["48"][flavor] for flavor in kw["96"]},
        "kspace_width_ratio_96_to_24": {flavor: kw["96"][flavor] / kw["24"][flavor] for flavor in kw["96"]},
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
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    write_protocol()
    if args.launch:
        launch_new()
    if args.launch or args.summarize:
        print(json.dumps(summarize(), indent=2))


if __name__ == "__main__":
    main()
