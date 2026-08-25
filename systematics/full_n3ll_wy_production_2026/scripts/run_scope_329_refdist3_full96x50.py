#!/usr/bin/env python3
"""Run and cross the isolated 96-start x 50-replica W+Y propagation.

This is the like-for-like analogue of the production empirical ensemble:
96 independent 1% perturbed starts for the fixed central data, plus 50
independently seeded experimental pseudo-data fits.  The final cross contains
96*50 empirical members per flavor; it is not interpreted as a Gaussian
confidence interval and never writes to the frozen production package. The
long-run variant deliberately uses a 50,000-epoch ceiling so that the W+Y
ensemble has a materially longer optimization horizon than the earlier 10k
diagnostic batch.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
REPORTS = BASE / "reports"
PYTHON = "/home/dustin/miniforge3/envs/pdf-fit/bin/python"
TRAINER = BASE.parent.parent / "train_bt_dnn_v19_localbcurv.py"
BACKEND = BASE.parent.parent / "v23/backends/bt_internal_css_backend_v22_tevatron.py"
DATA = REPORTS / "scope_353_fnp_inputs/data_with_old_effective_errors"
WGRID = REPORTS / "scope_353_fnp_inputs/scope_353_bspace_w_kinematic_corrected_nofidfactor.csv"
YGRID = REPORTS / "scope_353_fnp_inputs/scope_353_y_no_lhcb.csv"
REFERENCE = REPORTS / "baseline_reference_promoted96_allx.csv"
START_STATES = REPORTS / "wy_full96x50_start_states"
REPLICA_STATES = REPORTS / "wy_full96x50_replica_states"
LABEL = "scope_329_refdist3_full96x50_long50k"
TARGET = REPORTS / f"{LABEL}_propagation"
DATASETS = (
    "CDF_RUN_1", "CDF_RUN_2", "D0_RUN_1", "E288_200", "E288_300",
    "E288_400", "E605", "E772", "LHCb_7",
)


def fit_path(kind: str, member: int) -> Path:
    suffix = f"{kind}_{'s' if kind == 'start' else 'r'}{member}"
    return REPORTS / f"{LABEL}_{suffix}"


def log_path(kind: str, member: int) -> Path:
    suffix = f"{kind}_{'s' if kind == 'start' else 'r'}{member}"
    return REPORTS / f"{LABEL}_{suffix}.log"


def command(kind: str, member: int) -> list[str]:
    state_dir = START_STATES if kind == "start" else REPLICA_STATES
    out = fit_path(kind, member)
    # Keep the fit RNG stream separate from the pseudo-data RNG stream for
    # replica members, matching the production design's independent fit seed
    # and experimental-replica seed.
    fit_seed = member if kind == "start" else member + 10000
    cmd = [
        PYTHON, str(TRAINER), "--backend-script", str(BACKEND),
        "--data-dir", str(DATA), "--datasets", *DATASETS, "--mode", "matched",
        "--qT-max-over-Q", "0.2", "--tmd-qT-max-over-Q", "0.2",
        "--w-backend", "external", "--w-grid", str(WGRID),
        "--y-grid", str(YGRID), "--np-shape-mode", "monotone", "--np-a0", "0.05",
        "--soft-q-evolution", "none", "--fit-dataset-norms", "--lambda-dataset-norm", "1",
        "--norm-source", "csv", "--ptp-source", "csv",
        "--lambda-fnp-tail", "1", "--fnp-tail-bmin", "6", "--fnp-tail-target", "0.05",
        "--lambda-fnp-reference-distance", "3",
        "--fnp-reference-distance-csv", str(REFERENCE),
        "--fnp-reference-distance-bmin", "0.1", "--fnp-reference-distance-bmax", "8",
        "--epochs", "50000", "--batch-size", "10000", "--lr", "2e-5",
        "--patience", "0", "--min-delta", "1e-7", "--np-width", "48",
        "--np-cond-width", "32", "--np-blocks", "3", "--dtype", "float32",
        "--device", "cuda", "--log-every", "500", "--seed", str(fit_seed),
        "--init-model-state", str(state_dir / f"state_s{member}.pt"), "--out", str(out),
    ]
    if kind == "replica":
        cmd += ["--replica-seed", str(member)]
    return cmd


def run_one(kind: str, member: int, force: bool) -> dict:
    out = fit_path(kind, member)
    log = log_path(kind, member)
    if (out / "metrics.json").exists() and not force:
        return {"kind": kind, "member": member, "returncode": 0,
                "status": "existing", "out": str(out), "log": str(log)}
    out.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        proc = subprocess.run(command(kind, member), stdout=stream,
                              stderr=subprocess.STDOUT, env=os.environ.copy())
    result = {"kind": kind, "member": member, "returncode": int(proc.returncode),
              "status": "complete" if proc.returncode == 0 else "failed",
              "out": str(out), "log": str(log)}
    (out / "launcher_status.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def run_batch(kind: str, members: list[int], workers: int, force: bool) -> dict:
    results = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(run_one, kind, member, force) for member in members]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result), flush=True)
    results.sort(key=lambda row: row["member"])
    return {"kind": kind, "members": members, "workers": workers, "runs": results,
            "complete": all(row["returncode"] == 0 for row in results)}


def load_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path / "fnp_debug_grid.csv")
    frame = frame[np.isclose(frame["x"].to_numpy(float), 0.1)].sort_values("bT")
    if frame.empty or not np.all(np.isfinite(frame["F_NP"].to_numpy(float))):
        raise RuntimeError(f"invalid F_NP curve: {path}")
    return frame["bT"].to_numpy(float), np.log(np.maximum(frame["F_NP"].to_numpy(float), 1e-30))


def cross(start_members: list[int], replica_members: list[int]) -> dict:
    starts, replicas = [], []
    b_ref = None
    for member in start_members:
        b, curve = load_curve(fit_path("start", member))
        b_ref = b if b_ref is None else b_ref
        if not np.allclose(b, b_ref):
            raise RuntimeError(f"start grid mismatch for {member}")
        starts.append(curve)
    for member in replica_members:
        b, curve = load_curve(fit_path("replica", member))
        b_ref = b if b_ref is None else b_ref
        if not np.allclose(b, b_ref):
            raise RuntimeError(f"replica grid mismatch for {member}")
        replicas.append(curve)
    starts = np.asarray(starts, float)
    replicas = np.asarray(replicas, float)
    residuals = replicas - np.median(replicas, axis=0)
    crossed = (starts[:, None, :] + residuals[None, :, :]).reshape(-1, starts.shape[1])
    values = np.exp(crossed)
    q16, q50, q84 = np.quantile(values, [0.16, 0.50, 0.84], axis=0)
    s16, s50, s84 = np.quantile(np.exp(starts), [0.16, 0.50, 0.84], axis=0)
    r16, r50, r84 = np.quantile(np.exp(replicas), [0.16, 0.50, 0.84], axis=0)
    TARGET.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame({
        "bT": b_ref,
        "FNP_start_q16": s16, "FNP_start_q50": s50, "FNP_start_q84": s84,
        "FNP_replica_q16": r16, "FNP_replica_q50": r50, "FNP_replica_q84": r84,
        "FNP_crossed_q16": q16, "FNP_crossed_q50": q50, "FNP_crossed_q84": q84,
        "relative_crossed_full_width": (q84 - q16) / np.maximum(q50, 1e-30),
    })
    table.to_csv(TARGET / "fnp_start_replica_quantiles_x0p1.csv", index=False)
    long_rows = []
    for member, curve in enumerate(values):
        long_rows.extend({"member": member, "x": 0.1, "bT": float(b), "F_NP": float(f)}
                         for b, f in zip(b_ref, curve))
    pd.DataFrame(long_rows).to_csv(TARGET / "fnp_start_replica_crossed_long_x0p1.csv", index=False)
    active = (b_ref <= 4.0) & (q50 > 0.05 * np.max(q50[b_ref <= 4.0]))
    summary = {
        "status": "isolated_scope_329_refdist3_full96x50_cross_complete",
        "start_count": len(starts), "replica_count": len(replicas),
        "crossed_member_count": len(crossed),
        "cross_rule": "log(F_NP) start curves plus centered log(F_NP) replica residuals",
        "quantiles": [0.16, 0.50, 0.84],
        "max_relative_crossed_full_width_active": float(np.max(table.loc[active, "relative_crossed_full_width"])),
        "median_relative_crossed_full_width_active": float(np.median(table.loc[active, "relative_crossed_full_width"])),
        "interpretation": "empirical full procedural envelope; not assigned a Gaussian confidence level",
        "protocol": {"lambda_reference_distance": 3.0, "reference_bmin": 0.1,
                     "reference_bmax": 8.0, "lambda_tail": 1.0,
                     "tail_bmin": 6.0, "tail_target": 0.05,
                     "y_scope": "non-LHCb-Y", "data": "old effective errors"},
        "artifacts": {"quantiles": str(TARGET / "fnp_start_replica_quantiles_x0p1.csv"),
                      "crossed_long": str(TARGET / "fnp_start_replica_crossed_long_x0p1.csv")},
        "frozen_production_modified": False, "promotion_authorized": False,
    }
    (TARGET / "cross_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=("starts", "replicas", "both", "cross"), default="both")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    starts = list(range(303, 399))
    replicas = list(range(1001, 1051))
    TARGET.mkdir(parents=True, exist_ok=True)
    if args.kind in ("starts", "both"):
        result = run_batch("start", starts, args.workers, args.force)
        (TARGET / "starts_status.json").write_text(json.dumps(result, indent=2) + "\n")
        if not result["complete"]:
            raise SystemExit("start batch incomplete")
    if args.kind in ("replicas", "both"):
        result = run_batch("replica", replicas, args.workers, args.force)
        (TARGET / "replicas_status.json").write_text(json.dumps(result, indent=2) + "\n")
        if not result["complete"]:
            raise SystemExit("replica batch incomplete")
    if args.kind == "cross" or args.kind == "both":
        cross(starts, replicas)


if __name__ == "__main__":
    main()
