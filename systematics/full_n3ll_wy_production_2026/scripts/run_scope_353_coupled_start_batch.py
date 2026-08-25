#!/usr/bin/env python3
"""Run isolated independent-start fits for the 353-row candidate scope.

The central CSV-normalization fit at seed 303 is already present.  This
launcher runs additional seeds in bounded parallel batches, writes one log per
seed, and records completion without touching frozen production outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
FIT_ROOT = BASE / "reports"


def run_one(args: tuple[int, list[str], Path, Path]) -> dict:
    seed, common, out, log = args
    cmd = list(common) + ["--seed", str(seed), "--out", str(out)]
    with log.open("w") as stream:
        proc = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, env=os.environ.copy())
    status = {"seed": seed, "returncode": int(proc.returncode), "out": str(out), "log": str(log)}
    (out / "launcher_status.json").write_text(json.dumps(status, indent=2) + "\n") if out.exists() else None
    return status


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-start", type=int, default=304)
    ap.add_argument("--count", type=int, default=23)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=5000)
    ap.add_argument("--patience", type=int, default=1000)
    args = ap.parse_args()
    seeds = list(range(args.seed_start, args.seed_start + args.count))
    common = [
        "/home/dustin/miniforge3/envs/pdf-fit/bin/python",
        str(BASE.parent.parent / "train_bt_dnn_v19_localbcurv.py"),
        "--backend-script", str(BASE.parent.parent / "bt_internal_css_backend_v16.py"),
        "--data-dir", str(BASE / "reports/scope_353_fnp_inputs/data_with_csv_uncertainties"),
        "--datasets", "CDF_RUN_1", "CDF_RUN_2", "D0_RUN_1", "E288_200", "E288_300", "E288_400", "E605", "E772", "LHCb_7",
        "--mode", "matched", "--qT-max-over-Q", "1.0", "--tmd-qT-max-over-Q", "0.2",
        "--w-backend", "external",
        "--w-grid", str(BASE / "reports/scope_353_fnp_inputs/scope_353_bspace_w.csv"),
        "--y-grid", str(BASE / "reports/scope_353_fnp_inputs/scope_353_y.csv"),
        "--np-shape-mode", "monotone", "--np-a0", "0.05", "--soft-q-evolution", "none",
        "--fit-dataset-norms", "--lambda-dataset-norm", "1", "--norm-source", "csv", "--ptp-source", "csv",
        "--epochs", str(args.epochs), "--batch-size", "353", "--lr", "2e-3", "--patience", str(args.patience), "--min-delta", "1e-7",
        "--np-width", "48", "--np-cond-width", "32", "--np-blocks", "3",
        "--dtype", "float32", "--device", "cuda", "--log-every", "250",
    ]
    jobs = []
    for seed in seeds:
        out = FIT_ROOT / f"scope_353_coupled_fnp_fit_lambda1_candidate_s{seed}_csvnorm"
        log = FIT_ROOT / f"scope_353_coupled_fnp_fit_lambda1_candidate_s{seed}_csvnorm.log"
        jobs.append((seed, common, out, log))
    status = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [pool.submit(run_one, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            status.append(result)
            print(json.dumps(result), flush=True)
    status.sort(key=lambda x: x["seed"])
    summary = {
        "status": "isolated_scope_353_start_batch_complete" if all(x["returncode"] == 0 for x in status) else "isolated_scope_353_start_batch_incomplete",
        "seeds": seeds,
        "workers": int(args.workers),
        "epochs": int(args.epochs),
        "runs": status,
        "frozen_production_modified": False,
        "promotion_authorized": False,
    }
    out = BASE / "reports/scope_353_coupled_start_batch_status.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
