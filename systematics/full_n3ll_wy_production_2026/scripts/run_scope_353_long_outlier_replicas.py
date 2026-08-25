#!/usr/bin/env python3
"""Long-horizon validation for candidate experimental-replica outliers."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
REPORTS = BASE / "reports"


def one(args: tuple[int, list[str], Path, Path]) -> dict:
    seed, common, out, log = args
    cmd = list(common) + ["--replica-seed", str(seed), "--seed", "303", "--out", str(out)]
    with log.open("w") as stream:
        proc = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, env=os.environ.copy())
    status = {"replica_seed": seed, "returncode": int(proc.returncode), "out": str(out), "log": str(log)}
    if out.exists():
        (out / "launcher_status.json").write_text(json.dumps(status, indent=2) + "\n")
    return status


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=15000)
    ap.add_argument("--patience", type=int, default=2500)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
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
        "--dtype", "float32", "--device", "cuda", "--log-every", "500",
    ]
    jobs = []
    for seed in args.seeds:
        out = REPORTS / f"scope_353_coupled_fnp_fit_lambda1_candidate_r{seed}_csvnorm_long"
        log = REPORTS / f"scope_353_coupled_fnp_fit_lambda1_candidate_r{seed}_csvnorm_long.log"
        jobs.append((seed, common, out, log))
    statuses = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [pool.submit(one, job) for job in jobs]
        for future in as_completed(futures):
            status = future.result(); statuses.append(status); print(json.dumps(status), flush=True)
    statuses.sort(key=lambda x: x["replica_seed"])
    result = {"status": "isolated_scope_353_long_outlier_replica_batch_complete" if all(x["returncode"] == 0 for x in statuses) else "isolated_scope_353_long_outlier_replica_batch_incomplete",
              "seeds": args.seeds, "epochs": args.epochs, "patience": args.patience, "runs": statuses,
              "frozen_production_modified": False, "promotion_authorized": False}
    (REPORTS / "scope_353_long_outlier_replica_batch_status.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
