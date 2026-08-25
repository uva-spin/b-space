#!/usr/bin/env python3
"""Run isolated 353-row F_NP tail-penalty pilots.

These pilots diagnose the missing large-b damping in the coupled v19
candidate.  They are not production runs and never write to frozen paths.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
REPORTS = BASE / "reports"
PYTHON = "/home/dustin/miniforge3/envs/pdf-fit/bin/python"


def run_one(lam: float, epochs: int, patience: int) -> dict:
    label = str(lam).replace(".", "p").replace("-", "m")
    out = REPORTS / f"scope_353_tail_penalty_lambda{label}_s303"
    log = REPORTS / f"scope_353_tail_penalty_lambda{label}_s303.log"
    cmd = [
        PYTHON, str(BASE.parent.parent / "train_bt_dnn_v19_localbcurv.py"),
        "--backend-script", str(BASE.parent.parent / "bt_internal_css_backend_v16.py"),
        "--data-dir", str(REPORTS / "scope_353_fnp_inputs/data_with_csv_uncertainties"),
        "--datasets", "CDF_RUN_1", "CDF_RUN_2", "D0_RUN_1", "E288_200", "E288_300",
        "E288_400", "E605", "E772", "LHCb_7",
        "--mode", "matched", "--qT-max-over-Q", "1.0", "--tmd-qT-max-over-Q", "0.2",
        "--w-backend", "external",
        "--w-grid", str(REPORTS / "scope_353_fnp_inputs/scope_353_bspace_w.csv"),
        "--y-grid", str(REPORTS / "scope_353_fnp_inputs/scope_353_y.csv"),
        "--np-shape-mode", "monotone", "--np-a0", "0.05", "--soft-q-evolution", "none",
        "--fit-dataset-norms", "--lambda-dataset-norm", "1", "--norm-source", "csv", "--ptp-source", "csv",
        "--lambda-fnp-tail", str(lam), "--fnp-tail-bmin", "6.0", "--fnp-tail-target", "0.05",
        "--epochs", str(epochs), "--batch-size", "353", "--lr", "2e-3", "--patience", str(patience),
        "--min-delta", "1e-7", "--np-width", "48", "--np-cond-width", "32", "--np-blocks", "3",
        "--dtype", "float32", "--device", "cuda", "--log-every", "500", "--seed", "303",
        "--out", str(out),
    ]
    with log.open("w") as stream:
        proc = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, env=os.environ.copy())
    return {"lambda_fnp_tail": lam, "returncode": proc.returncode, "out": str(out), "log": str(log)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambdas", nargs="+", type=float, default=[1.0, 10.0, 100.0])
    ap.add_argument("--epochs", type=int, default=5000)
    ap.add_argument("--patience", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--tag", default="scope_353_tail_penalty_pilots")
    args = ap.parse_args()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(run_one, lam, args.epochs, args.patience) for lam in args.lambdas]
        results = [future.result() for future in as_completed(futures)]
    results.sort(key=lambda x: x["lambda_fnp_tail"])
    payload = {"status": "isolated_scope_353_tail_penalty_pilots_complete" if all(x["returncode"] == 0 for x in results) else "isolated_scope_353_tail_penalty_pilots_incomplete", "epochs": args.epochs, "patience": args.patience, "runs": results, "frozen_production_modified": False, "promotion_authorized": False}
    out = REPORTS / args.tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "status.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
