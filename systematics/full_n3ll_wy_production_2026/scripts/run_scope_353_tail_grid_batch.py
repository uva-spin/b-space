#!/usr/bin/env python3
"""Run an isolated scope-353 FiLM batch for a selected F_NP tail penalty.

This is deliberately separate from the completed tail1 campaign so that each
regularization strength has its own immutable output prefix and status file.
Frozen production inputs are read-only.
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


def run_one(kind: str, seed: int, epochs: int, patience: int, workers_tag: str,
            lambda_tail: float, bmin: float, target: float, long_run: bool) -> dict:
    lead = "s" if kind == "starts" else "r"
    suffix = "_long" if long_run else ""
    label = workers_tag
    out = REPORTS / f"{label}_coupled_fnp_fit_{lead}{seed}_csvnorm{suffix}"
    log = REPORTS / f"{label}_coupled_fnp_fit_{lead}{seed}_csvnorm{suffix}.log"
    cmd = [
        PYTHON, str(BASE.parent.parent / "train_bt_dnn_v19_localbcurv.py"),
        "--backend-script", str(BASE.parent.parent / "bt_internal_css_backend_v16.py"),
        "--data-dir", str(REPORTS / "scope_353_fnp_inputs/data_with_csv_uncertainties"),
        "--datasets", "CDF_RUN_1", "CDF_RUN_2", "D0_RUN_1", "E288_200", "E288_300",
        "E288_400", "E605", "E772", "LHCb_7", "--mode", "matched",
        "--qT-max-over-Q", "1.0", "--tmd-qT-max-over-Q", "0.2", "--w-backend", "external",
        "--w-grid", str(REPORTS / "scope_353_fnp_inputs/scope_353_bspace_w.csv"),
        "--y-grid", str(REPORTS / "scope_353_fnp_inputs/scope_353_y.csv"),
        "--np-shape-mode", "monotone", "--np-a0", "0.05", "--soft-q-evolution", "none",
        "--fit-dataset-norms", "--lambda-dataset-norm", "1", "--norm-source", "csv", "--ptp-source", "csv",
        "--lambda-fnp-tail", str(lambda_tail), "--fnp-tail-bmin", str(bmin), "--fnp-tail-target", str(target),
        "--epochs", str(epochs), "--batch-size", "353", "--lr", "2e-3", "--patience", str(patience),
        "--min-delta", "1e-7", "--np-width", "48", "--np-cond-width", "32", "--np-blocks", "3",
        "--dtype", "float32", "--device", "cuda", "--log-every", "500", "--out", str(out),
    ]
    if kind == "replicas":
        cmd += ["--replica-seed", str(seed), "--seed", "303"]
    else:
        cmd += ["--seed", str(seed)]
    with log.open("w") as stream:
        proc = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, env=os.environ.copy())
    if out.exists():
        (out / "launcher_status.json").write_text(json.dumps({"kind": kind, "seed": seed, "returncode": proc.returncode}, indent=2) + "\n")
    return {"kind": kind, "seed": seed, "returncode": proc.returncode, "out": str(out), "log": str(log)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["starts", "replicas"], required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=5000)
    ap.add_argument("--patience", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--label", required=True, help="output prefix, e.g. scope_353_tail10")
    ap.add_argument("--lambda-tail", type=float, required=True)
    ap.add_argument("--bmin", type=float, default=6.0)
    ap.add_argument("--target", type=float, default=0.05)
    ap.add_argument("--long", action="store_true")
    args = ap.parse_args()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(run_one, args.kind, seed, args.epochs, args.patience, args.label,
                                args.lambda_tail, args.bmin, args.target, args.long) for seed in args.seeds]
        results = [f.result() for f in as_completed(futures)]
    results.sort(key=lambda x: x["seed"])
    status = f"isolated_{args.label}_{args.kind}_{'long_' if args.long else ''}complete" if all(x["returncode"] == 0 for x in results) else f"isolated_{args.label}_{args.kind}_{'long_' if args.long else ''}incomplete"
    payload = {"status": status, "kind": args.kind, "label": args.label, "seeds": args.seeds,
               "lambda_tail": args.lambda_tail, "fnp_tail_bmin": args.bmin,
               "fnp_tail_target": args.target, "epochs": args.epochs, "patience": args.patience,
               "workers": args.workers, "runs": results, "frozen_production_modified": False,
               "promotion_authorized": False}
    path = REPORTS / f"{args.label}_{args.kind}_{'long_' if args.long else ''}batch_status.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
