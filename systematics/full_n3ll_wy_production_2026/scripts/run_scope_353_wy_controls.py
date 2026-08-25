#!/usr/bin/env python3
"""Matched W-only/W+Y controls on the identical isolated 353-row scope.

This is a diagnostic launcher.  It deliberately writes only under the new
full_n3ll_wy_production_2026/reports tree and never touches frozen production
outputs.  The only difference between the two controls is whether the fixed
external Y grid is passed to the trainer.
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
TRAINER = BASE.parent.parent / "train_bt_dnn_v19_localbcurv.py"
BACKEND = BASE.parent.parent / "bt_internal_css_backend_v16.py"
DATA = REPORTS / "scope_353_fnp_inputs/data_with_csv_uncertainties"
WGRID = REPORTS / "scope_353_fnp_inputs/scope_353_bspace_w.csv"
YGRID = REPORTS / "scope_353_fnp_inputs/scope_353_y.csv"
DATASETS = ("CDF_RUN_1", "CDF_RUN_2", "D0_RUN_1", "E288_200", "E288_300",
            "E288_400", "E605", "E772", "LHCb_7")


def run_one(mode: str, seed: int, epochs: int, patience: int, label: str,
            lambda_tail: float, bmin: float, target: float, qmax: float,
            data_dir: Path, w_grid: Path, lr: float, batch_size: int,
            min_delta: float, init_model_state: Path | None,
            backend_script: Path, y_grid: Path,
            init_state_dir: Path | None,
            lambda_reference_distance: float,
            reference_distance_csv: Path | None,
            reference_bmin: float,
            reference_bmax: float) -> dict:
    out = REPORTS / f"{label}_{mode}_s{seed}"
    log = REPORTS / f"{label}_{mode}_s{seed}.log"
    cmd = [
        PYTHON, str(TRAINER), "--backend-script", str(backend_script),
        "--data-dir", str(data_dir), "--datasets", *DATASETS, "--mode", "matched",
        "--qT-max-over-Q", str(qmax), "--tmd-qT-max-over-Q", "0.2",
        "--w-backend", "external", "--w-grid", str(w_grid),
        "--np-shape-mode", "monotone", "--np-a0", "0.05", "--soft-q-evolution", "none",
        "--fit-dataset-norms", "--lambda-dataset-norm", "1", "--norm-source", "csv",
        "--ptp-source", "csv", "--lambda-fnp-tail", str(lambda_tail),
        "--fnp-tail-bmin", str(bmin), "--fnp-tail-target", str(target),
        "--lambda-fnp-reference-distance", str(lambda_reference_distance),
        "--fnp-reference-distance-bmin", str(reference_bmin),
        "--fnp-reference-distance-bmax", str(reference_bmax),
        "--epochs", str(epochs), "--batch-size", str(batch_size), "--lr", str(lr),
        "--patience", str(patience), "--min-delta", str(min_delta), "--np-width", "48",
        "--np-cond-width", "32", "--np-blocks", "3", "--dtype", "float32",
        "--device", "cuda", "--log-every", "500", "--seed", str(seed), "--out", str(out),
    ]
    if mode == "wy":
        cmd += ["--y-grid", str(y_grid)]
    else:
        cmd += ["--allow-zero-y-in-matched"]
    selected_state = (init_state_dir / f"state_s{seed}.pt"
                      if init_state_dir is not None else init_model_state)
    if selected_state is not None:
        cmd += ["--init-model-state", str(selected_state)]
    if lambda_reference_distance > 0.0:
        if reference_distance_csv is None:
            raise ValueError("positive lambda_reference_distance requires reference_distance_csv")
        cmd += ["--fnp-reference-distance-csv", str(reference_distance_csv)]
    with log.open("w", encoding="utf-8") as stream:
        proc = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, env=os.environ.copy())
    status = {"mode": mode, "seed": seed, "returncode": int(proc.returncode),
              "out": str(out), "log": str(log)}
    if out.exists():
        (out / "launcher_status.json").write_text(json.dumps(status, indent=2) + "\n")
    return status


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", choices=("wonly", "wy"), default=("wonly", "wy"))
    ap.add_argument("--seeds", nargs="+", type=int, default=(303, 304, 305))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=5000)
    ap.add_argument("--patience", type=int, default=1000)
    ap.add_argument("--label", default="scope_353_wy_control_lambda10")
    ap.add_argument("--lambda-tail", type=float, default=10.0)
    ap.add_argument("--bmin", type=float, default=6.0)
    ap.add_argument("--target", type=float, default=0.05)
    ap.add_argument("--qmax", type=float, default=1.0,
                    help="qT/Q cut; 0.2 gives the 329-row low-qT core")
    ap.add_argument("--data-dir", type=Path, default=DATA)
    ap.add_argument("--w-grid", type=Path, default=WGRID)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch-size", type=int, default=353)
    ap.add_argument("--min-delta", type=float, default=1e-7)
    ap.add_argument("--init-model-state", type=Path, default=None)
    ap.add_argument("--backend-script", type=Path, default=BACKEND)
    ap.add_argument("--y-grid", type=Path, default=YGRID)
    ap.add_argument("--init-state-dir", type=Path, default=None,
                    help="Directory containing per-seed state_sSEED.pt warm starts.")
    ap.add_argument("--lambda-reference-distance", type=float, default=0.0,
                    help="Optional empirical baseline F_NP reference-distance strength.")
    ap.add_argument("--reference-distance-csv", type=Path, default=None,
                    help="CSV with x,bT,F_NP for --lambda-reference-distance.")
    ap.add_argument("--reference-bmin", type=float, default=0.10)
    ap.add_argument("--reference-bmax", type=float, default=2.0)
    args = ap.parse_args()
    jobs = [(mode, seed) for mode in args.modes for seed in args.seeds]
    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(run_one, mode, seed, args.epochs, args.patience,
                                args.label, args.lambda_tail, args.bmin, args.target, args.qmax,
                                args.data_dir, args.w_grid, args.lr, args.batch_size, args.min_delta,
                                args.init_model_state, args.backend_script, args.y_grid,
                                args.init_state_dir, args.lambda_reference_distance,
                                args.reference_distance_csv, args.reference_bmin,
                                args.reference_bmax)
                   for mode, seed in jobs]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result), flush=True)
    results.sort(key=lambda x: (x["mode"], x["seed"]))
    payload = {
        "status": "complete" if all(x["returncode"] == 0 for x in results) else "incomplete",
        "label": args.label, "modes": args.modes, "seeds": args.seeds,
        "epochs": args.epochs, "patience": args.patience, "lambda_tail": args.lambda_tail,
        "fnp_tail_bmin": args.bmin, "fnp_tail_target": args.target, "qT_max_over_Q": args.qmax,
        "data_dir": str(args.data_dir), "w_grid": str(args.w_grid), "lr": args.lr, "batch_size": args.batch_size,
        "min_delta": args.min_delta, "runs": results,
        "init_model_state": str(args.init_model_state) if args.init_model_state else None,
        "init_state_dir": str(args.init_state_dir) if args.init_state_dir else None,
        "lambda_reference_distance": args.lambda_reference_distance,
        "reference_distance_csv": str(args.reference_distance_csv) if args.reference_distance_csv else None,
        "reference_distance_bmin": args.reference_bmin,
        "reference_distance_bmax": args.reference_bmax,
        "backend_script": str(args.backend_script),
        "y_grid": str(args.y_grid),
        "purpose": "same 353 rows, same external W, same optimizer; W-only versus fixed W+Y",
        "frozen_production_modified": False, "promotion_authorized": False,
    }
    path = REPORTS / f"{args.label}_status.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
