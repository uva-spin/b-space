#!/usr/bin/env python3
"""Stress-test a regularized N3LL exponentiation on several E288 rows."""

from __future__ import annotations

import concurrent.futures
import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
SCRIPTS = BASE / "scripts"
OUT = BASE / "reports/expcreg_e288_row_stress"
DYTURBO = Path("/home/dustin/src/dyturbo-1.4.2/bin/dyturbo")
DYROOT = DYTURBO.parent.parent
ROWS = ("E288_200:30", "E288_200:44", "E288_200:57", "E288_200:69", "E288_200:80")
REG = 0.5
ORDER = 3


def load_fixed():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    common = importlib.import_module("run_dyturbo_full_n3ll_nnlo_probe")
    common.load_runner()
    spec = importlib.util.spec_from_file_location("fixed_grid_rows", SCRIPTS / "run_fixed_target_n3ll_nnlo_grid.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("fixed grid loader")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def one(row_id: str) -> dict:
    runner = load_fixed()
    dataset, _ = row_id.split(":")
    # Stress rows include high-Q rows outside the 329-row core authority;
    # read the fit-ready table directly for this diagnostic.
    frame = pd.read_csv(runner.DATA_ROOT / f"{dataset}.csv")
    row = frame[frame.row_id.astype(str).eq(row_id)].iloc[0]
    reg_tag = str(REG).replace('.', 'p')
    name = f"{row_id.replace(':', '_')}_order{ORDER}_expcreg{reg_tag}"
    card = OUT / "cards" / f"{name}.in"
    log = OUT / "logs" / f"{name}.log"
    table = DYROOT / f"{name}.txt"
    text = runner.make_card(row, name=name, g1=1.017, calls=100000, seed=20260860 + int(row_id.split(":")[1]), cores=4)
    text += f"\n# Isolated coefficient-exponentiation regularization probe.\norder = {ORDER}\nexpcreg = {REG}\n"
    card.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    card.write_text(text)
    table.unlink(missing_ok=True)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(x for x in ("/home/dustin/miniforge3/envs/pdf-fit/lib", str(DYROOT / "lib"), env.get("LD_LIBRARY_PATH", "")) if x)
    with log.open("w") as handle:
        try:
            proc = subprocess.run([str(DYTURBO), str(card)], cwd=DYROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, timeout=600, check=False)
            return_code, timed_out = proc.returncode, False
        except subprocess.TimeoutExpired:
            return_code, timed_out = None, True
    value = uncertainty = None
    if table.exists():
        for line in table.read_text(errors="replace").splitlines():
            if line.startswith("#") or not line.strip():
                continue
            try:
                value, uncertainty = map(float, line.split()[:2])
                break
            except ValueError:
                continue
    qhalf = runner.qT_half_width(dataset)
    qlow, qhigh = max(runner.QT_DIAGNOSTIC_FLOOR_GEV, float(row.qT) - qhalf), float(row.qT) + qhalf
    dq2 = qhigh * qhigh - qlow * qlow
    dy = float(row.y_High) - float(row.y_Low)
    target_a = runner.TARGETS[dataset][1]
    pred_a = None if value is None else value / (1000.0 * target_a * np.pi * dq2 * dy)
    ratio = None if pred_a is None else pred_a / float(row.A)
    warning_count = log.read_text(errors="replace").count("dequad abnormal termination")
    return {"row_id": row_id, "return_code": return_code, "timed_out": timed_out, "warning_count": warning_count,
            "output_exists": table.exists(), "raw_fb": value, "raw_unc_fb": uncertainty,
            "predicted_A_to_data": ratio, "finite": bool(value is not None and np.isfinite(value)),
            "positive": bool(value is not None and value > 0), "card": str(card), "log": str(log), "table": str(table)}


def main() -> None:
    global REG, ORDER
    ap = argparse.ArgumentParser()
    ap.add_argument("--expcreg", type=float, default=0.5)
    ap.add_argument("--rows", nargs="*", default=list(ROWS))
    ap.add_argument("--order", type=int, default=3)
    args = ap.parse_args()
    REG = float(args.expcreg)
    ORDER = int(args.order)
    OUT.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(one, args.rows))
    status = {"status": "isolated_expcreg_e288_row_stress", "expcreg": REG, "calls_per_component": 100000,
              "rows": results, "frozen_production_modified": False}
    (OUT / "rows_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
