#!/usr/bin/env python3
"""Automatically screen the lambda_tail=10 starts and, if viable, run replicas.

This is an isolated gate for the stronger-tail follow-up.  It deliberately
does not create production outputs or alter frozen files.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
REPORTS = BASE / "reports"
PYTHON = "/home/dustin/miniforge3/envs/pdf-fit/bin/python"
LABEL = "scope_353_tail10"
PREFIX = f"{LABEL}_coupled_fnp_fit"
STARTS = list(range(303, 327))
REPLICAS = list(range(1001, 1051))


def wait_status(path: Path, expected: str, stream) -> dict:
    while True:
        if path.exists():
            try:
                d = json.loads(path.read_text())
            except json.JSONDecodeError:
                d = {}
            if d.get("status") == expected:
                return d
            if str(d.get("status", "")).endswith("incomplete"):
                raise RuntimeError(f"incomplete batch: {path}")
        stream.write(f"waiting for {path.name}\n"); stream.flush(); time.sleep(30)


def run(cmd: list[str], stream) -> None:
    stream.write("running: " + " ".join(cmd) + "\n"); stream.flush()
    p = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT)
    if p.returncode:
        raise RuntimeError(f"failed command ({p.returncode}): {cmd[0]}")


def objective(seed: int, long: bool = False) -> float:
    suffix = "_long" if long else ""
    p = REPORTS / f"{PREFIX}_s{seed}_csvnorm{suffix}/metrics.json"
    d = json.loads(p.read_text())["train"]
    return float(d["best_chi2_like"] / d["n_points"])


def tail8(seed: int, long: bool = False) -> float:
    suffix = "_long" if long else ""
    p = REPORTS / f"{PREFIX}_s{seed}_csvnorm{suffix}/fnp_debug_grid.csv"
    d = pd.read_csv(p)
    g = d[np.isclose(d["x"], 0.1)]
    return float(g.iloc[(g["bT"] - 8.0).abs().argmin()]["F_NP"])


def main() -> None:
    log_path = REPORTS / f"{LABEL}_screen_supervisor.log"
    with log_path.open("a") as stream:
        wait_status(REPORTS / f"{LABEL}_starts_batch_status.json", f"isolated_{LABEL}_starts_complete", stream)
        rows = [{"seed": s, "objective_per_row": objective(s), "FNP_b8_x0p1": tail8(s)} for s in STARTS]
        vals = np.asarray([r["objective_per_row"] for r in rows])
        tails = np.asarray([r["FNP_b8_x0p1"] for r in rows])
        screen = {
            "label": LABEL, "lambda_tail": 10.0, "bmin": 6.0, "target": 0.05,
            "start_count": len(rows), "rows": rows,
            "objective_median": float(np.median(vals)), "objective_max": float(np.max(vals)),
            "FNP_b8_median": float(np.median(tails)), "FNP_b8_q84": float(np.quantile(tails, .84)),
            "FNP_b8_max": float(np.max(tails)),
            "screen_rule": "median objective <=0.06, max objective <=0.08, q84 FNP(b=8)<=0.10, max FNP(b=8)<=0.20",
        }
        screen["passed"] = bool(screen["objective_median"] <= .06 and screen["objective_max"] <= .08
                                  and screen["FNP_b8_q84"] <= .10 and screen["FNP_b8_max"] <= .20)
        (REPORTS / f"{LABEL}_start_screen.json").write_text(json.dumps(screen, indent=2) + "\n")
        if screen["passed"]:
            run([PYTHON, str(BASE / "scripts/run_scope_353_tail_grid_batch.py"), "--kind", "replicas",
                 "--seeds", *map(str, REPLICAS), "--epochs", "5000", "--patience", "1000", "--workers", "4",
                 "--label", LABEL, "--lambda-tail", "10", "--bmin", "6", "--target", ".05"], stream)
            wait_status(REPORTS / f"{LABEL}_replicas_batch_status.json", f"isolated_{LABEL}_replicas_complete", stream)
        else:
            stream.write("start screen failed; replica batch not launched\n")
        final = {"status": f"isolated_{LABEL}_screen_complete", "screen": screen,
                 "replica_batch_launched": bool(screen["passed"]),
                 "frozen_production_modified": False, "promotion_authorized": False}
        (REPORTS / f"{LABEL}_screen_final_status.json").write_text(json.dumps(final, indent=2) + "\n")
        stream.write(json.dumps(final, indent=2) + "\n")


if __name__ == "__main__":
    main()
