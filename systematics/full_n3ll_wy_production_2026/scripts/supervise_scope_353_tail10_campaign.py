#!/usr/bin/env python3
"""Complete the isolated lambda_tail=10 start/replica propagation.

The start screen found one poor optimizer basin.  This supervisor long-
revalidates that basin, then runs all replicas and applies the same robust
objective gate used by the tail1 campaign.  All outputs have a distinct
tail10 prefix; frozen production inputs are never written.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np


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


def objective(kind: str, seed: int, long: bool = False) -> tuple[float, int]:
    lead = "s" if kind == "starts" else "r"
    suffix = "_long" if long else ""
    p = REPORTS / f"{PREFIX}_{lead}{seed}_csvnorm{suffix}/metrics.json"
    d = json.loads(p.read_text())["train"]
    return float(d["best_chi2_like"] / d["n_points"]), int(d["epochs_run"])


def select_outliers(kind: str, seeds: list[int]) -> tuple[list[int], dict]:
    rows = [{"seed": s, "objective_per_row": objective(kind, s, False)[0],
             "epochs_run": objective(kind, s, False)[1]} for s in seeds]
    vals = np.asarray([x["objective_per_row"] for x in rows], float)
    med = float(np.median(vals)); mad = float(np.median(np.abs(vals - med)))
    threshold = med + max(3.0 * 1.4826 * mad, 0.03)
    selected = [int(x["seed"]) for x in rows
                if x["objective_per_row"] > threshold or x["epochs_run"] < 0.9 * 5000]
    return selected, {"median_objective_per_row": med, "mad_objective_per_row": mad,
                      "robust_threshold": threshold, "rows": rows,
                      "selected_seeds": selected}


def gate(kind: str, seeds: list[int], long_seeds: list[int]) -> tuple[list[int], dict]:
    rows = [{"seed": s, "objective_per_row": objective(kind, s, s in long_seeds)[0]}
            for s in seeds]
    vals = np.asarray([x["objective_per_row"] for x in rows], float)
    med = float(np.median(vals)); mad = float(np.median(np.abs(vals - med)))
    threshold = med + max(3.0 * 1.4826 * mad, 0.03)
    return [int(x["seed"]) for x in rows if x["objective_per_row"] > threshold], {
        "median": med, "mad": mad, "threshold": threshold, "rows": rows}


def main() -> None:
    log_path = REPORTS / f"{LABEL}_campaign_supervisor.log"
    with log_path.open("a") as stream:
        # Revalidate the one bad start identified by the all-start screen.
        run([PYTHON, str(BASE / "scripts/run_scope_353_tail_grid_batch.py"), "--kind", "starts",
             "--seeds", "310", "--epochs", "15000", "--patience", "2500", "--workers", "1",
             "--label", LABEL, "--lambda-tail", "10", "--bmin", "6", "--target", ".05", "--long"], stream)
        wait_status(REPORTS / f"{LABEL}_starts_long_batch_status.json", f"isolated_{LABEL}_starts_long_complete", stream)
        start_long = [310]

        start_out, start_selection = select_outliers("starts", STARTS)
        # Every selected start must have a long output before the crossed
        # ensemble is summarized.  The screen can select early-stop seeds in
        # addition to the initially known poor basin (seed 310).
        start_out = sorted(set(start_out) | {310})
        run([PYTHON, str(BASE / "scripts/run_scope_353_tail_grid_batch.py"), "--kind", "starts",
             "--seeds", *map(str, start_out), "--epochs", "15000", "--patience", "2500", "--workers", "4",
             "--label", LABEL, "--lambda-tail", "10", "--bmin", "6", "--target", ".05", "--long"], stream)
        wait_status(REPORTS / f"{LABEL}_starts_long_batch_status.json", f"isolated_{LABEL}_starts_long_complete", stream)
        (REPORTS / f"{LABEL}_outlier_selection.json").write_text(json.dumps({
            "starts": start_selection, "replicas": None, "long_start_seeds": start_out,
            "frozen_production_modified": False, "promotion_authorized": False}, indent=2) + "\n")

        run([PYTHON, str(BASE / "scripts/run_scope_353_tail_grid_batch.py"), "--kind", "replicas",
             "--seeds", *map(str, REPLICAS), "--epochs", "5000", "--patience", "1000", "--workers", "4",
             "--label", LABEL, "--lambda-tail", "10", "--bmin", "6", "--target", ".05"], stream)
        wait_status(REPORTS / f"{LABEL}_replicas_batch_status.json", f"isolated_{LABEL}_replicas_complete", stream)
        replica_out, replica_selection = select_outliers("replicas", REPLICAS)
        (REPORTS / f"{LABEL}_outlier_selection.json").write_text(json.dumps({
            "starts": start_selection, "replicas": replica_selection,
            "long_start_seeds": start_out, "long_replica_seeds": replica_out,
            "frozen_production_modified": False, "promotion_authorized": False}, indent=2) + "\n")

        if replica_out:
            run([PYTHON, str(BASE / "scripts/run_scope_353_tail_grid_batch.py"), "--kind", "replicas",
                 "--seeds", *map(str, replica_out), "--epochs", "15000", "--patience", "2500", "--workers", "4",
                 "--label", LABEL, "--lambda-tail", "10", "--bmin", "6", "--target", ".05", "--long"], stream)
            wait_status(REPORTS / f"{LABEL}_replicas_long_batch_status.json", f"isolated_{LABEL}_replicas_long_complete", stream)

        run([PYTHON, str(BASE / "scripts/summarize_scope_353_replica_batch.py"), "--prefix", PREFIX,
             "--start-seeds", *map(str, STARTS), "--long-start-seeds", *map(str, start_out),
             "--replica-seeds", *map(str, REPLICAS), "--long-replica-seeds", *map(str, replica_out),
             "--tag", f"{LABEL}_start_replica_propagation_full"], stream)
        reject_s, gate_s = gate("starts", STARTS, start_out)
        reject_r, gate_r = gate("replicas", REPLICAS, replica_out)
        (REPORTS / f"{LABEL}_fit_quality_gate.json").write_text(json.dumps({
            "rejected_start_seeds": reject_s, "rejected_replica_seeds": reject_r,
            "start_gate": gate_s, "replica_gate": gate_r,
            "frozen_production_modified": False, "promotion_authorized": False}, indent=2) + "\n")
        run([PYTHON, str(BASE / "scripts/summarize_scope_353_replica_batch.py"), "--prefix", PREFIX,
             "--start-seeds", *map(str, STARTS), "--long-start-seeds", *map(str, start_out),
             "--replica-seeds", *map(str, REPLICAS), "--long-replica-seeds", *map(str, replica_out),
             "--exclude-start-seeds", *map(str, reject_s), "--exclude-replica-seeds", *map(str, reject_r),
             "--tag", f"{LABEL}_start_replica_propagation_accepted"], stream)
        run([PYTHON, str(BASE / "scripts/plot_scope_353_fnp_start_replica_fig2_fig6.py"),
             "--source", str(REPORTS / f"{LABEL}_start_replica_propagation_full/fnp_start_replica_crossed_long_x0p1.csv"),
             "--target", str(REPORTS / f"{LABEL}_final_fig2_fig6_full")], stream)
        run([PYTHON, str(BASE / "scripts/plot_scope_353_fnp_start_replica_fig2_fig6.py"),
             "--source", str(REPORTS / f"{LABEL}_start_replica_propagation_accepted/fnp_start_replica_crossed_long_x0p1.csv"),
             "--target", str(REPORTS / f"{LABEL}_final_fig2_fig6_accepted"),
             "--start-count", str(len(STARTS) - len(reject_s)),
             "--replica-count", str(len(REPLICAS) - len(reject_r))], stream)
        final = {"status": f"isolated_{LABEL}_campaign_complete", "model": "v19 monotone FiLM with lambda_fnp_tail=10, bmin=6, target=0.05",
                 "start_count": len(STARTS), "replica_count": len(REPLICAS), "long_start_seeds": start_out,
                 "long_replica_seeds": replica_out, "rejected_start_seeds": reject_s,
                 "rejected_replica_seeds": reject_r, "accepted_start_count": len(STARTS) - len(reject_s),
                 "accepted_replica_count": len(REPLICAS) - len(reject_r),
                 "full_crossed_member_count": len(STARTS) * len(REPLICAS),
                 "accepted_crossed_member_count": (len(STARTS) - len(reject_s)) * (len(REPLICAS) - len(reject_r)),
                 "frozen_production_modified": False, "promotion_authorized": False}
        (REPORTS / f"{LABEL}_campaign_final_status.json").write_text(json.dumps(final, indent=2) + "\n")
        stream.write(json.dumps(final, indent=2) + "\n")


if __name__ == "__main__":
    main()
