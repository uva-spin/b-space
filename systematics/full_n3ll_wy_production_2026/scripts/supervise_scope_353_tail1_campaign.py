#!/usr/bin/env python3
"""Fully automate the corrected lambda-tail=1 scope-353 campaign."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np


BASE = Path(__file__).resolve().parents[1]
REPORTS = BASE / "reports"
PYTHON = "/home/dustin/miniforge3/envs/pdf-fit/bin/python"
PREFIX = "scope_353_tail1_coupled_fnp_fit"
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


def objective(kind: str, seed: int, long: bool) -> tuple[float, int]:
    lead = "s" if kind == "starts" else "r"
    suffix = "_long" if long else ""
    p = REPORTS / f"{PREFIX}_{lead}{seed}_csvnorm{suffix}/metrics.json"
    d = json.loads(p.read_text())["train"]
    return float(d["best_chi2_like"] / d["n_points"]), int(d["epochs_run"])


def select_outliers(kind: str, seeds: list[int]) -> tuple[list[int], dict]:
    rows = [{"seed": s, "objective_per_row": objective(kind, s, False)[0], "epochs_run": objective(kind, s, False)[1]} for s in seeds]
    vals = np.asarray([x["objective_per_row"] for x in rows], float)
    med = float(np.median(vals)); mad = float(np.median(np.abs(vals - med)))
    threshold = med + max(3.0 * 1.4826 * mad, 0.03)
    selected = [int(x["seed"]) for x in rows if x["objective_per_row"] > threshold or x["epochs_run"] < 0.9 * 5000]
    return selected, {"median_objective_per_row": med, "mad_objective_per_row": mad, "robust_threshold": threshold, "rows": rows, "selected_seeds": selected}


def main() -> None:
    log_path = REPORTS / "scope_353_tail1_campaign_supervisor.log"
    with log_path.open("a") as stream:
        stream.write("tail1 campaign supervisor started\n"); stream.flush()
        wait_status(REPORTS / "scope_353_tail1_starts_batch_status.json", "isolated_scope_353_tail1_starts_complete", stream)
        run([PYTHON, str(BASE / "scripts/summarize_scope_353_start_batch.py"), "--prefix", PREFIX, "--seeds", *map(str, STARTS), "--tag", "scope_353_tail1_start_summary_raw"], stream)
        run([PYTHON, str(BASE / "scripts/run_scope_353_tail1_batch.py"), "--kind", "replicas", "--seeds", *map(str, REPLICAS), "--epochs", "5000", "--patience", "1000", "--workers", "4"], stream)
        wait_status(REPORTS / "scope_353_tail1_replicas_batch_status.json", "isolated_scope_353_tail1_replicas_complete", stream)
        start_out, start_selection = select_outliers("starts", STARTS)
        replica_out, replica_selection = select_outliers("replicas", REPLICAS)
        (REPORTS / "scope_353_tail1_outlier_selection.json").write_text(json.dumps({"starts": start_selection, "replicas": replica_selection, "frozen_production_modified": False, "promotion_authorized": False}, indent=2) + "\n")
        if start_out:
            run([PYTHON, str(BASE / "scripts/run_scope_353_tail1_batch.py"), "--kind", "starts", "--seeds", *map(str, start_out), "--epochs", "15000", "--patience", "2500", "--workers", "4", "--long"], stream)
            wait_status(REPORTS / "scope_353_tail1_starts_long_batch_status.json", "isolated_scope_353_tail1_starts_long_complete", stream)
        if replica_out:
            run([PYTHON, str(BASE / "scripts/run_scope_353_tail1_batch.py"), "--kind", "replicas", "--seeds", *map(str, replica_out), "--epochs", "15000", "--patience", "2500", "--workers", "4", "--long"], stream)
            wait_status(REPORTS / "scope_353_tail1_replicas_long_batch_status.json", "isolated_scope_353_tail1_replicas_long_complete", stream)
        # Use validated long replacements in the raw crossed record.
        run([PYTHON, str(BASE / "scripts/summarize_scope_353_replica_batch.py"), "--prefix", PREFIX, "--start-seeds", *map(str, STARTS), "--long-start-seeds", *map(str, start_out), "--replica-seeds", *map(str, REPLICAS), "--long-replica-seeds", *map(str, replica_out), "--tag", "scope_353_tail1_start_replica_propagation_full"], stream)
        # Recompute robust gates from the long-replaced endpoints and create a
        # fit-quality-accepted record in parallel with the full raw record.
        final_start_rows = [(s, objective("starts", s, s in start_out)[0]) for s in STARTS]
        final_replica_rows = [(s, objective("replicas", s, s in replica_out)[0]) for s in REPLICAS]
        def gate(rows):
            vals = np.asarray([x[1] for x in rows]); med = float(np.median(vals)); mad = float(np.median(np.abs(vals - med))); threshold = med + max(3.0 * 1.4826 * mad, 0.03); return [s for s, v in rows if v > threshold], {"median": med, "mad": mad, "threshold": threshold, "rows": [{"seed": int(s), "objective_per_row": float(v)} for s, v in rows]}
        reject_s, gate_s = gate(final_start_rows); reject_r, gate_r = gate(final_replica_rows)
        (REPORTS / "scope_353_tail1_fit_quality_gate.json").write_text(json.dumps({"rejected_start_seeds": reject_s, "rejected_replica_seeds": reject_r, "start_gate": gate_s, "replica_gate": gate_r, "frozen_production_modified": False, "promotion_authorized": False}, indent=2) + "\n")
        run([PYTHON, str(BASE / "scripts/summarize_scope_353_replica_batch.py"), "--prefix", PREFIX, "--start-seeds", *map(str, STARTS), "--long-start-seeds", *map(str, start_out), "--replica-seeds", *map(str, REPLICAS), "--long-replica-seeds", *map(str, replica_out), "--exclude-start-seeds", *map(str, reject_s), "--exclude-replica-seeds", *map(str, reject_r), "--tag", "scope_353_tail1_start_replica_propagation_accepted"], stream)
        run([PYTHON, str(BASE / "scripts/plot_scope_353_fnp_start_replica_fig2_fig6.py"), "--source", str(REPORTS / "scope_353_tail1_start_replica_propagation_full/fnp_start_replica_crossed_long_x0p1.csv"), "--target", str(REPORTS / "scope_353_tail1_final_fig2_fig6_full")], stream)
        accepted_start = len(STARTS) - len(reject_s); accepted_replica = len(REPLICAS) - len(reject_r)
        run([PYTHON, str(BASE / "scripts/plot_scope_353_fnp_start_replica_fig2_fig6.py"), "--source", str(REPORTS / "scope_353_tail1_start_replica_propagation_accepted/fnp_start_replica_crossed_long_x0p1.csv"), "--target", str(REPORTS / "scope_353_tail1_final_fig2_fig6_accepted"), "--start-count", str(accepted_start), "--replica-count", str(accepted_replica)], stream)
        final = {"status": "isolated_scope_353_tail1_campaign_complete", "model": "v19 monotone FiLM with lambda_fnp_tail=1, bmin=6, target=0.05", "start_count": len(STARTS), "replica_count": len(REPLICAS), "long_start_seeds": start_out, "long_replica_seeds": replica_out, "rejected_start_seeds": reject_s, "rejected_replica_seeds": reject_r, "accepted_start_count": accepted_start, "accepted_replica_count": accepted_replica, "full_crossed_member_count": len(STARTS) * len(REPLICAS), "accepted_crossed_member_count": accepted_start * accepted_replica, "frozen_production_modified": False, "promotion_authorized": False}
        (REPORTS / "scope_353_tail1_campaign_final_status.json").write_text(json.dumps(final, indent=2) + "\n")
        stream.write(json.dumps(final, indent=2) + "\n")


if __name__ == "__main__":
    main()
