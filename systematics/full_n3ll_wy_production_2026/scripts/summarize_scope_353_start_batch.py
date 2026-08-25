#!/usr/bin/env python3
"""Summarize independent-start F_NP spread for the isolated 353 fit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
REPORTS = BASE / "reports"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(303, 327)))
    ap.add_argument("--long-seeds", nargs="*", type=int, default=[], help="seeds read from the long-horizon _long output")
    ap.add_argument("--prefix", default="scope_353_coupled_fnp_fit_lambda1_candidate")
    ap.add_argument("--tag", default="scope_353_coupled_start_summary")
    args = ap.parse_args()
    entries = []
    curves = []
    missing = []
    for seed in args.seeds:
        suffix = "_long" if seed in set(args.long_seeds) else ""
        fit = REPORTS / f"{args.prefix}_s{seed}_csvnorm{suffix}"
        metrics_path = fit / "metrics.json"
        grid_path = fit / "fnp_debug_grid.csv"
        if not metrics_path.exists() or not grid_path.exists():
            missing.append(seed)
            continue
        metrics = json.loads(metrics_path.read_text())
        train = metrics["train"]
        grid = pd.read_csv(grid_path)
        g = grid[np.isclose(grid["x"], 0.1)].sort_values("bT")
        if len(g) == 0:
            missing.append(seed)
            continue
        curves.append(g["F_NP"].to_numpy(float))
        entries.append({
            "seed": int(seed),
            "best_epoch": int(train["best_epoch"]),
            "epochs_run": int(train["epochs_run"]),
            "best_chi2_like": float(train["best_chi2_like"]),
            "last_chi2_like": float(train["last_epoch_chi2_like"]),
            "best_objective_per_row": float(train["best_chi2_like"] / train["n_points"]),
            "late_best": bool(train["best_epoch"] >= 0.9 * train["epochs_run"]),
            "last_minus_best": float(train["last_epoch_chi2_like"] - train["best_chi2_like"]),
        })
    if not curves:
        raise SystemExit("no completed start outputs found")
    arr = np.asarray(curves, float)
    # The x=0.1 grid is shared by all runs; use the first completed grid.
    b = pd.read_csv(REPORTS / f"{args.prefix}_s{entries[0]['seed']}_csvnorm/fnp_debug_grid.csv")
    b = b[np.isclose(b["x"], 0.1)].sort_values("bT")["bT"].to_numpy(float)
    q16, q50, q84 = np.quantile(arr, [0.16, 0.50, 0.84], axis=0)
    p5, p95 = np.quantile(arr, [0.05, 0.95], axis=0)
    active = (b <= 4.0) & (q50 > 0.05 * np.max(q50[b <= 4.0]))
    rel_full = (q84 - q16) / np.maximum(q50, 1.0e-30)
    rel_p90 = (p95 - p5) / np.maximum(q50, 1.0e-30)
    out_dir = REPORTS / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"bT": b, "q05": p5, "q16": q16, "q50": q50, "q84": q84, "q95": p95,
                  "relative_q16_q84": rel_full, "relative_q05_q95": rel_p90}).to_csv(out_dir / "fnp_start_quantiles_x0p1.csv", index=False)
    summary = {
        "status": "isolated_scope_353_start_summary_complete" if not missing else "isolated_scope_353_start_summary_partial",
        "requested_seeds": [int(x) for x in args.seeds],
        "completed_seeds": [int(x["seed"]) for x in entries],
        "missing_seeds": [int(x) for x in missing],
        "completed_count": len(entries),
        "all_late_best": bool(all(x["late_best"] for x in entries)),
        "starts": entries,
        "x": 0.1,
        "active_definition": "median F_NP > 5% of its b<=4 GeV^-1 maximum",
        "active_b_max_GeV_inv": float(b[active].max()),
        "max_relative_q16_q84_active": float(np.max(rel_full[active])),
        "median_relative_q16_q84_active": float(np.median(rel_full[active])),
        "max_relative_q05_q95_active": float(np.max(rel_p90[active])),
        "median_relative_q05_q95_active": float(np.median(rel_p90[active])),
        "quantile_interpretation": "descriptive start-distribution envelope; not a Gaussian confidence interval",
        "frozen_production_modified": False,
        "promotion_authorized": False,
        "artifact": str(out_dir / "fnp_start_quantiles_x0p1.csv"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
