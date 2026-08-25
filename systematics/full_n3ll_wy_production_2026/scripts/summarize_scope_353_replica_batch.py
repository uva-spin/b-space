#!/usr/bin/env python3
"""Summarize replica fits and form the isolated start/replica F_NP ensemble.

The cross is performed in log F_NP: experimental-replica deviations from the
replica median are added to each independent-start log F_NP curve.  This is a
clearly documented empirical propagation rule, not a Gaussian error model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
REPORTS = BASE / "reports"


def load_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    frame = frame[np.isclose(frame["x"], 0.1)].sort_values("bT")
    if frame.empty:
        raise ValueError(path)
    return frame["bT"].to_numpy(float), np.log(np.maximum(frame["F_NP"].to_numpy(float), 1.0e-30))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-seeds", nargs="+", type=int, default=list(range(303, 327)))
    ap.add_argument("--long-start-seeds", nargs="*", type=int, default=[])
    ap.add_argument("--replica-seeds", nargs="+", type=int, default=list(range(1001, 1051)))
    ap.add_argument("--long-replica-seeds", nargs="*", type=int, default=[])
    ap.add_argument("--exclude-start-seeds", nargs="*", type=int, default=[])
    ap.add_argument("--exclude-replica-seeds", nargs="*", type=int, default=[])
    ap.add_argument("--tag", default="scope_353_start_replica_propagation")
    ap.add_argument("--prefix", default="scope_353_coupled_fnp_fit_lambda1_candidate")
    args = ap.parse_args()
    starts, replicas = [], []
    missing_starts, missing_replicas = [], []
    entries = []
    b_ref = None
    excluded_starts = set(args.exclude_start_seeds)
    excluded_replicas = set(args.exclude_replica_seeds)
    for seed in args.start_seeds:
        if seed in excluded_starts:
            continue
        suffix = "_long" if seed in set(args.long_start_seeds) else ""
        path = REPORTS / f"{args.prefix}_s{seed}_csvnorm{suffix}/fnp_debug_grid.csv"
        if not path.exists():
            missing_starts.append(seed)
            continue
        b, curve = load_curve(path)
        b_ref = b if b_ref is None else b_ref
        if not np.allclose(b, b_ref):
            raise RuntimeError(f"start grid differs for seed {seed}")
        starts.append(curve)
        entries.append({"kind": "start", "seed": int(seed)})
    for seed in args.replica_seeds:
        if seed in excluded_replicas:
            continue
        suffix = "_long" if seed in set(args.long_replica_seeds) else ""
        path = REPORTS / f"{args.prefix}_r{seed}_csvnorm{suffix}/fnp_debug_grid.csv"
        if not path.exists():
            missing_replicas.append(seed)
            continue
        b, curve = load_curve(path)
        b_ref = b if b_ref is None else b_ref
        if not np.allclose(b, b_ref):
            raise RuntimeError(f"replica grid differs for seed {seed}")
        replicas.append(curve)
        entries.append({"kind": "replica", "seed": int(seed)})
    if not starts or not replicas:
        raise SystemExit("both completed start and replica ensembles are required")
    starts = np.asarray(starts, float)
    replicas = np.asarray(replicas, float)
    residuals = replicas - np.median(replicas, axis=0)
    crossed = (starts[:, None, :] + residuals[None, :, :]).reshape(-1, starts.shape[1])
    q16, q50, q84 = np.quantile(crossed, [0.16, 0.50, 0.84], axis=0)
    s16, s50, s84 = np.quantile(starts, [0.16, 0.50, 0.84], axis=0)
    r16, r50, r84 = np.quantile(replicas, [0.16, 0.50, 0.84], axis=0)
    out_dir = REPORTS / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame({
        "bT": b_ref,
        "FNP_start_q16": np.exp(s16), "FNP_start_q50": np.exp(s50), "FNP_start_q84": np.exp(s84),
        "FNP_replica_q16": np.exp(r16), "FNP_replica_q50": np.exp(r50), "FNP_replica_q84": np.exp(r84),
        "FNP_crossed_q16": np.exp(q16), "FNP_crossed_q50": np.exp(q50), "FNP_crossed_q84": np.exp(q84),
        "relative_crossed_full_width": (np.exp(q84) - np.exp(q16)) / np.maximum(np.exp(q50), 1.0e-30),
    })
    table.to_csv(out_dir / "fnp_start_replica_quantiles_x0p1.csv", index=False)
    long_rows = []
    for i, curve in enumerate(np.exp(crossed)):
        long_rows.extend({"member": i, "x": 0.1, "bT": float(bv), "F_NP": float(fv)}
                         for bv, fv in zip(b_ref, curve))
    pd.DataFrame(long_rows).to_csv(out_dir / "fnp_start_replica_crossed_long_x0p1.csv", index=False)
    active = (b_ref <= 4.0) & (np.exp(q50) > 0.05 * np.max(np.exp(q50)[b_ref <= 4.0]))
    summary = {
        "status": "isolated_scope_353_start_replica_cross_complete" if not missing_starts and not missing_replicas else "isolated_scope_353_start_replica_cross_partial",
        "start_count": int(len(starts)),
        "replica_count": int(len(replicas)),
        "requested_start_count": int(len(args.start_seeds)),
        "requested_replica_count": int(len(args.replica_seeds)),
        "excluded_start_seeds": sorted(int(x) for x in excluded_starts),
        "excluded_replica_seeds": sorted(int(x) for x in excluded_replicas),
        "missing_start_seeds": [int(x) for x in missing_starts],
        "missing_replica_seeds": [int(x) for x in missing_replicas],
        "crossed_member_count": int(len(crossed)),
        "cross_rule": "log(F_NP) start curves plus centered log(F_NP) replica residuals",
        "quantiles": [0.16, 0.50, 0.84],
        "active_definition": "crossed median F_NP > 5% of its b<=4 GeV^-1 maximum",
        "active_b_max_GeV_inv": float(b_ref[active].max()),
        "max_relative_crossed_full_width_active": float(np.max(table.loc[active, "relative_crossed_full_width"])),
        "median_relative_crossed_full_width_active": float(np.median(table.loc[active, "relative_crossed_full_width"])),
        "interpretation": "empirical propagated envelope; not assigned a Gaussian confidence level",
        "artifact": str(out_dir / "fnp_start_replica_quantiles_x0p1.csv"),
        "long_artifact": str(out_dir / "fnp_start_replica_crossed_long_x0p1.csv"),
        "frozen_production_modified": False,
        "promotion_authorized": False,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
