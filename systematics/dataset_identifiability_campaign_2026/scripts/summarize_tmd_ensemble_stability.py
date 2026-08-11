#!/usr/bin/env python3
"""Quantify percentile-curve stability for a propagated TMD ensemble."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble-long", type=Path, required=True)
    parser.add_argument("--coordinate", choices=("bT", "kT"), required=True)
    parser.add_argument("--value-column", default=None)
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=500)
    parser.add_argument("--split-draws", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=20260724)
    parser.add_argument("--minimum-member-count", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    frame = pd.read_csv(args.ensemble_long)
    member_column = "run_tag" if "run_tag" in frame else "_replica_key"
    value_column = args.value_column or (
        "ftilde" if args.coordinate == "bT" else "value")
    required = {
        member_column, args.coordinate, value_column, "flavor", "x", "Q"}
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"missing ensemble columns: {sorted(missing)}")
    if "quantity" in frame:
        frame = frame[frame["quantity"].astype(str).eq("ftilde")].copy()

    index_columns = ["x", "Q", "flavor", args.coordinate]
    if "pid" in frame:
        index_columns.insert(2, "pid")
    wide = frame.pivot(
        index=index_columns, columns=member_column, values=value_column)
    if wide.isna().any().any():
        raise RuntimeError("ensemble members do not share a complete curve grid")
    values = wide.to_numpy(float).T
    members = wide.columns.astype(str).tolist()
    n_members = len(members)
    if n_members < 3:
        raise RuntimeError("at least three ensemble members are required")

    full = np.quantile(values, [0.16, 0.50, 0.84], axis=0)
    index = wide.index.to_frame(index=False)
    bands = index.copy()
    bands["q16"], bands["median"], bands["q84"] = full

    active = np.zeros(values.shape[1], dtype=bool)
    for _, positions in index.groupby(["Q", "flavor"], sort=False).groups.items():
        positions = np.asarray(list(positions), dtype=int)
        peak = np.max(np.abs(full[1, positions]))
        active[positions] = np.abs(full[1, positions]) > 0.05 * peak
    scale = np.maximum(np.abs(full[1]), 1.0e-12)

    rng = np.random.default_rng(args.random_seed)
    bootstrap_max = []
    for _ in range(args.bootstrap_draws):
        sample = values[rng.integers(0, n_members, size=n_members)]
        endpoints = np.quantile(sample, [0.16, 0.84], axis=0)
        relative = np.abs(endpoints - full[[0, 2]]) / scale
        bootstrap_max.append(float(np.max(relative[:, active])))

    split_max = []
    half = n_members // 2
    if half >= 3:
        for _ in range(args.split_draws):
            order = rng.permutation(n_members)
            first = np.quantile(values[order[:half]], [0.16, 0.84], axis=0)
            second = np.quantile(values[order[half:2 * half]], [0.16, 0.84], axis=0)
            relative = np.abs(first - second) / scale
            split_max.append(float(np.max(relative[:, active])))

    target = BASE / "summaries" / args.target_name
    target.mkdir(parents=True, exist_ok=True)
    bands.to_csv(target / f"{args.coordinate}_tmd_bands.csv", index=False)
    pd.DataFrame({"member": members}).to_csv(
        target / "ensemble_members.csv", index=False)
    metrics = {
        "status": "isolated_tmd_ensemble_endpoint_stability_not_production",
        "source": str(args.ensemble_long.resolve()),
        "coordinate": args.coordinate,
        "value_column": value_column,
        "member_count": n_members,
        "minimum_member_count": args.minimum_member_count,
        "active_definition": "absolute full-ensemble median >5% of each flavor/Q peak",
        "bootstrap_draws": args.bootstrap_draws,
        "bootstrap_max_relative_endpoint_change": {
            "median": float(np.median(bootstrap_max)),
            "p90": float(np.quantile(bootstrap_max, 0.90)),
            "p95": float(np.quantile(bootstrap_max, 0.95)),
            "maximum": float(np.max(bootstrap_max)),
        },
        "split_draws": len(split_max),
        "split_half_max_relative_endpoint_difference": ({
            "median": float(np.median(split_max)),
            "p90": float(np.quantile(split_max, 0.90)),
            "p95": float(np.quantile(split_max, 0.95)),
            "maximum": float(np.max(split_max)),
        } if split_max else None),
        "declared_relative_endpoint_gate": 0.02,
        "endpoint_gate_pass": bool(
            n_members >= args.minimum_member_count
            and split_max
            and np.quantile(bootstrap_max, 0.95) <= 0.02
            and np.quantile(split_max, 0.95) <= 0.02),
        "production_sources_modified": False,
    }
    (target / "summary.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
