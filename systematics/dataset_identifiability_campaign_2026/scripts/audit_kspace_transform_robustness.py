#!/usr/bin/env python3
"""Audit Fig. 6 ensemble sensitivity to finite-b continuation choices."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
PROJECT = BASE.parents[1]
TRANSFORMER = PROJECT / "workflows/v23a/construction/construct_v23a_regularized_kspace_tmd_v2.py"
MODES = ("expb2", "expb", "zero", "taper")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bspace-ensemble", type=Path, required=True)
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--x", type=float, default=0.1)
    parser.add_argument("--Q", type=float, default=10.0)
    return parser.parse_args()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = arguments()
    transform = load_module("campaign_transform_audit", TRANSFORMER)
    all_curves = pd.read_csv(args.bspace_ensemble)
    curves = all_curves[
        np.isclose(all_curves["x"], args.x)
        & np.isclose(all_curves["Q"], args.Q)
        & all_curves["flavor"].astype(str).isin(["u", "d"])
    ].copy()
    curves["_replica_key"] = curves["run_tag"].astype(str)
    curves["seed"] = curves["fit_seed"].fillna(-1).astype(int)
    curves["pdf_member"] = 0

    transformed = []
    mode_meta = {}
    for mode in MODES:
        settings = argparse.Namespace(
            quantities=["ftilde"], tail_mode=mode, tail_fit_bmin=None,
            eps=1.0e-300, b_transform_max=24.0, n_b_transform=6001,
            k_max=4.0, n_k=401, end_taper_start_fraction=0.92)
        values, metadata = transform.transform_curves(curves, settings)
        values["tail_mode"] = mode
        transformed.append(values)
        mode_meta[mode] = metadata
    long = pd.concat(transformed, ignore_index=True)

    bands = (
        long.groupby(
            ["tail_mode", "flavor", "x", "Q", "kT"], observed=False)["value"]
        .quantile([0.16, 0.50, 0.84]).unstack()
        .rename(columns={0.16: "q16", 0.50: "median", 0.84: "q84"})
        .reset_index()
    )
    reference = bands[bands["tail_mode"].eq("expb2")].drop(
        columns="tail_mode")
    rows = []
    for mode in MODES:
        candidate = bands[bands["tail_mode"].eq(mode)].drop(
            columns="tail_mode")
        joined = reference.merge(
            candidate, on=["flavor", "x", "Q", "kT"],
            suffixes=("_reference", "_candidate"), validate="one_to_one")
        for flavor, group in joined.groupby("flavor", sort=False):
            group = group[group["kT"] <= 2.25]
            median = group["median_reference"].to_numpy(float)
            peak = np.max(median)
            active = median > 0.05 * peak
            scale = np.maximum(np.abs(median), 1.0e-12)
            central_change = np.abs(
                group["median_candidate"].to_numpy(float) - median) / scale
            endpoint_change = np.maximum(
                np.abs(group["q16_candidate"].to_numpy(float)
                       - group["q16_reference"].to_numpy(float)),
                np.abs(group["q84_candidate"].to_numpy(float)
                       - group["q84_reference"].to_numpy(float)),
            ) / scale
            candidate_median = group["median_candidate"].to_numpy(float)
            rows.append({
                "tail_mode": mode,
                "flavor": flavor,
                "max_relative_central_change_active": float(
                    np.max(central_change[active])),
                "max_relative_endpoint_change_active": float(
                    np.max(endpoint_change[active])),
                "minimum_median_over_reference_peak": float(
                    np.min(candidate_median) / peak),
                "minimum_active_median_over_reference_peak": float(
                    np.min(candidate_median[active]) / peak),
                "negative_point_fraction": float(
                    np.mean(candidate_median < 0.0)),
                "negative_active_point_fraction": float(
                    np.mean(candidate_median[active] < 0.0)),
            })
    metrics = pd.DataFrame(rows)
    alternatives = metrics[~metrics["tail_mode"].eq("expb2")]
    max_central = float(
        alternatives["max_relative_central_change_active"].max())
    max_endpoint = float(
        alternatives["max_relative_endpoint_change_active"].max())
    minimum_displayed = float(
        alternatives["minimum_median_over_reference_peak"].min())
    minimum_active = float(
        alternatives["minimum_active_median_over_reference_peak"].min())
    target = BASE / "summaries" / args.target_name
    target.mkdir(parents=True, exist_ok=True)
    long.to_csv(target / "kspace_tailmode_ensemble_long.csv", index=False)
    bands.to_csv(target / "kspace_tailmode_bands.csv", index=False)
    metrics.to_csv(target / "tailmode_metrics.csv", index=False)
    summary = {
        "status": "isolated_kspace_transform_robustness_not_production",
        "source_bspace_ensemble": str(args.bspace_ensemble.resolve()),
        "tail_modes": list(MODES),
        "reference_tail_mode": "expb2",
        "active_definition": (
            "reference median >5% of its positive flavor peak, "
            "kT<=2.25 GeV"),
        "max_alternative_relative_central_change_active": max_central,
        "max_alternative_relative_endpoint_change_active": max_endpoint,
        "minimum_alternative_median_over_reference_peak": minimum_displayed,
        "minimum_active_alternative_median_over_reference_peak": minimum_active,
        "declared_relative_robustness_gate": 0.02,
        "declared_negative_tolerance_over_peak": -0.01,
        "transform_gate_pass": bool(
            max_central <= 0.02
            and max_endpoint <= 0.02
            and minimum_active >= -0.01),
        "transform_metadata": mode_meta,
        "production_sources_modified": False,
    }
    (target / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
