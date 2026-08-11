#!/usr/bin/env python3
"""Deterministic Fourier preflight for the remote-tail closure coordinate."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
SOURCE = (
    BASE / "summaries/lengthrate_D020_E772_lrate1em03"
    / "bspace_tmd_ensemble_long.csv")
TARGET_ROOT = BASE / "summaries/closure_coordinate_preflight"
WINDOWS = (4.0, 4.5, 5.0, 5.5, 6.0, 6.5)
B_END = 8.0
THRESHOLD = 1.0e-4


def smoothstep(b: np.ndarray, start: float) -> np.ndarray:
    t = np.clip((b - start) / (B_END - start), 0.0, 1.0)
    return t**3 * (10.0 - 15.0 * t + 6.0 * t**2)


def main() -> None:
    source = pd.read_csv(SOURCE)
    rows = []
    reference_bands = {}
    for start in WINDOWS:
        adjusted = source.copy()
        endpoint = adjusted[np.isclose(adjusted["bT"], B_END)]
        endpoint_by_curve = endpoint.set_index(
            ["x", "Q", "flavor"])["F_NP"].to_dict()
        amplitudes = np.array([
            max(np.log(endpoint_by_curve[(x, q, str(flavor))]
                       / THRESHOLD), 0.0)
            for x, q, flavor in zip(
                adjusted["x"], adjusted["Q"],
                adjusted["flavor"].astype(str))
        ])
        factor = np.exp(
            -amplitudes * smoothstep(
                adjusted["bT"].to_numpy(float), start))
        adjusted["F_NP"] *= factor
        for column in ("ftilde", "x_ftilde", "b_ftilde", "b_x_ftilde"):
            adjusted[column] *= factor
        token = str(start).replace(".", "p")
        target_name = f"closure_coordinate_preflight_b{token}_8p0"
        adjusted["run_tag"] = target_name
        target = BASE / "summaries" / target_name
        target.mkdir(parents=True, exist_ok=True)
        bspace_path = target / "bspace_tmd_ensemble_long.csv"
        adjusted.to_csv(bspace_path, index=False)
        subprocess.run([
            sys.executable, "scripts/audit_kspace_transform_robustness.py",
            "--bspace-ensemble", str(bspace_path),
            "--target-name", target_name,
        ], cwd=BASE, check=True, stdout=subprocess.DEVNULL)
        audit = json.loads((target / "summary.json").read_text())
        bands = pd.read_csv(target / "kspace_tailmode_bands.csv")
        reference_bands[start] = bands[bands["tail_mode"].eq("expb2")]
        rows.append({
            "b_start": start,
            "b_end": B_END,
            "closure_threshold": THRESHOLD,
            "max_tailmode_central_change_active": audit[
                "max_alternative_relative_central_change_active"],
            "minimum_tailmode_median_over_peak": audit[
                "minimum_alternative_median_over_reference_peak"],
            "transform_gate_pass": audit["transform_gate_pass"],
        })
    adjacent = []
    for lower, upper in zip(WINDOWS[:-1], WINDOWS[1:]):
        joined = reference_bands[lower].merge(
            reference_bands[upper],
            on=["flavor", "x", "Q", "kT"],
            suffixes=("_lower", "_upper"), validate="one_to_one")
        for flavor, group in joined.groupby("flavor", sort=False):
            displayed = group[group["kT"] <= 2.25]
            reference = displayed["median_lower"].to_numpy(float)
            active = np.abs(reference) > 0.05 * np.max(np.abs(reference))
            relative = np.abs(
                displayed["median_upper"].to_numpy(float) - reference
            ) / np.maximum(np.abs(reference), 1.0e-12)
            adjacent.append({
                "lower_b_start": lower,
                "upper_b_start": upper,
                "flavor": str(flavor),
                "max_relative_central_change_active": float(
                    np.max(relative[active])),
            })
    TARGET_ROOT.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(TARGET_ROOT / "metrics.csv", index=False)
    pd.DataFrame(adjacent).to_csv(
        TARGET_ROOT / "adjacent_window_metrics.csv", index=False)
    (TARGET_ROOT / "summary.json").write_text(json.dumps({
        "status": "deterministic_remote_tail_coordinate_preflight_not_fit",
        "interpretation": (
            "Tests Fourier sufficiency only; fit stationarity and uniqueness "
            "must be established separately."),
        "production_sources_modified": False,
        "windows": rows,
        "adjacent_window_comparisons": adjacent,
    }, indent=2) + "\n")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
