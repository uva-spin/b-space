#!/usr/bin/env python3
"""Propagate and Fourier-audit completed transform-closure pilots."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
TARGET = BASE / "summaries/transform_closure_ladder"


def run(*arguments: str) -> None:
    subprocess.run(
        [sys.executable, *arguments], cwd=BASE, check=True,
        stdout=subprocess.DEVNULL)


def main() -> None:
    rows = []
    paths = list((BASE / "outputs").glob("closure_*/fit_status.json"))
    paths += list((BASE / "outputs").glob(
        "closurecoord_*/fit_status.json"))
    for path in sorted(paths):
        tag = path.parent.name
        status = json.loads(path.read_text())
        regularization = status["regularization"]
        closure = regularization["transform_closure"]
        target_name = tag
        target = BASE / "summaries" / target_name
        run("scripts/build_bspace_tmd_ensemble.py",
            "--tag-glob", tag, "--target-name", target_name)
        run("scripts/audit_kspace_transform_robustness.py",
            "--bspace-ensemble", str(target / "bspace_tmd_ensemble_long.csv"),
            "--target-name", target_name)
        audit = json.loads((target / "summary.json").read_text())
        grid = pd.read_csv(path.parent / "fnp_grid.csv")
        endpoint = grid.loc[
            grid["bT"] == grid["bT"].max(), ["x", "F_NP"]]
        x01 = endpoint.iloc[(endpoint["x"] - 0.1).abs().argsort()[:1]]
        final = status["final"]
        rows.append({
            "tag": tag,
            "lambda_closure": float(closure["lambda"]),
            "closure_max": float(closure["maximum_fnp"]),
            "model_constraint": status["model_constraint"]["kind"],
            "constraint_b_start": status["model_constraint"]["b_start"],
            "constraint_b_end": status["model_constraint"]["b_end"],
            "unpenalized_chi2": float(final["unpenalized_total_chi2"]),
            "fnp_gradient_l2": float(
                final["fnp_gradient_l2_per_row_objective"]),
            "closure_measure": float(
                final["transform_closure_penalty_per_row_objective"])
                / float(closure["lambda"]),
            "max_endpoint_fnp_b8": float(endpoint["F_NP"].max()),
            "x01_fnp_b8": float(x01["F_NP"].iloc[0]),
            "max_tailmode_central_change_active": float(
                audit["max_alternative_relative_central_change_active"]),
            "minimum_tailmode_median_over_peak": float(
                audit["minimum_alternative_median_over_reference_peak"]),
            "transform_gate_pass": bool(audit["transform_gate_pass"]),
        })
    frame = pd.DataFrame(rows)
    if len(frame):
        frame = frame.sort_values(["closure_max", "lambda_closure"])
    TARGET.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TARGET / "metrics.csv", index=False)
    (TARGET / "summary.json").write_text(json.dumps({
        "status": "isolated_transform_closure_ladder_not_production",
        "completed_count": len(rows),
        "passing_tags": (
            frame.loc[frame["transform_gate_pass"], "tag"].tolist()
            if len(frame) else []),
        "production_sources_modified": False,
        "pilots": rows,
    }, indent=2) + "\n")
    print(frame.to_string(index=False) if len(frame)
          else "No completed transform-closure pilots.")


if __name__ == "__main__":
    main()
