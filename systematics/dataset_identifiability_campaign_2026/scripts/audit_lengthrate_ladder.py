#!/usr/bin/env python3
"""Idempotently propagate and Fourier-audit every completed length-rate pilot."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
TARGET = BASE / "summaries/loglength_ratecurv_combo"


def run(*arguments: str) -> None:
    subprocess.run(
        [str(PYTHON), *arguments], cwd=BASE, check=True,
        stdout=subprocess.DEVNULL)


def main() -> None:
    rows = []
    statuses = sorted(
        (BASE / "outputs").glob("lengthrate_*/fit_status.json"))
    for status_path in statuses:
        status = json.loads(status_path.read_text())
        tag = status_path.parent.name
        if int(status["seed"]) != 701:
            continue
        candidate = tag.removeprefix("lengthrate_").split("_llen")[0]
        rate = float(status["regularization"]["ratecurv"]["lambda"])
        short_rate = tag.split("_lrate", 1)[1].rsplit("_s", 1)[0]
        target_name = f"lengthrate_{candidate}_lrate{short_rate}"
        target = BASE / "summaries" / target_name
        bspace = target / "bspace_tmd_ensemble_long.csv"
        run(
            "scripts/build_bspace_tmd_ensemble.py",
            "--tag-glob", tag, "--target-name", target_name)
        run(
            "scripts/audit_kspace_transform_robustness.py",
            "--bspace-ensemble", str(bspace),
            "--target-name", target_name)
        audit = json.loads((target / "summary.json").read_text())
        final = status["final"]
        rows.append({
            "candidate_id": candidate,
            "tag": tag,
            "lambda_loglength": float(
                status["regularization"]["logf_arc_length"]["lambda"]),
            "lambda_ratecurv": rate,
            "unpenalized_chi2": float(final["unpenalized_total_chi2"]),
            "fnp_gradient_l2": float(
                final["fnp_gradient_l2_per_row_objective"]),
            "max_tailmode_central_change_active": float(
                audit["max_alternative_relative_central_change_active"]),
            "max_tailmode_endpoint_change_active": float(
                audit["max_alternative_relative_endpoint_change_active"]),
            "minimum_tailmode_median_over_peak": float(
                audit["minimum_alternative_median_over_reference_peak"]),
            "transform_gate_pass": bool(audit["transform_gate_pass"]),
        })
    frame = pd.DataFrame(rows)
    if len(frame):
        frame = frame.sort_values(["candidate_id", "lambda_ratecurv"])
    TARGET.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TARGET / "transform_ladder_metrics.csv", index=False)
    summary = {
        "status": "isolated_lengthrate_transform_ladder_not_production",
        "completed_central_pilot_count": len(rows),
        "passing_tags": (
            frame.loc[frame["transform_gate_pass"], "tag"].tolist()
            if len(frame) else []),
        "production_sources_modified": False,
        "pilots": rows,
    }
    (TARGET / "transform_ladder_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(frame.to_string(index=False) if len(frame)
          else "No completed seed-701 length-rate pilots.")


if __name__ == "__main__":
    main()
