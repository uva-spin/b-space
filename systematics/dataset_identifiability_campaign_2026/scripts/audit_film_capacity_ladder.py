#!/usr/bin/env python3
"""Propagate and Fourier-audit every completed scientific capacity pilot."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
TARGET = BASE / "summaries/film_capacity_pilots"


def run(*arguments: str) -> None:
    subprocess.run(
        [str(PYTHON), *arguments], cwd=BASE, check=True,
        stdout=subprocess.DEVNULL)


def main() -> None:
    rows = []
    for path in sorted((BASE / "outputs").glob(
            "filmcap_*/fit_status.json")):
        status = json.loads(path.read_text())
        if int(status["epochs_run"]) <= 1:
            continue
        tag = path.parent.name
        candidate = tag.removeprefix("filmcap_").split("_w")[0]
        architecture = tag.split(f"filmcap_{candidate}_", 1)[1].rsplit(
            "_s", 1)[0]
        target_name = f"filmcap_{candidate}_{architecture}_s{status['seed']}"
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
        complexity = status["model_complexity"]
        rows.append({
            "candidate_id": candidate,
            "architecture": architecture,
            "tag": tag,
            "seed": int(status["seed"]),
            "np_width": int(complexity["np_width"]),
            "np_cond_width": int(complexity["np_cond_width"]),
            "np_blocks": int(complexity["np_blocks"]),
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
        frame = frame.sort_values([
            "candidate_id", "np_width", "np_cond_width", "np_blocks", "seed"])
    TARGET.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TARGET / "transform_ladder_metrics.csv", index=False)
    (TARGET / "transform_ladder_summary.json").write_text(json.dumps({
        "status": "isolated_capacity_transform_ladder_not_production",
        "completed_scientific_pilot_count": len(rows),
        "passing_tags": (
            frame.loc[frame["transform_gate_pass"], "tag"].tolist()
            if len(frame) else []),
        "production_sources_modified": False,
        "pilots": rows,
    }, indent=2) + "\n")
    print(frame.to_string(index=False) if len(frame)
          else "No completed scientific capacity pilots.")


if __name__ == "__main__":
    main()
