#!/usr/bin/env python3
"""Quantify the selected replica-robust prior on all verified endpoints."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np


BASE = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/dustin/miniforge3/envs/pdf-fit/bin/python")
SOURCE = BASE / "summaries/replica_robust_reference_full24/summary.json"
BRACKET = BASE / "summaries/full_reference_replica_strength_bracket/summary.json"
TARGET = BASE / "summaries/replica_robust_constraint_scale"
LOWER_CONTROL = BASE / "summaries/lambda637p5_fitbar_minimum_control/summary.json"
MINIMUM_SEARCH = BASE / "summaries/minimum_fitbar_constraint_search/summary.json"


def stats(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(array)), "median": float(np.median(array)),
        "max": float(np.max(array)),
    }


def main() -> None:
    # The selected lambda675 generation includes a fit-quality barrier that was
    # absent from the lambda637.5 failure.  Because that barrier is active on
    # the initial hard-replica objective, directly reject the combined lower
    # prescription before claiming lambda675 is minimal.
    minimum = (json.loads(MINIMUM_SEARCH.read_text())
               if MINIMUM_SEARCH.exists() else None)
    if minimum is None or minimum.get("status") != "complete":
        subprocess.run([
            str(PYTHON),
            str(BASE / "scripts/audit_lambda637p5_fitbar_minimum_control.py"),
        ], check=True)
        lower_control = json.loads(LOWER_CONTROL.read_text())
        if lower_control["status"] != "lower_candidate_rejected":
            raise RuntimeError(
                "lambda637.5+fit-barrier lower control has not been rejected"
            )
    # Independent fresh-block confirmation must precede central-initializer
    # selection.  This prevents an early two-block apparent plateau from
    # entering the non-uniqueness ensemble after the stress tests demonstrated
    # delayed reversals on unchanged objectives.
    source_before_confirmation = json.loads(SOURCE.read_text())
    fresh = source_before_confirmation.get("fresh_start_boundary_confirmation", {})
    if not fresh.get("all_24_starts_confirmed", False):
        subprocess.run([
            str(PYTHON), str(BASE / "scripts/confirm_reference_start_boundaries.py")
        ], check=True)
    source = json.loads(SOURCE.read_text())
    bracket = json.loads(BRACKET.read_text())
    if source["status"] != "complete" or not source["all_starts_fnp_plateaued_and_fit_preserved"]:
        raise RuntimeError("replica-robust full24 verification has not passed")
    if minimum is not None and minimum.get("status") == "complete":
        selected = float(minimum["selected_weakest_surviving_strength"])
        if float(source["selected_strength"]) != selected:
            raise RuntimeError("full24 verification does not match minimum search")
    if (not source.get("staged_prescription", False)
            and float(source["selected_strength"]) != float(bracket["weakest_tested_passing_strength"])):
        raise RuntimeError("verified strength does not match replica-stress bracket")
    # A verifier process may have been launched before the width-field naming was
    # hardened.  Normalize the terminal manifest here before any downstream
    # stage consumes it.  The legacy values were calculated on exactly this
    # domain; only their old ``active`` labels were ambiguous.
    width_definition = (
        "maximum over x=0.1 and 0.1<=bT<=4 of pointwise width divided by "
        "max(pointwise median,0.05); no active-mask points are omitted"
    )
    changed = False
    if "fnp_width_metric_definition" not in source:
        source["fnp_width_metric_definition"] = width_definition
        changed = True
    for precise, legacy in (
        ("max_endpoint_fnp_full_range_selected_domain_floor_normalized",
         "max_endpoint_fnp_full_range_active"),
        ("max_endpoint_fnp_central68_width_selected_domain_floor_normalized",
         "max_endpoint_fnp_central68_width_active"),
    ):
        if precise not in source:
            source[precise] = source[legacy]
            changed = True
    if changed:
        SOURCE.write_text(json.dumps(source, indent=2) + "\n")
    ratios, penalties, totals, deltas = [], [], [], []
    for tag in source["endpoint_tags"]:
        status = json.loads((BASE / "outputs" / tag / "fit_status.json").read_text())
        final = status["final"]
        penalty = float(final["reference_distance_penalty_per_row_objective"])
        total = float(final["objective_per_row"])
        penalties.append(penalty); totals.append(total); ratios.append(penalty / total)
    # The verifier is authoritative for source-relative unpenalized fit costs.
    import pandas as pd
    runs = pd.read_csv(SOURCE.parent / "runs.csv")
    endpoint_names = set(source["endpoint_tags"])
    deltas = runs.loc[runs.tag.isin(endpoint_names), "unpenalized_chi2_delta"].astype(float).tolist()
    summary = {
        "status": "complete", "selected_strength": source["selected_strength"],
        "fit_quality_barrier_strength": source.get("fit_quality_barrier_strength", 0.0),
        "fit_quality_barrier_power": source.get("fit_quality_barrier_power", 2),
        "selected_bmax": source["selected_bmax"], "endpoint_count": len(source["endpoint_tags"]),
        "selection_bracket": source["selection_fail_pass_bracket"],
        "metrics": {
            "reference_penalty_per_row_objective": stats(penalties),
            "total_objective_per_row": stats(totals),
            "reference_penalty_fraction_of_total_objective": stats(ratios),
            "source_relative_unpenalized_chi2_delta": stats(deltas),
        },
        "production_sources_modified": False,
    }
    TARGET.mkdir(parents=True, exist_ok=True)
    (TARGET / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
