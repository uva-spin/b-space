#!/usr/bin/env python3
"""Build a machine-readable table of all completed candidate phases."""

from __future__ import annotations

import json
from pathlib import Path
import re

import pandas as pd


BASE = Path(__file__).resolve().parents[1]


def main() -> None:
    registry = pd.read_csv(BASE / "manifests/dataset_candidates.csv")
    source_to_candidate = {
        str(Path(row.central_output).resolve()): row.candidate_id
        for row in registry.itertuples()
        if bool(row.central_output_exists)
    }
    records = []
    for path in sorted((BASE / "outputs").glob("*/fit_status.json")):
        status = json.loads(path.read_text())
        tag = path.parent.name
        source = str(Path(status["source_production"]).resolve())
        candidate = source_to_candidate.get(source, "")
        ordinary = re.fullmatch(
            r"(.+)_s(\d+)_(explore|settle|polish64)", tag)
        logcurv = re.fullmatch(
            r"logcurv_(.+)_lam([^_]+)_s(\d+)(?:_(continue|smoke))?", tag)
        loglength = re.fullmatch(
            r"loglength_(.+)_lam([^_]+)_s(\d+)(?:_(continue|smoke))?",
            tag)
        filmcap = re.fullmatch(
            r"filmcap_(.+)_(w\d+_c\d+_b\d+)_s(\d+)", tag)
        lengthrate = re.fullmatch(
            r"lengthrate_(.+)_llen([^_]+)_lrate([^_]+)_s(\d+)",
            tag)
        if ordinary:
            candidate = ordinary.group(1)
            seed = int(ordinary.group(2))
            phase = ordinary.group(3)
            family = "unregularized_multistart"
        elif logcurv:
            candidate = logcurv.group(1)
            seed = int(logcurv.group(3))
            phase = (
                "logcurv_" + logcurv.group(4)
                if logcurv.group(4) else "logcurv_pilot_or_multistart")
            family = "global_logF_curvature"
        elif loglength:
            candidate = loglength.group(1)
            seed = int(loglength.group(3))
            phase = (
                "loglength_" + loglength.group(4)
                if loglength.group(4)
                else "loglength_pilot_or_multistart")
            family = "global_logF_arc_length"
        elif filmcap:
            candidate = filmcap.group(1)
            seed = int(filmcap.group(3))
            phase = (
                "film_capacity_initialization_smoke"
                if int(status["epochs_run"]) <= 1
                else "film_capacity_pilot_or_multistart")
            family = "reduced_capacity_FiLM"
        elif lengthrate:
            candidate = lengthrate.group(1)
            seed = int(lengthrate.group(4))
            phase = "lengthrate_pilot_or_multistart"
            family = "logF_arc_length_plus_damping_rate_curvature"
        elif status.get("point_profile", {}).get("enabled"):
            seed = int(status["seed"])
            phase = "localized_point_profile"
            family = "localized_logF_profile"
        else:
            seed = status.get("seed")
            phase = "other_or_compatibility"
            family = status.get("model_constraint", {}).get("kind", "none")
        final = status["final"]
        regularization = status.get("regularization", {})
        logcurv_config = regularization.get("logf_curvature", {})
        loglength_config = regularization.get("logf_arc_length", {})
        ratecurv_config = regularization.get("ratecurv", {})
        model_constraint = status.get("model_constraint", {})
        model_complexity = status.get("model_complexity", {})
        records.append({
            "tag": tag,
            "candidate_id": candidate,
            "seed": seed,
            "phase": phase,
            "trial_family": family,
            "row_count": status["row_count"],
            "epochs_run": status["epochs_run"],
            "stopped_on_plateau": status["stopped_on_plateau"],
            "lbfgs_closures": status["lbfgs"]["closure_evaluations"],
            "total_chi2": final["total_chi2"],
            "data_chi2": final["data_chi2"],
            "unpenalized_total_chi2": final.get(
                "unpenalized_total_chi2", final["total_chi2"]),
            "fnp_gradient_l2": final["fnp_gradient_l2_per_row_objective"],
            "normalization_gradient_l2": final["normalization_gradient_l2_per_row_objective"],
            "lambda_logcurv": logcurv_config.get("lambda", 0.0),
            "lambda_loglength": loglength_config.get("lambda", 0.0),
            "lambda_ratecurv": ratecurv_config.get("lambda", 0.0),
            "model_constraint": model_constraint.get("kind", "none"),
            "np_width": model_complexity.get("np_width"),
            "np_cond_width": model_complexity.get("np_cond_width"),
            "np_blocks": model_complexity.get("np_blocks"),
            "replica_seed": status.get("replica_seed"),
            "initial_state": status.get("initial_state"),
            "source_production": source,
            "production_state_modified": status.get(
                "production_state_modified", False),
        })
    table = pd.DataFrame(records)
    target = BASE / "summaries"
    target.mkdir(parents=True, exist_ok=True)
    table.to_csv(target / "campaign_runs.csv", index=False)
    summary = {
        "completed_run_count": len(table),
        "candidate_count": int(table["candidate_id"].replace("", pd.NA).nunique())
        if len(table) else 0,
        "phase_counts": table["phase"].value_counts().to_dict() if len(table) else {},
        "family_counts": table["trial_family"].value_counts().to_dict()
        if len(table) else {},
        "all_report_production_unmodified": bool(
            (~table["production_state_modified"]).all()) if len(table) else True,
    }
    (target / "campaign_runs_status.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
