#!/usr/bin/env python3
"""Prove that the isolated study's declared final deliverables are complete."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import csv
import struct

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
SUMMARIES = BASE / "summaries"
TARGET = SUMMARIES / "campaign_completion_audit"
LOCKED_INCUMBENT_WIDTHS = {
    "u": 0.11772613918747582,
    "d": 0.12490071924111977,
}


def load(relative: str) -> dict:
    path = SUMMARIES / relative
    if not path.is_file():
        raise RuntimeError(f"missing required summary: {path}")
    return json.loads(path.read_text())


def digest(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"missing or empty artifact: {path}")
    result = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    if path.suffix.lower() == ".png":
        header = path.read_bytes()[:24]
        if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
            raise RuntimeError(f"invalid PNG header: {path}")
        width, height = struct.unpack(">II", header[16:24])
        if width < 100 or height < 100:
            raise RuntimeError(f"implausible PNG dimensions: {path}")
        result["pixel_width"] = width
        result["pixel_height"] = height
    return result


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def fnp_vector(tag: str) -> np.ndarray:
    frame = pd.read_csv(BASE / "outputs" / tag / "fnp_grid.csv")
    mask = np.isclose(frame.x, 0.1) & frame.bT.between(0.1, 4.0, inclusive="both")
    return frame.loc[mask, "F_NP"].to_numpy(float)


def sensitivity_confirmation_valid(rows: list[dict]) -> bool:
    """Require two wholly subsequent quiet blocks after the first >=1% event."""
    pass_indices = [index for index, row in enumerate(rows)
                    if row.get("stationarity_and_fit_pass") == "True"]
    if not pass_indices:
        return True
    passed_at = pass_indices[0]
    triggers = [index for index, row in enumerate(rows[:passed_at + 1])
                if row.get("eligible_post_mandatory_confirmation") == "True"
                and float(row["fnp_drift_from_previous_chunk"]) >= 0.01]
    if not triggers:
        return True
    trigger = triggers[0]
    subsequent = rows[trigger + 1:passed_at + 1]
    return (len(subsequent) >= 2
            and all(float(row["fnp_drift_from_previous_chunk"]) <= 0.02
                    for row in subsequent[-2:]))


def main() -> None:
    # Recompute this at the actual promotion boundary, rather than trusting an
    # earlier check made while the fitting campaign was still active.
    subprocess.run(
        [sys.executable, str(BASE / "scripts/audit_frozen_inputs.py")],
        check=True,
    )
    frozen = load("frozen_input_audit/summary.json")
    trial_inventory = load("campaign_runs_status.json")
    family_decision = json.loads((SUMMARIES /
        "final_constraint_family_decision/decision.json").read_text())
    dataset = load("dataset_selection_decision/summary.json")
    dataset_polish = load("start_spread_polish64/summary.json")
    dataset_phase_drift = load("phase_drift_settle_to_polish64/summary.json")
    selection = load("reference_strength_selected_extent_scan/summary.json")
    bracket = load("full_reference_replica_strength_bracket/summary.json")
    barrier_rescue = load("lambda675_fit_quality_barrier_rescue/summary.json")
    lower_control = load("lambda637p5_fitbar_minimum_control/summary.json")
    minimum_search_path = SUMMARIES / "minimum_fitbar_constraint_search/summary.json"
    minimum_search = (json.loads(minimum_search_path.read_text())
                      if minimum_search_path.exists() else None)
    barrier_search_path = SUMMARIES / "minimum_barrier_constraint_search/summary.json"
    barrier_search = (json.loads(barrier_search_path.read_text())
                      if barrier_search_path.exists() else None)
    staged = load("lambda675_fit_quality_barrier_stress/summary.json")
    failed_525 = load("selected_reference_central_replicas/failed_lam525_80k_summary.json")
    failed_562 = load("selected_reference_central_replicas/failed_lam562p5_80k_summary.json")
    failed_600 = load("selected_reference_central_replicas/failed_lam600_80k_summary.json")
    failed_637 = load("selected_reference_central_replicas/failed_80k_summary.json")
    starts = load("replica_robust_reference_full24/summary.json")
    start_boundary = load("selected_reference_start_boundary_confirmation/summary.json")
    start_sensitivity = load("selected_reference_start_sensitivity_confirmation/summary.json")
    constraint = load("replica_robust_constraint_scale/summary.json")
    replicas = load("selected_reference_central_replicas/summary.json")
    boundary = load("selected_reference_boundary_confirmation/summary.json")
    ensemble = load("final_combined_tmd_ensemble/summary.json")
    stability = load("final_combined_ensemble_stability/summary.json")
    figures = load("final_fig2_fig6/summary.json")

    require(frozen["status"] == "pass", "frozen production inputs changed")
    require(frozen["unchanged_input_count"] == frozen["registered_input_count"],
            "frozen-input audit has incomplete coverage")
    require(int(trial_inventory["completed_run_count"]) >= 347
            and int(trial_inventory["family_counts"][
                "unregularized_multistart"]) >= 99
            and int(trial_inventory["family_counts"][
                "global_logF_curvature"]) >= 31
            and int(trial_inventory["family_counts"][
                "global_logF_arc_length"]) >= 17
            and int(trial_inventory["family_counts"][
                "reduced_capacity_FiLM"]) >= 11
            and trial_inventory["all_report_production_unmodified"],
            "systematic prior/architecture trial inventory is incomplete")
    required_rejected_families = {
        "derivative_only_priors", "logF_curvature", "logF_arc_length",
        "arc_length_plus_rate_curvature", "matched_or_reduced_tails",
        "reduced_architectures", "C1_remote_closure", "C2_remote_closure",
    }
    require(family_decision["status"] ==
                "no_constraint_family_passes_all_promotion_gates"
            and required_rejected_families.issubset(
                family_decision["constraint_family_results"])
            and not family_decision["replica_promotion_authorized"]
            and not family_decision["production_sources_modified"],
            "rejected prior/architecture family evidence is incomplete")
    require(dataset["status"] == "complete"
            and dataset["selected_candidate"] == "D020_ALL"
            and dataset["selected_row_count"] == 329,
            "dataset selection decision is incomplete or inconsistent")
    require(dataset["all_ordinary_candidates_failed_unregularized_fnp_identifiability"]
            and not dataset["tier1_tevatron_joint_fnp_stationarity_established"],
            "dataset-only identifiability conclusion is unsupported")
    require(dataset_polish["phase"] == "polish64"
            and dataset_polish["candidate_count"] == 11
            and len(dataset_polish["candidates"]) == 11,
            "ordinary dataset polished comparison lacks full coverage")
    require(dataset_phase_drift["from_phase"] == "settle"
            and dataset_phase_drift["to_phase"] == "polish64"
            and dataset_phase_drift["run_count"] == 33
            and len(dataset_phase_drift["candidates"]) == 11,
            "ordinary dataset phase-drift evidence lacks full coverage")
    require(selection["status"] == "complete", "constraint selection incomplete")
    lower_search_complete = (minimum_search is not None
                             and minimum_search.get("status") == "complete")
    lower_grid_selected = (lower_search_complete and float(minimum_search[
        "selected_weakest_surviving_strength"]) < 675.0)
    require((lower_control["status"] in {
                "lower_candidate_rejected",
                "lower_candidate_survives_discriminator",
            })
            and float(lower_control["tested_reference_strength"]) == 637.5
            and float(lower_control["fit_quality_barrier_strength"]) == 100.0
            and int(lower_control["fit_quality_barrier_power"]) == 2
            and int(lower_control["mandatory_same_objective_iterations"]) >= 200_000,
            "lambda637.5+selected-barrier minimum-control evidence is incomplete")
    with (SUMMARIES / "lambda637p5_fitbar_minimum_control/runs.csv").open(
            newline="") as stream:
        lower_control_rows = list(csv.DictReader(stream))
    require(lower_control_rows
            and max(int(float(row["cumulative_lbfgs_iterations"]))
                    for row in lower_control_rows) >= 200_000
            and sum(int(float(row["cumulative_lbfgs_iterations"])) <= 200_000
                    for row in lower_control_rows) == 40
            and lower_control_rows[-1]["tag"] == lower_control["endpoint_tag"]
            and ((lower_control["status"] == "lower_candidate_rejected"
                  and not any(row["stationarity_and_fit_pass"] == "True"
                              for row in lower_control_rows))
                 or (lower_control["status"] ==
                     "lower_candidate_survives_discriminator"
                     and lower_control_rows[-1]["stationarity_and_fit_pass"] == "True")),
            "lambda637.5+selected-barrier control ledger is not terminal")
    for row in lower_control_rows:
        tag = row["tag"]
        status = json.loads((BASE / "outputs" / tag / "fit_status.json").read_text())
        reference = status["regularization"]["fnp_reference_distance"]
        barrier = status["regularization"]["fit_quality_barrier"]
        require(float(reference["lambda"]) == 637.5
                and float(reference["b_min"]) == 0.1
                and float(reference["b_max"]) == 4.0
                and float(barrier["lambda"]) == 100.0
                and int(barrier["power"]) == 2
                and int(status["replica_seed"]) == 1033,
                f"lower minimum-control objective provenance mismatch: {tag}")
        require(status["convergence_gate_pass"]
                and int(status["lbfgs"]["closure_evaluations"]) >= 1,
                f"lower minimum-control continuation did not converge: {tag}")
    if lower_search_complete:
        require([float(item["strength"]) for item in
                 minimum_search["tested_candidates"]] ==
                [637.5, 600.0, 562.5, 525.0, 487.5, 450.0, 412.5,
                 375.0, 337.5, 300.0, 262.5, 225.0, 187.5, 150.0,
                 112.5, 75.0, 37.5, 0.0],
                "minimum lambda search did not classify the complete registered grid")
        selected_lambda = float(
            minimum_search["selected_weakest_surviving_strength"])
        rejected_lambda = minimum_search[
            "strongest_rejected_strength_below_selection"]
        require(float(starts["selected_strength"]) == selected_lambda
                and ((selected_lambda == 0.0 and rejected_lambda is None)
                     or (rejected_lambda is not None
                         and float(rejected_lambda) < selected_lambda)),
                "minimum fit-barrier search does not bracket the full24 selection")
        for item in minimum_search["tested_candidates"]:
            evidence_path = Path(item["evidence"])
            evidence = json.loads(evidence_path.read_text())
            expected = ("candidate_survives_discriminator"
                        if item["outcome"] == "survives" else
                        "candidate_rejected")
            if float(item["strength"]) == 637.5:
                expected = ("lower_candidate_survives_discriminator"
                            if item["outcome"] in {
                                "survives", "rejected_post_mandatory_window"
                            } else "lower_candidate_rejected")
            require(evidence["status"] == expected
                    and float(evidence["fit_quality_barrier_strength"]) == 100.0
                    and int(evidence["mandatory_same_objective_iterations"]) >= 200_000,
                    f"minimum-search evidence is incomplete at lambda={item['strength']}")
            if float(item["strength"]) != 637.5:
                require("requested LBFGS max-iteration capacity" in evidence[
                            "iteration_accounting"]
                        and int(evidence[
                            "completed_continuation_checkpoint_count"]) >= 42
                        and int(evidence[
                            "executed_lbfgs_closure_evaluations_total"]) >= 42,
                        f"minimum-search optimizer accounting is incomplete at lambda={item['strength']}")
            with (evidence_path.parent / "runs.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            pass_key = "stationarity_and_fit_pass"
            require(rows
                    and max(int(float(row["cumulative_lbfgs_iterations"]))
                            for row in rows) >= 200_000
                    and sum(int(float(row["cumulative_lbfgs_iterations"])) <=
                            int(evidence["mandatory_same_objective_iterations"])
                            for row in rows) == int(evidence[
                                "mandatory_same_objective_iterations"]) // 5000
                    and rows[-1]["tag"] == evidence["endpoint_tag"]
                    and ((item["outcome"] in {
                              "survives", "rejected_post_mandatory_window"
                          }
                          and rows[-1][pass_key] == "True")
                         or (item["outcome"] == "rejected"
                             and not any(row[pass_key] == "True" for row in rows))),
                    f"minimum-search ledger is nonterminal at lambda={item['strength']}")
            if item["outcome"] == "survives":
                mandatory = int(evidence["mandatory_same_objective_iterations"])
                anchor_rows = [row for row in rows if int(float(
                    row["cumulative_lbfgs_iterations"])) == mandatory]
                require(len(anchor_rows) == 1,
                        f"minimum-search mandatory anchor is missing at lambda={item['strength']}")
                anchor = fnp_vector(anchor_rows[0]["tag"])
                terminal = fnp_vector(evidence["endpoint_tag"])
                cumulative = float(np.max(
                    np.abs(terminal - anchor) / np.maximum(anchor, 0.05)))
                require(cumulative <= 0.02,
                        f"minimum-search cumulative post-mandatory drift fails at lambda={item['strength']}")
                require(sensitivity_confirmation_valid(rows),
                        f"minimum-search sensitivity confirmation is incomplete at lambda={item['strength']}")
            if item["outcome"] == "rejected_post_mandatory_window":
                require(float(item["post_mandatory_window_fnp_drift"]) > 0.02,
                        "window-rejected candidate does not exceed the gate")
            for row in rows:
                fit = json.loads((BASE / "outputs" / row["tag"] /
                                  "fit_status.json").read_text())
                reference = fit["regularization"]["fnp_reference_distance"]
                barrier = fit["regularization"]["fit_quality_barrier"]
                require(float(reference["lambda"]) == float(item["strength"])
                        and float(reference["b_min"]) == 0.1
                        and float(reference["b_max"]) == 4.0
                        and float(barrier["lambda"]) == 100.0
                        and int(barrier["power"]) == 2
                        and int(fit["replica_seed"]) == int(evidence["replica_seed"]),
                        f"minimum-search objective mismatch: {row['tag']}")
                require(fit["convergence_gate_pass"]
                        and int(fit["lbfgs"]["closure_evaluations"]) >= 1,
                        f"minimum-search continuation did not converge: {row['tag']}")
    if lower_grid_selected:
        require(barrier_search is not None
                and barrier_search.get("status") == "complete"
                and float(barrier_search["selected_reference_strength"]) ==
                    float(starts["selected_strength"])
                and float(barrier_search[
                    "selected_weakest_surviving_barrier_strength"]) ==
                    float(starts["fit_quality_barrier_strength"]),
                "barrier-strength re-bracket does not match full24 selection")
        selected_mu = float(
            barrier_search["selected_weakest_surviving_barrier_strength"])
        require([float(item["barrier_strength"]) for item in
                 barrier_search["trials"]] ==
                [100.0, 30.0, 10.0, 3.0, 1.0, 0.3, 0.1, 0.0],
                "minimum barrier search did not classify the complete registered grid")
        rejected_mu = barrier_search[
            "strongest_rejected_barrier_strength_below_selection"]
        require((selected_mu == 0.0 and rejected_mu is None)
                or (rejected_mu is not None
                    and float(rejected_mu) < selected_mu),
                "selected barrier lacks either the exact zero boundary or a rejected weaker bracket")
        for item in barrier_search["trials"]:
            mu = float(item["barrier_strength"])
            if mu == 100.0:
                continue
            evidence_path = Path(item["evidence"])
            evidence = json.loads(evidence_path.read_text())
            expected = ("candidate_survives_discriminator"
                        if item["outcome"] == "survives" else
                        "candidate_rejected")
            require(evidence["status"] == expected
                    and float(evidence["tested_reference_strength"]) ==
                        float(starts["selected_strength"])
                    and float(evidence["fit_quality_barrier_strength"]) == mu
                    and int(evidence["mandatory_same_objective_iterations"]) >= 200_000,
                    f"barrier-search evidence is incomplete at mu={mu}")
            require("requested LBFGS max-iteration capacity" in evidence[
                        "iteration_accounting"]
                    and int(evidence[
                        "completed_continuation_checkpoint_count"]) >= 42
                    and int(evidence[
                        "executed_lbfgs_closure_evaluations_total"]) >= 42,
                    f"barrier-search optimizer accounting is incomplete at mu={mu}")
            with (evidence_path.parent / "runs.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            require(rows
                    and max(int(float(row["cumulative_lbfgs_iterations"]))
                            for row in rows) >= 200_000
                    and sum(int(float(row["cumulative_lbfgs_iterations"])) <=
                            int(evidence["mandatory_same_objective_iterations"])
                            for row in rows) == int(evidence[
                                "mandatory_same_objective_iterations"]) // 5000
                    and rows[-1]["tag"] == evidence["endpoint_tag"]
                    and ((item["outcome"] == "survives"
                          and rows[-1]["stationarity_and_fit_pass"] == "True")
                         or (item["outcome"] == "rejected"
                             and not any(row["stationarity_and_fit_pass"] == "True"
                                         for row in rows))),
                    f"barrier-search ledger is nonterminal at mu={mu}")
            if item["outcome"] == "survives":
                mandatory = int(evidence["mandatory_same_objective_iterations"])
                anchor_rows = [row for row in rows if int(float(
                    row["cumulative_lbfgs_iterations"])) == mandatory]
                require(len(anchor_rows) == 1,
                        f"barrier-search mandatory anchor is missing at mu={mu}")
                anchor = fnp_vector(anchor_rows[0]["tag"])
                terminal = fnp_vector(evidence["endpoint_tag"])
                cumulative = float(np.max(
                    np.abs(terminal - anchor) / np.maximum(anchor, 0.05)))
                require(cumulative <= 0.02,
                        f"barrier-search cumulative post-mandatory drift fails at mu={mu}")
                require(sensitivity_confirmation_valid(rows),
                        f"barrier-search sensitivity confirmation is incomplete at mu={mu}")
            for row in rows:
                fit = json.loads((BASE / "outputs" / row["tag"] /
                                  "fit_status.json").read_text())
                reference = fit["regularization"]["fnp_reference_distance"]
                barrier = fit["regularization"]["fit_quality_barrier"]
                require(float(reference["lambda"]) == float(
                            starts["selected_strength"])
                        and float(reference["b_min"]) == 0.1
                        and float(reference["b_max"]) == 4.0
                        and float(barrier["lambda"]) == mu
                        and int(barrier["power"]) == 2
                        and int(fit["replica_seed"]) == int(
                            evidence["replica_seed"]),
                        f"barrier-search objective mismatch: {row['tag']}")
                require(fit["convergence_gate_pass"]
                        and int(fit["lbfgs"]["closure_evaluations"]) >= 1,
                        f"barrier-search continuation did not converge: {row['tag']}")
    require(starts["status"] == "complete", "24-start verification incomplete")
    require("requested LBFGS max-iteration capacity" in starts[
                "iteration_accounting"],
            "24-start optimizer accounting semantics are missing")
    require(starts["member_count"] == 24, "24-start coverage is not exact")
    require(starts["all_starts_fnp_plateaued_and_fit_preserved"],
            "not every independent start passed stationarity and fit quality")
    require(start_boundary["status"] == "complete"
            and start_boundary["confirmed_count"] == 24
            and not start_boundary["failed_seeds"],
            "fresh boundary confirmation is incomplete for the 24 starts")
    require(int(start_boundary["minimum_cumulative_iterations"]) >= 40_000
            and int(start_boundary["maximum_cumulative_iterations"]) >= 160_000
            and all(int(item["cumulative_iterations"]) >= 50_000
                    for item in start_boundary["confirmations"]),
            "24-start confirmations did not test the delayed-reversal horizon")
    fresh_start = starts.get("fresh_start_boundary_confirmation", {})
    require(fresh_start.get("all_24_starts_confirmed", False)
            and len(fresh_start.get("confirmations", [])) == 24
            and not fresh_start.get("failed_seeds", []),
            "24-start summary does not contain complete fresh-boundary evidence")
    require(float(start_boundary["selected_strength"]) == float(starts["selected_strength"])
            and float(start_boundary["selected_bmax"]) == float(starts["selected_bmax"]),
            "fresh start confirmation used a different selected prescription")
    require(start_sensitivity["status"] == "complete"
            and not start_sensitivity["failed_seeds"],
            "near-boundary start reconfirmation is incomplete")
    require(float(start_sensitivity["acceptance_gate_value"]) == 0.02
            and float(start_sensitivity["selection_trigger_value"]) == 0.01,
            "near-boundary confirmation changed the stationarity gate semantics")
    sensitive_start = starts.get("sensitive_start_boundary_confirmation", {})
    expected_sensitive_seeds = sorted(
        int(item["seed"]) for item in start_boundary["confirmations"]
        if float(item["terminal_drift"]) >= 0.01
    )
    require(sensitive_start.get("all_selected_starts_confirmed", False)
            and not sensitive_start.get("failed_seeds", [])
            and sorted(int(seed) for seed in sensitive_start.get("selected_seeds", []))
            == expected_sensitive_seeds
            and sorted(int(seed) for seed in start_sensitivity.get("selected_seeds", []))
            == expected_sensitive_seeds
            and len(sensitive_start.get("confirmations", []))
            == len(expected_sensitive_seeds)
            and int(start_sensitivity.get("confirmed_count", -1))
            == len(expected_sensitive_seeds),
            "near-boundary evidence is absent or inconsistent in the recomputed 24-start summary")
    require("0.1<=bT<=4" in starts["fnp_width_metric_definition"]
            and "no active-mask points are omitted" in starts["fnp_width_metric_definition"],
            "24-start FNP-width domain/denominator semantics are missing")
    with (SUMMARIES / "replica_robust_reference_full24/runs.csv").open(
            newline="") as stream:
        start_ledger = {row["tag"]: row for row in csv.DictReader(stream)}
    require(len(starts["endpoint_tags"]) == 24
            and len(set(starts["endpoint_tags"])) == 24,
            "24-start endpoint identities are incomplete")
    for tag in starts["endpoint_tags"]:
        require(tag in start_ledger, f"24-start terminal ledger row missing: {tag}")
        row = start_ledger[tag]
        status = json.loads((BASE / "outputs" / tag / "fit_status.json").read_text())
        reference = status["regularization"]["fnp_reference_distance"]
        fit_barrier = status["regularization"]["fit_quality_barrier"]
        require(status["convergence_gate_pass"],
                f"24-start endpoint is not a converged fit: {tag}")
        require(int(float(row[
                    "executed_lbfgs_closure_evaluations_this_block"])) ==
                    int(status["lbfgs"]["closure_evaluations"])
                and int(float(row[
                    "requested_lbfgs_max_iterations_this_block"])) == 5000,
                f"24-start optimizer accounting mismatch: {tag}")
        require(row["stationarity_and_fit_pass"] == "True",
                f"24-start endpoint lacks terminal stationarity/fit passage: {tag}")
        require(float(reference["lambda"]) == float(starts["selected_strength"])
                and float(reference["b_min"]) == 0.1
                and float(reference["b_max"]) == float(starts["selected_bmax"]),
                f"24-start endpoint prior provenance mismatch: {tag}")
        require(float(fit_barrier["lambda"]) == float(
                    starts.get("fit_quality_barrier_strength", 0.0))
                and int(fit_barrier["power"]) == int(
                    starts.get("fit_quality_barrier_power", 2)),
                f"24-start endpoint fit-barrier provenance mismatch: {tag}")
    champion = load("champion_registry/current.json")
    require(all(float(starts["candidate_nonuniqueness_fig6_widths"][flavor])
                < float(champion["combined_fig6_max_active_relative_full_width"][flavor])
                for flavor in ("u", "d")),
            "selected start ensemble does not improve the registered champion")
    require(constraint["status"] == "complete" and constraint["endpoint_count"] == 24,
            "selected constraint scale was not calibrated on all endpoints")
    require(float(constraint["selected_strength"]) == float(starts["selected_strength"])
            and float(constraint["selected_bmax"]) == float(starts["selected_bmax"]),
            "constraint calibration does not match the selected prescription")
    require(float(starts["selected_bmax"]) == float(selection["selected_bmax"]),
            "replica-robust method changed the selected extent")
    if (starts.get("staged_prescription", False)
            and math.isclose(float(starts["selected_strength"]), 675.0)):
        selected_barrier = float(starts["fit_quality_barrier_strength"])
        lower_barrier_trials = [
            item for item in barrier_rescue["trials"]
            if float(item["barrier_strength"]) < selected_barrier
        ]
        selected_barrier_trials = [
            item for item in barrier_rescue["trials"]
            if float(item["barrier_strength"]) == selected_barrier
        ]
        require(barrier_rescue["status"] == "complete"
                and float(barrier_rescue["reference_strength_held_fixed"])
                    == float(starts["selected_strength"])
                and float(barrier_rescue["selected_weakest_barrier_strength"])
                    == selected_barrier
                and "quadratic" in barrier_rescue["selection_rule"]
                and int(barrier_rescue["replica_seed"]) in {
                    int(seed) for seed in
                    staged["stress_replica_seeds_hardest_first"]
                }
                and lower_barrier_trials
                and not any(item["fit_gate_pass"]
                            for item in lower_barrier_trials)
                and len(selected_barrier_trials) >= 2
                and all(item["fit_gate_pass"]
                        and item["fnp_stationarity_pass"]
                        for item in selected_barrier_trials),
                "minimum fit-quality barrier ladder is incomplete or inconsistent")
        require(staged["status"] == "complete"
                and float(starts["selected_strength"]) == float(staged["reference_strength"])
                and float(starts["fit_quality_barrier_strength"])
                    == float(staged["fit_quality_barrier_strength"])
                and int(starts["fit_quality_barrier_power"])
                    == int(staged["fit_quality_barrier_power"]),
                "staged replica-robust prescription is unresolved")
        require(int(staged.get("minimum_total_reference_iterations", 0)) >= 50_000,
                "staged hard-replica audit used insufficient total exposure")
        with (SUMMARIES / "lambda675_fit_quality_barrier_stress/runs.csv").open(
                newline="") as stream:
            staged_ledger_rows = list(csv.DictReader(stream))
        staged_by_tag = {row["tag"]: row for row in staged_ledger_rows}
        expected_stress_seeds = sorted(set(
            int(seed) for seed in staged["stress_replica_seeds_hardest_first"]
        ))
        observed_stress_seeds = []
        require(len(staged["endpoint_tags"]) == len(expected_stress_seeds)
                and len(set(staged["endpoint_tags"])) == len(expected_stress_seeds),
                "staged hard-replica endpoint coverage is incomplete")
        for tag in staged["endpoint_tags"]:
            require(tag in staged_by_tag,
                    f"staged hard-replica terminal ledger row missing: {tag}")
            row = staged_by_tag[tag]
            observed_stress_seeds.append(int(float(row["replica_seed"])))
            require(int(float(row["total_reference_iterations"])) >= 50_000,
                    f"staged hard replica stopped below50k total exposure: {tag}")
            require(row["fit_gate_pass"] == "True"
                    and row["stationarity_pass"] == "True",
                    f"staged hard replica lacks terminal fit/stationarity passage: {tag}")
        require(sorted(observed_stress_seeds) == expected_stress_seeds,
                "staged hard-replica identities differ from the declared set")
    elif starts.get("staged_prescription", False):
        require(lower_grid_selected and barrier_search is not None
                and barrier_search.get("status") == "complete",
                "lower staged prescription lacks two-dimensional bracket")
    else:
        require(float(starts["selected_strength"])
                    == float(bracket["weakest_tested_passing_strength"])
                and float(bracket["relative_fail_pass_bracket_width"]) <= 0.10,
                "minimum replica-robust strength bracket is unresolved")
    require(failed_525["status"] == "replica_stationarity_failed"
            and float(failed_525["selected_strength"]) == 525.0
            and 1032 in failed_525["failed_replica_seeds"],
            "archived lambda525 full-replica failure evidence is incomplete")
    require(failed_562["status"] == "replica_stationarity_failed"
            and float(failed_562["selected_strength"]) == 562.5
            and set(failed_562["failed_replica_seeds"]) == {1032, 1044},
            "archived lambda562.5 full-replica failure evidence is incomplete")
    require(failed_600["status"] == "replica_stationarity_failed"
            and float(failed_600["selected_strength"]) == 600.0
            and set(failed_600["failed_replica_seeds"]) == {1037, 1041, 1049},
            "archived lambda600 full-replica failure evidence is incomplete")
    require(failed_637["status"] == "replica_stationarity_failed"
            and float(failed_637["selected_strength"]) == 637.5
            and 1033 in failed_637["failed_replica_seeds"],
            "archived lambda637.5 full-replica failure evidence is incomplete")
    require(replicas["status"] == "complete", "experimental replicas incomplete")
    require("requested LBFGS max-iteration capacity" in replicas[
                "iteration_accounting"]
            and "requested LBFGS max-iteration capacity" in boundary[
                "iteration_accounting"],
            "central/replica optimizer accounting semantics are missing")
    require(replicas["replica_count"] == 50, "50-replica coverage is not exact")
    require(replicas["central_fnp_plateau_pass"] and replicas["all_replicas_fnp_plateaued"],
            "central or experimental-replica FNP stationarity failed")
    require(replicas.get("all_stress_replicas_cross_optimizer_consistent", False)
            and replicas.get("cross_optimizer_checks"),
            "hard experimental replicas lack cross-optimizer FNP consistency")
    require(all(item["agreement_pass"]
                for item in replicas["cross_optimizer_checks"]),
            "a hard experimental replica has unresolved optimizer-start ambiguity")
    require(int(replicas.get("minimum_cumulative_iterations", 0)) >= 40_000,
            "central/replica campaign did not enforce the delayed-reversal horizon")
    require(float(replicas["fnp_drift_domain"]["x"]) == 0.1
            and float(replicas["fnp_drift_domain"]["b_min"]) == 0.1
            and float(replicas["fnp_drift_domain"]["b_max"]) == float(starts["selected_bmax"]),
            "central/replica FNP drift domain differs from the selected prior domain")
    require(boundary["status"] == "complete" and not boundary["failed_replica_seeds"],
            "one-percent-sensitivity replica confirmations are incomplete")
    require(boundary["fnp_drift_domain"] == replicas["fnp_drift_domain"],
            "boundary confirmation drift domain differs from the replica campaign")
    initializer_status = json.loads((
        BASE / "outputs" / replicas["central_initializer_tag"] / "fit_status.json"
    ).read_text())
    central_status = json.loads((
        BASE / "outputs" / replicas["central_endpoint_tag"] / "fit_status.json"
    ).read_text())
    row_count = int(central_status["row_count"])
    central_unpenalized_delta = (
        float(central_status["final"]["unpenalized_total_chi2"])
        - float(initializer_status["final"]["unpenalized_total_chi2"])
    )
    require(central_unpenalized_delta <= math.sqrt(2.0 * row_count),
            "stationary selected central does not preserve unpenalized fit quality")
    # Replica fits fluctuate around N by construction.  Use a declared
    # chi-square sanity bound, rather than comparing their values with the
    # central-data chi-square or with the regularized training objective.
    replica_chi2_ceiling = row_count + 5.0 * math.sqrt(2.0 * row_count)
    replica_chi2 = []
    replica_ids = []
    with (SUMMARIES / "selected_reference_central_replicas/runs.csv").open(
            newline="") as stream:
        replica_ledger = list(csv.DictReader(stream))
    # Independently reconstruct delayed movers from the complete ledger.  Do
    # not trust the boundary selector's own summary as proof that it selected
    # every case.  Any replica that moved by more than 2% from 30k onward must have
    # two wholly fresh quiet blocks beyond its first primary passing endpoint.
    selected_boundary_seeds = {
        int(value) for value in boundary["selected_replica_seeds"]
    }
    confirmed_by_seed = {
        int(item["replica_seed"]): item
        for item in boundary["confirmations"]
    }
    empirical_delayed_seeds = set()
    for replica_seed in range(1001, 1051):
        seed_rows = sorted(
            (row for row in replica_ledger
             if row["kind"] == "experimental_replica"
             and int(float(row["replica_seed"])) == replica_seed),
            key=lambda row: int(float(row["cumulative_lbfgs_iterations"])),
        )
        late_horizon_rows = [
            row for row in seed_rows
            if int(float(row["cumulative_lbfgs_iterations"])) >= 30_000
        ]
        if (late_horizon_rows
                and max(float(row["fnp_drift_from_previous_chunk"])
                        for row in late_horizon_rows) > 0.02):
            empirical_delayed_seeds.add(replica_seed)
            require(replica_seed in selected_boundary_seeds,
                    f"delayed replica omitted from boundary audit: {replica_seed}")
            require(replica_seed in confirmed_by_seed,
                    f"delayed replica lacks a boundary confirmation: {replica_seed}")
            primary_pass_rows = [
                row for row in seed_rows
                if row["fnp_plateau_gate_pass"].strip().lower() == "true"
            ]
            require(primary_pass_rows,
                    f"delayed replica lacks a primary passing endpoint: {replica_seed}")
            primary_iterations = int(float(
                primary_pass_rows[0]["cumulative_lbfgs_iterations"]))
            confirmation_iterations = int(
                confirmed_by_seed[replica_seed]["cumulative_iterations"])
            require(confirmation_iterations >= primary_iterations + 10_000,
                    f"delayed replica lacks two fresh confirmation blocks: {replica_seed}")
            fresh_rows = [
                row for row in seed_rows
                if primary_iterations < int(float(
                    row["cumulative_lbfgs_iterations"])) <= confirmation_iterations
            ]
            require(len(fresh_rows) >= 2
                    and all(float(row["fnp_drift_from_previous_chunk"]) <= 0.02
                            for row in fresh_rows[-2:]),
                    f"delayed replica's final two fresh blocks are not quiet: {replica_seed}")
    require(empirical_delayed_seeds,
            "no empirical delayed movers were reconstructed from the final ledger")
    bracket_stress_seeds = {int(value) for value in bracket["stress_replica_seeds"]}
    require(bracket_stress_seeds,
            "replica-strength bracket has no stress identities")
    for replica_seed in sorted(bracket_stress_seeds):
        require(replica_seed in selected_boundary_seeds,
                f"bracket stress replica omitted from boundary audit: {replica_seed}")
        require(replica_seed in confirmed_by_seed,
                f"bracket stress replica lacks boundary confirmation: {replica_seed}")
        seed_rows = sorted(
            (row for row in replica_ledger
             if row["kind"] == "experimental_replica"
             and int(float(row["replica_seed"])) == replica_seed),
            key=lambda row: int(float(row["cumulative_lbfgs_iterations"])),
        )
        primary_pass_rows = [
            row for row in seed_rows
            if row["fnp_plateau_gate_pass"].strip().lower() == "true"
        ]
        require(primary_pass_rows,
                f"bracket stress replica lacks a primary passing endpoint: {replica_seed}")
        primary_iterations = int(float(
            primary_pass_rows[0]["cumulative_lbfgs_iterations"]))
        confirmation_iterations = int(
            confirmed_by_seed[replica_seed]["cumulative_iterations"])
        require(confirmation_iterations >= primary_iterations + 10_000,
                f"bracket stress replica lacks two fresh confirmation blocks: {replica_seed}")
        fresh_rows = [
            row for row in seed_rows
            if primary_iterations < int(float(
                row["cumulative_lbfgs_iterations"])) <= confirmation_iterations
        ]
        require(len(fresh_rows) >= 2
                and all(float(row["fnp_drift_from_previous_chunk"]) <= 0.02
                        for row in fresh_rows[-2:]),
                f"bracket stress replica's final two fresh blocks are not quiet: {replica_seed}")
    terminal_replica_rows = {
        row["tag"]: row for row in replica_ledger
        if row["tag"] in set(replicas["replica_endpoint_tags"])
    }
    central_rows = [row for row in replica_ledger
                    if row["tag"] == replicas["central_endpoint_tag"]]
    require(len(central_rows) == 1
            and int(float(central_rows[0]["cumulative_lbfgs_iterations"])) >= 50_000,
            "central endpoint lacks two fresh blocks beyond 40k")
    central_status = json.loads((BASE / "outputs" /
        replicas["central_endpoint_tag"] / "fit_status.json").read_text())
    require(central_status["convergence_gate_pass"]
            and int(float(central_rows[0][
                "executed_lbfgs_closure_evaluations_this_block"])) == int(
                    central_status["lbfgs"]["closure_evaluations"])
            and int(float(central_rows[0][
                "requested_lbfgs_max_iterations_this_block"])) == 5000,
            "central optimizer accounting mismatch")
    for tag in replicas["replica_endpoint_tags"]:
        require(tag in terminal_replica_rows
                and int(float(terminal_replica_rows[tag]["cumulative_lbfgs_iterations"])) >= 50_000,
                f"replica endpoint lacks two fresh blocks beyond 40k: {tag}")
        status = json.loads((BASE / "outputs" / tag / "fit_status.json").read_text())
        value = float(status["final"]["unpenalized_total_chi2"])
        require(status["convergence_gate_pass"] and math.isfinite(value),
                f"replica endpoint is not a finite converged fit: {tag}")
        require(int(float(terminal_replica_rows[tag][
                    "executed_lbfgs_closure_evaluations_this_block"])) ==
                    int(status["lbfgs"]["closure_evaluations"])
                and int(float(terminal_replica_rows[tag][
                    "requested_lbfgs_max_iterations_this_block"])) == 5000,
                f"replica optimizer accounting mismatch: {tag}")
        replica_chi2.append(value)
        require(status.get("replica_seed") is not None,
                f"replica endpoint lacks a pseudo-data identity: {tag}")
        replica_id = int(status["replica_seed"])
        replica_ids.append(replica_id)
        required_stress_exposure = int(
            replicas.get("stress_replica_required_iterations", {}).get(
                str(replica_id), 50_000)
        )
        require(int(float(terminal_replica_rows[tag][
                    "cumulative_lbfgs_iterations"])) >= required_stress_exposure,
                f"replica {replica_id} stopped before its independent stress horizon")
    require(sorted(replica_ids) == list(range(1001, 1051)),
            "experimental endpoints do not map exactly to replica seeds 1001--1050")
    require(max(replica_chi2) <= replica_chi2_ceiling,
            "an experimental replica fails the declared 5-sigma chi-square sanity gate")
    require(ensemble["status"] == "complete", "combined propagation incomplete")
    require(ensemble["start_count"] == 24 and ensemble["experimental_replica_count"] == 50,
            "combined ensemble source coverage is incomplete")
    require(ensemble["combined_member_count"] == 1200,
            "combined Cartesian ensemble must contain 24x50 members")
    require(ensemble.get("central_counted_once") is True
            and float(ensemble[
                "start_log_residual_pointwise_median_abs_max"]) <= 1.0e-12
            and float(ensemble[
                "experimental_log_residual_pointwise_median_abs_max"]) <= 1.0e-12
            and "operational separability assumption" in ensemble[
                "hierarchical_transfer_assumption"]
            and ensemble.get("joint_nested_start_by_replica_refits_performed") is False
            and ensemble.get("independent_sampling_axis_counts") == {
                "optimizer_starts": 24, "experimental_replicas": 50}
            and int(ensemble.get(
                "available_same_replica_cross_optimizer_check_count", -1))
                == len(replicas.get("cross_optimizer_checks", [])),
            "combined hierarchy is not demonstrably centered once about the central")
    require("empirical q16--q84" in ensemble.get("interval", "")
            and "neither a formal 68%" in ensemble.get(
                "interval_probability_semantics", ""),
            "combined interval probability semantics are overstated or missing")
    transform = ensemble["transform_settings"]
    require(transform["tail_mode"] == "expb2"
            and float(transform["b_transform_max"]) == 24.0
            and int(transform["n_b_transform"]) == 6001
            and float(transform["k_max"]) == 4.0
            and int(transform["n_k"]) == 401
            and float(transform["end_taper_start_fraction"]) == 0.92,
            "final Fig. 6 transform settings differ from the validated prescription")
    require(stability["status"] == "complete" and stability["endpoint_gate_pass"],
            "final sampling-robustness gate failed")
    require(stability["band_integrity_gate_pass"],
            "final persisted band-table integrity gate failed")
    require(stability["fnp_band_integrity"]["grid_points_per_component"] >= 3,
            "final persisted FNP band-table integrity gate failed")
    require(stability["coverage_gate_pass"], "final robustness coverage gate failed")
    allowances = stability["resampling_full_width_allowance_by_flavor"]
    final_widths = stability["final_max_active_relative_full_width"]
    champion_widths = stability[
        "comparison_champion_max_active_relative_full_width"]
    require(set(allowances) == {"u", "d"}
            and set(final_widths) == {"u", "d"}
            and set(champion_widths) == {"u", "d"}
            and all(math.isfinite(float(allowances[key]))
                    and float(allowances[key]) >= 0.0 for key in ("u", "d")),
            "flavor-specific resampling allowances are incomplete")
    require(all(math.isclose(float(champion_widths[key]),
                             LOCKED_INCUMBENT_WIDTHS[key],
                             rel_tol=0.0, abs_tol=1.0e-14)
                for key in ("u", "d")),
            "promotion gate does not use immutable registered lambda1 widths")
    union_diagnostic = stability.get(
        "comparison_champion_union_mask_relative_full_width", {})
    require(set(union_diagnostic) == {"u", "d"}
            and all(math.isfinite(float(union_diagnostic[key]))
                    and float(union_diagnostic[key]) > 0.0
                    for key in ("u", "d")),
            "secondary union-mask incumbent diagnostic is incomplete")
    for flavor in ("u", "d"):
        expected = (float(final_widths[flavor]) + float(allowances[flavor])
                    < float(champion_widths[flavor]))
        require(bool(stability["robust_improvement_gate_by_flavor"][flavor])
                == expected,
                f"{flavor} robust-improvement decision was not reconstructed flavor-wise")
    require(all(stability["robust_improvement_gate_by_flavor"].values()),
            "final Fig. 6 does not robustly improve both champion flavors")
    require(figures["status"] == "final_validated_figures",
            "validated final figures were not rendered")
    require(figures["updated_only"] and not figures["contains_individual_seed_curves"]
            and not figures["contains_legacy_conditional_result"]
            and figures.get("formal_confidence_level_assigned") is False
            and figures.get("one_sigma_claimed") is False
            and "empirical q16--q84" in figures.get("uncertainty", ""),
            "final figure content violates the frozen updated-only specification")

    figure_dir = SUMMARIES / "final_fig2_fig6"
    artifacts = [
        digest(figure_dir / "updated_fnp_bspace_full_empirical_q16q84.png"),
        digest(figure_dir / "updated_fnp_bspace_full_empirical_q16q84.pdf"),
        digest(figure_dir / "updated_fig2_bspace_full_empirical_q16q84.png"),
        digest(figure_dir / "updated_fig2_bspace_full_empirical_q16q84.pdf"),
        digest(figure_dir / "updated_fig6_kspace_ud_full_empirical_q16q84.png"),
        digest(figure_dir / "updated_fig6_kspace_ud_full_empirical_q16q84.pdf"),
        digest(SUMMARIES / "final_combined_tmd_ensemble/bT_tmd_bands.csv"),
        digest(SUMMARIES / "final_combined_tmd_ensemble/kT_tmd_bands.csv"),
        digest(SUMMARIES / "final_combined_tmd_ensemble/fnp_bands.csv"),
    ]
    handoff = BASE / "HANDOFF.md"
    require(handoff.is_file() and handoff.stat().st_size > 0,
            "detailed resumable handoff is missing")

    summary = {
        "status": "complete",
        "objective_requirements_verified": {
            "isolated_campaign": True,
            "systematic_prior_architecture_inventory_verified": True,
            "exploratory_completed_fit_count": int(
                trial_inventory["completed_run_count"]),
            "dataset_selection_audited": True,
            "frozen_production_inputs_unchanged": True,
            "minimum_constraint_selected": True,
            "minimum_fit_quality_barrier_selected": True,
            "full_replica_failure_generations_preserved": [
                525.0, 562.5, 600.0, 637.5,
            ],
            "selected_constraint_scale_calibrated": True,
            "fit_quality_and_fnp_stationarity": True,
            "central_unpenalized_chi2_delta_from_verified_initializer": central_unpenalized_delta,
            "independent_start_count": 24,
            "fresh_boundary_confirmed_start_count": start_boundary["confirmed_count"],
            "experimental_replica_count": 50,
            "hard_replica_cross_optimizer_fnp_consistency": True,
            "hard_replica_cross_optimizer_check_count":
                len(replicas["cross_optimizer_checks"]),
            "experimental_replica_identity_set": [1001, 1050],
            "boundary_confirmed_replica_seeds": boundary["selected_replica_seeds"],
            "delayed_movers_independently_reconstructed_and_freshly_confirmed":
                sorted(empirical_delayed_seeds),
            "bracket_stress_replicas_independently_freshly_confirmed":
                sorted(bracket_stress_seeds),
            "experimental_replica_unpenalized_chi2_range": [
                min(replica_chi2), max(replica_chi2)],
            "experimental_replica_unpenalized_chi2_ceiling": replica_chi2_ceiling,
            "combined_member_count": 1200,
            "validated_regularized_finite_b_transform": True,
            "combined_band_sampling_robust": True,
            "combined_band_tables_numerically_valid": True,
            "fig6_robustly_improves_registered_champion": True,
            "final_fig2_and_fig6_rendered": True,
            "detailed_handoff_present": True,
        },
        "selected_strength": starts["selected_strength"],
        "selected_bmax": starts["selected_bmax"],
        "fit_quality_barrier_strength": starts.get(
            "fit_quality_barrier_strength", 0.0),
        "fit_quality_barrier_power": starts.get(
            "fit_quality_barrier_power", 2),
        "final_fig6_widths": stability["final_max_active_relative_full_width"],
        "artifacts": artifacts,
        "production_sources_modified": False,
    }
    TARGET.mkdir(parents=True, exist_ok=True)
    (TARGET / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
