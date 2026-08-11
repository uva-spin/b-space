#!/usr/bin/env python3
"""Expand the strength bracket after a failed full-50 replica campaign.

The prospective six stress replicas remain in the set and every failure from
the complete 50-replica attempt is added.  Only strengths above the failed
prescription are considered.  Each candidate starts from the same lambda=300
central used by the original bracket, so stronger candidates do not inherit a
weaker candidate's endpoint.  Replicas exposed as late failures must also run
for at least 40k in later stress trials before an apparent plateau is accepted.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

from alternate_lambda_authorization import (
    require_alternate_lambda_authorization,
)


BASE = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/dustin/miniforge3/envs/pdf-fit/bin/python")
BRACKET_PATH = BASE / "summaries/full_reference_replica_strength_bracket/summary.json"
FAILED_PATH = BASE / "summaries/selected_reference_central_replicas/failed_80k_summary.json"
STRENGTHS = (562.5, 600.0, 637.5, 675.0, 712.5, 750.0, 825.0, 900.0,
             1000.0, 1100.0, 1200.0, 1350.0, 1500.0, 1650.0, 1800.0,
             2000.0, 2200.0, 2400.0, 2700.0, 3000.0)


def load_module():
    path = BASE / "scripts/bracket_full_reference_replica_strength.py"
    spec = importlib.util.spec_from_file_location("expanded_replica_bracket", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    require_alternate_lambda_authorization(
        "escalate_reference_strength_after_full50_failure")
    prior = json.loads(BRACKET_PATH.read_text())
    failed = json.loads(FAILED_PATH.read_text())
    failed_strength = float(failed["selected_strength"])
    failed_seeds = tuple(int(value) for value in failed["failed_replica_seeds"])
    if not failed_seeds:
        raise RuntimeError("escalation requested without failed replicas")
    # Put newly exposed full-ensemble failures first.  They are the most
    # discriminating cases for the next strength, and a failure there makes
    # running the older stress members scientifically redundant.  dict order
    # retains every unique member, so this changes cost but not coverage.
    stress = tuple(dict.fromkeys(failed_seeds + tuple(prior["stress_replica_seeds"])))
    bracket = load_module()
    # The same-strength independent confirmation must run two complete 5k
    # blocks beyond the longest failed full-ensemble trajectory.  The current
    # failure reached 155k, so its minimum qualifying endpoint is 165k.  Keep
    # one additional block of headroom: a 160k cap would make passing the
    # stated rule mathematically impossible and falsely force escalation.
    # Match the full-ensemble audit's 200k ceiling.  The previous 170k cap
    # could not confirm two quiet blocks after a reversal at a stress-required
    # 165k boundary.
    bracket.MAX_CHUNKS = 40
    bracket.TARGET.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    outcomes = dict(prior["candidate_outcomes"])
    outcomes[f"{failed_strength:g}"] = {
        "all_stress_replicas_pass": False,
        "failed_replica": failed_seeds[0],
        "evidence": "failed the full 50-replica campaign through the extended horizon",
    }
    selected = None
    selected_endpoints: list[str] = []
    # Re-evaluate the failed strength itself only when this invocation offers
    # a genuinely longer horizon than the failed full-ensemble run.  Once a
    # member has failed at the full 160k ceiling, replaying the same objective
    # cannot add evidence and would create an infinite recursive loop.
    prior_ledger = BASE / "summaries/selected_reference_central_replicas/runs.csv"
    prior_max_iterations = 0
    if prior_ledger.exists():
        import pandas as pd
        rows = pd.read_csv(prior_ledger)
        rows = rows[rows["replica_seed"].isin(failed_seeds)]
        if not rows.empty:
            prior_max_iterations = int(rows["cumulative_lbfgs_iterations"].max())
    horizon_iterations = 5000 * bracket.MAX_CHUNKS
    retest_failed_strength = prior_max_iterations < horizon_iterations
    eligible_strengths = (
        value for value in STRENGTHS
        if value > failed_strength
        or (value == failed_strength and retest_failed_strength)
    )
    for strength in eligible_strengths:
        endpoints = []
        failed_replica = None
        for replica_seed in stress:
            passed, endpoint = bracket.run_case(strength, replica_seed, records)
            endpoints.append(endpoint.name)
            if not passed:
                failed_replica = replica_seed
                break
        outcomes[f"{strength:g}"] = {
            "all_stress_replicas_pass": failed_replica is None,
            "failed_replica": failed_replica,
            "endpoint_tags": endpoints,
        }
        if failed_replica is None:
            selected = strength
            selected_endpoints = endpoints
            break
    if selected is None:
        raise RuntimeError("no expanded strength passed the augmented stress set")
    strongest_fail = max(float(key) for key, value in outcomes.items()
                         if not value["all_stress_replicas_pass"])
    relative_width = (selected - strongest_fail) / selected
    if relative_width > .10:
        raise RuntimeError("expanded fail/pass strength bracket remains wider than 10%")
    summary = {
        "status": "complete",
        "candidate_strengths_weakest_first": sorted(float(key) for key in outcomes),
        "stress_replica_seeds": list(stress),
        "initialization_rule": "same selected lambda=300 stationary central for every strength/replica case",
        "stationarity_rule": "two consecutive unchanged-objective 5k blocks with max selected-domain FNP drift <=2%; replicas previously failing an extended full ensemble require >=40k before acceptance; every accepted endpoint must also satisfy unpenalized chi2 <= N+5sqrt(2N)",
        "drift_sensitivity_thresholds": list(bracket.SENSITIVITY),
        "candidate_outcomes": outcomes,
        "weakest_tested_passing_strength": selected,
        "strongest_tested_failing_strength": strongest_fail,
        "relative_fail_pass_bracket_width": relative_width,
        "bracket_resolution_rule": "stop after fail/pass width is <=10% of passing strength",
        "selected_endpoint_tags": selected_endpoints,
        "expanded_after_full50_failure": True,
        "full50_failed_strength": failed_strength,
        "failed_full50_max_iterations": prior_max_iterations,
        "same_strength_retested_with_longer_horizon": retest_failed_strength,
        "added_stress_replica_seeds": list(failed_seeds),
        "next_step": "reverify all 24 starts, then restart central plus 50 replicas",
        "production_sources_modified": False,
    }
    BRACKET_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    subprocess.run([str(PYTHON), str(BASE / "scripts/verify_replica_robust_reference_full24.py")],
                   check=True)
    subprocess.run([str(PYTHON), str(BASE / "scripts/summarize_replica_robust_constraint_scale.py")],
                   check=True)
    subprocess.run([str(PYTHON), str(BASE / "scripts/supervise_selected_reference_central_replicas.py")],
                   check=True)
    current_summary = BASE / "summaries/selected_reference_central_replicas/summary.json"
    result = json.loads(current_summary.read_text())
    if result["status"] != "complete":
        # Preserve both generations of evidence, then recurse with the newly
        # failed full-ensemble members added to the augmented stress set.
        archive = FAILED_PATH.with_name(
            f"failed_lam{failed_strength:g}_80k_summary.json")
        if FAILED_PATH.exists():
            FAILED_PATH.replace(archive)
        current_summary.replace(FAILED_PATH)
        main()


if __name__ == "__main__":
    main()
