#!/usr/bin/env python3
"""Descend the fit-barrier reference-strength ladder after the 637.5 control."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from alternate_lambda_authorization import (
    require_alternate_lambda_authorization,
)


BASE = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/dustin/miniforge3/envs/pdf-fit/bin/python")
GENERIC = BASE / "scripts/audit_fitbar_candidate_long_horizon.py"
INITIAL = BASE / "summaries/lambda637p5_fitbar_minimum_control/summary.json"
TARGET = BASE / "summaries/minimum_fitbar_constraint_search"
CANDIDATES = (
    # strength, hard replica, fit seed, stationary central, mandatory horizon
    (600.0, 1041, 5041, "fullref_lam600_b4_central_polish64_15000", 200_000),
    (562.5, 1032, 2032, "fullref_lam562p5_b4_central_polish64_15000", 200_000),
    (525.0, 1032, 2032, "fullref_lam525_b4_central_polish64_15000", 200_000),
    # The earlier 487.5/450 stress classifications used only 40k capacity.
    # Lambda525's eventual fixed point after 155k proves that those short-run
    # failures cannot bracket the long-horizon minimum.  Resume their actual
    # hard-replica endpoints, then continue a fixed descending grid from the
    # common stationary lambda300 initializer.  Lambda0 is the concrete
    # unregularized lower endpoint; no monotonicity assumption is made.
    (487.5, 1001, 5001, "fullref_replica_stress_lam487p5_r1001_polish64_40000", 200_000),
    (450.0, 1001, 5001, "fullref_replica_stress_lam450_r1001_polish64_35000", 200_000),
    (412.5, 1001, 5001, "fullref_lam300_b4_central_polish64_15000", 200_000),
    (375.0, 1001, 5001, "fullref_lam300_b4_central_polish64_15000", 200_000),
    (337.5, 1001, 5001, "fullref_lam300_b4_central_polish64_15000", 200_000),
    (300.0, 1001, 5001, "fullref_lam300_b4_central_polish64_15000", 200_000),
    # Continue the same 37.5 spacing below the previously selected lambda300.
    # Jumping directly to zero can prove that no prior is insufficient but
    # cannot identify the minimum positive constraint required for robustness.
    (262.5, 1001, 5001, "fullref_lam300_b4_central_polish64_15000", 200_000),
    (225.0, 1001, 5001, "fullref_lam300_b4_central_polish64_15000", 200_000),
    (187.5, 1001, 5001, "fullref_lam300_b4_central_polish64_15000", 200_000),
    (150.0, 1001, 5001, "fullref_lam300_b4_central_polish64_15000", 200_000),
    (112.5, 1001, 5001, "fullref_lam300_b4_central_polish64_15000", 200_000),
    (75.0, 1001, 5001, "fullref_lam300_b4_central_polish64_15000", 200_000),
    (37.5, 1001, 5001, "fullref_lam300_b4_central_polish64_15000", 200_000),
    (0.0, 1001, 5001, "fullref_lam300_b4_central_polish64_15000", 200_000),
)

CANDIDATE_ORDER = (637.5,) + tuple(item[0] for item in CANDIDATES)


def vector(tag: str) -> np.ndarray:
    frame = pd.read_csv(BASE / "outputs" / tag / "fnp_grid.csv")
    mask = np.isclose(frame.x, 0.1) & frame.bT.between(0.1, 4.0, inclusive="both")
    return frame.loc[mask, "F_NP"].to_numpy(float)


def write(status: str, tested: list[dict], selected: float | None,
          rejected: float | None) -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "candidate_order_strong_to_weak": list(CANDIDATE_ORDER),
        "tested_candidates": tested,
        "selected_weakest_surviving_strength": selected,
        "strongest_rejected_strength_below_selection": rejected,
        "barrier_strength": 100.0,
        "barrier_power": 2,
        "selection_rule": (
            "independently classify every registered lambda after 40 unchanged-objective "
            "5k-capacity continuation checkpoints plus two fresh post-mandatory quiet blocks on its historically "
            "difficult replica; do not assume observed optimizer outcomes are monotonic"
        ),
        "production_sources_modified": False,
    }
    (TARGET / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    require_alternate_lambda_authorization("continue_minimum_fitbar_search")
    first = json.loads(INITIAL.read_text())
    if first["status"] == "lower_candidate_rejected":
        tested = [{"strength": 637.5, "outcome": "rejected",
                   "evidence": str(INITIAL)}]
    elif first["status"] != "lower_candidate_survives_discriminator":
        raise RuntimeError("lambda637.5 control is not terminal")
    else:
        ledger = pd.read_csv(INITIAL.parent / "runs.csv")
        anchor_rows = ledger[ledger.cumulative_lbfgs_iterations.eq(200_000)]
        if len(anchor_rows) != 1:
            raise RuntimeError("lambda637.5 control lacks its mandatory anchor")
        anchor = vector(str(anchor_rows.iloc[0].tag))
        terminal = vector(str(first["endpoint_tag"]))
        window_drift = float(np.max(np.abs(terminal - anchor) /
                                    np.maximum(anchor, 0.05)))
        if window_drift > 0.02:
            tested = [{
            "strength": 637.5, "outcome": "rejected_post_mandatory_window",
            "evidence": str(INITIAL),
            "post_mandatory_window_fnp_drift": window_drift,
            }]
        else:
            tested = [{"strength": 637.5, "outcome": "survives",
                       "evidence": str(INITIAL),
                       "post_mandatory_window_fnp_drift": window_drift}]
    initial_survivors = [float(item["strength"]) for item in tested
                         if item["outcome"] == "survives"]
    write("in_progress", tested,
          min(initial_survivors) if initial_survivors else 675.0, None)
    for strength, replica, fit_seed, initial_tag, mandatory in CANDIDATES:
        name = f"fitbar_candidate_lam{str(strength).replace('.', 'p')}_long_horizon"
        subprocess.run([
            str(PYTHON), str(GENERIC), "--strength", str(strength),
            "--replica-seed", str(replica), "--fit-seed", str(fit_seed),
            "--initial-tag", initial_tag, "--target-name", name,
            "--barrier-strength", "100", "--barrier-power", "2",
            "--mandatory-iterations", str(mandatory),
            "--maximum-iterations", "250000",
        ], check=True)
        evidence = BASE / "summaries" / name / "summary.json"
        result = json.loads(evidence.read_text())
        if result["status"] == "candidate_rejected":
            tested.append({"strength": strength, "outcome": "rejected",
                           "evidence": str(evidence)})
        elif result["status"] == "candidate_survives_discriminator":
            tested.append({"strength": strength, "outcome": "survives",
                           "evidence": str(evidence)})
        else:
            raise RuntimeError(f"nonterminal candidate result: {result['status']}")
        provisional = min([675.0] + [float(item["strength"]) for item in tested
                                     if item["outcome"] == "survives"])
        write("in_progress", tested, provisional, None)
    survivors = [675.0] + [float(item["strength"]) for item in tested
                 if item["outcome"] == "survives"]
    selected = min(survivors)
    rejected_below = [float(item["strength"]) for item in tested
                      if item["outcome"] == "rejected"
                      and float(item["strength"]) < selected]
    # Lambda=0 is the exact lower boundary of this non-negative penalty.  If
    # it survives, the reference-distance minimum is closed at zero even
    # though there cannot be a rejected strength below it.  The independent
    # barrier search must then determine whether mu can also be reduced.
    if selected == 0.0:
        write("complete", tested, selected, None)
    elif not rejected_below:
        write("unbracketed_below_registered_grid", tested, selected, None)
        raise RuntimeError(
            "the weakest registered positive reference strength survived "
            "without a rejected tested value below it"
        )
    else:
        write("complete", tested, selected, max(rejected_below))
    print((TARGET / "summary.json").read_text(), end="")


if __name__ == "__main__":
    main()
