#!/usr/bin/env python3
"""Re-bracket fit-barrier strength at the selected reference strength."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

from alternate_lambda_authorization import (
    require_alternate_lambda_authorization,
)


BASE = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/dustin/miniforge3/envs/pdf-fit/bin/python")
GENERIC = BASE / "scripts/audit_fitbar_candidate_long_horizon.py"
STRENGTH_SEARCH = BASE / "summaries/minimum_fitbar_constraint_search/summary.json"
TARGET = BASE / "summaries/minimum_barrier_constraint_search"
MU_VALUES = (30.0, 10.0, 3.0, 1.0, 0.3, 0.1, 0.0)
CASES = {
    637.5: (1033, 2033, "fullref_lam637p5_b4_central_polish64_50000"),
    600.0: (1041, 5041, "fullref_lam600_b4_central_polish64_15000"),
    562.5: (1032, 2032, "fullref_lam562p5_b4_central_polish64_15000"),
    525.0: (1032, 2032, "fullref_lam525_b4_central_polish64_15000"),
}
def token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def write(status: str, strength: float, trials: list[dict], selected: float | None,
          lower_rejected: float | None) -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "selected_reference_strength": strength,
        "barrier_power": 2,
        "tested_barrier_strengths": [100.0, *MU_VALUES],
        "trials": trials,
        "selected_weakest_surviving_barrier_strength": selected,
        "strongest_rejected_barrier_strength_below_selection": lower_rejected,
        "selection_rule": (
            "independently test every prospective lower mu from the same "
            "stationary central through 40 unchanged-objective 5k-capacity continuation checkpoints; select the smallest survivor only when mu=0 "
            "or another weaker tested value rejects"
        ),
        "production_sources_modified": False,
    }
    (TARGET / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    require_alternate_lambda_authorization("continue_minimum_barrier_search")
    strength_result = json.loads(STRENGTH_SEARCH.read_text())
    if strength_result["status"] != "complete":
        raise RuntimeError("reference-strength search is incomplete")
    strength = float(strength_result["selected_weakest_surviving_strength"])
    if strength == 675.0:
        # The existing fixed-lambda rescue ladder is already the authoritative
        # barrier bracket for lambda675.
        write("complete", strength, [], 100.0, 30.0)
        return
    if strength in CASES:
        replica, fit_seed, initial_tag = CASES[strength]
    else:
        lambda_evidence = next(
            item for item in strength_result["tested_candidates"]
            if float(item["strength"]) == strength
        )
        long_result = json.loads(Path(lambda_evidence["evidence"]).read_text())
        if long_result["status"] != "candidate_survives_discriminator":
            raise RuntimeError("selected lower-lambda hard case is not stationary")
        replica = int(long_result["replica_seed"])
        fit_seed = int(long_result["fit_seed"])
        initial_tag = str(long_result["endpoint_tag"])
    trials = [{"barrier_strength": 100.0, "outcome": "survives",
               "evidence": "minimum_fitbar_constraint_search"}]
    write("in_progress", strength, trials, 100.0, None)
    for mu in MU_VALUES:
        name = (f"barrier_candidate_lam{token(strength)}_mu{token(mu)}_"
                "long_horizon")
        subprocess.run([
            str(PYTHON), str(GENERIC), "--strength", str(strength),
            "--replica-seed", str(replica), "--fit-seed", str(fit_seed),
            "--initial-tag", initial_tag, "--target-name", name,
            "--barrier-strength", str(mu), "--barrier-power", "2",
            "--mandatory-iterations", "200000",
            "--maximum-iterations", "250000",
        ], check=True)
        evidence = BASE / "summaries" / name / "summary.json"
        result = json.loads(evidence.read_text())
        if result["status"] == "candidate_survives_discriminator":
            outcome = "survives"
        elif result["status"] == "candidate_rejected":
            outcome = "rejected"
        else:
            raise RuntimeError(f"nonterminal barrier result: {result['status']}")
        trials.append({"barrier_strength": mu, "outcome": outcome,
                       "evidence": str(evidence)})
        write("in_progress", strength, trials, None, None)
    survivors = [float(item["barrier_strength"]) for item in trials
                 if item["outcome"] == "survives"]
    selected = min(survivors)
    rejected_below = [float(item["barrier_strength"]) for item in trials
                      if item["outcome"] == "rejected"
                      and float(item["barrier_strength"]) < selected]
    if selected > 0.0 and not rejected_below:
        raise RuntimeError("barrier minimum has no rejected tested value below it")
    # Mu=0 is the exact lower boundary.  If it survives, no rejected value can
    # exist below it and the minimum is nevertheless closed rather than
    # unbracketed.
    write("complete", strength, trials, selected,
          max(rejected_below) if rejected_below else None)
    print((TARGET / "summary.json").read_text(), end="")


if __name__ == "__main__":
    main()
