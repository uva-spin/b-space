#!/usr/bin/env python3
"""Collect the isolated initial DY+SIDIS pilot and closure diagnostics.

This report deliberately keeps the central external-FF pilot, the corrected
longer/converged pilot, the circular HAPS comparison, and the NNFF10 replica
profile in one place.  It is a decision record, not a promotion script.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports"
TRIALS = {
    "nnff10_binavg_3000": ROOT / "outputs/initial_joint_dy_compass_collinear_binavg_pilot_reoptimized",
    "nnff10_binavg_2500": ROOT / "outputs/initial_joint_dy_compass_collinear_binavg_pilot_converged",
    "haps_binavg_1500_circular": ROOT / "outputs/initial_joint_dy_compass_collinear_haps_binavg_pilot_reoptimized",
    "apfel_nlo_full_den_diagnostic": ROOT / "outputs/initial_joint_dy_compass_apfel_nlo_full_den_probe_validated_named",
    "nnff10_midpoint_legacy": ROOT / "outputs/initial_joint_dy_compass_collinear_pilot",
}
PROFILE = ROOT / "reports/nnff10_replica_profile.json"
PROFILE_MIDPOINT = ROOT / "reports/nnff10_replica_profile_midpoint.json"


def _history_diagnostic(path: Path) -> dict[str, object]:
    history = pd.read_csv(path)
    objective = history["total_chi2"].to_numpy(float)
    if len(objective) < 2:
        return {"epochs_recorded": int(len(objective))}
    tail = objective[max(0, len(objective) - 5):]
    previous = objective[max(0, len(objective) - 10):max(0, len(objective) - 5)]
    return {
        "epochs_recorded": int(history.epoch.iloc[-1]),
        "last_total_chi2": float(objective[-1]),
        "last_five_total_chi2": [float(x) for x in tail],
        "tail_relative_change": float((tail[-1] - previous[0]) / max(abs(previous[0]), 1.0))
        if len(previous) else None,
    }


def _trial_summary(name: str, directory: Path) -> dict[str, object]:
    fit = json.loads((directory / "fit_summary.json").read_text())
    prediction = pd.read_csv(directory / "sidis_predictions.csv")
    prediction["group"] = prediction.hadron.astype(str) + prediction.charge.astype(str)
    groups = {}
    for group, rows in prediction.groupby("group", sort=True):
        pulls = rows.pull.to_numpy(float)
        groups[group] = {
            "rows": int(len(rows)),
            "chi2": float(np.sum(pulls * pulls)),
            "chi2_per_row": float(np.mean(pulls * pulls)),
            "mean_pull": float(np.mean(pulls)),
            "rms_pull": float(np.sqrt(np.mean(pulls * pulls))),
        }
    result = {
        "name": name,
        "path": str(directory),
        "ff_family": fit["ff"].get("family"),
        "kinematic_mode": fit.get("kinematics", {}).get("mode"),
        "rows_available": fit["sidis_rows_available"],
        "rows_fit": fit["sidis_rows_fit"],
        "rows_excluded": fit["sidis_rows_excluded"],
        "objective": fit["objective"],
        "sidis_normalizations": fit.get("sidis_normalizations", {}),
        "sidis_normalization_initialization": fit.get("sidis_normalization_initialization", {}),
        "groups": groups,
        "history": _history_diagnostic(directory / "loss_history.csv"),
        "classification": (
            "APFEL NLO numerator + NLO denominator diagnostic"
            if str(fit.get("ratio_source", "")).startswith("external ratio probe")
            else
            "external-FF primary diagnostic" if fit["ff"].get("family") == "nnff10_nnlo"
            else "circular external comparison: HAPS FFs were fitted using modern SIDIS data"
            if fit["ff"].get("family") == "haps_nnlo"
            else "legacy optimizer-control diagnostic"
        ),
    }
    return result


def main() -> None:
    trials = [_trial_summary(name, directory) for name, directory in TRIALS.items()]
    profile = {
        "midpoint": json.loads(PROFILE_MIDPOINT.read_text()),
        "bin_average": json.loads(PROFILE.read_text()),
    }
    report = {
        "status": "initial_joint_dy_sidis_validation_complete_not_production",
        "scope": "329 frozen lambda=1 DY rows plus 746 identified COMPASS 2026 pi/K collinear rows",
        "trials": trials,
        "nnff10_replica_profiles": profile,
        "conclusions": [
            "The reinitialized joint optimizer is materially better behaved than the legacy all-scales-near-one pilot, but central NNFF10 still gives a poor scalar SIDIS closure (17.13 chi2/row in the 2500-epoch run).",
            "The HAPS comparison reaches 2.94 chi2/row, but HAPS is not independent because its FFs incorporate modern COMPASS SIDIS information; it is a diagnostic, not evidence that the observable implementation is closed.",
            "Across all 101 NNFF10 members, the best raw member can make many rows non-positive and is invalid; the best member with all 746 rows positive remains a poor closure candidate. FF replicas alone do not resolve the mismatch.",
            "The APFEL SIDIS-NLO numerator with a full massless NLO inclusive-DIS denominator gives 12.98 chi2/row on 738 positive rows (the earlier LO-denominator diagnostic gave 12.75); eight rows remain non-positive and bin-integrated normalization is still unvalidated.",
            "DY non-regression is demonstrated in every pilot (about 0.394 chi2/row), while no pilot is authorized for production or for TMDFF uncertainty propagation.",
        ],
        "next_gate": "Implement and independently validate the NNLO SIDIS coefficient-function plus inclusive-DIS denominator/normalization interface, then rerun the same scope before adding HERMES or transverse TMDFF data.",
        "production_files_modified": False,
        "promotion_authorized": False,
    }
    (OUT / "initial_fit_trials.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Initial joint DY+SIDIS trial register (2026-08-26)",
        "",
        "This is an isolated validation record, not a production result.  The DY anchor is the frozen lambda=1 W-only 329-row solution; SIDIS is the 746-row identified COMPASS 2026 addendum collinear scope.",
        "",
        "| trial | FF | mode | DY chi2/row | SIDIS chi2/row | SIDIS rows | excluded | classification |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for trial in trials:
        obj = trial["objective"]
        lines.append(
            f"| {trial['name']} | {trial['ff_family']} | {trial['kinematic_mode']} | "
            f"{obj['dy_chi2_per_row']:.4f} | {obj['sidis_chi2_per_row']:.4f} | "
            f"{trial['rows_fit']} | {len(trial['rows_excluded'])} | {trial['classification']} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        *[f"- {item}" for item in report["conclusions"]],
        "",
        "The current fits establish an actual joint DY+SIDIS software path and expose a scalar observable-closure problem.  They do not justify a global/TMD production claim: the COMPASS addendum has no transverse axis, the public table lacks a full covariance, HERMES identity is unresolved, and the proper perturbative SIDIS coefficient/denominator interface remains to be independently validated.",
    ]
    (OUT / "initial_fit_trials.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
