#!/usr/bin/env python3
"""Build a reviewed Tevatron-only pilot table from the global-DY intake.

This is intentionally not a final fit-table builder. It filters the existing
canonical global intake to CDF/D0 Z-boson qT datasets, preserves provenance,
and adds the review fields needed before these rows can be used in the
Drell-Yan fit.

This script uses only the Python standard library so it can run even when the
local pandas/numpy environment is not available.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any


TEVATRON_DATASETS: dict[str, dict[str, Any]] = {
    "CDF_RUN_1": {
        "beam_config": "pbar_p",
        "sqrtS_GeV": 1800.0,
        "experiment": "CDF",
        "run_period": "Run I",
        "final_state": "mu+mu-",
        "observable_hint": "absolute_or_published_qT_spectrum_A",
        "norm_source_hint": "sys. B + / normalization-like 3.90%",
    },
    "CDF_RUN_2": {
        "beam_config": "pbar_p",
        "sqrtS_GeV": 1960.0,
        "experiment": "CDF",
        "run_period": "Run II",
        "final_state": "e+e-",
        "observable_hint": "absolute_or_published_qT_spectrum_A",
        "norm_source_hint": "sys. C +/- appears normalization-like 5.80%",
    },
    "D0_RUN_1": {
        "beam_config": "pbar_p",
        "sqrtS_GeV": 1800.0,
        "experiment": "D0",
        "run_period": "Run I",
        "final_state": "e+e-",
        "observable_hint": "absolute_or_published_qT_spectrum_A",
        "norm_source_hint": "sys,Normalisation error 4.40%",
    },
    "D0_RUN_2": {
        "beam_config": "pbar_p",
        "sqrtS_GeV": 1960.0,
        "experiment": "D0",
        "run_period": "Run II",
        "final_state": "e+e-",
        "observable_hint": "A derived from normalizedCS/fidcs columns",
        "norm_source_hint": "normalized spectrum; fiducial normalization requires review",
    },
    "D0_RUN_2N": {
        "beam_config": "pbar_p",
        "sqrtS_GeV": 1960.0,
        "experiment": "D0",
        "run_period": "Run II",
        "final_state": "mu+mu-",
        "observable_hint": "A derived from NormCS/fidcs columns",
        "norm_source_hint": "normalized spectrum; fiducial normalization requires review",
    },
}

OUTPUT_COLUMNS = [
    "dataset", "row_id", "source_file", "source_row", "experiment", "run_period",
    "beam_config", "process", "final_state", "sqrtS", "Q", "Q_low", "Q_high",
    "qT", "qT_low", "qT_high", "qT_bin_width", "qT_over_Q", "y", "y_low",
    "y_high", "eta", "x1", "x2", "observable_value", "observable_unc",
    "stat_unc", "syst_unc", "norm_unc", "unit", "unit_status", "observable_hint",
    "observable_status", "uncertainty_status", "fit_region", "kinematic_candidate",
    "diagnostic_fit_candidate", "physics_fit_ready", "review_notes",
]

ISSUE_COLUMNS = ["dataset", "row_id", "source_file", "source_row", "severity", "issue"]


def to_float(value: Any) -> float:
    if value is None:
        return math.nan
    text = str(value).strip().replace(",", "")
    if text == "" or text.lower() in {"nan", "none"}:
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def finite(value: Any) -> bool:
    return math.isfinite(to_float(value))


def fit_region(qt_over_q: Any) -> str:
    value = to_float(qt_over_q)
    if not math.isfinite(value):
        return "missing_qT_over_Q"
    if value <= 0.20:
        return "tmd_core_qT_over_Q_le_0p2"
    if value <= 0.50:
        return "matched_candidate_qT_over_Q_le_0p5"
    return "outside_matched_qT_over_Q_gt_0p5"


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def number_text(value: float) -> str:
    return "" if not math.isfinite(value) else f"{value:.15g}"


def finite_values(rows: list[dict[str, str]], column: str) -> list[float]:
    values = [to_float(row.get(column, "")) for row in rows]
    return [v for v in values if math.isfinite(v)]


def min_text(rows: list[dict[str, str]], column: str) -> str:
    values = finite_values(rows, column)
    return number_text(min(values)) if values else ""


def max_text(rows: list[dict[str, str]], column: str) -> str:
    values = finite_values(rows, column)
    return number_text(max(values)) if values else ""


def median_text(rows: list[dict[str, str]], column: str) -> str:
    values = finite_values(rows, column)
    return number_text(float(median(values))) if values else ""


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_pilot(canonical_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    pilot: list[dict[str, Any]] = []
    for row in canonical_rows:
        dataset = row.get("dataset", "")
        if dataset not in TEVATRON_DATASETS:
            continue
        cfg = TEVATRON_DATASETS[dataset]
        source_row = int(to_float(row.get("source_row", "")))
        qT_high = to_float(row.get("qT_high", ""))
        qT_low = to_float(row.get("qT_low", ""))
        qT_bin_width = qT_high - qT_low if math.isfinite(qT_high) and math.isfinite(qT_low) else math.nan
        observable_value = to_float(row.get("cross_section", ""))
        observable_unc = to_float(row.get("total_unc", ""))
        qt_over_q = to_float(row.get("qT_over_Q", ""))
        region = fit_region(qt_over_q)
        unit = str(row.get("unit", "")).strip()
        kinematic = all(
            finite(row.get(col, ""))
            for col in ["Q", "qT", "sqrtS", "x1", "x2"]
        ) and math.isfinite(observable_value) and observable_value > 0.0 and math.isfinite(observable_unc) and observable_unc > 0.0
        diagnostic = kinematic and math.isfinite(qt_over_q) and qt_over_q <= 0.50
        review_notes = (
            f"{cfg['observable_hint']}; {cfg['norm_source_hint']}; "
            "unit/bin-width/covariance review required before physics fit"
        )
        out = {
            "dataset": dataset,
            "row_id": f"{dataset}:{source_row}",
            "source_file": row.get("source_file", ""),
            "source_row": source_row,
            "experiment": cfg["experiment"],
            "run_period": cfg["run_period"],
            "beam_config": cfg["beam_config"],
            "process": "pbar p -> gamma*/Z -> lepton pair + X",
            "final_state": cfg["final_state"],
            "sqrtS": row.get("sqrtS", ""),
            "Q": row.get("Q", ""),
            "Q_low": row.get("Q_low", ""),
            "Q_high": row.get("Q_high", ""),
            "qT": row.get("qT", ""),
            "qT_low": row.get("qT_low", ""),
            "qT_high": row.get("qT_high", ""),
            "qT_bin_width": number_text(qT_bin_width),
            "qT_over_Q": row.get("qT_over_Q", ""),
            "y": row.get("y", ""),
            "y_low": row.get("y_low", ""),
            "y_high": row.get("y_high", ""),
            "eta": row.get("eta", ""),
            "x1": row.get("x1", ""),
            "x2": row.get("x2", ""),
            "observable_value": row.get("cross_section", ""),
            "observable_unc": row.get("total_unc", ""),
            "stat_unc": row.get("stat_unc", ""),
            "syst_unc": row.get("syst_unc", ""),
            "norm_unc": row.get("norm_unc", ""),
            "unit": unit,
            "unit_status": "provided" if unit else "missing",
            "observable_hint": cfg["observable_hint"],
            "observable_status": "needs_dataset_level_definition_review",
            "uncertainty_status": "diagonal_total_unc_only_for_now; correlated_systematics_not_encoded",
            "fit_region": region,
            "kinematic_candidate": bool_text(kinematic),
            "diagnostic_fit_candidate": bool_text(diagnostic),
            "physics_fit_ready": "false",
            "review_notes": review_notes,
        }
        pilot.append(out)
    pilot.sort(key=lambda r: (r["dataset"], int(r["source_row"])))
    return pilot


def build_summary(pilot: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pilot:
        grouped[row["dataset"]].append(row)
    summary = []
    for dataset in sorted(grouped):
        rows = grouped[dataset]
        summary.append({
            "dataset": dataset,
            "n_rows": len(rows),
            "n_tmd_core": sum(r["fit_region"] == "tmd_core_qT_over_Q_le_0p2" for r in rows),
            "n_matched_candidate": sum(r["diagnostic_fit_candidate"] == "true" for r in rows),
            "n_outside_matched": sum(r["fit_region"] == "outside_matched_qT_over_Q_gt_0p5" for r in rows),
            "Q_min": min_text(rows, "Q"),
            "Q_max": max_text(rows, "Q"),
            "qT_min": min_text(rows, "qT"),
            "qT_max": max_text(rows, "qT"),
            "qT_over_Q_median": median_text(rows, "qT_over_Q"),
            "y_min": min_text(rows, "y"),
            "y_max": max_text(rows, "y"),
            "observable_min": min_text(rows, "observable_value"),
            "observable_max": max_text(rows, "observable_value"),
            "all_units_provided": bool_text(all(r["unit_status"] == "provided" for r in rows)),
            "physics_fit_ready": "false",
        })
    return summary


def build_issues(pilot: list[dict[str, Any]]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for row in pilot:
        base = {
            "dataset": row["dataset"],
            "row_id": row["row_id"],
            "source_file": row["source_file"],
            "source_row": row["source_row"],
        }
        checks = [
            ("blocker", "unit_missing", row["unit_status"] == "missing"),
            ("blocker", "observable_definition_unreviewed", True),
            ("blocker", "correlated_systematics_not_encoded", True),
            ("warning", "outside_matched_qT_over_Q_gt_0p5", row["fit_region"] == "outside_matched_qT_over_Q_gt_0p5"),
            ("warning", "missing_or_nonpositive_uncertainty", not (finite(row["observable_unc"]) and to_float(row["observable_unc"]) > 0.0)),
            ("warning", "missing_or_nonpositive_observable", not (finite(row["observable_value"]) and to_float(row["observable_value"]) > 0.0)),
        ]
        for severity, issue, active in checks:
            if active:
                issues.append({**base, "severity": severity, "issue": issue})
    return issues


def issue_counts(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for issue in issues:
        counts[(issue["severity"], issue["issue"])] += 1
    return [
        {"severity": severity, "issue": issue, "n": n}
        for (severity, issue), n in sorted(counts.items())
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--canonical",
        default="v23/outputs/global_dy_data_intake_v2/canonical_candidate_rows.csv",
        help="Canonical candidate rows from audit_v23_global_dy_data_intake_v2.py",
    )
    parser.add_argument("--out", default="v23/outputs/tevatron_pilot_table")
    args = parser.parse_args()

    canonical_path = Path(args.canonical)
    if not canonical_path.exists():
        raise SystemExit(f"Missing canonical intake table: {canonical_path}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    canonical_rows = read_csv(canonical_path)
    pilot = build_pilot(canonical_rows)
    summary = build_summary(pilot)
    issues = build_issues(pilot)
    counts = issue_counts(issues)

    write_csv(out / "tevatron_pilot_rows.csv", pilot, OUTPUT_COLUMNS)
    write_csv(out / "tevatron_pilot_summary.csv", summary, list(summary[0].keys()) if summary else ["dataset"])
    write_csv(out / "tevatron_pilot_issues.csv", issues, ISSUE_COLUMNS)
    write_csv(out / "tevatron_pilot_issue_counts.csv", counts, ["severity", "issue", "n"])

    manifest = {
        "input": str(canonical_path),
        "out": str(out),
        "datasets": list(TEVATRON_DATASETS),
        "n_rows": len(pilot),
        "n_diagnostic_fit_candidates": sum(r["diagnostic_fit_candidate"] == "true" for r in pilot),
        "n_physics_fit_ready": sum(r["physics_fit_ready"] == "true" for r in pilot),
        "outputs": {
            "rows": str(out / "tevatron_pilot_rows.csv"),
            "summary": str(out / "tevatron_pilot_summary.csv"),
            "issues": str(out / "tevatron_pilot_issues.csv"),
            "issue_counts": str(out / "tevatron_pilot_issue_counts.csv"),
        },
        "interpretation": (
            "Rows marked diagnostic_fit_candidate have usable kinematics and qT/Q<=0.5, "
            "but physics_fit_ready is false until observable units, bin-width conventions, "
            "and correlated systematics are reviewed dataset by dataset."
        ),
        "next_steps": [
            "Resolve observable units and bin-width conventions for each CDF/D0 table.",
            "Decide whether D0 normalized spectra are fit as normalized distributions or converted to absolute spectra with fiducial normalization uncertainty.",
            "Encode luminosity/correlated systematics as nuisance parameters or covariance matrices.",
            "Only after those reviews, build a Tevatron check-only backend cache and baseline-vs-data audit.",
        ],
    }
    (out / "tevatron_pilot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print("\n=== Tevatron pilot manifest ===")
    print(json.dumps(manifest, indent=2))
    print("\n=== Tevatron pilot summary ===")
    if summary:
        for row in summary:
            print(row)
    else:
        print("(no rows)")
    print("\n=== Tevatron pilot issue counts ===")
    if counts:
        for row in counts:
            print(row)
    else:
        print("(no issues)")
    print("\nwrote:", out)


if __name__ == "__main__":
    main()
