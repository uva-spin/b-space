#!/usr/bin/env python3
"""Build reviewed Tevatron diagnostic tables from the pilot intake.

The output is still diagnostic, not a production physics table.  It makes the
implicit raw-table assumptions explicit: observable type, provisional units,
row exclusions, normalization groups, and covariance status.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


OUTPUT_COLUMNS = [
    "dataset", "row_id", "source_file", "source_row",
    "qT", "qT_low", "qT_high", "qT_bin_width", "QM", "QM_Low", "QM_High",
    "y", "y_Low", "y_High", "x1", "x2", "SqrtS", "BeamE",
    "CS", "error", "A", "dA", "PreFactor",
    "sysNorm", "sysNorm_rel", "sysP2P", "target", "unit",
    "observable_type", "observable_name", "unit_status", "bin_width_convention",
    "fit_mode", "norm_group", "covariance_group", "covariance_status",
    "final_state", "experiment", "run_period", "qT_over_Q", "fit_region",
    "diagnostic_fit_candidate", "review_status", "production_ready",
    "source_publication", "source_doi", "source_arxiv",
    "review_notes",
]


ISSUE_COLUMNS = ["dataset", "row_id", "source_file", "source_row", "severity", "issue", "detail"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def to_float(value: Any) -> float:
    if value is None:
        return math.nan
    text = str(value).strip().replace(",", "")
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def finite(value: Any) -> bool:
    return math.isfinite(to_float(value))


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def number_text(value: Any) -> str:
    num = to_float(value)
    return "" if not math.isfinite(num) else f"{num:.15g}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def exclusion_rows(dataset_cfg: dict[str, Any]) -> set[int]:
    rows = set()
    for item in dataset_cfg.get("row_exclusions", []):
        value = item.get("source_row")
        if value is not None:
            rows.add(int(value))
    return rows


def fit_mode(observable_type: str) -> str:
    if observable_type == "absolute_differential_cross_section":
        return "absolute_cross_section_with_dataset_norm_nuisance"
    if observable_type == "normalized_shape_with_raw_absolute_conversion":
        return "converted_absolute_diagnostic; normalized_shape_review_required"
    return "unknown"


def covariance_model_usable(status: str) -> bool:
    clean = str(status).strip().lower()
    return clean == "encoded" or clean.startswith("released_")


def region(qt_over_q: Any, tmd_core: float, matched: float) -> str:
    value = to_float(qt_over_q)
    if not math.isfinite(value):
        return "missing_qT_over_Q"
    if value <= tmd_core:
        return f"tmd_core_qT_over_Q_le_{str(tmd_core).replace('.', 'p')}"
    if value <= matched:
        return f"matched_candidate_qT_over_Q_le_{str(matched).replace('.', 'p')}"
    return f"outside_matched_qT_over_Q_gt_{str(matched).replace('.', 'p')}"


def build_rows(pilot: list[dict[str, str]], review: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    datasets = review["datasets"]
    global_policy = review.get("global_policy", {})
    matched_cut = float(global_policy.get("default_qT_over_Q_max", 0.5))
    core_cut = float(global_policy.get("tmd_core_qT_over_Q_max", 0.2))

    for row in pilot:
        dataset = row["dataset"]
        cfg = datasets.get(dataset)
        if cfg is None or not cfg.get("include", False):
            continue

        source_row = int(to_float(row["source_row"]))
        excluded = source_row in exclusion_rows(cfg)
        obs = cfg["observable"]
        unc = cfg["uncertainties"]
        src = cfg["source"]
        obs_type = obs["type"]
        qt_over_q = to_float(row.get("qT_over_Q", ""))
        this_region = region(qt_over_q, core_cut, matched_cut)
        in_diagnostic_region = math.isfinite(qt_over_q) and qt_over_q <= matched_cut

        sys_norm_rel = unc.get("normalization_relative")
        sys_norm_text = "" if sys_norm_rel is None else f"{100.0 * float(sys_norm_rel):.6g}%"
        review_notes = []
        if obs.get("unit_status", "").startswith("provisional"):
            review_notes.append("unit provisional")
        if obs_type != "absolute_differential_cross_section":
            review_notes.append("normalized-source conversion needs final review")
        cov_status = unc.get("production_covariance_status", "")
        if not covariance_model_usable(cov_status):
            review_notes.append("production covariance/nuisance model not encoded")
        if unc.get("correlated_systematic_columns"):
            review_notes.append("released correlated systematic component documented; current trainer uses diagonal dA for first fit")
        if excluded:
            review_notes.append("excluded by review metadata")

        diagnostic = (
            bool(cfg.get("diagnostic_ready", False))
            and in_diagnostic_region
            and not excluded
            and finite(row.get("observable_value"))
            and finite(row.get("observable_unc"))
            and to_float(row.get("observable_value")) > 0.0
            and to_float(row.get("observable_unc")) > 0.0
        )

        out = {
            "dataset": dataset,
            "row_id": row["row_id"],
            "source_file": row["source_file"],
            "source_row": source_row,
            "qT": number_text(row.get("qT")),
            "qT_low": number_text(row.get("qT_low")),
            "qT_high": number_text(row.get("qT_high")),
            "qT_bin_width": number_text(row.get("qT_bin_width")),
            "QM": number_text(row.get("Q")),
            "QM_Low": number_text(row.get("Q_low")),
            "QM_High": number_text(row.get("Q_high")),
            "y": number_text(row.get("y")),
            "y_Low": number_text(row.get("y_low")),
            "y_High": number_text(row.get("y_high")),
            "x1": number_text(row.get("x1")),
            "x2": number_text(row.get("x2")),
            "SqrtS": number_text(row.get("sqrtS")),
            "BeamE": "",
            "CS": number_text(row.get("observable_value")),
            "error": number_text(row.get("observable_unc")),
            "A": number_text(row.get("observable_value")),
            "dA": number_text(row.get("observable_unc")),
            "PreFactor": "1",
            "sysNorm": sys_norm_text,
            "sysNorm_rel": "" if sys_norm_rel is None else number_text(sys_norm_rel),
            "sysP2P": "",
            "target": "pbar_p",
            "unit": obs.get("unit", ""),
            "observable_type": obs_type,
            "observable_name": obs.get("name", ""),
            "unit_status": obs.get("unit_status", ""),
            "bin_width_convention": obs.get("bin_width_convention", ""),
            "fit_mode": fit_mode(obs_type),
            "norm_group": f"{dataset}:normalization" if sys_norm_rel is not None else "",
            "covariance_group": f"{dataset}:covariance",
            "covariance_status": unc.get("production_covariance_status", ""),
            "final_state": obs.get("final_state", ""),
            "experiment": cfg.get("experiment", ""),
            "run_period": cfg.get("run_period", ""),
            "qT_over_Q": number_text(row.get("qT_over_Q")),
            "fit_region": this_region,
            "diagnostic_fit_candidate": bool_text(diagnostic),
            "review_status": "diagnostic_ready" if diagnostic else "not_diagnostic_candidate",
            "production_ready": bool_text(bool(cfg.get("production_ready", False))),
            "source_publication": src.get("publication", ""),
            "source_doi": src.get("doi", ""),
            "source_arxiv": src.get("arxiv", ""),
            "review_notes": "; ".join(review_notes),
        }
        rows.append(out)

        checks = [
            ("blocker", "production_not_ready", not cfg.get("production_ready", False), "metadata production_ready is false"),
            ("blocker", "covariance_not_encoded", not covariance_model_usable(cov_status), "production covariance/nuisance model is not encoded"),
            ("warning", "provisional_unit", obs.get("unit_status", "").startswith("provisional"), obs.get("unit_status", "")),
            ("warning", "normalized_conversion_requires_review", obs_type != "absolute_differential_cross_section", obs_type),
            ("warning", "excluded_row", excluded, "row excluded by metadata"),
            ("warning", "outside_matched_qT_over_Q", not in_diagnostic_region, f"qT/Q={number_text(qt_over_q)}"),
            ("warning", "missing_or_nonpositive_CS", not (finite(out["CS"]) and to_float(out["CS"]) > 0.0), out["CS"]),
            ("warning", "missing_or_nonpositive_error", not (finite(out["error"]) and to_float(out["error"]) > 0.0), out["error"]),
        ]
        for severity, issue, active, detail in checks:
            if active:
                issues.append({
                    "dataset": dataset,
                    "row_id": row["row_id"],
                    "source_file": row["source_file"],
                    "source_row": source_row,
                    "severity": severity,
                    "issue": issue,
                    "detail": detail,
                })

    return rows, issues


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["dataset"]].append(row)
    out = []
    for dataset in sorted(grouped):
        ds_rows = grouped[dataset]
        diag = [r for r in ds_rows if r["diagnostic_fit_candidate"] == "true"]
        out.append({
            "dataset": dataset,
            "n_rows": len(ds_rows),
            "n_diagnostic_fit_candidates": len(diag),
            "n_tmd_core": sum(str(r["fit_region"]).startswith("tmd_core") and r["diagnostic_fit_candidate"] == "true" for r in ds_rows),
            "n_production_ready": sum(r["production_ready"] == "true" for r in ds_rows),
            "observable_type": ds_rows[0]["observable_type"] if ds_rows else "",
            "unit": ds_rows[0]["unit"] if ds_rows else "",
            "unit_status": ds_rows[0]["unit_status"] if ds_rows else "",
            "fit_mode": ds_rows[0]["fit_mode"] if ds_rows else "",
            "covariance_status": ds_rows[0]["covariance_status"] if ds_rows else "",
        })
    return out


def issue_counts(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for issue in issues:
        counts[(issue["severity"], issue["issue"])] += 1
    return [
        {"severity": severity, "issue": issue, "n": n}
        for (severity, issue), n in sorted(counts.items())
    ]


def covariance_plan(review: dict[str, Any]) -> dict[str, Any]:
    plan = {
        "status": "absolute_subset_fit_ready_with_documented_limitations",
        "notes": [
            "Absolute Tevatron rows are fit-ready with released table errors as diagonal point-to-point uncertainties plus explicit normalization nuisances.",
            "D0 Run II normalized spectra are not production-ready until a normalized-theory observable is implemented.",
            "CDF_RUN_2 has an explicitly correlated efficiency systematic in the publication; the current trainer does not support a separate correlated-shape nuisance, so first fits keep the released combined table dA as diagonal.",
        ],
        "datasets": {},
    }
    for dataset, cfg in review["datasets"].items():
        if not cfg.get("include", False):
            continue
        unc = cfg["uncertainties"]
        plan["datasets"][dataset] = {
            "diagonal_total_column": unc.get("diagonal_total_column"),
            "normalization_relative": unc.get("normalization_relative"),
            "normalization_columns": unc.get("normalization_columns", []),
            "stat_columns": unc.get("stat_columns", []),
            "uncorrelated_systematic_columns": unc.get("uncorrelated_systematic_columns", []),
            "correlated_systematic_columns": unc.get("correlated_systematic_columns", []),
            "covariance_available": unc.get("covariance_available"),
            "production_covariance_status": unc.get("production_covariance_status"),
        }
    return plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", default="v23/outputs/tevatron_pilot_table/tevatron_pilot_rows.csv")
    parser.add_argument("--review", default="v23/configs/tevatron_dataset_review.json")
    parser.add_argument("--out", default="v23/outputs/tevatron_reviewed_table")
    parser.add_argument("--out-data-dir", default="Data/v23_tevatron_reviewed_diagnostic")
    args = parser.parse_args()

    pilot_path = Path(args.pilot)
    review_path = Path(args.review)
    out = Path(args.out)
    out_data = Path(args.out_data_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_data.mkdir(parents=True, exist_ok=True)

    pilot = read_csv(pilot_path)
    review = json.loads(review_path.read_text())
    rows, issues = build_rows(pilot, review)
    summary = summarize(rows)
    counts = issue_counts(issues)
    diag_rows = [r for r in rows if r["diagnostic_fit_candidate"] == "true"]
    core_rows = [r for r in diag_rows if str(r["fit_region"]).startswith("tmd_core")]
    absolute_diag_rows = [
        r for r in diag_rows
        if r["observable_type"] == "absolute_differential_cross_section"
    ]

    write_csv(out / "tevatron_reviewed_all_rows.csv", rows, OUTPUT_COLUMNS)
    write_csv(out / "tevatron_reviewed_diagnostic_qToQ_le_0p5.csv", diag_rows, OUTPUT_COLUMNS)
    write_csv(out / "tevatron_reviewed_absolute_diagnostic_qToQ_le_0p5.csv", absolute_diag_rows, OUTPUT_COLUMNS)
    write_csv(out / "tevatron_reviewed_tmd_core_qToQ_le_0p2.csv", core_rows, OUTPUT_COLUMNS)
    write_csv(out / "tevatron_reviewed_summary.csv", summary, list(summary[0].keys()) if summary else ["dataset"])
    write_csv(out / "tevatron_reviewed_issues.csv", issues, ISSUE_COLUMNS)
    write_csv(out / "tevatron_reviewed_issue_counts.csv", counts, ["severity", "issue", "n"])

    # Trainer-style per-dataset files for check-only backend experiments.
    for dataset in sorted({r["dataset"] for r in diag_rows}):
        ds_rows = [r for r in diag_rows if r["dataset"] == dataset]
        write_csv(out_data / f"{dataset}.csv", ds_rows, OUTPUT_COLUMNS)

    cov = covariance_plan(review)
    (out / "tevatron_reviewed_covariance_plan.json").write_text(json.dumps(cov, indent=2) + "\n")

    manifest = {
        "pilot": str(pilot_path),
        "pilot_sha256": sha256(pilot_path),
        "review": str(review_path),
        "review_sha256": sha256(review_path),
        "out": str(out),
        "out_data_dir": str(out_data),
        "n_rows": len(rows),
        "n_diagnostic_fit_candidates": len(diag_rows),
        "n_absolute_diagnostic_fit_candidates": len(absolute_diag_rows),
        "n_tmd_core": len(core_rows),
        "n_production_ready": sum(r["production_ready"] == "true" for r in rows),
        "outputs": {
            "all_rows": str(out / "tevatron_reviewed_all_rows.csv"),
            "diagnostic": str(out / "tevatron_reviewed_diagnostic_qToQ_le_0p5.csv"),
            "absolute_diagnostic": str(out / "tevatron_reviewed_absolute_diagnostic_qToQ_le_0p5.csv"),
            "tmd_core": str(out / "tevatron_reviewed_tmd_core_qToQ_le_0p2.csv"),
            "summary": str(out / "tevatron_reviewed_summary.csv"),
            "issues": str(out / "tevatron_reviewed_issues.csv"),
            "issue_counts": str(out / "tevatron_reviewed_issue_counts.csv"),
            "covariance_plan": str(out / "tevatron_reviewed_covariance_plan.json"),
            "per_dataset_data_dir": str(out_data),
        },
        "interpretation": "Absolute Tevatron rows are usable with the released table errors as diagonal point-to-point uncertainties plus explicit normalization nuisances. D0 Run II normalized-source rows remain diagnostic until the normalized-shape policy is finalized.",
        "fit_ready_absolute_datasets": ["CDF_RUN_1", "CDF_RUN_2", "D0_RUN_1"],
        "excluded_until_normalized_observable": ["D0_RUN_2", "D0_RUN_2N"],
    }
    (out / "tevatron_reviewed_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(json.dumps(manifest, indent=2))
    print("\n=== Summary ===")
    for row in summary:
        print(row)
    print("\n=== Issue counts ===")
    for row in counts:
        print(row)


if __name__ == "__main__":
    main()
