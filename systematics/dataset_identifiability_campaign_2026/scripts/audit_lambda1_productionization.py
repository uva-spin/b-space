#!/usr/bin/env python3
"""Audit the complete lambda=1 study before production packaging.

This audit accepts the documented finite-ensemble resampling sensitivity as a
known limitation. It does not silently reinterpret that sensitivity as a
68-percent statistical gate.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
SUMMARIES = BASE / "summaries"
REGISTRY = SUMMARIES / "champion_registry"
CHAMPION = REGISTRY / "empirical_reference_lambda1_b0p1_2p0_full24.json"
SOURCE = SUMMARIES / "matched_baseline_reference_distance_lam1e00_full24_crossed_experimental"
BSPACE_RUNS = SUMMARIES / "matched_baseline_reference_distance_lam1e00_full24_bspace/ensemble_runs.csv"
BSPACE_STABILITY = SUMMARIES / "matched_baseline_reference_distance_lam1e00_full24_bspace_stability/summary.json"
KSPACE_STABILITY = SUMMARIES / "matched_baseline_reference_distance_lam1e00_full24_kspace_stability/summary.json"
TARGET = SUMMARIES / "lambda1_productionization_audit"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    champion = json.loads(CHAMPION.read_text())
    combined = json.loads((SOURCE / "summary.json").read_text())
    b_stability = json.loads(BSPACE_STABILITY.read_text())
    k_stability = json.loads(KSPACE_STABILITY.read_text())
    runs = pd.read_csv(BSPACE_RUNS)

    require(champion["champion_id"] == "empirical_reference_lambda1_b0p1_2p0_full24",
            "unexpected lambda=1 champion id")
    require(champion["start_count"] == 24 and champion["experimental_replica_count"] == 50,
            "incomplete lambda=1 ensemble counts")
    require(champion["combined_member_count_per_flavor"] == 1200,
            "unexpected crossed ensemble size")
    require(combined["start_count"] == 24 and combined["experimental_replica_count"] == 50,
            "combined summary lacks exact 24-start/50-replica coverage")
    require(set(runs["fit_seed"].astype(int)) == set(range(303, 327)),
            "b-space ensemble does not contain exactly seeds 303--326")
    require(not runs["production_state_modified"].astype(bool).any(),
            "source ensemble reports modified production state")

    historical = champion["historical_baseline_fig6_max_active_relative_full_width"]
    widths = champion["combined_fig6_max_active_relative_full_width"]
    for flavor in ("u", "d"):
        require(float(widths[flavor]) < float(historical[flavor]),
                f"lambda=1 does not improve historical Fig.6 width for {flavor}")

    artifacts = {
        "champion_record": CHAMPION,
        "combined_summary": SOURCE / "summary.json",
        "bspace_combined_bands": SOURCE / "bspace_combined_bands.csv",
        "kspace_combined_bands": SOURCE / "kspace_combined_bands.csv",
        "fig2_png": REGISTRY / "current_fig2_fig6/champion_fig2space_bT_ud_combined_1sigma.png",
        "fig2_pdf": REGISTRY / "current_fig2_fig6/champion_fig2space_bT_ud_combined_1sigma.pdf",
        "fig6_png": REGISTRY / "current_fig2_fig6/champion_fig6_kT_ud_combined_1sigma.png",
        "fig6_pdf": REGISTRY / "current_fig2_fig6/champion_fig6_kT_ud_combined_1sigma.pdf",
    }
    for name, path in artifacts.items():
        require(path.is_file() and path.stat().st_size > 0, f"missing artifact: {name}")

    record = {
        "status": "pass_for_productionization_with_documented_limitations",
        "champion_id": champion["champion_id"],
        "method": champion["method"],
        "coverage": {
            "start_count": 24,
            "start_seeds": list(range(303, 327)),
            "experimental_replica_count": 50,
            "combined_member_count_per_flavor": 1200,
        },
        "fit_preservation": {
            "max_source_relative_unpenalized_chi2_delta": champion[
                "max_source_relative_unpenalized_chi2_delta"],
            "median_source_relative_unpenalized_chi2_delta": champion[
                "median_source_relative_unpenalized_chi2_delta"],
        },
        "fig6_widths": widths,
        "historical_widths": historical,
        "resampling": {
            "bspace": b_stability["resampling_sensitivity"] if "resampling_sensitivity" in b_stability else {
                "bootstrap_p95": b_stability["bootstrap_max_relative_endpoint_change"]["p95"],
                "split_half_p95": b_stability["split_half_max_relative_endpoint_difference"]["p95"],
            },
            "kspace": k_stability["resampling_sensitivity"] if "resampling_sensitivity" in k_stability else {
                "bootstrap_p95": k_stability["bootstrap_max_relative_endpoint_change"]["p95"],
                "split_half_p95": k_stability["split_half_max_relative_endpoint_difference"]["p95"],
            },
            "interpretation": "documented finite-ensemble sensitivity; not added in quadrature and not relabeled as a confidence-level failure",
        },
        "limitations": champion["limitations"],
        "production_sources_modified_before_promotion": False,
        "artifacts": {name: str(path) for name, path in artifacts.items()},
        "artifact_sha256": {name: sha256(path) for name, path in artifacts.items()},
    }
    TARGET.mkdir(parents=True, exist_ok=True)
    (TARGET / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
