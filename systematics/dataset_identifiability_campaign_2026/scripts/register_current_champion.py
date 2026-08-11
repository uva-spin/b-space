#!/usr/bin/env python3
"""Register the best complete 24-start study result as the comparison champion.

This is a study registry, not a production promotion.  The first entry is the
lambda=1 empirical-reference ensemble already evaluated on all 24 exactly
reproduced historical starts and crossed with the 50 conditional experimental
replica residuals.  Later candidates must improve its complete Fig. 2/Fig. 6
uncertainty while preserving the fit; optimizer drift remains diagnostic.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
UNITARY = BASE.parent / "high_qt_direct_production_benchmark/experimental_unitary_transition"
SOURCE = BASE / "summaries/matched_baseline_reference_distance_lam1e00_full24_crossed_experimental"
BSPACE_STABILITY = BASE / "summaries/matched_baseline_reference_distance_lam1e00_full24_bspace_stability/summary.json"
KSPACE_STABILITY = BASE / "summaries/matched_baseline_reference_distance_lam1e00_full24_kspace_stability/summary.json"
RUNS = BASE / "summaries/matched_baseline_reference_distance_lam1e00_full24_bspace/ensemble_runs.csv"
FIGURE = BASE / "summaries/baseline_vs_reference_distance_lam1e00/baseline_vs_reference_distance.png"
UPDATED_FIG2SPACE = BASE / "summaries/champion_registry/current_fig2_fig6/champion_fig2space_bT_ud_combined_1sigma.png"
UPDATED_FIG6 = BASE / "summaries/champion_registry/current_fig2_fig6/champion_fig6_kT_ud_combined_1sigma.png"
TARGET = BASE / "summaries/champion_registry"
HISTORICAL = {"u": 0.23996770478880355, "d": 0.2553179594428333}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    combined = json.loads((SOURCE / "summary.json").read_text())
    b_stability = json.loads(BSPACE_STABILITY.read_text())
    k_stability = json.loads(KSPACE_STABILITY.read_text())
    runs = pd.read_csv(RUNS)
    deltas = []
    endpoint_tags = []
    for tag in runs.run_tag.astype(str):
        endpoint = BASE / "outputs" / tag
        source = UNITARY / "outputs" / f"fig6_lbfgs_stationary_s{int(json.loads((endpoint/'fit_status.json').read_text())['seed'])}"
        final_status = json.loads((endpoint / "fit_status.json").read_text())
        source_status = json.loads((source / "fit_status.json").read_text())
        deltas.append(float(final_status["final"]["unpenalized_total_chi2"]) -
                      float(source_status["final"]["total_chi2"]))
        endpoint_tags.append(tag)
    widths = {flavor: float(combined["metrics"][flavor]["max_full_width_relative"])
              for flavor in ("u", "d")}
    record = {
        "status": "provisional_complete_24start_champion_not_production",
        "champion_id": "empirical_reference_lambda1_b0p1_2p0_full24",
        "method": "pointwise distance to empirical median FNP on 0.1<=bT<=2, lambda=1",
        "start_count": 24,
        "experimental_replica_count": 50,
        "combined_member_count_per_flavor": 1200,
        "max_source_relative_unpenalized_chi2_delta": max(deltas),
        "median_source_relative_unpenalized_chi2_delta": float(pd.Series(deltas).median()),
        "combined_fig6_max_active_relative_full_width": widths,
        "historical_baseline_fig6_max_active_relative_full_width": HISTORICAL,
        "relative_width_reduction": {key: 1.0 - widths[key] / HISTORICAL[key] for key in widths},
        "resampling_sensitivity": {
            "bspace_bootstrap_p95_endpoint_movement": b_stability["bootstrap_max_relative_endpoint_change"]["p95"],
            "bspace_split_half_p95_endpoint_difference": b_stability["split_half_max_relative_endpoint_difference"]["p95"],
            "kspace_bootstrap_p95_endpoint_movement": k_stability["bootstrap_max_relative_endpoint_change"]["p95"],
            "kspace_split_half_p95_endpoint_difference": k_stability["split_half_max_relative_endpoint_difference"]["p95"],
        },
        "promotion_semantics": "first complete fit-preserving ensemble better than the historical Fig2/Fig6 baseline; retained as the study champion despite finite-ensemble sensitivity",
        "optimizer_drift_threshold_is_primary_gate": False,
        "limitations": [
            "reference median is empirical and not reciprocal-cross-fitted",
            "24-start interval endpoints retain documented 7-10% bootstrap/split-half sensitivity",
            "this registry does not modify or promote frozen production",
        ],
        "endpoint_tags": endpoint_tags,
        "artifacts": {
            "combined_summary": str(SOURCE / "summary.json"),
            "bspace_combined_bands": str(SOURCE / "bspace_combined_bands.csv"),
            "kspace_combined_bands": str(SOURCE / "kspace_combined_bands.csv"),
            "comparison_figure": str(FIGURE),
            "updated_only_fig2_space": str(UPDATED_FIG2SPACE),
            "updated_only_fig6": str(UPDATED_FIG6),
        },
        "artifact_sha256": {
            "combined_summary": sha256(SOURCE / "summary.json"),
            "bspace_combined_bands": sha256(SOURCE / "bspace_combined_bands.csv"),
            "kspace_combined_bands": sha256(SOURCE / "kspace_combined_bands.csv"),
            "comparison_figure": sha256(FIGURE),
            "updated_only_fig2_space": sha256(UPDATED_FIG2SPACE),
            "updated_only_fig6": sha256(UPDATED_FIG6),
        },
        "production_sources_modified": False,
    }
    TARGET.mkdir(parents=True, exist_ok=True)
    (TARGET / "current.json").write_text(json.dumps(record, indent=2) + "\n")
    (TARGET / f"{record['champion_id']}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
