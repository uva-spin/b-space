#!/usr/bin/env python3
"""Compare the full lambda=3 W+Y candidate cross with frozen lambda=1.

This is read-only with respect to the frozen production package.  It compares
the already-rendered 96-start x 50-replica q16--q84 bands, fit objectives, and
central-shape shifts, while recording the W/Y and prior differences that make
the comparison non-causal as a strict method test.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
SYSTEMATICS = BASE.parent
PRODUCTION = SYSTEMATICS / "dataset_identifiability_campaign_2026/production_lambda1_empirical_reference_full96x50"
CANDIDATE = BASE / "reports/scope_329_refdist3_full96x50_long50k_final_fig2_fig6"
CROSS = BASE / "reports/scope_329_refdist3_full96x50_long50k_propagation"
OUT = CANDIDATE / "full_production_comparison.json"
PROBES = CANDIDATE / "full_production_comparison_probes.csv"


def load_band(path: Path, x: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["flavor"] = frame["flavor"].astype(str)
    frame["width"] = frame["q84"] - frame["q16"]
    frame["relative_width"] = frame["width"] / frame["central"].abs().clip(lower=1.0e-30)
    frame["relative_halfwidth"] = 0.5 * frame["relative_width"]
    return frame.sort_values(["flavor", x]).reset_index(drop=True)


def nearest(frame: pd.DataFrame, flavor: str, x: str, value: float) -> pd.Series:
    subset = frame[frame.flavor.eq(flavor)].copy()
    return subset.iloc[(subset[x] - value).abs().argmin()]


def fit_objectives() -> dict:
    rows = []
    for seed in range(303, 399):
        path = BASE / f"reports/scope_329_refdist3_full96x50_long50k_start_s{seed}/metrics.json"
        if path.exists():
            data = json.loads(path.read_text())
            rows.append(float(data["train"]["final_chi2_like"]))
    baseline = []
    summary_path = SYSTEMATICS / "dataset_identifiability_campaign_2026/production_lambda1_empirical_reference_full96x50/start_only_summary.json"
    endpoint_tags = json.loads(summary_path.read_text())["endpoint_tags"]
    for tag in endpoint_tags:
        path = SYSTEMATICS / f"dataset_identifiability_campaign_2026/outputs/{tag}/fit_status.json"
        if path.exists():
            data = json.loads(path.read_text())
            baseline.append(float(data["final"]["objective_per_row"]))
    return {
        "candidate_start_count": len(rows),
        "candidate_start_objective_per_row": {
            "median": float(np.median(rows)), "min": float(np.min(rows)), "max": float(np.max(rows))
        },
        "baseline_stationary_start_count": len(baseline),
        "baseline_objective_per_row": {
            "median": float(np.median(baseline)), "min": float(np.min(baseline)), "max": float(np.max(baseline))
        },
    }


def main() -> None:
    b_base = load_band(PRODUCTION / "bspace_combined_bands.csv", "bT")
    b_cand = load_band(CANDIDATE / "fig2_bspace_start_replica_bands.csv", "bT")
    k_base = load_band(PRODUCTION / "kspace_combined_bands.csv", "kT")
    k_cand = load_band(CANDIDATE / "fig6_kspace_start_replica_bands.csv", "kT")

    probe_rows = []
    for space, x, base, cand, points in (
        ("b", "bT", b_base, b_cand, (0.5, 1.0, 2.0, 3.0, 4.0, 8.0)),
        ("k", "kT", k_base, k_cand, (0.0, 0.5, 1.0, 1.5, 2.0)),
    ):
        for flavor in ("u", "d"):
            for point in points:
                old = nearest(base, flavor, x, point)
                new = nearest(cand, flavor, x, point)
                probe_rows.append({
                    "space": space, "flavor": flavor, x: float(new[x]),
                    "baseline_central": float(old.central), "candidate_central": float(new.central),
                    "central_fractional_shift": float(new.central / old.central - 1.0) if old.central else None,
                    "baseline_relative_q16_q84_width": float(old.relative_width),
                    "candidate_relative_q16_q84_width": float(new.relative_width),
                    "width_ratio_candidate_over_baseline": float(new.relative_width / old.relative_width) if old.relative_width else None,
                })
    probes = pd.DataFrame(probe_rows)
    probes.to_csv(PROBES, index=False)

    metrics = {}
    for space, x, base, cand, xmax in (
        ("b", "bT", b_base, b_cand, 4.0),
        ("k", "kT", k_base, k_cand, 2.25),
    ):
        metrics[space] = {}
        for flavor in ("u", "d"):
            out = {}
            for label, frame in (("baseline", base), ("candidate", cand)):
                view = frame[(frame.flavor.eq(flavor)) & (frame[x] <= xmax)].copy()
                active = view.central > 0.05 * view.central.max()
                out[label] = {
                    "max_active_relative_full_width": float(view.loc[active, "relative_width"].max()),
                    "median_active_relative_full_width": float(view.loc[active, "relative_width"].median()),
                    "active_x_max": float(view.loc[active, x].max()),
                }
            metrics[space][flavor] = out

    fnp = pd.read_csv(CROSS / "fnp_start_replica_quantiles_x0p1.csv")
    fnp_probe = {}
    for value in (1.0, 2.0, 4.0, 8.0):
        row = fnp.iloc[(fnp.bT - value).abs().argmin()]
        fnp_probe[str(value)] = {
            "bT": float(row.bT),
            "candidate_crossed_relative_q16_q84_width": float(row.relative_crossed_full_width),
        }

    result = {
        "status": "complete_candidate_vs_frozen_lambda1_comparison",
        "candidate": {
            "lambda_reference_distance": 3.0,
            "reference_b_range": [0.1, 8.0],
            "finite_y": "non-LHCb rows; six LHCb_7 rows retained with Y=0",
            "start_count": 96, "replica_count": 50, "crossed_member_count": 4800,
        },
        "baseline": {
            "lambda_reference_distance": 1.0,
            "reference_b_range": [0.1, 2.0],
            "finite_y": "W-only historical production ensemble",
            "start_count": 96, "replica_count": 50, "crossed_member_count": 4800,
        },
        "fit_objectives": fit_objectives(),
        "band_metrics": metrics,
        "candidate_fnp_crossed_probes": fnp_probe,
        "interpretation": {
            "improvement": "The candidate narrows the kT≈0 endpoint and suppresses the bT≈3-4 spread.",
            "regression": "It broadens the bT<2 core and the intermediate/high active kT envelope; it is not a smaller total uncertainty band.",
            "central_shift": "The candidate central F_NP is about 6-8% lower than baseline at bT=1-2 and about 8% higher near bT=4, so it is a genuine shape change rather than only an error-band change.",
            "causal_limit": "The comparison changes the W grid, W-only versus non-LHCb finite-Y objective, and reference prior simultaneously.",
            "decision": "Do not promote lambda_ref=3 as an improved replacement. Retain it as a useful candidate showing that the stronger full-range prior relocates, rather than removes, the core non-uniqueness.",
        },
        "frozen_production_modified": False,
        "promotion_authorized": False,
        "artifacts": {"probes": str(PROBES), "candidate_figures": str(CANDIDATE)},
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
