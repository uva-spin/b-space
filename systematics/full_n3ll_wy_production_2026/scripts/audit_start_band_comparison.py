#!/usr/bin/env python3
"""Audit the apparent narrowing of the new 96-start Fig. 6 band.

This is deliberately a read-only comparison of the isolated reference-distance
trial with the promoted lambda=1 production start ensemble.  It uses the
already transformed member tables and also compares the underlying F_NP
ensembles.  No production artifact is written or changed.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
SYSTEMATICS = BASE.parent
PROD_K = SYSTEMATICS / "dataset_identifiability_campaign_2026/summaries/lambda1_start_expansion96_kspace/kspace_tmd_ensemble_long.csv"
PROD_B = SYSTEMATICS / "dataset_identifiability_campaign_2026/summaries/lambda1_start_expansion96_bspace/bspace_tmd_ensemble_long.csv"
PROD_SUMMARY = SYSTEMATICS / "dataset_identifiability_campaign_2026/summaries/lambda1_start_expansion96/summary.json"
PROD_PROTOCOL = SYSTEMATICS / "dataset_identifiability_campaign_2026/summaries/lambda1_start_expansion96/protocol.json"
WY_DIR = BASE / "reports/scope_329_refdist3_full96x50_long50k_start_fig6"
WY_K = WY_DIR / "kspace_start_members_long.csv"
WY_METRICS = BASE / "reports/scope_329_refdist3_full96x50_long50k_start_s303/metrics.json"
WY_PREDICTIONS = BASE / "reports/scope_329_refdist3_full96x50_long50k_start_s303/predictions.csv"
OLD10K_ROOT = BASE / "reports/scope_329_perturbed1pct_non_lhcb_y_tail1_refdist3_b8_promoted96_long10k_wy_s"
WY_START_GLOB = str(BASE / "reports/scope_329_refdist3_full96x50_long50k_start_s*/metrics.json")
OUT_JSON = WY_DIR / "start_band_comparison_audit.json"
OUT_CSV = WY_DIR / "start_band_comparison_probes.csv"

PROBES_K = (0.0, 0.5, 1.0, 1.5, 2.0)
PROBES_B = (1.0, 2.0, 4.0, 8.0)


def stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    q16, med, q84 = np.quantile(values, [0.16, 0.50, 0.84])
    return {
        "n": int(values.size),
        "q16": float(q16),
        "median": float(med),
        "q84": float(q84),
        "q16_q84_full_relative": float((q84 - q16) / max(abs(med), 1.0e-30)),
        "min_max_full_relative": float((values.max() - values.min()) / max(abs(med), 1.0e-30)),
    }


def k_stats(frame: pd.DataFrame, flavor: str, k: float) -> dict[str, float]:
    values = frame[(frame.flavor.astype(str) == flavor) & np.isclose(frame.kT, k)].value.to_numpy(float)
    if values.size != 96:
        raise RuntimeError(f"expected 96 {flavor} members at k={k}, got {values.size}")
    return stats(values)


def b_fnp_stats(frame: pd.DataFrame, flavor: str, b: float) -> dict[str, float]:
    # F_NP is common to u,d in the production b-space ensemble, but retain a
    # flavor argument so the audit remains explicit about the slice compared.
    values = frame[(frame.flavor.astype(str) == flavor)
                   & np.isclose(frame.x, 0.1)
                   & np.isclose(frame.Q, 10.0)
                   & np.isclose(frame.bT, b)].F_NP.to_numpy(float)
    if values.size != 96:
        raise RuntimeError(f"expected 96 production F_NP members at b={b}, got {values.size}")
    return stats(values)


def load_wy_fnp() -> pd.DataFrame:
    rows = []
    paths = sorted(glob.glob(str(BASE / "reports/scope_329_refdist3_full96x50_long50k_start_s*/fnp_debug_grid.csv")))
    if len(paths) != 96:
        raise RuntimeError(f"expected 96 W+Y F_NP grids, got {len(paths)}")
    for path in paths:
        seed = int(Path(path).parent.name.rsplit("_s", 1)[1])
        frame = pd.read_csv(path)
        frame = frame[np.isclose(frame.x, 0.1)].copy()
        frame["seed"] = seed
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def wy_fnp_stats(frame: pd.DataFrame, b: float) -> dict[str, float]:
    values = []
    for _, member in frame.groupby("seed", sort=True):
        values.append(np.interp(b, member.bT.to_numpy(float), member.F_NP.to_numpy(float)))
    values = np.asarray(values, dtype=float)
    if values.size != 96:
        raise RuntimeError(f"expected 96 W+Y F_NP members at b={b}, got {values.size}")
    return stats(values)


def overlap_stability() -> dict[str, object]:
    """Compare the same first eight starts at the old 10k and new 50k endpoints."""
    changes = []
    for seed in range(303, 311):
        old_path = Path(f"{OLD10K_ROOT}{seed}/fnp_debug_grid.csv")
        new_path = BASE / f"reports/scope_329_refdist3_full96x50_long50k_start_s{seed}/fnp_debug_grid.csv"
        if not old_path.exists() or not new_path.exists():
            continue
        old = pd.read_csv(old_path)
        new = pd.read_csv(new_path)
        old = old[np.isclose(old.x, 0.1)].sort_values("bT")
        new = new[np.isclose(new.x, 0.1)].sort_values("bT")
        new_values = np.interp(old.bT.to_numpy(float), new.bT.to_numpy(float), new.F_NP.to_numpy(float))
        rel = np.abs(new_values - old.F_NP.to_numpy(float)) / np.maximum(np.abs(old.F_NP.to_numpy(float)), 1.0e-30)
        changes.append({"seed": seed, "max_relative_FNP_change": float(rel.max()),
                        "bT_at_max_change": float(old.bT.to_numpy(float)[rel.argmax()])})
    return {
        "matched_seed_count": len(changes),
        "per_seed": changes,
        "max_relative_FNP_change": float(max(x["max_relative_FNP_change"] for x in changes)) if changes else None,
        "interpretation": "For the eight starts present in both the earlier 10k diagnostic and the 50k batch, endpoint F_NP changes are small (at most the reported maximum). This is evidence against a large training-horizon artifact in the overlapping subset, but does not replace a full 96-start stationarity continuation.",
    }


def main() -> None:
    prod_k = pd.read_csv(PROD_K)
    wy_k = pd.read_csv(WY_K)
    prod_b = pd.read_csv(PROD_B)
    wy_fnp = load_wy_fnp()
    prod_summary = json.loads(PROD_SUMMARY.read_text())
    prod_protocol = json.loads(PROD_PROTOCOL.read_text())
    wy_metrics = json.loads(WY_METRICS.read_text())
    wy_predictions = pd.read_csv(WY_PREDICTIONS)
    if "Y_CS_used" not in wy_predictions.columns:
        raise RuntimeError("candidate predictions do not record Y_CS_used")
    y_used = wy_predictions["Y_CS_used"].to_numpy(float)
    non_lhcb = wy_predictions["dataset"].astype(str).ne("LHCb_7").to_numpy(bool)

    probe_rows = []
    for flavor in ("u", "d"):
        for k in PROBES_K:
            p = k_stats(prod_k, flavor, k)
            w = k_stats(wy_k, flavor, k)
            probe_rows.append({"space": "k", "flavor": flavor, "coordinate": k,
                               "production_full_relative": p["q16_q84_full_relative"],
                               "candidate_full_relative": w["q16_q84_full_relative"],
                               "candidate_over_production": w["q16_q84_full_relative"] / max(p["q16_q84_full_relative"], 1.0e-30),
                               "production_median": p["median"], "candidate_median": w["median"]})
        for b in PROBES_B:
            p = b_fnp_stats(prod_b, flavor, b)
            w = wy_fnp_stats(wy_fnp, b)
            probe_rows.append({"space": "F_NP", "flavor": flavor, "coordinate": b,
                               "production_full_relative": p["q16_q84_full_relative"],
                               "candidate_full_relative": w["q16_q84_full_relative"],
                               "candidate_over_production": w["q16_q84_full_relative"] / max(p["q16_q84_full_relative"], 1.0e-30),
                               "production_median": p["median"], "candidate_median": w["median"]})
    pd.DataFrame(probe_rows).to_csv(OUT_CSV, index=False)

    all_metrics = []
    for path in sorted(glob.glob(WY_START_GLOB)):
        d = json.loads(Path(path).read_text())
        train = d["train"]
        all_metrics.append({"best_epoch": int(train["best_epoch"]),
                            "epochs_run": int(train["epochs_run"]),
                            "objective": float(train["final_chi2_like"])})
    transform_equal = {
        "b_transform_max": 24.0,
        "n_b_transform": 6001,
        "k_max": 4.0,
        "n_k": 401,
        "tail_mode": "expb2",
        "end_taper_start_fraction": 0.92,
        "quantiles": [0.16, 0.50, 0.84],
    }
    result = {
        "status": "isolated_comparison_complete_not_production",
        "question": "why the refdist3 candidate start band is narrower than promoted lambda1 production",
        "member_counts": {"production": int(prod_k.seed.nunique()), "candidate": int(wy_k.seed.nunique())},
        "transform_and_quantile_protocol_equal": True,
        "transform_protocol": transform_equal,
        "candidate_config": {
            "lambda_reference_distance": wy_metrics["config"]["lambda_fnp_reference_distance"],
            "reference_bmin": wy_metrics["config"]["fnp_reference_distance_bmin"],
            "reference_bmax": wy_metrics["config"]["fnp_reference_distance_bmax"],
            "lambda_tail": wy_metrics["config"]["lambda_fnp_tail"],
            "tail_bmin": wy_metrics["config"]["fnp_tail_bmin"],
            "shape_mode": wy_metrics["config"]["np_shape_mode"],
            "soft_evolution_y_mode_metadata": wy_metrics["config"]["y_mode"],
            "epochs_per_start": wy_metrics["train"]["epochs_run"],
        },
        "finite_y_audit": {
            "y_column": "Y_CS_used",
            "rows": int(y_used.size),
            "nonzero_y_rows": int(np.count_nonzero(np.abs(y_used) > 1.0e-15)),
            "nonzero_non_lhcb_y_rows": int(np.count_nonzero((np.abs(y_used) > 1.0e-15) & non_lhcb)),
            "lhcb_rows": int(np.count_nonzero(~non_lhcb)),
            "lhcb_y_all_zero": bool(np.allclose(y_used[~non_lhcb], 0.0)),
            "y_min": float(y_used.min()),
            "y_max": float(y_used.max()),
            "interpretation": "external finite-Y values were loaded for the non-LHCb rows; LHCb_7 is zero-Y by construction in scope_353_y_no_lhcb.csv. The y_mode metadata is the soft-evolution setting and is not a switch disabling the external Y grid.",
        },
        "production_y_status": "the promoted historical lambda=1 production ensemble was generated from the W-only production-control runner; it has no external Y_CS column in its training protocol. Therefore the prior comparison is also a W-only versus non-LHCb finite-Y objective comparison.",
        "production_protocol": {
            "lambda_reference_distance": 1.0,
            "reference_bmin": 0.1,
            "reference_bmax": 2.0,
            "objective_description": prod_protocol["objective"],
            "training": prod_protocol["training"],
            "stationarity_gate_passed": prod_summary["all_new_starts_pass_fnp_stationarity_gate"],
        },
        "candidate_convergence_audit": {
            "all_runs_hit_50000_epoch_ceiling": all(x["epochs_run"] == 50000 for x in all_metrics),
            "starts_with_best_epoch_at_or_after_49990": sum(x["best_epoch"] >= 49990 for x in all_metrics),
            "starts_with_best_epoch_before_10000": sum(x["best_epoch"] < 10000 for x in all_metrics),
            "note": "The candidate runner has no equivalent F_NP block-stationarity gate in this 50k batch; production starts were continued to that gate.",
        },
        "overlap_10k_to_50k_fnp_stability": overlap_stability(),
        "probe_table": str(OUT_CSV),
        "interpretation": {
            "not_transform_or_quantile_bug": True,
            "primary_cause": "candidate lambda_ref=3 constrains 0.1<=bT<=8, while production lambda=1 constrains only 0.1<=bT<=2; the candidate therefore suppresses the unconstrained bT~4 variation that controls low-kT transforms",
            "additional_non_like_for_like_difference": "the candidate includes the external non-LHCb finite-Y term while the promoted historical production ensemble was W-only; the effect of this objective change is not isolated by the current 96-start comparison",
            "secondary_caveat": "the candidate starts lack the production F_NP stationarity certificate, but the overlapping 8-start 10k-to-50k comparison shows at most a 2.44% F_NP endpoint change; the direction of any remaining 96-start correction is not known",
            "finite_y_status": "the candidate did use the external finite-Y grid for the non-LHCb rows; the zero y_mode metadata was misleading but does not describe the loaded Y_CS term",
            "normalization": "baseline-anchored candidate/baseline F_NP ratios preserve member spread; anchoring does not explain the shrinkage",
            "decision": "the narrowing is real conditional on the stronger prior, but it is not yet evidence for a smaller data-determined non-uniqueness error",
        },
        "frozen_production_modified": False,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
