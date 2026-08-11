#!/usr/bin/env python3
"""Cross the validated 96-start lambda=1 ensemble with 50 replica residuals.

This is an additive production-update diagnostic.  It reuses the existing
conditional experimental-replica residuals and replaces the 24-start
non-uniqueness component with the complete 96-start ensemble.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
SYSTEMATICS = BASE.parent
ROOT = SYSTEMATICS.parent
START_SUMMARY = BASE / "summaries/lambda1_start_expansion96/summary.json"
BSPACE = BASE / "summaries/lambda1_start_expansion96_bspace/bspace_tmd_ensemble_long.csv"
KSPACE = BASE / "summaries/lambda1_start_expansion96_kspace/kspace_tmd_ensemble_long.csv"
TARGET = BASE / "summaries/matched_baseline_reference_distance_lam1e00_full96_crossed_experimental"
REPLICA_B = (
    SYSTEMATICS / "collins_factorization_validity/replicas/"
    "rowidfix_stageFT_all_qmax0p20_lam0p50_lambda3_50rep/"
    "tmd_bspace_bands_exactx_50rep/v22_tmd_replica_bspace_long.csv")
REPLICA_K_ROOT = ROOT / "plots/prd_q020_figures"
BASELINE_K = (
    SYSTEMATICS
    / "high_qt_direct_production_benchmark/experimental_unitary_transition/"
    "summaries/fig6_updated_ud_band/fig6_updated_ud_central_1sigma.csv")


def crossed_quantiles(starts: np.ndarray, replicas: np.ndarray) -> np.ndarray:
    residuals = replicas - np.median(replicas, axis=0)
    crossed = (starts[:, None, :] + residuals[None, :, :]).reshape(
        -1, starts.shape[1])
    return np.quantile(crossed, [0.16, 0.50, 0.84], axis=0)


def main() -> None:
    start_summary = json.loads(START_SUMMARY.read_text())
    if start_summary["status"] != "complete" or not start_summary["all_new_starts_pass_fnp_stationarity_gate"]:
        raise RuntimeError("96-start stationarity audit is not complete")
    b_long = pd.read_csv(BSPACE)
    k_long = pd.read_csv(KSPACE)
    replica_b = pd.read_csv(REPLICA_B)
    baseline_k = pd.read_csv(BASELINE_K)
    b_rows, k_rows, metrics = [], [], {}
    for flavor in ("u", "d"):
        b_group = b_long[
            np.isclose(b_long["x"], .1) & np.isclose(b_long["Q"], 10)
            & b_long["flavor"].astype(str).eq(flavor)]
        b_wide = b_group.pivot(index="bT", columns="run_tag", values="ftilde").sort_index()
        rb = replica_b[
            np.isclose(replica_b["x"], .1) & np.isclose(replica_b["Q"], 10)
            & replica_b["flavor"].astype(str).eq(flavor)]
        rb_wide = rb.pivot(index="bT", columns="seed", values="ftilde").sort_index()
        if not np.allclose(b_wide.index, rb_wide.index):
            raise RuntimeError("b-space grids do not match")
        bq = crossed_quantiles(b_wide.to_numpy(float).T, rb_wide.to_numpy(float).T)
        for i, coordinate in enumerate(b_wide.index.to_numpy(float)):
            b_rows.append({"flavor": flavor, "bT": coordinate,
                           "q16": bq[0, i], "central": bq[1, i], "q84": bq[2, i]})

        k_group = k_long[
            k_long["quantity"].eq("ftilde")
            & np.isclose(k_long["x"], .1) & np.isclose(k_long["Q"], 10)
            & k_long["flavor"].astype(str).eq(flavor)]
        k_wide = k_group.pivot(index="kT", columns="_replica_key", values="value").sort_index()
        rk = pd.read_csv(REPLICA_K_ROOT / f"kspace_fixedx_q10_{flavor}_current/"
                         "v23a_regularized_kspace_replica_long.csv")
        rk = rk[
            rk["quantity"].eq("ftilde") & np.isclose(rk["x"], .1)
            & np.isclose(rk["Q"], 10) & rk["flavor"].astype(str).eq(flavor)]
        rk_wide = rk.pivot(index="kT", columns="seed", values="value").sort_index()
        if not np.allclose(k_wide.index, rk_wide.index):
            raise RuntimeError("k-space grids do not match")
        kq = crossed_quantiles(k_wide.to_numpy(float).T, rk_wide.to_numpy(float).T)
        for i, coordinate in enumerate(k_wide.index.to_numpy(float)):
            k_rows.append({"flavor": flavor, "kT": coordinate,
                           "q16": kq[0, i], "central": kq[1, i], "q84": kq[2, i]})

        k = k_wide.index.to_numpy(float)
        active = (k <= 2.25) & (kq[1] > .05 * np.max(kq[1, k <= 2.25]))
        old = baseline_k[baseline_k["flavor"].astype(str).eq(flavor)].sort_values("kT")
        oldq = old[["q16", "central", "q84"]].to_numpy(float).T
        metrics[flavor] = {
            "max_lower_relative_excursion": float(np.max((kq[1, active] - kq[0, active]) / kq[1, active])),
            "max_upper_relative_excursion": float(np.max((kq[2, active] - kq[1, active]) / kq[1, active])),
            "max_full_width_relative": float(np.max((kq[2, active] - kq[0, active]) / kq[1, active])),
            "baseline_max_full_width_relative_same_active": float(np.max((oldq[2, active] - oldq[0, active]) / oldq[1, active])),
        }
    TARGET.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(b_rows).to_csv(TARGET / "bspace_combined_bands.csv", index=False)
    pd.DataFrame(k_rows).to_csv(TARGET / "kspace_combined_bands.csv", index=False)
    summary = {
        "status": "isolated_matched_baseline_crossed_96start_experimental_not_production",
        "start_count": 96, "experimental_replica_count": 50,
        "crossed_member_count_per_flavor": 4800,
        "metrics": metrics,
        "band_interpretation": "empirical q16-q84 after crossing 96 stationary lambda=1 starts with conditional experimental-replica residuals",
        "start_only_summary": str(START_SUMMARY),
        "bspace_ensemble": str(BSPACE), "kspace_ensemble": str(KSPACE),
        "production_sources_modified": False,
    }
    (TARGET / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
