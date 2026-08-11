#!/usr/bin/env python3
"""Audit cross-section fidelity of truncated empirical log-FNP ranks."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
SYSTEMATICS = BASE.parent
ROOT = SYSTEMATICS.parent
TRAINER_PATH = ROOT / "v21_smoothedA_tail_candidate/train_bt_dnn_v21_smoothedA_tail.py"
SOURCE = (
    SYSTEMATICS / "collins_factorization_validity/outputs/"
    "rowidfix_stageFT_E772_qmax0p20_lam0p50_central_s303")
W_GRID = (
    ROOT / "outputs/v23a_tevatron_plus_lhcb7_fidacc_merged_cache/"
    "backend_cache/wpert_v23a_tevatron_plus_lhcb7_fidacc_b160.csv")
BASIS = BASE / "summaries/empirical_logf_pca_basis/basis.npz"
TARGET = BASE / "summaries/empirical_logf_pca_rank_coverage"
CEILING = 119.8021


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    trainer = load_module("rank_coverage_trainer", TRAINER_PATH)
    accepted = pd.read_csv(SOURCE / "predictions.csv")
    b, w = trainer.load_external_w_grid(
        accepted.row_id.astype(str), W_GRID)
    kernel = trainer.precompute_kernel_matrix(
        accepted.qT.to_numpy(float), b, w).astype(float)
    arrays = np.load(BASIS)
    x_knots = arrays["x"]
    if not np.allclose(b, arrays["bT"], rtol=0.0, atol=1.0e-12):
        raise RuntimeError("basis is not on the exact likelihood b grid")
    x_lookup = {float(value): i for i, value in enumerate(x_knots)}
    ix1 = np.asarray([x_lookup[float(value)] for value in accepted.x1])
    ix2 = np.asarray([x_lookup[float(value)] for value in accepted.x2])
    origin = arrays["mean_logf"]
    components = arrays["components"]
    scales = arrays["score_std"]
    scores = arrays["member_scores"]
    seeds = arrays["source_seeds"].astype(int)
    datasets = accepted.dataset.astype(str).to_numpy()
    data = accepted.target_used.to_numpy(float)
    error = accepted.sigma_used.to_numpy(float)
    y_term = accepted.Y_CS_used.to_numpy(float)
    norm_width = accepted.groupby("dataset").norm_rel_used.first()

    rows = []
    for member, seed in enumerate(seeds):
        norms = pd.read_csv(
            BASE / "outputs"
            / f"independent_datafit_D020_E772_init{seed}"
            / "dataset_norms.csv").set_index("dataset")
        norm_column = (
            "control_norm" if "control_norm" in norms else "norm_scale")
        row_norms = norms[norm_column].reindex(datasets).to_numpy(float)
        unique_norms = norms[norm_column].reindex(norm_width.index).to_numpy(float)
        norm_penalty = np.sum(
            ((unique_norms - 1.0)
             / np.where(norm_width.to_numpy(float) > 0.0,
                        norm_width.to_numpy(float), 1.0)) ** 2
            * (norm_width.to_numpy(float) > 0.0))
        for rank in range(1, len(components) + 1):
            logf = origin + np.sum(
                (scores[member, :rank] * scales[:rank]).reshape(-1, 1, 1)
                * components[:rank], axis=0)
            fnp = np.exp(np.clip(logf, -80.0, 80.0))
            raw = (
                np.sum(kernel * fnp[ix1] * fnp[ix2], axis=1) + y_term)
            prediction = row_norms * raw
            data_chi2 = np.sum(((prediction - data) / error) ** 2)
            rows.append({
                "seed": int(seed),
                "rank": rank,
                "data_chi2": float(data_chi2),
                "norm_penalty": float(norm_penalty),
                "unpenalized_total_chi2": float(data_chi2 + norm_penalty),
                "passes_frozen_fit_gate": bool(
                    data_chi2 + norm_penalty <= CEILING),
                "max_prediction_shift_over_sigma_from_source": float(
                    np.max(np.abs(
                        prediction
                        - pd.read_csv(
                            BASE / "outputs"
                            / f"independent_datafit_D020_E772_init{seed}"
                            / "accepted_predictions.csv"
                        ).control_prediction.to_numpy(float)
                    ) / error)),
            })
    table = pd.DataFrame(rows)
    ranks = table.groupby("rank", as_index=False).agg(
        passing_members=("passes_frozen_fit_gate", "sum"),
        maximum_total_chi2=("unpenalized_total_chi2", "max"),
        median_total_chi2=("unpenalized_total_chi2", "median"),
        maximum_prediction_shift_over_sigma=(
            "max_prediction_shift_over_sigma_from_source", "max"),
    )
    eligible = ranks[ranks.passing_members.eq(len(seeds))]
    selected_rank = (
        int(eligible.iloc[0]["rank"]) if len(eligible) else None)
    TARGET.mkdir(parents=True, exist_ok=True)
    table.to_csv(TARGET / "member_rank_coverage.csv", index=False)
    ranks.to_csv(TARGET / "rank_summary.csv", index=False)
    summary = {
        "status": "isolated_empirical_PCA_rank_coverage_not_production",
        "fit_gate": {
            "definition": "data chi2 plus fitted normalization penalty",
            "ceiling": CEILING,
        },
        "member_count": len(seeds),
        "selected_minimum_all-member_fit-admissible_rank": selected_rank,
        "selection_interpretation": (
            "projection fidelity gate only; selected rank must still pass "
            "fresh multistart stationarity and replica coverage"),
        "production_sources_modified": False,
    }
    (TARGET / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(ranks.to_string(index=False))


if __name__ == "__main__":
    main()
