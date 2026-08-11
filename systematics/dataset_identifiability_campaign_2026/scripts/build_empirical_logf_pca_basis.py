#!/usr/bin/env python3
"""Build a medoid-centred log-FNP basis on the exact likelihood grid."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


BASE = Path(__file__).resolve().parents[1]
SYSTEMATICS = BASE.parent
ROOT = SYSTEMATICS.parent
RUNNER_BASE = (
    SYSTEMATICS
    / "high_qt_direct_production_benchmark/experimental_unitary_transition")
TRAINER_PATH = ROOT / "v21_smoothedA_tail_candidate/train_bt_dnn_v21_smoothedA_tail.py"
REFIT_PATH = RUNNER_BASE / "scripts/run_differentiable_fnp_refit.py"
SOURCE = (
    SYSTEMATICS
    / "collins_factorization_validity/outputs/"
    "rowidfix_stageFT_E772_qmax0p20_lam0p50_central_s303")
W_GRID = (
    ROOT / "outputs/v23a_tevatron_plus_lhcb7_fidacc_merged_cache/"
    "backend_cache/wpert_v23a_tevatron_plus_lhcb7_fidacc_b160.csv")
SEEDS = (304, 305, 307, 310, 312, 313, 316, 317, 319, 320, 323, 324, 326)
DISPLAY_X = (0.001, 0.003, 0.01, 0.03, 0.1, 0.2, 0.4, 0.7)
TARGET = BASE / "summaries/empirical_logf_pca_basis"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    trainer = load_module("empirical_basis_trainer", TRAINER_PATH)
    refit = load_module("empirical_basis_refit", REFIT_PATH)
    metrics = json.loads((SOURCE / "metrics.json").read_text())
    accepted = pd.read_csv(SOURCE / "predictions.csv")
    b_np, _ = trainer.load_external_w_grid(
        accepted.row_id.astype(str), W_GRID)
    x_np = np.unique(np.concatenate((
        accepted.x1.to_numpy(float),
        accepted.x2.to_numpy(float),
        np.asarray(DISPLAY_X, dtype=float))))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    x = torch.tensor(x_np, dtype=dtype, device=device)
    b = torch.tensor(b_np, dtype=dtype, device=device)

    curves = []
    for seed in SEEDS:
        run = (
            BASE / "outputs"
            / f"independent_datafit_D020_E772_init{seed}")
        status = json.loads((run / "fit_status.json").read_text())
        if status["final"]["unpenalized_total_chi2"] > 119.8021:
            raise RuntimeError(f"seed {seed} no longer passes the frozen gate")
        model = refit.make_model(
            trainer, metrics["config"], device).to(dtype=dtype)
        saved = torch.load(
            run / "model_state.pt", map_location=device, weights_only=True)
        state = {
            key[len("np_factor."):]: value
            for key, value in saved.items()
            if key.startswith("np_factor.")
        }
        model.load_state_dict(state, strict=True)
        model.eval()
        with torch.no_grad():
            values = model(x, b).clamp_min(1.0e-30)
            curves.append(torch.log(values).cpu().numpy())

    matrix = np.stack(curves)
    active = b_np <= 2.0
    active_flat = matrix[:, :, active].reshape(len(SEEDS), -1)
    robust_center = np.median(active_flat, axis=0)
    medoid_index = int(np.argmin(np.mean(
        (active_flat - robust_center.reshape(1, -1)) ** 2, axis=1)))
    origin = matrix[medoid_index]
    differences = (matrix - origin).reshape(len(SEEDS), -1)
    _, singular, vt = np.linalg.svd(differences, full_matrices=False)
    numerical_rank = int(np.sum(
        singular > np.finfo(float).eps * max(differences.shape) * singular[0]))
    components = vt[:numerical_rank].reshape(
        numerical_rank, len(x_np), len(b_np))
    raw_scores = differences @ vt[:numerical_rank].T
    score_std = np.std(raw_scores, axis=0, ddof=1)
    standardized_scores = raw_scores / score_std.reshape(1, -1)
    variance = singular[:numerical_rank] ** 2
    explained = variance / variance.sum()
    cumulative = np.cumsum(explained)

    TARGET.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        TARGET / "basis.npz",
        x=x_np,
        bT=b_np,
        mean_logf=origin,
        components=components,
        score_std=score_std,
        member_scores=standardized_scores,
        source_seeds=np.asarray(SEEDS, dtype=int),
        medoid_seed=np.asarray(SEEDS[medoid_index], dtype=int),
        explained_variance_ratio=explained,
    )
    pd.DataFrame({
        "rank": np.arange(1, numerical_rank + 1),
        "singular_value": singular[:numerical_rank],
        "score_std": score_std,
        "explained_variance_ratio": explained,
        "cumulative_explained_variance_ratio": cumulative,
    }).to_csv(TARGET / "rank_spectrum.csv", index=False)
    pd.DataFrame(
        standardized_scores,
        index=pd.Index(SEEDS, name="seed"),
        columns=[f"pc{i}" for i in range(1, numerical_rank + 1)],
    ).reset_index().to_csv(TARGET / "member_scores.csv", index=False)
    summary = {
        "status": "isolated_empirical_logF_PCA_basis_not_production",
        "source_definition": (
            "all independent D020_E772 endpoints passing the frozen "
            "unpenalized cross-section fit-quality ceiling"),
        "source_seeds": list(SEEDS),
        "member_count": len(SEEDS),
        "origin": (
            "active-region log-FNP medoid of the admissible ensemble"),
        "medoid_seed": SEEDS[medoid_index],
        "grid": {
            "x_count": len(x_np),
            "bT_count": len(b_np),
            "bT_min": float(b_np[0]),
            "bT_max": float(b_np[-1]),
            "definition": (
                "exact union of likelihood x1/x2 coordinates and display x "
                "knots, on the exact likelihood b grid"),
        },
        "numerical_rank": numerical_rank,
        "ranks_for_90_95_99_percent_difference_energy": {
            str(level): int(np.searchsorted(cumulative, level) + 1)
            for level in (0.90, 0.95, 0.99)
        },
        "selection_rule": (
            "choose the smallest rank preserving cross-section adequacy for "
            "the admissible source manifold and pseudo-data replicas"),
        "production_sources_modified": False,
    }
    (TARGET / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
