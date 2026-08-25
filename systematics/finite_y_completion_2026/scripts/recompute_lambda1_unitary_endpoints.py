#!/usr/bin/env python3
"""Recompute finite-Y W/matched boundary predictions from the lambda=1 endpoints.

The registered lambda=1 production object is an ensemble of endpoint states,
not a single model.  This script evaluates every available endpoint with the
same kernels used by the isolated unitary Tevatron campaign.  It writes only
new diagnostic files under finite_y_completion_2026.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN = ROOT / "systematics/dataset_identifiability_campaign_2026"
LAMBDA1_OUTPUTS = CAMPAIGN / "outputs"
LAMBDA1_ENDPOINT = ROOT / "systematics/dataset_identifiability_campaign_2026/outputs/lambda1_start_expansion96_s353_cont120000"
CONFIG_SOURCE = ROOT / "systematics/collins_factorization_validity/outputs/rowidfix_stageFT_all_qmax0p20_lam0p50_central_s303"
UNITARY_SCRIPT = ROOT / "systematics/high_qt_direct_production_benchmark/experimental_unitary_transition/scripts/run_differentiable_fnp_refit.py"
UNITARY = ROOT / "systematics/high_qt_direct_production_benchmark/experimental_unitary_transition"
BOUNDARY_INPUT = UNITARY / "summaries/unitary_smootherstep_v1_nlo_24row/nlo_frozen_unitary_pulls.csv"
KERNEL_CACHE = UNITARY / "outputs/unitary_smootherstep_v1_differentiable_kernels_nb640_nqt2_ny2"
TARGET = ROOT / "systematics/finite_y_completion_2026/reports/lambda1_unitary_endpoint_recompute"
PROFILES = ("early_0p18_0p28", "central_0p20_0p30", "late_0p22_0p32")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def endpoint_paths() -> list[Path]:
    summary = json.loads((CAMPAIGN / "summaries/lambda1_start_expansion96/summary.json").read_text())
    tags = summary.get("endpoint_tags", [])
    if len(tags) != 96:
        raise RuntimeError(f"registered lambda=1 endpoint count is {len(tags)}, not 96")
    paths = [LAMBDA1_OUTPUTS / tag for tag in tags]
    missing = [str(path) for path in paths if not (path / "model_state.pt").exists()]
    if missing:
        raise RuntimeError(f"missing lambda=1 endpoint states: {missing[:3]}")
    return paths


def main() -> None:
    transform = load_module("unitary_refit_for_lambda1_recompute", UNITARY_SCRIPT)
    trainer = transform.load_trainer()
    metrics = json.loads((CONFIG_SOURCE / "metrics.json").read_text())
    config = metrics["config"]
    accepted = pd.read_csv(LAMBDA1_ENDPOINT / "accepted_predictions.csv")
    kinematics = pd.read_csv(CONFIG_SOURCE / "predictions.csv", usecols=["row_id", "x1", "x2"])
    accepted = accepted.merge(kinematics, on="row_id", how="left", validate="one_to_one")
    if accepted[["x1", "x2"]].isna().any().any():
        raise RuntimeError("lambda=1 accepted endpoint is missing x1/x2 kinematics")
    boundary, b_boundary_np, boundary_kernel_np, boundary_x_np = transform.prepare_boundary(torch.float32)
    b_low_np, low_kernel_np = trainer.load_external_w_grid(accepted.row_id.astype(str), transform.W_GRID)
    low_kernel_np = trainer.precompute_kernel_matrix(
        accepted.qT.to_numpy(), b_low_np, low_kernel_np, dtype=torch.float32
    ).astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    def tensor(values):
        return torch.tensor(np.array(values, copy=True), dtype=dtype, device=device)

    b_low = tensor(b_low_np)
    low_kernel = tensor(low_kernel_np)
    low_x1 = tensor(accepted.x1.to_numpy())
    low_x2 = tensor(accepted.x2.to_numpy())
    b_boundary = tensor(b_boundary_np)
    boundary_kernel = tensor(boundary_kernel_np)
    boundary_x1 = tensor(boundary_x_np[0])
    boundary_x2 = tensor(boundary_x_np[1])
    n_boundary = len(boundary)
    endpoint_rows = []
    accepted_rows = []
    for endpoint in endpoint_paths():
        model = trainer.FilmNPFactor(
            width=int(config["np_width"]), cond_width=int(config["np_cond_width"]),
            n_blocks=int(config["np_blocks"]), a0=float(config["np_a0"]),
            min_a=float(config["np_min_a"]), a_mode=str(config["np_a_mode"]),
            exponent_clip=float(config["fnp_exponent_clip"]),
            shape_mode=str(config["np_shape_mode"]),
            a_smooth_sigma=float(config["np_a_smooth_sigma"]),
            a_tail_amp=float(config["np_a_tail_amp"]),
            a_tail_b0=float(config["np_a_tail_b0"]),
            a_tail_width=float(config["np_a_tail_width"]), dtype=dtype,
        ).to(device)
        state = torch.load(endpoint / "model_state.pt", map_location=device, weights_only=True)
        model.load_state_dict(
            {key.removeprefix("np_factor."): value for key, value in state.items()},
            strict=True,
        )
        model.eval()
        with torch.no_grad():
            low_factors = model(low_x1, b_low) * model(low_x2, b_low)
            w_low = torch.sum(low_kernel * low_factors, dim=1).cpu().numpy()
            high_factors = model(boundary_x1, b_boundary) * model(boundary_x2, b_boundary)
            high_nodes = torch.sum(
                boundary_kernel.reshape(-1, len(b_boundary)) * high_factors, dim=1
            ).cpu().numpy().reshape(n_boundary, 4).sum(axis=1)
        endpoint_id = endpoint.name
        accepted_rows.extend(
            {"endpoint": endpoint_id, "row_id": row_id, "W_lambda1": float(value)}
            for row_id, value in zip(accepted.row_id.astype(str), w_low)
        )
        for profile in PROFILES:
            p = boundary[f"profile_{profile}"].to_numpy(float)
            nlo = boundary.mcfm_nlo_pb_per_GeV.to_numpy(float)
            matched = (1.0 - p) * high_nodes + p * nlo
            for row_id, w_value, p_value, nlo_value, matched_value in zip(
                boundary.row_id.astype(str), high_nodes, p, nlo, matched
            ):
                endpoint_rows.append({
                    "endpoint": endpoint_id,
                    "profile": profile,
                    "row_id": row_id,
                    "W_lambda1": float(w_value),
                    "profile_value": float(p_value),
                    "FO_NLO": float(nlo_value),
                    "matched_lambda1": float(matched_value),
                })
    TARGET.mkdir(parents=True, exist_ok=True)
    accepted_out = pd.DataFrame(accepted_rows)
    boundary_out = pd.DataFrame(endpoint_rows)
    accepted_out.to_csv(TARGET / "accepted_w_lambda1_endpoints.csv", index=False)
    boundary_out.to_csv(TARGET / "boundary_unitary_lambda1_endpoints.csv", index=False)
    old_boundary_w = boundary.w_fitted_pb_per_GeV.to_numpy(float)
    central = boundary_out[boundary_out.profile.eq("central_0p20_0p30")]
    endpoint_summary = central.groupby("endpoint").agg(
        max_abs_w_shift=("W_lambda1", lambda x: float(np.max(np.abs(x.to_numpy() - old_boundary_w)))),
        max_rel_w_shift=("W_lambda1", lambda x: float(np.max(np.abs(x.to_numpy() - old_boundary_w) / np.maximum(np.abs(old_boundary_w), 1e-12)))),
        max_matched=("matched_lambda1", "max"), min_matched=("matched_lambda1", "min"),
    ).reset_index()
    endpoint_summary.to_csv(TARGET / "endpoint_summary.csv", index=False)
    report = {
        "status": "lambda1_unitary_endpoint_recompute_complete",
        "endpoint_count": int(len(endpoint_paths())),
        "endpoint_tags": [p.name for p in endpoint_paths()],
        "device": str(device),
        "boundary_rows": int(len(boundary)),
        "profiles": list(PROFILES),
        "accepted_rows": int(len(accepted)),
        "max_relative_boundary_W_shift_vs_old_lambda0p5_source": float(endpoint_summary.max_rel_w_shift.max()),
        "median_relative_boundary_W_shift_vs_old_lambda0p5_source": float(endpoint_summary.max_rel_w_shift.median()),
        "all_endpoint_predictions_positive": bool((boundary_out.matched_lambda1 > 0).all()),
        "next_step": "Use these lambda=1 W values to rerun the finite-Y fit-impact and replica propagation; do not reuse the superseded lambda=0.5 boundary W column.",
        "production_outputs_modified": False,
    }
    (TARGET / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
