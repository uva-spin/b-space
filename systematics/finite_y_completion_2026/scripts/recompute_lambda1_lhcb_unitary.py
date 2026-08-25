#!/usr/bin/env python3
"""Recompute lambda=1 LHCb fiducial W and unitary endpoint predictions."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.integrate import simpson
from scipy.special import j0
import torch


ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN = ROOT / "systematics/dataset_identifiability_campaign_2026"
ENDPOINT_SUMMARY = CAMPAIGN / "summaries/lambda1_start_expansion96/summary.json"
ENDPOINTS = CAMPAIGN / "outputs"
TRAINER = ROOT / "v21_smoothedA_tail_candidate/train_bt_dnn_v21_smoothedA_tail.py"
ARCH_CONFIG = ROOT / "systematics/collins_factorization_validity/outputs/rowidfix_stageFT_all_qmax0p20_lam0p50_central_s303/metrics.json"
DATA = ROOT / "Data/v23a_tevatron_plus_lhcb7_fiducial_candidate/LHCb_7.csv"
KERNELS = ROOT / "systematics/finite_y_completion_2026/reports/lhcb_fiducial_w_kernels_nb640/rows"
DY_TRUE = ROOT / "systematics/finite_y_completion_2026/reports/lhcb7_external_true_nlo/dyturbo_true_nlo_summary.csv"
OUT = ROOT / "systematics/finite_y_completion_2026/reports/lambda1_lhcb_unitary"
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
    tags = json.loads(ENDPOINT_SUMMARY.read_text())["endpoint_tags"]
    paths = [ENDPOINTS / tag for tag in tags]
    missing = [str(path) for path in paths if not (path / "model_state.pt").exists()]
    if missing:
        raise RuntimeError(f"missing endpoint states: {missing[:3]}")
    return paths


def load_fo() -> pd.DataFrame:
    # Use the isolated cards with doVJREAL/doVJVIRT enabled for all four
    # boundary rows.  The earlier summaries are retained as provenance but
    # are not used here because their cards evaluated only the V+jet LO term.
    fo = pd.read_csv(DY_TRUE)
    # DYTurbo's text tables report fb/bin.  The data and internal W kernels
    # are in pb/GeV, so convert the table values before forming the unitary
    # transition.  (The fiducial/inclusive acceptance ratios are unitless and
    # were unaffected by this conversion.)
    return pd.DataFrame({
        "row_id": fo["row_id"],
        "FO_DYTurbo": fo["dyturbo_pb_per_GeV"],
        "FO_DYTurbo_unc": fo["dyturbo_pb_per_GeV_unc"],
    })


def main() -> None:
    transform = load_module("lhcb_lambda1_transform", TRAINER)
    config = json.loads(ARCH_CONFIG.read_text())["config"]
    data = pd.read_csv(DATA)
    rows = data[data.row_id.isin(["LHCb_7:10", "LHCb_7:11", "LHCb_7:12", "LHCb_7:13"])].copy()
    fo = load_fo()
    rows = rows.merge(fo, on="row_id", how="left", validate="one_to_one")
    if rows.FO_DYTurbo.isna().any():
        raise RuntimeError("missing DYTurbo FO rows")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    models = []
    for endpoint in endpoint_paths():
        model = transform.FilmNPFactor(
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
        model.load_state_dict({key.removeprefix("np_factor."): value for key, value in state.items()}, strict=True)
        model.eval()
        models.append((endpoint.name, model))
    b_rows, w_rows = [], []
    for _, row in rows.iterrows():
        with np.load(KERNELS / f"{row.row_id.lower().replace(':', '_')}.npz") as saved:
            b = saved["b_grid"].astype(float)
            sw = simpson(np.eye(len(b)), x=b, axis=1)
            q_nodes, y_nodes = saved["qT_nodes"], saved["y_nodes"]
            q_weights, y_weights = saved["qT_weights"], saved["y_weights"]
            kernel = saved["kernels"].astype(np.float32)
            x1 = saved["x1_nodes"].astype(np.float32)
            x2 = saved["x2_nodes"].astype(np.float32)
            q_half = 0.5 * (float(row.qT_high) - float(row.qT_low))
            y_half = 0.5 * (float(row.y_High) - float(row.y_Low))
            coefficients = q_half * y_half / (float(row.qT_high) - float(row.qT_low))
            b_t = torch.tensor(b, dtype=dtype, device=device)
            x1_t = torch.tensor(x1.reshape(-1), dtype=dtype, device=device)
            x2_t = torch.tensor(x2.reshape(-1), dtype=dtype, device=device)
            basis = torch.tensor(
                np.asarray([coefficients * q_weights[iq] * y_weights[iy] * sw * b * j0(float(q_nodes[iq]) * b)
                            for iq in range(len(q_nodes)) for iy in range(len(y_nodes))], dtype=np.float32),
                dtype=dtype, device=device,
            )
            kernel_t = torch.tensor(kernel.reshape(-1, len(b)), dtype=dtype, device=device)
            for endpoint_id, model in models:
                with torch.no_grad():
                    factors = model(x1_t, b_t) * model(x2_t, b_t)
                    w_value = float(torch.sum(basis * kernel_t * factors).cpu())
                w_rows.append({"endpoint": endpoint_id, "row_id": row.row_id,
                               "W_lambda1_fiducial": w_value})
    w_frame = pd.DataFrame(w_rows)
    out_rows = []
    for _, row in rows.iterrows():
        for profile in PROFILES:
            p = transform.smootherstep_profile if hasattr(transform, "smootherstep_profile") else None
            # Keep the profile implementation local and explicit.
            start, end = {"early_0p18_0p28": (0.18, 0.28), "central_0p20_0p30": (0.20, 0.30), "late_0p22_0p32": (0.22, 0.32)}[profile]
            nodes, weights = np.polynomial.legendre.leggauss(32)
            half = 0.5 * (row.qT_high - row.qT_low)
            q = 0.5 * (row.qT_high + row.qT_low) + half * nodes
            t = np.clip((q / row.QM - start) / (end - start), 0.0, 1.0)
            prof = float(half * np.sum(weights * t**3 * (t * (6 * t - 15) + 10)) / (row.qT_high - row.qT_low))
            for endpoint in w_frame[w_frame.row_id.eq(row.row_id)].itertuples(index=False):
                matched = (1.0 - prof) * endpoint.W_lambda1_fiducial + prof * row.FO_DYTurbo
                out_rows.append({"endpoint": endpoint.endpoint, "row_id": row.row_id,
                                 "profile": profile, "qT": row.qT,
                                 "W_lambda1_fiducial": endpoint.W_lambda1_fiducial,
                                 "FO_DYTurbo": row.FO_DYTurbo,
                                 "FO_DYTurbo_unc": row.FO_DYTurbo_unc,
                                 "profile_value": prof, "matched_lambda1": matched,
                                 "CS": row.CS, "error": row.error})
    OUT.mkdir(parents=True, exist_ok=True)
    w_frame.to_csv(OUT / "lhcb_w_lambda1_endpoints.csv", index=False)
    result = pd.DataFrame(out_rows)
    result.to_csv(OUT / "lhcb_unitary_lambda1_endpoints.csv", index=False)
    report = {
        "status": "lambda1_lhcb_unitary_endpoint_recompute_complete",
        "endpoint_count": int(len(models)), "row_count": int(len(rows)),
        "profiles": list(PROFILES), "device": str(device),
        "all_matched_positive": bool((result.matched_lambda1 > 0).all()),
        "matched_min": float(result.matched_lambda1.min()),
        "matched_max": float(result.matched_lambda1.max()),
        "fo_input_units": "DYTurbo text-table fb/bin converted to pb/GeV before matching",
        "fo_input_source": str(DY_TRUE),
        "production_outputs_modified": False,
        "next_step": "Fit the four LHCb rows with the lambda=1 endpoint ensemble and explicit correlated nuisance model; retain the Tevatron scope audit separately.",
    }
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
