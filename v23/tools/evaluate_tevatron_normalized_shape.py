#!/usr/bin/env python3
"""Evaluate D0 normalized Tevatron spectra as shape-only observables.

This is a diagnostic bridge before adding normalized-observable rows to the
trainer.  It reuses the v21 training/back-end code to build predictions from a
saved central b-space model, then compares data and theory after normalizing
each dataset over the same selected qT/Q range.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


def import_train_module(path: Path):
    spec = importlib.util.spec_from_file_location("bt_train_v21", str(path.resolve()))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import trainer from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bt_train_v21"] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def cfg_value(cfg: dict[str, Any], key: str, default: Any) -> Any:
    val = cfg.get(key, default)
    return default if val is None else val


def add_flag(args: list[str], name: str, value: Any) -> None:
    if isinstance(value, bool):
        if value:
            args.append(name)
        return
    args.extend([name, str(value)])


def build_train_args(ns: argparse.Namespace, cfg: dict[str, Any]) -> list[str]:
    args: list[str] = [
        "--backend-script", str(ns.backend_script),
        "--data-dir", str(ns.data_dir),
        "--datasets", *ns.datasets,
        "--mode", "matched",
        "--qT-max-over-Q", str(ns.max_qt_over_q),
        "--tmd-qT-max-over-Q", str(ns.max_qt_over_q),
        "--w-backend", "internal_css",
        "--out", str(ns.out),
        "--check-only",
    ]

    scalar_defaults = {
        "pdf_set": "NNPDF40_nnlo_as_01180",
        "pdf_member": 0,
        "resum_order": "n3llp",
        "match_order": "nlo",
        "nlo_singular_mode": "asymptotic_damped",
        "nlo_singular_rsub": 0.10,
        "nlo_singular_power": 2,
        "nlo_singular_damp_kind": "exp",
        "nlo_real_quad": 96,
        "nlo_real_tail_repair": "mcfm_logistic",
        "nlo_real_tail_r0": 0.530,
        "nlo_real_tail_width": 0.008,
        "nlo_real_tail_rinf": 0.180,
        "target_mode": "nuclear_isospin",
        "prefactor_scheme": "oldA_to_CS",
        "y_mode": "zero",
        "n_b": 160,
        "b_min": 1.0e-4,
        "b_max": 8,
        "b_star_max": 1.5,
        "mu_min": 1.3,
        "n_sudakov_quad": 32,
        "q0": 2.0,
        "dtype": "float32",
        "device": ns.device,
        "num_threads": ns.num_threads,
        "np_width": 48,
        "np_cond_width": 32,
        "np_blocks": 3,
        "np_a0": 0.05,
        "np_min_a": 0,
        "np_a_mode": "positive",
        "np_shape_mode": "monotone",
        "fnp_exponent_clip": 40,
        "np_a_smooth_sigma": 0.45,
        "np_a_tail_amp": 0.08,
        "np_a_tail_b0": 3.5,
        "np_a_tail_width": 0.25,
        "soft_q_evolution": "none",
        "norm_source": "csv",
        "ptp_source": "csv",
    }
    for key, default in scalar_defaults.items():
        add_flag(args, "--" + key.replace("_", "-"), cfg_value(cfg, key, default))
    return args


def load_model_state(model: torch.nn.Module, state_path: Path, device: torch.device) -> None:
    state = torch.load(state_path, map_location=device)
    if isinstance(state, dict) and "model_state" in state and isinstance(state["model_state"], dict):
        state = state["model_state"]
    own = model.state_dict()
    filtered: dict[str, Any] = {}
    skipped: list[tuple[str, str]] = []
    for key, value in state.items():
        if key in {"b", "kernel_matrix"}:
            skipped.append((key, "row-dependent buffer"))
            continue
        if key not in own:
            skipped.append((key, "not in audit model"))
            continue
        if tuple(own[key].shape) != tuple(value.shape):
            skipped.append((key, f"shape {tuple(value.shape)} != {tuple(own[key].shape)}"))
            continue
        filtered[key] = value.to(device=own[key].device, dtype=own[key].dtype) if torch.is_tensor(value) else value
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"loaded {len(filtered)} tensors from {state_path}")
    if skipped:
        print("skipped state entries:", skipped[:8])
    print("missing keys:", len(missing), "unexpected keys:", len(unexpected))


def normalized_metrics(pred_df: pd.DataFrame, cuts: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    pred_rows: list[pd.DataFrame] = []
    df = pred_df.copy()
    df["qT_bin_width"] = pd.to_numeric(df["qT_bin_width"], errors="coerce")
    df = df[np.isfinite(df["qT_bin_width"]) & (df["qT_bin_width"] > 0)].copy()

    for cut in cuts:
        cut_df = df[df["qT_over_Q"].astype(float) <= float(cut)].copy()
        for dataset, sub in cut_df.groupby("dataset", sort=False):
            sub = sub.copy()
            data_area = float(np.sum(sub["target_used"].to_numpy(float) * sub["qT_bin_width"].to_numpy(float)))
            pred_area = float(np.sum(sub["pred_match_CS"].to_numpy(float) * sub["qT_bin_width"].to_numpy(float)))
            if data_area <= 0 or pred_area <= 0 or len(sub) < 2:
                continue
            sub["shape_cut_qT_over_Q"] = float(cut)
            sub["data_norm_per_GeV"] = sub["target_used"].to_numpy(float) / data_area
            sub["pred_norm_per_GeV"] = sub["pred_match_CS"].to_numpy(float) / pred_area
            sub["sigma_norm_per_GeV_diag"] = sub["sigma_used"].to_numpy(float) / data_area
            sub["pull_norm_diag"] = (
                sub["pred_norm_per_GeV"].to_numpy(float) - sub["data_norm_per_GeV"].to_numpy(float)
            ) / np.maximum(sub["sigma_norm_per_GeV_diag"].to_numpy(float), 1.0e-30)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = sub["pred_norm_per_GeV"].to_numpy(float) / sub["data_norm_per_GeV"].to_numpy(float)
            pulls = sub["pull_norm_diag"].to_numpy(float)
            rows.append({
                "dataset": str(dataset),
                "shape_cut_qT_over_Q": float(cut),
                "n": int(len(sub)),
                "chi2_like_diag_shape": float(np.mean(pulls ** 2)),
                "median_abs_pull_diag_shape": float(np.median(np.abs(pulls))),
                "median_pred_over_data_shape": float(np.nanmedian(ratio)),
                "data_area_in_cut": data_area,
                "pred_area_in_cut": pred_area,
                "pred_area_over_data_area": pred_area / data_area,
            })
            pred_rows.append(sub)
    return (
        pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame(),
        pd.DataFrame(rows),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainer", default="v21_tail_release_amp0p019_candidate/train_bt_dnn_v21_smoothedA_tail.py")
    ap.add_argument("--backend-script", default="v23/backends/bt_internal_css_backend_v22_tevatron.py")
    ap.add_argument("--data-dir", default="Data/v23_tevatron_reviewed_diagnostic")
    ap.add_argument("--datasets", nargs="+", default=["D0_RUN_2", "D0_RUN_2N"])
    ap.add_argument("--model-state", default="outputs/v23a_tevatron_lowqt010_allchecked_tailpass_central_s303/model_state.pt")
    ap.add_argument("--fit-metrics", default="outputs/v23a_tevatron_lowqt010_allchecked_tailpass_central_s303/metrics.json")
    ap.add_argument("--out", default="outputs/v23a_d0_run2_normalized_shape_audit")
    ap.add_argument("--max-qt-over-q", type=float, default=0.5)
    ap.add_argument("--shape-cuts", type=float, nargs="+", default=[0.10, 0.15, 0.20, 0.50])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--num-threads", type=int, default=4)
    ns = ap.parse_args()

    out = Path(ns.out)
    if out.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {out}")
    out.mkdir(parents=True)

    metrics_path = Path(ns.fit_metrics)
    cfg: dict[str, Any] = {}
    if metrics_path.exists():
        cfg = json.loads(metrics_path.read_text()).get("config", {})

    train = import_train_module(Path(ns.trainer))
    train_args = train.build_argparser().parse_args(build_train_args(ns, cfg))
    if train_args.num_threads and train_args.num_threads > 0:
        torch.set_num_threads(int(train_args.num_threads))
    dtype = train.dtype_from_string(train_args.dtype)
    device = torch.device(ns.device)

    backend, backend_path = train.import_backend(train_args.backend_script)
    train_args.backend_script = str(backend_path)
    print("using backend:", backend_path)
    df_raw, b_grid, kernel, y_term, backend_meta = train.prepare_data_and_backend(train_args, backend, dtype)
    rep_cfg = train.ReplicaConfig(observable="CS", error_column="error", norm_source=train_args.norm_source, ptp_source=train_args.ptp_source)
    df = train.build_uncertainties(df_raw, rep_cfg)

    np_factor = train.FilmNPFactor(
        width=int(train_args.np_width),
        cond_width=int(train_args.np_cond_width),
        n_blocks=int(train_args.np_blocks),
        a0=float(train_args.np_a0),
        min_a=float(train_args.np_min_a),
        a_mode=str(train_args.np_a_mode),
        exponent_clip=float(train_args.fnp_exponent_clip),
        shape_mode=str(train_args.np_shape_mode),
        a_smooth_sigma=float(train_args.np_a_smooth_sigma),
        a_tail_amp=float(train_args.np_a_tail_amp),
        a_tail_b0=float(train_args.np_a_tail_b0),
        a_tail_width=float(train_args.np_a_tail_width),
        dtype=dtype,
    ).to(device)
    model = train.PrecomputedKernelModel(
        b_grid=b_grid,
        kernel_matrix=kernel,
        np_factor=np_factor,
        gk_model=train.ZeroGK().to(device),
        q0=float(train_args.q0),
        cs_log=str(train_args.cs_log),
        cs_kernel_convention=str(train_args.cs_kernel_convention),
        learn_global_norm=bool(train_args.learn_global_norm),
        global_norm_init=float(train_args.global_norm_init),
        dtype=dtype,
        device=device,
    ).to(device)
    load_model_state(model, Path(ns.model_state), device)

    data = train.TensorData(df, y_term=y_term, target=None, dtype=dtype, device=device)
    model.eval()
    with torch.no_grad():
        pred_w = model.sigma_w(data.row_index, data.Q, data.x1, data.x2)
        pred = pred_w + data.y_term
    pred_df = data.df.copy()
    pred_df["target_used"] = data.target.detach().cpu().numpy()
    pred_df["sigma_used"] = data.sigma.detach().cpu().numpy()
    pred_df["Y_CS_used"] = data.y_term.detach().cpu().numpy()
    pred_df["pred_W_CS"] = pred_w.detach().cpu().numpy()
    pred_df["pred_match_CS"] = pred.detach().cpu().numpy()
    pred_df["pull_absolute_diag"] = (pred_df["pred_match_CS"] - pred_df["target_used"]) / pred_df["sigma_used"]
    pred_df.to_csv(out / "absolute_predictions_for_shape_audit.csv", index=False)

    shape_pred, shape_metrics = normalized_metrics(pred_df, sorted(set(float(x) for x in ns.shape_cuts)))
    shape_pred.to_csv(out / "normalized_shape_predictions.csv", index=False)
    shape_metrics.to_csv(out / "normalized_shape_metrics.csv", index=False)

    summary = {
        "note": (
            "Shape metrics normalize data and theory over the same qT/Q cut. "
            "Uncertainties are diagonal approximations inherited from the reviewed tables; "
            "normalization-induced correlations are not included."
        ),
        "model_state": str(Path(ns.model_state)),
        "fit_metrics": str(metrics_path),
        "backend_meta": backend_meta,
        "datasets": list(ns.datasets),
        "shape_cuts": sorted(set(float(x) for x in ns.shape_cuts)),
        "metrics": shape_metrics.to_dict(orient="records"),
    }
    (out / "normalized_shape_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(shape_metrics.to_string(index=False))
    print("wrote", out)


if __name__ == "__main__":
    main()
