#!/usr/bin/env python3
"""Compare exact-bin resummed W with ASY and an external fixed-order anchor.

This is an isolated diagnostic.  It reads the accepted central model state but
does not load dataset normalization nuisances and does not write into the
accepted production tree.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from systematics.high_qt_direct_production_benchmark.experimental_matched_y.backend.exact_bin_asymptotic import (
    integrate_exact_bin,
    make_resummed_w_point_evaluators,
)
from systematics.high_qt_direct_production_benchmark.experimental_matched_y.scripts.run_asymptotic_pilot import (
    BACKEND_PATH,
    METRICS_PATH,
    load_backend,
    production_cfg,
)


HERE = ROOT / "systematics/high_qt_direct_production_benchmark/experimental_matched_y"
PRODUCTION_RUN = METRICS_PATH.parent
MODEL_STATE_PATH = PRODUCTION_RUN / "model_state.pt"
TRAINER_PATH = ROOT / "v21_smoothedA_tail_candidate/train_bt_dnn_v21_smoothedA_tail.py"
ASY_STATUS_PATH = HERE / "outputs/asymptotic_pilot/cdf_run_2_36/convergence_status.json"


def import_from_path(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_np_factor(config: dict, *, device: torch.device, dtype: torch.dtype):
    trainer = import_from_path(TRAINER_PATH, "_experimental_readonly_smoothed_a_trainer")
    model = trainer.FilmNPFactor(
        width=int(config.get("np_width", 48)),
        cond_width=int(config.get("np_cond_width", 32)),
        n_blocks=int(config.get("np_blocks", 3)),
        a0=float(config.get("np_a0", 0.05)),
        min_a=float(config.get("np_min_a", 0.0)),
        a_mode=str(config.get("np_a_mode", "positive")),
        exponent_clip=float(config.get("fnp_exponent_clip", 40.0)),
        shape_mode=str(config.get("np_shape_mode", "monotone")),
        a_smooth_sigma=float(config.get("np_a_smooth_sigma", 0.0)),
        a_tail_amp=float(config.get("np_a_tail_amp", 0.0)),
        a_tail_b0=float(config.get("np_a_tail_b0", 3.5)),
        a_tail_width=float(config.get("np_a_tail_width", 0.25)),
        dtype=dtype,
    ).to(device)
    state = torch.load(MODEL_STATE_PATH, map_location=device)
    prefix = "np_factor."
    selected = {key[len(prefix):]: value.to(device=device, dtype=dtype)
                for key, value in state.items() if key.startswith(prefix)}
    missing, unexpected = model.load_state_dict(selected, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"NP state mismatch: missing={missing}, unexpected={unexpected}")
    model.eval()
    return model


def make_pair_factor(model, *, device: torch.device, dtype: torch.dtype):
    def evaluate(x1: float, x2: float, b_grid: np.ndarray) -> np.ndarray:
        b = torch.as_tensor(b_grid, dtype=dtype, device=device)
        x = torch.tensor([x1, x2], dtype=dtype, device=device)
        with torch.no_grad():
            factors = model(x, b)
        return (factors[0] * factors[1]).detach().cpu().numpy().astype(float)
    return evaluate


def symmetric_shift(value: float, reference: float) -> float:
    return abs(value - reference) / max(0.5 * (abs(value) + abs(reference)), 1.0e-15)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="CDF_RUN_2")
    ap.add_argument("--row", default="CDF_RUN_2:36")
    ap.add_argument("--n-qt", type=int, default=2)
    ap.add_argument("--n-y", type=int, default=2)
    ap.add_argument("--n-b", type=int, default=320)
    ap.add_argument("--pdf-set", default="NNPDF40_nnlo_as_01180")
    ap.add_argument("--pdf-member", type=int, default=0)
    args = ap.parse_args()

    data_path = ROOT / "Data/v23a_tevatron_plus_lhcb7_fiducial_candidate" / f"{args.dataset}.csv"
    selected = pd.read_csv(data_path).loc[lambda frame: frame.row_id.eq(args.row)]
    if len(selected) != 1:
        raise SystemExit(f"Expected one row {args.row}, found {len(selected)}")
    row = selected.iloc[0]
    if str(row.get("target", "")).lower() == "pp":
        raise SystemExit("pp/fiducial rows require an explicit high-qT acceptance evaluator")

    metrics = json.loads(METRICS_PATH.read_text())
    config = metrics["config"]
    device = torch.device("cpu")
    dtype = torch.float32
    np_factor = load_np_factor(config, device=device, dtype=dtype)
    backend = load_backend()
    cfg = production_cfg(backend, n_b=args.n_b)
    pdf = backend.LHAPDFProvider(args.pdf_set, args.pdf_member, use_toy_pdf=False)
    perturbative, fitted = make_resummed_w_point_evaluators(
        backend=backend, pdf=pdf, cfg=cfg,
        np_pair_factor=make_pair_factor(np_factor, device=device, dtype=dtype),
    )

    started = time.monotonic()
    w_pert = integrate_exact_bin(row, point_evaluator=perturbative, n_qT=args.n_qt, n_y=args.n_y)
    w_fit = integrate_exact_bin(row, point_evaluator=fitted, n_qT=args.n_qt, n_y=args.n_y)
    elapsed = time.monotonic() - started

    asym_status = json.loads(ASY_STATUS_PATH.read_text())
    asym = float(asym_status["richardson_O_h2_asymptotic_pb_per_GeV"])
    fixed_order = float(asym_status["external_fo_average_pb_per_GeV"])
    y_term = fixed_order - asym
    matched_pert = w_pert.value_pb_per_GeV + y_term
    matched_fit = w_fit.value_pb_per_GeV + y_term
    record = {
        "status": "experimental_not_production",
        "dataset": args.dataset,
        "row_id": args.row,
        "source_data": str(data_path.relative_to(ROOT)),
        "backend_read_only": str(BACKEND_PATH.relative_to(ROOT)),
        "production_metrics_read_only": str(METRICS_PATH.relative_to(ROOT)),
        "production_model_state_read_only": str(MODEL_STATE_PATH.relative_to(ROOT)),
        "np_implementation_read_only": str(TRAINER_PATH.relative_to(ROOT)),
        "dataset_normalization_nuisances_applied": False,
        "learned_global_normalization_applied": False,
        "pdf_set": args.pdf_set,
        "pdf_member": args.pdf_member,
        "n_b": args.n_b,
        "n_qT": args.n_qt,
        "n_y": args.n_y,
        "resummed_w_perturbative_pb_per_GeV": w_pert.value_pb_per_GeV,
        "resummed_w_fitted_np_pb_per_GeV": w_fit.value_pb_per_GeV,
        "asymptotic_richardson_pb_per_GeV": asym,
        "external_fixed_order_average_pb_per_GeV": fixed_order,
        "formal_y_fo_minus_asym_pb_per_GeV": y_term,
        "matched_perturbative_pb_per_GeV": matched_pert,
        "matched_fitted_np_pb_per_GeV": matched_fit,
        "w_pert_vs_asym_symmetric_shift": symmetric_shift(w_pert.value_pb_per_GeV, asym),
        "w_fitted_vs_asym_symmetric_shift": symmetric_shift(w_fit.value_pb_per_GeV, asym),
        "matched_pert_vs_fo_symmetric_shift": symmetric_shift(matched_pert, fixed_order),
        "matched_fitted_vs_fo_symmetric_shift": symmetric_shift(matched_fit, fixed_order),
        "all_components_finite": all(math.isfinite(v) for v in (
            w_pert.value_pb_per_GeV, w_fit.value_pb_per_GeV, asym, fixed_order,
            y_term, matched_pert, matched_fit,
        )),
        "elapsed_seconds": elapsed,
        "interpretation": (
            "Component audit only: full resummed W is not required to equal its fixed-order "
            "asymptotic expansion at finite qT/Q; approval thresholds are not assigned by this pilot."
        ),
    }
    out = HERE / "outputs/resummed_w_cancellation_pilot" / args.row.replace(":", "_").lower()
    out.mkdir(parents=True, exist_ok=True)
    name = f"result_nb{args.n_b}_nqt{args.n_qt}_ny{args.n_y}.json"
    (out / name).write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
