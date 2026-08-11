#!/usr/bin/env python3
"""Experimental joint FNP refit of accepted and 24 unitary-boundary rows."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.integrate import simpson
from scipy.optimize import least_squares
from scipy.special import j0
import torch


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "systematics/high_qt_direct_production_benchmark/experimental_unitary_transition"
PRODUCTION = ROOT / "systematics/collins_factorization_validity/outputs/rowidfix_stageFT_all_qmax0p20_lam0p50_central_s303"
TRAINER_PATH = ROOT / "v21_smoothedA_tail_candidate/train_bt_dnn_v21_smoothedA_tail.py"
W_GRID = ROOT / "outputs/v23a_tevatron_plus_lhcb7_fidacc_merged_cache/backend_cache/wpert_v23a_tevatron_plus_lhcb7_fidacc_b160.csv"
BOUNDARY_INPUT = BASE / "summaries/unitary_smootherstep_v1_nlo_24row/nlo_frozen_unitary_pulls.csv"
KERNEL_CACHE = BASE / "outputs/unitary_smootherstep_v1_differentiable_kernels_nb640_nqt2_ny2"
PROFILE = "central_0p20_0p30"
UPWARD_NLO_SCALE = 0.19217428727157315
LEARNING_RATE = 2.0e-5
FNP_ANCHOR_X = [0.001, 0.003, 0.01, 0.03, 0.1, 0.2, 0.4, 0.7]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=303)
    parser.add_argument("--initial-perturbation", type=float, default=0.0,
                        help="Relative Gaussian parameter perturbation for local-basin tests")
    parser.add_argument("--max-epochs", type=int, default=20000)
    parser.add_argument("--min-epochs", type=int, default=5000)
    parser.add_argument("--plateau-patience", type=int, default=1500)
    parser.add_argument("--min-delta", type=float, default=1.0e-7,
                        help="Required improvement in total chi2 per row")
    parser.add_argument("--fnp-anchor-strength", type=float, default=0.0,
                        help="Coefficient of mean squared log-FNP displacement from the accepted state")
    parser.add_argument("--fnp-anchor-bmax", type=float, default=8.0)
    parser.add_argument("--fnp-anchor-nb", type=int, default=161)
    parser.add_argument("--initial-state-tag", help="Experimental output tag used to initialize model and nuisances")
    parser.add_argument("--learning-rate-stages", default="",
                        help="Comma-separated epoch:rate schedule, e.g. 0:2e-6,3000:2e-7")
    parser.add_argument("--lbfgs-max-iter", type=int, default=0)
    parser.add_argument("--stationarity-gradient-threshold", type=float, default=1.0e-4)
    parser.add_argument("--freeze-fnp", action="store_true",
                        help="Keep the initialized FNP fixed and polish nuisance parameters only")
    parser.add_argument("--scipy-nuisance-polish", action="store_true",
                        help="Finish a frozen-FNP fit with double-precision nonlinear least squares")
    parser.add_argument("--tag", default="unitary_smootherstep_v1_differentiable_fnp_refit_central_converged_s303")
    return parser.parse_args()


def load_trainer():
    spec = importlib.util.spec_from_file_location("accepted_fnp_trainer", TRAINER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load trainer at {TRAINER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def safe_tag(row_id: str) -> str:
    return row_id.lower().replace(":", "_")


def make_model(trainer, config: dict, device: torch.device):
    model = trainer.FilmNPFactor(
        width=int(config["np_width"]), cond_width=int(config["np_cond_width"]),
        n_blocks=int(config["np_blocks"]), a0=float(config["np_a0"]),
        min_a=float(config["np_min_a"]), a_mode=str(config["np_a_mode"]),
        exponent_clip=float(config["fnp_exponent_clip"]),
        shape_mode=str(config["np_shape_mode"]),
        a_smooth_sigma=float(config["np_a_smooth_sigma"]),
        a_tail_amp=float(config["np_a_tail_amp"]),
        a_tail_b0=float(config["np_a_tail_b0"]),
        a_tail_width=float(config["np_a_tail_width"]), dtype=torch.float32,
    ).to(device)
    state = torch.load(PRODUCTION / "model_state.pt", map_location=device, weights_only=True)
    np_state = {k.removeprefix("np_factor."): v for k, v in state.items() if k.startswith("np_factor.")}
    model.load_state_dict(np_state, strict=True)
    return model


def prepare_boundary(dtype: torch.dtype) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.read_csv(BOUNDARY_INPUT)
    all_kernels, x1_nodes, x2_nodes = [], [], []
    reference_b = None
    for row_id in frame.row_id.astype(str):
        with np.load(KERNEL_CACHE / "rows" / f"{safe_tag(row_id)}.npz") as saved:
            b = saved["b_grid"].astype(float)
            if reference_b is None:
                reference_b = b
            elif not np.array_equal(reference_b, b):
                raise ValueError("Boundary bundles do not share an identical b grid")
            # Obtain the exact linear weights used by scipy.integrate.simpson.
            sw = simpson(np.eye(len(b)), x=b, axis=1)
            node_kernels = []
            width = float(saved["qT_high"] - saved["qT_low"])
            for iq, qt in enumerate(saved["qt_nodes"]):
                for iy in range(len(saved["y_weights"])):
                    coefficient = float(saved["qt_weights"][iq] * saved["y_weights"][iy]) / width
                    node_kernels.append(
                        coefficient * sw * b * j0(float(qt) * b) * saved["kernels"][iq, iy]
                    )
                    x1_nodes.append(float(saved["x1_nodes"][iq, iy]))
                    x2_nodes.append(float(saved["x2_nodes"][iq, iy]))
            all_kernels.append(node_kernels)
    assert reference_b is not None
    return frame, reference_b, np.asarray(all_kernels, dtype=np.float32), np.asarray([x1_nodes, x2_nodes], dtype=np.float32)


def main() -> None:
    args = parse_args()
    if args.fnp_anchor_strength < 0.0:
        raise ValueError("--fnp-anchor-strength must be nonnegative")
    if args.fnp_anchor_bmax <= 0.0 or args.fnp_anchor_nb < 2:
        raise ValueError("The FNP anchor grid requires bmax > 0 and nb >= 2")
    if args.scipy_nuisance_polish and not args.freeze_fnp:
        raise ValueError("--scipy-nuisance-polish requires --freeze-fnp")
    lr_stages = []
    if args.learning_rate_stages:
        lr_stages = [(int(epoch), float(rate)) for epoch, rate in
                     (item.split(":", 1) for item in args.learning_rate_stages.split(","))]
        if not lr_stages or lr_stages[0][0] != 0 or any(rate <= 0 for _, rate in lr_stages):
            raise ValueError("Learning-rate stages must start at epoch zero and have positive rates")
        if [epoch for epoch, _ in lr_stages] != sorted(set(epoch for epoch, _ in lr_stages)):
            raise ValueError("Learning-rate stage epochs must be unique and increasing")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    target = BASE / "outputs" / args.tag
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    trainer = load_trainer()
    metrics = json.loads((PRODUCTION / "metrics.json").read_text())
    config = metrics["config"]
    accepted = pd.read_csv(PRODUCTION / "predictions.csv")
    boundary, b_boundary_np, boundary_kernel_np, boundary_x_np = prepare_boundary(dtype)

    b_low_np, w_matrix = trainer.load_external_w_grid(accepted.row_id.astype(str), W_GRID)
    low_kernel_np = trainer.precompute_kernel_matrix(
        accepted.qT.to_numpy(), b_low_np, w_matrix, dtype=dtype
    ).astype(np.float32)

    datasets = list(dict.fromkeys(accepted.dataset.astype(str)))
    dataset_to_i = {name: i for i, name in enumerate(datasets)}
    if not set(boundary.dataset.astype(str)).issubset(dataset_to_i):
        raise ValueError("Boundary data contain a dataset absent from the accepted fit")
    low_di = torch.tensor(accepted.dataset.map(dataset_to_i).to_numpy(), dtype=torch.long, device=device)
    high_di = torch.tensor(boundary.dataset.map(dataset_to_i).to_numpy(), dtype=torch.long, device=device)
    norm_width = accepted.groupby("dataset").norm_rel_used.first().reindex(datasets).to_numpy(float)
    norm_start = accepted.groupby("dataset").dataset_norm_factor.first().reindex(datasets).to_numpy(float)
    free_norm = norm_width > 0.0

    def tensor(values):
        return torch.tensor(np.array(values, copy=True), dtype=dtype, device=device)

    b_low, low_kernel = tensor(b_low_np), tensor(low_kernel_np)
    low_x1, low_x2 = tensor(accepted.x1.to_numpy()), tensor(accepted.x2.to_numpy())
    low_y = tensor(accepted.Y_CS_used.to_numpy())
    low_data, low_error = tensor(accepted.target_used.to_numpy()), tensor(accepted.sigma_used.to_numpy())
    b_boundary, boundary_kernel = tensor(b_boundary_np), tensor(boundary_kernel_np)
    boundary_x1, boundary_x2 = tensor(boundary_x_np[0]), tensor(boundary_x_np[1])
    high_data, high_error = tensor(boundary.CS.to_numpy()), tensor(boundary.error.to_numpy())
    p = tensor(boundary[f"profile_{PROFILE}"].to_numpy())
    nlo = tensor(boundary.mcfm_nlo_pb_per_GeV.to_numpy())
    norm_width_t = tensor(np.where(free_norm, norm_width, 1.0))
    free_norm_t = tensor(free_norm.astype(np.float32))

    model = make_model(trainer, config, device)
    anchor_x = tensor(FNP_ANCHOR_X)
    anchor_b = torch.linspace(0.0001, args.fnp_anchor_bmax, args.fnp_anchor_nb, dtype=dtype, device=device)
    with torch.no_grad():
        accepted_anchor_log_fnp = torch.log(model(anchor_x, anchor_b).clamp_min(1.0e-12)).detach()
    initial_nuisance_state = None
    if args.initial_state_tag:
        source = BASE / "outputs" / args.initial_state_tag
        state = torch.load(source / "model_state.pt", map_location=device, weights_only=True)
        model.load_state_dict({k.removeprefix("np_factor."): v for k, v in state.items()}, strict=True)
        initial_nuisance_state = torch.load(source / "nuisance_state.pt", map_location=device, weights_only=True)
    if args.initial_perturbation > 0.0:
        with torch.no_grad():
            for parameter in model.parameters():
                scale = torch.sqrt(torch.mean(parameter.square())).clamp(min=1.0e-6)
                parameter.add_(torch.randn_like(parameter) * scale * args.initial_perturbation)
    initial_np_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    nuisance_dtype = torch.float64 if args.freeze_fnp else dtype
    def nuisance_tensor(values):
        return torch.tensor(np.array(values, copy=True), dtype=nuisance_dtype, device=device)

    log_norms = torch.nn.Parameter(nuisance_tensor(np.where(free_norm, np.log(norm_start), 0.0)))
    eta_match = torch.nn.Parameter(nuisance_tensor(1.3505))
    eta_scale = torch.nn.Parameter(nuisance_tensor(0.9436))
    if initial_nuisance_state is not None:
        with torch.no_grad():
            log_norms.copy_(initial_nuisance_state["log_norms"])
            eta_match.copy_(initial_nuisance_state["eta_match"])
            eta_scale.copy_(initial_nuisance_state["eta_scale"])
    fnp_parameters = list(model.parameters())
    nuisance_parameters = [log_norms, eta_match, eta_scale]
    if args.freeze_fnp:
        for parameter in fnp_parameters:
            parameter.requires_grad_(False)
    parameters = nuisance_parameters if args.freeze_fnp else fnp_parameters + nuisance_parameters
    initial_lr = lr_stages[0][1] if lr_stages else LEARNING_RATE
    optimizer = torch.optim.AdamW(parameters, lr=initial_lr, weight_decay=0.0)
    scheduler = None if lr_stages else torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.3, patience=500, threshold=args.min_delta,
        threshold_mode="abs", min_lr=2.0e-7,
    )
    n_rows = len(accepted) + len(boundary)
    fixed_w_low = fixed_w_high = fixed_anchor_penalty = None
    if args.freeze_fnp:
        with torch.no_grad():
            fixed_factors_low = model(low_x1, b_low) * model(low_x2, b_low)
            fixed_w_low = torch.sum(low_kernel * fixed_factors_low, dim=1).to(torch.float64)
            fixed_factors_high = model(boundary_x1, b_boundary) * model(boundary_x2, b_boundary)
            fixed_node_values = torch.sum(
                boundary_kernel.reshape(-1, len(b_boundary)) * fixed_factors_high, dim=1)
            fixed_w_high = torch.sum(fixed_node_values.reshape(len(boundary), 4), dim=1).to(torch.float64)
            fixed_anchor_log = torch.log(model(anchor_x, anchor_b).clamp_min(1.0e-12))
            fixed_anchor_penalty = (
                args.fnp_anchor_strength * torch.mean((fixed_anchor_log - accepted_anchor_log_fnp).square())
            ).to(torch.float64)

    def evaluate():
        if fixed_w_low is None:
            factors_low = model(low_x1, b_low) * model(low_x2, b_low)
            w_low = torch.sum(low_kernel * factors_low, dim=1)
        else:
            w_low = fixed_w_low
        # A dataset with zero quoted normalization uncertainty is fixed exactly.
        norms = torch.exp(log_norms * free_norm_t)
        pred_low = norms[low_di] * (w_low + low_y)
        if fixed_w_high is None:
            factors_high = model(boundary_x1, b_boundary) * model(boundary_x2, b_boundary)
            node_values = torch.sum(boundary_kernel.reshape(-1, len(b_boundary)) * factors_high, dim=1)
            w_high = torch.sum(node_values.reshape(len(boundary), 4), dim=1)
        else:
            w_high = fixed_w_high
        base = (1.0 - p) * w_high + p * nlo
        matching = (1.0 - p) * (nlo - w_high)
        scale = p * nlo * UPWARD_NLO_SCALE
        pred_high = norms[high_di] * (base + eta_match * matching + eta_scale * scale)
        low_chi2 = torch.sum(((pred_low - low_data) / low_error).square())
        high_chi2 = torch.sum(((pred_high - high_data) / high_error).square())
        norm_penalty = torch.sum((free_norm_t * (norms - 1.0) / norm_width_t).square())
        theory_penalty = eta_match.square() + eta_scale.square()
        if fixed_anchor_penalty is None:
            anchor_log_fnp = torch.log(model(anchor_x, anchor_b).clamp_min(1.0e-12))
            fnp_anchor_penalty = args.fnp_anchor_strength * torch.mean(
                (anchor_log_fnp - accepted_anchor_log_fnp).square())
        else:
            fnp_anchor_penalty = fixed_anchor_penalty
        total = low_chi2 + high_chi2 + norm_penalty + theory_penalty + fnp_anchor_penalty
        return total, low_chi2, high_chi2, norm_penalty, theory_penalty, fnp_anchor_penalty, pred_low, pred_high, w_high

    with torch.no_grad():
        initial_values = evaluate()
        initial_predictions = [x.detach().cpu().numpy().copy() for x in initial_values[6:8]]
        initial_metrics = [float(x) for x in initial_values[:6]]

    best_loss = float("inf")
    best = None
    last_material_improvement = 0
    history = []
    stopped_early = False
    for epoch in range(args.max_epochs + 1):
        if lr_stages:
            active_lr = next(rate for start, rate in reversed(lr_stages) if epoch >= start)
            for group in optimizer.param_groups:
                group["lr"] = active_lr
        optimizer.zero_grad(set_to_none=True)
        values = evaluate()
        loss = values[0] / n_rows
        if epoch > 0:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, float(config["grad_clip"]))
            optimizer.step()
        current = float(loss.detach())
        if scheduler is not None:
            scheduler.step(current)
        if current < best_loss - args.min_delta:
            best_loss = current
            last_material_improvement = epoch
            best = {
                "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "log_norms": log_norms.detach().cpu().clone(),
                "eta_match": eta_match.detach().cpu().clone(),
                "eta_scale": eta_scale.detach().cpu().clone(),
                "epoch": epoch,
            }
        if epoch % 25 == 0 or epoch == args.max_epochs:
            history.append({
                "epoch": epoch, "total_chi2_per_row": current,
                "accepted_chi2": float(values[1].detach()), "boundary_chi2": float(values[2].detach()),
                "norm_penalty": float(values[3].detach()), "theory_penalty": float(values[4].detach()),
                "fnp_anchor_penalty": float(values[5].detach()),
                "learning_rate": optimizer.param_groups[0]["lr"],
            })
        if epoch % 200 == 0:
            print(
                f"epoch={epoch:4d} total/N={current:.6f} "
                f"low={float(values[1].detach()):.3f} boundary={float(values[2].detach()):.3f}",
                flush=True,
            )
        if not lr_stages and epoch >= args.min_epochs and epoch - last_material_improvement >= args.plateau_patience:
            stopped_early = True
            print(f"plateau stop at epoch={epoch}; last material improvement={last_material_improvement}", flush=True)
            break

    assert best is not None
    model.load_state_dict(best["model"])
    with torch.no_grad():
        log_norms.copy_(best["log_norms"].to(device))
        eta_match.copy_(best["eta_match"].to(device))
        eta_scale.copy_(best["eta_scale"].to(device))

    lbfgs_start = float(evaluate()[0].detach()) / n_rows
    lbfgs_steps = 0
    if args.lbfgs_max_iter > 0:
        polish = torch.optim.LBFGS(
            parameters, lr=1.0, max_iter=args.lbfgs_max_iter, tolerance_grad=1.0e-7,
            tolerance_change=1.0e-10, history_size=50, line_search_fn="strong_wolfe")

        def closure():
            nonlocal lbfgs_steps
            polish.zero_grad(set_to_none=True)
            objective = evaluate()[0] / n_rows
            objective.backward()
            lbfgs_steps += 1
            return objective

        polish.step(closure)

    scipy_status = None
    if args.scipy_nuisance_polish:
        free_indices = np.flatnonzero(free_norm)
        w_low_np = fixed_w_low.detach().cpu().numpy()
        w_high_np = fixed_w_high.detach().cpu().numpy()
        low_base_np = w_low_np + accepted.Y_CS_used.to_numpy(float)
        profile_np = boundary[f"profile_{PROFILE}"].to_numpy(float)
        nlo_np = boundary.mcfm_nlo_pb_per_GeV.to_numpy(float)
        high_base_np = (1.0 - profile_np) * w_high_np + profile_np * nlo_np
        matching_np = (1.0 - profile_np) * (nlo_np - w_high_np)
        scale_np = profile_np * nlo_np * UPWARD_NLO_SCALE
        low_di_np, high_di_np = low_di.cpu().numpy(), high_di.cpu().numpy()
        low_data_np, low_error_np = accepted.target_used.to_numpy(float), accepted.sigma_used.to_numpy(float)
        high_data_np, high_error_np = boundary.CS.to_numpy(float), boundary.error.to_numpy(float)

        def residual_vector(vector):
            logs = log_norms.detach().cpu().numpy().copy()
            logs[free_indices] = vector[:len(free_indices)]
            norms_np = np.exp(logs * free_norm.astype(float))
            match_value, scale_value = vector[-2:]
            low_residual = (norms_np[low_di_np] * low_base_np - low_data_np) / low_error_np
            high_prediction = norms_np[high_di_np] * (
                high_base_np + match_value * matching_np + scale_value * scale_np)
            high_residual = (high_prediction - high_data_np) / high_error_np
            norm_residual = (norms_np[free_indices] - 1.0) / norm_width[free_indices]
            return np.concatenate((low_residual, high_residual, norm_residual, [match_value, scale_value]))

        scipy_start = np.concatenate((
            log_norms.detach().cpu().numpy()[free_indices],
            [float(eta_match.detach()), float(eta_scale.detach())]))
        scipy_result = least_squares(
            residual_vector, scipy_start, method="trf", ftol=1.0e-13, xtol=1.0e-13,
            gtol=1.0e-13, max_nfev=5000)
        with torch.no_grad():
            updated_logs = log_norms.detach().clone()
            updated_logs[torch.tensor(free_indices, device=device)] = torch.tensor(
                scipy_result.x[:len(free_indices)], dtype=nuisance_dtype, device=device)
            log_norms.copy_(updated_logs)
            eta_match.copy_(torch.tensor(scipy_result.x[-2], dtype=nuisance_dtype, device=device))
            eta_scale.copy_(torch.tensor(scipy_result.x[-1], dtype=nuisance_dtype, device=device))
        scipy_status = {
            "success": bool(scipy_result.success), "status": int(scipy_result.status),
            "message": str(scipy_result.message), "function_evaluations": int(scipy_result.nfev),
            "cost": float(scipy_result.cost), "optimality": float(scipy_result.optimality),
        }

    stationarity_loss = evaluate()[0] / n_rows
    stationarity_gradients = torch.autograd.grad(stationarity_loss, parameters)
    if args.freeze_fnp:
        fnp_gradient_l2 = None
        nuisance_gradients = stationarity_gradients
    else:
        fnp_gradient_l2 = float(torch.sqrt(sum(torch.sum(g.square()) for g in stationarity_gradients[:-3])).detach())
        nuisance_gradients = stationarity_gradients[-3:]
    nuisance_gradient_l2 = float(torch.sqrt(sum(torch.sum(g.square()) for g in nuisance_gradients)).detach())
    maximum_gradient_l2 = nuisance_gradient_l2 if fnp_gradient_l2 is None else max(fnp_gradient_l2, nuisance_gradient_l2)
    stationarity_pass = maximum_gradient_l2 <= args.stationarity_gradient_threshold

    with torch.no_grad():
        final_values = evaluate()
        total, low_chi2, high_chi2, norm_penalty, theory_penalty, fnp_anchor_penalty, pred_low, pred_high, w_high = final_values
        norms = torch.exp(log_norms * free_norm_t)
        low_pull = (pred_low - low_data) / low_error
        high_pull = (pred_high - high_data) / high_error

    accepted_out = accepted[["dataset", "row_id", "qT", "target_used", "sigma_used"]].copy()
    accepted_out["initial_prediction"] = initial_predictions[0]
    accepted_out["refit_prediction"] = pred_low.cpu().numpy()
    accepted_out["initial_pull"] = (initial_predictions[0] - accepted.target_used.to_numpy()) / accepted.sigma_used.to_numpy()
    accepted_out["refit_pull"] = low_pull.cpu().numpy()
    boundary_out = boundary[["dataset", "row_id", "qT_over_Q", "CS", "error"]].copy()
    boundary_out["initial_prediction"] = initial_predictions[1]
    boundary_out["refit_w_pb_per_GeV"] = w_high.cpu().numpy()
    boundary_out["refit_prediction"] = pred_high.cpu().numpy()
    boundary_out["initial_pull"] = (initial_predictions[1] - boundary.CS.to_numpy()) / boundary.error.to_numpy()
    boundary_out["refit_pull"] = high_pull.cpu().numpy()
    norm_out = pd.DataFrame({
        "dataset": datasets, "norm_prior_width": norm_width, "initial_norm": norm_start,
        "refit_norm": norms.cpu().numpy(),
        "refit_norm_pull": np.where(free_norm, (norms.cpu().numpy() - 1.0) / np.where(free_norm, norm_width, 1.0), 0.0),
    })
    parameter_delta = np.sqrt(sum(float(torch.sum((model.state_dict()[k].cpu() - v.cpu()).square())) for k, v in initial_np_state.items()))
    initial_parameter_norm = np.sqrt(sum(float(torch.sum(v.cpu().square())) for v in initial_np_state.values()))
    status = {
        "status": "experimental_unitary_transition_not_production",
        "fit": "joint_accepted_329_plus_boundary_24_differentiable_central_fnp_refit",
        "device": str(device), "profile": PROFILE, "epochs_run": epoch,
        "max_epochs": args.max_epochs, "stopped_on_plateau": stopped_early,
        "convergence_gate_pass": bool(stopped_early or stationarity_pass),
        "seed": args.seed, "initial_relative_parameter_perturbation": args.initial_perturbation,
        "initial_state_tag": args.initial_state_tag,
        "fnp_frozen_during_polish": args.freeze_fnp,
        "optimizer_polish": {
            "learning_rate_stages": lr_stages,
            "lbfgs_max_iter": args.lbfgs_max_iter, "lbfgs_closure_evaluations": lbfgs_steps,
            "lbfgs_start_chi2_per_row": lbfgs_start,
            "lbfgs_final_chi2_per_row": float(total) / n_rows,
            "fnp_gradient_l2_per_row_objective": fnp_gradient_l2,
            "nuisance_gradient_l2_per_row_objective": nuisance_gradient_l2,
            "stationarity_gradient_threshold": args.stationarity_gradient_threshold,
            "stationarity_gate_pass": stationarity_pass,
            "scipy_nuisance_polish": scipy_status,
        },
        "regularization": {
            "kind": "mean_squared_log_fnp_displacement_from_accepted_state",
            "strength": args.fnp_anchor_strength,
            "x_grid": FNP_ANCHOR_X,
            "b_min": 0.0001, "b_max": args.fnp_anchor_bmax, "b_count": args.fnp_anchor_nb,
        },
        "best_epoch": int(best["epoch"]), "learning_rate": initial_lr,
        "accepted_row_count": len(accepted), "boundary_row_count": len(boundary),
        "initial": {"total_chi2": initial_metrics[0], "accepted_chi2": initial_metrics[1], "boundary_chi2": initial_metrics[2], "norm_penalty": initial_metrics[3], "theory_penalty": initial_metrics[4], "fnp_anchor_penalty": initial_metrics[5]},
        "refit": {
            "total_chi2": float(total), "total_chi2_per_row": float(total) / n_rows,
            "accepted_chi2": float(low_chi2), "accepted_chi2_per_row": float(low_chi2) / len(accepted),
            "boundary_chi2": float(high_chi2), "boundary_chi2_per_row": float(high_chi2) / len(boundary),
            "norm_penalty": float(norm_penalty), "theory_penalty": float(theory_penalty),
            "fnp_anchor_penalty": float(fnp_anchor_penalty),
            "matching_nuisance_sigma": float(eta_match.detach()), "nlo_scale_nuisance_sigma": float(eta_scale.detach()),
            "accepted_max_absolute_pull": float(torch.max(torch.abs(low_pull))),
            "boundary_max_absolute_pull": float(torch.max(torch.abs(high_pull))),
            "accepted_max_relative_prediction_shift": float(np.max(np.abs(pred_low.cpu().numpy() / initial_predictions[0] - 1.0))),
            "fnp_parameter_l2_shift": parameter_delta,
            "fnp_parameter_relative_l2_shift": parameter_delta / initial_parameter_norm,
        },
        "production_state_modified": False,
        "promotion_authorized": False,
        "replica_stability_authorized": False,
        "next_gate": "interpret central fit impact, then decide whether replica stability is justified",
    }
    target.mkdir(parents=True, exist_ok=True)
    accepted_out.to_csv(target / "accepted_predictions.csv", index=False)
    boundary_out.to_csv(target / "boundary_predictions.csv", index=False)
    norm_out.to_csv(target / "dataset_norms.csv", index=False)
    pd.DataFrame(history).to_csv(target / "loss_history.csv", index=False)
    torch.save({f"np_factor.{k}": v.detach().cpu() for k, v in model.state_dict().items()}, target / "model_state.pt")
    torch.save({"log_norms": log_norms.detach().cpu(), "eta_match": eta_match.detach().cpu(), "eta_scale": eta_scale.detach().cpu()}, target / "nuisance_state.pt")
    (target / "fit_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
