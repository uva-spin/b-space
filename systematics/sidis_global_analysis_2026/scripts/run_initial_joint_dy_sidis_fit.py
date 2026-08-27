#!/usr/bin/env python3
"""Run the first isolated joint DY + identified-SIDIS fit.

This is a closure pilot, not production.  DY uses the frozen lambda=1
W-only kernel and FiLM F_NP start; SIDIS uses the 2026 COMPASS identified
pi/K collinear multiplicities and the fixed NNFF10 NNLO central member.  The
SIDIS source has no P_hT axis, so it constrains the collinear FF/PDF ratio and
normalization terms but cannot yet identify a TMDFF transverse width.  The
shared FiLM factor is therefore trained by the DY component, while all
reported objective components remain separate.

The one negative central NNFF10 K- prediction is excluded explicitly (signed
fixed-order FFs are not positivity-clamped); its row ID and reason are saved
in the fit manifest.  No frozen production file is read-write opened.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import lhapdf
import numpy as np
import pandas as pd
import torch

from sidis_global_analysis_2026.sidis_ff import LHAPDFFMember


ROOT = Path(__file__).resolve().parents[1]
SYSTEMATICS = ROOT.parent
DY_DIR = ROOT.parent / "dataset_identifiability_campaign_2026/outputs/lambda1_start_expansion96_s353_cont120000"
DY_W_GRID = SYSTEMATICS.parent / "outputs/v23a_tevatron_plus_lhcb7_fidacc_merged_cache/backend_cache/wpert_v23a_tevatron_plus_lhcb7_fidacc_b160.csv"
DY_STATE = DY_DIR / "model_state.pt"
DY_PREDICTIONS = DY_DIR / "accepted_predictions.csv"
DY_NORMS = DY_DIR / "dataset_norms.csv"
SIDIS_DATA = ROOT / "data/derived/compass_collinear_provisional/compass_collinear_provisional.csv"
OUT = ROOT / "outputs/initial_joint_dy_compass_collinear_pilot"
TRAINER = SYSTEMATICS.parent / "v21_smoothedA_tail_candidate/train_bt_dnn_v21_smoothedA_tail.py"

PID_E2 = {1: 1.0 / 9.0, 2: 4.0 / 9.0, 3: 1.0 / 9.0,
          -1: 1.0 / 9.0, -2: 4.0 / 9.0, -3: 1.0 / 9.0}
NEUTRON_SWAP = {1: 2, 2: 1, -1: -2, -2: -1, 3: 3, -3: -3}
FF_FAMILIES = {
    "nnff10_nnlo": {
        ("pi", "+"): "NNFF10_PIp_nnlo", ("pi", "-"): "NNFF10_PIm_nnlo",
        ("K", "+"): "NNFF10_KAp_nnlo", ("K", "-"): "NNFF10_KAm_nnlo",
    },
    # HAPS is kept as a deliberately explicit external comparison.  It is
    # not the primary choice because it was fitted using modern COMPASS SIDIS
    # inputs, so using it in the first closure run would be circular.
    "haps_nnlo": {
        ("pi", "+"): "HAPS-PiFF1.0-plus-NNLO", ("pi", "-"): "HAPS-PiFF1.0-minus-NNLO",
        ("K", "+"): "HAPS-KaFF1.0-plus-NNLO", ("K", "-"): "HAPS-KaFF1.0-minus-NNLO",
    },
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _central_collinear_ratio(row, pdf, ff: LHAPDFFMember) -> float:
    """Massless LO multiplicity ratio at one published central point."""

    x, z, q = float(row.x), float(row.z), float(row.Q_reconstructed)
    numerator = 0.0
    denominator = 0.0
    for pid, charge2 in PID_E2.items():
        neutron_pid = NEUTRON_SWAP[pid]
        fp = float(pdf.xfxQ(pid, x, q)) / x
        fn = float(pdf.xfxQ(neutron_pid, x, q)) / x
        f_iso = 0.5 * (fp + fn)
        denominator += charge2 * f_iso
        numerator += charge2 * f_iso * float(ff.density(pid, z, q))
    return numerator / denominator if denominator > 0.0 else np.nan


def _bin_averaged_collinear_ratio(row, pdf, ff: LHAPDFFMember, order: int) -> float:
    """Bin-average the LO ratio over the published x/y/z edges.

    The multiplicity is differential in z, so the numerator is averaged over
    the z interval while numerator and DIS denominator are integrated over x
    and y.  The massless LO DIS weight ``[1+(1-y)^2]/y`` is retained rather
    than averaging ratios uniformly.  This is a scalar closure diagnostic;
    it is not a substitute for the NLO/NNLO SIDIS coefficient functions.
    """

    if order < 2:
        raise ValueError("bin-average quadrature order must be at least 2")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    xlo, xhi = float(row.x_low), float(row.x_high)
    ylo, yhi = float(row.y_low), float(row.y_high)
    zlo, zhi = float(row.z_low), float(row.z_high)
    xnodes = 0.5 * (xhi + xlo) + 0.5 * (xhi - xlo) * nodes
    ynodes = 0.5 * (yhi + ylo) + 0.5 * (yhi - ylo) * nodes
    znodes = 0.5 * (zhi + zlo) + 0.5 * (zhi - zlo) * nodes
    nx, ny, nz = 0.5 * (xhi - xlo), 0.5 * (yhi - ylo), 0.5 * (zhi - zlo)
    numerator_integral = 0.0
    denominator_integral = 0.0
    for ix, x in enumerate(xnodes):
        for iy, y in enumerate(ynodes):
            q2 = 2.0 * 0.9382720813 * 160.0 * x * y
            q = float(np.sqrt(q2))
            if q < 1.0:
                raise ValueError(
                    f"published bin contains Q={q:.6g} below FF grid boundary Q=1 GeV"
                )
            pdf_values = {}
            denominator = 0.0
            for pid, charge2 in PID_E2.items():
                neutron_pid = NEUTRON_SWAP[pid]
                fp = float(pdf.xfxQ(pid, x, q)) / x
                fn = float(pdf.xfxQ(neutron_pid, x, q)) / x
                f_iso = 0.5 * (fp + fn)
                pdf_values[pid] = f_iso
                denominator += charge2 * f_iso
            dis_weight = (1.0 + (1.0 - y) ** 2) / y
            xy_weight = weights[ix] * weights[iy] * nx * ny * dis_weight
            denominator_integral += xy_weight * denominator
            z_average = 0.0
            for iz, z in enumerate(znodes):
                numerator = sum(
                    charge2 * pdf_values[pid] * float(ff.density(pid, z, q))
                    for pid, charge2 in PID_E2.items()
                )
                z_average += weights[iz] * numerator
            numerator_integral += xy_weight * (nz * z_average / (zhi - zlo))
    return numerator_integral / denominator_integral if denominator_integral > 0.0 else np.nan


def compute_collinear_ratio(
    frame: pd.DataFrame,
    pdf,
    ff_members: dict[tuple[str, str], LHAPDFFMember],
    *,
    mode: str = "midpoint",
    quadrature_order: int = 4,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    if mode not in {"midpoint", "bin_average"}:
        raise ValueError(f"unknown kinematic mode {mode!r}")
    ratios = np.full(len(frame), np.nan, dtype=float)
    diagnostics: list[dict[str, object]] = []
    for i, row in enumerate(frame.itertuples(index=False)):
        hadron_key = (str(row.hadron), str(row.charge))
        ff = ff_members[hadron_key]
        try:
            ratio = (
                _central_collinear_ratio(row, pdf, ff)
                if mode == "midpoint"
                else _bin_averaged_collinear_ratio(row, pdf, ff, quadrature_order)
            )
        except ValueError as exc:
            ratio = np.nan
            diagnostics.append({
                "row_id": str(row.row_id), "hadron": str(row.hadron),
                "charge": str(row.charge), "ratio": None,
                "reason": "bin_average_domain_failure", "detail": str(exc),
            })
        ratios[i] = ratio
        if not np.isfinite(ratio) or ratio <= 0.0:
            diagnostics.append({
                "row_id": str(row.row_id),
                "hadron": str(row.hadron),
                "charge": str(row.charge),
                "ratio": None if not np.isfinite(ratio) else float(ratio),
                "reason": "nonpositive_or_nonfinite_fixed_order_collinear_ratio",
            })
    return ratios, diagnostics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2500)
    ap.add_argument("--lr", type=float, default=2.0e-5)
    ap.add_argument("--sidis-lr", type=float, default=2.0e-3,
                    help="learning rate for the independent SIDIS normalization directions")
    ap.add_argument("--seed", type=int, default=260826)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--kinematic-mode", choices=("midpoint", "bin_average"), default="bin_average")
    ap.add_argument("--quadrature-order", type=int, default=4)
    ap.add_argument("--ff-family", choices=tuple(FF_FAMILIES), default="nnff10_nnlo")
    ap.add_argument("--ratio-csv", type=Path, default=None,
                    help="optional row_id,ratio CSV from an external theory probe")
    ap.add_argument("--ratio-column", default="ratio",
                    help="column in --ratio-csv containing the SIDIS ratio")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.float64

    trainer = load_module("sidis_initial_fit_trainer", TRAINER)
    dy = pd.read_csv(DY_PREDICTIONS)
    norms = pd.read_csv(DY_NORMS).set_index("dataset")
    w_grid = pd.read_csv(DY_W_GRID)
    b_grid, w_matrix = trainer.load_external_w_grid(dy.row_id.astype(str), DY_W_GRID)
    kernel = trainer.precompute_kernel_matrix(dy.qT.to_numpy(float), b_grid, w_matrix, dtype=dtype)
    wtab = w_grid.drop_duplicates("row_id").set_index("row_id").loc[dy.row_id.astype(str)]

    sidis = pd.read_csv(SIDIS_DATA)
    pdf = lhapdf.mkPDF("NNPDF40_nnlo_as_01180", 0)
    ff_sets = FF_FAMILIES[args.ff_family]
    ff_members = {key: LHAPDFFMember(name, 0) for key, name in ff_sets.items()}
    if args.ratio_csv is None:
        ratio, excluded = compute_collinear_ratio(
            sidis, pdf, ff_members, mode=args.kinematic_mode,
            quadrature_order=args.quadrature_order,
        )
        ratio_source = "internal massless LO PDF/FF ratio diagnostic"
    else:
        external = pd.read_csv(args.ratio_csv)
        required = {"row_id", args.ratio_column}
        missing = required.difference(external.columns)
        if missing:
            raise ValueError(f"external ratio CSV missing columns: {sorted(missing)}")
        if external["row_id"].astype(str).duplicated().any():
            raise ValueError("external ratio CSV has duplicate row_id values")
        ratio_map = external.set_index(external["row_id"].astype(str))[args.ratio_column]
        ratio = sidis["row_id"].astype(str).map(ratio_map).to_numpy(float)
        excluded = []
        for row_id, value, hadron, charge in zip(
                sidis.row_id.astype(str), ratio, sidis.hadron, sidis.charge):
            if not np.isfinite(value) or value <= 0.0:
                excluded.append({
                    "row_id": row_id, "hadron": str(hadron), "charge": str(charge),
                    "ratio": None if not np.isfinite(value) else float(value),
                    "reason": "external_theory_ratio_nonpositive_or_missing",
                })
        ratio_source = f"external ratio probe: {args.ratio_csv} column {args.ratio_column}"
    valid = np.isfinite(ratio) & (ratio > 0.0)
    sidis_fit = sidis.loc[valid].reset_index(drop=True)
    ratio_fit = ratio[valid]
    sidis_index = {name: i for i, name in enumerate(sorted(set(sidis_fit.hadron + sidis_fit.charge)))}
    # Four explicit hadron-charge normalization directions, each with a 10%
    # prior to absorb only the unresolved bin-integral/FF normalization in
    # this first collinear closure pilot.
    sidis_labels = ["K+", "K-", "pi+", "pi-"]
    sidis_group = np.asarray([f"{h}{c}" for h, c in zip(sidis_fit.hadron, sidis_fit.charge)])
    sidis_group_idx = torch.tensor([sidis_labels.index(x) for x in sidis_group], dtype=torch.long, device=device)

    def tensor(values):
        return torch.tensor(np.asarray(values), dtype=dtype, device=device)

    b = tensor(b_grid)
    k = tensor(kernel)
    x1, x2, q = tensor(wtab.x1.to_numpy()), tensor(wtab.x2.to_numpy()), tensor(dy.qT.to_numpy() * 0 + wtab.QM.to_numpy())
    target_dy, sigma_dy = tensor(dy.target_used.to_numpy()), tensor(np.maximum(dy.sigma_used.to_numpy(), 1.0e-12))
    dy_dataset_names = list(norms.index)
    dataset_idx = torch.tensor([dy_dataset_names.index(x) for x in dy.dataset], dtype=torch.long, device=device)
    norm_width = tensor(norms.loc[dy_dataset_names, "norm_width"].to_numpy(float))
    norm_mask = norm_width > 0.0
    norm_start = norms.loc[dy_dataset_names, "production_norm"].to_numpy(float)
    target_sidis = tensor(sidis_fit.multiplicity.to_numpy())
    sigma_sidis = tensor(np.maximum(sidis_fit.sigma_uncorrelated.to_numpy(), 1.0e-12))
    ratio_t = tensor(ratio_fit)

    np_factor = trainer.FilmNPFactor(width=48, cond_width=32, n_blocks=3, a0=0.05,
                                     min_a=0.0, a_mode="positive", exponent_clip=40.0,
                                     shape_mode="monotone", a_smooth_sigma=0.45,
                                     a_tail_amp=0.08, a_tail_b0=3.5, a_tail_width=0.25,
                                     dtype=dtype).to(device)
    saved = torch.load(DY_STATE, map_location=device, weights_only=True)
    saved = {key[len("np_factor."):]: val for key, val in saved.items() if key.startswith("np_factor.")}
    np_factor.load_state_dict(saved, strict=True)
    log_norm = torch.nn.Parameter(torch.tensor(np.log(np.maximum(norm_start, 1.0e-6)), dtype=dtype, device=device))
    # Start each SIDIS normalization near its data/model scale before the
    # joint optimizer runs.  The scale is only an initialization; the fit
    # still carries the declared log-normal prior.  Starting all four values
    # at one can leave a badly mismatched scalar closure pilot far from its
    # minimum at the deliberately small shared-F_iLM learning rate.
    sidis_initial_scales = []
    observed_sidis = sidis_fit.multiplicity.to_numpy(float)
    for label in sidis_labels:
        mask = sidis_group == label
        finite = ratio_fit[mask] > 0.0
        if np.any(finite):
            sidis_initial_scales.append(float(np.median(observed_sidis[mask][finite] / ratio_fit[mask][finite])))
        else:
            sidis_initial_scales.append(1.0)
    log_sidis_norm = torch.nn.Parameter(
        torch.tensor(np.log(np.maximum(sidis_initial_scales, 1.0e-6)), dtype=dtype, device=device)
    )
    parameters = list(np_factor.parameters()) + [log_norm]
    optimizer = torch.optim.Adam(parameters, lr=float(args.lr))
    sidis_optimizer = torch.optim.Adam([log_sidis_norm], lr=float(args.sidis_lr))
    history: list[dict[str, float]] = []
    best_total = float("inf")
    best_np = best_norm = best_sidis_norm = None
    for epoch in range(int(args.epochs) + 1):
        optimizer.zero_grad(set_to_none=True)
        sidis_optimizer.zero_grad(set_to_none=True)
        fnp1, fnp2 = np_factor(x1, b), np_factor(x2, b)
        dy_raw = torch.sum(k * fnp1 * fnp2, dim=1)
        dy_pred = torch.exp(log_norm[dataset_idx]) * dy_raw
        dy_chi2 = torch.sum(((dy_pred - target_dy) / sigma_dy).square())
        norm_penalty = torch.sum(
            (((torch.exp(log_norm) - 1.0) / torch.where(norm_mask, norm_width, torch.ones_like(norm_width))).square())
            * norm_mask.to(dtype)
        )
        sidis_pred = torch.exp(log_sidis_norm[sidis_group_idx]) * ratio_t
        sidis_chi2 = torch.sum(((sidis_pred - target_sidis) / sigma_sidis).square())
        sidis_norm_penalty = torch.sum((log_sidis_norm / 0.10).square())
        total = dy_chi2 + norm_penalty + sidis_chi2 + sidis_norm_penalty
        if epoch:
            (total / (len(dy) + len(sidis_fit))).backward()
            torch.nn.utils.clip_grad_norm_(parameters, 10.0)
            torch.nn.utils.clip_grad_norm_([log_sidis_norm], 10.0)
            optimizer.step()
            sidis_optimizer.step()
        current = float(total.detach())
        if not np.isfinite(current):
            raise RuntimeError(f"non-finite joint objective at epoch {epoch}")
        if current < best_total:
            best_total = current
            best_np = {k: v.detach().cpu().clone() for k, v in np_factor.state_dict().items()}
            best_norm = log_norm.detach().cpu().clone()
            best_sidis_norm = log_sidis_norm.detach().cpu().clone()
        if epoch % 100 == 0 or epoch == int(args.epochs):
            history.append({"epoch": epoch, "total_chi2": current,
                            "dy_chi2": float(dy_chi2.detach()),
                            "dy_norm_penalty": float(norm_penalty.detach()),
                            "sidis_chi2": float(sidis_chi2.detach()),
                            "sidis_norm_penalty": float(sidis_norm_penalty.detach())})
    if best_np is None or best_norm is None or best_sidis_norm is None:
        raise RuntimeError("joint optimizer never produced a finite objective")
    np_factor.load_state_dict(best_np, strict=True)
    with torch.no_grad():
        log_norm.copy_(best_norm.to(device)); log_sidis_norm.copy_(best_sidis_norm.to(device))
        dy_pred = torch.exp(log_norm[dataset_idx]) * torch.sum(k * np_factor(x1, b) * np_factor(x2, b), dim=1)
        sidis_pred = torch.exp(log_sidis_norm[sidis_group_idx]) * ratio_t
    dy_out = dy[["dataset", "row_id", "qT", "target_used", "sigma_used"]].copy()
    dy_out["prediction"] = dy_pred.detach().cpu().numpy()
    dy_out["pull"] = (dy_out.prediction - dy_out.target_used) / dy_out.sigma_used
    sidis_out = sidis_fit.copy()
    sidis_out["ff_collinear_ratio"] = ratio_fit
    sidis_out["prediction"] = sidis_pred.detach().cpu().numpy()
    sidis_out["pull"] = (sidis_out.prediction - sidis_out.multiplicity) / sidis_out.sigma_uncorrelated
    args.out.mkdir(parents=True, exist_ok=True)
    dy_out.to_csv(args.out / "dy_predictions.csv", index=False)
    sidis_out.to_csv(args.out / "sidis_predictions.csv", index=False)
    pd.DataFrame(history).to_csv(args.out / "loss_history.csv", index=False)
    torch.save({f"np_factor.{k}": v.detach().cpu() for k, v in np_factor.state_dict().items()}, args.out / "model_state.pt")
    grid_x = np.asarray([0.001, 0.003, 0.01, 0.03, 0.1, 0.2, 0.4, 0.7])
    grid_b = np.linspace(0.0001, 8.0, 321)
    with torch.no_grad():
        grid = np_factor(tensor(grid_x), tensor(grid_b)).cpu().numpy()
    pd.DataFrame({"x": np.repeat(grid_x, len(grid_b)), "bT": np.tile(grid_b, len(grid_x)), "F_NP": grid.ravel()}).to_csv(args.out / "fnp_grid.csv", index=False)
    dy_chi2_final = float(np.sum(dy_out.pull.to_numpy() ** 2))
    sidis_chi2_final = float(np.sum(sidis_out.pull.to_numpy() ** 2))
    summary = {
        "status": "initial_joint_dy_sidis_collinear_pilot_complete_not_production",
        "model": "frozen lambda=1 DY W-only FiLM F_NP + external identified COMPASS collinear pi/K",
        "dy_rows": int(len(dy_out)), "sidis_rows_available": int(len(sidis)), "sidis_rows_fit": int(len(sidis_out)),
        "sidis_rows_excluded": excluded,
        "objective": {"dy_chi2": dy_chi2_final, "sidis_chi2": sidis_chi2_final,
                       "total_chi2": dy_chi2_final + sidis_chi2_final,
                       "dy_chi2_per_row": dy_chi2_final / len(dy_out),
                       "sidis_chi2_per_row": sidis_chi2_final / len(sidis_out)},
        "sidis_normalizations": {label: float(np.exp(log_sidis_norm.detach().cpu().numpy()[i])) for i, label in enumerate(sidis_labels)},
        "sidis_normalization_initialization": {label: float(sidis_initial_scales[i]) for i, label in enumerate(sidis_labels)},
        "dy_normalizations": {name: float(np.exp(log_norm.detach().cpu().numpy()[i])) for i, name in enumerate(dy_dataset_names)},
        "ff": {"pdf_set": "NNPDF40_nnlo_as_01180", "member": 0,
               "family": args.ff_family,
               "sets": {f"{h}{c}": name for (h, c), name in ff_sets.items()},
               "member_policy": "central only in pilot"},
        "kinematics": {"mode": args.kinematic_mode, "quadrature_order": int(args.quadrature_order),
                       "bin_average_weight": "massless LO DIS [1+(1-y)^2]/y; diagnostic only"},
        "ratio_source": ratio_source,
        "sidis_scope": str(SIDIS_DATA),
        "dy_scope": str(DY_PREDICTIONS),
        "limitations": [
            "COMPASS addendum has no P_hT axis; this initial SIDIS term is collinear and cannot constrain a TMDFF transverse factor.",
            "Public COMPASS table has no full covariance; stat and point-to-point sys are quadratured only for this pilot.",
            "HERMES zxpt-3D archive/covariance remains unresolved; no HERMES rows are fitted.",
            "DY finite-Y is outside this pilot; the frozen lambda=1 W-only kernel is used as the shared DY anchor.",
        ],
        "production_files_modified": False,
        "promotion_authorized": False,
    }
    (args.out / "fit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
