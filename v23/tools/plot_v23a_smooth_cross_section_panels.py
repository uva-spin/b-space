#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import math
import sys
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter


def import_from_path(path: Path, name: str):
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def namespace_from_config(config: dict) -> SimpleNamespace:
    ns = SimpleNamespace(**config)
    aliases = {
        "b_star_max": "b_star_max",
        "no_cap_mub_at_Q": "no_cap_mub_at_Q",
        "no_nlo_dev_switch": "no_nlo_dev_switch",
        "no_upsilon_veto": "no_upsilon_veto",
        "allow_zero_y_in_matched": "allow_zero_y_in_matched",
    }
    for name in aliases:
        if not hasattr(ns, name):
            setattr(ns, name, False)
    return ns


def backend_cfg_from_config(backend, config: dict):
    overrides = {
        "b_min": float(config.get("b_min", 1.0e-4)),
        "b_max": float(config.get("b_max", 8.0)),
        "n_b": int(config.get("n_b", 160)),
        "bstar_bmax": float(config.get("b_star_max", 1.5)),
        "mu_min": float(config.get("mu_min", 1.3)),
        "cap_mub_at_Q": not bool(config.get("no_cap_mub_at_Q", False)),
        "q0": float(config.get("q0", 2.0)),
        "resum_order": str(config.get("resum_order", "n3llp")),
        "nf": int(config.get("nf", 5)),
        "n_sudakov_quad": int(config.get("n_sudakov_quad", 32)),
        "alpha_em": float(config.get("alpha_em", 1.0 / 137.035999084)),
        "hc_factor": float(config.get("hc_factor", 3.893793656e8)),
        "prefactor_scheme": str(config.get("prefactor_scheme", "oldA_to_CS")),
        "global_norm": float(config.get("backend_global_norm", 1.0)),
        "flavors": tuple(int(f) for f in config.get("flavors", [1, 2, 3])),
        "target_mode": str(config.get("target_mode", "nuclear_isospin")),
        "y_mode": "zero",
    }
    optional = {
        "nlo_y_pilot_strength": 1.0,
        "nlo_y_transition": 0.2,
        "nlo_y_transition_width": 0.15,
        "nlo_real_quad": 96,
        "nlo_real_norm": 1.0,
        "nlo_singular_norm": 1.0,
        "nlo_y_component": "raw",
        "nlo_y_clip_multiple": 5.0,
        "nlo_dev_min_qt_over_q": 1.0e-4,
        "nlo_singular_mode": "asymptotic_damped",
        "nlo_singular_rsub": 0.1,
        "nlo_singular_power": 2.0,
        "nlo_singular_damp_kind": "exp",
        "nlo_real_convention": "base",
        "nlo_singular_convention": "base",
        "nlo_alpha_convention": "alpha_over_pi",
        "nlo_real_tail_repair": "mcfm_logistic",
        "nlo_real_tail_r0": 0.53,
        "nlo_real_tail_width": 0.008,
        "nlo_real_tail_rinf": 0.18,
        "nlo_dev_use_switch": True,
    }
    for key, default in optional.items():
        overrides[key] = config.get(key, default)
    # The accepted production panels are restricted to qT/Q <= 0.10. In the
    # validated cache the development Y term is switched off in this region, so
    # force W-only dense evaluation instead of spending time recomputing Y_NLO.
    overrides["match_order"] = "none"
    overrides["y_mode"] = "zero"

    config_type = backend.CSSConfig
    if is_dataclass(config_type):
        valid = {field.name for field in fields(config_type)}
        overrides = {k: v for k, v in overrides.items() if k in valid}
    return config_type(**overrides)


def build_model(trainer, config: dict, b_grid: np.ndarray, kernel: np.ndarray, device: torch.device, dtype: torch.dtype):
    np_factor = trainer.FilmNPFactor(
        width=int(config.get("np_width", 48)),
        cond_width=int(config.get("np_cond_width", 32)),
        n_blocks=int(config.get("np_blocks", 3)),
        a0=float(config.get("np_a0", 0.08)),
        min_a=float(config.get("np_min_a", 0.0)),
        a_mode=str(config.get("np_a_mode", "positive")),
        exponent_clip=float(config.get("fnp_exponent_clip", 40.0)),
        shape_mode=str(config.get("np_shape_mode", "direct")),
        a_smooth_sigma=float(config.get("np_a_smooth_sigma", 0.0)),
        a_tail_amp=float(config.get("np_a_tail_amp", 0.0)),
        a_tail_b0=float(config.get("np_a_tail_b0", 3.5)),
        a_tail_width=float(config.get("np_a_tail_width", 0.25)),
        dtype=dtype,
    ).to(device)
    if bool(config.get("learn_gk", False)):
        if str(config.get("gk_mode", "bounded")) == "bounded":
            gk_model = trainer.BoundedGKModel(
                width=int(config.get("gk_width", 24)),
                n_layers=int(config.get("gk_layers", 2)),
                b0=float(config.get("gk_b0", 0.02)),
                bmax=float(config.get("gk_bmax", 0.08)),
                dtype=dtype,
            ).to(device)
        else:
            gk_model = trainer.GKModel(
                width=int(config.get("gk_width", 24)),
                n_layers=int(config.get("gk_layers", 2)),
                b0=float(config.get("gk_b0", 0.02)),
                dtype=dtype,
            ).to(device)
    else:
        gk_model = trainer.ZeroGK().to(device)

    return trainer.PrecomputedKernelModel(
        b_grid=b_grid,
        kernel_matrix=kernel,
        np_factor=np_factor,
        gk_model=gk_model,
        q0=float(config.get("q0", 2.0)),
        cs_log=str(config.get("cs_log", "lnQ")),
        cs_kernel_convention=str(config.get("cs_kernel_convention", "pair")),
        learn_global_norm=bool(config.get("learn_global_norm", False)),
        global_norm_init=float(config.get("global_norm_init", 1.0)),
        dtype=dtype,
        device=device,
    ).to(device)


def load_model_state(model, state_path: Path, device: torch.device) -> None:
    state = torch.load(state_path, map_location=device)
    own = model.state_dict()
    filtered = {}
    for key, value in state.items():
        if key in {"b", "kernel_matrix"} or key.startswith("dataset_norms."):
            continue
        if key in own and tuple(own[key].shape) == tuple(value.shape):
            filtered[key] = value.to(device=own[key].device, dtype=own[key].dtype)
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    bad_missing = [k for k in missing if not k.startswith(("b", "kernel_matrix"))]
    if bad_missing or unexpected:
        print("state load missing:", bad_missing[:10])
        print("state load unexpected:", unexpected[:10])


def state_prediction(
    trainer,
    config: dict,
    dense: pd.DataFrame,
    b_grid: np.ndarray,
    kernel: np.ndarray,
    state_path: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> np.ndarray:
    model = build_model(trainer, config, b_grid, kernel, device, dtype)
    load_model_state(model, state_path, device)
    model.eval()
    with torch.no_grad():
        q = torch.tensor(dense["QM"].to_numpy(float), dtype=dtype, device=device)
        x1 = torch.tensor(dense["x1"].to_numpy(float), dtype=dtype, device=device)
        x2 = torch.tensor(dense["x2"].to_numpy(float), dtype=dtype, device=device)
        idx = torch.arange(len(dense), dtype=torch.long, device=device)
        y = torch.zeros(len(dense), dtype=dtype, device=device)
        return model(idx, q, x1, x2, y).detach().cpu().numpy()


def norm_factors_for_run(run: Path, dense: pd.DataFrame) -> np.ndarray:
    norms_path = run / "dataset_norms.csv"
    norm_map = {}
    if norms_path.exists():
        norms = pd.read_csv(norms_path)
        norm_map = dict(zip(norms["dataset"].astype(str), norms["norm_scale"].astype(float)))
    return dense["dataset"].astype(str).map(norm_map).fillna(1.0).astype(float).to_numpy()


def make_groups(df: pd.DataFrame, datasets: list[str]) -> list[tuple[str, str, pd.DataFrame]]:
    groups: list[tuple[str, str, pd.DataFrame]] = []
    for dataset in datasets:
        sub = df[df["dataset"].astype(str).eq(dataset)].copy()
        if sub.empty:
            continue
        if dataset.startswith(("E288", "E605", "E772")):
            for qm, g in sub.groupby("QM", observed=False, sort=True):
                groups.append((dataset, f"QM={float(qm):g}", g.sort_values("qT")))
        else:
            groups.append((dataset, dataset, sub.sort_values("qT")))
    return groups


def dense_rows_for_group(g: pd.DataFrame, n_qt: int) -> pd.DataFrame:
    template = g.iloc[0].copy()
    is_collider = str(template.get("target", "")).lower().replace("-", "_") in {"pbar_p", "pp", "p_p"} or str(template.get("beam_config", "")).lower().replace("-", "_") in {"pbar_p", "pp", "p_p"}
    qmax = float(np.nanmax(g["qT"].to_numpy(float)))
    low_edges = pd.to_numeric(g.get("qT_low", pd.Series(dtype=float)), errors="coerce").dropna()
    if is_collider and not low_edges.empty:
        qmin = max(0.0, float(low_edges.min()))
    else:
        qmin = max(1.0e-6, float(np.nanmin(g["qT"].to_numpy(float))))
    if qmax <= qmin:
        qmin = max(0.0, float(np.nanmin(g["qT"].to_numpy(float))) * 0.5)
    q_grid = np.linspace(qmin, qmax, int(n_qt))
    source_qt = g["qT"].to_numpy(float)
    order = np.argsort(source_qt)
    source_qt = source_qt[order]
    interp_cols = {}
    for col in ["PreFactor"]:
        if col in g.columns:
            values = pd.to_numeric(g[col], errors="coerce").to_numpy(float)[order]
            if np.isfinite(values).all() and len(np.unique(source_qt)) >= 2:
                product = values * source_qt
                rel_spread = float(np.nanstd(product) / max(abs(np.nanmean(product)), 1.0e-300))
                if col == "PreFactor" and not is_collider and rel_spread < 1.0e-6:
                    interp_cols[col] = float(np.nanmean(product)) / np.maximum(q_grid, 1.0e-12)
                else:
                    interp_cols[col] = np.interp(q_grid, source_qt, values)
    rows = []
    for i, qt in enumerate(q_grid):
        row = template.copy()
        row["qT"] = float(qt)
        row["qT_over_Q"] = float(qt) / max(float(row["QM"]), 1.0e-12)
        row["row_id"] = f"smooth:{template['dataset']}:{template.get('QM', np.nan)}:{template.get('y', np.nan)}:{i}"
        row["local_index"] = int(i)
        row["smooth_prefactor_scale"] = 1.0
        for col, values in interp_cols.items():
            if col == "PreFactor" and not is_collider:
                base = float(template[col])
                row["smooth_prefactor_scale"] = float(base / values[i]) if values[i] != 0.0 else 1.0
            else:
                row[col] = float(values[i])
        if is_collider:
            row["qT_low"] = np.nan
            row["qT_high"] = np.nan
            row["qT_bin_width"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def compute_or_load_dense_kernel(
    trainer,
    backend,
    config: dict,
    groups,
    n_qt: int,
    dtype: torch.dtype,
    *,
    cache: Path | None,
    progress: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    dense = pd.concat([dense_rows_for_group(g, n_qt) for _, _, g in groups], ignore_index=True)
    if cache is not None and cache.exists():
        loaded = np.load(cache, allow_pickle=False)
        cached_dense = pd.read_csv(cache.with_suffix(".rows.csv"))
        return cached_dense, loaded["b_grid"], loaded["kernel"]

    cfg = backend_cfg_from_config(backend, config)
    pdf = backend.LHAPDFProvider(config.get("pdf_set", "NNPDF40_nnlo_as_01180"), int(config.get("pdf_member", 0)), use_toy_pdf=False)
    b_grid, w_matrix, _ = backend.compute_backend_grids(dense, pdf, cfg, progress=progress)
    kernel = trainer.precompute_kernel_matrix(dense["qT"].to_numpy(float), b_grid, w_matrix, dtype=dtype)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, b_grid=b_grid, kernel=kernel)
        dense.to_csv(cache.with_suffix(".rows.csv"), index=False)
    return dense, b_grid, kernel


def attach_central_prediction(trainer, config: dict, dense: pd.DataFrame, b_grid: np.ndarray, kernel: np.ndarray, run: Path, device: torch.device, dtype: torch.dtype) -> pd.DataFrame:
    pred_raw = state_prediction(trainer, config, dense, b_grid, kernel, run / "model_state.pt", device, dtype)
    dense["pred_smooth_raw_before_dataset_norm"] = pred_raw
    dense["dataset_norm_factor"] = norm_factors_for_run(run, dense)
    scale = pd.to_numeric(dense.get("smooth_prefactor_scale", 1.0), errors="coerce").fillna(1.0).to_numpy(float)
    dense["pred_smooth_CS"] = dense["pred_smooth_raw_before_dataset_norm"] * scale * dense["dataset_norm_factor"]
    return dense


def attach_replica_bands(
    trainer,
    config: dict,
    dense: pd.DataFrame,
    b_grid: np.ndarray,
    kernel: np.ndarray,
    pattern: str,
    device: torch.device,
    dtype: torch.dtype,
    *,
    norm_mode: str,
    central_run: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = sorted(Path(p).parent for p in glob.glob(pattern))
    if not runs:
        raise SystemExit(f"No replica model states matched: {pattern}")
    scale = pd.to_numeric(dense.get("smooth_prefactor_scale", 1.0), errors="coerce").fillna(1.0).to_numpy(float)
    central_norm = norm_factors_for_run(central_run, dense)
    values = []
    records = []
    for i, run in enumerate(runs, start=1):
        raw = state_prediction(trainer, config, dense, b_grid, kernel, run / "model_state.pt", device, dtype)
        if norm_mode == "replica":
            norm = norm_factors_for_run(run, dense)
        elif norm_mode == "central":
            norm = central_norm
        elif norm_mode == "none":
            norm = np.ones(len(dense), dtype=float)
        else:
            raise ValueError(f"unknown replica band norm mode: {norm_mode}")
        pred = raw * norm * scale
        values.append(pred)
        records.append({"replica_index": i, "run": str(run), "n_points": int(len(pred)), "finite": bool(np.isfinite(pred).all())})
        print(f"  replica smooth prediction: {i}/{len(runs)} {run.name}", flush=True)
    arr = np.vstack(values)
    dense["pred_smooth_replica_q16"] = np.nanquantile(arr, 0.16, axis=0)
    dense["pred_smooth_replica_q50"] = np.nanquantile(arr, 0.50, axis=0)
    dense["pred_smooth_replica_q84"] = np.nanquantile(arr, 0.84, axis=0)
    dense["pred_smooth_replica_mean"] = np.nanmean(arr, axis=0)
    dense["pred_smooth_replica_std"] = np.nanstd(arr, axis=0, ddof=1)
    dense["n_smooth_replicas"] = arr.shape[0]
    return dense, pd.DataFrame(records)


def panel_label(g: pd.DataFrame) -> str:
    row = g.iloc[0]
    bits = [rf"$Q_M={float(row['QM']):g}$"]
    qlo = pd.to_numeric(row.get("QM_Low", np.nan), errors="coerce")
    qhi = pd.to_numeric(row.get("QM_High", np.nan), errors="coerce")
    if np.isfinite(qlo) and np.isfinite(qhi):
        bits.append(rf"$[{float(qlo):g},{float(qhi):g}]$")
    ylo = pd.to_numeric(row.get("y_Low", np.nan), errors="coerce")
    yhi = pd.to_numeric(row.get("y_High", np.nan), errors="coerce")
    if np.isfinite(ylo) and np.isfinite(yhi):
        bits.append(rf"$y\in[{float(ylo):g},{float(yhi):g}]$")
    elif np.isfinite(pd.to_numeric(row.get("xF", np.nan), errors="coerce")):
        bits.append(rf"$x_F={float(row['xF']):g}$")
    return "\n".join(bits)


def balanced_grid(n: int, max_cols: int) -> tuple[int, int]:
    if n <= 1:
        return 1, 1
    ncols = min(int(max_cols), int(math.ceil(math.sqrt(n))))
    nrows = int(math.ceil(n / ncols))
    return nrows, ncols


def plot_dataset(
    central: pd.DataFrame,
    dense: pd.DataFrame,
    dataset: str,
    out_dir: Path,
    max_cols: int,
    *,
    with_bands: bool,
    band_label: str,
) -> dict:
    groups = [(name, label, g) for name, label, g in make_groups(central, [dataset])]
    nrows, ncols = balanced_grid(len(groups), max_cols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(max(4.8, 3.45 * ncols), max(3.7, 2.85 * nrows)), dpi=180, squeeze=False)
    scale_vals = pd.to_numeric(central.loc[central["dataset"].astype(str).eq(dataset), "CS"], errors="coerce").dropna()
    yscale = "log" if len(scale_vals) and scale_vals.max() / max(scale_vals[scale_vals > 0].min(), 1.0e-300) > 80 else "linear"

    for ax, (_, _, g) in zip(axes.ravel(), groups):
        data_g = g.sort_values("qT")
        dsub = dense[dense["dataset"].astype(str).eq(dataset)].copy()
        if dataset.startswith(("E288", "E605", "E772")):
            dsub = dsub[np.isclose(dsub["QM"].astype(float), float(data_g["QM"].iloc[0]))]
        dsub = dsub.sort_values("qT")

        if with_bands and {"pred_smooth_replica_q16", "pred_smooth_replica_q84"}.issubset(dsub.columns):
            ax.fill_between(
                dsub["qT"].to_numpy(float),
                dsub["pred_smooth_replica_q16"].to_numpy(float),
                dsub["pred_smooth_replica_q84"].to_numpy(float),
                color="#1f77b4",
                alpha=0.22,
                linewidth=0,
                zorder=1,
            )
        ax.plot(dsub["qT"], dsub["pred_smooth_CS"], color="#1f77b4", lw=2.0, zorder=2)
        ax.errorbar(
            data_g["qT"],
            data_g["CS"],
            yerr=data_g["sigma_used"],
            fmt="o",
            ms=3.3,
            mfc="white",
            mec="black",
            mew=0.8,
            ecolor="black",
            elinewidth=0.8,
            capsize=1.7,
            zorder=3,
        )
        ax.text(0.06, 0.91, panel_label(data_g), transform=ax.transAxes, va="top", ha="left", fontsize=8.5)
        ax.set_yscale(yscale)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        if yscale == "log":
            ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
            ax.yaxis.set_minor_formatter(NullFormatter())
        else:
            ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which="major", direction="in", top=True, right=True, length=4.5, width=0.9, labelsize=8.5)
        ax.tick_params(which="minor", direction="in", top=True, right=True, length=2.5, width=0.7)

    for ax in axes.ravel()[len(groups):]:
        ax.axis("off")
    for ax in axes[-1, :]:
        if ax.has_data():
            ax.set_xlabel(r"$q_T$ [GeV]", fontsize=10.5)
    for ax in axes[:, 0]:
        if ax.has_data():
            ax.set_ylabel(r"$d\sigma$ (table units)", fontsize=10.5)
    handles = [
        Line2D([0], [0], color="#1f77b4", lw=2.0, label="smooth fitted TMD prediction"),
    ]
    if with_bands:
        handles.append(Line2D([0], [0], color="#1f77b4", lw=7, alpha=0.22, label=band_label))
    handles.append(Line2D([0], [0], color="black", marker="o", linestyle="None", mfc="white", label="data"))
    if len(groups) == 1:
        axes.ravel()[0].set_title(dataset, fontsize=13, pad=8)
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=min(3, len(handles)), frameon=False, fontsize=10)
        fig.tight_layout(rect=(0, 0.10, 1, 1))
    else:
        fig.suptitle(dataset, x=0.02, y=0.99, ha="left", va="top", fontsize=13)
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.57, 0.985), ncol=min(3, len(handles)), frameon=False, fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.88))
    suffix = "smooth_cross_section_bands" if with_bands else "smooth_cross_section_panels"
    png = out_dir / f"{dataset}_{suffix}.png"
    pdf = out_dir / f"{dataset}_{suffix}.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"dataset": dataset, "png": str(png), "pdf": str(pdf), "n_panels": len(groups), "yscale": yscale}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--trainer-script", default="v21_tail_release_amp0p019_candidate/train_bt_dnn_v21_replica_stable.py")
    ap.add_argument("--central-predictions", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-qt", type=int, default=100)
    ap.add_argument("--max-cols", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--kernel-cache", default=None)
    ap.add_argument("--replica-model-glob", default=None)
    ap.add_argument(
        "--replica-band-norm-mode",
        choices=["central", "replica", "none"],
        default="central",
        help=(
            "central: scale every replica by the central profiled normalization, "
            "isolating TMD/fit-shape uncertainty; replica: include each replica's "
            "profiled normalization nuisance; none: no dataset normalization."
        ),
    )
    args = ap.parse_args()

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.linewidth": 1.0,
        "savefig.dpi": 300,
    })

    run = Path(args.run)
    metrics = json.loads((run / "metrics.json").read_text())
    config = metrics["config"]
    dtype = torch.float32 if str(config.get("dtype", "float32")) == "float32" else torch.float64
    device = torch.device(args.device)
    trainer = import_from_path(Path(args.trainer_script), "v23_smooth_xsec_trainer")
    backend = import_from_path(Path(config["backend_script"]), "v23_smooth_xsec_backend")
    central = pd.read_csv(args.central_predictions)
    groups = make_groups(central, [str(d) for d in args.datasets])
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.kernel_cache) if args.kernel_cache else out / "v23a_smooth_cross_section_dense_kernel_cache.npz"
    dense, b_grid, kernel = compute_or_load_dense_kernel(
        trainer,
        backend,
        config,
        groups,
        int(args.n_qt),
        dtype,
        cache=cache_path,
    )
    dense = attach_central_prediction(trainer, config, dense, b_grid, kernel, run, device, dtype)
    replica_summary = pd.DataFrame()
    with_bands = bool(args.replica_model_glob)
    if with_bands:
        dense, replica_summary = attach_replica_bands(
            trainer,
            config,
            dense,
            b_grid,
            kernel,
            str(args.replica_model_glob),
            device,
            dtype,
            norm_mode=str(args.replica_band_norm_mode),
            central_run=run,
        )
        replica_summary.to_csv(out / "v23a_smooth_cross_section_replica_summary.csv", index=False)
    dense.to_csv(out / "v23a_smooth_cross_section_predictions.csv", index=False)
    if str(args.replica_band_norm_mode) == "central":
        band_label = "68% TMD-shape replicas"
    elif str(args.replica_band_norm_mode) == "replica":
        band_label = "68% replicas incl. norm"
    else:
        band_label = "68% raw replicas"
    summaries = [
        plot_dataset(
            central,
            dense,
            str(dataset),
            out,
            int(args.max_cols),
            with_bands=with_bands,
            band_label=band_label,
        )
        for dataset in args.datasets
    ]
    manifest = {
        "run": str(run),
        "central_predictions": str(args.central_predictions),
        "kernel_cache": str(cache_path),
        "n_qt_per_panel": int(args.n_qt),
        "datasets": summaries,
        "definition": "Smooth qT curves from recomputed W(b) kernels multiplied by the fitted b-space F_NP factors and the profiled dataset normalizations.",
        "uncertainty_note": "Bands, when present, are q16/q84 over smooth predictions from the fitted experimental-data replica models evaluated on the same dense W-kernel cache.",
        "replica_model_glob": str(args.replica_model_glob) if args.replica_model_glob else None,
        "replica_band_norm_mode": str(args.replica_band_norm_mode),
        "n_replicas": int(len(replica_summary)) if not replica_summary.empty else 0,
        "collider_note": "For dense collider qT points, qT bin edges are cleared so the backend uses the differential 2*pi*qT Jacobian; data points remain the published binned measurements.",
    }
    (out / "v23a_smooth_cross_section_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
