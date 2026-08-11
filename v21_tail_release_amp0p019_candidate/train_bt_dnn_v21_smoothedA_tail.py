#!/usr/bin/env python3
"""
train_bt_dnn_v18.py

Single-file PyTorch training script for the FiLM/DNN b_T-space nonperturbative
factor.  It is designed to live next to bt_internal_css_backend_v18.py and to avoid
the multi-module btfit package while we are developing the backend.

Typical usage, from the directory containing ./Data and both scripts:

  python3 train_bt_dnn.py \
    --data-dir ./Data \
    --mode tmd_only \
    --w-backend internal_css \
    --pdf-set NNPDF40_nnlo_as_01180 \
    --resum-order nnll \
    --epochs 300 \
    --batch-size 128 \
    --learn-gk \
    --out outputs/tmd_only_test

Matched-mode pilot, currently W-only unless a real Y grid/backend is supplied:

  python3 train_bt_dnn.py \
    --data-dir ./Data \
    --mode matched \
    --qT-max-over-Q 1.0 \
    --w-backend internal_css \
    --y-mode zero \
    --epochs 500 \
    --out outputs/internal_css_matched_pilot

External precomputed-grid mode is also supported:

  python3 train_bt_dnn.py \
    --data-dir ./Data \
    --mode matched \
    --w-backend external \
    --w-grid internal_css_grids/wpert_internal_css_matched.csv \
    --y-grid internal_css_grids/y_internal_css_matched.csv \
    --epochs 500 \
    --out outputs/external_grid_fit

The fitted object is the nonperturbative factor

  F_NP(x,b) = exp[-b^2 A_theta(x,b)] or a monotone cumulative variant; A_theta >= 0,

and optionally

  g_K(b) = b^2 B_theta(b),  B_theta >= 0.

The perturbative W kernel and the Y term are not learned by the DNN.

Version note: v11 adds --match-order and forwards N3LL'-ready/matched-mode
backend settings to bt_internal_css_backend_v18.py. The built-in nlo_pilot finite
tail is for matched-mode plumbing; audited production Y_NLO must still replace it.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Backend import
# -----------------------------------------------------------------------------


def import_backend(path: str | Path | None = None):
    """Import bt_internal_css_backend_v18.py from an explicit path or this directory."""
    candidates = []
    if path is not None:
        candidates.append(Path(path).expanduser())
    here = Path(__file__).resolve().parent
    candidates.append(here / "bt_internal_css_backend_v18.py")
    candidates.append(Path.cwd() / "bt_internal_css_backend_v18.py")
    candidates.append(here / "bt_internal_css_backend.py")
    candidates.append(Path.cwd() / "bt_internal_css_backend.py")

    for p in candidates:
        if p.exists():
            spec = importlib.util.spec_from_file_location("bt_internal_css_backend", str(p.resolve()))
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            sys.modules["bt_internal_css_backend"] = mod
            spec.loader.exec_module(mod)  # type: ignore[attr-defined]
            return mod, p.resolve()
    raise FileNotFoundError(
        "Could not find bt_internal_css_backend_v18.py. Put it in the same directory as "
        "train_bt_dnn.py or pass --backend-script /path/to/bt_internal_css_backend_v18.py."
    )


# -----------------------------------------------------------------------------
# Replica / uncertainty handling
# -----------------------------------------------------------------------------


PAPER_NORM_REL: Dict[str, float] = {
    "E288_200": 0.25,
    "E288_300": 0.25,
    "E288_400": 0.25,
    "E605": 0.15,
    "E772": 0.0,
}

PAPER_EXTRA_PTP_REL: Dict[str, float] = {
    "E288_200": 0.0,
    "E288_300": 0.0,
    "E288_400": 0.0,
    "E605": 0.10,
    "E772": 0.0,
}


@dataclass(frozen=True)
class ReplicaConfig:
    observable: str = "CS"
    error_column: str = "error"
    norm_source: str = "paper"  # paper, csv, none
    ptp_source: str = "paper"   # paper, csv, none


def _row_ptp_rel(df: pd.DataFrame, cfg: ReplicaConfig) -> np.ndarray:
    if cfg.ptp_source == "paper":
        return df["dataset"].map(lambda d: PAPER_EXTRA_PTP_REL.get(str(d), 0.0)).to_numpy(float)
    if cfg.ptp_source == "csv":
        return df.get("sysP2P_rel", pd.Series(np.zeros(len(df)))).to_numpy(float)
    if cfg.ptp_source == "none":
        return np.zeros(len(df), dtype=float)
    raise ValueError(f"Unknown ptp_source={cfg.ptp_source!r}")


def _row_norm_rel(df: pd.DataFrame, cfg: ReplicaConfig) -> np.ndarray:
    if cfg.norm_source == "paper":
        return df["dataset"].map(lambda d: PAPER_NORM_REL.get(str(d), 0.0)).to_numpy(float)
    if cfg.norm_source == "csv":
        return df.get("sysNorm_rel", pd.Series(np.zeros(len(df)))).to_numpy(float)
    if cfg.norm_source == "none":
        return np.zeros(len(df), dtype=float)
    raise ValueError(f"Unknown norm_source={cfg.norm_source!r}")


def build_uncertainties(df: pd.DataFrame, cfg: ReplicaConfig) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    y = out[cfg.observable].to_numpy(float)
    sigma_base = out[cfg.error_column].to_numpy(float)
    ptp_rel = _row_ptp_rel(out, cfg)
    norm_rel = _row_norm_rel(out, cfg)
    out["sigma_uncorr"] = np.sqrt(np.maximum(sigma_base, 0.0) ** 2 + (ptp_rel * y) ** 2)
    out["norm_rel"] = norm_rel
    out["ptp_rel_used"] = ptp_rel
    out["norm_rel_used"] = norm_rel
    return out




def backend_scale_diagnostics(
    *,
    df: pd.DataFrame,
    baseline: np.ndarray,
    target: np.ndarray,
    sigma: np.ndarray | None = None,
) -> Dict[str, Any]:
    """Compute robust scalar diagnostics for a fixed backend baseline.

    baseline is W+Y with F_NP=1 and gK=0.  The most useful scale factors are:
      inverse_median: 1 / median(W/data), using all finite nonzero data rows.
      median_data_over_w: median(data/W), using positive-baseline rows only.
      weighted_ls: argmin_g sum_i ((g W_i - data_i)/sigma_i)^2.
    """
    baseline = np.asarray(baseline, dtype=float)
    target = np.asarray(target, dtype=float)
    if sigma is None:
        sigma = np.ones_like(target, dtype=float)
    else:
        sigma = np.asarray(sigma, dtype=float)
    finite = np.isfinite(baseline) & np.isfinite(target) & np.isfinite(sigma) & (target != 0.0) & (sigma > 0.0)
    positive = finite & (baseline > 0.0) & (target > 0.0)
    ratio = np.full_like(target, np.nan, dtype=float)
    ratio[finite] = baseline[finite] / target[finite]
    data_over_w = np.full_like(target, np.nan, dtype=float)
    data_over_w[positive] = target[positive] / baseline[positive]

    out: Dict[str, Any] = {
        "n": int(len(target)),
        "n_finite": int(np.sum(finite)),
        "n_positive_baseline": int(np.sum(positive)),
        "n_nonpositive_baseline": int(np.sum(finite & (baseline <= 0.0))),
        "median_w_over_data": float(np.nanmedian(ratio[finite])) if finite.any() else float("nan"),
        "median_data_over_w_positive": float(np.nanmedian(data_over_w[positive])) if positive.any() else float("nan"),
    }
    med = out["median_w_over_data"]
    out["inverse_median_w_over_data"] = float(1.0 / med) if np.isfinite(med) and med != 0.0 else float("nan")
    if finite.any():
        w = 1.0 / np.maximum(sigma[finite], 1e-300) ** 2
        b = baseline[finite]
        y = target[finite]
        den = float(np.sum(w * b * b))
        num = float(np.sum(w * b * y))
        out["weighted_ls_scale"] = float(num / den) if den > 0.0 else float("nan")
    else:
        out["weighted_ls_scale"] = float("nan")

    per_dataset = []
    for ds, sub in df.assign(_baseline=baseline, _target=target, _sigma=sigma).groupby("dataset", sort=False):
        b = sub["_baseline"].to_numpy(float)
        y = sub["_target"].to_numpy(float)
        sg = sub["_sigma"].to_numpy(float)
        f = np.isfinite(b) & np.isfinite(y) & np.isfinite(sg) & (y != 0.0) & (sg > 0.0)
        p = f & (b > 0.0) & (y > 0.0)
        r = b[f] / y[f] if f.any() else np.array([], dtype=float)
        wls = float("nan")
        if f.any():
            ww = 1.0 / np.maximum(sg[f], 1e-300) ** 2
            den = float(np.sum(ww * b[f] * b[f]))
            num = float(np.sum(ww * b[f] * y[f]))
            if den > 0.0:
                wls = num / den
        per_dataset.append({
            "dataset": str(ds),
            "n": int(len(sub)),
            "n_positive_baseline": int(np.sum(p)),
            "median_w_over_data": float(np.nanmedian(r)) if f.any() else float("nan"),
            "inverse_median_w_over_data": float(1.0 / np.nanmedian(r)) if f.any() and np.nanmedian(r) != 0.0 else float("nan"),
            "median_data_over_w_positive": float(np.nanmedian(y[p] / b[p])) if p.any() else float("nan"),
            "weighted_ls_scale": float(wls),
        })
    out["per_dataset"] = per_dataset
    return out


def choose_auto_global_norm(diag: Mapping[str, Any], method: str) -> float:
    method = str(method).lower()
    key_map = {
        "inverse_median": "inverse_median_w_over_data",
        "median_data_over_w": "median_data_over_w_positive",
        "weighted_ls": "weighted_ls_scale",
    }
    if method not in key_map:
        raise ValueError(f"Unknown auto global norm method {method!r}")
    val = float(diag.get(key_map[method], float("nan")))
    if not np.isfinite(val) or val <= 0.0:
        # Fall back in a deterministic order.
        for key in ("inverse_median_w_over_data", "weighted_ls_scale", "median_data_over_w_positive"):
            val = float(diag.get(key, float("nan")))
            if np.isfinite(val) and val > 0.0:
                return val
        raise RuntimeError("Could not determine a finite positive auto global normalization initialization")
    return val



def soft_log_multiplier_np(
    q: np.ndarray,
    *,
    q0: float,
    cs_log: str = "lnQ",
    cs_kernel_convention: str = "pair",
) -> np.ndarray:
    """Return the logarithmic factor multiplying the learned nonperturbative CS kernel.

    The training model defines the pair-level soft evolution as

        exp[- G_CS(b) * L_Q]

    when --cs-kernel-convention=pair.  With --cs-kernel-convention=single_tmd,
    G_CS is interpreted as a single-TMD kernel and the DY pair receives a factor
    of two.

    --cs-log=lnQ  uses L_Q = ln(Q/Q0), preserving the older --learn-gk behavior.
    --cs-log=lnQ2 uses L_Q = ln(Q^2/Q0^2) = 2 ln(Q/Q0).
    """
    q = np.asarray(q, dtype=float)
    logq = np.log(np.clip(q / max(float(q0), 1.0e-12), 1.0e-12, None))
    cs_log = str(cs_log).lower()
    if cs_log == "lnq":
        lq = logq
    elif cs_log == "lnq2":
        lq = 2.0 * logq
    else:
        raise ValueError("cs_log must be 'lnQ' or 'lnQ2'")
    conv = str(cs_kernel_convention).lower()
    if conv == "pair":
        factor = 1.0
    elif conv == "single_tmd":
        factor = 2.0
    else:
        raise ValueError("cs_kernel_convention must be 'pair' or 'single_tmd'")
    return factor * lq


def soft_log_multiplier_torch(
    q: torch.Tensor,
    *,
    q0: float,
    cs_log: str = "lnQ",
    cs_kernel_convention: str = "pair",
) -> torch.Tensor:
    """Torch analogue of soft_log_multiplier_np."""
    logq = torch.log(torch.clamp(q / float(q0), min=1.0e-12))
    cs_log = str(cs_log).lower()
    if cs_log == "lnq":
        lq = logq
    elif cs_log == "lnq2":
        lq = 2.0 * logq
    else:
        raise ValueError("cs_log must be 'lnQ' or 'lnQ2'")
    conv = str(cs_kernel_convention).lower()
    if conv == "pair":
        factor = 1.0
    elif conv == "single_tmd":
        factor = 2.0
    else:
        raise ValueError("cs_kernel_convention must be 'pair' or 'single_tmd'")
    return float(factor) * lq


def initial_bounded_gk(
    b_grid: np.ndarray,
    *,
    learn_gk: bool,
    gk_b0: float,
    gk_mode: str = "bounded",
    gk_bmax: float = 0.08,
) -> np.ndarray:
    """Analytic initialization for the learned nonperturbative CS kernel."""
    b = np.asarray(b_grid, dtype=float)
    if not learn_gk:
        return np.zeros_like(b)
    b0 = max(float(gk_b0), 0.0)
    if str(gk_mode).lower() == "bounded":
        bcoef = min(max(b0, 0.0), max(float(gk_bmax), 1.0e-300))
    else:
        bcoef = b0
    return (b ** 2) * bcoef


def initial_np_prediction(
    *,
    kernel: np.ndarray,
    b_grid: np.ndarray,
    df: pd.DataFrame,
    y_term: np.ndarray,
    np_a0: float,
    np_min_a: float,
    learn_gk: bool,
    gk_b0: float,
    q0: float,
    gk_mode: str = "bounded",
    gk_bmax: float = 0.08,
    cs_log: str = "lnQ",
    cs_kernel_convention: str = "pair",
) -> np.ndarray:
    """Return W+Y at the analytic initialization of F_NP and gK.

    The FiLM A-head and gK-head are initialized with zero final weights and
    calibrated softplus biases, so initially A(x,b)=np_a0+np_min_a and, if
    enabled, gK(b)=b^2*gk_b0.  This diagnostic is much more useful than the
    bare W baseline for setting a starting global norm, because the undamped
    Bessel integral can be highly oscillatory.
    """
    b = np.asarray(b_grid, dtype=float)
    k = np.asarray(kernel, dtype=float)
    a_init = max(float(np_a0) + float(np_min_a), 0.0)
    fnp_pair = np.exp(-2.0 * a_init * b[None, :] ** 2)
    evol = 1.0
    if learn_gk:
        q = df["QM"].to_numpy(float)
        logq = soft_log_multiplier_np(
            q,
            q0=float(q0),
            cs_log=str(cs_log),
            cs_kernel_convention=str(cs_kernel_convention),
        )[:, None]
        gk = initial_bounded_gk(
            b,
            learn_gk=bool(learn_gk),
            gk_b0=float(gk_b0),
            gk_mode=str(gk_mode),
            gk_bmax=float(gk_bmax),
        )[None, :]
        evol = np.exp(-gk * logq)
    return np.sum(k * fnp_pair * evol, axis=1) + np.asarray(y_term, dtype=float)


def chi2_like_np(pred: np.ndarray, target: np.ndarray, sigma: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    m = np.isfinite(pred) & np.isfinite(target) & np.isfinite(sigma) & (sigma > 0.0)
    if not np.any(m):
        return float("nan")
    return float(np.mean(((pred[m] - target[m]) / np.maximum(sigma[m], 1e-300)) ** 2))


def initial_damping_scan(
    *,
    df: pd.DataFrame,
    kernel: np.ndarray,
    b_grid: np.ndarray,
    y_term: np.ndarray,
    target: np.ndarray,
    sigma: np.ndarray,
    a0_values: Sequence[float],
    np_min_a: float,
    learn_gk: bool,
    gk_b0: float,
    q0: float,
    gk_mode: str = "bounded",
    gk_bmax: float = 0.08,
    cs_log: str = "lnQ",
    cs_kernel_convention: str = "pair",
) -> pd.DataFrame:
    rows = []
    for a0 in a0_values:
        pred = initial_np_prediction(
            kernel=kernel,
            b_grid=b_grid,
            df=df,
            y_term=y_term,
            np_a0=float(a0),
            np_min_a=float(np_min_a),
            learn_gk=bool(learn_gk),
            gk_b0=float(gk_b0),
            q0=float(q0),
            gk_mode=str(gk_mode),
            gk_bmax=float(gk_bmax),
            cs_log=str(cs_log),
            cs_kernel_convention=str(cs_kernel_convention),
        )
        diag = backend_scale_diagnostics(df=df, baseline=pred, target=target, sigma=sigma)
        finite = np.isfinite(pred) & np.isfinite(target) & (target != 0.0)
        ratio = pred[finite] / target[finite] if np.any(finite) else np.array([], dtype=float)
        wls = float(diag.get("weighted_ls_scale", float("nan")))
        invmed = float(diag.get("inverse_median_w_over_data", float("nan")))
        rows.append({
            "np_a0": float(a0),
            "n_nonpositive": int(diag.get("n_nonpositive_baseline", 0)),
            "n_positive": int(diag.get("n_positive_baseline", 0)),
            "median_w_over_data": float(diag.get("median_w_over_data", float("nan"))),
            "inverse_median_scale": invmed,
            "weighted_ls_scale": wls,
            "chi2_scale1": chi2_like_np(pred, target, sigma),
            "chi2_weighted_ls": chi2_like_np(wls * pred, target, sigma) if np.isfinite(wls) and wls > 0 else float("nan"),
            "chi2_inverse_median": chi2_like_np(invmed * pred, target, sigma) if np.isfinite(invmed) and invmed > 0 else float("nan"),
            "min_w_over_data": float(np.nanmin(ratio)) if ratio.size else float("nan"),
            "max_w_over_data": float(np.nanmax(ratio)) if ratio.size else float("nan"),
        })
    return pd.DataFrame(rows)


def make_experimental_replica(df_with_unc: pd.DataFrame, cfg: ReplicaConfig, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    y = df_with_unc[cfg.observable].to_numpy(float)
    sigma = df_with_unc["sigma_uncorr"].to_numpy(float)
    replica = y.copy()
    for ds, sub in df_with_unc.groupby("dataset", sort=False):
        idx = sub.index.to_numpy()
        norm_rel = float(sub["norm_rel"].iloc[0]) if len(sub) else 0.0
        norm = 1.0 + rng.normal(0.0, 1.0) * norm_rel
        replica[idx] = norm * y[idx]
    replica = replica + rng.normal(0.0, 1.0, size=len(df_with_unc)) * sigma
    return replica


# -----------------------------------------------------------------------------
# External W/Y grid loading
# -----------------------------------------------------------------------------


def load_y_grid(data_df: pd.DataFrame, path: str | Path, *, allow_zero_for_nonfinite: bool = False) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    tab = pd.read_csv(path)
    if "row_id" not in tab.columns:
        raise ValueError("Y grid must contain row_id")
    tab = tab.drop_duplicates("row_id", keep="last").set_index("row_id")
    y_vals = np.full(len(data_df), np.nan, dtype=float)
    missing: list[str] = []
    nonfinite: list[str] = []
    for i, rid in enumerate(data_df["row_id"].astype(str)):
        if rid not in tab.index:
            missing.append(rid)
            continue
        row = tab.loc[rid]
        val = np.nan
        if "Y_CS" in tab.columns:
            try:
                val = float(row["Y_CS"])
            except Exception:
                val = np.nan
        if not np.isfinite(val) and {"FO_CS", "singular_CS"}.issubset(tab.columns):
            try:
                val = float(row["FO_CS"]) - float(row["singular_CS"])
            except Exception:
                val = np.nan
        if not np.isfinite(val):
            if allow_zero_for_nonfinite:
                val = 0.0
            else:
                nonfinite.append(rid)
        y_vals[i] = val
    if missing:
        raise ValueError(f"Y grid missing {len(missing)} selected row_ids; first missing {missing[:10]}")
    if nonfinite:
        raise ValueError(
            "Y grid has blank/non-finite values. This usually means a template was passed. "
            f"First affected rows: {nonfinite[:10]}"
        )
    bad = ~np.isfinite(y_vals)
    if bad.any():
        rows = data_df.loc[bad, "row_id"].head(10).tolist()
        raise ValueError(f"Y grid produced non-finite values for rows: {rows}")
    return y_vals


def load_external_w_grid(data_row_ids: Sequence[str], path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    tab = pd.read_csv(path)
    required = {"row_id", "bT", "Wpert_CS"}
    missing = sorted(required.difference(tab.columns))
    if missing:
        raise ValueError(f"External W grid missing columns: {missing}")
    data_row_ids = [str(x) for x in data_row_ids]
    tab = tab.loc[tab["row_id"].astype(str).isin(data_row_ids)].copy()
    if tab.empty:
        raise ValueError("External W grid has no row_ids matching the selected data.")
    b_vals = np.sort(tab["bT"].astype(float).unique())
    if len(b_vals) < 3:
        raise ValueError("External W grid needs at least 3 bT nodes")
    row_to_i = {rid: i for i, rid in enumerate(data_row_ids)}
    b_to_j = {float(b): j for j, b in enumerate(b_vals)}
    matrix = np.full((len(data_row_ids), len(b_vals)), np.nan, dtype=float)
    filled = np.zeros_like(matrix, dtype=bool)
    bad_rows: list[str] = []
    for r in tab.itertuples(index=False):
        rid = str(getattr(r, "row_id"))
        i = row_to_i.get(rid)
        if i is None:
            continue
        j = b_to_j[float(getattr(r, "bT"))]
        try:
            val = float(getattr(r, "Wpert_CS"))
        except Exception:
            val = np.nan
        matrix[i, j] = val
        filled[i, j] = True
        if not np.isfinite(val) and len(bad_rows) < 10:
            bad_rows.append(rid)
    missing_rows = [data_row_ids[i] for i in range(len(data_row_ids)) if not filled[i].all()]
    if missing_rows:
        raise ValueError(f"External W grid missing bT entries for {len(missing_rows)} rows; first {missing_rows[:10]}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(
            "External W grid has blank/non-finite Wpert_CS values. "
            f"First affected rows: {bad_rows[:10]}"
        )
    return b_vals, matrix


# -----------------------------------------------------------------------------
# Torch model pieces
# -----------------------------------------------------------------------------


def dtype_from_string(name: str) -> torch.dtype:
    name = str(name).lower()
    if name in ("float64", "double"):
        return torch.float64
    if name in ("float32", "single"):
        return torch.float32
    raise ValueError(f"Unknown dtype: {name}")


def _softplus_inverse(y: float) -> float:
    y = max(float(y), 1e-12)
    return math.log(math.expm1(y))


class FiLMBlock(nn.Module):
    def __init__(self, width: int, cond_width: int, *, dtype: torch.dtype | None = None):
        super().__init__()
        self.lin1 = nn.Linear(width, width, dtype=dtype)
        self.lin2 = nn.Linear(width, width, dtype=dtype)
        self.to_gamma = nn.Linear(cond_width, width, dtype=dtype)
        self.to_beta = nn.Linear(cond_width, width, dtype=dtype)

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        u = torch.tanh(self.lin1(h))
        gamma = F.softplus(self.to_gamma(c)).unsqueeze(1) + 1e-6
        beta = torch.tanh(self.to_beta(c)).unsqueeze(1)
        return torch.tanh(self.lin2(gamma * u + beta) + h)


class FilmNPFactor(nn.Module):
    r"""Shared nonperturbative b-space factor.

    direct mode:
        F_NP(x,b)=exp[-b^2 A_theta(x,b)], A_theta >= 0.
        This preserves F_NP(x,0)=1 and 0<F_NP<=1, but it does not strictly
        enforce monotonicity in b because A_theta may decrease with b.

    monotone mode:
        F_NP(x,b)=exp[- integral_0^b db' 2 b' A_theta(x,b')], A_theta >= 0.
        This keeps the same Gaussian initialization for constant A=a0 while
        guaranteeing dF_NP/db <= 0 on the supplied ordered b grid.
    """

    def __init__(
        self,
        *,
        width: int = 48,
        cond_width: int = 32,
        n_blocks: int = 3,
        a0: float = 0.08,
        min_a: float = 0.0,
        a_mode: str = "positive",
        exponent_clip: float = 40.0,
        shape_mode: str = "direct",
        a_smooth_sigma: float = 0.0,
        a_tail_amp: float = 0.0,
        a_tail_b0: float = 3.5,
        a_tail_width: float = 0.25,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.min_a = float(min_a)
        self.a_mode = str(a_mode).lower()
        self.shape_mode = str(shape_mode).lower()
        self.a_smooth_sigma = float(a_smooth_sigma)
        self.a_tail_amp = float(a_tail_amp)
        self.a_tail_b0 = float(a_tail_b0)
        self.a_tail_width = float(a_tail_width)
        if self.a_smooth_sigma < 0.0:
            raise ValueError("a_smooth_sigma must be nonnegative")
        if self.a_tail_amp < 0.0:
            raise ValueError("a_tail_amp must be nonnegative")
        if self.a_tail_width <= 0.0:
            raise ValueError("a_tail_width must be positive")
        if self.a_mode not in ("positive", "signed"):
            raise ValueError("a_mode must be 'positive' or 'signed'")
        if self.shape_mode not in ("direct", "monotone"):
            raise ValueError("shape_mode must be 'direct' or 'monotone'")
        if self.shape_mode == "monotone" and self.a_mode != "positive":
            raise ValueError("monotone F_NP requires --np-a-mode positive")
        self.exponent_clip = float(exponent_clip)
        self.radial = nn.Linear(4, width, dtype=dtype)
        self.cond = nn.Sequential(
            nn.Linear(2, cond_width, dtype=dtype), nn.SiLU(),
            nn.Linear(cond_width, cond_width, dtype=dtype), nn.SiLU(),
        )
        self.blocks = nn.ModuleList([FiLMBlock(width, cond_width, dtype=dtype) for _ in range(int(n_blocks))])
        self.head = nn.Linear(width, 1, dtype=dtype)
        nn.init.zeros_(self.head.weight)
        if self.a_mode == "positive":
            nn.init.constant_(self.head.bias, _softplus_inverse(max(float(a0), 1.0e-12)))
        else:
            # Signed mode treats the head output directly as A.
            nn.init.constant_(self.head.bias, float(a0))

    @staticmethod
    def radial_features(b: torch.Tensor) -> torch.Tensor:
        b = torch.clamp(b, min=0.0)
        return torch.stack([b, b.square(), torch.sqrt(b + 1e-8), torch.log1p(b)], dim=-1)

    @staticmethod
    def condition_features(x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, 1e-6, 1.0 - 1e-6)
        return torch.stack([x, torch.log(x / (1.0 - x))], dim=-1)

    def A_raw(self, x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if x.ndim != 1:
            raise ValueError("x must have shape [batch]")
        b = b.to(dtype=x.dtype, device=x.device)
        b2 = b.unsqueeze(0).expand(x.numel(), -1)
        h = torch.tanh(self.radial(self.radial_features(b2)))
        c = self.cond(self.condition_features(x))
        for block in self.blocks:
            h = block(h, c)
        raw = self.head(h).squeeze(-1)
        if self.a_mode == "signed":
            return raw
        return F.softplus(raw) + self.min_a

    @staticmethod
    def _gaussian_smoother_matrix(
        b: torch.Tensor,
        sigma: float,
    ) -> torch.Tensor:
        """Normalized physical-b Gaussian convolution matrix.

        The quadrature weights make this a convolution in b_T rather than
        index space. Row normalization handles the finite-grid boundaries.
        """
        if b.numel() < 2 or float(sigma) <= 0.0:
            return torch.eye(
                b.numel(),
                dtype=b.dtype,
                device=b.device,
            )

        db = b[1:] - b[:-1]
        quad = torch.cat(
            [
                0.5 * db[:1],
                0.5 * (b[2:] - b[:-2]),
                0.5 * db[-1:],
            ],
            dim=0,
        )

        delta = b.reshape(-1, 1) - b.reshape(1, -1)
        kernel = torch.exp(
            -0.5 * (delta / float(sigma)).square()
        )
        kernel = kernel * quad.reshape(1, -1)

        return kernel / torch.clamp(
            torch.sum(kernel, dim=1, keepdim=True),
            min=1.0e-30,
        )

    def A(self, x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        raw_A = self.A_raw(x, b)

        if self.a_smooth_sigma > 0.0 and b.numel() >= 2:
            smoother = self._gaussian_smoother_matrix(
                b,
                self.a_smooth_sigma,
            )
            A_value = torch.matmul(
                raw_A,
                smoother.transpose(0, 1),
            )
        else:
            A_value = raw_A

        # Fixed, smooth late-b damping floor.  It is negligible in the
        # perturbative/small-b region and enforces a decaying single-TMD tail.
        if self.a_tail_amp > 0.0:
            gate = torch.sigmoid(
                (b - self.a_tail_b0) / self.a_tail_width
            )
            A_value = A_value + self.a_tail_amp * gate.reshape(1, -1)

        return A_value

    @staticmethod
    def _cumulative_trapezoid(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # y has shape [batch, nb], x has shape [nb].  Return integral from x[0].
        if x.numel() < 2:
            return torch.zeros_like(y)
        dx = x[1:] - x[:-1]
        area = 0.5 * (y[:, 1:] + y[:, :-1]) * dx.unsqueeze(0)
        zero = torch.zeros((y.shape[0], 1), dtype=y.dtype, device=y.device)
        return torch.cat([zero, torch.cumsum(area, dim=1)], dim=1)

    def forward(self, x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        b = b.to(dtype=x.dtype, device=x.device)
        b2 = b.unsqueeze(0).expand(x.numel(), -1)
        A = self.A(x, b)
        if self.shape_mode == "monotone":
            # Constant A=a0 gives exponent approximately -a0*b^2, matching
            # the old Gaussian initialization, while enforcing monotonicity.
            integrand = 2.0 * b2 * A
            exponent = -self._cumulative_trapezoid(integrand, b)
        else:
            exponent = -b2.square() * A
        if self.exponent_clip > 0:
            exponent = torch.clamp(exponent, -self.exponent_clip, self.exponent_clip)
        return torch.exp(exponent)


class GKModel(nn.Module):
    r"""Unbounded pilot model: g_K(b)=b^2 B_theta(b), B_theta >= 0.

    This is kept for diagnostics, but it is too flexible for fixed-target-only
    central fits: with Q0 below all fitted Q values, it can mimic an arbitrary
    large-b damping factor and become degenerate with F_NP.
    """

    def __init__(self, *, width: int = 24, n_layers: int = 2, b0: float = 0.02, dtype: torch.dtype | None = None):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(4, width, dtype=dtype), nn.SiLU()]
        for _ in range(max(0, int(n_layers) - 1)):
            layers += [nn.Linear(width, width, dtype=dtype), nn.SiLU()]
        layers.append(nn.Linear(width, 1, dtype=dtype))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, _softplus_inverse(b0))

    @staticmethod
    def features(b: torch.Tensor) -> torch.Tensor:
        b = torch.clamp(b, min=0.0)
        return torch.stack([b, b.square(), torch.sqrt(b + 1e-8), torch.log1p(b)], dim=-1)

    def forward(self, b: torch.Tensor) -> torch.Tensor:
        flat = b.reshape(-1)
        B = F.softplus(self.net(self.features(flat)).squeeze(-1))
        return (flat.square() * B).reshape(b.shape)


class BoundedGKModel(nn.Module):
    r"""Safer pilot model: g_K(b)=b^2 B_theta(b), 0 <= B_theta <= B_max.

    The cap is on the quadratic coefficient B_theta, not directly on g_K.
    For example B_max=0.08 gives g_K(8 GeV^-1) <= 5.12, preventing the
    enormous large-b values that the unbounded network can learn while still
    allowing visible nonperturbative Collins-Soper evolution.
    """

    def __init__(
        self,
        *,
        width: int = 24,
        n_layers: int = 2,
        b0: float = 1.0e-4,
        bmax: float = 0.08,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.bmax = float(bmax)
        if self.bmax <= 0.0:
            raise ValueError("BoundedGKModel requires bmax > 0")
        frac = min(max(float(b0) / self.bmax, 1.0e-8), 1.0 - 1.0e-8)
        layers: list[nn.Module] = [nn.Linear(4, width, dtype=dtype), nn.SiLU()]
        for _ in range(max(0, int(n_layers) - 1)):
            layers += [nn.Linear(width, width, dtype=dtype), nn.SiLU()]
        layers.append(nn.Linear(width, 1, dtype=dtype))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, math.log(frac / (1.0 - frac)))

    @staticmethod
    def features(b: torch.Tensor) -> torch.Tensor:
        b = torch.clamp(b, min=0.0)
        return torch.stack([b, b.square(), torch.sqrt(b + 1e-8), torch.log1p(b)], dim=-1)

    def coefficient_B(self, b: torch.Tensor) -> torch.Tensor:
        flat = b.reshape(-1)
        return self.bmax * torch.sigmoid(self.net(self.features(flat)).squeeze(-1))

    def forward(self, b: torch.Tensor) -> torch.Tensor:
        flat = b.reshape(-1)
        B = self.coefficient_B(flat)
        return (flat.square() * B).reshape(b.shape)


class ZeroGK(nn.Module):
    def forward(self, b: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(b)


class OptionalGlobalNorm(nn.Module):
    """Optional global multiplicative normalization for development diagnostics."""

    def __init__(self, *, enabled: bool, init: float = 1.0, dtype: torch.dtype | None = None):
        super().__init__()
        self.enabled = bool(enabled)
        if self.enabled:
            self.log_norm = nn.Parameter(torch.tensor(math.log(max(float(init), 1e-12)), dtype=dtype))
        else:
            self.register_buffer("log_norm", torch.tensor(0.0, dtype=dtype))

    def forward(self) -> torch.Tensor:
        return torch.exp(self.log_norm)




class DatasetNormNuisance(nn.Module):
    """Gaussian-constrained correlated normalization factors, one per dataset.

    The factor for dataset d is scale_d = exp(log_scale_d).  Its Gaussian pull is
    (scale_d - 1) / delta_d, where delta_d is the relative normalization
    uncertainty.  The penalty returned by penalty(normalize_by=N) is
    sum_d pull_d^2 / N so the training loss remains approximately chi2 per point.
    """

    def __init__(
        self,
        *,
        names: Sequence[str],
        deltas: Sequence[float],
        enabled: bool,
        dtype: torch.dtype | None = None,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.names = [str(x) for x in names]
        self.enabled = bool(enabled)
        d = torch.tensor(np.asarray(deltas, dtype=float), dtype=dtype, device=device)
        self.register_buffer("deltas", torch.clamp(d, min=1.0e-12))
        if self.enabled:
            self.log_scales = nn.Parameter(torch.zeros(len(self.names), dtype=dtype, device=device))
        else:
            self.register_buffer("log_scales", torch.zeros(len(self.names), dtype=dtype, device=device))

    def scales(self) -> torch.Tensor:
        return torch.exp(self.log_scales)

    def forward(self, values: torch.Tensor, dataset_index: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return values
        return values * self.scales().to(values.device, values.dtype)[dataset_index]

    def penalty(self, *, normalize_by: int) -> torch.Tensor:
        if not self.enabled:
            return torch.zeros((), dtype=self.log_scales.dtype, device=self.log_scales.device)
        pull = (self.scales() - 1.0) / self.deltas.to(self.log_scales.device, self.log_scales.dtype)
        return torch.sum(pull.square()) / max(int(normalize_by), 1)

    def as_dict(self) -> Dict[str, float]:
        vals = self.scales().detach().cpu().numpy().tolist()
        return {name: float(val) for name, val in zip(self.names, vals)}

    def pulls_dict(self) -> Dict[str, float]:
        vals = ((self.scales() - 1.0) / self.deltas).detach().cpu().numpy().tolist()
        return {name: float(val) for name, val in zip(self.names, vals)}


class PrecomputedKernelModel(nn.Module):
    """DNN NP factor multiplying a fixed precomputed b-space W kernel.

    kernel_matrix[row,b] already contains the trapezoid weight, b, J0(qT*b), and
    Wpert_CS(row,b).  The forward pass only multiplies by the trainable NP factors
    and sums over b.
    """

    def __init__(
        self,
        *,
        b_grid: np.ndarray,
        kernel_matrix: np.ndarray,
        np_factor: nn.Module,
        gk_model: nn.Module,
        q0: float = 2.0,
        cs_log: str = "lnQ",
        cs_kernel_convention: str = "pair",
        learn_global_norm: bool = False,
        global_norm_init: float = 1.0,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        b = torch.tensor(np.asarray(b_grid, dtype=float), dtype=dtype, device=device)
        km = torch.tensor(np.asarray(kernel_matrix, dtype=float), dtype=dtype, device=device)
        self.register_buffer("b", b)
        self.register_buffer("kernel_matrix", km)
        self.np_factor = np_factor
        self.gk_model = gk_model
        self.q0 = float(q0)
        self.cs_log = str(cs_log)
        self.cs_kernel_convention = str(cs_kernel_convention)
        self.global_norm = OptionalGlobalNorm(enabled=learn_global_norm, init=global_norm_init, dtype=dtype)

    def sigma_w(self, row_index: torch.Tensor, q: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        b = self.b.to(dtype=q.dtype, device=q.device)
        kernel = self.kernel_matrix[row_index].to(dtype=q.dtype, device=q.device)
        fnp1 = self.np_factor(x1, b)
        fnp2 = self.np_factor(x2, b)
        gk = self.gk_model(b).unsqueeze(0).to(dtype=q.dtype, device=q.device)
        logq = soft_log_multiplier_torch(
            q,
            q0=self.q0,
            cs_log=self.cs_log,
            cs_kernel_convention=self.cs_kernel_convention,
        ).unsqueeze(-1)
        evol_np = torch.exp(-gk * logq)
        return self.global_norm() * torch.sum(kernel * fnp1 * fnp2 * evol_np, dim=-1)

    def forward(self, row_index: torch.Tensor, q: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor, y_term: torch.Tensor) -> torch.Tensor:
        return self.sigma_w(row_index, q, x1, x2) + y_term


# -----------------------------------------------------------------------------
# Tensor data and training
# -----------------------------------------------------------------------------


class TensorData:
    def __init__(
        self,
        df: pd.DataFrame,
        *,
        y_term: np.ndarray,
        target: np.ndarray | None,
        dtype: torch.dtype,
        device: torch.device | str,
    ):
        self.df = df.reset_index(drop=True).copy()
        n = len(self.df)
        self.row_index = torch.arange(n, dtype=torch.long, device=device)
        self.qT = torch.tensor(self.df["qT"].to_numpy(float), dtype=dtype, device=device)
        self.Q = torch.tensor(self.df["QM"].to_numpy(float), dtype=dtype, device=device)
        self.x1 = torch.tensor(self.df["x1"].to_numpy(float), dtype=dtype, device=device)
        self.x2 = torch.tensor(self.df["x2"].to_numpy(float), dtype=dtype, device=device)
        y_np = target if target is not None else self.df["CS"].to_numpy(float)
        self.target = torch.tensor(y_np, dtype=dtype, device=device)
        sigma = np.maximum(self.df["sigma_uncorr"].to_numpy(float), 1e-12)
        self.sigma = torch.tensor(sigma, dtype=dtype, device=device)
        self.y_term = torch.tensor(y_term, dtype=dtype, device=device)
        self.dataset_names = list(dict.fromkeys(self.df["dataset"].astype(str).tolist()))
        ds_to_i = {name: i for i, name in enumerate(self.dataset_names)}
        ds_idx = self.df["dataset"].astype(str).map(ds_to_i).to_numpy(int)
        self.dataset_index = torch.tensor(ds_idx, dtype=torch.long, device=device)
        rels = []
        for name in self.dataset_names:
            vals = self.df.loc[self.df["dataset"].astype(str) == name, "norm_rel"].to_numpy(float)
            rels.append(float(vals[0]) if len(vals) else 0.0)
        self.dataset_norm_rel = np.asarray(rels, dtype=float)

    def __len__(self) -> int:
        return len(self.df)

    def batch(self, idx: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "row_index": self.row_index[idx],
            "qT": self.qT[idx],
            "Q": self.Q[idx],
            "x1": self.x1[idx],
            "x2": self.x2[idx],
            "target": self.target[idx],
            "sigma": self.sigma[idx],
            "y_term": self.y_term[idx],
            "dataset_index": self.dataset_index[idx],
        }


def scaled_mse(pred: torch.Tensor, target: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return torch.mean(((pred - target) / sigma) ** 2)


def _safe_unique_x_grid(data: TensorData, *, max_points: int = 80) -> torch.Tensor:
    """Return a sorted x-grid from the fitted support, thinned if needed.

    The curvature penalties are diagnostics/regularizers, not the primary loss.
    They should not explode cost on large data sets, so we thin to quantiles when
    the combined x1/x2 support is large.
    """
    x = torch.unique(torch.cat([data.x1.detach(), data.x2.detach()]))
    x = torch.sort(torch.clamp(x, 1.0e-6, 1.0 - 1.0e-6)).values
    if x.numel() <= max_points:
        return x
    # Quantile-like thinning by index while preserving endpoints.
    idx = torch.linspace(0, x.numel() - 1, max_points, device=x.device).round().long()
    return torch.unique(x[idx])


def _second_diff_nonuniform(y: torch.Tensor, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Second derivative-like finite difference on a nonuniform 1D grid.

    y may be [nx, nb].  If dim=-1, x is the b-grid and the result is over b.
    If dim=0, x is the x/logit-x grid and the result is over x.
    """
    if x.numel() < 3:
        return torch.zeros_like(y.narrow(dim, 0, 0))
    if dim < 0:
        dim = y.ndim + dim
    ym = y.narrow(dim, 0, y.shape[dim] - 2)
    y0 = y.narrow(dim, 1, y.shape[dim] - 2)
    yp = y.narrow(dim, 2, y.shape[dim] - 2)
    xm = x[:-2]
    x0 = x[1:-1]
    xp = x[2:]
    dxm = torch.clamp(x0 - xm, min=1.0e-12)
    dxp = torch.clamp(xp - x0, min=1.0e-12)
    # reshape grid spacings for broadcasting on requested dimension
    shape = [1] * y.ndim
    shape[dim] = -1
    dxm = dxm.reshape(shape)
    dxp = dxp.reshape(shape)
    slope_l = (y0 - ym) / dxm
    slope_r = (yp - y0) / dxp
    return 2.0 * (slope_r - slope_l) / (dxm + dxp)


def _reg_multiplier_for_epoch(epoch: int, cfg: "TrainConfig") -> float:
    """Cosine backoff multiplier for removable regularizers."""
    start = max(0.0, min(1.0, float(cfg.reg_backoff_start_frac))) * float(cfg.epochs)
    end = max(0.0, min(1.0, float(cfg.reg_backoff_end_frac))) * float(cfg.epochs)
    final = float(cfg.reg_final_scale)
    if end <= start:
        return final if epoch >= end else 1.0
    if epoch <= start:
        return 1.0
    if epoch >= end:
        return final
    t = (float(epoch) - start) / (end - start)
    # Smooth cosine interpolation from 1 to final.
    return final + (1.0 - final) * 0.5 * (1.0 + math.cos(math.pi * t))


def regularization_loss(
    model: PrecomputedKernelModel,
    data: TensorData,
    *,
    lambda_a_l2: float = 0.0,
    lambda_gk_l2: float = 0.0,
    lambda_fnp_mono: float = 0.0,
    lambda_fnp_bcurv: float = 0.0,
    lambda_fnp_xcurv: float = 0.0,
    lambda_fnp_pair_bcurv: float = 0.0,
    lambda_fnp_local_bcurv: float = 0.0,
    fnp_local_bcurv_x_values: tuple[float, ...] = (
        0.15, 0.20, 0.30, 0.40, 0.50,
    ),
    fnp_local_bcurv_bmin: float = 0.5,
    fnp_local_bcurv_bmax: float = 3.5,
    lambda_fnp_lowpass: float = 0.0,
    fnp_lowpass_x_values: tuple[float, ...] = (
        0.15, 0.20, 0.30, 0.40, 0.50,
    ),
    fnp_lowpass_bmin: float = 0.5,
    fnp_lowpass_bmax: float = 3.5,
    fnp_lowpass_sigma: float = 0.30,
    lambda_fnp_ratecurv: float = 0.0,
    fnp_ratecurv_bmin: float = 0.25,
    fnp_ratecurv_bmax: float = 4.0,
    lambda_fnp_tail: float = 0.0,
    fnp_tail_bmin: float = 6.0,
    fnp_tail_target: float = 0.35,
    mono_tol: float = 1.0e-4,
) -> torch.Tensor:
    """Removable inverse-problem scaffolding for the bT-space extraction.

    v18 is deliberately modeled on the kT-space strategy: use strong smoothness
    and tail scaffolds to condition the inverse problem, then back them off.  The
    hard constraints should be carried by --np-shape-mode monotone; these lambda
    terms are optional and meant to be reduced/removed in ladder runs.
    """
    loss = torch.zeros((), dtype=data.Q.dtype, device=data.Q.device)
    if lambda_a_l2 > 0.0:
        # Use the selected data x support and b grid, but detach x values from any graph.
        a1 = model.np_factor.A(data.x1, model.b)
        a2 = model.np_factor.A(data.x2, model.b)
        loss = loss + float(lambda_a_l2) * 0.5 * (torch.mean(a1.square()) + torch.mean(a2.square()))
    if lambda_gk_l2 > 0.0:
        gk = model.gk_model(model.b)
        loss = loss + float(lambda_gk_l2) * torch.mean(gk.square())

    needs_fnp_grid = any(v > 0.0 for v in [
        lambda_fnp_mono,
        lambda_fnp_bcurv,
        lambda_fnp_xcurv,
        lambda_fnp_ratecurv,
        lambda_fnp_tail,
    ])
    x_support = None
    fnp = None
    logf = None
    if needs_fnp_grid:
        x_support = _safe_unique_x_grid(data)
        fnp = model.np_factor(x_support, model.b)
        logf = torch.log(torch.clamp(fnp, min=1.0e-30))

    if lambda_fnp_mono > 0.0 and fnp is not None:
        # Soft monotonicity prior for direct F_NP.  In monotone mode this should
        # be identically zero except for numerical noise.
        increases = fnp[:, 1:] - fnp[:, :-1] - float(mono_tol)
        loss = loss + float(lambda_fnp_mono) * torch.mean(torch.relu(increases).square())

    if lambda_fnp_bcurv > 0.0 and logf is not None:
        d2b = _second_diff_nonuniform(logf, model.b, dim=1)
        # Scale by a characteristic b range so lambda values are less grid-dependent.
        brange = torch.clamp(model.b[-1] - model.b[0], min=1.0)
        loss = loss + float(lambda_fnp_bcurv) * torch.mean((d2b * brange.square()).square())

    if lambda_fnp_ratecurv > 0.0 and x_support is not None:
        # For the monotone parameterization
        #
        #   F_NP(x,b) = exp[- integral_0^b 2 beta A_theta(x,beta) d beta],
        #
        # the exact continuum damping rate is
        #
        #   h(x,b) = -d_b log F_NP(x,b) = 2 b A_theta(x,b).
        #
        # Penalize curvature of h rather than its slope. This removes
        # localized spikes/knees while still allowing h to vary broadly.
        if not hasattr(model.np_factor, "A"):
            raise RuntimeError(
                "--lambda-fnp-ratecurv requires --np-shape-mode monotone "
                "with model.np_factor.A available."
            )

        rate_mask = (
            (model.b >= float(fnp_ratecurv_bmin))
            & (model.b <= float(fnp_ratecurv_bmax))
        )

        if int(torch.count_nonzero(rate_mask).item()) >= 3:
            b_rate = model.b[rate_mask]
            a_rate = model.np_factor.A(x_support, b_rate)
            h_rate = 2.0 * b_rate.reshape(1, -1) * a_rate

            d2h = _second_diff_nonuniform(h_rate, b_rate, dim=1)

            brange_rate = torch.clamp(
                b_rate[-1] - b_rate[0],
                min=1.0,
            )

            # Normalize by the detached RMS damping rate so the lambda has
            # a similar meaning across fits and x-support choices.
            hscale = torch.clamp(
                torch.sqrt(torch.mean(h_rate.detach().square())),
                min=0.25,
            )

            scaled_d2h = (
                d2h
                * brange_rate.square()
                / hscale
            )

            loss = loss + float(lambda_fnp_ratecurv) * torch.mean(
                scaled_d2h.square()
            )

    if lambda_fnp_local_bcurv > 0.0:
        # Localized version of the log-F_NP curvature scaffold.
        #
        # This deliberately matches the localized gradient audit:
        #
        #   x = supplied probe values
        #   bT in [bmin, bmax]
        #   same full-grid brange scaling as lambda_fnp_bcurv
        #
        # Therefore the lambda values inferred from that audit apply
        # directly to this term.
        x_probe = torch.tensor(
            tuple(float(v) for v in fnp_local_bcurv_x_values),
            dtype=data.Q.dtype,
            device=data.Q.device,
        )

        x_probe = torch.clamp(
            x_probe,
            min=1.0e-6,
            max=1.0 - 1.0e-6,
        )

        fnp_local = model.np_factor(x_probe, model.b)
        logf_local = torch.log(
            torch.clamp(fnp_local, min=1.0e-30)
        )

        d2_local = _second_diff_nonuniform(
            logf_local,
            model.b,
            dim=1,
        )

        if d2_local.shape[1] == model.b.numel() - 2:
            b_for_local = model.b[1:-1]
        else:
            b_for_local = model.b[:d2_local.shape[1]]

        local_mask = (
            (b_for_local >= float(fnp_local_bcurv_bmin))
            & (b_for_local <= float(fnp_local_bcurv_bmax))
        )

        if bool(torch.any(local_mask)):
            # Keep the same scaling convention as the global term so
            # the gradient-audit calibration remains valid.
            brange_global = torch.clamp(
                model.b[-1] - model.b[0],
                min=1.0,
            )

            local_density = (
                d2_local[:, local_mask]
                * brange_global.square()
            ).square()

            loss = loss + float(
                lambda_fnp_local_bcurv
            ) * torch.mean(local_density)

    if lambda_fnp_lowpass > 0.0:
        if not hasattr(model.np_factor, "A"):
            raise RuntimeError(
                "--lambda-fnp-lowpass requires --np-shape-mode monotone "
                "with model.np_factor.A available."
            )

        b_lp = model.b

        if b_lp.numel() < 3:
            raise RuntimeError(
                "The low-pass scaffold requires at least three bT points."
            )

        x_lp = torch.tensor(
            tuple(float(v) for v in fnp_lowpass_x_values),
            dtype=data.Q.dtype,
            device=data.Q.device,
        )

        x_lp = torch.clamp(
            x_lp,
            min=1.0e-6,
            max=1.0 - 1.0e-6,
        )

        # Exact damping rate of the monotone-integral parameterization.
        a_lp = model.np_factor.A(x_lp, b_lp)
        h_lp = 2.0 * b_lp.reshape(1, -1) * a_lp

        # Trapezoidal integration weights for a potentially nonuniform grid.
        db_lp = b_lp[1:] - b_lp[:-1]

        quad_lp = torch.cat(
            [
                0.5 * db_lp[:1],
                0.5 * (b_lp[2:] - b_lp[:-2]),
                0.5 * db_lp[-1:],
            ],
            dim=0,
        )

        sigma_lp = max(float(fnp_lowpass_sigma), 1.0e-6)

        # S_ij is a normalized Gaussian convolution kernel:
        #
        #   h_smooth(b_i) = sum_j S_ij h(b_j).
        #
        # Multiplication by quad_lp makes this an approximation to a
        # continuous convolution in physical bT, not an index-space filter.
        delta_lp = (
            b_lp.reshape(-1, 1)
            - b_lp.reshape(1, -1)
        )

        kernel_lp = torch.exp(
            -0.5 * (delta_lp / sigma_lp).square()
        )

        kernel_lp = (
            kernel_lp
            * quad_lp.reshape(1, -1)
        )

        smoother_lp = kernel_lp / torch.clamp(
            torch.sum(kernel_lp, dim=1, keepdim=True),
            min=1.0e-30,
        )

        h_smooth_lp = torch.matmul(
            h_lp,
            smoother_lp.transpose(0, 1),
        )

        mask_lp = (
            (b_lp >= float(fnp_lowpass_bmin))
            & (b_lp <= float(fnp_lowpass_bmax))
        )

        if bool(torch.any(mask_lp)):
            local_weights_lp = quad_lp[mask_lp]

            residual_lp = (
                h_lp[:, mask_lp]
                - h_smooth_lp[:, mask_lp]
            )

            numerator_lp = torch.sum(
                residual_lp.square()
                * local_weights_lp.reshape(1, -1)
            )

            denominator_lp = torch.clamp(
                float(x_lp.numel())
                * torch.sum(local_weights_lp),
                min=1.0e-30,
            )

            raw_lowpass_lp = numerator_lp / denominator_lp

            loss = (
                loss
                + float(lambda_fnp_lowpass)
                * raw_lowpass_lp
            )

    if lambda_fnp_xcurv > 0.0 and logf is not None and x_support is not None and x_support.numel() >= 3:
        z = torch.log(x_support / (1.0 - x_support))
        d2x = _second_diff_nonuniform(logf, z, dim=0)
        zrange = torch.clamp(z[-1] - z[0], min=1.0)
        loss = loss + float(lambda_fnp_xcurv) * torch.mean((d2x * zrange.square()).square())

    if lambda_fnp_pair_bcurv > 0.0:
        # Pair damping is closer to what DY directly constrains.  Penalize rough
        # b-dependence of log[F(x1,b)F(x2,b)] on actual data kinematics.
        f1 = model.np_factor(data.x1.detach(), model.b)
        f2 = model.np_factor(data.x2.detach(), model.b)
        logpair = torch.log(torch.clamp(f1 * f2, min=1.0e-30))
        d2p = _second_diff_nonuniform(logpair, model.b, dim=1)
        brange = torch.clamp(model.b[-1] - model.b[0], min=1.0)
        loss = loss + float(lambda_fnp_pair_bcurv) * torch.mean((d2p * brange.square()).square())

    if lambda_fnp_tail > 0.0 and fnp is not None:
        mask = model.b >= float(fnp_tail_bmin)
        if bool(torch.any(mask)):
            # Weak large-b scaffold: discourage undamped plateaus but do not force
            # a specific functional form.  This is meant to be removable.
            excess = torch.relu(fnp[:, mask] - float(fnp_tail_target))
            loss = loss + float(lambda_fnp_tail) * torch.mean(excess.square())
    return loss


@dataclass
class TrainConfig:
    epochs: int = 300
    batch_size: int = 128
    lr: float = 2e-3
    weight_decay: float = 0.0
    grad_clip: float = 10.0
    seed: int = 123
    patience: int = 0
    min_delta: float = 1e-7
    restore_best: bool = True
    log_every: int = 0
    lambda_a_l2: float = 0.0
    lambda_gk_l2: float = 0.0
    lambda_fnp_mono: float = 0.0
    lambda_fnp_bcurv: float = 0.0
    lambda_fnp_xcurv: float = 0.0
    lambda_fnp_pair_bcurv: float = 0.0
    lambda_fnp_local_bcurv: float = 0.0
    fnp_local_bcurv_x_values: tuple[float, ...] = (
        0.15, 0.20, 0.30, 0.40, 0.50,
    )
    fnp_local_bcurv_bmin: float = 0.5
    fnp_local_bcurv_bmax: float = 3.5
    lambda_fnp_lowpass: float = 0.0
    fnp_lowpass_x_values: tuple[float, ...] = (
        0.15, 0.20, 0.30, 0.40, 0.50,
    )
    fnp_lowpass_bmin: float = 0.5
    fnp_lowpass_bmax: float = 3.5
    fnp_lowpass_sigma: float = 0.30
    lambda_fnp_ratecurv: float = 0.0
    fnp_ratecurv_bmin: float = 0.25
    fnp_ratecurv_bmax: float = 4.0
    lambda_fnp_tail: float = 0.0
    fnp_tail_bmin: float = 6.0
    fnp_tail_target: float = 0.35
    mono_tol: float = 1.0e-4
    reg_backoff_start_frac: float = 1.0
    reg_backoff_end_frac: float = 1.0
    reg_final_scale: float = 1.0
    fit_dataset_norms: bool = False
    lambda_dataset_norm: float = 1.0
    dataset_norm_init_scales: tuple[float, ...] | None = None


def fit_model(model: PrecomputedKernelModel, data: TensorData, cfg: TrainConfig) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    torch.manual_seed(int(cfg.seed))
    n = len(data)
    batch_size = min(max(1, int(cfg.batch_size)), n)
    dataset_norms = DatasetNormNuisance(
        names=data.dataset_names,
        deltas=data.dataset_norm_rel,
        enabled=bool(cfg.fit_dataset_norms),
        dtype=data.Q.dtype,
        device=data.Q.device,
    )
    if cfg.fit_dataset_norms and cfg.dataset_norm_init_scales is not None:
        values = torch.tensor(
            tuple(float(v) for v in cfg.dataset_norm_init_scales),
            dtype=data.Q.dtype,
            device=data.Q.device,
        )
        if values.numel() != len(data.dataset_names):
            raise ValueError(
                "dataset_norm_init_scales length does not match dataset names"
            )
        with torch.no_grad():
            dataset_norms.log_scales.copy_(
                torch.log(torch.clamp(values, min=1.0e-12))
            )
        print(
            "initialized dataset norms:",
            {name: float(value) for name, value in zip(data.dataset_names, values.cpu())},
        )
    params = list(model.parameters()) + (list(dataset_norms.parameters()) if cfg.fit_dataset_norms else [])
    opt = torch.optim.AdamW(params, lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=max(10, cfg.patience // 4) if cfg.patience else 50)
    history: list[dict[str, float]] = []
    best = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_norm_state: dict[str, torch.Tensor] | None = None
    stale = 0
    t0 = time.time()
    log_every = int(cfg.log_every) if cfg.log_every else max(1, int(cfg.epochs) // 10)

    for epoch in range(1, int(cfg.epochs) + 1):
        model.train()
        perm = torch.randperm(n, device=data.Q.device)
        losses = []
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            batch = data.batch(idx)
            opt.zero_grad(set_to_none=True)
            pred_raw = model(batch["row_index"], batch["Q"], batch["x1"], batch["x2"], batch["y_term"])
            pred = dataset_norms(pred_raw, batch["dataset_index"])
            loss_data = scaled_mse(pred, batch["target"], batch["sigma"])
            reg_mult = _reg_multiplier_for_epoch(epoch, cfg)
            loss_reg = regularization_loss(
                model,
                data,
                lambda_a_l2=cfg.lambda_a_l2 * reg_mult,
                lambda_gk_l2=cfg.lambda_gk_l2 * reg_mult,
                lambda_fnp_mono=cfg.lambda_fnp_mono * reg_mult,
                lambda_fnp_bcurv=cfg.lambda_fnp_bcurv * reg_mult,
                lambda_fnp_xcurv=cfg.lambda_fnp_xcurv * reg_mult,
                lambda_fnp_pair_bcurv=cfg.lambda_fnp_pair_bcurv * reg_mult,
                lambda_fnp_local_bcurv=cfg.lambda_fnp_local_bcurv * reg_mult,
                fnp_local_bcurv_x_values=cfg.fnp_local_bcurv_x_values,
                fnp_local_bcurv_bmin=cfg.fnp_local_bcurv_bmin,
                fnp_local_bcurv_bmax=cfg.fnp_local_bcurv_bmax,
                lambda_fnp_lowpass=cfg.lambda_fnp_lowpass * reg_mult,
                fnp_lowpass_x_values=cfg.fnp_lowpass_x_values,
                fnp_lowpass_bmin=cfg.fnp_lowpass_bmin,
                fnp_lowpass_bmax=cfg.fnp_lowpass_bmax,
                fnp_lowpass_sigma=cfg.fnp_lowpass_sigma,
                lambda_fnp_ratecurv=cfg.lambda_fnp_ratecurv * reg_mult,
                fnp_ratecurv_bmin=cfg.fnp_ratecurv_bmin,
                fnp_ratecurv_bmax=cfg.fnp_ratecurv_bmax,
                lambda_fnp_tail=cfg.lambda_fnp_tail * reg_mult,
                fnp_tail_bmin=cfg.fnp_tail_bmin,
                fnp_tail_target=cfg.fnp_tail_target,
                mono_tol=cfg.mono_tol,
            )
            loss_norm = float(cfg.lambda_dataset_norm) * dataset_norms.penalty(normalize_by=len(data))
            loss = loss_data + loss_reg + loss_norm
            loss.backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
            opt.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            pred_raw_all = model(data.row_index, data.Q, data.x1, data.x2, data.y_term)
            pred_all = dataset_norms(pred_raw_all, data.dataset_index)
            chi2_like = float(scaled_mse(pred_all, data.target, data.sigma).detach().cpu())
            norm_penalty = float(dataset_norms.penalty(normalize_by=len(data)).detach().cpu())
            rel_rmse = float(torch.sqrt(torch.mean(((pred_all - data.target) / torch.clamp(torch.abs(data.target), min=1e-12)) ** 2)).detach().cpu())
            rmse = float(torch.sqrt(torch.mean((pred_all - data.target) ** 2)).detach().cpu())
            lr_now = float(opt.param_groups[0]["lr"])
            gnorm = float(model.global_norm().detach().cpu())
        scheduler.step(chi2_like)
        history.append({
            "epoch": float(epoch),
            "batch_loss": float(np.mean(losses)),
            "reg_multiplier": float(_reg_multiplier_for_epoch(epoch, cfg)),
            "chi2_like": chi2_like,
            "relative_rmse": rel_rmse,
            "rmse": rmse,
            "lr": lr_now,
            "global_norm": gnorm,
            "norm_penalty": norm_penalty,
        })
        if epoch == 1 or epoch % log_every == 0 or epoch == int(cfg.epochs):
            ds_txt = ""
            if cfg.fit_dataset_norms:
                ds_txt = ", ds_norm=" + json.dumps(dataset_norms.as_dict(), sort_keys=True)
            print(f"epoch {epoch:5d}/{cfg.epochs}: chi2_like={chi2_like:.6g}, rel_rmse={rel_rmse:.6g}, lr={lr_now:.3g}, norm={gnorm:.4g}, norm_penalty={norm_penalty:.3g}{ds_txt}", flush=True)
        improved = chi2_like < best - float(cfg.min_delta)
        if improved:
            best = chi2_like
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_norm_state = {k: v.detach().cpu().clone() for k, v in dataset_norms.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if cfg.patience and cfg.patience > 0 and stale >= int(cfg.patience):
            print(f"early stopping at epoch {epoch}; best epoch={best_epoch}, best chi2_like={best:.6g}", flush=True)
            break

    restored_best = False
    if bool(cfg.restore_best) and best_state is not None:
        model.load_state_dict(best_state)
        if best_norm_state is not None:
            dataset_norms.load_state_dict(best_norm_state)
        restored_best = True
        print(f"restored best epoch {best_epoch} with chi2_like={best:.6g}", flush=True)
    meta = {
        "seconds": time.time() - t0,
        "n_points": n,
        "epochs_run": len(history),
        "last_epoch_chi2_like": history[-1]["chi2_like"],
        "last_epoch_relative_rmse": history[-1]["relative_rmse"],
        "best_epoch": int(best_epoch),
        "best_chi2_like": float(best),
        "restored_best": bool(restored_best),
        "final_chi2_like": float(best if restored_best else history[-1]["chi2_like"]),
        "final_relative_rmse": float(history[best_epoch-1]["relative_rmse"] if restored_best and best_epoch > 0 else history[-1]["relative_rmse"]),
        "fit_dataset_norms": bool(cfg.fit_dataset_norms),
        "dataset_norms": dataset_norms.as_dict(),
        "dataset_norm_pulls": dataset_norms.pulls_dict(),
        "dataset_norm_rel": {name: float(data.dataset_norm_rel[i]) for i, name in enumerate(data.dataset_names)},
    }
    model.dataset_norms = dataset_norms  # type: ignore[attr-defined]
    return pd.DataFrame(history), meta


# -----------------------------------------------------------------------------
# Backend preparation and outputs
# -----------------------------------------------------------------------------


def trapezoid_weights(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) < 3:
        raise ValueError("b grid must be one-dimensional with at least 3 nodes")
    if not np.all(np.diff(x) > 0):
        raise ValueError("b grid must be strictly increasing")
    w = np.empty_like(x)
    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])
    w[1:-1] = 0.5 * (x[2:] - x[:-2])
    return w


def precompute_kernel_matrix(qT: np.ndarray, b_grid: np.ndarray, w_matrix: np.ndarray, *, dtype: torch.dtype) -> np.ndarray:
    qT = np.asarray(qT, dtype=float)
    b = np.asarray(b_grid, dtype=float)
    w = np.asarray(w_matrix, dtype=float)
    if w.shape != (len(qT), len(b)):
        raise ValueError(f"w_matrix shape {w.shape} incompatible with qT/b shapes {(len(qT), len(b))}")
    weights = trapezoid_weights(b)
    # Compute J0 with torch because PyTorch is a required dependency here.
    tq = torch.tensor(qT[:, None], dtype=dtype)
    tb = torch.tensor(b[None, :], dtype=dtype)
    j0 = torch.special.bessel_j0(tq * tb).detach().cpu().numpy()
    kernel = weights[None, :] * b[None, :] * j0 * w
    if not np.all(np.isfinite(kernel)):
        raise ValueError("Precomputed W kernel has non-finite values")
    return kernel


def make_backend_config(args: argparse.Namespace, backend) -> Any:
    kwargs = dict(
        b_min=float(args.b_min),
        b_max=float(args.b_max),
        n_b=int(args.n_b),
        bstar_bmax=float(args.b_star_max),
        mu_min=float(args.mu_min),
        cap_mub_at_Q=not bool(args.no_cap_mub_at_Q),
        q0=float(args.q0),
        resum_order=str(args.resum_order),
        nf=int(args.nf),
        n_sudakov_quad=int(args.n_sudakov_quad),
        alpha_em=float(args.alpha_em),
        hc_factor=float(args.hc_factor),
        prefactor_scheme=str(args.prefactor_scheme),
        global_norm=float(args.backend_global_norm),
        flavors=tuple(int(f) for f in args.flavors),
        target_mode=str(args.target_mode),
        y_mode=str(args.y_mode),
    )
    fields = getattr(backend.CSSConfig, "__dataclass_fields__", {})
    if "match_order" in fields:
        kwargs["match_order"] = str(args.match_order)
    if "nlo_y_pilot_strength" in fields:
        kwargs["nlo_y_pilot_strength"] = float(args.nlo_y_pilot_strength)
    if "nlo_y_transition" in fields:
        kwargs["nlo_y_transition"] = float(args.nlo_y_transition)
    if "nlo_y_transition_width" in fields:
        kwargs["nlo_y_transition_width"] = float(args.nlo_y_transition_width)
    for name, typ in [
        ("nlo_real_quad", int),
        ("nlo_real_norm", float),
        ("nlo_singular_norm", float),
        ("nlo_y_component", str),
        ("nlo_y_clip_multiple", float),
        ("nlo_dev_min_qt_over_q", float),
        ("nlo_singular_mode", str),
        ("nlo_singular_rsub", float),
        ("nlo_singular_power", float),
        ("nlo_singular_damp_kind", str),
        ("nlo_real_convention", str),
        ("nlo_singular_convention", str),
        ("nlo_alpha_convention", str),
        ("nlo_real_tail_repair", str),
        ("nlo_real_tail_r0", float),
        ("nlo_real_tail_width", float),
        ("nlo_real_tail_rinf", float),
    ]:
        if name in fields and hasattr(args, name):
            kwargs[name] = typ(getattr(args, name))
    if "nlo_dev_use_switch" in fields:
        kwargs["nlo_dev_use_switch"] = not bool(getattr(args, "no_nlo_dev_switch", False))
    return backend.CSSConfig(**kwargs)


def prepare_data_and_backend(args: argparse.Namespace, backend, dtype: torch.dtype) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    cuts = backend.CutConfig(
        mode=str(args.mode),
        qT_max_over_Q=float(args.qT_max_over_Q),
        tmd_qT_max_over_Q=float(args.tmd_qT_max_over_Q),
        apply_upsilon_veto=not bool(args.no_upsilon_veto),
    )
    df = backend.load_fixed_target_data(args.data_dir, args.datasets, cuts)
    if len(df) == 0:
        raise ValueError("No data rows remain after cuts")
    print(f"loaded rows after cuts: {len(df)}")
    print(backend.summarize(df).to_string(index=False))

    meta: Dict[str, Any] = {
        "backend_script": str(args.backend_script) if args.backend_script else "bt_internal_css_backend_v18.py",
        "w_backend": args.w_backend,
        "cuts": asdict(cuts),
    }

    if args.w_backend == "external":
        if not args.w_grid:
            raise SystemExit("--w-backend external requires --w-grid")
        b_grid, w_matrix = load_external_w_grid(df["row_id"].astype(str).tolist(), args.w_grid)
        if args.y_grid:
            y_term = load_y_grid(df, args.y_grid, allow_zero_for_nonfinite=args.allow_zero_y_in_matched)
        else:
            if args.mode == "matched" and not args.allow_zero_y_in_matched:
                raise SystemExit("matched external-W mode requires --y-grid, unless --allow-zero-y-in-matched is set")
            y_term = np.zeros(len(df), dtype=float)
        meta.update({"w_grid": str(args.w_grid), "y_grid": str(args.y_grid) if args.y_grid else None})
    elif args.w_backend == "internal_css":
        cfg = make_backend_config(args, backend)
        pdf = backend.LHAPDFProvider(args.pdf_set, args.pdf_member, use_toy_pdf=args.toy_pdf)
        print("building internal CSS W grid in memory...")
        print("CSS backend config:")
        print(json.dumps(asdict(cfg), indent=2, sort_keys=True))
        b_grid, w_matrix, y_term = backend.compute_backend_grids(df, pdf, cfg, progress=not args.quiet_backend)
        diag = getattr(backend, "LAST_BACKEND_ROW_DIAGNOSTICS", None)
        if isinstance(diag, pd.DataFrame):
            outp = Path(args.out)
            outp.mkdir(parents=True, exist_ok=True)
            diag_path = outp / "backend_y_diagnostics.csv"
            diag.to_csv(diag_path, index=False)
            print("wrote backend Y diagnostics to", diag_path)
            meta["backend_y_diagnostics"] = str(diag_path)
            # Compact audit summaries. These are useful in check-only runs and do
            # not affect training.
            try:
                cols = [c for c in [
                    "qT_over_Q", "W_CS_baseline", "FO_NLO_real_dev_CS",
                    "singular_NLO_dev_CS", "raw_Y_NLO_dev_CS", "component_Y_CS",
                    "tail_switch", "Y_unclipped_CS", "Y_CS", "Y_over_W",
                    "FO_over_singular"
                ] if c in diag.columns]
                if cols:
                    by_dataset = diag.groupby("dataset", dropna=False)[cols].agg(["count", "mean", "median", "min", "max"])
                    by_dataset_path = outp / "backend_y_summary_by_dataset.csv"
                    by_dataset.to_csv(by_dataset_path)
                    print("wrote backend Y summary by dataset to", by_dataset_path)
                    meta["backend_y_summary_by_dataset"] = str(by_dataset_path)
                    # qT/Q bins: TMD, transition, matched-tail.
                    qbins = [-1e-9, 0.2, 0.5, 1.0, float("inf")]
                    qlabs = ["0_0p2", "0p2_0p5", "0p5_1p0", "gt1p0"]
                    tmp = diag.copy()
                    tmp["qT_over_Q_bin"] = pd.cut(tmp["qT_over_Q"], bins=qbins, labels=qlabs)
                    by_bin = tmp.groupby(["qT_over_Q_bin", "dataset"], dropna=False)[cols].agg(["count", "mean", "median", "min", "max"])
                    by_bin_path = outp / "backend_y_summary_by_qToQ_bin.csv"
                    by_bin.to_csv(by_bin_path)
                    print("wrote backend Y summary by qT/Q bin to", by_bin_path)
                    meta["backend_y_summary_by_qToQ_bin"] = str(by_bin_path)
            except Exception as exc:
                print("WARNING: failed to write backend Y summaries:", repr(exc))
        if args.y_grid:
            y_term = load_y_grid(df, args.y_grid, allow_zero_for_nonfinite=args.allow_zero_y_in_matched)
            print("overrode internal Y term with external Y grid:", args.y_grid)
        if args.cache_backend_grids:
            cache_dir = Path(args.out) / "backend_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            tag = args.cache_tag or f"{args.mode}_{args.resum_order}_{args.match_order}"
            w_path = cache_dir / f"wpert_{tag}.csv"
            y_path = cache_dir / f"y_{tag}.csv"
            base_path = cache_dir / f"baseline_{tag}.csv"
            meta_path = cache_dir / f"metadata_{tag}.json"
            # A baseline integral without neural factors is useful for sanity checks.
            baseline = backend.torch_bessel_integral(df["qT"].to_numpy(float), b_grid, w_matrix)
            backend.write_w_grid(df, w_path, b_grid, w_matrix)
            backend.write_y_grid(df, y_path, y_term, mode=args.mode)
            backend.write_baseline(base_path, df, baseline, y_term)
            backend.write_metadata(meta_path, df, cfg, vars(args), baseline, y_term)
            print("cached backend grids to", cache_dir)
        meta.update({"internal_css_config": asdict(cfg), "pdf_set": args.pdf_set, "pdf_member": args.pdf_member})
    else:
        raise ValueError(f"Unknown w_backend={args.w_backend!r}")

    if args.mode == "matched" and np.allclose(y_term, 0.0):
        print("NOTE: matched mode has Y_CS=0. This is W-only unless --match-order nlo_pilot or an external Y grid is used.")
    if args.mode == "matched" and str(args.match_order) in {"nlo", "nlo_dev"}:
        print("WARNING: using v16 NLO finite-tail development path. It includes localized asymptotic-subtraction diagnostics but still needs independent benchmarking before production Y_NLO claims.")
    elif args.mode == "matched" and str(args.match_order) == "nlo_pilot":
        print("WARNING: using built-in nlo_pilot finite tail. This is a matched-mode development scaffold, not audited production Y_NLO.")

    kernel = precompute_kernel_matrix(df["qT"].to_numpy(float), b_grid, w_matrix, dtype=dtype)
    with np.errstate(divide="ignore", invalid="ignore"):
        baseline_w = kernel.sum(axis=1)
        ratio = baseline_w / np.where(df["CS"].to_numpy(float) != 0, df["CS"].to_numpy(float), np.nan)
    med_ratio = float(np.nanmedian(ratio))
    print("baseline W/data median:", med_ratio)
    if np.isfinite(med_ratio) and med_ratio < 0.85:
        print("NOTE: the bare W Bessel integral is below the data median. Because J0 oscillates, a positive F_NP damping factor can change the integrated W up or down by suppressing large-b cancellations. Use --np-a0-scan in check-only mode before interpreting an overall normalization.")
    print("kernel shape:", kernel.shape)
    meta["kernel_shape"] = list(kernel.shape)
    meta["baseline_w_over_data_median"] = med_ratio
    return df, b_grid, kernel, y_term, meta


def write_outputs(
    *,
    model: PrecomputedKernelModel,
    data: TensorData,
    history: pd.DataFrame,
    train_meta: Dict[str, Any],
    config: Dict[str, Any],
    out_dir: str | Path,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        pred_w_raw = model.sigma_w(data.row_index, data.Q, data.x1, data.x2)
        pred_raw = pred_w_raw + data.y_term
        dataset_norms = getattr(model, "dataset_norms", None)
        if dataset_norms is not None:
            pred_w = dataset_norms(pred_w_raw, data.dataset_index)
            pred = dataset_norms(pred_raw, data.dataset_index)
            ds_norm_factor = dataset_norms.scales().to(data.Q.device, data.Q.dtype)[data.dataset_index]
        else:
            pred_w = pred_w_raw
            pred = pred_raw
            ds_norm_factor = torch.ones_like(pred)
    pred_df = data.df.copy()
    pred_df["target_used"] = data.target.detach().cpu().numpy()
    pred_df["sigma_used"] = data.sigma.detach().cpu().numpy()
    pred_df["Y_CS_used"] = data.y_term.detach().cpu().numpy()
    pred_df["dataset_norm_factor"] = ds_norm_factor.detach().cpu().numpy()
    pred_df["pred_W_CS_raw_before_dataset_norm"] = pred_w_raw.detach().cpu().numpy()
    pred_df["pred_match_CS_raw_before_dataset_norm"] = pred_raw.detach().cpu().numpy()
    pred_df["pred_W_CS"] = pred_w.detach().cpu().numpy()
    pred_df["pred_match_CS"] = pred.detach().cpu().numpy()
    pred_df["pull"] = (pred_df["pred_match_CS"] - pred_df["target_used"]) / pred_df["sigma_used"]
    pred_df.to_csv(out / "predictions.csv", index=False)
    history.to_csv(out / "loss_history.csv", index=False)
    torch.save(model.state_dict(), out / "model_state.pt")

    # Compact per-dataset metrics for command-line checking.
    per_dataset = []
    for ds, sub in pred_df.groupby("dataset", sort=False):
        per_dataset.append({
            "dataset": ds,
            "n": int(len(sub)),
            "chi2_like": float(np.mean(sub["pull"].to_numpy(float) ** 2)),
            "median_abs_pull": float(np.median(np.abs(sub["pull"].to_numpy(float)))),
            "median_pred_over_target": float(np.median(sub["pred_match_CS"].to_numpy(float) / np.where(sub["target_used"].to_numpy(float) != 0, sub["target_used"].to_numpy(float), np.nan))),
        })
    metrics = {"train": train_meta, "per_dataset": per_dataset, "config": config}
    dataset_norms_obj = getattr(model, "dataset_norms", None)
    if dataset_norms_obj is not None:
        metrics["dataset_norms"] = dataset_norms_obj.as_dict()
        metrics["dataset_norm_pulls"] = dataset_norms_obj.pulls_dict()
        pd.DataFrame([
            {"dataset": k, "norm_scale": dataset_norms_obj.as_dict()[k], "norm_pull": dataset_norms_obj.pulls_dict()[k]}
            for k in dataset_norms_obj.as_dict().keys()
        ]).to_csv(out / "dataset_norms.csv", index=False)
    with (out / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    # Save a lightweight nonperturbative table for debugging/inspection.
    with torch.no_grad():
        b = model.b.detach().cpu().numpy()
        x_grid = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.7], dtype=float)
        xb = torch.tensor(x_grid, dtype=model.b.dtype, device=model.b.device)
        fnp = model.np_factor(xb, model.b).detach().cpu().numpy()
        rows = []
        for ix, xval in enumerate(x_grid):
            for j, bv in enumerate(b):
                rows.append({"x": float(xval), "bT": float(bv), "F_NP": float(fnp[ix, j])})
        pd.DataFrame(rows).to_csv(out / "fnp_debug_grid.csv", index=False)
        mono_rows = []
        for ix, xval in enumerate(x_grid):
            vals = fnp[ix, :]
            inc = np.diff(vals)
            mono_rows.append({
                "x": float(xval),
                "min_F_NP": float(np.min(vals)),
                "max_F_NP": float(np.max(vals)),
                "max_increase": float(np.max(inc)) if inc.size else 0.0,
                "n_increases_gt_1e_minus_4": int(np.sum(inc > 1.0e-4)),
                "is_monotone_tol_1e_minus_4": bool(np.all(inc <= 1.0e-4)),
            })
        pd.DataFrame(mono_rows).to_csv(out / "fnp_monotonicity.csv", index=False)
        # Pair-damping grid on representative real-data kinematics.  This is
        # closer to what Drell-Yan directly constrains than arbitrary single-x
        # slices of F_NP.
        pair_rows = []
        pair_source = data.df[["x1", "x2", "QM"]].copy()
        if len(pair_source) > 0:
            # Pick quantile rows in qT/Q-independent x1*x2 support.
            pair_source["xprod"] = pair_source["x1"].to_numpy(float) * pair_source["x2"].to_numpy(float)
            qidx = np.unique(np.linspace(0, len(pair_source) - 1, min(12, len(pair_source))).round().astype(int))
            pair_source = pair_source.sort_values("xprod").iloc[qidx]
            x1t = torch.tensor(pair_source["x1"].to_numpy(float), dtype=model.b.dtype, device=model.b.device)
            x2t = torch.tensor(pair_source["x2"].to_numpy(float), dtype=model.b.dtype, device=model.b.device)
            fpair = (model.np_factor(x1t, model.b) * model.np_factor(x2t, model.b)).detach().cpu().numpy()
            for ip, (_, rr) in enumerate(pair_source.reset_index(drop=True).iterrows()):
                for j, bv in enumerate(b):
                    pair_rows.append({
                        "pair_index": int(ip),
                        "x1": float(rr["x1"]),
                        "x2": float(rr["x2"]),
                        "QM": float(rr["QM"]),
                        "bT": float(bv),
                        "F_pair": float(fpair[ip, j]),
                    })
        pd.DataFrame(pair_rows).to_csv(out / "fnp_pair_debug_grid.csv", index=False)
        gk = model.gk_model(model.b).detach().cpu().numpy()
        pd.DataFrame({"bT": b, "G_CS": gk, "gK_legacy": gk}).to_csv(out / "gk_debug_grid.csv", index=False)
        q_debug = np.array([4.5, 6.5, 10.0, 15.75, 91.1876], dtype=float)
        evol_rows = []
        for qv in q_debug:
            Lq = soft_log_multiplier_np(
                np.array([qv], dtype=float),
                q0=float(getattr(model, "q0", 2.0)),
                cs_log=str(getattr(model, "cs_log", "lnQ")),
                cs_kernel_convention=str(getattr(model, "cs_kernel_convention", "pair")),
            )[0]
            factor = np.exp(-gk * Lq)
            for j, bv in enumerate(b):
                evol_rows.append({
                    "Q": float(qv),
                    "bT": float(bv),
                    "L_Q": float(Lq),
                    "G_CS": float(gk[j]),
                    "soft_evolution_factor": float(factor[j]),
                })
        pd.DataFrame(evol_rows).to_csv(out / "soft_evolution_debug_grid.csv", index=False)

    print("wrote outputs to", out)
    print("per-dataset metrics:")
    print(pd.DataFrame(per_dataset).to_string(index=False))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Self-contained bT-space FiLM/DNN trainer using bt_internal_css_backend_v18.py for W/Y kernels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Script/backend organization.
    ap.add_argument("--backend-script", default=None, help="Path to bt_internal_css_backend_v18.py. Defaults to same directory as this script.")

    # Data/cuts.
    ap.add_argument("--data-dir", default="./Data")
    ap.add_argument("--datasets", nargs="+", default=["E288_200", "E288_300", "E288_400", "E605"])
    ap.add_argument("--mode", choices=["matched", "tmd_only", "none"], default="matched")
    ap.add_argument("--qT-max-over-Q", type=float, default=1.0)
    ap.add_argument("--tmd-qT-max-over-Q", type=float, default=0.2)
    ap.add_argument("--no-upsilon-veto", action="store_true")

    # W/Y source.
    ap.add_argument("--w-backend", choices=["internal_css", "external"], default="internal_css")
    ap.add_argument("--w-grid", default=None, help="External long CSV with row_id,bT,Wpert_CS.")
    ap.add_argument("--y-grid", default=None, help="External CSV with row_id,Y_CS or row_id,FO_CS,singular_CS.")
    ap.add_argument("--allow-zero-y-in-matched", action="store_true", help="Allow missing/non-finite Y values as zero; development only.")
    ap.add_argument("--cache-backend-grids", action="store_true", help="When using internal_css, also write W/Y cache CSVs under OUT/backend_cache.")
    ap.add_argument("--cache-tag", default=None)

    # Internal backend theory settings; forwarded to bt_internal_css_backend.CSSConfig.
    ap.add_argument("--pdf-set", default="NNPDF40_nnlo_as_01180")
    ap.add_argument("--pdf-member", type=int, default=0)
    ap.add_argument("--toy-pdf", action="store_true", help="Nonphysical smoke-test PDFs if LHAPDF is unavailable.")
    ap.add_argument("--target-mode", choices=["proton_approx", "isoscalar", "nuclear_isospin", "nuclear_pdf"], default="proton_approx")
    ap.add_argument("--flavors", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--resum-order", choices=["ll", "nll", "nnll", "n3llp", "n3ll_pilot"], default="nnll")
    ap.add_argument("--nf", type=int, default=5)
    ap.add_argument("--b-min", type=float, default=1.0e-4)
    ap.add_argument("--b-max", type=float, default=8.0)
    ap.add_argument("--n-b", type=int, default=160)
    ap.add_argument("--b-star-max", type=float, default=1.5)
    ap.add_argument("--mu-min", type=float, default=1.3)
    ap.add_argument("--no-cap-mub-at-Q", action="store_true")
    ap.add_argument("--q0", type=float, default=2.0)
    ap.add_argument("--n-sudakov-quad", type=int, default=32)
    ap.add_argument("--prefactor-scheme", choices=["oldA_to_CS", "unit"], default="oldA_to_CS")
    ap.add_argument("--backend-global-norm", type=float, default=1.0)
    ap.add_argument("--hc-factor", type=float, default=3.893793656e8)
    ap.add_argument("--alpha-em", type=float, default=1.0 / 137.035999084)
    ap.add_argument("--match-order", choices=["none", "nlo_pilot", "nlo_dev", "nlo"], default="none", help="Finite-tail matching order. nlo currently aliases the backend nlo_dev analytic-development path in backend v18.")
    ap.add_argument("--nlo-y-pilot-strength", type=float, default=1.0, help="Scale factor for backend nlo_pilot finite tail; development only.")
    ap.add_argument("--nlo-y-transition", type=float, default=0.20, help="qT/Q turn-on point for backend nlo_pilot Y.")
    ap.add_argument("--nlo-y-transition-width", type=float, default=0.15, help="Smooth turn-on width for backend nlo_pilot/nlo_dev Y.")
    ap.add_argument("--nlo-real-quad", type=int, default=48, help="Gauss-Legendre nodes for the nlo_dev recoil-rapidity integral.")
    ap.add_argument("--nlo-real-norm", type=float, default=1.0, help="Development normalization factor for FO_NLO_real_dev.")
    ap.add_argument("--nlo-singular-norm", type=float, default=1.0, help="Development normalization factor for singular_NLO_dev.")
    ap.add_argument("--nlo-y-component", choices=["raw", "positive", "fo_only", "minus_sing", "singular_only", "zero"], default="raw", help="Audit knob for nlo_dev Y before switch/clip: raw=FO-sing, positive=max(FO-sing,0), fo_only=FO, minus_sing=-sing, singular_only=sing.")
    ap.add_argument("--nlo-y-clip-multiple", type=float, default=5.0, help="Clip |Y| to this multiple of max(|W|,|data|); use <=0 to disable. Development only.")
    ap.add_argument("--no-nlo-dev-switch", action="store_true", help="Disable smooth qT/Q turn-on for nlo_dev Y.")
    ap.add_argument("--nlo-dev-min-qt-over-q", type=float, default=1.0e-4, help="qT/Q floor in singular_NLO_dev.")
    ap.add_argument("--nlo-singular-mode", choices=["analytic", "asymptotic_damped", "wexp_numeric", "wexp_positive", "none"], default="asymptotic_damped", help="V18 singular subtraction mode. asymptotic_damped uses a localized qT-space asymptotic subtraction; wexp_numeric retained for diagnostics.")
    ap.add_argument("--nlo-singular-rsub", type=float, default=0.20, help="r_sub in qT/Q for asymptotic_damped singular subtraction localization.")
    ap.add_argument("--nlo-singular-power", type=float, default=4.0, help="Power p in the asymptotic_damped singular subtraction profile.")
    ap.add_argument("--nlo-singular-damp-kind", choices=["exp", "rational"], default="exp", help="Damping profile for asymptotic_damped singular subtraction.")
    ap.add_argument("--nlo-real-convention", default="base", help="Audit multiplier for real term: base, times_qt, times_2qt, div_qt, times_Q, times_2pi, etc.; combine with commas or *.")
    ap.add_argument("--nlo-singular-convention", default="base", help="Audit multiplier for singular term; same tokens as --nlo-real-convention.")
    ap.add_argument("--nlo-alpha-convention", choices=["alpha_over_pi", "alpha_over_2pi"], default="alpha_over_pi", help="Alpha_s normalization used in NLO real and singular terms.")
    ap.add_argument("--nlo-real-tail-repair", choices=["none", "mcfm_logistic", "external_logistic", "logistic"], default="none", help="V18 external-code calibrated high-qT/Q multiplier for FO_NLO_real_dev; none reproduces v15.")
    ap.add_argument("--nlo-real-tail-r0", type=float, default=0.520, help="Center r=qT/Q of the MCFM/DYTurbo logistic real-tail repair.")
    ap.add_argument("--nlo-real-tail-width", type=float, default=0.010, help="Width in r=qT/Q of the MCFM/DYTurbo logistic real-tail repair.")
    ap.add_argument("--nlo-real-tail-rinf", type=float, default=0.180, help="Large-r limiting multiplier for the MCFM/DYTurbo real-tail repair.")
    ap.add_argument("--y-mode", choices=["zero", "nlo_pilot", "data_minus_w_debug"], default="zero")
    ap.add_argument("--quiet-backend", action="store_true")

    # Experimental replicas and uncertainties.
    ap.add_argument("--replica-seed", type=int, default=None, help="Train on one experimental replica with correlated dataset normalizations.")
    ap.add_argument("--fit-dataset-norms", action="store_true", help="Profile paper/csv correlated dataset-normalization nuisances with Gaussian priors.")
    ap.add_argument("--lambda-dataset-norm", type=float, default=1.0, help="Multiplier for the dataset-normalization nuisance penalty.")
    ap.add_argument("--norm-source", choices=["paper", "csv", "none"], default="paper")
    ap.add_argument("--ptp-source", choices=["paper", "csv", "none"], default="paper")

    # DNN architecture.
    ap.add_argument("--np-width", type=int, default=48)
    ap.add_argument("--np-cond-width", type=int, default=32)
    ap.add_argument("--np-blocks", type=int, default=3)
    ap.add_argument("--np-a0", type=float, default=0.08)
    ap.add_argument("--np-min-a", type=float, default=0.0)
    ap.add_argument("--np-a-mode", choices=["positive", "signed"], default="positive", help="positive enforces damping; signed allows F_NP>1 for backend diagnostics while keeping F_NP(x,0)=1.")
    ap.add_argument(
        "--np-a-smooth-sigma",
        type=float,
        default=0.0,
        help=(
            "Gaussian smoothing width in physical bT [GeV^-1] applied "
            "to the positive A_theta field before monotone integration. "
            "Zero restores the legacy parameterization."
        ),
    )
    ap.add_argument("--np-a-tail-amp", type=float, default=0.0, help="Positive late-b A_theta floor amplitude.")
    ap.add_argument("--np-a-tail-b0", type=float, default=3.5, help="Center bT [GeV^-1] of the smooth late-b floor.")
    ap.add_argument("--np-a-tail-width", type=float, default=0.25, help="Logistic width [GeV^-1] of the smooth late-b floor.")
    ap.add_argument("--np-shape-mode", choices=["direct", "monotone"], default="direct", help="direct uses F_NP=exp[-b^2 A(x,b)] and may be nonmonotonic; monotone uses a cumulative positive slope so F_NP decreases with b by construction.")
    ap.add_argument("--np-a0-scan", nargs="*", type=float, default=None, help="Check-only diagnostic: print initial W/data sign/scale diagnostics for candidate Gaussian A0 values. If passed without values, a default list is used.")
    ap.add_argument("--np-a0-scan-exit", action="store_true", help="Exit immediately after printing the --np-a0-scan table.")
    ap.add_argument("--strict-positive-init", action="store_true", help="Abort if the initial F_NP/gK prediction has nonpositive rows; useful for choosing a stable initial damping.")
    ap.add_argument("--fnp-exponent-clip", type=float, default=40.0, help="Clamp exponent in F_NP=exp[-b^2 A] for numerical stability.")
    # Controlled nonperturbative soft-Q / Collins-Soper evolution.
    # Preferred new interface: --soft-q-evolution cs_kernel with the --cs-* aliases.
    # Backward-compatible old interface: --learn-gk with --gk-* flags.
    ap.add_argument("--soft-q-evolution", choices=["none", "cs_kernel"], default=None, help="Controlled nonperturbative soft/CS evolution. 'none' freezes it; 'cs_kernel' learns G_CS(b) in exp[-G_CS(b)*L_Q]. If omitted, the legacy --learn-gk flag controls this.")
    ap.add_argument("--cs-log", choices=["lnQ", "lnQ2"], default="lnQ", help="Logarithm used in the nonperturbative soft evolution: lnQ means ln(Q/Q0); lnQ2 means ln(Q^2/Q0^2)=2 ln(Q/Q0).")
    ap.add_argument("--cs-kernel-convention", choices=["pair", "single_tmd"], default="pair", help="pair: network is the pair-level DY kernel in exp[-G_pair L_Q]. single_tmd: network is one single-TMD kernel and the DY pair exponent gets an extra factor of two.")
    ap.add_argument("--cs-kernel-mode", choices=["bounded", "unbounded"], default=None, help="Alias for --gk-mode.")
    ap.add_argument("--cs-kernel-b0", type=float, default=None, help="Alias for --gk-b0; initial quadratic coefficient of G_CS(b)=b^2 B(b).")
    ap.add_argument("--cs-kernel-bmax", type=float, default=None, help="Alias for --gk-bmax; upper bound on B(b) when bounded.")
    ap.add_argument("--cs-kernel-cap", type=float, default=None, help="Alias for --gk-cap; direct cap on G_CS at b_max.")
    ap.add_argument("--learn-gk", action="store_true", help="Legacy alias for --soft-q-evolution cs_kernel.")
    ap.add_argument("--gk-mode", choices=["bounded", "unbounded"], default="bounded", help="bounded caps the quadratic coefficient B in G_CS=b^2 B; unbounded is diagnostic only.")
    ap.add_argument("--gk-width", type=int, default=24)
    ap.add_argument("--gk-layers", type=int, default=2)
    ap.add_argument("--gk-b0", type=float, default=0.02)
    ap.add_argument("--gk-bmax", type=float, default=0.08, help="Upper bound on the quadratic coefficient B_theta when --gk-mode bounded; G_CS(b) <= b^2*gk_bmax. For b_max=8, gk_bmax=0.078125 gives G_CS(8)<=5.")
    ap.add_argument("--gk-cap", type=float, default=None, help="Convenience alias for a direct cap on G_CS at the largest b grid point: sets --gk-bmax = gk_cap / b_max**2. Example: --gk-cap 5 with --b-max 8 gives --gk-bmax 0.078125.")
    ap.add_argument("--learn-global-norm", action="store_true", help="Development diagnostic; do not use for final physics without documenting it.")
    ap.add_argument("--global-norm-init", type=float, default=1.0)
    ap.add_argument("--auto-global-norm-init", action="store_true", help="Initialize --learn-global-norm from the configured initial F_NP/gK prediction. Development diagnostic only.")
    ap.add_argument("--auto-global-norm-method", choices=["inverse_median", "median_data_over_w", "weighted_ls"], default="weighted_ls", help="Scale estimator used by --auto-global-norm-init. inverse_median means 1/median(W/data), matching the printed baseline diagnostic.")

    # Optimization/runtime.
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=10.0)
    ap.add_argument("--lambda-a-l2", type=float, default=0.0)
    ap.add_argument("--lambda-gk-l2", type=float, default=0.0)
    ap.add_argument("--lambda-cs-kernel-l2", type=float, default=None, help="Alias for --lambda-gk-l2 with the preferred CS-kernel naming.")
    ap.add_argument("--lambda-fnp-mono", type=float, default=0.0, help="Soft monotonicity penalty for direct F_NP: penalizes positive finite differences in bT on the fitted x support. Use with --np-shape-mode direct.")
    ap.add_argument("--lambda-fnp-bcurv", type=float, default=0.0, help="Removable v18 scaffold: curvature penalty on log F_NP along bT.")
    ap.add_argument(
        "--lambda-fnp-ratecurv",
        type=float,
        default=0.0,
        help=(
            "Removable v19 scaffold: curvature penalty on the damping "
            "rate h(x,bT)=-d_b log F_NP=2 bT A_theta."
        ),
    )
    ap.add_argument(
        "--fnp-ratecurv-bmin",
        type=float,
        default=0.25,
        help="Lower bT edge for damping-rate curvature regularization.",
    )
    ap.add_argument(
        "--fnp-ratecurv-bmax",
        type=float,
        default=4.0,
        help="Upper bT edge for damping-rate curvature regularization.",
    )
    ap.add_argument(
        "--lambda-fnp-local-bcurv",
        type=float,
        default=0.0,
        help=(
            "Localized curvature penalty on log F_NP over selected "
            "x values and bT interval."
        ),
    )
    ap.add_argument(
        "--fnp-local-bcurv-x-values",
        type=float,
        nargs="+",
        default=[0.15, 0.20, 0.30, 0.40, 0.50],
        help="x probe values used by the localized curvature scaffold.",
    )
    ap.add_argument(
        "--fnp-local-bcurv-bmin",
        type=float,
        default=0.5,
        help="Lower bT edge of localized curvature scaffold.",
    )
    ap.add_argument(
        "--fnp-local-bcurv-bmax",
        type=float,
        default=3.5,
        help="Upper bT edge of localized curvature scaffold.",
    )
    ap.add_argument(
        "--lambda-fnp-lowpass",
        type=float,
        default=0.0,
        help=(
            "Localized low-pass scaffold on the damping rate "
            "h=-d_b log F_NP=2 b A_theta."
        ),
    )
    ap.add_argument(
        "--fnp-lowpass-x-values",
        type=float,
        nargs="+",
        default=[0.15, 0.20, 0.30, 0.40, 0.50],
        help="x values used by the localized damping-rate low-pass scaffold.",
    )
    ap.add_argument(
        "--fnp-lowpass-bmin",
        type=float,
        default=0.5,
        help="Lower bT edge of the low-pass penalty interval.",
    )
    ap.add_argument(
        "--fnp-lowpass-bmax",
        type=float,
        default=3.5,
        help="Upper bT edge of the low-pass penalty interval.",
    )
    ap.add_argument(
        "--fnp-lowpass-sigma",
        type=float,
        default=0.30,
        help=(
            "Gaussian smoothing width in physical bT units "
            "[GeV^-1] for the damping-rate low-pass scaffold."
        ),
    )
    ap.add_argument("--lambda-fnp-xcurv", type=float, default=0.0, help="Removable v18 scaffold: curvature penalty on log F_NP along logit(x).")
    ap.add_argument("--lambda-fnp-pair-bcurv", type=float, default=0.0, help="Removable v18 scaffold: curvature penalty on log[F_NP(x1,b) F_NP(x2,b)] for fitted data pairs.")
    ap.add_argument("--lambda-fnp-tail", type=float, default=0.0, help="Removable v18 scaffold: weak large-b plateau penalty on F_NP.")
    ap.add_argument("--fnp-tail-bmin", type=float, default=6.0, help="bT threshold for --lambda-fnp-tail.")
    ap.add_argument("--fnp-tail-target", type=float, default=0.35, help="Allowed large-b F_NP value before --lambda-fnp-tail penalizes excess.")
    ap.add_argument("--reg-backoff-start-frac", type=float, default=1.0, help="Epoch fraction where removable regularizer backoff starts. 1 disables backoff.")
    ap.add_argument("--reg-backoff-end-frac", type=float, default=1.0, help="Epoch fraction where removable regularizer backoff ends.")
    ap.add_argument("--reg-final-scale", type=float, default=1.0, help="Final multiplier for removable regularizers after backoff; set 0 to remove by the end.")
    ap.add_argument("--mono-tol", type=float, default=1.0e-4, help="Tolerance for the soft F_NP monotonicity penalty.")
    ap.add_argument("--patience", type=int, default=0, help="Early-stop after this many epochs without chi2_like improvement. Best epoch is restored even when patience=0 unless --no-restore-best is used.")
    ap.add_argument("--min-delta", type=float, default=1e-7, help="Minimum chi2_like improvement needed to update the best epoch.")
    ap.add_argument("--no-restore-best", action="store_true", help="Keep the last epoch instead of restoring the best chi2_like epoch.")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    ap.add_argument("--device", default="cpu", help="cpu, cuda, or mps. CPU is recommended for the current small fixed-target fits.")
    ap.add_argument("--num-threads", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=0)
    ap.add_argument("--check-only", action="store_true", help="Prepare data/backend and write no model; useful for sanity checks.")
    ap.add_argument(
        "--init-dataset-norms-from",
        default=None,
        help=(
            "Optional output directory or dataset_norms.csv used to initialize "
            "profiled dataset normalization scales."
        ),
    )
    ap.add_argument("--init-model-state", default=None,
                    help="Warm-start model weights from a saved model_state.pt. Skips b/kernel buffers if shapes differ.")
    ap.add_argument("--out", default="outputs/bt_dnn_fit")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = build_argparser()
    args = ap.parse_args(argv)

    # Resolve matched-tail aliases.
    if args.match_order == "nlo":
        print("WARNING: --match-order nlo uses the v18 same-backend NLO development path. Benchmark against an independent FO code before production Y_NLO claims.")

    # Resolve new CS-kernel aliases into the legacy internal variable names.
    if args.soft_q_evolution == "cs_kernel":
        args.learn_gk = True
    elif args.soft_q_evolution == "none":
        args.learn_gk = False
    else:
        args.soft_q_evolution = "cs_kernel" if bool(args.learn_gk) else "none"
    if args.cs_kernel_mode is not None:
        args.gk_mode = str(args.cs_kernel_mode)
    if args.cs_kernel_b0 is not None:
        args.gk_b0 = float(args.cs_kernel_b0)
    if args.cs_kernel_bmax is not None:
        args.gk_bmax = float(args.cs_kernel_bmax)
    if args.cs_kernel_cap is not None:
        args.gk_cap = float(args.cs_kernel_cap)
    if args.lambda_cs_kernel_l2 is not None:
        args.lambda_gk_l2 = float(args.lambda_cs_kernel_l2)

    if getattr(args, "gk_cap", None) is not None:
        if float(args.gk_cap) <= 0.0:
            raise SystemExit("--gk-cap must be positive")
        if float(args.b_max) <= 0.0:
            raise SystemExit("--b-max must be positive when using --gk-cap")
        args.gk_bmax = float(args.gk_cap) / (float(args.b_max) ** 2)
        print(f"interpreting --gk-cap={float(args.gk_cap):.6g} as --gk-bmax={float(args.gk_bmax):.6g} for b_max={float(args.b_max):.6g}")
    if args.num_threads and args.num_threads > 0:
        torch.set_num_threads(int(args.num_threads))
    dtype = dtype_from_string(args.dtype)
    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    backend, backend_path = import_backend(args.backend_script)
    args.backend_script = str(backend_path)
    print("using backend:", backend_path)
    print("torch:", torch.__version__, "device:", device, "dtype:", args.dtype)

    df_raw, b_grid, kernel, y_term, backend_meta = prepare_data_and_backend(args, backend, dtype)
    rep_cfg = ReplicaConfig(observable="CS", error_column="error", norm_source=args.norm_source, ptp_source=args.ptp_source)
    df = build_uncertainties(df_raw, rep_cfg)
    baseline = kernel.sum(axis=1) + np.asarray(y_term, dtype=float)
    y_central = df["CS"].to_numpy(float)
    sigma_central = df["sigma_uncorr"].to_numpy(float)
    scale_diag = backend_scale_diagnostics(df=df, baseline=baseline, target=y_central, sigma=sigma_central)
    print("backend scale diagnostics for bare W+Y:")
    print(json.dumps({k: v for k, v in scale_diag.items() if k != "per_dataset"}, indent=2, sort_keys=True))
    print("backend scale diagnostics by dataset for bare W+Y:")
    print(pd.DataFrame(scale_diag["per_dataset"]).to_string(index=False))

    init_baseline = initial_np_prediction(
        kernel=kernel,
        b_grid=b_grid,
        df=df,
        y_term=y_term,
        np_a0=float(args.np_a0),
        np_min_a=float(args.np_min_a),
        learn_gk=bool(args.learn_gk),
        gk_b0=float(args.gk_b0),
        q0=float(args.q0),
        gk_mode=str(args.gk_mode),
        gk_bmax=float(args.gk_bmax),
        cs_log=str(args.cs_log),
        cs_kernel_convention=str(args.cs_kernel_convention),
    )
    init_scale_diag = backend_scale_diagnostics(df=df, baseline=init_baseline, target=y_central, sigma=sigma_central)
    print("backend scale diagnostics for initial F_NP/gK:")
    print(json.dumps({k: v for k, v in init_scale_diag.items() if k != "per_dataset"}, indent=2, sort_keys=True))
    print("backend scale diagnostics by dataset for initial F_NP/gK:")
    print(pd.DataFrame(init_scale_diag["per_dataset"]).to_string(index=False))
    if args.np_a0_scan is not None:
        scan_values = list(args.np_a0_scan) if len(args.np_a0_scan) > 0 else [0.0, 1e-4, 0.01, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35]
        scan_df = initial_damping_scan(
            df=df,
            kernel=kernel,
            b_grid=b_grid,
            y_term=y_term,
            target=y_central,
            sigma=sigma_central,
            a0_values=scan_values,
            np_min_a=float(args.np_min_a),
            learn_gk=bool(args.learn_gk),
            gk_b0=float(args.gk_b0),
            q0=float(args.q0),
            gk_mode=str(args.gk_mode),
            gk_bmax=float(args.gk_bmax),
            cs_log=str(args.cs_log),
            cs_kernel_convention=str(args.cs_kernel_convention),
        )
        print("initial F_NP damping scan:")
        print(scan_df.to_string(index=False))
        Path(args.out).mkdir(parents=True, exist_ok=True)
        scan_df.to_csv(Path(args.out) / "initial_damping_scan.csv", index=False)
        if args.np_a0_scan_exit:
            print("np-a0 scan complete; exiting before training.")
            return
    if args.strict_positive_init and int(init_scale_diag.get("n_nonpositive_baseline", 0)) > 0:
        raise SystemExit("Initial W+Y has nonpositive rows; increase --np-a0 or run --np-a0-scan before training.")
    if args.auto_global_norm_init:
        init = choose_auto_global_norm(init_scale_diag, args.auto_global_norm_method)
        args.global_norm_init = float(np.clip(init, 1.0e-3, 1.0e3))
        args.learn_global_norm = True
        print(f"auto global norm init from initial F_NP/gK ({args.auto_global_norm_method}): {args.global_norm_init:.6g}")
    if args.learn_global_norm and args.fit_dataset_norms:
        print("NOTE: both --learn-global-norm and --fit-dataset-norms are enabled. This is useful for diagnostics but introduces a near-degeneracy in the overall scale; prefer one at a time for clean interpretation.")
    target = None
    if args.replica_seed is not None:
        target = make_experimental_replica(df, rep_cfg, int(args.replica_seed))
        print(f"using experimental replica seed {args.replica_seed} with norm_source={args.norm_source}, ptp_source={args.ptp_source}")

    if args.check_only:
        print("check-only complete; no training run.")
        return

    if str(args.np_shape_mode) == "monotone":
        print("using monotone-integral F_NP scaffold: F=exp[-∫_0^b 2 b' A_theta(x,b') db']; this guarantees F_NP decreases with b on the grid")
    np_factor = FilmNPFactor(
        width=int(args.np_width),
        cond_width=int(args.np_cond_width),
        n_blocks=int(args.np_blocks),
        a0=float(args.np_a0),
        min_a=float(args.np_min_a),
        a_mode=str(args.np_a_mode),
        exponent_clip=float(args.fnp_exponent_clip),
        shape_mode=str(args.np_shape_mode),
        a_smooth_sigma=float(args.np_a_smooth_sigma),
        a_tail_amp=float(args.np_a_tail_amp),
        a_tail_b0=float(args.np_a_tail_b0),
        a_tail_width=float(args.np_a_tail_width),
        dtype=dtype,
    ).to(device)
    gk_model: nn.Module
    if args.learn_gk:
        if str(args.gk_mode) == "bounded":
            gk_model = BoundedGKModel(
                width=int(args.gk_width),
                n_layers=int(args.gk_layers),
                b0=float(args.gk_b0),
                bmax=float(args.gk_bmax),
                dtype=dtype,
            ).to(device)
            print(f"using bounded nonperturbative CS kernel: 0 <= B(b) <= {float(args.gk_bmax):.6g}, so G_CS(b) <= b^2 Bmax")
            print(f"soft-Q evolution convention: exp[-G_CS(b)*L_Q], cs_log={args.cs_log}, cs_kernel_convention={args.cs_kernel_convention}")
        else:
            gk_model = GKModel(width=int(args.gk_width), n_layers=int(args.gk_layers), b0=float(args.gk_b0), dtype=dtype).to(device)
            print("WARNING: using unbounded nonperturbative CS kernel. This is diagnostic only and can be degenerate with F_NP in fixed-target-only fits.")
            print(f"soft-Q evolution convention: exp[-G_CS(b)*L_Q], cs_log={args.cs_log}, cs_kernel_convention={args.cs_kernel_convention}")
    else:
        gk_model = ZeroGK().to(device)

    model = PrecomputedKernelModel(
        b_grid=b_grid,
        kernel_matrix=kernel,
        np_factor=np_factor,
        gk_model=gk_model,
        q0=float(args.q0),
        cs_log=str(args.cs_log),
        cs_kernel_convention=str(args.cs_kernel_convention),
        learn_global_norm=bool(args.learn_global_norm),
        global_norm_init=float(args.global_norm_init),
        dtype=dtype,
        device=device,
    ).to(device)
    if args.init_model_state is not None:
        state_path = Path(args.init_model_state)
        print(f"warm-starting model from {state_path}")
        state = torch.load(state_path, map_location=device)

        # Allow either a raw state_dict or a wrapped checkpoint.
        if isinstance(state, dict) and "model_state" in state and isinstance(state["model_state"], dict):
            state = state["model_state"]

        own = model.state_dict()
        filtered = {}
        skipped = []

        for k, v in state.items():
            if k in {"b", "kernel_matrix"}:
                skipped.append((k, "buffer skipped"))
                continue
            if k not in own:
                skipped.append((k, "not in current model"))
                continue
            if tuple(own[k].shape) != tuple(v.shape):
                skipped.append((k, f"shape {tuple(v.shape)} != {tuple(own[k].shape)}"))
                continue
            if torch.is_tensor(v):
                filtered[k] = v.to(device=own[k].device, dtype=own[k].dtype)
            else:
                filtered[k] = v

        missing, unexpected = model.load_state_dict(filtered, strict=False)
        print(f"warm-start loaded {len(filtered)} tensors from {state_path}")
        if skipped:
            print("warm-start skipped first entries:", skipped[:8])
        print(f"warm-start missing keys count: {len(missing)}")
        print(f"warm-start unexpected keys count: {len(unexpected)}")

    data = TensorData(df, y_term=y_term, target=target, dtype=dtype, device=device)

    dataset_norm_init_scales = None
    if args.init_dataset_norms_from is not None:
        norm_path = Path(args.init_dataset_norms_from)
        if norm_path.is_dir():
            norm_path = norm_path / "dataset_norms.csv"
        if not norm_path.exists():
            raise FileNotFoundError(f"Missing dataset norm initializer: {norm_path}")
        norm_df = pd.read_csv(norm_path)
        if not {"dataset", "norm_scale"}.issubset(norm_df.columns):
            raise ValueError(
                f"{norm_path} must contain dataset,norm_scale columns"
            )
        norm_map = dict(zip(norm_df["dataset"].astype(str), norm_df["norm_scale"].astype(float)))
        missing_norms = [name for name in data.dataset_names if name not in norm_map]
        if missing_norms:
            raise ValueError(f"Missing initial norms for datasets: {missing_norms}")
        dataset_norm_init_scales = tuple(norm_map[name] for name in data.dataset_names)

    train_cfg = TrainConfig(
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        grad_clip=float(args.grad_clip),
        seed=int(args.seed),
        patience=int(args.patience),
        min_delta=float(args.min_delta),
        restore_best=not bool(args.no_restore_best),
        log_every=int(args.log_every),
        lambda_a_l2=float(args.lambda_a_l2),
        lambda_gk_l2=float(args.lambda_gk_l2),
        lambda_fnp_mono=float(args.lambda_fnp_mono),
        lambda_fnp_bcurv=float(args.lambda_fnp_bcurv),
        lambda_fnp_xcurv=float(args.lambda_fnp_xcurv),
        lambda_fnp_pair_bcurv=float(args.lambda_fnp_pair_bcurv),
        lambda_fnp_local_bcurv=float(args.lambda_fnp_local_bcurv),
        fnp_local_bcurv_x_values=tuple(
            float(v) for v in args.fnp_local_bcurv_x_values
        ),
        fnp_local_bcurv_bmin=float(args.fnp_local_bcurv_bmin),
        fnp_local_bcurv_bmax=float(args.fnp_local_bcurv_bmax),
        lambda_fnp_lowpass=float(args.lambda_fnp_lowpass),
        fnp_lowpass_x_values=tuple(
            float(v) for v in args.fnp_lowpass_x_values
        ),
        fnp_lowpass_bmin=float(args.fnp_lowpass_bmin),
        fnp_lowpass_bmax=float(args.fnp_lowpass_bmax),
        fnp_lowpass_sigma=float(args.fnp_lowpass_sigma),
        lambda_fnp_ratecurv=float(args.lambda_fnp_ratecurv),
        fnp_ratecurv_bmin=float(args.fnp_ratecurv_bmin),
        fnp_ratecurv_bmax=float(args.fnp_ratecurv_bmax),
        lambda_fnp_tail=float(args.lambda_fnp_tail),
        fnp_tail_bmin=float(args.fnp_tail_bmin),
        fnp_tail_target=float(args.fnp_tail_target),
        mono_tol=float(args.mono_tol),
        reg_backoff_start_frac=float(args.reg_backoff_start_frac),
        reg_backoff_end_frac=float(args.reg_backoff_end_frac),
        reg_final_scale=float(args.reg_final_scale),
        fit_dataset_norms=bool(args.fit_dataset_norms),
        lambda_dataset_norm=float(args.lambda_dataset_norm),
        dataset_norm_init_scales=dataset_norm_init_scales,
    )
    history, train_meta = fit_model(model, data, train_cfg)
    config = vars(args).copy()
    config["replica_config"] = asdict(rep_cfg)
    config["backend_meta"] = backend_meta
    config["note"] = (
        "This script trains only the nonperturbative bT-space factors. "
        "The current internal CSS backend is a pilot backend; y_mode=zero is not a final W+Y finite-tail calculation. "
        "When --fit-dataset-norms is used, correlated dataset-normalization uncertainties are profiled with Gaussian penalties."
    )
    write_outputs(model=model, data=data, history=history, train_meta=train_meta, config=config, out_dir=args.out)


if __name__ == "__main__":
    main()
