#!/usr/bin/env python3
"""Traditional single-panel k_T-space TMDPDF plot with 68% bands.

This script is the k_T companion to the traditional b_T plot.  It reads the
regularized k_T-space outputs from construct_v23a_regularized_kspace_tmd.py:

  v23a_regularized_kspace_bands.csv
  v23a_regularized_kspace_replica_long.csv

Optionally, it computes the dashed central curve by applying the same
regularized finite-b_T Hankel transform to a central b-space grid.

Default example:
  d quark, x=0.1, Q=10 GeV, ftilde, 0 <= kT <= 4 GeV.

Transform convention for optional central curve:
  f(kT) = 1/(2*pi) int db b J0(kb) ftilde(b)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from scipy.interpolate import PchipInterpolator
from scipy.special import j0

try:
    from numpy import trapezoid as _trapezoid
except Exception:
    from scipy.integrate import trapezoid as _trapezoid


FLAVOR_TO_PID = {"dbar": -1, "ubar": -2, "d": 1, "u": 2}

YLABEL = {
    "ftilde": r"$f_1(x,k_T)$",
    "x_ftilde": r"$x f_1(x,k_T)$",
    "F_NP": r"${\cal F}[F_{\rm NP}](k_T)$",
}


def first_col(df: pd.DataFrame, names: Iterable[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for name in names:
        if name in df.columns:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def add_derived_bspace(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "ftilde" in df.columns:
        if "x_ftilde" not in df.columns and "x" in df.columns:
            df["x_ftilde"] = pd.to_numeric(df["x"], errors="coerce") * pd.to_numeric(df["ftilde"], errors="coerce")
        if "b_ftilde" not in df.columns and "bT" in df.columns:
            df["b_ftilde"] = pd.to_numeric(df["bT"], errors="coerce") * pd.to_numeric(df["ftilde"], errors="coerce")
        if "b_x_ftilde" not in df.columns and {"bT", "x"}.issubset(df.columns):
            df["b_x_ftilde"] = (
                pd.to_numeric(df["bT"], errors="coerce")
                * pd.to_numeric(df["x"], errors="coerce")
                * pd.to_numeric(df["ftilde"], errors="coerce")
            )
    return df


def taper_window(b: np.ndarray, start_fraction: float) -> np.ndarray:
    if start_fraction >= 1:
        return np.ones_like(b)
    B = float(np.max(b))
    start = float(start_fraction) * B
    w = np.ones_like(b)
    m = b > start
    if np.any(m):
        t = (b[m] - start) / max(B - start, 1e-300)
        w[m] = 0.5 * (1.0 + np.cos(np.pi * t))
    w[b >= B] = 0.0
    return w


def trapezoid_weights_uniform(x: np.ndarray) -> np.ndarray:
    dx = float(x[1] - x[0])
    w = np.full_like(x, dx, dtype=float)
    w[0] *= 0.5
    w[-1] *= 0.5
    return w


def fit_tail_expb(b: np.ndarray, y: np.ndarray, b_fit_min: float, eps: float) -> float:
    m = (b >= b_fit_min) & np.isfinite(y) & (np.abs(y) > eps)
    if np.sum(m) < 3:
        return 1.0
    slope, _intercept = np.polyfit(b[m], np.log(np.maximum(np.abs(y[m]), eps)), 1)
    return max(1e-4, -float(slope))


def fit_tail_expb2(b: np.ndarray, y: np.ndarray, b_fit_min: float, eps: float) -> float:
    m = (b >= b_fit_min) & np.isfinite(y) & (np.abs(y) > eps)
    if np.sum(m) < 3:
        return 0.10
    slope, _intercept = np.polyfit(b[m] ** 2, np.log(np.maximum(np.abs(y[m]), eps)), 1)
    return max(1e-5, -float(slope))


def extend_curve(
    b_in: np.ndarray,
    y_in: np.ndarray,
    b_grid: np.ndarray,
    *,
    tail_mode: str,
    tail_fit_bmin: float | None,
    eps: float,
) -> np.ndarray:
    order = np.argsort(b_in)
    b = np.asarray(b_in[order], dtype=float)
    y = np.asarray(y_in[order], dtype=float)
    good = np.isfinite(b) & np.isfinite(y)
    b = b[good]
    y = y[good]
    tmp = pd.DataFrame({"b": b, "y": y}).groupby("b", observed=False)["y"].mean().reset_index()
    b = tmp["b"].to_numpy(float)
    y = tmp["y"].to_numpy(float)

    if len(b) < 3:
        raise ValueError("not enough b points to transform")

    b0, bN = float(b[0]), float(b[-1])
    yN = float(y[-1])
    signN = 1.0 if yN >= 0 else -1.0
    yN_abs = max(abs(yN), eps)
    fit_min = float(tail_fit_bmin) if tail_fit_bmin is not None else max(0.65 * bN, bN - 2.0)

    interp = PchipInterpolator(b, y, extrapolate=False)
    out = interp(np.clip(b_grid, b0, bN))
    out[b_grid < b0] = y[0]

    beyond = b_grid > bN
    if np.any(beyond):
        if tail_mode == "expb2":
            a = fit_tail_expb2(b, y, fit_min, eps)
            out[beyond] = signN * yN_abs * np.exp(-a * (b_grid[beyond] ** 2 - bN ** 2))
        elif tail_mode == "expb":
            lam = fit_tail_expb(b, y, fit_min, eps)
            out[beyond] = signN * yN_abs * np.exp(-lam * (b_grid[beyond] - bN))
        elif tail_mode == "taper":
            out[beyond] = 0.0
        elif tail_mode == "zero":
            out[beyond] = 0.0
        elif tail_mode == "hold":
            out[beyond] = yN
        else:
            raise ValueError(f"unknown tail mode: {tail_mode}")

    if tail_mode == "taper":
        m = b_grid > fit_min
        if np.any(m):
            t = np.clip((b_grid[m] - fit_min) / max(bN - fit_min, 1e-300), 0, 1)
            out[m] *= 0.5 * (1 + np.cos(np.pi * t))
            out[b_grid >= bN] = 0.0

    return np.asarray(out, dtype=float)


def transform_central_from_bspace(
    central_grid: Path,
    *,
    quantity: str,
    flavor: str,
    pid: int,
    x: float,
    Q: float,
    k_grid: np.ndarray,
    tail_mode: str,
    b_transform_max: float,
    n_b_transform: int,
    end_taper_start_fraction: float,
    tail_fit_bmin: float | None,
) -> pd.DataFrame:
    df = pd.read_csv(central_grid)
    df = add_derived_bspace(df)
    if quantity not in df.columns:
        raise SystemExit(f"central grid missing quantity {quantity}. Columns: {list(df.columns)}")

    m = (
        (df["pid"].astype(int) == int(pid))
        & df["flavor"].astype(str).eq(flavor)
        & np.isclose(pd.to_numeric(df["x"], errors="coerce"), float(x), rtol=0, atol=1e-10)
        & np.isclose(pd.to_numeric(df["Q"], errors="coerce"), float(Q), rtol=0, atol=1e-10)
    )
    sub = df[m].copy()
    if sub.empty:
        raise SystemExit(f"No central b-space curve for {flavor}, x={x}, Q={Q}")

    b_grid = np.linspace(0.0, float(b_transform_max), int(n_b_transform))
    win = taper_window(b_grid, float(end_taper_start_fraction))
    trap_w = trapezoid_weights_uniform(b_grid)
    quad_w = b_grid * win * trap_w / (2.0 * np.pi)

    y_ext = extend_curve(
        pd.to_numeric(sub["bT"], errors="coerce").to_numpy(float),
        pd.to_numeric(sub[quantity], errors="coerce").to_numpy(float),
        b_grid,
        tail_mode=tail_mode,
        tail_fit_bmin=tail_fit_bmin,
        eps=1e-300,
    )
    J = j0(np.outer(k_grid, b_grid))
    val = J @ (y_ext * quad_w)
    return pd.DataFrame({"kT": k_grid, "central": val})


def load_curve(
    band_dir: Path,
    *,
    quantity: str,
    flavor: str,
    pid: int,
    x: float,
    Q: float,
    k_max: float,
) -> pd.DataFrame:
    p = band_dir / "v23a_regularized_kspace_bands.csv"
    if not p.exists():
        raise SystemExit(f"Missing k-space band CSV: {p}")

    df = pd.read_csv(p)
    m = (
        df["quantity"].astype(str).eq(quantity)
        & df["pid"].astype(int).eq(int(pid))
        & df["flavor"].astype(str).eq(flavor)
        & np.isclose(pd.to_numeric(df["x"], errors="coerce"), float(x), rtol=0, atol=1e-10)
        & np.isclose(pd.to_numeric(df["Q"], errors="coerce"), float(Q), rtol=0, atol=1e-10)
        & (pd.to_numeric(df["kT"], errors="coerce") <= float(k_max))
    )
    out = df[m].copy()
    if out.empty:
        raise SystemExit(f"No k-space curve found for {quantity}, {flavor}, x={x}, Q={Q}")
    return out.sort_values("kT")


def load_thin_replicas(
    band_dir: Path,
    *,
    quantity: str,
    flavor: str,
    pid: int,
    x: float,
    Q: float,
    k_max: float,
    n: int,
    seed: int,
) -> list[pd.DataFrame]:
    if n <= 0:
        return []
    p = band_dir / "v23a_regularized_kspace_replica_long.csv"
    if not p.exists():
        return []
    df = pd.read_csv(p)
    key_col = "_replica_key" if "_replica_key" in df.columns else None
    if key_col is None:
        if {"seed", "pdf_member"}.issubset(df.columns):
            df["_replica_key"] = df["seed"].astype(str) + "|pdf" + df["pdf_member"].astype(str)
            key_col = "_replica_key"
        else:
            return []
    m = (
        df["quantity"].astype(str).eq(quantity)
        & df["pid"].astype(int).eq(int(pid))
        & df["flavor"].astype(str).eq(flavor)
        & np.isclose(pd.to_numeric(df["x"], errors="coerce"), float(x), rtol=0, atol=1e-10)
        & np.isclose(pd.to_numeric(df["Q"], errors="coerce"), float(Q), rtol=0, atol=1e-10)
        & (pd.to_numeric(df["kT"], errors="coerce") <= float(k_max))
    )
    df = df[m].copy()
    keys = sorted(df[key_col].dropna().astype(str).unique().tolist())
    if len(keys) > n:
        rng = np.random.default_rng(int(seed))
        keys = sorted(rng.choice(keys, size=int(n), replace=False).tolist())
    curves = []
    for key in keys:
        g = df[df[key_col].astype(str).eq(str(key))].sort_values("kT")
        curves.append(g[["kT", "value"]].copy())
    return curves


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--band-dir", required=True)
    ap.add_argument("--central-bspace-grid", default="")
    ap.add_argument("--quantity", default="ftilde", choices=["ftilde", "x_ftilde", "F_NP"])
    ap.add_argument("--flavor", default="d", choices=["u", "d", "ubar", "dbar"])
    ap.add_argument("--pid", type=int, default=None)
    ap.add_argument("--x", type=float, default=0.10)
    ap.add_argument("--Q", type=float, default=10.0)
    ap.add_argument("--k-max", type=float, default=4.0)
    ap.add_argument("--title", default="TMD PDFs")
    ap.add_argument("--label", default="v23a FT-DY")
    ap.add_argument("--band-label", default="68% exp+PDF overlay")
    ap.add_argument("--central-label", default="central fit, PDF0")
    ap.add_argument("--thin-replicas", type=int, default=0)
    ap.add_argument("--thin-alpha", type=float, default=0.10)
    ap.add_argument("--random-seed", type=int, default=303)
    ap.add_argument("--with-inset", action="store_true")
    ap.add_argument("--show-zero", action="store_true")
    ap.add_argument("--out", required=True)

    # Needed only if central-bspace-grid is supplied.
    ap.add_argument("--tail-mode", choices=["expb2", "expb", "taper", "zero", "hold"], default="expb2")
    ap.add_argument("--b-transform-max", type=float, default=24.0)
    ap.add_argument("--n-b-transform", type=int, default=6001)
    ap.add_argument("--end-taper-start-fraction", type=float, default=0.92)
    ap.add_argument("--tail-fit-bmin", type=float, default=None)
    args = ap.parse_args()

    pid = args.pid if args.pid is not None else FLAVOR_TO_PID[args.flavor]
    band_dir = Path(args.band_dir)
    curve = load_curve(
        band_dir,
        quantity=args.quantity,
        flavor=args.flavor,
        pid=pid,
        x=args.x,
        Q=args.Q,
        k_max=args.k_max,
    )

    k_grid = pd.to_numeric(curve["kT"], errors="coerce").to_numpy(float)
    central = None
    if args.central_bspace_grid:
        central = transform_central_from_bspace(
            Path(args.central_bspace_grid),
            quantity=args.quantity,
            flavor=args.flavor,
            pid=pid,
            x=args.x,
            Q=args.Q,
            k_grid=k_grid,
            tail_mode=args.tail_mode,
            b_transform_max=args.b_transform_max,
            n_b_transform=args.n_b_transform,
            end_taper_start_fraction=args.end_taper_start_fraction,
            tail_fit_bmin=args.tail_fit_bmin,
        )

    thin = load_thin_replicas(
        band_dir,
        quantity=args.quantity,
        flavor=args.flavor,
        pid=pid,
        x=args.x,
        Q=args.Q,
        k_max=args.k_max,
        n=args.thin_replicas,
        seed=args.random_seed,
    )

    # Diagnostics.
    eps = 1e-300
    curve["abs_halfwidth"] = 0.5 * (curve["q84"] - curve["q16"])
    curve["rel_halfwidth"] = curve["abs_halfwidth"] / np.maximum(np.abs(curve["median"]), eps)
    peak = float(np.nanmax(np.abs(curve["median"])))
    curve["active"] = np.abs(curve["median"]) > 0.05 * max(peak, eps)
    active = curve[curve["active"]]

    min_over_peak = float(np.nanmin(curve["median"]) / max(peak, eps))
    neg_fraction = float(np.mean(curve["median"] < 0))
    diag = {
        "quantity": args.quantity,
        "flavor": args.flavor,
        "pid": int(pid),
        "x": float(args.x),
        "Q": float(args.Q),
        "k_max": float(args.k_max),
        "peak_abs_median": peak,
        "min_over_peak": min_over_peak,
        "negative_point_fraction": neg_fraction,
        "relative_68_halfwidth_median_active": float(active["rel_halfwidth"].median()) if len(active) else None,
        "relative_68_halfwidth_p90_active": float(active["rel_halfwidth"].quantile(0.90)) if len(active) else None,
        "relative_68_halfwidth_max_active": float(active["rel_halfwidth"].max()) if len(active) else None,
        "n_thin_replicas": int(len(thin)),
        "transform_label": (
            f"regularized finite-bT Hankel transform, tail={args.tail_mode}, "
            f"bmax={args.b_transform_max:g} GeV^-1"
        ),
    }

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.linewidth": 1.25,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "legend.frameon": False,
    })

    fig, ax = plt.subplots(figsize=(7.4, 5.2), dpi=180)

    if args.show_zero:
        ax.axhline(0, color="0.5", lw=0.8, zorder=0)

    for g in thin:
        ax.plot(
            g["kT"],
            g["value"],
            color="0.45",
            alpha=float(args.thin_alpha),
            lw=0.7,
            label="thin replicas" if g is thin[0] else None,
            zorder=1,
        )

    ax.fill_between(
        curve["kT"].to_numpy(float),
        curve["q16"].to_numpy(float),
        curve["q84"].to_numpy(float),
        color="tab:blue",
        alpha=0.25,
        linewidth=0,
        label=args.band_label,
        zorder=2,
    )
    ax.plot(curve["kT"], curve["median"], color="tab:blue", lw=2.2, label=f"{args.label} median", zorder=4)

    if central is not None:
        ax.plot(
            central["kT"],
            central["central"],
            color="black",
            lw=1.7,
            ls="--",
            alpha=0.9,
            label=args.central_label,
            zorder=5,
        )

    ymax = max(float(np.nanmax(curve["q84"])), float(np.nanmax(curve["median"])))
    ymin = min(float(np.nanmin(curve["q16"])), float(np.nanmin(curve["median"])))
    if central is not None:
        ymax = max(ymax, float(np.nanmax(central["central"])))
        ymin = min(ymin, float(np.nanmin(central["central"])))
    if ymin >= 0:
        ax.set_ylim(0, ymax * 1.18 if ymax > 0 else 1.0)
    else:
        pad = 0.12 * (ymax - ymin)
        ax.set_ylim(ymin - pad, ymax + pad)

    ax.set_xlim(0, args.k_max)
    ax.set_title(args.title, fontsize=24, fontweight="bold", pad=15)
    ax.set_xlabel(r"$k_T\;(\mathrm{GeV})$", fontsize=18)
    ax.set_ylabel(YLABEL.get(args.quantity, args.quantity), fontsize=18)

    ax.text(
        0.03,
        0.93,
        rf"$x={args.x:g}\quad Q={args.Q:g}\,\mathrm{{GeV}}\quad {args.flavor}$-quark",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2},
    )

    ax.tick_params(which="major", length=6, width=1.2, labelsize=13)
    ax.tick_params(which="minor", length=3, width=1.0)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(alpha=0.18)

    ax.legend(loc="upper right", fontsize=10)

    if args.with_inset:
        inset = inset_axes(ax, width="37%", height="28%", loc="center right", borderpad=1.2)
        inset.plot(curve["kT"], 100 * curve["rel_halfwidth"], color="black", lw=1.2)
        inset.set_title("68% halfwidth", fontsize=8)
        inset.set_xlabel(r"$k_T$", fontsize=8)
        inset.set_ylabel("%", fontsize=8)
        inset.tick_params(labelsize=7, direction="in", top=True, right=True)
        inset.set_xlim(0, args.k_max)
        mh = float(np.nanmax(100 * curve["rel_halfwidth"].replace([np.inf, -np.inf], np.nan)))
        inset.set_ylim(0, mh * 1.15 if np.isfinite(mh) and mh > 0 else 1.0)

    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    curve.to_csv(out.with_suffix(".curve.csv"), index=False)
    out.with_suffix(".diagnostics.json").write_text(json.dumps(diag, indent=2) + "\n")

    print(json.dumps(diag, indent=2))
    print("wrote:", out)
    print("wrote:", out.with_suffix(".png"))
    print("wrote:", out.with_suffix(".curve.csv"))
    print("wrote:", out.with_suffix(".diagnostics.json"))


if __name__ == "__main__":
    main()
