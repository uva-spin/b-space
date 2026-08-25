#!/usr/bin/env python3
"""Construct first v22 scheme-defined b-space TMDPDF grids.

This builds the unweighted b-space quark TMDPDF

    ftilde_1,q/p(x,bT; mu=Q, zeta=Q^2)

using the v22 perturbative OPE leg, single-leg Sudakov evolution, and the
fitted F_NP grid from a central refit.

Definition used here:

    ftilde_q(x,bT;Q,Q^2)
      = [C_{q<-j}^{NLO} tensor f_j](x; mu_b, zeta_b=mu_b^2)
        * exp[-S(bT,Q)/2]
        * F_NP(x,bT)

The DY hard factor is intentionally not included in a single-hadron TMDPDF.

This is the first central grid, not an uncertainty ensemble.
"""

from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.special import j0

from v22.src.dy_w_nlo_reference import build_quark_leg_nlo
from v22.src.small_b_profile import b_ope_profile


PID_LABEL = {
    2: "u",
    1: "d",
    3: "s",
    -2: "ubar",
    -1: "dbar",
    -3: "sbar",
}


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


def construct_cfg(backend, args):
    overrides: dict[str, Any] = {
        "b_min": max(float(args.b_min), 1.0e-4),
        "b_max": float(args.b_max),
        "n_b": int(args.n_backend_b),
        "bstar_bmax": float(args.bstar_bmax),
        "mu_min": float(args.mu_min),
        "mu_floor_smooth_width": float(args.mu_floor_smooth_width),
        "cap_mub_at_Q": True,
        "q0": 2.0,
        "resum_order": str(args.resum_order),
        "match_order": "nlo",
        "nf": 5,
        "n_sudakov_quad": int(args.n_sudakov_quad),
        "prefactor_scheme": "oldA_to_CS",
        "global_norm": 1.0,
        "flavors": tuple(sorted({abs(int(pid)) for pid in args.pids} | {3})),
        "target_mode": "nuclear_isospin",
        "y_mode": "zero",
    }

    config_type = backend.CSSConfig
    if is_dataclass(config_type):
        valid = {field.name for field in fields(config_type)}
        overrides = {
            key: value
            for key, value in overrides.items()
            if key in valid
        }

    return config_type(**overrides)


def find_fnp_file(run: Path) -> Path:
    candidates = [
        run / "fnp_debug_grid.csv",
        run / "run" / "fnp_debug_grid.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    found = sorted(run.rglob("fnp_debug_grid.csv"))
    if found:
        return found[0]

    raise SystemExit(
        f"Could not find fnp_debug_grid.csv under {run}. "
        "Run the central fit or point --run to the fit output directory."
    )


def infer_column(frame: pd.DataFrame, names: list[str], purpose: str) -> str:
    lower = {column.lower(): column for column in frame.columns}
    for name in names:
        if name in frame.columns:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    raise SystemExit(
        f"Could not infer {purpose} column. Available columns: {list(frame.columns)}"
    )


class FNPGrid:
    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        allow_x_interpolation: bool = False,
    ) -> None:
        self.x_col = infer_column(frame, ["x"], "x")
        self.b_col = infer_column(frame, ["bT", "b", "bt", "b_GeV_inv"], "bT")
        self.f_col = infer_column(
            frame,
            ["F_NP", "F_NP_mean", "fnp", "FNP", "f_np"],
            "F_NP",
        )
        self.allow_x_interpolation = bool(allow_x_interpolation)

        self.frame = frame[
            [self.x_col, self.b_col, self.f_col]
        ].dropna().copy()
        self.frame = self.frame.sort_values([self.x_col, self.b_col])
        self.x_values = np.array(sorted(self.frame[self.x_col].unique()), dtype=float)

        self.by_x: dict[float, tuple[np.ndarray, np.ndarray]] = {}
        for x, group in self.frame.groupby(self.x_col, observed=False):
            b = group[self.b_col].to_numpy(float)
            f = group[self.f_col].to_numpy(float)
            order = np.argsort(b)
            self.by_x[float(x)] = (b[order], f[order])

    def evaluate_at_known_x(self, x: float, b: float) -> float:
        # Snap to the nearest available x if it is numerically identical.
        idx = int(np.argmin(np.abs(self.x_values - float(x))))
        x0 = float(self.x_values[idx])
        if abs(x0 - float(x)) > 5.0e-8:
            raise KeyError(x)
        b_grid, f_grid = self.by_x[x0]
        return float(np.interp(float(b), b_grid, f_grid, left=f_grid[0], right=f_grid[-1]))

    def __call__(self, x: float, b: float) -> float:
        try:
            return self.evaluate_at_known_x(x, b)
        except KeyError:
            if not self.allow_x_interpolation:
                raise SystemExit(
                    f"x={x} is not in the fitted F_NP debug grid. "
                    f"Available x values: {self.x_values.tolist()}. "
                    "Use --allow-x-interpolation only for explicitly labeled extrapolation/interpolation."
                )

        xs = self.x_values
        if float(x) < xs.min() or float(x) > xs.max():
            raise SystemExit(
                f"x={x} lies outside the F_NP grid range [{xs.min()}, {xs.max()}]. "
                "This script does not extrapolate F_NP outside fitted support."
            )

        hi = int(np.searchsorted(xs, float(x), side="right"))
        lo = hi - 1
        x_lo = float(xs[lo])
        x_hi = float(xs[hi])
        f_lo = self.evaluate_at_known_x(x_lo, b)
        f_hi = self.evaluate_at_known_x(x_hi, b)
        t = (float(x) - x_lo) / (x_hi - x_lo)
        return float((1.0 - t) * f_lo + t * f_hi)


def proton_density(pdf, pid: int, mu: float) -> Callable[[float], float]:
    def evaluate(x: float) -> float:
        x = float(x)
        if not 0.0 < x < 1.0:
            return 0.0
        return float(pdf.xf_proton(int(pid), x, mu)) / x

    return evaluate


def compute_curve(
    *,
    backend,
    cfg,
    pdf,
    fnp: FNPGrid,
    x: float,
    Q: float,
    pid: int,
    b_values: np.ndarray,
    C5: float,
    profile_power: float,
    profile_kind: str,
    epsabs: float,
    epsrel: float,
) -> pd.DataFrame:
    rows = []

    for bT in b_values:
        bT = float(bT)
        b_for_backend = max(bT, 1.0e-4)

        b_star = float(
            np.asarray(
                backend.bstar(
                    b_for_backend,
                    float(cfg.bstar_bmax),
                )
            )
        )
        mu_b = float(backend.mu_b_of_b(b_for_backend, float(Q), cfg))
        zeta_b = mu_b * mu_b
        alpha_s_mu = float(pdf.alphas(mu_b))

        b_pert = b_ope_profile(
            b_star_GeV_inv=b_star,
            Q_GeV=float(Q),
            C5=float(C5),
            power=float(profile_power),
            kind=str(profile_kind),
        )

        leg = build_quark_leg_nlo(
            pid=int(pid),
            x=float(x),
            alpha_s_mu=alpha_s_mu,
            b_pert_GeV_inv=b_pert,
            mu_GeV=mu_b,
            zeta_GeV2=zeta_b,
            quark_pdf=proton_density(pdf, int(pid), mu_b),
            gluon_pdf=proton_density(pdf, 21, mu_b),
            epsabs=float(epsabs),
            epsrel=float(epsrel),
        )

        S = float(backend.sudakov_s(b_for_backend, float(Q), pdf, cfg))
        evol_half = math.exp(-0.5 * S)
        F_NP = float(fnp(float(x), bT))

        ftilde_boundary_born = leg.born
        ftilde_boundary_ope = leg.matched
        ftilde_no_np = ftilde_boundary_ope * evol_half
        ftilde = ftilde_no_np * F_NP

        rows.append({
            "x": float(x),
            "Q": float(Q),
            "mu": float(Q),
            "zeta": float(Q) ** 2,
            "pid": int(pid),
            "flavor": PID_LABEL.get(int(pid), str(pid)),
            "bT": bT,
            "b_star": b_star,
            "b_pert": b_pert,
            "mu_b": mu_b,
            "alpha_s_mu_b": alpha_s_mu,
            "L_b": leg.L_b,
            "born_pdf": leg.born,
            "ope_delta_qq_coeff": leg.delta_qq_coefficient,
            "ope_delta_qg_coeff": leg.delta_qg_coefficient,
            "a_s_mu_b": leg.a_s,
            "ope_boundary_born": ftilde_boundary_born,
            "ope_boundary_nlo": ftilde_boundary_ope,
            "sudakov_S_pair": S,
            "evol_half": evol_half,
            "F_NP": F_NP,
            "ftilde_no_np": ftilde_no_np,
            "ftilde": ftilde,
            "x_ftilde": float(x) * ftilde,
            "b_ftilde": bT * ftilde,
            "b_x_ftilde": bT * float(x) * ftilde,
        })

    return pd.DataFrame(rows)


def hankel_transform_curve(group: pd.DataFrame, k_values: np.ndarray) -> pd.DataFrame:
    group = group.sort_values("bT")
    b = group["bT"].to_numpy(float)
    f = group["ftilde"].to_numpy(float)

    rows = []
    meta = group.iloc[0]
    for kT in k_values:
        value = float(np.trapezoid(b * j0(float(kT) * b) * f, x=b) / (2.0 * math.pi))
        rows.append({
            "x": float(meta["x"]),
            "Q": float(meta["Q"]),
            "mu": float(meta["mu"]),
            "zeta": float(meta["zeta"]),
            "pid": int(meta["pid"]),
            "flavor": str(meta["flavor"]),
            "kT": float(kT),
            "f_kT": value,
            "x_f_kT": float(meta["x"]) * value,
        })
    return pd.DataFrame(rows)


def plot_bspace(long: pd.DataFrame, out: Path) -> None:
    for quantity, ylabel in [
        ("ftilde", r"$\widetilde f_{1,q/p}(x,b_T;Q,Q^2)$"),
        ("x_ftilde", r"$x\,\widetilde f_{1,q/p}(x,b_T;Q,Q^2)$"),
        ("F_NP", r"$F_{\rm NP}(x,b_T)$"),
        ("ope_boundary_nlo", r"$C\otimes f$ at $(\mu_b,\zeta_b)$"),
        ("evol_half", r"$\exp[-S(b_T,Q)/2]$"),
    ]:
        pdf_path = out / f"{quantity}_curves.pdf"
        with PdfPages(pdf_path) as pdf:
            for Q in sorted(long["Q"].unique()):
                for pid in sorted(long["pid"].unique()):
                    page = long[
                        np.isclose(long["Q"], Q)
                        & (long["pid"].astype(int) == int(pid))
                    ]
                    if page.empty:
                        continue
                    fig, ax = plt.subplots(figsize=(8.5, 5.5))
                    for x in sorted(page["x"].unique()):
                        g = page[np.isclose(page["x"], x)].sort_values("bT")
                        ax.plot(g["bT"], g[quantity], label=f"x={x:g}")
                    flavor = str(page["flavor"].iloc[0])
                    ax.set_title(f"{quantity}: {flavor}, Q={Q:g} GeV")
                    ax.set_xlabel(r"$b_T\,[{\rm GeV}^{-1}]$")
                    ax.set_ylabel(ylabel)
                    ax.grid(True, alpha=0.3)
                    ax.legend(ncol=2)
                    fig.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)


def plot_kspace(k_long: pd.DataFrame, out: Path) -> None:
    if k_long.empty:
        return
    pdf_path = out / "kT_ftilde_curves.pdf"
    with PdfPages(pdf_path) as pdf:
        for Q in sorted(k_long["Q"].unique()):
            for pid in sorted(k_long["pid"].unique()):
                page = k_long[
                    np.isclose(k_long["Q"], Q)
                    & (k_long["pid"].astype(int) == int(pid))
                ]
                if page.empty:
                    continue
                fig, ax = plt.subplots(figsize=(8.5, 5.5))
                for x in sorted(page["x"].unique()):
                    g = page[np.isclose(page["x"], x)].sort_values("kT")
                    ax.plot(g["kT"], g["f_kT"], label=f"x={x:g}")
                flavor = str(page["flavor"].iloc[0])
                ax.set_title(f"k-space TMD: {flavor}, Q={Q:g} GeV")
                ax.set_xlabel(r"$k_T\,[{\rm GeV}]$")
                ax.set_ylabel(r"$f_{1,q/p}(x,k_T;Q,Q^2)\,[{\rm GeV}^{-2}]$")
                ax.grid(True, alpha=0.3)
                ax.legend(ncol=2)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="outputs/v22_full_backend_central_refit_stage1_s303")
    parser.add_argument("--backend-script", default="v22/backends/bt_internal_css_backend_v22_full.py")
    parser.add_argument("--pdf-set", default="NNPDF40_nnlo_as_01180")
    parser.add_argument("--pdf-member", type=int, default=0)
    parser.add_argument("--resum-order", default="n3llp")
    parser.add_argument("--pids", nargs="+", type=int, default=[2, 1, -2, -1])
    parser.add_argument("--x-values", nargs="+", type=float, default=[0.15, 0.20, 0.30, 0.40, 0.50])
    parser.add_argument("--Q-values", nargs="+", type=float, default=[5.0, 10.0])
    parser.add_argument("--b-min", type=float, default=0.0)
    parser.add_argument("--b-max", type=float, default=8.0)
    parser.add_argument("--n-b", type=int, default=321)
    parser.add_argument("--n-backend-b", type=int, default=160)
    parser.add_argument("--bstar-bmax", type=float, default=1.5)
    parser.add_argument("--mu-min", type=float, default=1.3)
    parser.add_argument("--mu-floor-smooth-width", type=float, default=0.12)
    parser.add_argument("--n-sudakov-quad", type=int, default=32)
    parser.add_argument("--C5", type=float, default=1.0)
    parser.add_argument("--profile-power", type=float, default=16.0)
    parser.add_argument("--profile-kind", choices=["smooth", "hard"], default="smooth")
    parser.add_argument("--epsabs", type=float, default=1.0e-8)
    parser.add_argument("--epsrel", type=float, default=1.0e-7)
    parser.add_argument("--allow-x-interpolation", action="store_true")
    parser.add_argument("--make-kT", action="store_true")
    parser.add_argument("--kT-max", type=float, default=4.0)
    parser.add_argument("--n-kT", type=int, default=241)
    parser.add_argument("--out", default="plots/v22_scheme_tmd_stage1_s303")
    args = parser.parse_args()

    run = Path(args.run)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    backend = import_from_path(Path(args.backend_script), "v22_scheme_tmd_backend")
    cfg = construct_cfg(backend, args)
    pdf = backend.LHAPDFProvider(args.pdf_set, int(args.pdf_member), use_toy_pdf=False)

    fnp_path = find_fnp_file(run)
    fnp_frame = pd.read_csv(fnp_path)
    fnp = FNPGrid(fnp_frame, allow_x_interpolation=bool(args.allow_x_interpolation))

    b_values = np.linspace(float(args.b_min), float(args.b_max), int(args.n_b))

    pieces = []
    for Q in args.Q_values:
        for pid in args.pids:
            for x in args.x_values:
                pieces.append(
                    compute_curve(
                        backend=backend,
                        cfg=cfg,
                        pdf=pdf,
                        fnp=fnp,
                        x=float(x),
                        Q=float(Q),
                        pid=int(pid),
                        b_values=b_values,
                        C5=float(args.C5),
                        profile_power=float(args.profile_power),
                        profile_kind=str(args.profile_kind),
                        epsabs=float(args.epsabs),
                        epsrel=float(args.epsrel),
                    )
                )

    long = pd.concat(pieces, ignore_index=True)
    long_path = out / "v22_scheme_tmd_bspace_long.csv"
    long.to_csv(long_path, index=False)

    summary = (
        long.groupby(["pid", "flavor", "x", "Q"], observed=False)
        .agg(
            ftilde_b0=("ftilde", "first"),
            ftilde_max=("ftilde", "max"),
            b_at_ftilde_max=("bT", lambda s: float(long.loc[s.index, :].sort_values("ftilde").iloc[-1]["bT"])),
            F_NP_min=("F_NP", "min"),
            F_NP_b8=("F_NP", "last"),
            mu_b_min=("mu_b", "min"),
            mu_b_max=("mu_b", "max"),
            L_b_min=("L_b", "min"),
            L_b_max=("L_b", "max"),
        )
        .reset_index()
    )
    summary_path = out / "v22_scheme_tmd_summary.csv"
    summary.to_csv(summary_path, index=False)

    k_long = pd.DataFrame()
    if args.make_kT:
        k_values = np.linspace(0.0, float(args.kT_max), int(args.n_kT))
        k_pieces = []
        for _, group in long.groupby(["pid", "x", "Q"], observed=False):
            k_pieces.append(hankel_transform_curve(group, k_values))
        k_long = pd.concat(k_pieces, ignore_index=True)
        k_long.to_csv(out / "v22_scheme_tmd_kspace_long.csv", index=False)

    plot_bspace(long, out)
    if args.make_kT:
        plot_kspace(k_long, out)

    metadata = {
        "run": str(run),
        "backend_script": str(Path(args.backend_script).resolve()),
        "fnp_grid": str(fnp_path),
        "pdf_set": args.pdf_set,
        "pdf_member": int(args.pdf_member),
        "x_values": [float(v) for v in args.x_values],
        "Q_values": [float(v) for v in args.Q_values],
        "pids": [int(v) for v in args.pids],
        "definition": (
            "ftilde = [C_q<-j^NLO tensor f_j](x; mu_b,zeta_b=mu_b^2) "
            "* exp[-S(bT,Q)/2] * F_NP(x,bT); hard factor excluded"
        ),
        "warning": (
            "This is a central fitted grid only. It does not include replica, PDF, "
            "scale, profile, nuclear, or model-form uncertainties."
        ),
    }
    (out / "v22_scheme_tmd_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print("\n=== v22 scheme TMD grid created ===")
    print("b-space:", long_path)
    print("summary:", summary_path)
    if args.make_kT:
        print("k-space:", out / "v22_scheme_tmd_kspace_long.csv")
    print("plots:", out)
    print("\nSummary preview:")
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
