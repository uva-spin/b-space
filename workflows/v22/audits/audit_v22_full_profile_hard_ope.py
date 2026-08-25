#!/usr/bin/env python3
"""Audit the one-loop hard+OPE correction over the full existing b profile.

This is the last diagnostic before building a standalone v22 W kernel.

For representative fixed-target DY rows, the script evaluates on the full
backend b grid:

  * the Born luminosity bridge to the old W kernel;
  * the general-scale one-loop q<-q and q<-g OPE corrections;
  * the one-loop DY hard factor at mu_H=Q;
  * the strict O(alpha_s) ratio
        1 + delta_H + delta_OPE;
  * the naive multiplicative ratio, shown only to expose beyond-NLO terms.

The perturbative coefficient functions use b_pert=b_star and
zeta_b=mu_profile^2. The physical large-b coordinate is not inserted into
the OPE.

No nonperturbative factor and no Y term are included in this audit.
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

import numpy as np
import pandas as pd
from scipy.special import j0

from v22.src.css2_ope_nlo_general import (
    general_scale_quark_ope_nlo_components,
)
from v22.src.dy_hard_nlo import dy_hard_nlo_at_Q


FOURIER_NORM = 1.0 / (2.0 * math.pi)


def import_module_from_path(path: Path):
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    name = "v22_full_profile_backend"
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def relative_error(a: float, b: float) -> float:
    return abs(float(a) - float(b)) / max(
        abs(float(a)),
        abs(float(b)),
        1.0e-300,
    )


def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    good = (
        np.isfinite(values)
        & np.isfinite(weights)
        & (weights >= 0.0)
    )
    values = values[good]
    weights = weights[good]

    if len(values) == 0 or float(np.sum(weights)) <= 0.0:
        return math.nan

    order = np.argsort(values)
    values = values[order]
    weights = weights[order]

    cumulative = np.cumsum(weights)
    target = float(quantile) * cumulative[-1]

    return float(np.interp(target, cumulative, values))


def trapezoid_node_weights(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)

    if x.ndim != 1 or len(x) < 2:
        raise ValueError("x must be a one-dimensional grid of length >=2")

    dx = np.diff(x)
    if not np.all(dx > 0.0):
        raise ValueError("x grid must be strictly increasing")

    weights = np.empty_like(x)
    weights[0] = 0.5 * dx[0]
    weights[-1] = 0.5 * dx[-1]

    if len(x) > 2:
        weights[1:-1] = 0.5 * (dx[:-1] + dx[1:])

    return weights


def construct_css_config(backend, args):
    overrides: dict[str, Any] = {
        "b_min": float(args.b_min),
        "b_max": float(args.b_max),
        "n_b": int(args.n_b),
        "bstar_bmax": float(args.bstar_bmax),
        "mu_min": float(args.mu_min),
        "cap_mub_at_Q": True,
        "q0": 2.0,
        "resum_order": str(args.resum_order),
        "match_order": "nlo",
        "nf": 5,
        "n_sudakov_quad": int(args.n_sudakov_quad),
        "prefactor_scheme": "oldA_to_CS",
        "global_norm": 1.0,
        "flavors": tuple(int(pid) for pid in args.flavors),
        "target_mode": str(args.target_mode),
        "y_mode": "zero",
        "mu_floor_smooth_width": float(args.mu_floor_smooth_width),
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


def select_representative_rows(
    frame: pd.DataFrame,
    rows_per_dataset: int,
) -> pd.DataFrame:
    selected = []

    for _, group in frame.groupby("dataset", observed=False, sort=True):
        columns = [
            column
            for column in ["QM", "qT", "row_id"]
            if column in group.columns
        ]
        group = group.sort_values(columns)

        n_pick = min(int(rows_per_dataset), len(group))
        indices = np.unique(
            np.linspace(
                0,
                len(group) - 1,
                n_pick,
                dtype=int,
            )
        )
        selected.append(group.iloc[indices])

    return pd.concat(selected, ignore_index=True)


def proton_density(
    pdf,
    pid: int,
    mu: float,
) -> Callable[[float], float]:
    def evaluate(x: float) -> float:
        x = float(x)
        if not 0.0 < x < 1.0:
            return 0.0
        return float(pdf.xf_proton(int(pid), x, mu)) / x

    return evaluate


def target_density(
    pdf,
    pid: int,
    mu: float,
    *,
    dataset: str,
    target_mode: str,
) -> Callable[[float], float]:
    def evaluate(x: float) -> float:
        x = float(x)
        if not 0.0 < x < 1.0:
            return 0.0
        return (
            float(
                pdf.xf_target(
                    int(pid),
                    x,
                    mu,
                    dataset=dataset,
                    target_mode=target_mode,
                )
            )
            / x
        )

    return evaluate


def leg(
    *,
    x: float,
    alpha_s: float,
    b_star: float,
    mu: float,
    zeta: float,
    quark_pdf: Callable[[float], float],
    gluon_pdf: Callable[[float], float],
    epsabs: float,
    epsrel: float,
) -> dict[str, float]:
    components = general_scale_quark_ope_nlo_components(
        x=float(x),
        alpha_s=float(alpha_s),
        b_pert_GeV_inv=float(b_star),
        mu_GeV=float(mu),
        zeta_GeV2=float(zeta),
        quark_pdf=quark_pdf,
        gluon_pdf=gluon_pdf,
        epsabs=float(epsabs),
        epsrel=float(epsrel),
    )

    return {
        "born": float(components.born_quark),
        "delta_qq": float(components.one_loop_qq_total),
        "delta_qg": float(components.one_loop_qg_total),
        "delta_total": float(components.one_loop_total),
        "a_s": float(components.a_s),
        "L_b": float(components.L_b),
        "l_zeta": float(components.l_zeta),
        "matched": float(components.matched_tmd),
    }


def luminosity_corrections(
    *,
    backend,
    row: pd.Series,
    b_star: float,
    mu: float,
    zeta: float,
    alpha_s: float,
    pdf,
    cfg,
    epsabs: float,
    epsrel: float,
) -> dict[str, float]:
    x1 = float(row["x1"])
    x2 = float(row["x2"])
    dataset = str(row["dataset"])

    g_a = proton_density(pdf, 21, mu)
    g_b = target_density(
        pdf,
        21,
        mu,
        dataset=dataset,
        target_mode=cfg.target_mode,
    )

    born_lum = 0.0
    delta_qq_lum = 0.0
    delta_qg_lum = 0.0
    product_lum = 0.0
    a_s_values: list[float] = []
    L_values: list[float] = []

    for flavor in cfg.flavors:
        pid = abs(int(flavor))
        charge2 = float(backend.CHARGE2.get(pid, 0.0))

        if charge2 == 0.0:
            continue

        q_a = leg(
            x=x1,
            alpha_s=alpha_s,
            b_star=b_star,
            mu=mu,
            zeta=zeta,
            quark_pdf=proton_density(pdf, pid, mu),
            gluon_pdf=g_a,
            epsabs=epsabs,
            epsrel=epsrel,
        )
        qb_a = leg(
            x=x1,
            alpha_s=alpha_s,
            b_star=b_star,
            mu=mu,
            zeta=zeta,
            quark_pdf=proton_density(pdf, -pid, mu),
            gluon_pdf=g_a,
            epsabs=epsabs,
            epsrel=epsrel,
        )
        q_b = leg(
            x=x2,
            alpha_s=alpha_s,
            b_star=b_star,
            mu=mu,
            zeta=zeta,
            quark_pdf=target_density(
                pdf,
                pid,
                mu,
                dataset=dataset,
                target_mode=cfg.target_mode,
            ),
            gluon_pdf=g_b,
            epsabs=epsabs,
            epsrel=epsrel,
        )
        qb_b = leg(
            x=x2,
            alpha_s=alpha_s,
            b_star=b_star,
            mu=mu,
            zeta=zeta,
            quark_pdf=target_density(
                pdf,
                -pid,
                mu,
                dataset=dataset,
                target_mode=cfg.target_mode,
            ),
            gluon_pdf=g_b,
            epsabs=epsabs,
            epsrel=epsrel,
        )

        a_s_values.extend(
            [q_a["a_s"], qb_a["a_s"], q_b["a_s"], qb_b["a_s"]]
        )
        L_values.extend(
            [q_a["L_b"], qb_a["L_b"], q_b["L_b"], qb_b["L_b"]]
        )

        born = (
            q_a["born"] * qb_b["born"]
            + qb_a["born"] * q_b["born"]
        )

        delta_qq = (
            q_a["delta_qq"] * qb_b["born"]
            + q_a["born"] * qb_b["delta_qq"]
            + qb_a["delta_qq"] * q_b["born"]
            + qb_a["born"] * q_b["delta_qq"]
        )

        delta_qg = (
            q_a["delta_qg"] * qb_b["born"]
            + q_a["born"] * qb_b["delta_qg"]
            + qb_a["delta_qg"] * q_b["born"]
            + qb_a["born"] * q_b["delta_qg"]
        )

        product = (
            q_a["matched"] * qb_b["matched"]
            + qb_a["matched"] * q_b["matched"]
        )

        born_lum += charge2 * born
        delta_qq_lum += charge2 * delta_qq
        delta_qg_lum += charge2 * delta_qg
        product_lum += charge2 * product

    if not a_s_values:
        raise RuntimeError("no active flavor channels")

    a_s = float(a_s_values[0])

    if max(abs(value - a_s) for value in a_s_values) > 1.0e-14:
        raise RuntimeError("inconsistent a_s among legs")

    L_b = float(L_values[0])

    if max(abs(value - L_b) for value in L_values) > 1.0e-12:
        raise RuntimeError("inconsistent L_b among legs")

    delta_total_lum = delta_qq_lum + delta_qg_lum

    return {
        "born_luminosity_f": born_lum,
        "delta_qq_coefficient_luminosity": delta_qq_lum,
        "delta_qg_coefficient_luminosity": delta_qg_lum,
        "delta_total_coefficient_luminosity": delta_total_lum,
        "naive_product_luminosity": product_lum,
        "a_s": a_s,
        "L_b": L_b,
        "l_zeta": 0.0,
    }


def classify_region(
    *,
    mu_canonical: float,
    Q: float,
    mu_min: float,
) -> str:
    if mu_canonical > Q:
        return "q_cap"
    if mu_canonical < mu_min:
        return "mu_floor"
    return "canonical"


def evaluate_row(
    *,
    backend,
    row: pd.Series,
    pdf,
    cfg,
    epsabs: float,
    epsrel: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    Q = float(row["QM"])
    qT = float(row["qT"])
    x1 = float(row["x1"])
    x2 = float(row["x2"])
    dataset = str(row["dataset"])

    b = np.asarray(backend.make_b_grid(cfg), dtype=float)
    node_weights = trapezoid_node_weights(b)

    old_w = np.asarray(
        backend.wpert_cs_for_row(row, b, pdf, cfg),
        dtype=float,
    )

    prefactor = float(backend.fixed_target_prefactor_cs(row, cfg))
    hard_factor = float(
        dy_hard_nlo_at_Q(
            Q_GeV=Q,
            alpha_s_at_Q=float(pdf.alphas(Q)),
        )
    )
    hard_fraction = hard_factor - 1.0

    rows: list[dict[str, Any]] = []
    strict_w = np.empty_like(old_w)
    naive_w = np.empty_like(old_w)
    born_reconstructed = np.empty_like(old_w)

    for index, bT in enumerate(b):
        b_star = float(
            np.asarray(
                backend.bstar(
                    float(bT),
                    float(cfg.bstar_bmax),
                )
            )
        )

        mu_canonical = float(backend.C0) / max(b_star, 1.0e-300)
        mu_profile = float(backend.mu_b_of_b(float(bT), Q, cfg))
        zeta_boundary = mu_profile * mu_profile
        alpha_s_mu = float(pdf.alphas(mu_profile))

        luminosity = luminosity_corrections(
            backend=backend,
            row=row,
            b_star=b_star,
            mu=mu_profile,
            zeta=zeta_boundary,
            alpha_s=alpha_s_mu,
            pdf=pdf,
            cfg=cfg,
            epsabs=epsabs,
            epsrel=epsrel,
        )

        born_lum = float(luminosity["born_luminosity_f"])
        born_scale = max(abs(born_lum), 1.0e-300)
        a_s = float(luminosity["a_s"])

        qq_fraction = (
            a_s
            * float(luminosity["delta_qq_coefficient_luminosity"])
            / born_scale
        )
        qg_fraction = (
            a_s
            * float(luminosity["delta_qg_coefficient_luminosity"])
            / born_scale
        )
        ope_fraction = qq_fraction + qg_fraction
        strict_total_fraction = hard_fraction + ope_fraction
        strict_ratio = 1.0 + strict_total_fraction

        naive_leg_ratio = (
            float(luminosity["naive_product_luminosity"])
            / born_scale
        )
        naive_full_ratio = hard_factor * naive_leg_ratio
        beyond_nlo = naive_full_ratio - strict_ratio

        sudakov = float(backend.sudakov_s(float(bT), Q, pdf, cfg))
        common = (
            prefactor
            * FOURIER_NORM
            * x1
            * x2
            * math.exp(-sudakov)
        )
        born_value = common * born_lum

        born_reconstructed[index] = born_value
        strict_w[index] = old_w[index] * strict_ratio
        naive_w[index] = old_w[index] * naive_full_ratio

        bessel_weight = (
            node_weights[index]
            * abs(
                bT
                * j0(qT * bT)
                * old_w[index]
            )
        )

        rows.append({
            "row_id": str(row["row_id"]),
            "dataset": dataset,
            "Q": Q,
            "qT": qT,
            "qT_over_Q": qT / Q,
            "x1": x1,
            "x2": x2,
            "bT": float(bT),
            "b_star": b_star,
            "mu_canonical": mu_canonical,
            "mu_profile": mu_profile,
            "zeta_boundary": zeta_boundary,
            "region": classify_region(
                mu_canonical=mu_canonical,
                Q=Q,
                mu_min=float(cfg.mu_min),
            ),
            "L_b": float(luminosity["L_b"]),
            "alpha_s_mu": alpha_s_mu,
            "a_s_mu": a_s,
            "hard_fraction": hard_fraction,
            "qq_fraction": qq_fraction,
            "qg_fraction": qg_fraction,
            "ope_fraction": ope_fraction,
            "strict_total_fraction": strict_total_fraction,
            "strict_ratio": strict_ratio,
            "naive_leg_ratio": naive_leg_ratio,
            "naive_full_ratio": naive_full_ratio,
            "beyond_nlo_ratio_difference": beyond_nlo,
            "old_W": float(old_w[index]),
            "born_W_reconstructed": born_value,
            "strict_v22_W": float(strict_w[index]),
            "naive_multiplicative_W": float(naive_w[index]),
            "born_bridge_relerr": relative_error(
                old_w[index],
                born_value,
            ),
            "absolute_old_bessel_weight": bessel_weight,
        })

    point_table = pd.DataFrame(rows)

    old_integral = float(
        np.trapezoid(
            b * j0(qT * b) * old_w,
            x=b,
        )
    )
    strict_integral = float(
        np.trapezoid(
            b * j0(qT * b) * strict_w,
            x=b,
        )
    )
    naive_integral = float(
        np.trapezoid(
            b * j0(qT * b) * naive_w,
            x=b,
        )
    )

    summary = {
        "row_id": str(row["row_id"]),
        "dataset": dataset,
        "Q": Q,
        "qT": qT,
        "x1": x1,
        "x2": x2,
        "old_W_bessel_integral": old_integral,
        "strict_v22_W_bessel_integral": strict_integral,
        "naive_v22_W_bessel_integral": naive_integral,
        "strict_integral_over_old": (
            strict_integral / old_integral
            if abs(old_integral) > 1.0e-300
            else math.nan
        ),
        "naive_integral_over_old": (
            naive_integral / old_integral
            if abs(old_integral) > 1.0e-300
            else math.nan
        ),
        "max_born_bridge_relerr": float(
            point_table["born_bridge_relerr"].max()
        ),
        "strict_ratio_min": float(point_table["strict_ratio"].min()),
        "strict_ratio_max": float(point_table["strict_ratio"].max()),
        "max_abs_strict_total_fraction": float(
            np.max(np.abs(point_table["strict_total_fraction"]))
        ),
        "bessel_weighted_abs_total_p50": weighted_quantile(
            np.abs(point_table["strict_total_fraction"].to_numpy(float)),
            point_table["absolute_old_bessel_weight"].to_numpy(float),
            0.50,
        ),
        "bessel_weighted_abs_total_p90": weighted_quantile(
            np.abs(point_table["strict_total_fraction"].to_numpy(float)),
            point_table["absolute_old_bessel_weight"].to_numpy(float),
            0.90,
        ),
        "max_abs_beyond_nlo": float(
            np.max(np.abs(point_table["beyond_nlo_ratio_difference"]))
        ),
        "n_nonpositive_strict_ratio": int(
            np.sum(point_table["strict_ratio"] <= 0.0)
        ),
    }

    return point_table, summary


def region_summary(points: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (dataset, region), group in points.groupby(
        ["dataset", "region"],
        observed=False,
        sort=True,
    ):
        weights = group["absolute_old_bessel_weight"].to_numpy(float)
        abs_total = np.abs(group["strict_total_fraction"].to_numpy(float))

        rows.append({
            "dataset": dataset,
            "region": region,
            "n_points": int(len(group)),
            "b_min": float(group["bT"].min()),
            "b_max": float(group["bT"].max()),
            "L_b_min": float(group["L_b"].min()),
            "L_b_median": float(group["L_b"].median()),
            "L_b_max": float(group["L_b"].max()),
            "qq_fraction_median": float(group["qq_fraction"].median()),
            "qg_fraction_median": float(group["qg_fraction"].median()),
            "ope_fraction_min": float(group["ope_fraction"].min()),
            "ope_fraction_median": float(group["ope_fraction"].median()),
            "ope_fraction_max": float(group["ope_fraction"].max()),
            "strict_total_fraction_min": float(
                group["strict_total_fraction"].min()
            ),
            "strict_total_fraction_median": float(
                group["strict_total_fraction"].median()
            ),
            "strict_total_fraction_max": float(
                group["strict_total_fraction"].max()
            ),
            "strict_ratio_min": float(group["strict_ratio"].min()),
            "strict_ratio_max": float(group["strict_ratio"].max()),
            "bessel_support_fraction": float(
                np.sum(weights)
                / max(
                    np.sum(
                        points.loc[
                            points["dataset"] == dataset,
                            "absolute_old_bessel_weight",
                        ].to_numpy(float)
                    ),
                    1.0e-300,
                )
            ),
            "bessel_weighted_abs_total_p50": weighted_quantile(
                abs_total,
                weights,
                0.50,
            ),
            "bessel_weighted_abs_total_p90": weighted_quantile(
                abs_total,
                weights,
                0.90,
            ),
            "max_abs_beyond_nlo": float(
                np.max(
                    np.abs(
                        group[
                            "beyond_nlo_ratio_difference"
                        ].to_numpy(float)
                    )
                )
            ),
            "n_nonpositive_strict_ratio": int(
                np.sum(group["strict_ratio"] <= 0.0)
            ),
        })

    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--backend-script",
        default="./bt_internal_css_backend_v19_smoothprofile.py",
    )
    parser.add_argument("--data-dir", default="./Data")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["E288_200", "E288_300", "E288_400", "E605"],
    )
    parser.add_argument(
        "--pdf-set",
        default="NNPDF40_nnlo_as_01180",
    )
    parser.add_argument("--pdf-member", type=int, default=0)
    parser.add_argument(
        "--target-mode",
        default="nuclear_isospin",
    )
    parser.add_argument("--resum-order", default="n3llp")
    parser.add_argument(
        "--flavors",
        nargs="+",
        type=int,
        default=[1, 2, 3],
    )
    parser.add_argument("--rows-per-dataset", type=int, default=1)
    parser.add_argument("--qT-max-over-Q", type=float, default=0.5)
    parser.add_argument("--b-min", type=float, default=1.0e-4)
    parser.add_argument("--b-max", type=float, default=8.0)
    parser.add_argument("--n-b", type=int, default=101)
    parser.add_argument("--bstar-bmax", type=float, default=1.5)
    parser.add_argument("--mu-min", type=float, default=1.3)
    parser.add_argument(
        "--mu-floor-smooth-width",
        type=float,
        default=0.12,
    )
    parser.add_argument("--n-sudakov-quad", type=int, default=32)
    parser.add_argument("--epsabs", type=float, default=1.0e-8)
    parser.add_argument("--epsrel", type=float, default=1.0e-7)
    parser.add_argument(
        "--bridge-tolerance",
        type=float,
        default=1.0e-8,
    )
    parser.add_argument(
        "--out",
        default="v22/outputs/full_profile_hard_ope_audit",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    backend = import_module_from_path(Path(args.backend_script))

    cuts = backend.CutConfig(
        mode="matched",
        qT_max_over_Q=float(args.qT_max_over_Q),
        tmd_qT_max_over_Q=0.2,
        apply_upsilon_veto=True,
    )

    data = backend.load_fixed_target_data(
        args.data_dir,
        args.datasets,
        cuts,
    )

    selected = select_representative_rows(
        data,
        rows_per_dataset=int(args.rows_per_dataset),
    )

    cfg = construct_css_config(backend, args)

    pdf = backend.LHAPDFProvider(
        args.pdf_set,
        int(args.pdf_member),
        use_toy_pdf=False,
    )

    point_tables = []
    row_summaries = []

    for _, row in selected.iterrows():
        points, summary = evaluate_row(
            backend=backend,
            row=row,
            pdf=pdf,
            cfg=cfg,
            epsabs=float(args.epsabs),
            epsrel=float(args.epsrel),
        )
        point_tables.append(points)
        row_summaries.append(summary)

    points = pd.concat(point_tables, ignore_index=True)
    rows = pd.DataFrame(row_summaries)
    regions = region_summary(points)

    numeric = points.select_dtypes(include=[np.number]).to_numpy(float)
    all_finite = bool(np.isfinite(numeric).all())

    checks = {
        "rows_evaluated": len(rows) > 0,
        "all_values_finite": all_finite,
        "born_bridge_closes": (
            float(points["born_bridge_relerr"].max())
            < float(args.bridge_tolerance)
        ),
        "strict_ratio_positive_everywhere": bool(
            (points["strict_ratio"] > 0.0).all()
        ),
    }

    implementation_pass = bool(
        checks["rows_evaluated"]
        and checks["all_values_finite"]
        and checks["born_bridge_closes"]
    )

    # Operational diagnostics only; not theory theorems.
    perturbative_control = bool(
        checks["strict_ratio_positive_everywhere"]
        and float(rows["bessel_weighted_abs_total_p90"].max()) < 0.50
        and float(rows["max_abs_beyond_nlo"].max()) < 0.10
    )

    summary = {
        "backend": str(Path(args.backend_script).resolve()),
        "pdf_set": args.pdf_set,
        "pdf_member": int(args.pdf_member),
        "n_selected_rows": int(len(rows)),
        "n_pointwise_values": int(len(points)),
        "max_born_bridge_error": float(
            points["born_bridge_relerr"].max()
        ),
        "global_strict_ratio_min": float(
            points["strict_ratio"].min()
        ),
        "global_strict_ratio_max": float(
            points["strict_ratio"].max()
        ),
        "max_abs_pointwise_total_fraction": float(
            np.max(np.abs(points["strict_total_fraction"]))
        ),
        "max_bessel_weighted_abs_total_p90": float(
            rows["bessel_weighted_abs_total_p90"].max()
        ),
        "max_abs_beyond_nlo": float(
            rows["max_abs_beyond_nlo"].max()
        ),
        "strict_integral_over_old_min": float(
            rows["strict_integral_over_old"].min()
        ),
        "strict_integral_over_old_median": float(
            rows["strict_integral_over_old"].median()
        ),
        "strict_integral_over_old_max": float(
            rows["strict_integral_over_old"].max()
        ),
        "checks": checks,
        "GENERAL_SCALE_IMPLEMENTATION_PASS": implementation_pass,
        "OPERATIONAL_PERTURBATIVE_CONTROL": perturbative_control,
        "interpretation": (
            "The audit uses the bare perturbative W without F_NP and without "
            "the Y term. Large-b support is therefore deliberately conservative. "
            "The operational-control flag is a development diagnostic, not a "
            "physics acceptance criterion."
        ),
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    points.to_csv(out / "full_profile_points.csv", index=False)
    rows.to_csv(out / "full_profile_by_row.csv", index=False)
    regions.to_csv(out / "full_profile_by_region.csv", index=False)
    (out / "full_profile_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print("\n=== Full-profile correction by region ===")
    print(regions.to_string(index=False))

    print("\n=== Bessel-integrated effect by selected row ===")
    print(rows.to_string(index=False))

    print("\n=== Global summary ===")
    for key, value in summary.items():
        if key != "checks":
            print(f"{key}: {value}")

    print("\nChecks:")
    for key, value in checks.items():
        print(f"  {key}: {value}")

    print(
        "\nGENERAL_SCALE_IMPLEMENTATION_PASS:",
        summary["GENERAL_SCALE_IMPLEMENTATION_PASS"],
    )
    print(
        "OPERATIONAL_PERTURBATIVE_CONTROL:",
        summary["OPERATIONAL_PERTURBATIVE_CONTROL"],
    )
    print("wrote:", out)

    if not implementation_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
