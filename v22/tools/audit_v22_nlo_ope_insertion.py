#!/usr/bin/env python3
"""Audit the separate one-loop CSS2 OPE insertions in the v21 W kernel.

This is a pointwise bridge diagnostic.  It evaluates only bT points where
the existing scale profile is genuinely canonical,

    mu_profile = C0 / b_star,

with neither the small-b cap at Q nor the large-b mu_min floor active.

At each accepted point it computes, consistently through O(alpha_s),

    L = L_Born + a_s [Delta_L_qq + Delta_L_qg],

and reports the q<-q and q<-g effects separately.  It also reports the
difference between the strict O(alpha_s) expansion and the naive product
of two NLO-matched legs; that difference is O(alpha_s^2) and must not be
silently included in an NLO calculation.

The DY hard factor is intentionally not included here.
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

from v22.src.css2_ope_nlo import (
    CanonicalOPEComponents,
    canonical_css2_quark_ope_nlo_components,
)


FOURIER_NORM = 1.0 / (2.0 * math.pi)


def import_module_from_path(path: Path):
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    name = "v22_ope_insertion_backend"
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

    for _, group in frame.groupby(
        "dataset",
        observed=False,
        sort=True,
    ):
        sort_columns = [
            column
            for column in ["QM", "qT", "row_id"]
            if column in group.columns
        ]
        group = group.sort_values(sort_columns)

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


def proton_density(pdf, pid: int, mu: float) -> Callable[[float], float]:
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


def split_corrections(
    components: CanonicalOPEComponents,
) -> tuple[float, float]:
    """Return coefficient-level qq and qg corrections, without a_s."""

    delta_qq = (
        components.one_loop_qq_delta
        + components.one_loop_qq_regular
    )
    delta_qg = components.one_loop_qg_regular

    return float(delta_qq), float(delta_qg)


def profile_status(
    *,
    backend,
    bT: float,
    Q: float,
    cfg,
    tolerance: float,
) -> dict[str, float | bool | str]:
    bstar_value = float(
        np.asarray(
            backend.bstar(
                float(bT),
                float(cfg.bstar_bmax),
            )
        )
    )

    mu_unclipped = float(backend.C0) / max(
        bstar_value,
        1.0e-300,
    )

    mu_profile = float(
        backend.mu_b_of_b(
            float(bT),
            float(Q),
            cfg,
        )
    )

    cap_active = (
        bool(getattr(cfg, "cap_mub_at_Q", True))
        and mu_unclipped > max(float(Q), float(cfg.mu_min))
        * (1.0 + tolerance)
    )

    floor_active = (
        mu_unclipped
        < float(cfg.mu_min) * (1.0 - tolerance)
    )

    canonical = relative_error(
        mu_profile,
        mu_unclipped,
    ) < tolerance

    if canonical:
        reason = "canonical"
    elif cap_active:
        reason = "Q_cap_active"
    elif floor_active:
        reason = "mu_floor_active"
    else:
        reason = "profile_modified_other"

    return {
        "b_star": bstar_value,
        "mu_unclipped_C0_over_bstar": mu_unclipped,
        "mu_profile": mu_profile,
        "cap_active": cap_active,
        "floor_active": floor_active,
        "canonical": canonical,
        "profile_reason": reason,
    }


def leg_components(
    *,
    x: float,
    mu: float,
    alpha_s: float,
    quark_pdf: Callable[[float], float],
    gluon_pdf: Callable[[float], float],
    epsabs: float,
    epsrel: float,
) -> dict[str, float]:
    components = canonical_css2_quark_ope_nlo_components(
        x=float(x),
        alpha_s=float(alpha_s),
        quark_pdf=quark_pdf,
        gluon_pdf=gluon_pdf,
        epsabs=float(epsabs),
        epsrel=float(epsrel),
    )

    delta_qq, delta_qg = split_corrections(components)

    return {
        "born": float(components.born_quark),
        "delta_qq": delta_qq,
        "delta_qg": delta_qg,
        "a_s": float(components.a_s),
        "matched_qq": float(
            components.born_quark
            + components.a_s * delta_qq
        ),
        "matched_full": float(
            components.born_quark
            + components.a_s * (delta_qq + delta_qg)
        ),
    }


def evaluate_flavor(
    *,
    backend,
    row: pd.Series,
    pid: int,
    mu: float,
    alpha_s: float,
    pdf,
    cfg,
    epsabs: float,
    epsrel: float,
) -> dict[str, float | str | int]:
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

    q_a = leg_components(
        x=x1,
        mu=mu,
        alpha_s=alpha_s,
        quark_pdf=proton_density(pdf, pid, mu),
        gluon_pdf=g_a,
        epsabs=epsabs,
        epsrel=epsrel,
    )
    qb_a = leg_components(
        x=x1,
        mu=mu,
        alpha_s=alpha_s,
        quark_pdf=proton_density(pdf, -pid, mu),
        gluon_pdf=g_a,
        epsabs=epsabs,
        epsrel=epsrel,
    )
    q_b = leg_components(
        x=x2,
        mu=mu,
        alpha_s=alpha_s,
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
    qb_b = leg_components(
        x=x2,
        mu=mu,
        alpha_s=alpha_s,
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

    a_s_values = {
        q_a["a_s"],
        qb_a["a_s"],
        q_b["a_s"],
        qb_b["a_s"],
    }
    if len(a_s_values) != 1:
        raise RuntimeError("inconsistent a_s values across the four legs")

    a_s = q_a["a_s"]

    born_channel = (
        q_a["born"] * qb_b["born"]
        + qb_a["born"] * q_b["born"]
    )

    delta_qq_channel = (
        q_a["delta_qq"] * qb_b["born"]
        + q_a["born"] * qb_b["delta_qq"]
        + qb_a["delta_qq"] * q_b["born"]
        + qb_a["born"] * q_b["delta_qq"]
    )

    delta_qg_channel = (
        q_a["delta_qg"] * qb_b["born"]
        + q_a["born"] * qb_b["delta_qg"]
        + qb_a["delta_qg"] * q_b["born"]
        + qb_a["born"] * q_b["delta_qg"]
    )

    linear_qq_channel = (
        born_channel
        + a_s * delta_qq_channel
    )

    linear_full_channel = (
        born_channel
        + a_s * (
            delta_qq_channel
            + delta_qg_channel
        )
    )

    product_qq_channel = (
        q_a["matched_qq"] * qb_b["matched_qq"]
        + qb_a["matched_qq"] * q_b["matched_qq"]
    )

    product_full_channel = (
        q_a["matched_full"] * qb_b["matched_full"]
        + qb_a["matched_full"] * q_b["matched_full"]
    )

    charge2 = float(
        backend.CHARGE2.get(
            abs(int(pid)),
            0.0,
        )
    )

    return {
        "pid": int(pid),
        "charge2": charge2,
        "born_channel": float(born_channel),
        "delta_qq_channel": float(delta_qq_channel),
        "delta_qg_channel": float(delta_qg_channel),
        "linear_qq_channel": float(linear_qq_channel),
        "linear_full_channel": float(linear_full_channel),
        "product_qq_channel": float(product_qq_channel),
        "product_full_channel": float(product_full_channel),
        "a_s": float(a_s),
        "weighted_born": charge2 * born_channel,
        "weighted_delta_qq": charge2 * delta_qq_channel,
        "weighted_delta_qg": charge2 * delta_qg_channel,
        "weighted_linear_qq": charge2 * linear_qq_channel,
        "weighted_linear_full": charge2 * linear_full_channel,
        "weighted_product_qq": charge2 * product_qq_channel,
        "weighted_product_full": charge2 * product_full_channel,
    }


def evaluate_point(
    *,
    backend,
    row: pd.Series,
    bT: float,
    pdf,
    cfg,
    epsabs: float,
    epsrel: float,
    profile_tolerance: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    Q = float(row["QM"])
    x1 = float(row["x1"])
    x2 = float(row["x2"])

    profile = profile_status(
        backend=backend,
        bT=float(bT),
        Q=Q,
        cfg=cfg,
        tolerance=float(profile_tolerance),
    )

    base = {
        "row_id": str(row["row_id"]),
        "dataset": str(row["dataset"]),
        "Q": Q,
        "qT": float(row["qT"]),
        "x1": x1,
        "x2": x2,
        "bT": float(bT),
        **profile,
    }

    if not bool(profile["canonical"]):
        return {
            **base,
            "evaluated": False,
        }, []

    mu = float(profile["mu_profile"])
    alpha_s = float(pdf.alphas(mu))
    a_s_expected = alpha_s / (4.0 * math.pi)

    flavor_rows: list[dict[str, Any]] = []

    for flavor in cfg.flavors:
        pid = abs(int(flavor))

        result = evaluate_flavor(
            backend=backend,
            row=row,
            pid=pid,
            mu=mu,
            alpha_s=alpha_s,
            pdf=pdf,
            cfg=cfg,
            epsabs=epsabs,
            epsrel=epsrel,
        )

        flavor_rows.append({
            **base,
            "evaluated": True,
            "mu": mu,
            "alpha_s": alpha_s,
            "a_s_expected": a_s_expected,
            **result,
        })

    born_luminosity = sum(
        float(item["weighted_born"])
        for item in flavor_rows
    )
    delta_qq_luminosity = sum(
        float(item["weighted_delta_qq"])
        for item in flavor_rows
    )
    delta_qg_luminosity = sum(
        float(item["weighted_delta_qg"])
        for item in flavor_rows
    )
    linear_qq_luminosity = sum(
        float(item["weighted_linear_qq"])
        for item in flavor_rows
    )
    linear_full_luminosity = sum(
        float(item["weighted_linear_full"])
        for item in flavor_rows
    )
    product_qq_luminosity = sum(
        float(item["weighted_product_qq"])
        for item in flavor_rows
    )
    product_full_luminosity = sum(
        float(item["weighted_product_full"])
        for item in flavor_rows
    )

    backend_luminosity_xf = float(
        backend.charge_weighted_lumi(
            row,
            mu,
            pdf,
            cfg,
        )
    )

    born_luminosity_xf = (
        x1 * x2 * born_luminosity
    )

    prefactor = float(
        backend.fixed_target_prefactor_cs(
            row,
            cfg,
        )
    )
    sudakov = float(
        backend.sudakov_s(
            float(bT),
            Q,
            pdf,
            cfg,
        )
    )

    common = (
        prefactor
        * FOURIER_NORM
        * x1
        * x2
        * math.exp(-sudakov)
    )

    old_w = float(
        backend.wpert_cs_for_row(
            row,
            np.asarray([bT], dtype=float),
            pdf,
            cfg,
        )[0]
    )

    w_born_reconstructed = (
        common * born_luminosity
    )
    w_qq_linear = (
        common * linear_qq_luminosity
    )
    w_full_linear = (
        common * linear_full_luminosity
    )
    w_qq_product = (
        common * product_qq_luminosity
    )
    w_full_product = (
        common * product_full_luminosity
    )

    born_scale = max(
        abs(born_luminosity),
        1.0e-300,
    )

    summary = {
        **base,
        "evaluated": True,
        "mu": mu,
        "alpha_s": alpha_s,
        "a_s": a_s_expected,
        "backend_luminosity_xf": backend_luminosity_xf,
        "born_luminosity_f": born_luminosity,
        "born_luminosity_xf": born_luminosity_xf,
        "delta_qq_coefficient_luminosity": delta_qq_luminosity,
        "delta_qg_coefficient_luminosity": delta_qg_luminosity,
        "linear_qq_luminosity": linear_qq_luminosity,
        "linear_full_luminosity": linear_full_luminosity,
        "product_qq_luminosity": product_qq_luminosity,
        "product_full_luminosity": product_full_luminosity,
        "qq_fraction_of_born": (
            a_s_expected
            * delta_qq_luminosity
            / born_scale
        ),
        "qg_fraction_of_born": (
            a_s_expected
            * delta_qg_luminosity
            / born_scale
        ),
        "full_nlo_fraction_of_born": (
            a_s_expected
            * (
                delta_qq_luminosity
                + delta_qg_luminosity
            )
            / born_scale
        ),
        "qq_product_minus_linear_over_born": (
            product_qq_luminosity
            - linear_qq_luminosity
        ) / born_scale,
        "full_product_minus_linear_over_born": (
            product_full_luminosity
            - linear_full_luminosity
        ) / born_scale,
        "old_Wpert_CS": old_w,
        "W_born_reconstructed": w_born_reconstructed,
        "W_qq_linear_no_hard": w_qq_linear,
        "W_full_linear_no_hard": w_full_linear,
        "W_qq_naive_product": w_qq_product,
        "W_full_naive_product": w_full_product,
        "W_qq_linear_over_old": (
            w_qq_linear / old_w
            if abs(old_w) > 1.0e-300
            else math.nan
        ),
        "W_full_linear_over_old": (
            w_full_linear / old_w
            if abs(old_w) > 1.0e-300
            else math.nan
        ),
        "relerr_backend_xf_vs_born_xf": relative_error(
            backend_luminosity_xf,
            born_luminosity_xf,
        ),
        "relerr_old_W_vs_reconstructed_born": relative_error(
            old_w,
            w_born_reconstructed,
        ),
        "relerr_a_s_expected_vs_legs": max(
            relative_error(
                a_s_expected,
                float(item["a_s"]),
            )
            for item in flavor_rows
        ),
    }

    return summary, flavor_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit separate q<-q and q<-g one-loop CSS2 OPE "
            "insertions in the canonical profile window."
        )
    )

    parser.add_argument(
        "--backend-script",
        default="./bt_internal_css_backend_v19_smoothprofile.py",
    )
    parser.add_argument(
        "--data-dir",
        default="./Data",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "E288_200",
            "E288_300",
            "E288_400",
            "E605",
        ],
    )
    parser.add_argument(
        "--pdf-set",
        default="NNPDF40_nnlo_as_01180",
    )
    parser.add_argument(
        "--pdf-member",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--target-mode",
        default="nuclear_isospin",
    )
    parser.add_argument(
        "--resum-order",
        default="n3llp",
    )
    parser.add_argument(
        "--flavors",
        nargs="+",
        type=int,
        default=[1, 2, 3],
    )
    parser.add_argument(
        "--rows-per-dataset",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--b-points",
        nargs="+",
        type=float,
        default=[
            0.20,
            0.30,
            0.50,
            0.75,
            1.00,
            1.20,
        ],
    )
    parser.add_argument(
        "--b-min",
        type=float,
        default=1.0e-4,
    )
    parser.add_argument(
        "--b-max",
        type=float,
        default=8.0,
    )
    parser.add_argument(
        "--n-b",
        type=int,
        default=160,
    )
    parser.add_argument(
        "--bstar-bmax",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--mu-min",
        type=float,
        default=1.3,
    )
    parser.add_argument(
        "--n-sudakov-quad",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--qT-max-over-Q",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--epsabs",
        type=float,
        default=1.0e-9,
    )
    parser.add_argument(
        "--epsrel",
        type=float,
        default=1.0e-8,
    )
    parser.add_argument(
        "--profile-tolerance",
        type=float,
        default=1.0e-10,
    )
    parser.add_argument(
        "--closure-tolerance",
        type=float,
        default=1.0e-9,
    )
    parser.add_argument(
        "--out",
        default="v22/outputs/nlo_ope_insertion_audit",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    backend = import_module_from_path(
        Path(args.backend_script)
    )

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

    cfg = construct_css_config(
        backend,
        args,
    )

    pdf = backend.LHAPDFProvider(
        args.pdf_set,
        int(args.pdf_member),
        use_toy_pdf=False,
    )

    point_rows: list[dict[str, Any]] = []
    flavor_rows: list[dict[str, Any]] = []

    for _, row in selected.iterrows():
        for bT in args.b_points:
            point, flavors = evaluate_point(
                backend=backend,
                row=row,
                bT=float(bT),
                pdf=pdf,
                cfg=cfg,
                epsabs=float(args.epsabs),
                epsrel=float(args.epsrel),
                profile_tolerance=float(args.profile_tolerance),
            )
            point_rows.append(point)
            flavor_rows.extend(flavors)

    points = pd.DataFrame(point_rows)
    flavors = pd.DataFrame(flavor_rows)

    evaluated = points[
        points["evaluated"].astype(bool)
    ].copy()

    skipped = points[
        ~points["evaluated"].astype(bool)
    ].copy()

    if evaluated.empty:
        raise SystemExit(
            "No canonical-profile points were found. "
            "Adjust --b-points."
        )

    numeric_columns = [
        "mu",
        "alpha_s",
        "a_s",
        "qq_fraction_of_born",
        "qg_fraction_of_born",
        "full_nlo_fraction_of_born",
        "qq_product_minus_linear_over_born",
        "full_product_minus_linear_over_born",
        "W_qq_linear_over_old",
        "W_full_linear_over_old",
        "relerr_backend_xf_vs_born_xf",
        "relerr_old_W_vs_reconstructed_born",
        "relerr_a_s_expected_vs_legs",
    ]

    all_finite = bool(
        np.isfinite(
            evaluated[numeric_columns].to_numpy(float)
        ).all()
    )

    checks = {
        "canonical_points_found": len(evaluated) > 0,
        "all_reported_values_finite": all_finite,
        "born_luminosity_bridge_closes": (
            float(
                evaluated[
                    "relerr_backend_xf_vs_born_xf"
                ].max()
            )
            < float(args.closure_tolerance)
        ),
        "born_W_bridge_closes": (
            float(
                evaluated[
                    "relerr_old_W_vs_reconstructed_born"
                ].max()
            )
            < float(args.closure_tolerance)
        ),
        "alpha_convention_closes": (
            float(
                evaluated[
                    "relerr_a_s_expected_vs_legs"
                ].max()
            )
            < float(args.closure_tolerance)
        ),
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    points.to_csv(
        out / "nlo_ope_insertion_points.csv",
        index=False,
    )
    flavors.to_csv(
        out / "nlo_ope_insertion_by_flavor.csv",
        index=False,
    )
    skipped.to_csv(
        out / "profile_modified_points_skipped.csv",
        index=False,
    )

    grouped = (
        evaluated.groupby(
            ["dataset", "bT"],
            observed=False,
        )
        .agg(
            n=("row_id", "size"),
            mu_min=("mu", "min"),
            mu_max=("mu", "max"),
            alpha_s_median=("alpha_s", "median"),
            qq_fraction_median=(
                "qq_fraction_of_born",
                "median",
            ),
            qg_fraction_median=(
                "qg_fraction_of_born",
                "median",
            ),
            full_nlo_fraction_median=(
                "full_nlo_fraction_of_born",
                "median",
            ),
            full_nlo_fraction_min=(
                "full_nlo_fraction_of_born",
                "min",
            ),
            full_nlo_fraction_max=(
                "full_nlo_fraction_of_born",
                "max",
            ),
            product_minus_linear_abs_max=(
                "full_product_minus_linear_over_born",
                lambda values: float(
                    np.max(np.abs(values))
                ),
            ),
        )
        .reset_index()
    )

    grouped.to_csv(
        out / "nlo_ope_insertion_summary_by_dataset_b.csv",
        index=False,
    )

    summary = {
        "backend": str(
            Path(args.backend_script).resolve()
        ),
        "pdf_set": args.pdf_set,
        "pdf_member": int(args.pdf_member),
        "target_mode": args.target_mode,
        "n_selected_rows": int(len(selected)),
        "n_requested_points": int(len(points)),
        "n_canonical_points_evaluated": int(
            len(evaluated)
        ),
        "n_profile_modified_points_skipped": int(
            len(skipped)
        ),
        "profile_skip_reasons": (
            skipped["profile_reason"]
            .value_counts()
            .to_dict()
            if not skipped.empty
            else {}
        ),
        "qq_fraction_of_born_min": float(
            evaluated["qq_fraction_of_born"].min()
        ),
        "qq_fraction_of_born_median": float(
            evaluated[
                "qq_fraction_of_born"
            ].median()
        ),
        "qq_fraction_of_born_max": float(
            evaluated["qq_fraction_of_born"].max()
        ),
        "qg_fraction_of_born_min": float(
            evaluated["qg_fraction_of_born"].min()
        ),
        "qg_fraction_of_born_median": float(
            evaluated[
                "qg_fraction_of_born"
            ].median()
        ),
        "qg_fraction_of_born_max": float(
            evaluated["qg_fraction_of_born"].max()
        ),
        "full_nlo_fraction_of_born_min": float(
            evaluated[
                "full_nlo_fraction_of_born"
            ].min()
        ),
        "full_nlo_fraction_of_born_median": float(
            evaluated[
                "full_nlo_fraction_of_born"
            ].median()
        ),
        "full_nlo_fraction_of_born_max": float(
            evaluated[
                "full_nlo_fraction_of_born"
            ].max()
        ),
        "max_abs_naive_product_minus_linear_over_born": float(
            np.max(
                np.abs(
                    evaluated[
                        "full_product_minus_linear_over_born"
                    ].to_numpy(float)
                )
            )
        ),
        "max_born_luminosity_bridge_error": float(
            evaluated[
                "relerr_backend_xf_vs_born_xf"
            ].max()
        ),
        "max_born_W_bridge_error": float(
            evaluated[
                "relerr_old_W_vs_reconstructed_born"
            ].max()
        ),
        "checks": checks,
        "NLO_OPE_INSERTION_AUDIT_PASS": bool(
            all(checks.values())
        ),
        "interpretation": (
            "The reported NLO ratios include only the CSS2 OPE "
            "coefficients. The DY hard factor and profile-region "
            "logarithms are not included. Therefore the NLO ratios "
            "are diagnostics, not final physical W predictions."
        ),
    }

    (
        out / "nlo_ope_insertion_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n"
    )

    print("\n=== Canonical-window OPE insertion summary ===")
    print(grouped.to_string(index=False))

    print("\n=== Global ranges ===")
    print(
        "qq correction / Born:",
        (
            summary["qq_fraction_of_born_min"],
            summary["qq_fraction_of_born_median"],
            summary["qq_fraction_of_born_max"],
        ),
    )
    print(
        "qg correction / Born:",
        (
            summary["qg_fraction_of_born_min"],
            summary["qg_fraction_of_born_median"],
            summary["qg_fraction_of_born_max"],
        ),
    )
    print(
        "full OPE correction / Born:",
        (
            summary[
                "full_nlo_fraction_of_born_min"
            ],
            summary[
                "full_nlo_fraction_of_born_median"
            ],
            summary[
                "full_nlo_fraction_of_born_max"
            ],
        ),
    )
    print(
        "max |naive NLO product - strict linear NLO| / Born:",
        summary[
            "max_abs_naive_product_minus_linear_over_born"
        ],
    )

    print("\n=== Bridge checks ===")
    for key, value in checks.items():
        print(f"{key}: {value}")

    print(
        "\nNLO_OPE_INSERTION_AUDIT_PASS:",
        summary["NLO_OPE_INSERTION_AUDIT_PASS"],
    )
    print("wrote:", out)

    if not summary["NLO_OPE_INSERTION_AUDIT_PASS"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
