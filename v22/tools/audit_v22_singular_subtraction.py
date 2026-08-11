#!/usr/bin/env python3
"""Audit the NLO singular subtraction after upgrading W to the v22 kernel.

The existing matched backend forms a development Y term as

    Y = S_Y(qT/Q) [FO_real - singular],

where `singular` is selected independently of the resummed W kernel.

The old numerical W-expansion subtraction contains only the one-loop
Sudakov expansion,

    delta W_old = - W_Born * S1.

The v22 strict one-loop W also contains the hard factor and the OPE
matching coefficients.  Its numerical expansion is

    delta W_v22 =
        - W_Born * S1
        + delta_H * W_Born
        + delta W_OPE.

This script compares the existing analytic/damped subtraction, the old
Sudakov-only numerical expansion, and the complete v22 numerical
expansion.  It does not alter the backend or fit.
"""

from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import j0

from v22.src.dy_hard_nlo import dy_hard_nlo_at_Q
from v22.src.small_b_profile import b_ope_profile


FOURIER_NORM = 1.0 / (2.0 * math.pi)


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


def relative_error(a: float, b: float) -> float:
    return abs(float(a) - float(b)) / max(
        abs(float(a)),
        abs(float(b)),
        1.0e-300,
    )


def symmetric_difference(
    a: float,
    b: float,
    reference_scale: float = 0.0,
) -> float:
    return abs(float(a) - float(b)) / max(
        abs(float(a)),
        abs(float(b)),
        abs(float(reference_scale)),
        1.0e-300,
    )


def construct_cfg(backend, args):
    overrides: dict[str, Any] = {
        "b_min": float(args.b_min),
        "b_max": float(args.b_max),
        "n_b": int(args.n_b),
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
        "flavors": tuple(int(pid) for pid in args.flavors),
        "target_mode": str(args.target_mode),
        "y_mode": "zero",
        "nlo_real_quad": int(args.nlo_real_quad),
        "nlo_real_norm": 1.0,
        "nlo_singular_norm": 1.0,
        "nlo_y_component": "raw",
        "nlo_y_clip_multiple": 0.0,
        "nlo_dev_use_switch": True,
        "nlo_dev_min_qt_over_q": 1.0e-4,
        "nlo_y_transition": float(args.nlo_y_transition),
        "nlo_y_transition_width": float(args.nlo_y_transition_width),
        "nlo_singular_mode": "asymptotic_damped",
        "nlo_singular_rsub": float(args.nlo_singular_rsub),
        "nlo_singular_power": float(args.nlo_singular_power),
        "nlo_singular_damp_kind": str(args.nlo_singular_damp_kind),
        "nlo_real_convention": "base",
        "nlo_singular_convention": "base",
        "nlo_alpha_convention": "alpha_over_pi",
        "nlo_real_tail_repair": str(args.nlo_real_tail_repair),
        "nlo_real_tail_r0": float(args.nlo_real_tail_r0),
        "nlo_real_tail_width": float(args.nlo_real_tail_width),
        "nlo_real_tail_rinf": float(args.nlo_real_tail_rinf),
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


def select_rows(
    frame: pd.DataFrame,
    rows_per_dataset: int,
) -> pd.DataFrame:
    selected = []

    work = frame.copy()
    work["qT_over_Q"] = (
        work["qT"].astype(float)
        / work["QM"].astype(float)
    )

    for _, group in work.groupby(
        "dataset",
        observed=False,
        sort=True,
    ):
        group = group.sort_values(
            ["qT_over_Q", "QM", "qT", "row_id"]
        )

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


def v22_expansion_integrand(
    *,
    helper,
    backend,
    row: pd.Series,
    pdf,
    cfg,
    b: np.ndarray,
    C5: float,
    profile_power: float,
    profile_kind: str,
    epsabs: float,
    epsrel: float,
) -> tuple[np.ndarray, dict[str, np.ndarray | float]]:
    """Return the complete strict one-loop v22 delta-W integrand."""

    Q = float(row["QM"])
    x1 = float(row["x1"])
    x2 = float(row["x2"])

    prefactor = float(
        backend.fixed_target_prefactor_cs(
            row,
            cfg,
        )
    )

    hard_factor = float(
        dy_hard_nlo_at_Q(
            Q_GeV=Q,
            alpha_s_at_Q=float(pdf.alphas(Q)),
        )
    )
    hard_fraction = hard_factor - 1.0

    total = np.empty_like(b, dtype=float)
    sudakov_piece = np.empty_like(b, dtype=float)
    hard_piece = np.empty_like(b, dtype=float)
    ope_piece = np.empty_like(b, dtype=float)
    born_piece = np.empty_like(b, dtype=float)
    L_b_values = np.empty_like(b, dtype=float)

    for index, bT in enumerate(b):
        b_star = float(
            np.asarray(
                backend.bstar(
                    float(bT),
                    float(cfg.bstar_bmax),
                )
            )
        )

        b_pert = b_ope_profile(
            b_star_GeV_inv=b_star,
            Q_GeV=Q,
            C5=float(C5),
            power=float(profile_power),
            kind=str(profile_kind),
        )

        mu = float(
            backend.mu_b_of_b(
                float(bT),
                Q,
                cfg,
            )
        )

        zeta = mu * mu
        alpha_s_mu = float(pdf.alphas(mu))

        luminosity = helper.luminosity_corrections(
            backend=backend,
            row=row,
            b_star=b_pert,
            mu=mu,
            zeta=zeta,
            alpha_s=alpha_s_mu,
            pdf=pdf,
            cfg=cfg,
            epsabs=float(epsabs),
            epsrel=float(epsrel),
        )

        born_lum = float(
            luminosity["born_luminosity_f"]
        )

        delta_ope = (
            float(luminosity["a_s"])
            * (
                float(
                    luminosity[
                        "delta_qq_coefficient_luminosity"
                    ]
                )
                + float(
                    luminosity[
                        "delta_qg_coefficient_luminosity"
                    ]
                )
            )
        )

        S1 = float(
            backend.sudakov_s_one_loop(
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
        )

        born_piece[index] = common * born_lum
        sudakov_piece[index] = (
            common * (-born_lum * S1)
        )
        hard_piece[index] = (
            common * (hard_fraction * born_lum)
        )
        ope_piece[index] = common * delta_ope

        total[index] = (
            sudakov_piece[index]
            + hard_piece[index]
            + ope_piece[index]
        )

        L_b_values[index] = float(
            luminosity["L_b"]
        )

    return total, {
        "born": born_piece,
        "sudakov": sudakov_piece,
        "hard": hard_piece,
        "ope": ope_piece,
        "L_b": L_b_values,
        "hard_fraction": hard_fraction,
    }


def bessel_transform(
    b: np.ndarray,
    values: np.ndarray,
    qT: float,
) -> float:
    return float(
        np.trapezoid(
            b
            * j0(float(qT) * b)
            * values,
            x=b,
        )
    )


def backend_convention_multiplier(
    backend,
    row: pd.Series,
    cfg,
) -> float:
    function = getattr(
        backend,
        "_nlo_convention_multiplier",
        None,
    )

    if function is None:
        return 1.0

    return float(
        function(
            getattr(
                cfg,
                "nlo_singular_convention",
                "base",
            ),
            row,
        )
    )


def evaluate_row(
    *,
    helper,
    backend,
    row: pd.Series,
    pdf,
    cfg,
    args,
) -> dict[str, Any]:
    Q = float(row["QM"])
    qT = float(row["qT"])
    r = qT / Q

    b = np.asarray(
        backend.make_b_grid(cfg),
        dtype=float,
    )

    delta_integrand, pieces = v22_expansion_integrand(
        helper=helper,
        backend=backend,
        row=row,
        pdf=pdf,
        cfg=cfg,
        b=b,
        C5=float(args.C5),
        profile_power=float(args.profile_power),
        profile_kind=str(args.profile_kind),
        epsabs=float(args.epsabs),
        epsrel=float(args.epsrel),
    )

    norm = float(
        getattr(
            cfg,
            "nlo_singular_norm",
            1.0,
        )
    )

    convention = backend_convention_multiplier(
        backend,
        row,
        cfg,
    )

    v22_raw = (
        bessel_transform(
            b,
            delta_integrand,
            qT,
        )
        * norm
        * convention
    )

    v22_sudakov = (
        bessel_transform(
            b,
            np.asarray(pieces["sudakov"]),
            qT,
        )
        * norm
        * convention
    )

    v22_hard = (
        bessel_transform(
            b,
            np.asarray(pieces["hard"]),
            qT,
        )
        * norm
        * convention
    )

    v22_ope = (
        bessel_transform(
            b,
            np.asarray(pieces["ope"]),
            qT,
        )
        * norm
        * convention
    )

    old_wexp = float(
        backend.singular_nlo_wexp_numeric_for_row(
            row,
            pdf,
            cfg,
            positive=False,
        )
    )

    analytic_raw = float(
        backend.singular_nlo_analytic_for_row(
            row,
            pdf,
            cfg,
        )
    )

    damping = float(
        backend.nlo_singular_damping_factor(
            row,
            cfg,
        )
    )

    current_selected = float(
        backend.singular_nlo_dev_for_row(
            row,
            pdf,
            cfg,
        )
    )

    v22_damped = v22_raw * damping

    fo_raw = float(
        backend.fo_nlo_real_dev_for_row(
            row,
            pdf,
            cfg,
        )
    )

    repair = float(
        backend.nlo_real_tail_repair_factor(
            r,
            cfg,
        )
    )

    fo_repaired = fo_raw * repair

    switch = (
        float(
            backend.smooth_tail_switch(
                r,
                float(
                    getattr(
                        cfg,
                        "nlo_y_transition",
                        0.2,
                    )
                ),
                float(
                    getattr(
                        cfg,
                        "nlo_y_transition_width",
                        0.15,
                    )
                ),
            )
        )
        if bool(
            getattr(
                cfg,
                "nlo_dev_use_switch",
                True,
            )
        )
        else 1.0
    )

    y_current = switch * (
        fo_repaired - current_selected
    )

    y_v22 = switch * (
        fo_repaired - v22_damped
    )

    matched_exp_current = (
        current_selected + y_current
    )

    matched_exp_v22 = (
        v22_damped + y_v22
    )

    current_target = (
        (1.0 - switch) * current_selected
        + switch * fo_repaired
    )

    v22_target = (
        (1.0 - switch) * v22_damped
        + switch * fo_repaired
    )

    scale = max(
        abs(fo_repaired),
        abs(current_selected),
        abs(v22_damped),
        1.0e-300,
    )

    return {
        "row_id": str(row["row_id"]),
        "dataset": str(row["dataset"]),
        "Q": Q,
        "qT": qT,
        "qT_over_Q": r,
        "x1": float(row["x1"]),
        "x2": float(row["x2"]),
        "tail_switch": switch,
        "singular_damping": damping,
        "FO_real_raw": fo_raw,
        "FO_real_tail_repair_factor": repair,
        "FO_real_repaired": fo_repaired,
        "singular_analytic_raw": analytic_raw,
        "singular_current_selected": current_selected,
        "singular_old_wexp_sudakov_only": old_wexp,
        "singular_v22_raw": v22_raw,
        "singular_v22_damped": v22_damped,
        "v22_sudakov_piece": v22_sudakov,
        "v22_hard_piece": v22_hard,
        "v22_ope_piece": v22_ope,
        "v22_piece_sum_residual": (
            v22_raw
            - (
                v22_sudakov
                + v22_hard
                + v22_ope
            )
        ),
        "Y_current": y_current,
        "Y_v22_consistent": y_v22,
        "matched_expansion_current": matched_exp_current,
        "matched_expansion_v22": matched_exp_v22,
        "current_algebraic_target": current_target,
        "v22_algebraic_target": v22_target,
        "current_closure_abs": abs(
            matched_exp_current
            - current_target
        ),
        "v22_closure_abs": abs(
            matched_exp_v22
            - v22_target
        ),
        "current_vs_v22_symmetric_difference": (
            symmetric_difference(
                current_selected,
                v22_damped,
                reference_scale=0.05 * scale,
            )
        ),
        "old_wexp_vs_v22_raw_symmetric_difference": (
            symmetric_difference(
                old_wexp,
                v22_raw,
                reference_scale=0.05 * scale,
            )
        ),
        "Y_change_over_scale": (
            (y_v22 - y_current) / scale
        ),
        "max_abs_profiled_L_b": float(
            np.max(
                np.abs(
                    np.asarray(
                        pieces["L_b"],
                        dtype=float,
                    )
                )
            )
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--helper-script",
        default=(
            "v22/tools/"
            "audit_v22_full_profile_hard_ope.py"
        ),
    )

    parser.add_argument(
        "--backend-script",
        default=(
            "./bt_internal_css_backend_v19_"
            "smoothprofile.py"
        ),
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
        default=5,
    )

    parser.add_argument(
        "--qT-max-over-Q",
        type=float,
        default=0.5,
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
        default=121,
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
        "--mu-floor-smooth-width",
        type=float,
        default=0.12,
    )

    parser.add_argument(
        "--n-sudakov-quad",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--epsabs",
        type=float,
        default=1.0e-8,
    )

    parser.add_argument(
        "--epsrel",
        type=float,
        default=1.0e-7,
    )

    parser.add_argument(
        "--C5",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--profile-power",
        type=float,
        default=16.0,
    )

    parser.add_argument(
        "--profile-kind",
        choices=["smooth", "hard"],
        default="smooth",
    )

    parser.add_argument(
        "--nlo-real-quad",
        type=int,
        default=96,
    )

    parser.add_argument(
        "--nlo-y-transition",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--nlo-y-transition-width",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--nlo-singular-rsub",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--nlo-singular-power",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--nlo-singular-damp-kind",
        default="exp",
    )

    parser.add_argument(
        "--nlo-real-tail-repair",
        default="mcfm_logistic",
    )

    parser.add_argument(
        "--nlo-real-tail-r0",
        type=float,
        default=0.530,
    )

    parser.add_argument(
        "--nlo-real-tail-width",
        type=float,
        default=0.008,
    )

    parser.add_argument(
        "--nlo-real-tail-rinf",
        type=float,
        default=0.180,
    )

    parser.add_argument(
        "--difference-threshold",
        type=float,
        default=0.05,
        help=(
            "Operational threshold for deciding that "
            "the Y subtraction must be rebuilt."
        ),
    )

    parser.add_argument(
        "--out",
        default=(
            "v22/outputs/"
            "singular_subtraction_audit"
        ),
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    helper = import_from_path(
        Path(args.helper_script),
        "v22_singular_helper",
    )

    backend = helper.import_module_from_path(
        Path(args.backend_script)
    )

    required_backend_functions = [
        "sudakov_s_one_loop",
        "singular_nlo_wexp_numeric_for_row",
        "singular_nlo_analytic_for_row",
        "nlo_singular_damping_factor",
        "singular_nlo_dev_for_row",
        "fo_nlo_real_dev_for_row",
        "nlo_real_tail_repair_factor",
        "smooth_tail_switch",
    ]

    missing_functions = [
        name
        for name in required_backend_functions
        if not hasattr(backend, name)
    ]

    if missing_functions:
        raise SystemExit(
            "Backend is missing required functions: "
            f"{missing_functions}"
        )

    cuts = backend.CutConfig(
        mode="matched",
        qT_max_over_Q=float(
            args.qT_max_over_Q
        ),
        tmd_qT_max_over_Q=0.2,
        apply_upsilon_veto=True,
    )

    data = backend.load_fixed_target_data(
        args.data_dir,
        args.datasets,
        cuts,
    )

    selected = select_rows(
        data,
        rows_per_dataset=int(
            args.rows_per_dataset
        ),
    )

    cfg = construct_cfg(
        backend,
        args,
    )

    pdf = backend.LHAPDFProvider(
        args.pdf_set,
        int(args.pdf_member),
        use_toy_pdf=False,
    )

    rows = []

    for _, row in selected.iterrows():
        rows.append(
            evaluate_row(
                helper=helper,
                backend=backend,
                row=row,
                pdf=pdf,
                cfg=cfg,
                args=args,
            )
        )

    result = pd.DataFrame(rows)

    bins = [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
    ]

    labels = [
        "0-0.1",
        "0.1-0.2",
        "0.2-0.3",
        "0.3-0.4",
        "0.4-0.5",
    ]

    result["qT_over_Q_bin"] = pd.cut(
        result["qT_over_Q"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    grouped = (
        result.groupby(
            ["dataset", "qT_over_Q_bin"],
            observed=False,
        )
        .agg(
            n=("row_id", "size"),
            qT_over_Q_min=(
                "qT_over_Q",
                "min",
            ),
            qT_over_Q_max=(
                "qT_over_Q",
                "max",
            ),
            switch_median=(
                "tail_switch",
                "median",
            ),
            damping_median=(
                "singular_damping",
                "median",
            ),
            current_singular_median=(
                "singular_current_selected",
                "median",
            ),
            v22_singular_median=(
                "singular_v22_damped",
                "median",
            ),
            current_vs_v22_diff_median=(
                "current_vs_v22_symmetric_difference",
                "median",
            ),
            current_vs_v22_diff_max=(
                "current_vs_v22_symmetric_difference",
                "max",
            ),
            old_wexp_vs_v22_diff_median=(
                "old_wexp_vs_v22_raw_symmetric_difference",
                "median",
            ),
            Y_change_over_scale_median=(
                "Y_change_over_scale",
                "median",
            ),
            Y_change_over_scale_max_abs=(
                "Y_change_over_scale",
                lambda values: float(
                    np.max(
                        np.abs(values)
                    )
                ),
            ),
        )
        .reset_index()
    )

    numeric = result.select_dtypes(
        include=[np.number]
    ).to_numpy(float)

    all_finite = bool(
        np.isfinite(numeric).all()
    )

    max_piece_residual = float(
        result[
            "v22_piece_sum_residual"
        ].abs().max()
    )

    max_current_closure = float(
        result[
            "current_closure_abs"
        ].max()
    )

    max_v22_closure = float(
        result[
            "v22_closure_abs"
        ].max()
    )

    overlap = result[
        (result["tail_switch"] > 0.05)
        | (
            result[
                "singular_damping"
            ] > 0.05
        )
    ].copy()

    if overlap.empty:
        overlap = result.copy()

    difference_median = float(
        overlap[
            "current_vs_v22_symmetric_difference"
        ].median()
    )

    difference_q90 = float(
        overlap[
            "current_vs_v22_symmetric_difference"
        ].quantile(0.90)
    )

    Y_rebuild_required = bool(
        difference_median
        > float(args.difference_threshold)
        or difference_q90
        > 2.0
        * float(
            args.difference_threshold
        )
    )

    checks = {
        "rows_evaluated": len(result) > 0,
        "all_values_finite": all_finite,
        "v22_piece_decomposition_closes": (
            max_piece_residual < 1.0e-9
        ),
        "current_Y_algebra_closes": (
            max_current_closure < 1.0e-9
        ),
        "v22_Y_algebra_closes": (
            max_v22_closure < 1.0e-9
        ),
    }

    implementation_pass = bool(
        all(checks.values())
    )

    summary = {
        "n_rows": int(len(result)),
        "difference_threshold": float(
            args.difference_threshold
        ),
        "current_vs_v22_difference_median_overlap": (
            difference_median
        ),
        "current_vs_v22_difference_q90_overlap": (
            difference_q90
        ),
        "max_abs_Y_change_over_scale": float(
            result[
                "Y_change_over_scale"
            ].abs().max()
        ),
        "max_v22_piece_sum_residual": (
            max_piece_residual
        ),
        "max_current_Y_closure_abs": (
            max_current_closure
        ),
        "max_v22_Y_closure_abs": (
            max_v22_closure
        ),
        "checks": checks,
        "SINGULAR_SUBTRACTION_AUDIT_PASS": (
            implementation_pass
        ),
        "Y_REBUILD_REQUIRED": (
            Y_rebuild_required
        ),
        "interpretation": (
            "Y_REBUILD_REQUIRED is a scheme-consistency "
            "decision. It does not assess the absolute "
            "normalization of the development FO real term; "
            "that still requires MCFM/DYTurbo closure."
        ),
    }

    out = Path(args.out)
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        out / "singular_subtraction_by_row.csv",
        index=False,
    )

    grouped.to_csv(
        out / "singular_subtraction_by_dataset_bin.csv",
        index=False,
    )

    (
        out / "singular_subtraction_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n"
    )

    print(
        "\n=== Singular subtraction by dataset and qT/Q bin ==="
    )
    print(grouped.to_string(index=False))

    print("\n=== Largest current-v22 differences ===")
    print(
        result.sort_values(
            "current_vs_v22_symmetric_difference",
            ascending=False,
        )[
            [
                "dataset",
                "row_id",
                "Q",
                "qT",
                "qT_over_Q",
                "tail_switch",
                "singular_damping",
                "FO_real_repaired",
                "singular_current_selected",
                "singular_old_wexp_sudakov_only",
                "singular_v22_damped",
                "v22_sudakov_piece",
                "v22_hard_piece",
                "v22_ope_piece",
                "current_vs_v22_symmetric_difference",
                "Y_change_over_scale",
            ]
        ]
        .head(16)
        .to_string(index=False)
    )

    print("\n=== Summary ===")

    for key, value in summary.items():
        if key != "checks":
            print(f"{key}: {value}")

    print("\nChecks:")

    for key, value in checks.items():
        print(f"  {key}: {value}")

    print(
        "\nSINGULAR_SUBTRACTION_AUDIT_PASS:",
        implementation_pass,
    )

    print(
        "Y_REBUILD_REQUIRED:",
        Y_rebuild_required,
    )

    print("wrote:", out)

    if not implementation_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
