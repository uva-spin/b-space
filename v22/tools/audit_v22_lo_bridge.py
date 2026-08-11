#!/usr/bin/env python3
"""Audit the exact LO bridge from the v21 W kernel to factorized v22 legs.

This test does *not* turn on the new one-loop OPE coefficients.

It checks, using the same backend, PDFs, scale profile, Sudakov factor and
observable prefactor, that

    W_old(b)
      = prefactor/(2*pi)
        * sum_q e_q^2 [
            (x1 f_q/A e^{-S/2})(x2 f_qbar/B e^{-S/2})
          + (x1 f_qbar/A e^{-S/2})(x2 f_q/B e^{-S/2})
        ]

and equivalently

    W_old(b)
      = prefactor/(2*pi)
        * x1*x2
        * sum_q e_q^2 [
            (f_q/A e^{-S/2})(f_qbar/B e^{-S/2})
          + (f_qbar/A e^{-S/2})(f_q/B e^{-S/2})
        ].

The second identity is the bridge to the v22 convention, where individual
TMDPDFs are unweighted f rather than LHAPDF's x*f.
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


FOURIER_NORM = 1.0 / (2.0 * math.pi)


def import_module_from_path(path: Path):
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    module_name = "v22_bridge_backend"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def relative_error(a: float, b: float) -> float:
    return abs(float(a) - float(b)) / max(
        abs(float(a)),
        abs(float(b)),
        1.0e-300,
    )


def construct_css_config(backend, args):
    """Instantiate the backend config while tolerating v18/v19 field changes."""

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

    for _, group in frame.groupby("dataset", observed=False, sort=True):
        group = group.sort_values(
            [
                column
                for column in ["QM", "qT", "row_id"]
                if column in group.columns
            ]
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


def manual_luminosities(
    *,
    backend,
    row: pd.Series,
    mu: float,
    pdf,
    cfg,
) -> dict[str, float]:
    x1 = float(row["x1"])
    x2 = float(row["x2"])
    dataset = str(row["dataset"])

    luminosity_xf = 0.0
    luminosity_f = 0.0

    for flavor in cfg.flavors:
        pid = abs(int(flavor))
        charge2 = float(backend.CHARGE2.get(pid, 0.0))

        if charge2 == 0.0:
            continue

        q1_xf = float(pdf.xf_proton(pid, x1, mu))
        qb1_xf = float(pdf.xf_proton(-pid, x1, mu))

        q2_xf = float(
            pdf.xf_target(
                pid,
                x2,
                mu,
                dataset=dataset,
                target_mode=cfg.target_mode,
            )
        )
        qb2_xf = float(
            pdf.xf_target(
                -pid,
                x2,
                mu,
                dataset=dataset,
                target_mode=cfg.target_mode,
            )
        )

        luminosity_xf += charge2 * (
            q1_xf * qb2_xf
            + qb1_xf * q2_xf
        )

        q1_f = q1_xf / x1
        qb1_f = qb1_xf / x1
        q2_f = q2_xf / x2
        qb2_f = qb2_xf / x2

        luminosity_f += charge2 * (
            q1_f * qb2_f
            + qb1_f * q2_f
        )

    return {
        "luminosity_xf": float(luminosity_xf),
        "luminosity_f": float(luminosity_f),
    }


def evaluate_point(
    *,
    backend,
    row: pd.Series,
    bT: float,
    pdf,
    cfg,
) -> dict[str, float | str | int]:
    q = float(row["QM"])
    x1 = float(row["x1"])
    x2 = float(row["x2"])

    mu = float(backend.mu_b_of_b(float(bT), q, cfg))
    sudakov = float(backend.sudakov_s(float(bT), q, pdf, cfg))
    half_evolution = math.exp(-0.5 * sudakov)
    pair_evolution = half_evolution * half_evolution

    prefactor = float(backend.fixed_target_prefactor_cs(row, cfg))

    backend_luminosity_xf = float(
        backend.charge_weighted_lumi(
            row,
            mu,
            pdf,
            cfg,
        )
    )

    luminosities = manual_luminosities(
        backend=backend,
        row=row,
        mu=mu,
        pdf=pdf,
        cfg=cfg,
    )

    manual_luminosity_xf = luminosities["luminosity_xf"]
    manual_luminosity_f = luminosities["luminosity_f"]

    old_w = float(
        backend.wpert_cs_for_row(
            row,
            np.asarray([bT], dtype=float),
            pdf,
            cfg,
        )[0]
    )

    # Explicit product of two old x*f single-hadron legs.
    reconstructed_from_xf_legs = (
        prefactor
        * FOURIER_NORM
        * manual_luminosity_xf
        * pair_evolution
    )

    # Explicit product of two v22 unweighted-f legs. The x1*x2 factor
    # belongs in the observable bridge because the old kernel used x*f.
    reconstructed_from_f_legs = (
        prefactor
        * FOURIER_NORM
        * x1
        * x2
        * manual_luminosity_f
        * pair_evolution
    )

    # Deliberately omit x1*x2 to expose the convention mismatch.
    raw_unweighted_f_product = (
        prefactor
        * FOURIER_NORM
        * manual_luminosity_f
        * pair_evolution
    )

    raw_ratio = (
        raw_unweighted_f_product / old_w
        if abs(old_w) > 1.0e-300
        else math.nan
    )
    expected_raw_ratio = 1.0 / (x1 * x2)

    return {
        "row_id": str(row["row_id"]),
        "dataset": str(row["dataset"]),
        "Q": q,
        "qT": float(row["qT"]),
        "x1": x1,
        "x2": x2,
        "bT": float(bT),
        "mu_b": mu,
        "sudakov_S": sudakov,
        "half_evolution": half_evolution,
        "pair_evolution": pair_evolution,
        "backend_luminosity_xf": backend_luminosity_xf,
        "manual_luminosity_xf": manual_luminosity_xf,
        "manual_luminosity_f": manual_luminosity_f,
        "old_Wpert_CS": old_w,
        "reconstructed_from_xf_legs": reconstructed_from_xf_legs,
        "reconstructed_from_f_legs_times_x1x2": reconstructed_from_f_legs,
        "raw_unweighted_f_product": raw_unweighted_f_product,
        "raw_unweighted_f_over_old": raw_ratio,
        "expected_raw_ratio_1_over_x1x2": expected_raw_ratio,
        "relerr_backend_lumi_vs_manual_xf": relative_error(
            backend_luminosity_xf,
            manual_luminosity_xf,
        ),
        "relerr_old_vs_xf_legs": relative_error(
            old_w,
            reconstructed_from_xf_legs,
        ),
        "relerr_old_vs_f_legs_times_x1x2": relative_error(
            old_w,
            reconstructed_from_f_legs,
        ),
        "relerr_raw_ratio_vs_1_over_x1x2": relative_error(
            raw_ratio,
            expected_raw_ratio,
        ),
    }


def bessel_integral(
    b_grid: np.ndarray,
    integrand: np.ndarray,
    qT: float,
) -> float:
    return float(
        np.trapezoid(
            b_grid
            * j0(float(qT) * b_grid)
            * integrand,
            x=b_grid,
        )
    )


def evaluate_integrated_row(
    *,
    backend,
    row: pd.Series,
    pdf,
    cfg,
) -> dict[str, float | str | int]:
    b_grid = np.asarray(
        backend.make_b_grid(cfg),
        dtype=float,
    )

    old_grid = np.asarray(
        backend.wpert_cs_for_row(
            row,
            b_grid,
            pdf,
            cfg,
        ),
        dtype=float,
    )

    reconstructed_grid = np.empty_like(old_grid)

    for index, bT in enumerate(b_grid):
        point = evaluate_point(
            backend=backend,
            row=row,
            bT=float(bT),
            pdf=pdf,
            cfg=cfg,
        )
        reconstructed_grid[index] = float(
            point["reconstructed_from_f_legs_times_x1x2"]
        )

    old_integral = bessel_integral(
        b_grid,
        old_grid,
        float(row["qT"]),
    )
    reconstructed_integral = bessel_integral(
        b_grid,
        reconstructed_grid,
        float(row["qT"]),
    )

    return {
        "row_id": str(row["row_id"]),
        "dataset": str(row["dataset"]),
        "Q": float(row["QM"]),
        "qT": float(row["qT"]),
        "x1": float(row["x1"]),
        "x2": float(row["x2"]),
        "old_W_CS_bessel_integral": old_integral,
        "reconstructed_W_CS_bessel_integral": reconstructed_integral,
        "relerr_integrated": relative_error(
            old_integral,
            reconstructed_integral,
        ),
        "max_pointwise_relerr_on_backend_grid": float(
            np.max(
                np.abs(old_grid - reconstructed_grid)
                / np.maximum(
                    np.maximum(
                        np.abs(old_grid),
                        np.abs(reconstructed_grid),
                    ),
                    1.0e-300,
                )
            )
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit exact LO factorization of the v21 W kernel into "
            "two v22 unweighted-f single-hadron legs."
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
        default=3,
    )
    parser.add_argument(
        "--b-points",
        nargs="+",
        type=float,
        default=[
            0.05,
            0.10,
            0.25,
            0.50,
            1.00,
            2.00,
            4.00,
            6.00,
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
        "--out",
        default="v22/outputs/lo_bridge_audit",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1.0e-10,
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

    point_rows = []

    for _, row in selected.iterrows():
        for bT in args.b_points:
            point_rows.append(
                evaluate_point(
                    backend=backend,
                    row=row,
                    bT=float(bT),
                    pdf=pdf,
                    cfg=cfg,
                )
            )

    pointwise = pd.DataFrame(point_rows)

    # One integrated closure row per dataset, chosen near the middle
    # of each selected set.
    integrated_rows = []

    for _, group in selected.groupby(
        "dataset",
        observed=False,
        sort=True,
    ):
        row = group.iloc[len(group) // 2]
        integrated_rows.append(
            evaluate_integrated_row(
                backend=backend,
                row=row,
                pdf=pdf,
                cfg=cfg,
            )
        )

    integrated = pd.DataFrame(integrated_rows)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pointwise.to_csv(
        out / "lo_bridge_pointwise.csv",
        index=False,
    )
    integrated.to_csv(
        out / "lo_bridge_integrated.csv",
        index=False,
    )

    summary = {
        "backend": str(Path(args.backend_script).resolve()),
        "pdf_set": args.pdf_set,
        "pdf_member": int(args.pdf_member),
        "target_mode": args.target_mode,
        "resum_order": args.resum_order,
        "n_selected_rows": int(len(selected)),
        "n_pointwise_checks": int(len(pointwise)),
        "n_integrated_checks": int(len(integrated)),
        "max_relerr_backend_lumi_vs_manual_xf": float(
            pointwise[
                "relerr_backend_lumi_vs_manual_xf"
            ].max()
        ),
        "max_relerr_old_vs_xf_legs": float(
            pointwise[
                "relerr_old_vs_xf_legs"
            ].max()
        ),
        "max_relerr_old_vs_f_legs_times_x1x2": float(
            pointwise[
                "relerr_old_vs_f_legs_times_x1x2"
            ].max()
        ),
        "max_relerr_raw_ratio_vs_1_over_x1x2": float(
            pointwise[
                "relerr_raw_ratio_vs_1_over_x1x2"
            ].max()
        ),
        "max_relerr_integrated": float(
            integrated["relerr_integrated"].max()
        ),
        "max_pointwise_relerr_on_backend_grid": float(
            integrated[
                "max_pointwise_relerr_on_backend_grid"
            ].max()
        ),
        "tolerance": float(args.tolerance),
    }

    checks = {
        "luminosity_identity": (
            summary[
                "max_relerr_backend_lumi_vs_manual_xf"
            ]
            < args.tolerance
        ),
        "old_xf_leg_factorization": (
            summary[
                "max_relerr_old_vs_xf_legs"
            ]
            < args.tolerance
        ),
        "v22_f_leg_factorization_with_x1x2": (
            summary[
                "max_relerr_old_vs_f_legs_times_x1x2"
            ]
            < args.tolerance
        ),
        "missing_x1x2_mismatch_understood": (
            summary[
                "max_relerr_raw_ratio_vs_1_over_x1x2"
            ]
            < args.tolerance
        ),
        "bessel_integrated_closure": (
            summary["max_relerr_integrated"]
            < args.tolerance
        ),
    }

    summary["checks"] = checks
    summary["LO_BRIDGE_PASS"] = bool(all(checks.values()))

    (out / "lo_bridge_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print("\n=== Pointwise LO bridge ===")
    print(
        pointwise[
            [
                "dataset",
                "row_id",
                "Q",
                "qT",
                "x1",
                "x2",
                "bT",
                "relerr_backend_lumi_vs_manual_xf",
                "relerr_old_vs_xf_legs",
                "relerr_old_vs_f_legs_times_x1x2",
                "raw_unweighted_f_over_old",
                "expected_raw_ratio_1_over_x1x2",
            ]
        ]
        .sort_values(
            "relerr_old_vs_f_legs_times_x1x2",
            ascending=False,
        )
        .head(16)
        .to_string(index=False)
    )

    print("\n=== Integrated LO bridge ===")
    print(integrated.to_string(index=False))

    print("\n=== Summary ===")
    for key, value in summary.items():
        if key != "checks":
            print(f"{key}: {value}")

    print("\nChecks:")
    for key, value in checks.items():
        print(f"  {key}: {value}")

    print("\nLO_BRIDGE_PASS:", summary["LO_BRIDGE_PASS"])
    print("wrote:", out)

    if not summary["LO_BRIDGE_PASS"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
