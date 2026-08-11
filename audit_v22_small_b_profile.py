#!/usr/bin/env python3
"""Compare raw and small-b-profiled OPE coordinates over the full b grid.

This script reuses the validated luminosity and hard-factor machinery in
`audit_v22_full_profile_hard_ope.py`.  It changes only the perturbative
coordinate entering the OPE coefficient logarithms:

    raw      : b_pert = b_star
    profiled : b_pert = smooth_max(b_star, b0/(C5 Q))

The backend scale profile, Sudakov exponent, Born luminosity and observable
prefactor remain unchanged.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import j0

from v22.src.small_b_profile import (
    b_min_from_Q,
    b_ope_profile,
)


FOURIER_NORM = 1.0 / (2.0 * math.pi)


def import_from_path(
    path: Path,
    name: str,
):
    path = path.expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(
        name,
        str(path),
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


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

    return float(
        np.interp(
            float(quantile) * cumulative[-1],
            cumulative,
            values,
        )
    )


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


def correction_from_luminosity(
    *,
    luminosity: dict[str, float],
    hard_factor: float,
) -> dict[str, float]:
    born = float(
        luminosity["born_luminosity_f"]
    )
    scale = max(abs(born), 1.0e-300)
    a_s = float(luminosity["a_s"])

    qq = (
        a_s
        * float(
            luminosity[
                "delta_qq_coefficient_luminosity"
            ]
        )
        / scale
    )

    qg = (
        a_s
        * float(
            luminosity[
                "delta_qg_coefficient_luminosity"
            ]
        )
        / scale
    )

    ope = qq + qg
    hard = float(hard_factor) - 1.0
    strict = 1.0 + hard + ope

    naive_leg = (
        float(
            luminosity[
                "naive_product_luminosity"
            ]
        )
        / scale
    )

    naive_full = float(hard_factor) * naive_leg

    return {
        "born": born,
        "a_s": a_s,
        "L_b": float(luminosity["L_b"]),
        "qq_fraction": qq,
        "qg_fraction": qg,
        "ope_fraction": ope,
        "hard_fraction": hard,
        "strict_ratio": strict,
        "naive_full_ratio": naive_full,
        "beyond_nlo": naive_full - strict,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--full-profile-audit-script",
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

    parser.add_argument("--data-dir", default="./Data")

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
        default=1,
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
        default=61,
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
        "--bridge-tolerance",
        type=float,
        default=1.0e-8,
    )

    parser.add_argument(
        "--out",
        default=(
            "v22/outputs/"
            "small_b_profile_audit"
        ),
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    full = import_from_path(
        Path(args.full_profile_audit_script),
        "v22_full_profile_helpers",
    )

    backend = full.import_module_from_path(
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

    selected = full.select_representative_rows(
        data,
        rows_per_dataset=int(
            args.rows_per_dataset
        ),
    )

    cfg = full.construct_css_config(
        backend,
        args,
    )

    pdf = backend.LHAPDFProvider(
        args.pdf_set,
        int(args.pdf_member),
        use_toy_pdf=False,
    )

    point_rows: list[dict[str, Any]] = []
    row_rows: list[dict[str, Any]] = []

    for _, row in selected.iterrows():
        Q = float(row["QM"])
        qT = float(row["qT"])
        x1 = float(row["x1"])
        x2 = float(row["x2"])
        dataset = str(row["dataset"])

        b = np.asarray(
            backend.make_b_grid(cfg),
            dtype=float,
        )

        node_weights = full.trapezoid_node_weights(b)

        old_w = np.asarray(
            backend.wpert_cs_for_row(
                row,
                b,
                pdf,
                cfg,
            ),
            dtype=float,
        )

        hard_factor = float(
            full.dy_hard_nlo_at_Q(
                Q_GeV=Q,
                alpha_s_at_Q=float(pdf.alphas(Q)),
            )
        )

        prefactor = float(
            backend.fixed_target_prefactor_cs(
                row,
                cfg,
            )
        )

        raw_ratios = np.empty_like(b)
        profiled_ratios = np.empty_like(b)
        raw_beyond = np.empty_like(b)
        profiled_beyond = np.empty_like(b)
        born_values = np.empty_like(b)

        for index, bT in enumerate(b):
            b_star = float(
                np.asarray(
                    backend.bstar(
                        float(bT),
                        float(cfg.bstar_bmax),
                    )
                )
            )

            b_profiled = b_ope_profile(
                b_star_GeV_inv=b_star,
                Q_GeV=Q,
                C5=float(args.C5),
                power=float(
                    args.profile_power
                ),
                kind=str(args.profile_kind),
            )

            mu_canonical = (
                float(backend.C0)
                / max(b_star, 1.0e-300)
            )

            mu_profile = float(
                backend.mu_b_of_b(
                    float(bT),
                    Q,
                    cfg,
                )
            )

            zeta = mu_profile * mu_profile
            alpha_s_mu = float(
                pdf.alphas(mu_profile)
            )

            raw_lum = full.luminosity_corrections(
                backend=backend,
                row=row,
                b_star=b_star,
                mu=mu_profile,
                zeta=zeta,
                alpha_s=alpha_s_mu,
                pdf=pdf,
                cfg=cfg,
                epsabs=float(args.epsabs),
                epsrel=float(args.epsrel),
            )

            profiled_lum = (
                full.luminosity_corrections(
                    backend=backend,
                    row=row,
                    b_star=b_profiled,
                    mu=mu_profile,
                    zeta=zeta,
                    alpha_s=alpha_s_mu,
                    pdf=pdf,
                    cfg=cfg,
                    epsabs=float(args.epsabs),
                    epsrel=float(args.epsrel),
                )
            )

            raw = correction_from_luminosity(
                luminosity=raw_lum,
                hard_factor=hard_factor,
            )

            profiled = correction_from_luminosity(
                luminosity=profiled_lum,
                hard_factor=hard_factor,
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

            born_value = (
                common * raw["born"]
            )

            bessel_weight = (
                node_weights[index]
                * abs(
                    bT
                    * j0(qT * bT)
                    * old_w[index]
                )
            )

            raw_ratios[index] = raw["strict_ratio"]
            profiled_ratios[index] = (
                profiled["strict_ratio"]
            )
            raw_beyond[index] = raw["beyond_nlo"]
            profiled_beyond[index] = (
                profiled["beyond_nlo"]
            )
            born_values[index] = born_value

            point_rows.append({
                "row_id": str(row["row_id"]),
                "dataset": dataset,
                "Q": Q,
                "qT": qT,
                "x1": x1,
                "x2": x2,
                "bT": float(bT),
                "b_star": b_star,
                "b_min_Q": b_min_from_Q(
                    Q_GeV=Q,
                    C5=float(args.C5),
                ),
                "b_ope_profiled": b_profiled,
                "mu_canonical": mu_canonical,
                "mu_profile": mu_profile,
                "region": classify_region(
                    mu_canonical=mu_canonical,
                    Q=Q,
                    mu_min=float(cfg.mu_min),
                ),
                "raw_L_b": raw["L_b"],
                "profiled_L_b": profiled["L_b"],
                "raw_strict_ratio": raw["strict_ratio"],
                "profiled_strict_ratio": (
                    profiled["strict_ratio"]
                ),
                "raw_beyond_nlo": raw["beyond_nlo"],
                "profiled_beyond_nlo": (
                    profiled["beyond_nlo"]
                ),
                "old_W": float(old_w[index]),
                "born_W_reconstructed": born_value,
                "born_bridge_relerr": full.relative_error(
                    old_w[index],
                    born_value,
                ),
                "absolute_old_bessel_weight": (
                    bessel_weight
                ),
            })

        raw_integral = float(
            np.trapezoid(
                b * j0(qT * b)
                * old_w * raw_ratios,
                x=b,
            )
        )

        profiled_integral = float(
            np.trapezoid(
                b * j0(qT * b)
                * old_w * profiled_ratios,
                x=b,
            )
        )

        old_integral = float(
            np.trapezoid(
                b * j0(qT * b) * old_w,
                x=b,
            )
        )

        weights = (
            node_weights
            * np.abs(
                b * j0(qT * b) * old_w
            )
        )

        row_rows.append({
            "row_id": str(row["row_id"]),
            "dataset": dataset,
            "Q": Q,
            "qT": qT,
            "old_integral": old_integral,
            "raw_corrected_integral": raw_integral,
            "profiled_corrected_integral": (
                profiled_integral
            ),
            "raw_integral_over_old": (
                raw_integral / old_integral
            ),
            "profiled_integral_over_old": (
                profiled_integral
                / old_integral
            ),
            "profiled_minus_raw_over_old": (
                (profiled_integral - raw_integral)
                / old_integral
            ),
            "raw_ratio_min": float(
                np.min(raw_ratios)
            ),
            "profiled_ratio_min": float(
                np.min(profiled_ratios)
            ),
            "raw_nonpositive_count": int(
                np.sum(raw_ratios <= 0.0)
            ),
            "profiled_nonpositive_count": int(
                np.sum(profiled_ratios <= 0.0)
            ),
            "raw_weighted_abs_correction_p90": (
                weighted_quantile(
                    np.abs(raw_ratios - 1.0),
                    weights,
                    0.90,
                )
            ),
            "profiled_weighted_abs_correction_p90": (
                weighted_quantile(
                    np.abs(
                        profiled_ratios - 1.0
                    ),
                    weights,
                    0.90,
                )
            ),
            "raw_max_abs_beyond_nlo": float(
                np.max(np.abs(raw_beyond))
            ),
            "profiled_max_abs_beyond_nlo": float(
                np.max(
                    np.abs(profiled_beyond)
                )
            ),
        })

    points = pd.DataFrame(point_rows)
    rows = pd.DataFrame(row_rows)

    finite = bool(
        np.isfinite(
            points.select_dtypes(
                include=[np.number]
            ).to_numpy(float)
        ).all()
        and np.isfinite(
            rows.select_dtypes(
                include=[np.number]
            ).to_numpy(float)
        ).all()
    )

    qcap = points[
        points["region"] == "q_cap"
    ]

    checks = {
        "rows_evaluated": len(rows) > 0,
        "all_values_finite": finite,
        "born_bridge_closes": (
            float(
                points[
                    "born_bridge_relerr"
                ].max()
            )
            < float(args.bridge_tolerance)
        ),
        "profiled_ratio_positive_everywhere": bool(
            (
                points[
                    "profiled_strict_ratio"
                ]
                > 0.0
            ).all()
        ),
        "q_cap_profiled_logs_controlled": bool(
            qcap.empty
            or (
                np.abs(
                    qcap[
                        "profiled_L_b"
                    ].to_numpy(float)
                )
                < 0.25
            ).all()
        ),
        "profiled_beyond_nlo_below_10pct": (
            float(
                rows[
                    "profiled_max_abs_beyond_nlo"
                ].max()
            )
            < 0.10
        ),
        "profiled_weighted_p90_below_50pct": (
            float(
                rows[
                    "profiled_weighted_abs_correction_p90"
                ].max()
            )
            < 0.50
        ),
    }

    implementation_pass = bool(
        checks["rows_evaluated"]
        and checks["all_values_finite"]
        and checks["born_bridge_closes"]
    )

    operational_pass = bool(
        all(checks.values())
    )

    region_summary = (
        points.groupby(
            ["dataset", "region"],
            observed=False,
        )
        .agg(
            n=("bT", "size"),
            b_min=("bT", "min"),
            b_max=("bT", "max"),
            raw_L_min=("raw_L_b", "min"),
            raw_L_max=("raw_L_b", "max"),
            profiled_L_min=(
                "profiled_L_b",
                "min",
            ),
            profiled_L_max=(
                "profiled_L_b",
                "max",
            ),
            raw_ratio_min=(
                "raw_strict_ratio",
                "min",
            ),
            profiled_ratio_min=(
                "profiled_strict_ratio",
                "min",
            ),
            raw_beyond_abs_max=(
                "raw_beyond_nlo",
                lambda values: float(
                    np.max(np.abs(values))
                ),
            ),
            profiled_beyond_abs_max=(
                "profiled_beyond_nlo",
                lambda values: float(
                    np.max(np.abs(values))
                ),
            ),
        )
        .reset_index()
    )

    summary = {
        "profile_kind": args.profile_kind,
        "C5": float(args.C5),
        "profile_power": float(
            args.profile_power
        ),
        "n_rows": int(len(rows)),
        "n_points": int(len(points)),
        "raw_global_ratio_min": float(
            points["raw_strict_ratio"].min()
        ),
        "profiled_global_ratio_min": float(
            points[
                "profiled_strict_ratio"
            ].min()
        ),
        "raw_nonpositive_total": int(
            np.sum(
                points[
                    "raw_strict_ratio"
                ] <= 0.0
            )
        ),
        "profiled_nonpositive_total": int(
            np.sum(
                points[
                    "profiled_strict_ratio"
                ] <= 0.0
            )
        ),
        "raw_max_abs_beyond_nlo": float(
            rows[
                "raw_max_abs_beyond_nlo"
            ].max()
        ),
        "profiled_max_abs_beyond_nlo": float(
            rows[
                "profiled_max_abs_beyond_nlo"
            ].max()
        ),
        "max_abs_profiled_minus_raw_integral_over_old": float(
            np.max(
                np.abs(
                    rows[
                        "profiled_minus_raw_over_old"
                    ].to_numpy(float)
                )
            )
        ),
        "checks": checks,
        "SMALL_B_PROFILE_IMPLEMENTATION_PASS": (
            implementation_pass
        ),
        "SMALL_B_PROFILE_OPERATIONAL_PASS": (
            operational_pass
        ),
        "interpretation": (
            "The small-b profile changes only the perturbative coordinate "
            "used in the OPE logarithms.  It is a theory/profile choice "
            "that must later be varied; it is not fitted to data."
        ),
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    points.to_csv(
        out / "small_b_profile_points.csv",
        index=False,
    )

    rows.to_csv(
        out / "small_b_profile_by_row.csv",
        index=False,
    )

    region_summary.to_csv(
        out / "small_b_profile_by_region.csv",
        index=False,
    )

    (
        out / "small_b_profile_summary.json"
    ).write_text(
        json.dumps(summary, indent=2)
        + "\n"
    )

    print(
        "\n=== Raw versus profiled small-b behavior by region ==="
    )
    print(region_summary.to_string(index=False))

    print(
        "\n=== Integrated comparison by selected row ==="
    )
    print(rows.to_string(index=False))

    print("\n=== Summary ===")
    for key, value in summary.items():
        if key != "checks":
            print(f"{key}: {value}")

    print("\nChecks:")
    for key, value in checks.items():
        print(f"  {key}: {value}")

    print(
        "\nSMALL_B_PROFILE_IMPLEMENTATION_PASS:",
        implementation_pass,
    )

    print(
        "SMALL_B_PROFILE_OPERATIONAL_PASS:",
        operational_pass,
    )

    print("wrote:", out)

    if not implementation_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
