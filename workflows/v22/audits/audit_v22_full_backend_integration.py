#!/usr/bin/env python3
"""Integration audit for the complete v22 W+Y backend wrapper.

The audit checks three separate interfaces:

1. The intermediate scheme-Y backend still reproduces the old Born W values
   stored by the standalone W audit.
2. The full backend reproduces the standalone multiplicative v22 W values.
3. With Y clipping disabled, changing W does not alter the already validated
   scheme-consistent Y term.

The change in W+Y relative to the old Born-W backend is reported but is not
used as a pass/fail threshold. External observable closure against MCFM and
DYTurbo is the following milestone.
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


def import_from_path(path: Path, name: str):
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


def relative_error(a: float, b: float) -> float:
    return abs(float(a) - float(b)) / max(
        abs(float(a)),
        abs(float(b)),
        1.0e-300,
    )


def construct_cfg(module, args):
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
        "nlo_y_transition": 0.20,
        "nlo_y_transition_width": 0.15,
        "nlo_singular_mode": "asymptotic_damped",
        "nlo_singular_rsub": 0.10,
        "nlo_singular_power": 2.0,
        "nlo_singular_damp_kind": "exp",
        "nlo_real_convention": "base",
        "nlo_singular_convention": "base",
        "nlo_alpha_convention": "alpha_over_pi",
        "nlo_real_tail_repair": "mcfm_logistic",
        "nlo_real_tail_r0": 0.530,
        "nlo_real_tail_width": 0.008,
        "nlo_real_tail_rinf": 0.180,
    }

    config_type = module.CSSConfig

    if is_dataclass(config_type):
        valid = {field.name for field in fields(config_type)}
        overrides = {
            key: value
            for key, value in overrides.items()
            if key in valid
        }

    return config_type(**overrides)


def bessel_integral(
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scheme-y-backend",
        default=(
            "v22/backends/"
            "bt_internal_css_backend_v22_scheme_y.py"
        ),
    )

    parser.add_argument(
        "--full-backend",
        default=(
            "v22/backends/"
            "bt_internal_css_backend_v22_full.py"
        ),
    )

    parser.add_argument(
        "--standalone-points",
        default=(
            "v22/outputs/"
            "standalone_w_kernel_audit/"
            "standalone_w_points.csv"
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
        "--nlo-real-quad",
        type=int,
        default=96,
    )

    parser.add_argument(
        "--point-tolerance",
        type=float,
        default=1.0e-8,
    )

    parser.add_argument(
        "--y-tolerance",
        type=float,
        default=1.0e-10,
    )

    parser.add_argument(
        "--out",
        default=(
            "v22/outputs/"
            "full_backend_integration_audit"
        ),
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    reference_path = Path(args.standalone_points)

    if not reference_path.exists():
        raise SystemExit(
            f"Missing {reference_path}. "
            "Run the standalone W-kernel audit first."
        )

    reference = pd.read_csv(reference_path)

    required_columns = {
        "row_id",
        "dataset",
        "Q",
        "qT",
        "bT",
        "old_W",
        "standalone_multiplicative_W",
    }

    missing = required_columns.difference(reference.columns)

    if missing:
        raise SystemExit(
            f"Standalone reference is missing: {sorted(missing)}"
        )

    scheme = import_from_path(
        Path(args.scheme_y_backend),
        "v22_scheme_y_for_full_audit",
    )

    full = import_from_path(
        Path(args.full_backend),
        "v22_full_for_full_audit",
    )

    cfg_scheme = construct_cfg(
        scheme,
        args,
    )

    cfg_full = construct_cfg(
        full,
        args,
    )

    cuts = scheme.CutConfig(
        mode="matched",
        qT_max_over_Q=0.5,
        tmd_qT_max_over_Q=0.2,
        apply_upsilon_veto=True,
    )

    data = scheme.load_fixed_target_data(
        args.data_dir,
        args.datasets,
        cuts,
    ).copy()

    data["_row_id_string"] = data["row_id"].astype(str)

    pdf_scheme = scheme.LHAPDFProvider(
        args.pdf_set,
        int(args.pdf_member),
        use_toy_pdf=False,
    )

    pdf_full = full.LHAPDFProvider(
        args.pdf_set,
        int(args.pdf_member),
        use_toy_pdf=False,
    )

    point_rows = []
    selected_rows = []

    for row_id, group in reference.groupby(
        "row_id",
        observed=False,
        sort=True,
    ):
        matches = data[
            data["_row_id_string"]
            == str(row_id)
        ]

        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one data row for row_id={row_id}, "
                f"found {len(matches)}"
            )

        row = matches.iloc[0].drop(
            labels=["_row_id_string"]
        )

        group = group.sort_values("bT")
        b = group["bT"].to_numpy(float)

        old_values = scheme.wpert_cs_for_row(
            row,
            b,
            pdf_scheme,
            cfg_scheme,
        )

        full_values = full.wpert_cs_for_row(
            row,
            b,
            pdf_full,
            cfg_full,
        )

        old_reference = group["old_W"].to_numpy(float)
        full_reference = group[
            "standalone_multiplicative_W"
        ].to_numpy(float)

        for index, bT in enumerate(b):
            point_rows.append({
                "row_id": str(row_id),
                "dataset": str(row["dataset"]),
                "Q": float(row["QM"]),
                "qT": float(row["qT"]),
                "bT": float(bT),
                "scheme_y_old_W": float(old_values[index]),
                "standalone_old_W": float(old_reference[index]),
                "full_backend_W": float(full_values[index]),
                "standalone_multiplicative_W": float(
                    full_reference[index]
                ),
                "old_W_relerr": relative_error(
                    old_values[index],
                    old_reference[index],
                ),
                "full_W_relerr": relative_error(
                    full_values[index],
                    full_reference[index],
                ),
            })

        selected_rows.append(row)

    points = pd.DataFrame(point_rows)
    selected = pd.DataFrame(selected_rows)

    b_grid = np.asarray(
        scheme.make_b_grid(cfg_scheme),
        dtype=float,
    )

    old_matrix = np.vstack([
        scheme.wpert_cs_for_row(
            row,
            b_grid,
            pdf_scheme,
            cfg_scheme,
        )
        for _, row in selected.iterrows()
    ])

    full_matrix = np.vstack([
        full.wpert_cs_for_row(
            row,
            b_grid,
            pdf_full,
            cfg_full,
        )
        for _, row in selected.iterrows()
    ])

    qT = selected["qT"].to_numpy(float)

    old_baseline = np.asarray([
        bessel_integral(
            b_grid,
            old_matrix[index],
            qT[index],
        )
        for index in range(len(selected))
    ])

    full_baseline = np.asarray([
        bessel_integral(
            b_grid,
            full_matrix[index],
            qT[index],
        )
        for index in range(len(selected))
    ])

    y_old = scheme.y_nlo_dev_for_rows(
        selected,
        old_baseline,
        pdf_scheme,
        cfg_scheme,
    )

    y_full = full.y_nlo_dev_for_rows(
        selected,
        full_baseline,
        pdf_full,
        cfg_full,
    )

    row_summary = pd.DataFrame({
        "row_id": selected["row_id"].astype(str),
        "dataset": selected["dataset"].astype(str),
        "Q": selected["QM"].to_numpy(float),
        "qT": qT,
        "old_W_integral": old_baseline,
        "full_v22_W_integral": full_baseline,
        "W_ratio_full_over_old": (
            full_baseline
            / np.where(
                np.abs(old_baseline) > 1.0e-300,
                old_baseline,
                np.nan,
            )
        ),
        "scheme_y_Y": y_old,
        "full_backend_Y": y_full,
        "Y_abs_difference": np.abs(y_full - y_old),
        "old_matched_W_plus_Y": old_baseline + y_old,
        "full_matched_W_plus_Y": full_baseline + y_full,
        "matched_ratio_full_over_old": (
            (full_baseline + y_full)
            / np.where(
                np.abs(old_baseline + y_old) > 1.0e-300,
                old_baseline + y_old,
                np.nan,
            )
        ),
    })

    numeric_points = points.select_dtypes(
        include=[np.number]
    ).to_numpy(float)

    numeric_rows = row_summary.select_dtypes(
        include=[np.number]
    ).to_numpy(float)

    all_finite = bool(
        np.isfinite(numeric_points).all()
        and np.isfinite(numeric_rows).all()
    )

    checks = {
        "full_backend_flag_active": bool(
            getattr(
                full,
                "V22_FULL_PERTURBATIVE_BACKEND_ACTIVE",
                False,
            )
        ),
        "scheme_y_flag_active": bool(
            getattr(
                full,
                "V22_SCHEME_CONSISTENT_Y_ACTIVE",
                False,
            )
        ),
        "all_values_finite": all_finite,
        "old_W_reference_closes": (
            float(points["old_W_relerr"].max())
            < float(args.point_tolerance)
        ),
        "full_W_reference_closes": (
            float(points["full_W_relerr"].max())
            < float(args.point_tolerance)
        ),
        "Y_unchanged_when_clipping_disabled": (
            float(row_summary["Y_abs_difference"].max())
            < float(args.y_tolerance)
        ),
    }

    passed = bool(all(checks.values()))

    summary = {
        "n_reference_rows": int(len(row_summary)),
        "n_pointwise_checks": int(len(points)),
        "max_old_W_reference_relerr": float(
            points["old_W_relerr"].max()
        ),
        "max_full_W_reference_relerr": float(
            points["full_W_relerr"].max()
        ),
        "max_Y_abs_difference": float(
            row_summary["Y_abs_difference"].max()
        ),
        "W_ratio_full_over_old_min": float(
            row_summary["W_ratio_full_over_old"].min()
        ),
        "W_ratio_full_over_old_median": float(
            row_summary["W_ratio_full_over_old"].median()
        ),
        "W_ratio_full_over_old_max": float(
            row_summary["W_ratio_full_over_old"].max()
        ),
        "matched_ratio_full_over_old_min": float(
            row_summary[
                "matched_ratio_full_over_old"
            ].min()
        ),
        "matched_ratio_full_over_old_median": float(
            row_summary[
                "matched_ratio_full_over_old"
            ].median()
        ),
        "matched_ratio_full_over_old_max": float(
            row_summary[
                "matched_ratio_full_over_old"
            ].max()
        ),
        "checks": checks,
        "FULL_V22_BACKEND_INTEGRATION_PASS": passed,
        "interpretation": (
            "This is an internal integration closure. "
            "It does not replace the external MCFM/DYTurbo "
            "observable benchmark."
        ),
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    points.to_csv(
        out / "full_backend_pointwise.csv",
        index=False,
    )

    row_summary.to_csv(
        out / "full_backend_by_row.csv",
        index=False,
    )

    (
        out / "full_backend_summary.json"
    ).write_text(
        json.dumps(summary, indent=2)
        + "\n"
    )

    print("\n=== Complete v22 backend by selected row ===")
    print(row_summary.to_string(index=False))

    print("\n=== Summary ===")

    for key, value in summary.items():
        if key != "checks":
            print(f"{key}: {value}")

    print("\nChecks:")

    for key, value in checks.items():
        print(f"  {key}: {value}")

    print(
        "\nFULL_V22_BACKEND_INTEGRATION_PASS:",
        passed,
    )

    print("wrote:", out)

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
