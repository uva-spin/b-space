#!/usr/bin/env python3
"""Combine the one-loop DY hard factor with the audited OPE insertion.

The input is the canonical-window CSV written by
audit_v22_nlo_ope_insertion.py.

The script reports both strict NLO bookkeeping and common multiplicative
forms so that formally higher-order contamination is explicit.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from v22.src.dy_hard_nlo import (
    dy_hard_coefficient_1,
    dy_hard_nlo_at_Q,
)


def import_module_from_path(path: Path):
    path = path.expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(path)

    name = "v22_hard_ope_backend"
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ope-points",
        default=(
            "v22/outputs/nlo_ope_insertion_audit/"
            "nlo_ope_insertion_points.csv"
        ),
    )

    parser.add_argument(
        "--backend-script",
        default=(
            "./bt_internal_css_backend_v19_smoothprofile.py"
        ),
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
        "--bridge-tolerance",
        type=float,
        default=1.0e-9,
    )

    parser.add_argument(
        "--out",
        default=(
            "v22/outputs/"
            "hard_plus_ope_audit"
        ),
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    source = Path(args.ope_points)

    if not source.exists():
        raise SystemExit(
            f"Missing {source}. Run the NLO OPE insertion audit first."
        )

    points = pd.read_csv(source)

    required = {
        "evaluated",
        "dataset",
        "row_id",
        "Q",
        "bT",
        "full_nlo_fraction_of_born",
        "full_product_minus_linear_over_born",
        "relerr_backend_xf_vs_born_xf",
        "relerr_old_W_vs_reconstructed_born",
    }

    missing = required.difference(points.columns)

    if missing:
        raise SystemExit(
            f"Missing columns: {sorted(missing)}"
        )

    points = points[
        points["evaluated"].astype(bool)
    ].copy()

    if points.empty:
        raise SystemExit(
            "No evaluated canonical-window points found."
        )

    backend = import_module_from_path(
        Path(args.backend_script)
    )

    pdf = backend.LHAPDFProvider(
        args.pdf_set,
        int(args.pdf_member),
        use_toy_pdf=False,
    )

    alpha_s_Q = []
    hard_factor = []
    hard_fraction = []
    hard_coefficient = []

    for Q in points["Q"].to_numpy(float):
        alpha = float(pdf.alphas(float(Q)))
        H = float(
            dy_hard_nlo_at_Q(
                Q_GeV=float(Q),
                alpha_s_at_Q=alpha,
            )
        )

        alpha_s_Q.append(alpha)
        hard_factor.append(H)
        hard_fraction.append(H - 1.0)
        hard_coefficient.append(
            dy_hard_coefficient_1(
                Q_GeV=float(Q),
                mu_GeV=float(Q),
            )
        )

    points["alpha_s_Q"] = alpha_s_Q
    points["hard_coefficient_H1_mu_eq_Q"] = hard_coefficient
    points["hard_factor_nlo"] = hard_factor
    points["hard_fraction_of_born"] = hard_fraction

    ope = points[
        "full_nlo_fraction_of_born"
    ].to_numpy(float)

    leg_product_extra = points[
        "full_product_minus_linear_over_born"
    ].to_numpy(float)

    hard = points[
        "hard_fraction_of_born"
    ].to_numpy(float)

    # Strict O(alpha_s): no hard*OPE or leg*leg cross terms.
    points["strict_hard_plus_ope_fraction"] = (
        hard + ope
    )

    points["strict_hard_plus_ope_ratio"] = (
        1.0 + hard + ope
    )

    # Multiplying H_NLO by the strict linear OPE luminosity introduces
    # the hard*OPE cross term.
    points["hard_times_linear_ope_ratio"] = (
        (1.0 + hard)
        * (1.0 + ope)
    )

    # The naive product of two complete NLO legs also contains leg*leg
    # terms. Include both sources of beyond-NLO contamination here.
    points["hard_times_naive_nlo_legs_ratio"] = (
        (1.0 + hard)
        * (
            1.0
            + ope
            + leg_product_extra
        )
    )

    points["hard_ope_cross_term"] = (
        points["hard_times_linear_ope_ratio"]
        - points["strict_hard_plus_ope_ratio"]
    )

    points["all_beyond_nlo_terms"] = (
        points[
            "hard_times_naive_nlo_legs_ratio"
        ]
        - points[
            "strict_hard_plus_ope_ratio"
        ]
    )

    numeric_columns = [
        "alpha_s_Q",
        "hard_factor_nlo",
        "hard_fraction_of_born",
        "strict_hard_plus_ope_fraction",
        "strict_hard_plus_ope_ratio",
        "hard_times_linear_ope_ratio",
        "hard_times_naive_nlo_legs_ratio",
        "hard_ope_cross_term",
        "all_beyond_nlo_terms",
    ]

    all_finite = bool(
        np.isfinite(
            points[numeric_columns].to_numpy(float)
        ).all()
    )

    checks = {
        "canonical_points_found": len(points) > 0,
        "all_values_finite": all_finite,
        "born_luminosity_bridge_still_closes": (
            float(
                points[
                    "relerr_backend_xf_vs_born_xf"
                ].max()
            )
            < float(args.bridge_tolerance)
        ),
        "born_W_bridge_still_closes": (
            float(
                points[
                    "relerr_old_W_vs_reconstructed_born"
                ].max()
            )
            < float(args.bridge_tolerance)
        ),
        "strict_combined_ratio_positive": bool(
            (
                points[
                    "strict_hard_plus_ope_ratio"
                ]
                > 0.0
            ).all()
        ),
    }

    grouped = (
        points.groupby(
            ["dataset", "bT"],
            observed=False,
        )
        .agg(
            n=("row_id", "size"),
            Q_min=("Q", "min"),
            Q_max=("Q", "max"),
            alpha_s_Q_median=("alpha_s_Q", "median"),
            hard_fraction_median=(
                "hard_fraction_of_born",
                "median",
            ),
            ope_fraction_median=(
                "full_nlo_fraction_of_born",
                "median",
            ),
            strict_combined_fraction_median=(
                "strict_hard_plus_ope_fraction",
                "median",
            ),
            strict_combined_fraction_min=(
                "strict_hard_plus_ope_fraction",
                "min",
            ),
            strict_combined_fraction_max=(
                "strict_hard_plus_ope_fraction",
                "max",
            ),
            beyond_nlo_abs_max=(
                "all_beyond_nlo_terms",
                lambda values: float(
                    np.max(np.abs(values))
                ),
            ),
        )
        .reset_index()
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    points.to_csv(
        out / "hard_plus_ope_points.csv",
        index=False,
    )

    grouped.to_csv(
        out / "hard_plus_ope_summary_by_dataset_b.csv",
        index=False,
    )

    summary = {
        "n_points": int(len(points)),
        "hard_fraction_min": float(
            points["hard_fraction_of_born"].min()
        ),
        "hard_fraction_median": float(
            points["hard_fraction_of_born"].median()
        ),
        "hard_fraction_max": float(
            points["hard_fraction_of_born"].max()
        ),
        "ope_fraction_min": float(
            points[
                "full_nlo_fraction_of_born"
            ].min()
        ),
        "ope_fraction_median": float(
            points[
                "full_nlo_fraction_of_born"
            ].median()
        ),
        "ope_fraction_max": float(
            points[
                "full_nlo_fraction_of_born"
            ].max()
        ),
        "strict_combined_fraction_min": float(
            points[
                "strict_hard_plus_ope_fraction"
            ].min()
        ),
        "strict_combined_fraction_median": float(
            points[
                "strict_hard_plus_ope_fraction"
            ].median()
        ),
        "strict_combined_fraction_max": float(
            points[
                "strict_hard_plus_ope_fraction"
            ].max()
        ),
        "max_abs_hard_ope_cross_term": float(
            np.max(
                np.abs(
                    points[
                        "hard_ope_cross_term"
                    ].to_numpy(float)
                )
            )
        ),
        "max_abs_all_beyond_nlo_terms": float(
            np.max(
                np.abs(
                    points[
                        "all_beyond_nlo_terms"
                    ].to_numpy(float)
                )
            )
        ),
        "checks": checks,
        "HARD_PLUS_OPE_AUDIT_PASS": bool(
            all(checks.values())
        ),
        "interpretation": (
            "The strict combined fraction is the one-loop hard "
            "correction plus the one-loop OPE correction. "
            "The multiplicative alternatives contain formally "
            "higher-order terms and are reported, not vetoed."
        ),
    }

    (
        out / "hard_plus_ope_summary.json"
    ).write_text(
        json.dumps(summary, indent=2)
        + "\n"
    )

    print("\n=== Hard + OPE summary by dataset and bT ===")
    print(grouped.to_string(index=False))

    print("\n=== Global ranges ===")
    print(
        "hard correction / Born:",
        (
            summary["hard_fraction_min"],
            summary["hard_fraction_median"],
            summary["hard_fraction_max"],
        ),
    )
    print(
        "OPE correction / Born:",
        (
            summary["ope_fraction_min"],
            summary["ope_fraction_median"],
            summary["ope_fraction_max"],
        ),
    )
    print(
        "strict hard+OPE correction / Born:",
        (
            summary["strict_combined_fraction_min"],
            summary["strict_combined_fraction_median"],
            summary["strict_combined_fraction_max"],
        ),
    )
    print(
        "max |hard*OPE cross term|:",
        summary["max_abs_hard_ope_cross_term"],
    )
    print(
        "max |all beyond-NLO terms|:",
        summary["max_abs_all_beyond_nlo_terms"],
    )

    print("\n=== Checks ===")
    for key, value in checks.items():
        print(f"{key}: {value}")

    print(
        "\nHARD_PLUS_OPE_AUDIT_PASS:",
        summary["HARD_PLUS_OPE_AUDIT_PASS"],
    )
    print("wrote:", out)

    if not summary["HARD_PLUS_OPE_AUDIT_PASS"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
