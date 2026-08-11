#!/usr/bin/env python3
"""Audit where the existing b* scale profile departs from canonical OPE scales.

This diagnostic does not change the fit and does not insert new coefficients.
It answers a narrow Phase-A question:

    How much of the perturbative W Bessel integral comes from regions where
    mu_profile differs from mu_can = C0 / b_star?

The three geometric regions are

    Q-cap       : mu_can > Q
    canonical   : mu_min <= mu_can <= Q
    mu-floor    : mu_can < mu_min

For a smooth profile, the script also records the actual logarithmic mismatch

    L_profile = ln(mu_profile^2 / mu_can^2).

Two support measures are reported:

  * positive b-space support: b * W(b), independent of the measured qT;
  * absolute Bessel support: |b J0(qT b) W(b)| for each selected data row.

The current audit uses the perturbative W kernel only.  A fitted
nonperturbative factor suppresses large b and will generally reduce the
large-b floor contribution; it does not remove the need to handle the
small-b Q-cap consistently.
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


def import_module_from_path(path: Path):
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    name = "v22_profile_support_backend"
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


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

    return float(
        np.interp(
            target,
            cumulative,
            values,
        )
    )


def support_quantile(
    b_mid: np.ndarray,
    cell_weights: np.ndarray,
    quantile: float,
) -> float:
    order = np.argsort(b_mid)
    b = np.asarray(b_mid, dtype=float)[order]
    weights = np.asarray(cell_weights, dtype=float)[order]

    total = float(np.sum(weights))
    if total <= 0.0:
        return math.nan

    cumulative = np.cumsum(weights)
    return float(
        np.interp(
            float(quantile) * total,
            cumulative,
            b,
        )
    )


def fractions_by_mask(
    weights: np.ndarray,
    masks: dict[str, np.ndarray],
) -> dict[str, float]:
    total = float(np.sum(weights))
    if total <= 0.0:
        return {
            key: math.nan
            for key in masks
        }

    return {
        key: float(np.sum(weights[mask]) / total)
        for key, mask in masks.items()
    }


def evaluate_row(
    *,
    backend,
    row: pd.Series,
    pdf,
    cfg,
    modification_log_tolerance: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    Q = float(row["QM"])
    qT = float(row["qT"])

    b = np.asarray(
        backend.make_b_grid(cfg),
        dtype=float,
    )

    if len(b) < 2 or not np.all(np.diff(b) > 0.0):
        raise ValueError("backend b grid must be strictly increasing")

    W = np.asarray(
        backend.wpert_cs_for_row(
            row,
            b,
            pdf,
            cfg,
        ),
        dtype=float,
    )

    if W.shape != b.shape:
        raise ValueError("W grid shape does not match b grid")

    if not np.all(np.isfinite(W)):
        raise FloatingPointError("nonfinite W values")

    b_mid = 0.5 * (b[:-1] + b[1:])
    db = np.diff(b)

    bstar_mid = np.asarray(
        backend.bstar(
            b_mid,
            float(cfg.bstar_bmax),
        ),
        dtype=float,
    )

    mu_can = float(backend.C0) / np.maximum(
        bstar_mid,
        1.0e-300,
    )

    mu_profile = np.asarray(
        [
            backend.mu_b_of_b(
                float(value),
                Q,
                cfg,
            )
            for value in b_mid
        ],
        dtype=float,
    )

    L_profile = 2.0 * np.log(
        np.maximum(mu_profile, 1.0e-300)
        / np.maximum(mu_can, 1.0e-300)
    )

    masks = {
        "q_cap": mu_can > Q,
        "canonical": (
            (mu_can <= Q)
            & (mu_can >= float(cfg.mu_min))
        ),
        "mu_floor": mu_can < float(cfg.mu_min),
        "profile_modified": (
            np.abs(L_profile)
            > float(modification_log_tolerance)
        ),
    }

    # Cellwise trapezoid weights.
    positive_integrand = b * W
    positive_cell = (
        0.5
        * (
            positive_integrand[:-1]
            + positive_integrand[1:]
        )
        * db
    )

    bessel_integrand = (
        b
        * j0(qT * b)
        * W
    )

    signed_bessel_cell = (
        0.5
        * (
            bessel_integrand[:-1]
            + bessel_integrand[1:]
        )
        * db
    )

    abs_bessel_cell = (
        0.5
        * (
            np.abs(bessel_integrand[:-1])
            + np.abs(bessel_integrand[1:])
        )
        * db
    )

    positive_fractions = fractions_by_mask(
        positive_cell,
        masks,
    )
    bessel_fractions = fractions_by_mask(
        abs_bessel_cell,
        masks,
    )

    positive_total = float(np.sum(positive_cell))
    signed_bessel_total = float(
        np.sum(signed_bessel_cell)
    )
    absolute_bessel_total = float(
        np.sum(abs_bessel_cell)
    )

    cancellation_ratio = (
        abs(signed_bessel_total)
        / absolute_bessel_total
        if absolute_bessel_total > 0.0
        else math.nan
    )

    summary = {
        "row_id": str(row["row_id"]),
        "dataset": str(row["dataset"]),
        "Q": Q,
        "qT": qT,
        "qT_over_Q": qT / Q,
        "x1": float(row["x1"]),
        "x2": float(row["x2"]),
        "positive_bW_integral": positive_total,
        "signed_bessel_integral": signed_bessel_total,
        "absolute_bessel_integral": absolute_bessel_total,
        "bessel_cancellation_ratio": cancellation_ratio,
        "positive_q_cap_fraction": positive_fractions["q_cap"],
        "positive_canonical_fraction": positive_fractions["canonical"],
        "positive_mu_floor_fraction": positive_fractions["mu_floor"],
        "positive_profile_modified_fraction": positive_fractions[
            "profile_modified"
        ],
        "bessel_q_cap_fraction": bessel_fractions["q_cap"],
        "bessel_canonical_fraction": bessel_fractions["canonical"],
        "bessel_mu_floor_fraction": bessel_fractions["mu_floor"],
        "bessel_profile_modified_fraction": bessel_fractions[
            "profile_modified"
        ],
        "positive_abs_L_profile_p50": weighted_quantile(
            np.abs(L_profile),
            positive_cell,
            0.50,
        ),
        "positive_abs_L_profile_p90": weighted_quantile(
            np.abs(L_profile),
            positive_cell,
            0.90,
        ),
        "positive_abs_L_profile_max": float(
            np.max(np.abs(L_profile))
        ),
        "bessel_abs_L_profile_p50": weighted_quantile(
            np.abs(L_profile),
            abs_bessel_cell,
            0.50,
        ),
        "bessel_abs_L_profile_p90": weighted_quantile(
            np.abs(L_profile),
            abs_bessel_cell,
            0.90,
        ),
        "bessel_abs_L_profile_max": float(
            np.max(np.abs(L_profile))
        ),
        "positive_b50": support_quantile(
            b_mid,
            positive_cell,
            0.50,
        ),
        "positive_b90": support_quantile(
            b_mid,
            positive_cell,
            0.90,
        ),
        "positive_b99": support_quantile(
            b_mid,
            positive_cell,
            0.99,
        ),
        "bessel_b50": support_quantile(
            b_mid,
            abs_bessel_cell,
            0.50,
        ),
        "bessel_b90": support_quantile(
            b_mid,
            abs_bessel_cell,
            0.90,
        ),
        "bessel_b99": support_quantile(
            b_mid,
            abs_bessel_cell,
            0.99,
        ),
    }

    cells = pd.DataFrame({
        "row_id": str(row["row_id"]),
        "dataset": str(row["dataset"]),
        "Q": Q,
        "qT": qT,
        "x1": float(row["x1"]),
        "x2": float(row["x2"]),
        "b_mid": b_mid,
        "b_star_mid": bstar_mid,
        "mu_canonical": mu_can,
        "mu_profile": mu_profile,
        "L_profile": L_profile,
        "abs_L_profile": np.abs(L_profile),
        "region": np.select(
            [
                masks["q_cap"],
                masks["mu_floor"],
            ],
            [
                "q_cap",
                "mu_floor",
            ],
            default="canonical",
        ),
        "profile_modified": masks["profile_modified"],
        "positive_bW_cell_weight": positive_cell,
        "signed_bessel_cell_weight": signed_bessel_cell,
        "absolute_bessel_cell_weight": abs_bessel_cell,
    })

    return summary, cells


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Quantify Q-cap, canonical and mu-floor support in "
            "the existing perturbative W kernel."
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
        default=4,
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
        default=801,
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
        "--modification-log-tolerance",
        type=float,
        default=1.0e-3,
        help=(
            "A point is marked profile-modified when "
            "|ln(mu_profile^2/mu_can^2)| exceeds this value."
        ),
    )
    parser.add_argument(
        "--out",
        default="v22/outputs/profile_support_audit",
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

    row_summaries = []
    cell_tables = []

    for _, row in selected.iterrows():
        summary, cells = evaluate_row(
            backend=backend,
            row=row,
            pdf=pdf,
            cfg=cfg,
            modification_log_tolerance=(
                args.modification_log_tolerance
            ),
        )
        row_summaries.append(summary)
        cell_tables.append(cells)

    rows = pd.DataFrame(row_summaries)
    cells = pd.concat(cell_tables, ignore_index=True)

    fraction_columns = [
        "positive_q_cap_fraction",
        "positive_canonical_fraction",
        "positive_mu_floor_fraction",
        "bessel_q_cap_fraction",
        "bessel_canonical_fraction",
        "bessel_mu_floor_fraction",
    ]

    finite = bool(
        np.isfinite(
            rows.select_dtypes(include=[np.number]).to_numpy(float)
        ).all()
    )

    positive_fraction_closure = np.abs(
        rows["positive_q_cap_fraction"]
        + rows["positive_canonical_fraction"]
        + rows["positive_mu_floor_fraction"]
        - 1.0
    )

    bessel_fraction_closure = np.abs(
        rows["bessel_q_cap_fraction"]
        + rows["bessel_canonical_fraction"]
        + rows["bessel_mu_floor_fraction"]
        - 1.0
    )

    checks = {
        "rows_evaluated": len(rows) > 0,
        "all_numeric_outputs_finite": finite,
        "positive_support_fractions_close": bool(
            (positive_fraction_closure < 1.0e-10).all()
        ),
        "bessel_support_fractions_close": bool(
            (bessel_fraction_closure < 1.0e-10).all()
        ),
    }

    grouped = (
        rows.groupby("dataset", observed=False)
        .agg(
            n=("row_id", "size"),
            positive_q_cap_median=(
                "positive_q_cap_fraction",
                "median",
            ),
            positive_q_cap_max=(
                "positive_q_cap_fraction",
                "max",
            ),
            positive_canonical_median=(
                "positive_canonical_fraction",
                "median",
            ),
            positive_mu_floor_median=(
                "positive_mu_floor_fraction",
                "median",
            ),
            positive_mu_floor_max=(
                "positive_mu_floor_fraction",
                "max",
            ),
            bessel_q_cap_median=(
                "bessel_q_cap_fraction",
                "median",
            ),
            bessel_q_cap_max=(
                "bessel_q_cap_fraction",
                "max",
            ),
            bessel_canonical_median=(
                "bessel_canonical_fraction",
                "median",
            ),
            bessel_mu_floor_median=(
                "bessel_mu_floor_fraction",
                "median",
            ),
            bessel_mu_floor_max=(
                "bessel_mu_floor_fraction",
                "max",
            ),
            bessel_modified_median=(
                "bessel_profile_modified_fraction",
                "median",
            ),
            bessel_modified_max=(
                "bessel_profile_modified_fraction",
                "max",
            ),
            bessel_abs_L_p90_median=(
                "bessel_abs_L_profile_p90",
                "median",
            ),
            bessel_abs_L_p90_max=(
                "bessel_abs_L_profile_p90",
                "max",
            ),
            bessel_b90_median=("bessel_b90", "median"),
            bessel_b99_median=("bessel_b99", "median"),
        )
        .reset_index()
    )

    max_modified_fraction = float(
        rows["bessel_profile_modified_fraction"].max()
    )
    max_weighted_log = float(
        rows["bessel_abs_L_profile_p90"].max()
    )

    # Operational decision aid, not a physics theorem.
    general_scale_required = bool(
        max_modified_fraction > 0.01
        or max_weighted_log > 0.05
    )

    summary = {
        "backend": str(Path(args.backend_script).resolve()),
        "pdf_set": args.pdf_set,
        "pdf_member": int(args.pdf_member),
        "n_selected_rows": int(len(rows)),
        "modification_log_tolerance": float(
            args.modification_log_tolerance
        ),
        "global_positive_q_cap_fraction_median": float(
            rows["positive_q_cap_fraction"].median()
        ),
        "global_positive_canonical_fraction_median": float(
            rows["positive_canonical_fraction"].median()
        ),
        "global_positive_mu_floor_fraction_median": float(
            rows["positive_mu_floor_fraction"].median()
        ),
        "global_bessel_q_cap_fraction_median": float(
            rows["bessel_q_cap_fraction"].median()
        ),
        "global_bessel_canonical_fraction_median": float(
            rows["bessel_canonical_fraction"].median()
        ),
        "global_bessel_mu_floor_fraction_median": float(
            rows["bessel_mu_floor_fraction"].median()
        ),
        "max_bessel_profile_modified_fraction": (
            max_modified_fraction
        ),
        "max_bessel_weighted_abs_L_profile_p90": (
            max_weighted_log
        ),
        "checks": checks,
        "PROFILE_SUPPORT_AUDIT_PASS": bool(
            all(checks.values())
        ),
        "GENERAL_SCALE_OPE_REQUIRED": general_scale_required,
        "interpretation": (
            "GENERAL_SCALE_OPE_REQUIRED is an operational code-development "
            "decision. The audit uses bare perturbative W support; fitted "
            "nonperturbative damping generally reduces large-b floor support."
        ),
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows.to_csv(
        out / "profile_support_by_row.csv",
        index=False,
    )
    grouped.to_csv(
        out / "profile_support_by_dataset.csv",
        index=False,
    )
    cells.to_csv(
        out / "profile_support_cells.csv",
        index=False,
    )
    (
        out / "profile_support_summary.json"
    ).write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print("\n=== Profile support by dataset ===")
    print(grouped.to_string(index=False))

    print("\n=== Global summary ===")
    for key, value in summary.items():
        if key != "checks":
            print(f"{key}: {value}")

    print("\nChecks:")
    for key, value in checks.items():
        print(f"  {key}: {value}")

    print(
        "\nPROFILE_SUPPORT_AUDIT_PASS:",
        summary["PROFILE_SUPPORT_AUDIT_PASS"],
    )
    print(
        "GENERAL_SCALE_OPE_REQUIRED:",
        summary["GENERAL_SCALE_OPE_REQUIRED"],
    )
    print("wrote:", out)

    if not summary["PROFILE_SUPPORT_AUDIT_PASS"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
