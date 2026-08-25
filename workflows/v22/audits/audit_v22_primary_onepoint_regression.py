#!/usr/bin/env python3
"""Lock the documented onepoint MCFM bridge target and evaluate v22 predictions.

This script intentionally separates two things:

  1. external target provenance:
       v15_onepoint_mcfm_bridgecut.log -> E288_400:80,
       final MCFM value in fb;

  2. internal backend predictions:
       legacy v19, v22 scheme-consistent-Y, and v22 full W+Y
       evaluated at the same data-row kinematics.

The backend predictions are in the existing bT-TMD observable units, not MCFM
fb.  A direct MCFM pass/fail is therefore not made unless a conversion factor
is supplied explicitly with --model-to-fb.
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
            b * j0(float(qT) * b) * values,
            x=b,
        )
    )


def find_primary_target(reduced_dir: Path) -> dict[str, Any]:
    path = reduced_dir / "mcfm_point_mapping_candidates.csv"

    if not path.exists():
        raise SystemExit(
            f"Missing {path}. Run reduce_v22_external_tail_benchmarks.py first."
        )

    table = pd.read_csv(path)

    mask = (
        table["point_label"].astype(str).eq("onepoint")
        & table["cut_label"].astype(str).eq("bridgecut")
        & table["mapping_status"].astype(str).str.contains(
            "documented",
            case=False,
            na=False,
        )
        & table["row_id"].astype(str).eq("E288_400:80")
    )

    selected = table[mask].copy()

    if len(selected) != 1:
        raise SystemExit(
            "Expected exactly one documented onepoint bridgecut target; "
            f"found {len(selected)}"
        )

    row = selected.iloc[0].to_dict()

    return {
        "point_label": row["point_label"],
        "cut_label": row["cut_label"],
        "mcfm_filename": row["mcfm_filename"],
        "mcfm_final_integral_fb": float(row["mcfm_final_integral"]),
        "mcfm_final_uncertainty_fb": float(row["mcfm_final_uncertainty"]),
        "mcfm_final_rel_uncertainty": float(
            row["mcfm_final_rel_uncertainty"]
        ),
        "row_id": str(row["row_id"]),
        "dataset": str(row["dataset"]),
        "qT": float(row["qT"]),
        "Q": float(row["QM"]),
        "y": float(row["y"]),
        "xF": float(row["xF"]),
        "x1": float(row["x1"]),
        "x2": float(row["x2"]),
        "SqrtS": float(row["SqrtS"]),
        "mapping_status": str(row["mapping_status"]),
    }


def load_target_data_row(module, args, row_id: str) -> pd.Series:
    cuts = module.CutConfig(
        mode="matched",
        qT_max_over_Q=10.0,
        tmd_qT_max_over_Q=10.0,
        apply_upsilon_veto=False,
    )

    dataset = row_id.split(":", 1)[0]

    data = module.load_fixed_target_data(
        args.data_dir,
        [dataset],
        cuts,
    ).copy()

    matches = data[data["row_id"].astype(str).eq(row_id)]

    if len(matches) != 1:
        raise SystemExit(
            f"Expected one row {row_id}; found {len(matches)}"
        )

    return matches.iloc[0]


def evaluate_backend(
    *,
    label: str,
    module,
    args,
    data_row: pd.Series,
) -> dict[str, Any]:
    cfg = construct_cfg(module, args)

    pdf = module.LHAPDFProvider(
        args.pdf_set,
        int(args.pdf_member),
        use_toy_pdf=False,
    )

    b = np.asarray(module.make_b_grid(cfg), dtype=float)
    qT = float(data_row["qT"])
    Q = float(data_row["QM"])
    r = qT / Q

    W_grid = np.asarray(
        module.wpert_cs_for_row(data_row, b, pdf, cfg),
        dtype=float,
    )

    W_integral = bessel_integral(b, W_grid, qT)

    rows = pd.DataFrame([data_row])
    baseline = np.asarray([W_integral], dtype=float)

    Y = float(module.y_nlo_dev_for_rows(rows, baseline, pdf, cfg)[0])

    matched = W_integral + Y

    fo_raw = float(module.fo_nlo_real_dev_for_row(data_row, pdf, cfg))
    repair = float(module.nlo_real_tail_repair_factor(r, cfg))
    fo_repaired = fo_raw * repair

    singular = float(module.singular_nlo_dev_for_row(data_row, pdf, cfg))

    switch = (
        float(
            module.smooth_tail_switch(
                r,
                float(getattr(cfg, "nlo_y_transition", 0.2)),
                float(getattr(cfg, "nlo_y_transition_width", 0.15)),
            )
        )
        if bool(getattr(cfg, "nlo_dev_use_switch", True))
        else 1.0
    )

    Y_formula = switch * (fo_repaired - singular)

    return {
        "backend_label": label,
        "backend_path": str(Path(module.__file__).resolve()) if hasattr(module, "__file__") else "",
        "Q": Q,
        "qT": qT,
        "qT_over_Q": r,
        "W_integral": W_integral,
        "Y": Y,
        "W_plus_Y": matched,
        "FO_real_raw": fo_raw,
        "FO_tail_repair": repair,
        "FO_real_repaired": fo_repaired,
        "singular_subtraction": singular,
        "tail_switch": switch,
        "Y_formula": Y_formula,
        "Y_formula_minus_backend_Y": Y_formula - Y,
        "W_min": float(np.min(W_grid)),
        "W_max": float(np.max(W_grid)),
        "n_b": int(len(b)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--reduced-dir",
        default="v22/outputs/external_tail_benchmark_reduced",
    )
    parser.add_argument(
        "--legacy-backend",
        default="./bt_internal_css_backend_v19_smoothprofile.py",
    )
    parser.add_argument(
        "--scheme-y-backend",
        default="v22/backends/bt_internal_css_backend_v22_scheme_y.py",
    )
    parser.add_argument(
        "--full-backend",
        default="v22/backends/bt_internal_css_backend_v22_full.py",
    )
    parser.add_argument("--data-dir", default="./Data")
    parser.add_argument("--pdf-set", default="NNPDF40_nnlo_as_01180")
    parser.add_argument("--pdf-member", type=int, default=0)
    parser.add_argument("--target-mode", default="nuclear_isospin")
    parser.add_argument("--resum-order", default="n3llp")
    parser.add_argument("--flavors", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--b-min", type=float, default=1.0e-4)
    parser.add_argument("--b-max", type=float, default=8.0)
    parser.add_argument("--n-b", type=int, default=160)
    parser.add_argument("--bstar-bmax", type=float, default=1.5)
    parser.add_argument("--mu-min", type=float, default=1.3)
    parser.add_argument("--mu-floor-smooth-width", type=float, default=0.12)
    parser.add_argument("--n-sudakov-quad", type=int, default=32)
    parser.add_argument("--nlo-real-quad", type=int, default=96)
    parser.add_argument("--nlo-y-transition", type=float, default=0.20)
    parser.add_argument("--nlo-y-transition-width", type=float, default=0.15)
    parser.add_argument("--nlo-singular-rsub", type=float, default=0.10)
    parser.add_argument("--nlo-singular-power", type=float, default=2.0)
    parser.add_argument("--nlo-singular-damp-kind", default="exp")
    parser.add_argument("--nlo-real-tail-repair", default="mcfm_logistic")
    parser.add_argument("--nlo-real-tail-r0", type=float, default=0.530)
    parser.add_argument("--nlo-real-tail-width", type=float, default=0.008)
    parser.add_argument("--nlo-real-tail-rinf", type=float, default=0.180)
    parser.add_argument(
        "--model-to-fb",
        type=float,
        default=float("nan"),
        help=(
            "Optional conversion factor from backend W+Y units to fb. "
            "If omitted, no direct MCFM pass/fail is made."
        ),
    )
    parser.add_argument(
        "--out",
        default="v22/outputs/primary_onepoint_regression",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    target = find_primary_target(Path(args.reduced_dir))

    modules = [
        (
            "legacy_v19",
            import_from_path(Path(args.legacy_backend), "v22_regression_legacy"),
        ),
        (
            "v22_scheme_y_only",
            import_from_path(Path(args.scheme_y_backend), "v22_regression_scheme_y"),
        ),
        (
            "v22_full",
            import_from_path(Path(args.full_backend), "v22_regression_full"),
        ),
    ]

    # Use the first module only to load the row; all wrappers share the loader.
    data_row = load_target_data_row(modules[0][1], args, target["row_id"])

    predictions = pd.DataFrame([
        evaluate_backend(
            label=label,
            module=module,
            args=args,
            data_row=data_row,
        )
        for label, module in modules
    ])

    # Add ratios relative to the legacy backend.
    legacy = predictions[predictions["backend_label"] == "legacy_v19"].iloc[0]
    full = predictions[predictions["backend_label"] == "v22_full"].iloc[0]

    predictions["W_plus_Y_over_legacy"] = (
        predictions["W_plus_Y"]
        / float(legacy["W_plus_Y"])
        if abs(float(legacy["W_plus_Y"])) > 1.0e-300
        else np.nan
    )
    predictions["W_integral_over_legacy"] = (
        predictions["W_integral"]
        / float(legacy["W_integral"])
        if abs(float(legacy["W_integral"])) > 1.0e-300
        else np.nan
    )

    conversion_supplied = bool(np.isfinite(args.model_to_fb))

    if conversion_supplied:
        predictions["W_plus_Y_fb"] = predictions["W_plus_Y"] * float(args.model_to_fb)
        predictions["mcfm_pull_sigma"] = (
            predictions["W_plus_Y_fb"]
            - target["mcfm_final_integral_fb"]
        ) / max(target["mcfm_final_uncertainty_fb"], 1.0e-300)
        predictions["mcfm_ratio"] = (
            predictions["W_plus_Y_fb"]
            / target["mcfm_final_integral_fb"]
        )
    else:
        predictions["W_plus_Y_fb"] = np.nan
        predictions["mcfm_pull_sigma"] = np.nan
        predictions["mcfm_ratio"] = np.nan

    summary = {
        "target": target,
        "conversion_supplied": conversion_supplied,
        "model_to_fb": (
            float(args.model_to_fb)
            if conversion_supplied
            else None
        ),
        "legacy_W_plus_Y": float(legacy["W_plus_Y"]),
        "v22_full_W_plus_Y": float(full["W_plus_Y"]),
        "v22_full_over_legacy_W_plus_Y": (
            float(full["W_plus_Y"]) / float(legacy["W_plus_Y"])
        ),
        "legacy_W_integral": float(legacy["W_integral"]),
        "v22_full_W_integral": float(full["W_integral"]),
        "v22_full_over_legacy_W_integral": (
            float(full["W_integral"]) / float(legacy["W_integral"])
        ),
        "direct_mcfm_comparison_status": (
            "enabled_with_user_supplied_conversion"
            if conversion_supplied
            else "not_enabled_backend_units_not_fb"
        ),
        "notes": [
            "Primary documented target is onepoint bridgecut -> E288_400:80.",
            "Backend W+Y values are in bT-TMD observable units, not fb.",
            "Use this to track old-vs-v22 perturbative changes before locking a unit conversion to MCFM.",
            "Do not tune parameters to this one point."
        ],
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    predictions.to_csv(out / "onepoint_backend_predictions.csv", index=False)
    (out / "primary_target.json").write_text(json.dumps(target, indent=2) + "\n")
    (out / "onepoint_regression_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("\n=== Primary external target ===")
    print(json.dumps(target, indent=2))

    print("\n=== Backend predictions at the mapped row ===")
    display = [
        "backend_label", "W_integral", "Y", "W_plus_Y",
        "W_integral_over_legacy", "W_plus_Y_over_legacy",
        "FO_real_repaired", "singular_subtraction", "tail_switch",
        "Y_formula_minus_backend_Y"
    ]
    print(predictions[display].to_string(index=False))

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))

    print("\nwrote:", out)


if __name__ == "__main__":
    main()
