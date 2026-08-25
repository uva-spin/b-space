#!/usr/bin/env python3
"""Local-cut quadrature for the documented onepoint MCFM bridge target.

This script integrates the internal backend quantities over the same *local*
(Q, qT, y) bridge window used for the documented MCFM onepoint target.

Default target:
    E288_400:80
    Q  in [8.475, 8.525] GeV
    qT in [2.673, 2.727] GeV
    y  in [0.020, 0.040]
    MCFM bridgecut = 0.00145969 +/- 0.0000014298 fb

Important:
    The backend quantities are in the bT-TMD observable units.  The script
    therefore reports the conversion factor implied by the FO-real integral.
    It does not tune anything and it does not alter any backend.
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

    spec = importlib.util.spec_from_file_location(name, str(path))
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
    return float(np.trapezoid(b * j0(float(qT) * b) * values, x=b))


def mutate_row_kinematics(row: pd.Series, *, Q: float, qT: float, y: float) -> pd.Series:
    out = row.copy()
    sqrts = float(out["SqrtS"])
    x1 = Q / sqrts * math.exp(y)
    x2 = Q / sqrts * math.exp(-y)

    out["QM"] = Q
    if "Q" in out.index:
        out["Q"] = Q
    out["qT"] = qT
    out["y"] = y
    out["x1"] = x1
    out["x2"] = x2
    if "xF" in out.index:
        out["xF"] = x1 - x2
    return out


def load_base_row(module, args, row_id: str) -> pd.Series:
    cuts = module.CutConfig(
        mode="matched",
        qT_max_over_Q=10.0,
        tmd_qT_max_over_Q=10.0,
        apply_upsilon_veto=False,
    )
    dataset = row_id.split(":", 1)[0]
    data = module.load_fixed_target_data(args.data_dir, [dataset], cuts)
    matches = data[data["row_id"].astype(str).eq(row_id)]
    if len(matches) != 1:
        raise SystemExit(f"Expected exactly one row {row_id}; found {len(matches)}")
    return matches.iloc[0]


def evaluate_at_point(module, cfg, pdf, row: pd.Series) -> dict[str, float]:
    Q = float(row["QM"])
    qT = float(row["qT"])
    r = qT / Q

    b = np.asarray(module.make_b_grid(cfg), dtype=float)
    W_grid = np.asarray(module.wpert_cs_for_row(row, b, pdf, cfg), dtype=float)
    W = bessel_integral(b, W_grid, qT)

    Y = float(module.y_nlo_dev_for_rows(pd.DataFrame([row]), np.asarray([W]), pdf, cfg)[0])
    FO_raw = float(module.fo_nlo_real_dev_for_row(row, pdf, cfg))
    repair = float(module.nlo_real_tail_repair_factor(r, cfg))
    FO = FO_raw * repair
    singular = float(module.singular_nlo_dev_for_row(row, pdf, cfg))
    switch = (
        float(module.smooth_tail_switch(
            r,
            float(getattr(cfg, "nlo_y_transition", 0.2)),
            float(getattr(cfg, "nlo_y_transition_width", 0.15)),
        ))
        if bool(getattr(cfg, "nlo_dev_use_switch", True))
        else 1.0
    )

    return {
        "W": W,
        "Y": Y,
        "W_plus_Y": W + Y,
        "FO_real_raw": FO_raw,
        "FO_tail_repair": repair,
        "FO_real_repaired": FO,
        "singular": singular,
        "switch": switch,
        "Y_formula": switch * (FO - singular),
    }


def integrate_backend(label: str, module, args, base_row: pd.Series) -> tuple[dict[str, float], pd.DataFrame]:
    cfg = construct_cfg(module, args)
    pdf = module.LHAPDFProvider(args.pdf_set, int(args.pdf_member), use_toy_pdf=False)

    xq, wq = np.polynomial.legendre.leggauss(int(args.n_quad_Q))
    xt, wt = np.polynomial.legendre.leggauss(int(args.n_quad_qT))
    xy, wy = np.polynomial.legendre.leggauss(int(args.n_quad_y))

    Qs = 0.5 * (args.Q_max - args.Q_min) * xq + 0.5 * (args.Q_max + args.Q_min)
    qTs = 0.5 * (args.qT_max - args.qT_min) * xt + 0.5 * (args.qT_max + args.qT_min)
    ys = 0.5 * (args.y_max - args.y_min) * xy + 0.5 * (args.y_max + args.y_min)

    wQ = 0.5 * (args.Q_max - args.Q_min) * wq
    wqT = 0.5 * (args.qT_max - args.qT_min) * wt
    wyv = 0.5 * (args.y_max - args.y_min) * wy

    totals = {
        "W": 0.0,
        "Y": 0.0,
        "W_plus_Y": 0.0,
        "FO_real_raw": 0.0,
        "FO_real_repaired": 0.0,
        "singular": 0.0,
        "Y_formula": 0.0,
    }
    point_rows = []

    for iQ, Q in enumerate(Qs):
        for iq, qT in enumerate(qTs):
            for iy, y in enumerate(ys):
                weight = float(wQ[iQ] * wqT[iq] * wyv[iy])
                row = mutate_row_kinematics(base_row, Q=float(Q), qT=float(qT), y=float(y))
                values = evaluate_at_point(module, cfg, pdf, row)

                for key in totals:
                    totals[key] += weight * float(values[key])

                point_rows.append({
                    "backend_label": label,
                    "Q": float(Q),
                    "qT": float(qT),
                    "y": float(y),
                    "weight": weight,
                    **values,
                })

    totals["backend_label"] = label
    totals["volume"] = (
        (args.Q_max - args.Q_min)
        * (args.qT_max - args.qT_min)
        * (args.y_max - args.y_min)
    )
    totals["center_Q"] = 0.5 * (args.Q_min + args.Q_max)
    totals["center_qT"] = 0.5 * (args.qT_min + args.qT_max)
    totals["center_y"] = 0.5 * (args.y_min + args.y_max)
    return totals, pd.DataFrame(point_rows)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    p.add_argument("--legacy-backend", default="./bt_internal_css_backend_v19_smoothprofile.py")
    p.add_argument("--scheme-y-backend", default="v22/backends/bt_internal_css_backend_v22_scheme_y.py")
    p.add_argument("--full-backend", default="v22/backends/bt_internal_css_backend_v22_full.py")
    p.add_argument("--data-dir", default="./Data")
    p.add_argument("--row-id", default="E288_400:80")

    p.add_argument("--Q-min", type=float, default=8.475)
    p.add_argument("--Q-max", type=float, default=8.525)
    p.add_argument("--qT-min", type=float, default=2.673)
    p.add_argument("--qT-max", type=float, default=2.727)
    p.add_argument("--y-min", type=float, default=0.020)
    p.add_argument("--y-max", type=float, default=0.040)

    p.add_argument("--mcfm-fb", type=float, default=0.00145969)
    p.add_argument("--mcfm-err-fb", type=float, default=1.4298e-06)

    p.add_argument("--pdf-set", default="NNPDF40_nnlo_as_01180")
    p.add_argument("--pdf-member", type=int, default=0)
    p.add_argument("--target-mode", default="nuclear_isospin")
    p.add_argument("--resum-order", default="n3llp")
    p.add_argument("--flavors", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--b-min", type=float, default=1.0e-4)
    p.add_argument("--b-max", type=float, default=8.0)
    p.add_argument("--n-b", type=int, default=160)
    p.add_argument("--bstar-bmax", type=float, default=1.5)
    p.add_argument("--mu-min", type=float, default=1.3)
    p.add_argument("--mu-floor-smooth-width", type=float, default=0.12)
    p.add_argument("--n-sudakov-quad", type=int, default=32)
    p.add_argument("--nlo-real-quad", type=int, default=96)
    p.add_argument("--nlo-y-transition", type=float, default=0.20)
    p.add_argument("--nlo-y-transition-width", type=float, default=0.15)
    p.add_argument("--nlo-singular-rsub", type=float, default=0.10)
    p.add_argument("--nlo-singular-power", type=float, default=2.0)
    p.add_argument("--nlo-singular-damp-kind", default="exp")
    p.add_argument("--nlo-real-tail-repair", default="mcfm_logistic")
    p.add_argument("--nlo-real-tail-r0", type=float, default=0.530)
    p.add_argument("--nlo-real-tail-width", type=float, default=0.008)
    p.add_argument("--nlo-real-tail-rinf", type=float, default=0.180)

    p.add_argument("--n-quad-Q", type=int, default=3)
    p.add_argument("--n-quad-qT", type=int, default=3)
    p.add_argument("--n-quad-y", type=int, default=3)

    p.add_argument("--out", default="v22/outputs/onepoint_local_cut_quadrature")
    return p


def main() -> None:
    args = build_parser().parse_args()

    backend_specs = [
        ("legacy_v19", Path(args.legacy_backend)),
        ("v22_scheme_y_only", Path(args.scheme_y_backend)),
        ("v22_full", Path(args.full_backend)),
    ]
    modules = [(label, import_from_path(path, f"local_cut_{label}")) for label, path in backend_specs]
    base_row = load_base_row(modules[0][1], args, args.row_id)

    totals = []
    point_tables = []
    for label, module in modules:
        total, points = integrate_backend(label, module, args, base_row)
        totals.append(total)
        point_tables.append(points)

    summary = pd.DataFrame(totals)
    points = pd.concat(point_tables, ignore_index=True)

    legacy = summary[summary["backend_label"] == "legacy_v19"].iloc[0]
    v22 = summary[summary["backend_label"] == "v22_full"].iloc[0]

    # Use the FO real integral as the external-unit anchor, because the MCFM
    # onepoint bridge target is a finite-order one-jet local-cut integral.
    model_to_fb = float(args.mcfm_fb) / float(legacy["FO_real_repaired"])

    for column in ["W", "Y", "W_plus_Y", "FO_real_raw", "FO_real_repaired", "singular", "Y_formula"]:
        summary[column + "_fb_via_legacy_FO"] = summary[column] * model_to_fb

    summary["FO_ratio_to_MCFM"] = summary["FO_real_repaired_fb_via_legacy_FO"] / float(args.mcfm_fb)
    summary["FO_pull_sigma"] = (
        summary["FO_real_repaired_fb_via_legacy_FO"] - float(args.mcfm_fb)
    ) / float(args.mcfm_err_fb)
    summary["WplusY_ratio_to_MCFM"] = summary["W_plus_Y_fb_via_legacy_FO"] / float(args.mcfm_fb)

    decision = {
        "target": {
            "row_id": args.row_id,
            "Q_range": [args.Q_min, args.Q_max],
            "qT_range": [args.qT_min, args.qT_max],
            "y_range": [args.y_min, args.y_max],
            "mcfm_fb": args.mcfm_fb,
            "mcfm_err_fb": args.mcfm_err_fb,
        },
        "model_to_fb_from_legacy_FO": model_to_fb,
        "legacy_FO_integral_backend_units": float(legacy["FO_real_repaired"]),
        "v22_FO_integral_backend_units": float(v22["FO_real_repaired"]),
        "FO_external_closure_by_construction": True,
        "v22_full_WplusY_over_legacy_WplusY": (
            float(v22["W_plus_Y"]) / float(legacy["W_plus_Y"])
        ),
        "interpretation": (
            "The MCFM target is used to lock a local-cut finite-order unit "
            "conversion through FO_real_repaired. W+Y converted with that "
            "same factor is diagnostic only; it is not an MCFM pass/fail."
        ),
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "onepoint_local_cut_summary.csv", index=False)
    points.to_csv(out / "onepoint_local_cut_quadrature_points.csv", index=False)
    (out / "onepoint_local_cut_decision.json").write_text(json.dumps(decision, indent=2) + "\n")

    print("\n=== Local-cut backend integrals ===")
    show = [
        "backend_label",
        "FO_real_repaired",
        "W",
        "Y",
        "W_plus_Y",
        "FO_real_repaired_fb_via_legacy_FO",
        "FO_ratio_to_MCFM",
        "FO_pull_sigma",
        "WplusY_ratio_to_MCFM",
    ]
    print(summary[show].to_string(index=False))

    print("\n=== Decision ===")
    print(json.dumps(decision, indent=2))
    print("\nwrote:", out)


if __name__ == "__main__":
    main()
