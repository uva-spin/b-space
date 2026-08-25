#!/usr/bin/env python3
"""Audit the standalone v22 W-kernel module against the existing backend."""

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

from v22.src.dy_hard_nlo import dy_hard_nlo_at_Q
from v22.src.dy_w_nlo_reference import (
    assemble_dy_w_nlo,
    build_dy_luminosity_nlo,
    build_quark_leg_nlo,
)
from v22.src.small_b_profile import b_ope_profile


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--helper-script",
        default="v22/tools/audit_v22_full_profile_hard_ope.py",
    )
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
    parser.add_argument("--n-b", type=int, default=61)
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
    parser.add_argument("--C5", type=float, default=1.0)
    parser.add_argument("--profile-power", type=float, default=16.0)
    parser.add_argument(
        "--profile-kind",
        choices=["smooth", "hard"],
        default="smooth",
    )
    parser.add_argument(
        "--born-tolerance",
        type=float,
        default=1.0e-8,
    )
    parser.add_argument(
        "--higher-order-tolerance",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--out",
        default="v22/outputs/standalone_w_kernel_audit",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    helper = import_from_path(
        Path(args.helper_script),
        "v22_standalone_w_helpers",
    )

    backend = helper.import_module_from_path(
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

    selected = helper.select_representative_rows(
        data,
        rows_per_dataset=int(args.rows_per_dataset),
    )

    cfg = helper.construct_css_config(backend, args)

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

        b = np.asarray(backend.make_b_grid(cfg), dtype=float)
        old_w = np.asarray(
            backend.wpert_cs_for_row(row, b, pdf, cfg),
            dtype=float,
        )

        strict = np.empty_like(b)
        multiplicative = np.empty_like(b)
        born = np.empty_like(b)
        beyond = np.empty_like(b)

        hard = dy_hard_nlo_at_Q(
            Q_GeV=Q,
            alpha_s_at_Q=float(pdf.alphas(Q)),
        )

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
                C5=float(args.C5),
                power=float(args.profile_power),
                kind=str(args.profile_kind),
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

            g_a = helper.proton_density(pdf, 21, mu)
            g_b = helper.target_density(
                pdf,
                21,
                mu,
                dataset=dataset,
                target_mode=cfg.target_mode,
            )

            legs_a = {}
            legs_b = {}

            for flavor in cfg.flavors:
                pid = abs(int(flavor))

                for signed_pid in (pid, -pid):
                    legs_a[signed_pid] = build_quark_leg_nlo(
                        pid=signed_pid,
                        x=x1,
                        alpha_s_mu=alpha_s_mu,
                        b_pert_GeV_inv=b_pert,
                        mu_GeV=mu,
                        zeta_GeV2=zeta,
                        quark_pdf=helper.proton_density(
                            pdf,
                            signed_pid,
                            mu,
                        ),
                        gluon_pdf=g_a,
                        epsabs=float(args.epsabs),
                        epsrel=float(args.epsrel),
                    )

                    legs_b[signed_pid] = build_quark_leg_nlo(
                        pid=signed_pid,
                        x=x2,
                        alpha_s_mu=alpha_s_mu,
                        b_pert_GeV_inv=b_pert,
                        mu_GeV=mu,
                        zeta_GeV2=zeta,
                        quark_pdf=helper.target_density(
                            pdf,
                            signed_pid,
                            mu,
                            dataset=dataset,
                            target_mode=cfg.target_mode,
                        ),
                        gluon_pdf=g_b,
                        epsabs=float(args.epsabs),
                        epsrel=float(args.epsrel),
                    )

            luminosity = build_dy_luminosity_nlo(
                legs_a=legs_a,
                legs_b=legs_b,
                charge_squared=backend.CHARGE2,
                flavors=tuple(
                    abs(int(pid))
                    for pid in cfg.flavors
                ),
            )

            W = assemble_dy_w_nlo(
                luminosity=luminosity,
                hard_factor=hard,
                observable_prefactor=float(
                    backend.fixed_target_prefactor_cs(
                        row,
                        cfg,
                    )
                ),
                x1=x1,
                x2=x2,
                sudakov_pair_exponent=float(
                    backend.sudakov_s(
                        float(bT),
                        Q,
                        pdf,
                        cfg,
                    )
                ),
            )

            born[index] = W.born
            strict[index] = W.strict_nlo
            multiplicative[index] = W.multiplicative_nlo
            beyond[index] = W.beyond_nlo_fraction_of_born

            point_rows.append({
                "row_id": str(row["row_id"]),
                "dataset": dataset,
                "Q": Q,
                "qT": qT,
                "x1": x1,
                "x2": x2,
                "bT": float(bT),
                "b_star": b_star,
                "b_pert": b_pert,
                "mu": mu,
                "hard_factor": hard,
                "old_W": float(old_w[index]),
                "standalone_born_W": W.born,
                "standalone_strict_W": W.strict_nlo,
                "standalone_multiplicative_W": W.multiplicative_nlo,
                "strict_ratio_to_born": W.strict_ratio_to_born,
                "multiplicative_ratio_to_born": (
                    W.multiplicative_ratio_to_born
                ),
                "beyond_nlo_fraction_of_born": (
                    W.beyond_nlo_fraction_of_born
                ),
                "born_bridge_relerr": helper.relative_error(
                    old_w[index],
                    W.born,
                ),
            })

        old_integral = float(
            np.trapezoid(
                b * j0(qT * b) * old_w,
                x=b,
            )
        )
        strict_integral = float(
            np.trapezoid(
                b * j0(qT * b) * strict,
                x=b,
            )
        )
        multiplicative_integral = float(
            np.trapezoid(
                b * j0(qT * b) * multiplicative,
                x=b,
            )
        )

        row_rows.append({
            "row_id": str(row["row_id"]),
            "dataset": dataset,
            "Q": Q,
            "qT": qT,
            "old_integral": old_integral,
            "strict_integral": strict_integral,
            "multiplicative_integral": multiplicative_integral,
            "strict_over_old": strict_integral / old_integral,
            "multiplicative_over_old": (
                multiplicative_integral / old_integral
            ),
            "max_born_bridge_relerr": float(
                np.max(
                    np.abs(born - old_w)
                    / np.maximum(
                        np.maximum(np.abs(born), np.abs(old_w)),
                        1.0e-300,
                    )
                )
            ),
            "strict_ratio_min": float(
                np.min(strict / np.maximum(np.abs(born), 1.0e-300))
            ),
            "strict_ratio_max": float(
                np.max(strict / np.maximum(np.abs(born), 1.0e-300))
            ),
            "max_abs_beyond_nlo": float(
                np.max(np.abs(beyond))
            ),
        })

    points = pd.DataFrame(point_rows)
    rows = pd.DataFrame(row_rows)

    all_finite = bool(
        np.isfinite(
            points.select_dtypes(include=[np.number]).to_numpy(float)
        ).all()
        and np.isfinite(
            rows.select_dtypes(include=[np.number]).to_numpy(float)
        ).all()
    )

    checks = {
        "rows_evaluated": len(rows) > 0,
        "all_values_finite": all_finite,
        "born_bridge_closes": (
            float(points["born_bridge_relerr"].max())
            < float(args.born_tolerance)
        ),
        "strict_kernel_positive": bool(
            (points["standalone_strict_W"] > 0.0).all()
        ),
        "higher_order_difference_controlled": (
            float(
                points[
                    "beyond_nlo_fraction_of_born"
                ].abs().max()
            )
            < float(args.higher_order_tolerance)
        ),
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    points.to_csv(out / "standalone_w_points.csv", index=False)
    rows.to_csv(out / "standalone_w_by_row.csv", index=False)

    summary = {
        "n_rows": int(len(rows)),
        "n_points": int(len(points)),
        "max_born_bridge_error": float(
            points["born_bridge_relerr"].max()
        ),
        "strict_ratio_min": float(
            points["strict_ratio_to_born"].min()
        ),
        "strict_ratio_median": float(
            points["strict_ratio_to_born"].median()
        ),
        "strict_ratio_max": float(
            points["strict_ratio_to_born"].max()
        ),
        "max_abs_beyond_nlo": float(
            points[
                "beyond_nlo_fraction_of_born"
            ].abs().max()
        ),
        "strict_integral_over_old_min": float(
            rows["strict_over_old"].min()
        ),
        "strict_integral_over_old_median": float(
            rows["strict_over_old"].median()
        ),
        "strict_integral_over_old_max": float(
            rows["strict_over_old"].max()
        ),
        "checks": checks,
        "STANDALONE_W_KERNEL_PASS": bool(all(checks.values())),
    }

    (out / "standalone_w_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print("\n=== Standalone W-kernel by selected row ===")
    print(rows.to_string(index=False))

    print("\n=== Summary ===")
    for key, value in summary.items():
        if key != "checks":
            print(f"{key}: {value}")

    print("\nChecks:")
    for key, value in checks.items():
        print(f"  {key}: {value}")

    print(
        "\nSTANDALONE_W_KERNEL_PASS:",
        summary["STANDALONE_W_KERNEL_PASS"],
    )
    print("wrote:", out)

    if not summary["STANDALONE_W_KERNEL_PASS"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
