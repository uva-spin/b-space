#!/usr/bin/env python3
"""Run an isolated exact-bin strict-one-loop asymptotic pilot."""

from __future__ import annotations

import argparse
from dataclasses import fields
import importlib.util
import json
from pathlib import Path
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from systematics.high_qt_direct_production_benchmark.experimental_matched_y.backend.exact_bin_asymptotic import (
    integrate_exact_bin,
    make_v22_point_evaluator,
)


HERE = ROOT / "systematics/high_qt_direct_production_benchmark/experimental_matched_y"
BACKEND_PATH = ROOT / "v23/backends/bt_internal_css_backend_v22_tevatron.py"
METRICS_PATH = ROOT / "systematics/collins_factorization_validity/outputs/rowidfix_stageFT_all_qmax0p20_lam0p50_central_s303/metrics.json"


def load_backend():
    spec = importlib.util.spec_from_file_location("_experimental_readonly_v23_backend", BACKEND_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(BACKEND_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def production_cfg(backend, *, n_b: int | None = None):
    source = json.loads(METRICS_PATH.read_text())["config"]
    mapping = {
        "b_min": source["b_min"], "b_max": source["b_max"],
        "n_b": int(n_b if n_b is not None else source["n_b"]),
        "bstar_bmax": source["b_star_max"], "mu_min": source["mu_min"],
        "cap_mub_at_Q": not source["no_cap_mub_at_Q"], "q0": source["q0"],
        "resum_order": source["resum_order"], "match_order": source["match_order"],
        "nf": source["nf"], "n_sudakov_quad": source["n_sudakov_quad"],
        "alpha_em": source["alpha_em"], "hc_factor": source["hc_factor"],
        "prefactor_scheme": source["prefactor_scheme"], "global_norm": source["backend_global_norm"],
        "flavors": tuple(source["flavors"]), "target_mode": source["target_mode"], "y_mode": "zero",
        "nlo_singular_norm": source["nlo_singular_norm"],
        "nlo_singular_convention": source["nlo_singular_convention"],
        "nlo_alpha_convention": source["nlo_alpha_convention"],
    }
    valid = {field.name for field in fields(backend.CSSConfig)}
    return backend.CSSConfig(**{key: value for key, value in mapping.items() if key in valid})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="CDF_RUN_2")
    ap.add_argument("--row", default="CDF_RUN_2:36")
    ap.add_argument("--n-qt", type=int, default=2)
    ap.add_argument("--n-y", type=int, default=2)
    ap.add_argument("--n-b", type=int, default=160)
    ap.add_argument("--pdf-set", default="NNPDF40_nnlo_as_01180")
    ap.add_argument("--pdf-member", type=int, default=0)
    args = ap.parse_args()
    data_path = ROOT / "Data/v23a_tevatron_plus_lhcb7_fiducial_candidate" / f"{args.dataset}.csv"
    selected = pd.read_csv(data_path).loc[lambda frame: frame.row_id.eq(args.row)]
    if len(selected) != 1:
        raise SystemExit(f"Expected one row {args.row}, found {len(selected)}")
    row = selected.iloc[0]
    if str(row.get("target", "")).lower() == "pp":
        raise SystemExit("LHCb/pp pilot requires an explicit high-qT fiducial acceptance evaluator")
    backend = load_backend()
    cfg = production_cfg(backend, n_b=args.n_b)
    pdf = backend.LHAPDFProvider(args.pdf_set, args.pdf_member, use_toy_pdf=False)
    evaluator = make_v22_point_evaluator(backend=backend, pdf=pdf, cfg=cfg)
    started = time.monotonic()
    result = integrate_exact_bin(row, point_evaluator=evaluator, n_qT=args.n_qt, n_y=args.n_y)
    record = {
        "status": "experimental_not_production", "dataset": args.dataset, "row_id": args.row,
        "source_data": str(data_path.relative_to(ROOT)), "backend_read_only": str(BACKEND_PATH.relative_to(ROOT)),
        "production_metrics_read_only": str(METRICS_PATH.relative_to(ROOT)),
        "pdf_set": args.pdf_set, "pdf_member": args.pdf_member, "n_b": args.n_b,
        "n_qT": args.n_qt, "n_y": args.n_y, "value_pb_per_GeV": result.value_pb_per_GeV,
        "qT_low": result.qT_low, "qT_high": result.qT_high,
        "y_low": result.y_low, "y_high": result.y_high,
        "acceptance_mode": result.acceptance_mode, "elapsed_seconds": time.monotonic() - started,
        "kinematics": "leading-power x1,2=(Q/sqrtS) exp(+/-y); explicit qT-bin average and rapidity integral",
    }
    out = HERE / "outputs/asymptotic_pilot" / args.row.replace(":", "_").lower()
    out.mkdir(parents=True, exist_ok=True)
    result_name = f"result_nb{args.n_b}_nqt{args.n_qt}_ny{args.n_y}.json"
    (out / result_name).write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
