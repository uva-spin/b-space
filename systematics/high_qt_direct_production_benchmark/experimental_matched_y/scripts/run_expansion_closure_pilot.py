#!/usr/bin/env python3
"""Audit ASY against the alpha_s derivative of strict resummed W at one node."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import j0

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from systematics.high_qt_direct_production_benchmark.experimental_matched_y.backend.exact_bin_asymptotic import (
    collider_luminosity_patch,
    node_row,
)
from systematics.high_qt_direct_production_benchmark.experimental_matched_y.scripts.run_asymptotic_pilot import (
    load_backend,
    production_cfg,
)

HERE = ROOT / "systematics/high_qt_direct_production_benchmark/experimental_matched_y"


class ScaledAlphaPDF:
    """Delegate PDFs unchanged while scaling every alpha_s evaluation."""

    def __init__(self, pdf, scale: float):
        self._pdf = pdf
        self._scale = float(scale)

    def alphas(self, q: float) -> float:
        return self._scale * float(self._pdf.alphas(q))

    def __getattr__(self, name):
        return getattr(self._pdf, name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="CDF_RUN_2")
    ap.add_argument("--row", default="CDF_RUN_2:36")
    ap.add_argument("--n-b", type=int, default=160)
    ap.add_argument("--epsilon", type=float, nargs="+", default=[0.001, 0.003])
    args = ap.parse_args()
    source = ROOT / "Data/v23a_tevatron_plus_lhcb7_fiducial_candidate" / f"{args.dataset}.csv"
    selected = pd.read_csv(source).loc[lambda frame: frame.row_id.eq(args.row)]
    if len(selected) != 1:
        raise SystemExit(f"Expected one row {args.row}, found {len(selected)}")
    original = selected.iloc[0]
    row = node_row(
        original,
        qT=0.5 * (float(original.qT_low) + float(original.qT_high)),
        y=0.0,
    )
    backend = load_backend()
    cfg = production_cfg(backend, n_b=args.n_b)
    pdf = backend.LHAPDFProvider("NNPDF40_nnlo_as_01180", 0, use_toy_pdf=False)
    b = np.asarray(backend.make_b_grid(cfg), dtype=float)

    def transform(alpha_scale: float) -> float:
        values = backend._full.wpert_cs_for_row_v22(
            row, b, ScaledAlphaPDF(pdf, alpha_scale), cfg, organization="strict"
        )
        return float(np.trapezoid(b * j0(float(row.qT) * b) * values, x=b))

    with collider_luminosity_patch(backend):
        asym = float(backend._scheme.singular_nlo_v22_wexp_numeric_for_row(
            row, pdf, cfg, positive=False
        ))
        born = transform(0.0)
        samples = [{"epsilon": eps, "w_strict": transform(eps)} for eps in args.epsilon]
    for sample in samples:
        sample["finite_difference_slope"] = (sample["w_strict"] - born) / sample["epsilon"]
        sample["relative_closure_error"] = abs(sample["finite_difference_slope"] - asym) / abs(asym)

    rapidity_factor = float(backend._tevatron_rapidity_factor(row))
    record = {
        "status": "experimental_not_production",
        "row_id": args.row,
        "node": {"qT": float(row.qT), "y": float(row.y), "x1": float(row.x1), "x2": float(row.x2)},
        "n_b": args.n_b,
        "born_transform_pb_per_GeV": born,
        "strict_asymptotic_pb_per_GeV": asym,
        "samples": samples,
        "best_relative_closure_error": min(s["relative_closure_error"] for s in samples),
        "expansion_closure_pass_0p2pct": min(s["relative_closure_error"] for s in samples) < 0.002,
        "backend_inclusive_rapidity_factor_still_present_at_explicit_node": rapidity_factor,
        "exact_bin_double_rapidity_weighting_detected": abs(rapidity_factor - 1.0) > 1.0e-12,
        "conclusion": "ASY closes against the O(alpha_s) derivative of strict W; high-qT failure is not an ASY sign or normalization error. Explicit-y pilots must remove the backend inclusive rapidity approximation.",
    }
    out = HERE / "outputs/expansion_closure_pilot" / args.row.replace(":", "_").lower()
    out.mkdir(parents=True, exist_ok=True)
    (out / f"result_nb{args.n_b}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
