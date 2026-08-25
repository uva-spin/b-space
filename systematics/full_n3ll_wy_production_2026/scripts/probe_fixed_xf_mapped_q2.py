#!/usr/bin/env python3
"""Diagnostic fixed-xF conversion using fixed-y DYTurbo cards.

DYTurbo's native ``dsdxf`` restriction failed an identity test.  This probe
instead partitions each Q bin and evaluates fixed-y rectangles whose edges
are obtained from xF=2*mT*sinh(y)/sqrt(s), thereby approximating the exact
fixed-xF integral while retaining the validated dsdqt2 observable.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
RUNNER = BASE / "scripts/run_fixed_target_n3ll_nnlo_grid.py"
DYTURBO = Path("/home/dustin/src/dyturbo-1.4.2/bin/dyturbo")
DYROOT = DYTURBO.parent.parent
OUT = BASE / "reports/fixed_xf_mapped_q2_probe"


def load_runner():
    if str(RUNNER.parent) not in sys.path:
        sys.path.insert(0, str(RUNNER.parent))
    common = importlib.import_module("run_dyturbo_full_n3ll_nnlo_probe")
    common.load_runner()
    spec = importlib.util.spec_from_file_location("fixed_grid", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def map_y(xf: float, m: float, sroot: float) -> float:
    return math.asinh(xf * sroot / (2.0 * m))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("E605", "E772"), required=True)
    ap.add_argument("--row", default="0")
    ap.add_argument("--calls", type=int, default=100_000)
    ap.add_argument("--q-slices", type=int, default=4)
    ap.add_argument("--q-floor", type=float, default=None)
    ap.add_argument("--expcreg", type=float, default=None,
                    help="isolated DYTurbo coefficient-exponentiation regularization override")
    args = ap.parse_args()
    runner = load_runner()
    row = runner.resolve_rows(args.dataset)
    row = row[row.row_id.astype(str).eq(f"{args.dataset}:{args.row}")]
    if len(row) != 1:
        raise RuntimeError(f"row not found: {args.dataset}:{args.row}")
    row = row.iloc[0].copy()
    reg_tag = "default" if args.expcreg is None else f"expcreg{str(float(args.expcreg)).replace('.', 'p')}"
    out = OUT / f"{args.dataset}_{args.row}_{reg_tag}"
    (out / "cards").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(x for x in ("/home/dustin/miniforge3/envs/pdf-fit/lib", str(DYROOT / "lib"), env.get("LD_LIBRARY_PATH", "")) if x)
    qhalf = runner.qT_half_width(args.dataset)
    qfloor = runner.QT_DIAGNOSTIC_FLOOR_GEV if args.q_floor is None else float(args.q_floor)
    qlow = max(qfloor, float(row.qT) - qhalf)
    qhigh = float(row.qT) + qhalf
    qmid = 0.5 * (qlow + qhigh)
    sroot = float(row.SqrtS)
    raw_xf_bounds = (float(row.xF_Low), float(row.xF_High))
    if abs(raw_xf_bounds[1] - raw_xf_bounds[0]) < 1.0e-12:
        # E605 is tabulated at xF=-0.1 without an explicit width.  The
        # earlier identity probe used the published-looking +/-0.005 window;
        # retain that as an explicit diagnostic assumption.
        xf_low, xf_high = float(row.xF) - 0.005, float(row.xF) + 0.005
    else:
        xf_low, xf_high = sorted(raw_xf_bounds)
    m_edges = np.linspace(float(row.QM_Low), float(row.QM_High), args.q_slices + 1)
    records = []
    raw_total = 0.0
    raw_unc2 = 0.0
    for i, (mlo, mhi) in enumerate(zip(m_edges[:-1], m_edges[1:])):
        mmid = math.sqrt(0.5 * (mlo * mlo + mhi * mhi) + qmid * qmid)
        ylo, yhi = sorted((map_y(xf_low, mmid, sroot), map_y(xf_high, mmid, sroot)))
        sub = row.copy()
        sub["QM_Low"], sub["QM_High"] = mlo, mhi
        sub["y_Low"], sub["y_High"] = ylo, yhi
        name = f"{args.dataset}_{args.row}_q2slice{i}_calls{args.calls}"
        card = out / "cards" / f"{name}.in"
        log = out / "logs" / f"{name}.log"
        text = runner.make_card(sub, name=name, g1=1.017, calls=args.calls, seed=20260860 + i, cores=4)
        if args.expcreg is not None:
            text += f"\n# Isolated regularization audit; not a production setting.\nexpcreg = {float(args.expcreg)}\n"
        card.write_text(text)
        with log.open("w") as handle:
            subprocess.run([str(DYTURBO), str(card)], cwd=DYROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True, timeout=900)
        table = DYROOT / f"{name}.txt"
        value, unc = runner.load_runner().parse_first_value(table)
        raw_total += float(value)
        raw_unc2 += float(unc) ** 2
        records.append({"slice": i, "Q_low": mlo, "Q_high": mhi, "y_low": ylo, "y_high": yhi,
                        "raw_fb_per_nucleus": float(value), "raw_unc_fb_per_nucleus": float(unc),
                        "card": str(card), "log": str(log), "table": str(table)})
    z, target_a, target_status = runner.TARGETS[args.dataset]
    dq2 = qhigh * qhigh - qlow * qlow
    dxf = xf_high - xf_low
    qmid_data = float(row.qT)
    m_data = math.sqrt(float(row.QM) ** 2 + qmid_data * qmid_data)
    jac_center = 2.0 * m_data * math.cosh(float(row.y)) / sroot
    pred_a = raw_total * jac_center / (1000.0 * target_a * math.pi * dq2 * dxf)
    pred_cs = pred_a / float(row.PreFactor)
    summary = {
        "status": "isolated_fixed_xf_mapped_q2_probe",
        "dataset": args.dataset, "row_id": str(row.row_id), "q2_slices": args.q_slices,
        "calls_per_slice": args.calls, "qT_low": qlow, "qT_high": qhigh,
        "qT_floor_GeV": qfloor,
        "expcreg": args.expcreg,
        "xF_low": xf_low, "xF_high": xf_high, "target_Z": z, "target_A": target_a,
        "target_status": target_status, "raw_total_fb_per_nucleus": raw_total,
        "raw_total_unc_fb_per_nucleus": math.sqrt(raw_unc2), "central_xF_jacobian": jac_center,
        "predicted_A": pred_a, "data_A": float(row.A), "predicted_A_to_data": pred_a / float(row.A),
        "predicted_CS": pred_cs, "data_CS": float(row.CS), "predicted_CS_to_data": pred_cs / float(row.CS),
        "observable": "fixed-y dsdqt2 cards with Q-dependent xF-to-y mapping; A=raw*J/[1000*A*pi*Delta(qT2)*Delta(xF)]",
        "slices": records, "frozen_production_modified": False,
    }
    (out / "probe_status.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
