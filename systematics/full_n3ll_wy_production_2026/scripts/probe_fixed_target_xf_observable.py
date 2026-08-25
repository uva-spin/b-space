#!/usr/bin/env python3
"""Isolated fixed-target x_F/invariant-observable convention probe.

The earlier observable probe used a narrow central-y slice for E288 even
though the published E288 rows carry x_F bins.  This diagnostic evaluates the
same representative row with DYTurbo's explicit ``dsdxf`` Jacobian and keeps
the x_F interval in the audit metadata.  It never writes a production grid.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import pandas as pd

from run_dyturbo_full_n3ll_nnlo_probe import DYTURBO, DYROOT, full_card_text, load_runner


SYSTEMATICS = Path(__file__).resolve().parents[2]
PROJECT = SYSTEMATICS.parent
DATA_ROOT = PROJECT / "Data/v23a_fixed_target_plus_tevatron_absolute_fit_ready"
OUT_ROOT = SYSTEMATICS / "full_n3ll_wy_production_2026/reports/fixed_target_xf_observable_probe"


def parse_first(path: Path) -> tuple[float, float]:
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2 and not parts[0].startswith("#"):
            try:
                return float(parts[0]), float(parts[1])
            except ValueError:
                continue
    raise RuntimeError(f"no numeric table row in {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="E288_200")
    ap.add_argument("--row", default="E288_200:0")
    ap.add_argument("--calls", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=20260826)
    ap.add_argument("--target-z", type=float, default=4.0)
    ap.add_argument("--target-a", type=float, default=9.0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    load_runner()
    source = pd.read_csv(DATA_ROOT / f"{args.dataset}.csv")
    selected = source[source.row_id.eq(args.row)]
    if len(selected) != 1:
        raise SystemExit(f"row is not unique: {args.row}")
    row = selected.iloc[0].copy()
    q = float(row.qT)
    dq = max(1.0e-3, 1.0e-3 * q)
    qlo, qhi = max(1.0e-6, q - dq), q + dq
    row["qT_low"], row["qT_high"] = qlo, qhi
    xflo, xfhi = float(row.xF_Low), float(row.xF_High)
    if not xfhi > xflo:
        raise SystemExit(f"row has no positive x_F interval: {args.row}")
    name = f"{args.dataset}_{args.row.split(':')[-1]}_xf_full_wy"
    out = OUT_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    card = out / f"{name}.in"
    log = out / f"{name}.log"
    table = DYROOT / f"{name}.txt"

    text = full_card_text(row, output_name=name, pdf_set="NNPDF40_nnlo_as_01180",
                          pdf_member=0, cores=1, calls=int(args.calls), seed=int(args.seed))
    text = text.replace("ih2          = -1", "ih2          = 1", 1)
    text = text.replace("g1 = 0.0", "g1 = 1.017", 1)
    text = text.replace("VJquad = false", "VJquad = true", 1)
    text = text.replace("intDimVJ   = -1", "intDimVJ   = 3", 1)
    text = text.replace("makecuts = true", "makecuts = false", 1)
    text = text.replace("ptbinwidth = false", "ptbinwidth = true", 1)
    text = text.replace("ybinwidth = false", "ybinwidth = false", 1)
    text = text.replace("nproc        = 3", (
        "nproc        = 3\n"
        "nuclearpdf   = true\n"
        "Z1           = 1\nA1           = 1\n"
        f"Z2           = {args.target_z:.12g}\nA2           = {args.target_a:.12g}"
    ), 1)
    text += "\ndsdqt2 = false\ndsdxf = true\nedsdp3 = true\n"
    text += f"xfmin = {xflo:.12g}\nxfmax = {xfhi:.12g}\n"
    text = re.sub(r"qt_bins = \[ [^\]]+ \]", f"qt_bins = [ {qlo:.12g} {qhi:.12g} ]", text, count=1)
    text = re.sub(r"y_bins  = \[ [^\]]+ \]", "y_bins  = [ -10 10 ]", text, count=1)
    text = re.sub(r"m_bins  = \[ [^\]]+ \]", f"m_bins  = [ {float(row.QM_Low):.12g} {float(row.QM_High):.12g} ]", text, count=1)
    card.write_text(text)

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(p for p in ("/home/dustin/miniforge3/envs/pdf-fit/lib", str(DYROOT / "lib"), env.get("LD_LIBRARY_PATH", "")) if p)
    if args.force and table.exists():
        table.unlink()
    proc = None
    if not table.exists():
        with log.open("w") as handle:
            proc = subprocess.run([str(DYTURBO), str(card)], cwd=DYROOT, env=env,
                                  stdout=handle, stderr=subprocess.STDOUT,
                                  timeout=3600, check=False)
    finite = table.exists() and table.stat().st_size > 0
    value = uncertainty = None
    if finite:
        value, uncertainty = parse_first(table)
    result = {
        "status": "fixed_target_xf_observable_probe_complete_not_production" if finite and (proc is None or proc.returncode == 0) else "fixed_target_xf_observable_probe_failed",
        "dataset": args.dataset, "row_id": args.row,
        "formula": "W_N3LL + (FO_NNLO - ASY_NNLO)", "engine": str(DYTURBO),
        "order": {"resummation": "N3LL_unprimed", "fixed_order": "NNLO_VJ", "primed": False},
        "observable": "dsdxf=true plus edsdp3=true",
        "qT_GeV": q, "qT_bin_GeV": [qlo, qhi],
        "xF_bin": [xflo, xfhi], "xF_width": xfhi - xflo,
        "QM_bin_GeV": [float(row.QM_Low), float(row.QM_High)],
        "target": {"Z": float(args.target_z), "A": float(args.target_a), "normalization_note": "DYTurbo nuclearpdf whole-nucleus output must be divided by A for a per-nucleon comparison"},
        "raw_dyturbo_value": value, "raw_dyturbo_uncertainty": uncertainty,
        "data_CS": float(row.CS), "data_error": float(row.error),
        "raw_to_data_ratio": None if value is None else value / float(row.CS),
        "per_nucleon_raw_to_data_ratio": None if value is None else value / float(args.target_a) / float(row.CS),
        "card": str(card), "log": str(log), "table": str(table),
        "return_code": None if proc is None else proc.returncode,
        "interpretation": "observable/Jacobian and target-normalization diagnostic only; no fixed-target production authorization",
        "frozen_baseline_unchanged": True, "production_outputs_modified": False,
    }
    (OUT_ROOT / "probe_status.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result["status"].endswith("failed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
