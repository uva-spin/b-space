#!/usr/bin/env python3
"""Test deterministic/quadrature V+jet integration for fixed-target W+Y.

The first fixed-target capability probe used the automatic Vegas choice for
the NNLO V+jet term and was dominated by RES--CT--VJ cancellation.  DYTurbo
supports lower-dimensional quadrature when no lepton cuts are requested;
fixed-target bins have only qT, y, and mass-bin boundaries.  This isolated
probe changes only that integration choice and never touches the frozen
production package.
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
OUT_ROOT = SYSTEMATICS / "full_n3ll_wy_production_2026/reports/fixed_target_quadrature_probes"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="E288_200")
    ap.add_argument("--row", default="E288_200:0")
    ap.add_argument("--g1", type=float, default=1.017)
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--intdim-vj", type=int, choices=(3, 5), default=3)
    ap.add_argument("--calls", type=int, default=1_000_000)
    ap.add_argument("--target-z", type=float, default=None,
                    help="nuclear target proton count; omit for the prior proton-target probe")
    ap.add_argument("--target-a", type=float, default=None,
                    help="nuclear target mass count; requires --target-z")
    ap.add_argument("--makecuts", action="store_true", help="retain DYTurbo's cut-aware quadrature path")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    data = pd.read_csv(DATA_ROOT / f"{args.dataset}.csv")
    selected = data[data.row_id.eq(args.row)]
    if len(selected) != 1:
        raise SystemExit(f"requested row is not unique: {args.row}")
    row = selected.iloc[0].copy()
    if (args.target_z is None) != (args.target_a is None):
        raise SystemExit("--target-z and --target-a must be provided together")
    qlow, qhigh = max(0.0, float(row.qT) - 0.05), float(row.qT) + 0.05
    row["qT_low"], row["qT_high"] = qlow, qhigh
    ylow, yhigh = sorted((float(row.y_Low), float(row.y_High)))
    cut_tag = "cuts" if args.makecuts else "nocuts"
    target_tag = "proton" if args.target_z is None else f"nuc_z{args.target_z:g}_a{args.target_a:g}".replace(".", "p")
    name = f"{args.dataset}_{args.row.split(':')[-1]}_quadrature_vj_d{args.intdim_vj}_{cut_tag}_{target_tag}"
    out = OUT_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    card = out / f"{name}.in"
    log = out / f"{name}.log"
    table = DYROOT / f"{name}.txt"

    load_runner()
    text = full_card_text(row, output_name=name, pdf_set="NNPDF40_nnlo_as_01180",
                          pdf_member=0, cores=1, calls=int(args.calls), seed=int(args.seed))
    text = text.replace("ih2          = -1", "ih2          = 1", 1)
    if args.target_z is not None:
        # DYTurbo's nuclearpdf branch constructs the proton/neutron sum from
        # Z and A.  Keep this card-side declaration explicit and record the
        # target counts in the diagnostic metadata; no production files are
        # touched.
        text = text.replace("nproc        = 3", "nproc        = 3\nnuclearpdf   = true\nZ1           = 1\nZ2           = %.12g\nA1           = 1\nA2           = %.12g" % (args.target_z, args.target_a), 1)
    text = text.replace("g1 = 0.0", f"g1 = {args.g1:.12g}", 1)
    text = text.replace("VJquad = false", "VJquad = true", 1)
    text = text.replace("intDimVJ   = -1", f"intDimVJ   = {args.intdim_vj}", 1)
    # No lepton cuts are present in the fixed-target input.  The default
    # diagnostic therefore disables the cut-aware path; --makecuts tests the
    # formally supported 5D quadrature alternative explicitly.
    text = text.replace("makecuts = true", f"makecuts = {'true' if args.makecuts else 'false'}", 1)
    text = re.sub(r"qt_bins = \[ [^\]]+ \]", f"qt_bins = [ {qlow:.12g} {qhigh:.12g} ]", text, count=1)
    text = re.sub(r"y_bins  = \[ [^\]]+ \]", f"y_bins  = [ {ylow:.12g} {yhigh:.12g} ]", text, count=1)
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
    result = {
        "status": "fixed_target_quadrature_vj_probe_complete" if finite and (proc is None or proc.returncode == 0) else "fixed_target_quadrature_vj_probe_failed",
        "dataset": args.dataset,
        "row_id": args.row,
        "formula": "W_N3LL + (FO_NNLO - ASY_NNLO)",
        "engine": str(DYTURBO),
        "order": {"resummation": "N3LL_unprimed", "fixed_order": "NNLO_VJ", "primed": False},
        "g1_GeV2": float(args.g1),
        "integration": {"VJquad": True, "intDimVJ": int(args.intdim_vj), "makecuts": bool(args.makecuts), "calls_per_vegas_component": int(args.calls)},
        "target": {"mode": "nuclearpdf" if args.target_z is not None else "proton", "Z": args.target_z, "A": args.target_a, "normalization_note": "DYTurbo nuclearpdf returns the whole-nucleus PDF sum; per-nucleon comparison requires division by A" if args.target_z is not None else "proton target"},
        "random_seed": int(args.seed),
        "card": str(card), "log": str(log), "table": str(table),
        "return_code": None if proc is None else proc.returncode,
        "table_nonempty": bool(finite),
        "interpretation": "isolated fixed-target integration-method test; not production authorization",
        "frozen_baseline_unchanged": True,
        "production_outputs_modified": False,
    }
    (out / "probe_status.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result["status"].endswith("failed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
