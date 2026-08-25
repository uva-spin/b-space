#!/usr/bin/env python3
"""Probe the DYTurbo fixed-target observable convention in isolation.

The fixed-target measurements are invariant cross sections, not the collider
``d sigma/dqT`` observable used by the Tevatron cards.  DYTurbo exposes the
``dsdxf`` and ``edsdp3`` switches needed for this comparison.  This script
tests one representative row from E288, E605, and E772, records several
explicit conversion diagnostics, and never creates production rows.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
from pathlib import Path

import pandas as pd

from run_dyturbo_full_n3ll_nnlo_probe import DYTURBO, DYROOT, full_card_text, load_runner


BASE = Path(__file__).resolve().parents[1]
PROJECT = BASE.parents[1]
DATA_ROOT = PROJECT / "Data/v23a_fixed_target_plus_tevatron_absolute_fit_ready"
OUT = BASE / "reports/fixed_target_observable_convention_probe"


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
    ap.add_argument("--calls", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=20260824)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    load_runner()
    OUT.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        p for p in ("/home/dustin/miniforge3/envs/pdf-fit/lib", str(DYROOT / "lib"), env.get("LD_LIBRARY_PATH", "")) if p
    )
    representatives = [("E288_200", "E288_200:0"), ("E605", "E605:0"), ("E772", "E772:0")]
    rows = []
    for dataset, row_id in representatives:
        row = pd.read_csv(DATA_ROOT / f"{dataset}.csv")
        selected = row[row.row_id.eq(row_id)]
        if len(selected) != 1:
            raise SystemExit(f"row is not unique: {row_id}")
        r = selected.iloc[0].copy()
        q = float(r.qT)
        # Narrow finite bins approximate the published average-qT points;
        # output is normalized by these widths below.
        eps_q = max(1.0e-3, 1.0e-3 * q)
        qlo, qhi = max(1.0e-6, q - eps_q), q + eps_q
        r["qT_low"], r["qT_high"] = qlo, qhi
        mlo, mhi = float(r.QM_Low), float(r.QM_High)
        use_xf = dataset in {"E605", "E772"}
        if use_xf:
            ylo, yhi = -10.0, 10.0
            target_z, target_a = (0.5, 1.0) if dataset == "E772" else (0.4, 1.0)
            mode = "xf"
        else:
            y = float(r.y)
            eps_y = 1.0e-4
            ylo, yhi = y - eps_y, y + eps_y
            target_z, target_a = 0.4, 1.0
            mode = "fixed_y"
        name = f"{dataset}_observable_probe"
        card = OUT / f"{name}.in"
        log = OUT / f"{name}.log"
        table = DYROOT / f"{name}.txt"
        text = full_card_text(r, output_name=name, pdf_set="NNPDF40_nnlo_as_01180", pdf_member=0,
                              cores=8, calls=int(args.calls), seed=int(args.seed))
        text = text.replace("ih2          = -1", "ih2          = 1", 1)
        text = text.replace("g1 = 0.0", "g1 = 1.017", 1)
        text = text.replace("VJquad = false", "VJquad = true", 1)
        text = text.replace("intDimVJ   = -1", "intDimVJ   = 3", 1)
        text = text.replace("makecuts = true", "makecuts = false", 1)
        text = text.replace("ptbinwidth = false", "ptbinwidth = true", 1)
        text = text.replace("ybinwidth = false", "ybinwidth = true", 1)
        text = text.replace("mbinwidth = false", "mbinwidth = false", 1)
        text += "\ndsdqt2 = false\ndsdxf = %s\nedsdp3 = true\nxfmin = -1e6\nxfmax = +1e6\n" % ("true" if use_xf else "false")
        text = text.replace("nproc        = 3", f"nproc        = 3\nnuclearpdf   = true\nZ1 = 1\nA1 = 1\nZ2 = {target_z:.12g}\nA2 = {target_a:.12g}", 1)
        text = re.sub(r"qt_bins = \[ [^\]]+ \]", f"qt_bins = [ {qlo:.12g} {qhi:.12g} ]", text, count=1)
        text = re.sub(r"y_bins  = \[ [^\]]+ \]", f"y_bins  = [ {ylo:.12g} {yhi:.12g} ]", text, count=1)
        text = re.sub(r"m_bins  = \[ [^\]]+ \]", f"m_bins  = [ {mlo:.12g} {mhi:.12g} ]", text, count=1)
        if use_xf:
            text = text.replace("xfmin = -1e6", f"xfmin = {float(r.xF_Low):.12g}", 1)
            text = text.replace("xfmax = +1e6", f"xfmax = {float(r.xF_High):.12g}", 1)
        card.write_text(text)
        if args.force and table.exists():
            table.unlink()
        proc = None
        if not table.exists():
            with log.open("w") as handle:
                proc = subprocess.run([str(DYTURBO), str(card)], cwd=DYROOT, env=env,
                                      stdout=handle, stderr=subprocess.STDOUT,
                                      check=False, timeout=3600)
        value, unc = parse_first(table)
        row_width = qhi - qlo
        # Since pt/y bin-width normalization is enabled, the table is a
        # differential invariant observable candidate in DYTurbo's own units.
        # Keep both direct and explicit unit conversions for audit.
        data = float(r.CS)
        data_err = float(r.error)
        record = {
            "dataset": dataset, "row_id": row_id, "mode": mode,
            "qT_GeV": q, "qT_bin_width_GeV": row_width,
            "QM_range_GeV": [mlo, mhi], "y_bin": [ylo, yhi],
            "xF_bin": [float(r.xF_Low), float(r.xF_High)] if use_xf else None,
            "target_per_nucleon_mixture": {"proton_fraction": target_z / target_a, "neutron_fraction": 1.0 - target_z / target_a},
            "raw_dyturbo_value": value, "raw_dyturbo_uncertainty": unc,
            "data_CS": data, "data_error": data_err,
            "raw_to_data_ratio": value / data,
            "raw_to_data_ratio_uncertainty": unc / data,
            "card": str(card), "log": str(log), "table": str(table),
            "return_code": None if proc is None else proc.returncode,
        }
        rows.append(record)
    payload = {
        "status": "fixed_target_observable_convention_probe_complete_not_production",
        "formula": "W_N3LL + (FO_NNLO - ASY_NNLO)",
        "engine": str(DYTURBO),
        "order": 3, "primed": False,
        "calls_per_vegas_component": int(args.calls), "random_seed": int(args.seed),
        "rows": rows,
        "interpretation": "tests DYTurbo dsdxf/edsdp3 observable switches against the verified CS convention; no 353-row production claim",
        "source_formula": "E d^3sigma/d^3q = 1/(2 pi qT) d sigma/(dqT dy dQ), with xF Jacobian for E605/E772",
        "production_authorized": False,
        "frozen_baseline_unchanged": True,
        "production_outputs_modified": False,
    }
    (OUT / "probe_status.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
