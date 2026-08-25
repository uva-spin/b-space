#!/usr/bin/env python3
"""Test Eq. (3.3) at the published fixed-qT point rather than a qT bin average.

The fit-ready fixed-target rows contain a qT point (the first E288/E605 rows
are qT=0.1 GeV).  The earlier Eq. (3.3) probe integrated over [qT-0.1,
qT+0.1] and compared that bin average with the point value.  This isolated
diagnostic repeats the calculation in narrow qT windows and records the
extrapolation toward the point limit.  It never changes data or production.
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
OUT = BASE / "reports/fixed_target_eq33_point_probe"


def parse_first(path: Path) -> tuple[float, float]:
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) >= 2 and not p[0].startswith("#"):
            try:
                return float(p[0]), float(p[1])
            except ValueError:
                pass
    raise RuntimeError(f"no numeric table row in {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, default=10_000_000)
    ap.add_argument("--seed", type=int, default=20260894)
    ap.add_argument("--g1", type=float, default=1.017)
    ap.add_argument("--qT-widths", nargs="+", type=float, default=[0.2, 0.05, 0.02, 0.01])
    ap.add_argument("--target-mode", choices=("isotope_approx", "p40n60"), default="isotope_approx")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    load_runner()
    out = OUT / args.target_mode
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        p for p in ("/home/dustin/miniforge3/envs/pdf-fit/lib", str(DYROOT / "lib"), env.get("LD_LIBRARY_PATH", "")) if p
    )
    if args.target_mode == "p40n60":
        reps = [("E288_200", "E288_200:0", 0.4, 1.0), ("E605", "E605:0", 0.4, 1.0)]
    else:
        reps = [("E288_200", "E288_200:0", 4.0, 9.0121831), ("E605", "E605:0", 29.0, 63.546)]
    rows = []
    serial = 0
    for dataset, row_id, z, a in reps:
        frame = pd.read_csv(DATA_ROOT / f"{dataset}.csv")
        hit = frame[frame.row_id.eq(row_id)]
        if len(hit) != 1:
            raise RuntimeError(row_id)
        row = hit.iloc[0].copy()
        qcenter = float(row.qT)
        ylo, yhi = sorted((float(row.y_Low), float(row.y_High)))
        mlo, mhi = float(row.QM_Low), float(row.QM_High)
        for width in args.qT_widths:
            if width <= 0.0 or width > 2.0 * qcenter:
                raise ValueError(f"invalid narrow qT width {width} at qT={qcenter}")
            qlo, qhi = qcenter - 0.5 * width, qcenter + 0.5 * width
            name = f"{dataset}_eq33_point_qw{str(width).replace('.', 'p')}"
            card, log, table = out / f"{name}.in", out / f"{name}.log", DYROOT / f"{name}.txt"
            row["qT_low"], row["qT_high"] = qlo, qhi
            text = full_card_text(row, output_name=name, pdf_set="NNPDF40_nnlo_as_01180", pdf_member=0,
                                  cores=4, calls=int(args.calls), seed=int(args.seed) + serial)
            serial += 1
            text = text.replace("makecuts = true", "makecuts = false", 1)
            text = text.replace("ih2          = -1", "ih2          = 1", 1)
            text = text.replace("g1 = 0.0", f"g1 = {args.g1:.12g}", 1)
            text = text.replace("nproc        = 3", f"nproc        = 3\nnuclearpdf = true\nZ1 = 1\nA1 = 1\nZ2 = {z:.12g}\nA2 = {a:.12g}", 1)
            text = re.sub(r"qt_bins = \[ [^\]]+ \]", f"qt_bins = [ {qlo:.12g} {qhi:.12g} ]", text, count=1)
            text = re.sub(r"y_bins  = \[ [^\]]+ \]", f"y_bins  = [ {ylo:.12g} {yhi:.12g} ]", text, count=1)
            text = re.sub(r"m_bins  = \[ [^\]]+ \]", f"m_bins  = [ {mlo:.12g} {mhi:.12g} ]", text, count=1)
            text = text.replace("ptbinwidth = true", "ptbinwidth = false", 1)
            text = text.replace("ybinwidth = true", "ybinwidth = false", 1)
            text += "\n# Eq. (3.3) narrow-qT point audit.\ndsdqt2 = true\ndsdxf = false\nedsdp3 = false\n"
            card.write_text(text)
            if args.force:
                table.unlink(missing_ok=True)
            proc = None
            if not table.exists():
                with log.open("w") as handle:
                    proc = subprocess.run([str(DYTURBO), str(card)], cwd=DYROOT, env=env,
                                          stdout=handle, stderr=subprocess.STDOUT, check=False, timeout=3600)
            value, unc = parse_first(table)
            dq2 = qhi * qhi - qlo * qlo
            dy = yhi - ylo
            candidate = value / (1000.0 * a * math.pi * dq2 * dy)
            candidate_unc = unc / (1000.0 * a * math.pi * dq2 * dy)
            rows.append({"dataset": dataset, "row_id": row_id, "target_Z": z, "target_A": a,
                         "qT_center": qcenter, "qT_width": width, "qT_bin": [qlo, qhi],
                         "qT2_width": dq2, "Q_bin": [mlo, mhi], "y_bin": [ylo, yhi],
                         "raw_integrated_pb": value / 1000.0, "raw_unc_pb": unc / 1000.0,
                         "eq33_CS": candidate, "eq33_unc": candidate_unc,
                         "data_CS": float(row.CS), "data_error": float(row.error),
                         "eq33_to_data": candidate / float(row.CS), "card": str(card),
                         "log": str(log), "table": str(table),
                         "return_code": None if proc is None else proc.returncode})
            pd.DataFrame(rows).to_csv(out / "probe_rows.csv", index=False)
            print(json.dumps(rows[-1]), flush=True)
    result = {"status": "fixed_target_eq33_point_probe_complete_not_production",
              "formula": "W_N3LL + (FO_NNLO - ASY_NNLO)",
              "observable": "Eq. (3.3) evaluated in shrinking qT windows about the published point",
              "calls_per_vegas_component": int(args.calls), "g1_GeV2": float(args.g1),
              "rows": rows, "finding": "point-limit conversion diagnostic; no promotion authorized",
              "frozen_baseline_unchanged": True, "production_outputs_modified": False,
              "promotion_authorized": False}
    result["target_mode"] = args.target_mode
    (out / "probe_status.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
