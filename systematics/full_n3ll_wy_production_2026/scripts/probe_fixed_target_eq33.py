#!/usr/bin/env python3
"""Test the published fixed-target Eq. (3.3) measure in isolation.

This uses DYTurbo with ``dsdqt2=true`` and fixed-y bins, then records the
candidate invariant conversion

    (1/pi) * integral(dQ^2) d sigma/(dQ^2 d(qT^2) d eta)

by dividing the bin-integrated result by the qT-squared and rapidity widths.
It is a convention audit only and never modifies fit-ready data or frozen
production outputs.
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
OUT = BASE / "reports/fixed_target_eq33_probe"


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
    ap.add_argument("--calls", type=int, default=30_000_000)
    ap.add_argument("--seed", type=int, default=20260892)
    ap.add_argument("--g1", type=float, default=1.017)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    load_runner()
    OUT.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        p for p in ("/home/dustin/miniforge3/envs/pdf-fit/lib", str(DYROOT / "lib"), env.get("LD_LIBRARY_PATH", "")) if p
    )
    reps = [("E288_200", "E288_200:0", 4.0, 9.0121831), ("E605", "E605:0", 29.0, 63.546)]
    rows = []
    for i, (dataset, row_id, z, a) in enumerate(reps):
        frame = pd.read_csv(DATA_ROOT / f"{dataset}.csv")
        hit = frame[frame.row_id.eq(row_id)]
        if len(hit) != 1:
            raise RuntimeError(row_id)
        row = hit.iloc[0].copy()
        qhalf = 0.125 if dataset == "E772" else 0.1
        qlo, qhi = max(0.0, float(row.qT) - qhalf), float(row.qT) + qhalf
        row["qT_low"], row["qT_high"] = qlo, qhi
        ylo, yhi = sorted((float(row.y_Low), float(row.y_High)))
        mlo, mhi = float(row.QM_Low), float(row.QM_High)
        name = f"{dataset}_eq33"
        card, log, table = OUT / f"{name}.in", OUT / f"{name}.log", DYROOT / f"{name}.txt"
        text = full_card_text(row, output_name=name, pdf_set="NNPDF40_nnlo_as_01180", pdf_member=0,
                              cores=4, calls=int(args.calls), seed=int(args.seed) + i)
        text = text.replace("makecuts = true", "makecuts = false", 1)
        text = text.replace("ih2          = -1", "ih2          = 1", 1)
        text = text.replace("g1 = 0.0", f"g1 = {args.g1:.12g}", 1)
        text = text.replace("nproc        = 3", f"nproc        = 3\nnuclearpdf = true\nZ1 = 1\nA1 = 1\nZ2 = {z:.12g}\nA2 = {a:.12g}", 1)
        text = re.sub(r"qt_bins = \[ [^\]]+ \]", f"qt_bins = [ {qlo:.12g} {qhi:.12g} ]", text, count=1)
        text = re.sub(r"y_bins  = \[ [^\]]+ \]", f"y_bins  = [ {ylo:.12g} {yhi:.12g} ]", text, count=1)
        text = re.sub(r"m_bins  = \[ [^\]]+ \]", f"m_bins  = [ {mlo:.12g} {mhi:.12g} ]", text, count=1)
        text = text.replace("ptbinwidth = true", "ptbinwidth = false", 1)
        text = text.replace("ybinwidth = true", "ybinwidth = false", 1)
        text += "\n# Eq. (3.3) observable audit.\ndsdqt2 = true\ndsdxf = false\nedsdp3 = false\n"
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
        eq33 = value / (1000.0 * a * math.pi * dq2 * dy)
        eq33_unc = unc / (1000.0 * a * math.pi * dq2 * dy)
        rows.append({"dataset": dataset, "row_id": row_id, "target_Z": z, "target_A": a,
                     "qT_bin": [qlo, qhi], "qT2_width": dq2, "Q_bin": [mlo, mhi],
                     "y_bin": [ylo, yhi], "raw_integrated_pb": value / 1000.0,
                     "raw_unc_pb": unc / 1000.0, "eq33_CS": eq33, "eq33_unc": eq33_unc,
                     "data_CS": float(row.CS), "data_error": float(row.error),
                     "eq33_to_data": eq33 / float(row.CS), "card": str(card), "log": str(log),
                     "table": str(table), "return_code": None if proc is None else proc.returncode})
    result = {"status": "fixed_target_eq33_probe_complete_not_production",
              "formula": "W_N3LL + (FO_NNLO - ASY_NNLO)",
              "observable": "Eq. (3.3): (1/pi) integral dQ^2 dSigma/(dQ^2 d(qT^2) deta)",
              "calls_per_vegas_component": int(args.calls), "g1_GeV2": float(args.g1),
              "rows": rows, "frozen_baseline_unchanged": True,
              "production_outputs_modified": False, "promotion_authorized": False}
    (OUT / "probe_status.json").write_text(json.dumps(result, indent=2) + "\n")
    pd.DataFrame(rows).to_csv(OUT / "probe_rows.csv", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
