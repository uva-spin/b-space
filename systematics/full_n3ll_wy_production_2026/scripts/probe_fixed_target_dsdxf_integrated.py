#!/usr/bin/env python3
"""Isolated audit of DYTurbo's integrated ``dsdxf`` normalization.

The previous closure divided a non-integrable ``dsdxf`` result by a nominal
Delta-xF after also enabling ``ybinwidth``.  This probe instead lets DYTurbo
integrate the finite xF interval with its own Jacobian, leaves all bin-width
normalizations off, and records several explicit conversions.  It is a
diagnostic only; no fit-ready or frozen file is changed.
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
OUT = BASE / "reports/fixed_target_dsdxf_integrated_probe"


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
    ap.add_argument("--calls", type=int, default=30_000_000)
    ap.add_argument("--seed", type=int, default=20260891)
    ap.add_argument("--g1", type=float, default=1.017)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    load_runner()
    OUT.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        p for p in ("/home/dustin/miniforge3/envs/pdf-fit/lib", str(DYROOT / "lib"), env.get("LD_LIBRARY_PATH", "")) if p
    )
    reps = [
        ("E605", "E605:0", 29.0, 63.546),
        ("E772", "E772:0", 0.5, 1.0),
    ]
    records = []
    for index, (dataset, row_id, z, a) in enumerate(reps):
        frame = pd.read_csv(DATA_ROOT / f"{dataset}.csv")
        match = frame[frame.row_id.eq(row_id)]
        if len(match) != 1:
            raise RuntimeError(f"row is not unique: {row_id}")
        row = match.iloc[0].copy()
        q_half = 0.125 if dataset == "E772" else 0.1
        qlo, qhi = max(0.0, float(row.qT) - q_half), float(row.qT) + q_half
        row["qT_low"], row["qT_high"] = qlo, qhi
        mlo, mhi = float(row.QM_Low), float(row.QM_High)
        sqrts = float(row.SqrtS)
        if dataset == "E605":
            # The source row has no xF limits, but its y limits define the
            # published fixed-xF bin in the fit-ready table.  Map those limits
            # with the same xF(y,m) relation used by DYTurbo.
            yvals = sorted((float(row.y_Low), float(row.y_High)))
            xfvals = [2.0 * float(row.QM) / sqrts * math.sinh(y) for y in yvals]
            xflo, xfhi = min(xfvals), max(xfvals)
        else:
            xflo, xfhi = float(row.xF_Low), float(row.xF_High)
        name = f"{dataset}_dsdxf_integrated"
        card = OUT / f"{name}.in"
        log = OUT / f"{name}.log"
        table = DYROOT / f"{name}.txt"
        text = full_card_text(
            row, output_name=name, pdf_set="NNPDF40_nnlo_as_01180", pdf_member=0,
            cores=4, calls=int(args.calls), seed=int(args.seed) + index,
        )
        text = text.replace("makecuts = true", "makecuts = false", 1)
        text = text.replace("ih2          = -1", "ih2          = 1", 1)
        text = text.replace("g1 = 0.0", f"g1 = {args.g1:.12g}", 1)
        text = text.replace(
            "nproc        = 3",
            "nproc        = 3\n"
            f"nuclearpdf   = true\nZ1 = 1\nA1 = 1\nZ2 = {z:.12g}\nA2 = {a:.12g}",
            1,
        )
        text = re.sub(r"qt_bins = \[ [^\]]+ \]", f"qt_bins = [ {qlo:.12g} {qhi:.12g} ]", text, count=1)
        text = re.sub(r"y_bins  = \[ [^\]]+ \]", "y_bins  = [ -10 10 ]", text, count=1)
        text = re.sub(r"m_bins  = \[ [^\]]+ \]", f"m_bins  = [ {mlo:.12g} {mhi:.12g} ]", text, count=1)
        text = text.replace("ptbinwidth = true", "ptbinwidth = false", 1)
        text = text.replace("ybinwidth = true", "ybinwidth = false", 1)
        text += (
            "\n# Integrated fixed-xF diagnostic: DYTurbo supplies dxF/dy.\n"
            "dsdqt2 = false\n"
            "dsdxf = true\n"
            "edsdp3 = true\n"
            f"xfmin = {xflo:.12g}\nxfmax = {xfhi:.12g}\n"
        )
        card.write_text(text)
        if args.force:
            table.unlink(missing_ok=True)
        proc = None
        if not table.exists():
            with log.open("w") as handle:
                proc = subprocess.run(
                    [str(DYTURBO), str(card)], cwd=DYROOT, env=env,
                    stdout=handle, stderr=subprocess.STDOUT, check=False, timeout=3600,
                )
        value, uncertainty = parse_first(table)
        q_width = qhi - qlo
        m_width = mhi - mlo
        # Candidate conversions are intentionally all retained: the source
        # convention is what this probe is determining.
        records.append({
            "dataset": dataset, "row_id": row_id,
            "target_Z": z, "target_A": a,
            "qT_bin": [qlo, qhi], "Q_bin": [mlo, mhi],
            "xF_bin": [xflo, xfhi], "xF_width": xfhi - xflo,
            "raw_integrated_fb": value, "raw_unc_fb": uncertainty,
            "raw_per_nucleon_fb": value / a, "raw_per_nucleon_unc_fb": uncertainty / a,
            "candidate_pb_per_GeV": math.pi * value / (1000.0 * a * q_width * m_width),
            "candidate_pb_per_GeV_per_xf": math.pi * value / (1000.0 * a * q_width * m_width * (xfhi - xflo)),
            "data_CS": float(row.CS), "data_error": float(row.error),
            "candidate_to_data": math.pi * value / (1000.0 * a * q_width * m_width) / float(row.CS),
            "card": str(card), "log": str(log), "table": str(table),
            "return_code": None if proc is None else proc.returncode,
        })
    result = {
        "status": "fixed_target_dsdxf_integrated_probe_complete_not_production",
        "formula": "W_N3LL + (FO_NNLO - ASY_NNLO)",
        "observable": "integrated finite-xF DYTurbo dsdxf with edsdp3, no bin-width flags",
        "calls_per_vegas_component": int(args.calls), "g1_GeV2": float(args.g1),
        "rows": records,
        "finding": "diagnostic candidates retained; no conversion or production promotion is authorized",
        "frozen_baseline_unchanged": True, "production_outputs_modified": False,
        "promotion_authorized": False,
    }
    (OUT / "probe_status.json").write_text(json.dumps(result, indent=2) + "\n")
    pd.DataFrame(records).to_csv(OUT / "probe_rows.csv", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
