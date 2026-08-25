#!/usr/bin/env python3
"""Representative fixed-target CS-convention closure for the external W+Y engine.

This is deliberately separate from the fixed-target production runner.  The
fit-ready tables use the historical CS convention, while DYTurbo returns a
cross section in fb after integrating the requested phase-space bin.  E288 is
handled as a fixed-y bin; E605 and E772 are evaluated in finite x_F bins with
``dsdxf``.  The conversion is recorded explicitly as

    For E288 this is applied to the integrated fixed-y bin.  For the
    dsdxf/edsdp3 cards, DYTurbo returns the result normalized by the qT and
    auxiliary y-bin widths, so the audit explicitly replaces Delta-y by the
    requested Delta-xF.

It is a diagnostic closure only and never writes frozen or production files.
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
OUT = BASE / "reports/fixed_target_cs_convention_closure"


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
    ap.add_argument("--seed", type=int, default=20260890)
    ap.add_argument("--g1", type=float, default=1.017)
    ap.add_argument("--xf-half-width", type=float, default=1.0e-3)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    load_runner()
    OUT.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        p for p in ("/home/dustin/miniforge3/envs/pdf-fit/lib", str(DYROOT / "lib"), env.get("LD_LIBRARY_PATH", "")) if p
    )
    reps = [
        ("E288_200", "E288_200:0", "fixed_y", 0.4, 1.0),
        ("E605", "E605:0", "fixed_xf", 0.4, 1.0),
        ("E772", "E772:0", "fixed_xf", 0.5, 1.0),
    ]
    records = []
    for index, (dataset, row_id, mode, z, a) in enumerate(reps):
        data = pd.read_csv(DATA_ROOT / f"{dataset}.csv")
        selected = data[data.row_id.eq(row_id)]
        if len(selected) != 1:
            raise RuntimeError(f"row is not unique: {row_id}")
        row = selected.iloc[0].copy()
        q_half = 0.125 if dataset == "E772" else 0.1
        qlo, qhi = max(0.0, float(row.qT) - q_half), float(row.qT) + q_half
        row["qT_low"], row["qT_high"] = qlo, qhi
        mlo, mhi = float(row.QM_Low), float(row.QM_High)
        if mode == "fixed_y":
            ylo, yhi = sorted((float(row.y_Low), float(row.y_High)))
            variable_width = yhi - ylo
            xflo = xfhi = None
        else:
            xcenter = float(row.xF)
            xflo, xfhi = xcenter - args.xf_half_width, xcenter + args.xf_half_width
            ylo, yhi = -10.0, 10.0
            variable_width = xfhi - xflo

        name = f"{dataset}_cs_closure_{mode}"
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
        text = re.sub(r"y_bins  = \[ [^\]]+ \]", f"y_bins  = [ {ylo:.12g} {yhi:.12g} ]", text, count=1)
        text = re.sub(r"m_bins  = \[ [^\]]+ \]", f"m_bins  = [ {mlo:.12g} {mhi:.12g} ]", text, count=1)
        if mode == "fixed_xf":
            text = text.replace("ptbinwidth = false", "ptbinwidth = true", 1)
            text = text.replace("ybinwidth = false", "ybinwidth = true", 1)
            text += (
                "\n# Explicit fixed-xF observable diagnostic.\n"
                "dsdqt2 = true\n"
                "dsdxf = true\n"
                "edsdp3 = true\n"
                f"xfmin = {xflo:.12g}\nxfmax = {xfhi:.12g}\n"
            )
        else:
            text += "\n# Fixed-y E288 diagnostic; no xF Jacobian.\ndsdxf = false\nedsdp3 = false\n"
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
        q_width_mass = mhi - mlo
        # DYTurbo's returned value is an integrated fb bin.  A=1 is the
        # per-nucleon 40/60 or 50/50 mixture used here, but retain A in the
        # formula so a whole-nucleus test remains auditable.
        if mode == "fixed_y":
            conversion_factor = 1.0 / (q_width * variable_width)
        else:
            conversion_factor = (yhi - ylo) / variable_width
        cs = math.pi * value * conversion_factor / (1000.0 * a * q_width_mass)
        cs_unc = math.pi * uncertainty * conversion_factor / (1000.0 * a * q_width_mass)
        data_cs = float(row.CS)
        records.append({
            "dataset": dataset, "row_id": row_id, "mode": mode,
            "target_Z": z, "target_A": a,
            "qT_bin": [qlo, qhi], "Q_bin": [mlo, mhi],
            "y_bin": [ylo, yhi] if mode == "fixed_y" else None,
            "xF_bin": [xflo, xfhi] if mode == "fixed_xf" else None,
            "integrated_variable_width": variable_width,
            "raw_dyturbo_fb_per_bin": value,
            "raw_dyturbo_unc_fb_per_bin": uncertainty,
            "converted_CS": cs, "converted_CS_unc": cs_unc,
            "data_CS": data_cs, "data_error": float(row.error),
            "prediction_to_data": cs / data_cs,
            "card": str(card), "log": str(log), "table": str(table),
            "return_code": None if proc is None else proc.returncode,
        })
    result = {
        "status": "fixed_target_cs_convention_closure_complete_not_production",
        "formula": "W_N3LL + (FO_NNLO - ASY_NNLO)",
        "conversion": "CS = pi * sigma_fb / (1000*A*Delta_qT*Delta_Q*Delta_y_or_Delta_xF)",
        "calls_per_vegas_component": int(args.calls), "g1_GeV2": float(args.g1),
        "xf_half_width": float(args.xf_half_width), "rows": records,
        "frozen_baseline_unchanged": True, "production_outputs_modified": False,
        "promotion_authorized": False,
    }
    (OUT / "closure_status.json").write_text(json.dumps(result, indent=2) + "\n")
    pd.DataFrame(records).to_csv(OUT / "closure_rows.csv", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
