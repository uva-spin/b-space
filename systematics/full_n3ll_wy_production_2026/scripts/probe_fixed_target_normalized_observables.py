#!/usr/bin/env python3
"""Isolated fixed-target observable conversion audit.

This diagnostic keeps the DYTurbo calculation in its native differential
variables and converts the result to the fit-ready fixed-target convention
explicitly.  It tests two different published observables:

* E288 fixed-rapidity rows: use Eq. (3.3) to obtain the invariant cross
  section and then apply the row's ``PreFactor``;
* E605/E772 fixed-x_F rows: request ``dsdxf`` and convert directly to
  ``d sigma/(dq_T dx_F)`` before applying ``PreFactor``.

No input data, frozen production cache, or accepted result is modified.
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
OUT = BASE / "reports/fixed_target_normalized_observable_probe"


def parse_first(path: Path) -> tuple[float, float]:
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) >= 2 and not p[0].startswith("#"):
            try:
                return float(p[0]), float(p[1])
            except ValueError:
                continue
    raise RuntimeError(f"no numeric table row in {path}")


def card_for(row: pd.Series, *, name: str, mode: str, z: float, a: float,
             q_width: float, xf_width: float, calls: int, seed: int) -> tuple[str, dict]:
    q = float(row.qT)
    qlo, qhi = max(1.0e-5, q - 0.5 * q_width), q + 0.5 * q_width
    work = row.copy()
    work["qT_low"], work["qT_high"] = qlo, qhi
    if mode == "fixed_y":
        ylo, yhi = sorted((float(row.y_Low), float(row.y_High)))
        xflo = xfhi = None
    elif mode == "fixed_xf":
        xfc = float(row.xF)
        xflo, xfhi = xfc - 0.5 * xf_width, xfc + 0.5 * xf_width
        ylo, yhi = -10.0, 10.0
    else:
        raise ValueError(mode)
    text = full_card_text(work, output_name=name, pdf_set="NNPDF40_nnlo_as_01180",
                          pdf_member=0, cores=4, calls=calls, seed=seed)
    text = text.replace("makecuts = true", "makecuts = false", 1)
    text = text.replace("ih2          = -1", "ih2          = 1", 1)
    text = text.replace(
        "nproc        = 3",
        f"nproc        = 3\nnuclearpdf   = true\nZ1 = 1\nA1 = 1\nZ2 = {z:.12g}\nA2 = {a:.12g}",
        1,
    )
    text = text.replace("g1 = 0.0", "g1 = 1.017", 1)
    text = re.sub(r"qt_bins = \[ [^\]]+ \]", f"qt_bins = [ {qlo:.12g} {qhi:.12g} ]", text, count=1)
    text = re.sub(r"y_bins  = \[ [^\]]+ \]", f"y_bins  = [ {ylo:.12g} {yhi:.12g} ]", text, count=1)
    text = re.sub(r"m_bins  = \[ [^\]]+ \]",
                  f"m_bins  = [ {float(row.QM_Low):.12g} {float(row.QM_High):.12g} ]", text, count=1)
    text = text.replace("ptbinwidth = true", "ptbinwidth = false", 1)
    text = text.replace("ybinwidth = true", "ybinwidth = false", 1)
    text = text.replace("mbinwidth = true", "mbinwidth = false", 1)
    text += "\n# Native differential observable conversion audit.\n"
    text += "dsdqt2 = true\n"
    text += f"dsdxf = {'true' if mode == 'fixed_xf' else 'false'}\nedsdp3 = false\n"
    if mode == "fixed_xf":
        # Replace the defaults already present in the common card. DYTurbo's
        # parser keeps the first occurrence, so appending a second pair would
        # silently leave the unrestricted cuts active.
        if "xfmin" in text:
            text = text.replace("xfmin = -1e6", f"xfmin = {xflo:.12g}", 1)
        else:
            text += f"xfmin = {xflo:.12g}\n"
        if "xfmax" in text:
            text = text.replace("xfmax = +1e6", f"xfmax = {xfhi:.12g}", 1)
        else:
            text += f"xfmax = {xfhi:.12g}\n"
    return text, {"qT_bin": [qlo, qhi], "qT2_width": qhi * qhi - qlo * qlo,
                  "y_bin": [ylo, yhi], "xF_bin": None if xflo is None else [xflo, xfhi],
                  "xF_width": None if xflo is None else xfhi - xflo}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=20260896)
    ap.add_argument("--qT-width", type=float, default=0.01)
    ap.add_argument("--xf-width", type=float, default=0.01,
                    help="local x_F width for the E605 fixed-x_F point")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    load_runner()
    OUT.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        p for p in ("/home/dustin/miniforge3/envs/pdf-fit/lib", str(DYROOT / "lib"), env.get("LD_LIBRARY_PATH", "")) if p
    )
    reps = [
        ("E288_200", "E288_200:0", "fixed_y", 4.0, 9.0121831),
        ("E605", "E605:0", "fixed_xf", 29.0, 63.546),
        ("E772", "E772:0", "fixed_xf", 0.5, 1.0),
    ]
    rows = []
    for serial, (dataset, row_id, mode, z, a) in enumerate(reps):
        frame = pd.read_csv(DATA_ROOT / f"{dataset}.csv")
        hit = frame[frame.row_id.eq(row_id)]
        if len(hit) != 1:
            raise RuntimeError(row_id)
        row = hit.iloc[0].copy()
        name = f"{dataset}_normalized_{mode}"
        card = OUT / f"{name}.in"
        log = OUT / f"{name}.log"
        table = DYROOT / f"{name}.txt"
        text, bins = card_for(row, name=name, mode=mode, z=z, a=a,
                              q_width=float(args.qT_width), xf_width=float(args.xf_width),
                              calls=int(args.calls), seed=int(args.seed) + serial)
        card.write_text(text)
        if args.force:
            table.unlink(missing_ok=True)
        proc = None
        if not table.exists():
            with log.open("w") as handle:
                proc = subprocess.run([str(DYTURBO), str(card)], cwd=DYROOT, env=env,
                                      stdout=handle, stderr=subprocess.STDOUT,
                                      check=False, timeout=3600)
        value, unc = parse_first(table)
        dq2 = bins["qT2_width"]
        dy = bins["y_bin"][1] - bins["y_bin"][0]
        data_a = float(row.A)
        data_cs = float(row.CS)
        pref = float(row.PreFactor)
        if mode == "fixed_y":
            # Eq. (3.3): invariant A = 1/pi integral dQ^2 d qT^2 d y.
            pred_a = (value / 1000.0) / (a * math.pi * dq2 * dy)
            pred_cs = pred_a / pref
            formula = "A_pred=I/(1000*A*pi*Delta(qT^2)*Delta(y)); CS_pred=A_pred/PreFactor"
        else:
            dxf = float(bins["xF_width"])
            # dsdxf=true already supplies dxF/dy.  Convert dqT^2 to dqT.
            pred_cs = (value / 1000.0) / a * (2.0 * float(row.qT)) / (dq2 * dxf)
            pred_a = pref * pred_cs
            formula = "CS_pred=I/(1000*A)*2*qT/(Delta(qT^2)*Delta(xF)); A_pred=PreFactor*CS_pred"
        record = {
            "dataset": dataset, "row_id": row_id, "mode": mode,
            "target_Z": z, "target_A": a, "qT_GeV": float(row.qT),
            "QM_bin_GeV": [float(row.QM_Low), float(row.QM_High)], **bins,
            "raw_integrated_fb_per_nucleus": value, "raw_unc_fb_per_nucleus": unc,
            "predicted_A_invariant": pred_a, "data_A": data_a,
            "predicted_CS_fit_ready": pred_cs, "data_CS_fit_ready": data_cs,
            "ratio_A": pred_a / data_a, "ratio_CS": pred_cs / data_cs,
            "formula": formula, "card": str(card), "log": str(log), "table": str(table),
            "return_code": None if proc is None else proc.returncode,
        }
        rows.append(record)
        pd.DataFrame(rows).to_csv(OUT / "probe_rows.csv", index=False)
        print(json.dumps(record), flush=True)
    result = {
        "status": "fixed_target_normalized_observable_probe_complete_not_production",
        "formula": "W_N3LL + (FO_NNLO - ASY_NNLO)", "engine": str(DYTURBO),
        "order": {"resummation": "N3LL_unprimed", "fixed_order": "NNLO_VJ", "primed": False},
        "calls_per_vegas_component": int(args.calls), "rows": rows,
        "interpretation": "native DYTurbo fixed-target observable conversion audit; no 353-row promotion",
        "frozen_baseline_unchanged": True, "production_outputs_modified": False,
        "promotion_authorized": False,
    }
    (OUT / "probe_status.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
