#!/usr/bin/env python3
"""Isolated pilot for DYTurbo's low-dimensional nonperturbative factor.

This does not select a production NP model.  It measures how the genuine
external N3LL+NNLO W+Y oracle responds to the built-in Gaussian ``npff=0``
parameter on one low-qT and one transition-bin Tevatron point.  The pilot is
used only to decide whether an external-engine candidate fit is practical.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import argparse
from pathlib import Path

import pandas as pd

from run_dyturbo_full_n3ll_nnlo_probe import DYTURBO, DYROOT, full_card_text, load_runner


SYSTEMATICS = Path(__file__).resolve().parents[2]
PROJECT = SYSTEMATICS.parent
DATA = PROJECT / "Data/v23a_tevatron_plus_lhcb7_fiducial_candidate/CDF_RUN_2.csv"
OUT = SYSTEMATICS / "full_n3ll_wy_production_2026/reports/dyturbo_npff_pilot"
ROWS = ("CDF_RUN_2:0", "CDF_RUN_2:17")
G1_VALUES = (0.0, 0.1, 0.2, 0.3, 0.5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g1-values", nargs="+", type=float, default=list(G1_VALUES))
    parser.add_argument("--output", default="npff_pilot.csv")
    args = parser.parse_args()
    g1_values = tuple(args.g1_values)
    runner = load_runner()
    data = pd.read_csv(DATA)
    selected = data[data.row_id.isin(ROWS)].copy()
    OUT.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(p for p in ("/home/dustin/miniforge3/envs/pdf-fit/lib", str(DYROOT / "lib"), env.get("LD_LIBRARY_PATH", "")) if p)
    records = []
    for _, row in selected.iterrows():
        for g1 in g1_values:
            tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{row.row_id}_g1_{g1:g}")
            name = f"{tag}_full_n3ll_nnlo"
            card = OUT / "cards" / f"{name}.in"
            log = OUT / "logs" / f"{name}.log"
            table = DYROOT / f"{name}.txt"
            card.parent.mkdir(parents=True, exist_ok=True)
            log.parent.mkdir(parents=True, exist_ok=True)
            text = full_card_text(row, output_name=name, pdf_set="NNPDF40_nnlo_as_01180", pdf_member=0, cores=8, calls=3_000_000)
            text = text.replace("g1 = 0.0", f"g1 = {g1:.12g}", 1)
            card.write_text(text)
            if table.exists():
                table.unlink()
            with log.open("w") as handle:
                subprocess.run([str(DYTURBO), str(card)], cwd=DYROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True, timeout=1800)
            value, uncertainty = runner.parse_first_value(table)
            width = float(row.qT_high - row.qT_low)
            records.append({
                "row_id": row.row_id,
                "g1_GeV2": g1,
                "full_wy_pb_per_GeV": value / width / 1000.0,
                "full_wy_unc_pb_per_GeV": uncertainty / width / 1000.0,
                "data_pb_per_GeV": float(row.CS),
                "ratio_to_data": value / width / 1000.0 / float(row.CS),
                "card": str(card),
                "log": str(log),
            })
            pd.DataFrame(records).to_csv(OUT / args.output, index=False)
    status = {
        "status": "isolated_dyturbo_npff_gaussian_pilot_complete",
        "model": "DYTurbo npff=0 Gaussian exp[-(g1 + g2 log(Q/Q0) + g3 log(100 x1 x2)) b^2]",
        "rows": list(ROWS),
        "g1_values_GeV2": list(g1_values),
        "interpretation": "response diagnostic only; does not establish a fitted NP model or replace the DNN baseline",
        "production_outputs_modified": False,
    }
    (OUT / (Path(args.output).stem + "_status.json")).write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
