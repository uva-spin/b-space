#!/usr/bin/env python3
"""Isolated MCFM check with an explicit boson-qT (pt34) bin cut.

The earlier LHCb MCFM cards used ``basicjets ptjetmin/max`` as if it were a
boson-qT bin.  This check leaves the jet acceptance broad and puts the
requested qT range in ``masscuts pt34min/max``, which is the MCFM cut on the
Z/gamma* transverse momentum.  It is diagnostic only and never touches the
older benchmark outputs.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import re
import shutil
import subprocess

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "Data/v23a_tevatron_plus_lhcb7_fiducial_candidate/LHCb_7.csv"
RUNNER = ROOT / "systematics/finite_y_tail_benchmark/scripts/run_lhcb7_mcfm_benchmark.py"
DEFAULT_OUT = ROOT / "systematics/finite_y_completion_2026/reports/lhcb7_external_mcfm_bosonqt"
MCFM_BIN = Path("/home/dustin/work/MCFM-10.3/Bin")


def load_runner():
    spec = importlib.util.spec_from_file_location("existing_lhcb_mcfm_runner", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_total(path: Path) -> tuple[float, float]:
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 4 and not line.lstrip().startswith("#"):
            return float(parts[2]), float(parts[3])
    raise RuntimeError(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", default=["LHCb_7:10"])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--calls", type=int, default=200000)
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--no-lepton-cuts", action="store_true",
                    help="diagnostic boson-level comparison while retaining qT/y/m cuts")
    ap.add_argument("--also-jet-cuts", action="store_true",
                    help="retain the basicjets qT-range cuts in addition to pt34 cuts")
    ap.add_argument("--no-resummation", action="store_true",
                    help="disable MCFM's resummation grid for a pure fixed-order comparison")
    args = ap.parse_args()
    runner = load_runner()
    data = pd.read_csv(DATA)
    selected = data[data.row_id.isin(args.rows)].copy()
    if len(selected) != len(args.rows):
        raise SystemExit("requested rows are missing")
    out = Path(args.out)
    cards, logs, tables = out / "cards", out / "logs", out / "tables"
    for path in (cards, logs, tables):
        path.mkdir(parents=True, exist_ok=True)
    rundir = "lhcb7_mcfm_bosonqt_check"
    (MCFM_BIN / rundir).mkdir(exist_ok=True)
    records = []
    for _, row in selected.iterrows():
        tag = re.sub(r"[^A-Za-z0-9]+", "", str(row.row_id).split(":")[-1])
        runstring = f"lhcbqt{tag}"
        text = runner.card_text(row, runstring=runstring, rundir=rundir,
                                pdf_set="NNPDF40_nnlo_as_01180", calls=args.calls,
                                pdf_member=0, seed=0)
        if args.no_resummation:
            text = text.replace("    usegrid = .true.", "    usegrid = .false.")
        if not args.also_jet_cuts:
            text = text.replace(f"ptjetmin = {float(row.qT_low):.12g}", "ptjetmin = 0.0")
            text = text.replace(f"ptjetmax = {float(row.qT_high):.12g}", "ptjetmax = 900.0")
        text = text.replace("    m3456min = 0\n", f"    m3456min = 0\n    pt34min = {float(row.qT_low):.12g}\n    pt34max = {float(row.qT_high):.12g}\n")
        text = text.replace("    makecuts = .true.\n", f"    makecuts = .true.\n    y34min = {float(row.y_Low):.12g}\n    y34max = {float(row.y_High):.12g}\n")
        if args.no_lepton_cuts:
            text = text.replace("    ptleptmin = 20.0", "    ptleptmin = 0.0")
            text = text.replace("    ptlept2min = 20.0", "    ptlept2min = 0.0")
            text = text.replace("    etaleptmin = 2.0", "    etaleptmin = -1000.0")
            text = text.replace("    etaleptmax = 4.5", "    etaleptmax = 1000.0")
            text = text.replace("    etalept2min = 2.0", "    etalept2min = -1000.0")
            text = text.replace("    etalept2max = 4.5", "    etalept2max = 1000.0")
        card = cards / f"{runstring}.ini"
        card.write_text(text)
        work_card = MCFM_BIN / card.name
        shutil.copy2(card, work_card)
        for old in (MCFM_BIN / rundir).glob(f"*{runstring}*"):
            old.unlink()
        log = logs / f"{runstring}.log"
        env = {"PATH": "/usr/bin:/bin", "HOME": os.environ.get("HOME", "/home/dustin"),
               "LHAPDF_DATA_PATH": str((MCFM_BIN / "PDFs").resolve()), "OMP_NUM_THREADS": "1"}
        with log.open("w") as handle:
            subprocess.run(["/bin/bash", "-lc", f"ulimit -s unlimited; exec {MCFM_BIN / 'mcfm'} {work_card.name}"],
                           cwd=MCFM_BIN, env=env, stdout=handle, stderr=subprocess.STDOUT,
                           check=True, timeout=args.timeout)
        matches = sorted((MCFM_BIN / rundir).glob(f"*{runstring}_total_cross.txt"))
        if not matches:
            raise RuntimeError(f"missing total cross for {runstring}")
        table = tables / matches[-1].name
        shutil.copy2(matches[-1], table)
        val, unc = parse_total(table)
        width = float(row.qT_high - row.qT_low)
        # MCFM histogram text tables are fb/bin even when the log's final
        # integral is printed in pb.  Convert explicitly before comparing to
        # the pb/GeV data table.
        records.append({"row_id": row.row_id, "qT_low": row.qT_low, "qT_high": row.qT_high,
                        "data_pb_per_GeV": row.CS, "mcfm_raw_fb_bin": val,
                        "mcfm_raw_fb_bin_unc": unc, "mcfm_pb_bin": val / 1000.0,
                        "mcfm_pb_per_GeV": val / (1000.0 * width), "mcfm_pb_per_GeV_unc": unc / (1000.0 * width),
                        "card": str(card), "log": str(log), "table": str(table),
                        "explicit_pt34_cut": True})
        work_card.unlink(missing_ok=True)
    result = pd.DataFrame(records)
    result.to_csv(out / "mcfm_bosonqt_summary.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
