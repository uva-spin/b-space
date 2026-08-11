#!/usr/bin/env python3
"""Run MCFM LO Z+jet benchmarks for processed LHCb 7 TeV rows."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import pandas as pd


DEFAULT_MCFM_BIN = Path("/home/dustin/work/MCFM-10.3/Bin")
DEFAULT_MCFM_EXE = DEFAULT_MCFM_BIN / "mcfm"
DEFAULT_LHAPDF_DATA = DEFAULT_MCFM_BIN / "PDFs"


def card_text(row: pd.Series, *, runstring: str, rundir: str, pdf_set: str, calls: int,
              pdf_member: int = 0, mu_r_factor: float = 1.0,
              mu_f_factor: float = 1.0, seed: int = 0) -> str:
    return f"""mcfm_version = 10.3
writerefs = .false.

[general]
    nproc = 41
    part = lo
    runstring = {runstring}
    rundir = {rundir}
    sqrts = {float(row.SqrtS):.12g}
    ih1 = +1
    ih2 = +1
    zerowidth = .false.
    removebr = .false.
    ewcorr = none

[nnlo]

[resummation]
    usegrid = .true.
    makegrid = .false.
    gridoutpath = PDFs/
    gridinpath = PDFs/
    res_range = 0.0 270.0
    resexp_range = 1.0 270.0
    fo_cutoff = 1.0
    transitionswitch = 0.4

[pdf]
    pdlabel = 'CT14.NN'

[lhapdf]
    lhapdfset = {pdf_set}
    lhapdfmember = {pdf_member}
    dopdferrors = .false.

[scales]
    dynamicscale = none
    renscale = {float(row.QM) * mu_r_factor:.12g}
    facscale = {float(row.QM) * mu_f_factor:.12g}
    doscalevar = .false.
    maxscalevar = 6

[masses]
    hmass = 125
    mt = 173.3
    mb = 4.66
    mc = 1.275

[basicjets]
    inclusive = .true.
    algorithm = ankt
    ptjetmin = {float(row.qT_low):.12g}
    ptjetmax = {float(row.qT_high):.12g}
    etajetmax = 99.0
    Rcutjet = 0.5

[masscuts]
    m34min = {float(row.QM_Low):.12g}
    m34max = {float(row.QM_High):.12g}
    m56min = 0
    m3456min = 0

[cuts]
    makecuts = .true.
    ptleptmin = 20.0
    etaleptmin = 2.0
    etaleptmax = 4.5
    etaleptveto = 0.0 0.0
    ptminmiss = 0.0
    ptlept2min = 20.0
    etalept2min = 2.0
    etalept2max = 4.5
    etalept2veto = 0.0 0.0
    m34transmin = 0.0
    Rjlmin = 0.0
    Rllmin = 0.0
    delyjjmin = 0.0
    jetsopphem = .false.
    lbjscheme = 0
    ptbjetmin = 0.0
    etabjetmax = 99.0

[photon]
    fragmentation = .false.
    fragmentation_set = GdRG__LO
    fragmentation_scale = 1.0
    gammptmin = 40
    gammrapmax = 2.5
    gammpt2 = 25
    gammpt3 = 25
    Rgalmin = 0
    Rgagamin = 0.4
    Rgajetmin = 0
    cone_ang = 0.4
    epsilon_h = 0.5
    n_pow = 1

[histogram]
    writetxt = .true.
    newstyle = .true.

[integration]
    initcallslord = {calls}
    initcallsnloreal=1000000
    initcallsnlovirt=200000
    initcallsnnlobelow=200000
    initcallsnnlovirtabove=400000
    initcallsnnlorealabove=2000000
    initcallsnloresummed=1000
    initcallsnloresabove=200000
    usesobol = .true.
    seed = {seed}
    precisiongoal = 0.003
    readin = .false.
    writeintermediate = .false.
    warmupprecisiongoal = 0.25
    warmupchisqgoal = 3.0
"""


def parse_total_cross(path: Path) -> tuple[float, float]:
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) >= 4:
            return float(parts[2]), float(parts[3])
    raise ValueError(f"Could not parse MCFM total cross table: {path}")


def parse_log_result(log: Path) -> tuple[float, float, str]:
    text = log.read_text(errors="replace")
    matches = re.findall(
        r"Value of integral is\s+([-+0-9.Ee]+)\s*±\s*([-+0-9.Ee]+)\s+([fp]b)",
        text,
    )
    if not matches or "=== Result for PDF set" not in text:
        raise ValueError(f"Could not parse final MCFM result from {log}")
    value, unc, unit = matches[-1]
    return float(value), float(unc), unit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="Data/v23a_tevatron_plus_lhcb7_fiducial_candidate/LHCb_7.csv")
    ap.add_argument("--rows", nargs="+", default=["LHCb_7:6", "LHCb_7:8", "LHCb_7:9", "LHCb_7:11"])
    ap.add_argument("--out", default="systematics/finite_y_tail_benchmark/outputs/lhcb7_mcfm")
    ap.add_argument("--mcfm-bin", default=str(DEFAULT_MCFM_BIN))
    ap.add_argument("--mcfm-exe", default=str(DEFAULT_MCFM_EXE))
    ap.add_argument("--pdf-set", default="NNPDF40_nnlo_as_01180")
    ap.add_argument("--pdf-member", type=int, default=0)
    ap.add_argument("--mu-r-factor", type=float, default=1.0)
    ap.add_argument("--mu-f-factor", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lhapdf-data", default=str(DEFAULT_LHAPDF_DATA))
    ap.add_argument("--calls", type=int, default=200000)
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    data = pd.read_csv(args.data)
    selected = data[data["row_id"].isin(args.rows)].copy()
    missing = sorted(set(args.rows) - set(selected["row_id"]))
    if missing:
        raise SystemExit(f"Missing requested rows: {missing}")

    out = Path(args.out)
    cards = out / "cards"
    logs = out / "logs"
    tables = out / "tables"
    for d in (cards, logs, tables):
        d.mkdir(parents=True, exist_ok=True)

    mcfm_bin = Path(args.mcfm_bin).resolve()
    mcfm_exe = Path(args.mcfm_exe).resolve()
    rundir = "lhcb7_mcfm_benchmark"
    (mcfm_bin / rundir).mkdir(exist_ok=True)

    rows = []
    for _, row in selected.iterrows():
        tag = re.sub(r"[^A-Za-z0-9]+", "", str(row.row_id).split(":")[-1])
        runstring = f"lhcb{tag}"
        card = (cards / f"{runstring}.ini").resolve()
        card.write_text(card_text(row, runstring=runstring, rundir=rundir, pdf_set=args.pdf_set,
                                 calls=args.calls, pdf_member=args.pdf_member,
                                 mu_r_factor=args.mu_r_factor, mu_f_factor=args.mu_f_factor,
                                 seed=args.seed))
        work_card = mcfm_bin / card.name
        shutil.copy2(card, work_card)

        for old in (mcfm_bin / rundir).glob(f"*{runstring}*"):
            old.unlink()

        log = logs / f"{runstring}.log"
        env = {
            "PATH": "/usr/bin:/bin",
            "HOME": os.environ.get("HOME", "/home/dustin"),
            "LHAPDF_DATA_PATH": str(Path(args.lhapdf_data).resolve()),
            "OMP_NUM_THREADS": "1",
        }
        with log.open("w") as fh:
            subprocess.run(
                ["/bin/bash", "-lc", f"ulimit -s unlimited; exec {shlex.quote(str(mcfm_exe))} {shlex.quote(work_card.name)}"],
                cwd=mcfm_bin,
                env=env,
                stdout=fh,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=args.timeout,
            )

        matches = sorted((mcfm_bin / rundir).glob(f"*{runstring}_total_cross.txt"))
        if not matches:
            raise FileNotFoundError(f"No MCFM total_cross output for {runstring}; see {log}")
        table_src = matches[-1]
        table_dst = tables / table_src.name
        shutil.copy2(table_src, table_dst)
        value_table, unc_table = parse_total_cross(table_dst)
        value_raw, unc_raw, unit = parse_log_result(log)
        unit_to_pb = 1.0 if unit == "pb" else 1.0 / 1000.0
        bin_width = float(row.qT_high) - float(row.qT_low)
        rows.append(
            {
                "dataset": row.dataset,
                "row_id": row.row_id,
                "qT": float(row.qT),
                "qT_low": float(row.qT_low),
                "qT_high": float(row.qT_high),
                "qT_over_Q": float(row.qT_over_Q),
                "QM_Low": float(row.QM_Low),
                "QM_High": float(row.QM_High),
                "y_Low": float(row.y_Low),
                "y_High": float(row.y_High),
                "data_pb_per_GeV": float(row.CS),
                "data_bin_pb": float(row.CS) * bin_width,
                "mcfm_table_bin": value_table,
                "mcfm_table_bin_unc": unc_table,
                "mcfm_log_bin": value_raw,
                "mcfm_log_bin_unc": unc_raw,
                "mcfm_log_unit": unit,
                "mcfm_pb_bin": value_raw * unit_to_pb,
                "mcfm_pb_bin_unc": unc_raw * unit_to_pb,
                "mcfm_pb_per_GeV": value_raw * unit_to_pb / bin_width,
                "mcfm_pb_per_GeV_unc": unc_raw * unit_to_pb / bin_width,
                "card": str(card),
                "log": str(log),
                "txt": str(table_dst),
            }
        )
        try:
            work_card.unlink()
        except FileNotFoundError:
            pass

    result = pd.DataFrame(rows)
    result.to_csv(out / "mcfm_benchmark_summary.csv", index=False)
    print(result.to_string(index=False))
    print(f"wrote {out / 'mcfm_benchmark_summary.csv'}")


if __name__ == "__main__":
    main()
