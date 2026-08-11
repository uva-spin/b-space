#!/usr/bin/env python3
"""Run DYTurbo fixed-order V+jet benchmarks for processed LHCb 7 TeV rows."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path

import pandas as pd


DEFAULT_DYTURBO = Path("/home/dustin/src/dyturbo-1.4.2/bin/dyturbo")
DEFAULT_DYTURBO_ROOT = Path("/home/dustin/src/dyturbo-1.4.2")
DEFAULT_CONDA_LIB = Path("/home/dustin/miniforge3/envs/pdf-fit/lib")


def card_text(row: pd.Series, *, output_name: str, pdf_set: str, pdf_member: int, cores: int,
              mu_r_factor: float = 1.0, mu_f_factor: float = 1.0, seed: int = 123456) -> str:
    return f"""# LHCb 7 TeV DYTurbo fixed-order V+jet benchmark
# Row: {row.row_id}

sroot        = {float(row.SqrtS):.12g}
ih1          = 1
ih2          = 1
nproc        = 3

fixedorder_only = true
order           = 1
primed          = true
qed             = false
qedorder        = 0

npff = 0
g1 = 0.0
g2 = 0.0
g3 = 0.0
Q0 = 1

LHAPDFset    = {pdf_set}
LHAPDFmember = {pdf_member}

fmuren = 1
fmufac = 1
fmures = 1
kmuren = {mu_r_factor:.12g}
kmufac = {mu_f_factor:.12g}
kmures = 1.0

ewscheme = 1
Gf    = 1.1663787e-5
zmass = 91.1876
wmass = 80.385
xw    = 0.23153
aemmz = 7.7585538055706e-03
zwidth = 2.4950
wwidth = 2.091
runningwidth = false
conv2fixw = false
useGamma = true

xqtcut = 0.008
qtcut = 0.0
rseed = {seed}
mcutoff = 1e-3

doBORN = false
doCT   = false
doVJ   = true
doFPC  = false
doVJREAL = false
doVJVIRT = false
VJquad = true

intDimRes  = -1
intDimBorn = -1
intDimCT   = -1
intDimVJ   = 3
intDimFPC  = -1

threading = 0
cores     = {cores}
cubaverbosity = 0
cubanbatch    = 1000
niterVJ       = 10
vegasncallsBORN   = 1000
vegasncallsCT     = 100000
vegasncallsVJLO   = 10000000
vegasncallsVJREAL = 100000
vegasncallsVJVIRT = 100000
vegascollect      = true
vegascorr         = false

pcubature   = true
relaccuracy = 1e-3
absaccuracy = 0
level = 3

makecuts = true
lptcut = 0
lycut  = 1000
lepptcut = 0
lepycut  = 1000
alpptcut = 0
alpycut  = 1000
lcptcut = 20
lcymin = 2.0
lcymax = 4.5
lfptcut = 20
lfymin = 2.0
lfymax = 4.5
cthCSmin = -1
cthCSmax = +1

output_filename = {output_name}
texttable   = true
redirect    = false
unicode     = false
silent      = false
makehistos  = false
gridverbose = false

force_binsampling = true
ptbinwidth = false
ybinwidth = false
mbinwidth = false

qt_bins = [ {float(row.qT_low):.12g} {float(row.qT_high):.12g} ]
y_bins  = [ {float(row.y_Low):.12g} {float(row.y_High):.12g} ]
m_bins  = [ {float(row.QM_Low):.12g} {float(row.QM_High):.12g} ]

hqt   = false
hy    = false
hm    = false
hmt   = false
haiqt = false
haiy  = false
haim  = false
"""


def parse_first_value(path: Path) -> tuple[float, float]:
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) >= 2:
            return float(parts[0]), float(parts[1])
    raise ValueError(f"Could not parse DYTurbo output: {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="Data/v23a_tevatron_plus_lhcb7_fiducial_candidate/LHCb_7.csv")
    ap.add_argument("--rows", nargs="+", default=["LHCb_7:6", "LHCb_7:8", "LHCb_7:9", "LHCb_7:11"])
    ap.add_argument("--out", default="systematics/finite_y_tail_benchmark/outputs/lhcb7_dyturbo")
    ap.add_argument("--dyturbo", default=str(DEFAULT_DYTURBO))
    ap.add_argument("--dyturbo-root", default=str(DEFAULT_DYTURBO_ROOT))
    ap.add_argument("--pdf-set", default="NNPDF40_nnlo_as_01180")
    ap.add_argument("--pdf-member", type=int, default=0)
    ap.add_argument("--mu-r-factor", type=float, default=1.0)
    ap.add_argument("--mu-f-factor", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=123456)
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=600)
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

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        p for p in [str(DEFAULT_CONDA_LIB), str(Path(args.dyturbo_root) / "lib"), env.get("LD_LIBRARY_PATH", "")] if p
    )

    rows = []
    for _, row in selected.iterrows():
        tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(row.row_id))
        output_name = f"{tag}_dyturbo_vj_nlo_fid"
        card = (cards / f"{output_name}.in").resolve()
        card.write_text(card_text(row, output_name=output_name, pdf_set=args.pdf_set,
                                 pdf_member=args.pdf_member, cores=args.cores,
                                 mu_r_factor=args.mu_r_factor, mu_f_factor=args.mu_f_factor,
                                 seed=args.seed))
        log = logs / f"{output_name}.log"
        txt_src = Path(args.dyturbo_root) / f"{output_name}.txt"
        dat_src = Path(args.dyturbo_root) / f"{output_name}.dat"
        for p in (txt_src, dat_src):
            if p.exists():
                p.unlink()
        with log.open("w") as fh:
            subprocess.run(
                [str(Path(args.dyturbo).resolve()), str(card)],
                cwd=args.dyturbo_root,
                env=env,
                stdout=fh,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=args.timeout,
            )
        txt_dst = tables / txt_src.name
        dat_dst = tables / dat_src.name
        if txt_src.exists():
            txt_src.replace(txt_dst)
        if dat_src.exists():
            dat_src.replace(dat_dst)
        value, unc = parse_first_value(txt_dst)
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
                "dyturbo_raw": value,
                "dyturbo_raw_unc": unc,
                "dyturbo_raw_per_GeV": value / bin_width,
                "dyturbo_raw_unc_per_GeV": unc / bin_width,
                "card": str(card),
                "log": str(log),
                "txt": str(txt_dst),
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(out / "dyturbo_benchmark_summary.csv", index=False)
    print(result.to_string(index=False))
    print(f"wrote {out / 'dyturbo_benchmark_summary.csv'}")


if __name__ == "__main__":
    main()
