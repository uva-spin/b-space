#!/usr/bin/env python3
"""Second, isolated MCFM NLO closure attempt for the LHCb boundary bins.

The first probe used the generic MCFM defaults and never got through the very
large NLO-real warmup.  This retry follows the settings in MCFM's own
input_Zjet.ini (larger real-emission warmup, relaxed warmup chi2 criterion,
and intermediate snapshots).  It is a diagnostic only: no MCFM source,
benchmark output, or frozen production input is changed.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "Data/v23a_tevatron_plus_lhcb7_fiducial_candidate/LHCb_7.csv"
MCFM_RUNNER = ROOT / "systematics/finite_y_tail_benchmark/scripts/run_lhcb7_mcfm_benchmark.py"
MCFM_BIN = Path("/home/dustin/work/MCFM-10.3/Bin")
MCFM_EXE = MCFM_BIN / "mcfm"
LHAPDF_DATA = MCFM_BIN / "PDFs"
OUT = ROOT / "systematics/finite_y_completion_2026/reports/lhcb7_external_mcfm_true_nlo_retry"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys_modules = __import__("sys").modules
    sys_modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_log(path: Path):
    matches = re.findall(
        r"Value of integral is\s+([-+0-9.Ee]+)\s*±\s*([-+0-9.Ee]+)\s+([fp]b)",
        path.read_text(errors="replace"),
    )
    if not matches:
        raise RuntimeError(f"no final MCFM result in {path}")
    return matches[-1]


def tune_card(card: str, *, calls: int, real_calls: int, warmup_chi2: float,
              warmup_precision: float, writeintermediate: bool, readin: bool) -> str:
    card = card.replace("part = lo", "part = nlo")
    replacements = {
        r"initcallslord\s*=\s*\d+": f"initcallslord = {calls}",
        r"initcallsnloreal\s*=\s*\d+": f"initcallsnloreal = {real_calls}",
        r"initcallsnlovirt\s*=\s*\d+": "initcallsnlovirt = 50000",
        r"initcallsnnlobelow\s*=\s*\d+": "initcallsnnlobelow = 1000000",
        r"initcallsnnlorealabove\s*=\s*\d+": "initcallsnnlorealabove = 10000000",
        r"precisiongoal\s*=\s*[0-9.eE+-]+": "precisiongoal = 0.003",
        r"warmupprecisiongoal\s*=\s*[0-9.eE+-]+": f"warmupprecisiongoal = {warmup_precision}",
        r"warmupchisqgoal\s*=\s*[0-9.eE+-]+": f"warmupchisqgoal = {warmup_chi2}",
        r"writeintermediate\s*=\s*\.\w+": f"writeintermediate = {str(writeintermediate).lower()}",
        r"readin\s*=\s*\.\w+": f"readin = {str(readin).lower()}",
    }
    for pattern, replacement in replacements.items():
        card, n = re.subn(pattern, replacement, card, count=1)
        if n != 1:
            raise RuntimeError(f"could not replace {pattern}")
    # These are present in input_Zjet.ini and make the warmup map less noisy.
    card += "\n    ndmx = 40\n    iterbatch1 = 10\n    iterbatch2 = 3\n    iterbatchwarmup = 10\n    callboost = 10.0\n"
    return card


def run_one(row: pd.Series, runner, *, calls: int, real_calls: int, timeout: int,
            workdir_name: str, readin: bool) -> dict:
    cards, logs = OUT / "cards", OUT / "logs"
    cards.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    tag = str(row.row_id).replace(":", "_")
    runstring = f"lhcb_nlo_retry_{tag}"
    card_text = runner.card_text(
        row, runstring=runstring, rundir=workdir_name,
        pdf_set="NNPDF40_nnlo_as_01180", calls=calls,
        pdf_member=0, mu_r_factor=1.0, mu_f_factor=1.0, seed=17,
    )
    card_text = tune_card(card_text, calls=calls, real_calls=real_calls,
                          warmup_chi2=12.5, warmup_precision=0.25,
                          writeintermediate=True, readin=readin)
    card = (cards / f"{runstring}.ini").resolve()
    card.write_text(card_text)
    work_card = MCFM_BIN / card.name
    shutil.copy2(card, work_card)
    rundir = MCFM_BIN / workdir_name
    rundir.mkdir(exist_ok=True)
    # Keep snapshots from this run, but remove stale files for the same tag.
    for old in rundir.glob(f"*{runstring}*"):
        old.unlink()
    log = logs / f"{runstring}.log"
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": os.environ.get("HOME", "/home/dustin"),
        "LHAPDF_DATA_PATH": str(LHAPDF_DATA.resolve()),
        "OMP_NUM_THREADS": "1",
    }
    with log.open("w") as handle:
        subprocess.run(
            ["/bin/bash", "-lc", f"ulimit -s unlimited; exec {shlex.quote(str(MCFM_EXE))} {shlex.quote(work_card.name)}"],
            cwd=MCFM_BIN, env=env, stdout=handle, stderr=subprocess.STDOUT,
            check=True, timeout=timeout,
        )
    value, unc, unit = parse_log(log)
    factor = 1.0 if unit == "pb" else 1.0 / 1000.0
    width = float(row.qT_high - row.qT_low)
    result = {
        "row_id": row.row_id,
        "data_pb_per_GeV": float(row.CS),
        "mcfm_raw_abs_y_pb_per_GeV": float(value) * factor / width,
        "mcfm_raw_abs_y_unc_pb_per_GeV": float(unc) * factor / width,
        "mcfm_single_positive_arm_pb_per_GeV": 0.5 * float(value) * factor / width,
        "mcfm_single_arm_over_data": 0.5 * float(value) * factor / width / float(row.CS),
        "unit": unit, "log": str(log), "card": str(card),
    }
    (OUT / "retry_result.json").write_text(__import__("json").dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", default="LHCb_7:10")
    ap.add_argument("--calls", type=int, default=50000)
    ap.add_argument("--real-calls", type=int, default=5000000)
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--readin", action="store_true")
    args = ap.parse_args()
    runner = load_module("mcfm_lhcb_retry_runner", MCFM_RUNNER)
    data = pd.read_csv(DATA)
    selected = data[data.row_id.eq(args.row)]
    if len(selected) != 1:
        raise RuntimeError(f"missing or duplicate row {args.row}")
    result = run_one(selected.iloc[0], runner, calls=args.calls, real_calls=args.real_calls,
                     timeout=args.timeout, workdir_name="lhcb7_mcfm_true_nlo_retry",
                     readin=args.readin)
    print(result)


if __name__ == "__main__":
    main()
