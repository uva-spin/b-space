#!/usr/bin/env python3
"""Isolated MCFM NLO cross-check for the LHCb positive-arm boundary.

MCFM's local user cut is an absolute-rapidity cut, so its result is divided
by two for comparison with the positive-arm LHCb/DYTurbo observable.  This
probe never edits MCFM sources or the benchmark outputs.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
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
OUT = ROOT / "systematics/finite_y_completion_2026/reports/lhcb7_external_mcfm_true_nlo_probe"
ROWS = ("LHCb_7:10", "LHCb_7:11")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_log(path: Path):
    import re
    matches = re.findall(
        r"Value of integral is\s+([-+0-9.Ee]+)\s*±\s*([-+0-9.Ee]+)\s+([fp]b)",
        path.read_text(errors="replace"),
    )
    if not matches:
        raise RuntimeError(f"no final MCFM result in {path}")
    return matches[-1]


def main() -> None:
    runner = load_module("mcfm_lhcb_runner", MCFM_RUNNER)
    data = pd.read_csv(DATA)
    selected = data[data.row_id.isin(ROWS)].copy()
    out = OUT
    cards, logs = out / "cards", out / "logs"
    cards.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    workdir_name = "lhcb7_mcfm_true_nlo_probe"
    (MCFM_BIN / workdir_name).mkdir(exist_ok=True)
    records = []
    for _, row in selected.iterrows():
        tag = str(row.row_id).replace(":", "_")
        runstring = f"lhcb_nlo_probe_{tag}"
        card_text = runner.card_text(
            row, runstring=runstring, rundir=workdir_name,
            pdf_set="NNPDF40_nnlo_as_01180", calls=100000,
            pdf_member=0, mu_r_factor=1.0, mu_f_factor=1.0, seed=17,
        ).replace("part = lo", "part = nlo")
        card = (cards / f"{runstring}.ini").resolve()
        card.write_text(card_text)
        work_card = MCFM_BIN / card.name
        shutil.copy2(card, work_card)
        for old in (MCFM_BIN / workdir_name).glob(f"*{runstring}*"):
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
                check=True, timeout=3600,
            )
        value, unc, unit = parse_log(log)
        factor = 1.0 if unit == "pb" else 1.0 / 1000.0
        width = float(row.qT_high - row.qT_low)
        records.append({
            "row_id": row.row_id,
            "data_pb_per_GeV": row.CS,
            "mcfm_raw_abs_y_pb_per_GeV": float(value) * factor / width,
            "mcfm_raw_abs_y_unc_pb_per_GeV": float(unc) * factor / width,
            "mcfm_single_positive_arm_pb_per_GeV": 0.5 * float(value) * factor / width,
            "mcfm_single_arm_over_data": 0.5 * float(value) * factor / width / float(row.CS),
            "unit": unit,
            "log": str(log),
            "card": str(card),
        })
    result = pd.DataFrame(records)
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "mcfm_true_nlo_probe.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
