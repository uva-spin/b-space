#!/usr/bin/env python3
"""Probe DYTurbo's complete unprimed N3LL+NNLO W+Y observable.

Unlike the fixed-order boundary oracle, this enables the resummed Born and
counterterm pieces as well as NNLO V+jet.  It is a candidate-side one-row
observable probe only; no fitted F_NP or production cache is touched.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


SYSTEMATICS = Path(__file__).resolve().parents[2]
PROJECT = SYSTEMATICS.parent
RUNNER_PATH = PROJECT / "b-space-public/v23/tools/run_tevatron_dyturbo_benchmark.py"
DATA = PROJECT / "Data/v23a_tevatron_plus_lhcb7_fiducial_candidate/CDF_RUN_2.csv"
OUT = SYSTEMATICS / "full_n3ll_wy_production_2026/reports/dyturbo_full_n3ll_nnlo_probe"
DYTURBO = Path("/home/dustin/src/dyturbo-1.4.2/bin/dyturbo")
DYROOT = Path("/home/dustin/src/dyturbo-1.4.2")


def load_runner():
    spec = importlib.util.spec_from_file_location("full_n3ll_probe_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def full_card_text(row: pd.Series, *, output_name: str, pdf_set: str, pdf_member: int, cores: int,
                   calls: int = 1_000_000, seed: int = 246810) -> str:
    runner = sys.modules["full_n3ll_probe_runner"]
    text = runner.card_text(
        row, output_name=output_name, pdf_set=pdf_set, pdf_member=pdf_member,
        cores=cores, mu_r_factor=1.0, mu_f_factor=1.0, seed=seed,
    )
    text = text.replace("fixedorder_only = true", "fixedorder_only = false", 1)
    text = text.replace("order           = 1", "order           = 3", 1)
    text = text.replace("primed          = true", "primed          = false", 1)
    text = text.replace("doBORN = false", "doBORN = true", 1)
    text = text.replace("doCT   = false", "doCT   = true", 1)
    text = text.replace("doVJREAL = false", "doVJREAL = true", 1)
    text = text.replace("doVJVIRT = false", "doVJVIRT = true", 1)
    text = text.replace("VJquad = true", "VJquad = false", 1)
    text = text.replace("intDimVJ   = 3", "intDimVJ   = -1", 1)
    text = text.replace("makecuts = false", "makecuts = true", 1)
    text = text.replace("vegasncallsBORN   = 1000", f"vegasncallsBORN   = {calls}", 1)
    text = text.replace("vegasncallsCT     = 100000", f"vegasncallsCT     = {calls}", 1)
    text = text.replace("vegasncallsVJREAL = 100000", f"vegasncallsVJREAL = {calls}", 1)
    text = text.replace("vegasncallsVJVIRT = 100000", f"vegasncallsVJVIRT = {calls}", 1)
    return text


def main() -> None:
    runner = load_runner()
    data = pd.read_csv(DATA)
    row = data[data["row_id"].eq("CDF_RUN_2:17")].iloc[0]
    OUT.mkdir(parents=True, exist_ok=True)
    card_path = OUT / "CDF_RUN_2_17_full_n3ll_nnlo.in"
    log_path = OUT / "CDF_RUN_2_17_full_n3ll_nnlo.log"
    table_path = DYROOT / "CDF_RUN_2_17_full_n3ll_nnlo.txt"
    output_name = "CDF_RUN_2_17_full_n3ll_nnlo"
    card_path.write_text(full_card_text(row, output_name=output_name, pdf_set="NNPDF40_nnlo_as_01180", pdf_member=0, cores=8))
    if table_path.exists():
        table_path.unlink()
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        p for p in ("/home/dustin/miniforge3/envs/pdf-fit/lib", str(DYROOT / "lib"), env.get("LD_LIBRARY_PATH", "")) if p
    )
    with log_path.open("w") as handle:
        subprocess.run([str(DYTURBO), str(card_path)], cwd=DYROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True, timeout=1800)
    if not table_path.exists():
        raise RuntimeError(f"missing DYTurbo output {table_path}")
    value, uncertainty = runner.parse_first_value(table_path)
    width = float(row.qT_high - row.qT_low)
    result = {
        "status": "isolated_full_unprimed_n3ll_nnlo_dyturbo_probe_passed",
        "row_id": str(row.row_id),
        "engine": str(DYTURBO),
        "order": 3,
        "primed": False,
        "fixedorder_only": False,
        "enabled_terms": ["resummation_W", "counterterm_ASY", "VJ_NNLO"],
        "raw_dyturbo_fb_per_bin": value,
        "raw_dyturbo_unc_fb_per_bin": uncertainty,
        "dyturbo_pb_per_GeV": value / width / 1000.0,
        "dyturbo_unc_pb_per_GeV": uncertainty / width / 1000.0,
        "data_pb_per_GeV": float(row.CS),
        "card": str(card_path),
        "log": str(log_path),
        "table": str(table_path),
        "interpretation": "external full W+Y observable probe; no custom fitted F_NP and no production authorization",
        "frozen_baseline_unchanged": True,
        "production_outputs_modified": False,
    }
    (OUT / "probe_status.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
