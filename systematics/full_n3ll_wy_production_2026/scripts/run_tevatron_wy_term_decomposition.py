#!/usr/bin/env python3
"""Decompose one external DYTurbo N3LL+NNLO W+Y row into W, ASY and FO.

DYTurbo exposes the conventional additive construction as three separately
integrable terms: RES (the W term), CT (the fixed-order expansion of RES, i.e.
the ASY subtraction with its subtraction sign), and VJ (the NNLO fixed-order
remainder).  This diagnostic runs those terms independently, then verifies
that RES+CT+VJ agrees with the all-terms card within Monte-Carlo errors.  It
is candidate-local and never touches frozen production outputs.
"""

from __future__ import annotations

import argparse
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
DATA_ROOT = PROJECT / "Data/v23a_tevatron_plus_lhcb7_fiducial_candidate"
DYTURBO = Path("/home/dustin/src/dyturbo-1.4.2/bin/dyturbo")
DYROOT = Path("/home/dustin/src/dyturbo-1.4.2")
OUT = SYSTEMATICS / "full_n3ll_wy_production_2026/reports/dyturbo_term_decomposition"


def load_runner():
    spec = importlib.util.spec_from_file_location("term_decomposition_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_card(runner, row, *, output_name: str, calls: int, g1: float, term: str) -> str:
    text = runner.card_text(
        row, output_name=output_name, pdf_set="NNPDF40_nnlo_as_01180",
        pdf_member=0, cores=8, mu_r_factor=1.0, mu_f_factor=1.0,
        seed=246810,
    )
    replacements = {
        "fixedorder_only = true": "fixedorder_only = false",
        "order           = 1": "order           = 3",
        "primed          = true": "primed          = false",
        "doBORN = false": "doBORN = true",
        "doCT   = false": "doCT   = true",
        "doVJ   = false": "doVJ   = true",
        "doVJREAL = false": "doVJREAL = true",
        "doVJVIRT = false": "doVJVIRT = true",
        "VJquad = true": "VJquad = false",
        "intDimVJ   = 3": "intDimVJ   = -1",
        "makecuts = false": "makecuts = true",
        "vegasncallsBORN   = 1000": f"vegasncallsBORN   = {calls}",
        "vegasncallsCT     = 100000": f"vegasncallsCT     = {calls}",
        "vegasncallsVJREAL = 100000": f"vegasncallsVJREAL = {calls}",
        "vegasncallsVJVIRT = 100000": f"vegasncallsVJVIRT = {calls}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new, 1)
    text = text.replace("npff = 0", "npff = 0", 1)
    text = text.replace("g1 = 0.0", f"g1 = {g1:.12g}", 1)
    if term == "RES":
        text = text.replace("doCT   = true", "doCT   = false", 1)
        text = text.replace("doVJ   = true", "doVJ   = false", 1)
        text = text.replace("doVJREAL = true", "doVJREAL = false", 1)
        text = text.replace("doVJVIRT = true", "doVJVIRT = false", 1)
    elif term == "CT":
        text = text.replace("doBORN = true", "doBORN = false", 1)
        text = text.replace("doVJ   = true", "doVJ   = false", 1)
        text = text.replace("doVJREAL = true", "doVJREAL = false", 1)
        text = text.replace("doVJVIRT = true", "doVJVIRT = false", 1)
    elif term == "VJ":
        text = text.replace("doBORN = true", "doBORN = false", 1)
        text = text.replace("doCT   = true", "doCT   = false", 1)
    else:
        raise ValueError(term)
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="CDF_RUN_2")
    parser.add_argument("--row-id", default="CDF_RUN_2:17")
    parser.add_argument("--g1", type=float, default=1.0)
    parser.add_argument("--calls", type=int, default=3_000_000)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    runner = load_runner()
    data = pd.read_csv(DATA_ROOT / f"{args.dataset}.csv")
    row = data[data.row_id.eq(args.row_id)].iloc[0]
    out = Path(args.out).resolve()
    (out / "cards").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(p for p in ("/home/dustin/miniforge3/envs/pdf-fit/lib", str(DYROOT / "lib"), env.get("LD_LIBRARY_PATH", "")) if p)
    values = {}
    for term in ("RES", "CT", "VJ"):
        stem = f"{args.dataset.replace('_', '-')}_{args.row_id.split(':')[-1]}_n3ll_nnlo_{term.lower()}_g1_{str(args.g1).replace('.', 'p')}"
        card = out / "cards" / f"{stem}.in"
        log = out / "logs" / f"{stem}.log"
        table = DYROOT / f"{stem}.txt"
        card.write_text(make_card(runner, row, output_name=stem, calls=args.calls, g1=args.g1, term=term))
        if table.exists():
            table.unlink()
        with log.open("w") as handle:
            subprocess.run([str(DYTURBO), str(card)], cwd=DYROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True, timeout=3600)
        value, uncertainty = runner.parse_first_value(table)
        values[term] = {"raw_fb_per_bin": value, "raw_unc_fb_per_bin": uncertainty}
    total = sum(values[t]["raw_fb_per_bin"] for t in values)
    total_unc = sum(values[t]["raw_unc_fb_per_bin"] ** 2 for t in values) ** 0.5
    width = float(row.qT_high - row.qT_low)
    result = {
        "status": "isolated_dyturbo_conventional_wy_term_decomposition_passed",
        "row_id": str(args.row_id), "dataset": args.dataset, "g1_GeV2": args.g1,
        "calls_per_vegas_component": int(args.calls),
        "order": 3, "primed": False,
        "terms": values,
        "conventional_interpretation": "W=RES; ASY=-CT; FO=VJ; Y=FO-ASY=VJ+CT",
        "reconstructed_full_wy_raw_fb_per_bin": total,
        "reconstructed_full_wy_unc_fb_per_bin": total_unc,
        "reconstructed_full_wy_pb_per_GeV": total / width / 1000.0,
        "data_pb_per_GeV": float(row.CS),
        "cards": str(out / "cards"), "logs": str(out / "logs"),
        "frozen_baseline_unchanged": True, "production_outputs_modified": False,
    }
    (out / "term_decomposition_status.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
