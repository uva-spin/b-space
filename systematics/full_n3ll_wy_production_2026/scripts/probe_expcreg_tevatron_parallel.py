#!/usr/bin/env python3
"""Parallel isolated expcreg scan on the CDF Run-2 W+Y control card."""

from __future__ import annotations

import concurrent.futures
import argparse
import json
import os
from pathlib import Path
import re
import subprocess


BASE = Path(__file__).resolve().parents[1]
SOURCE = BASE / "reports/tevatron_n3ll_nnlo_wy_final_g1_1p017/cards/CDF_RUN_2_full_n3ll_nnlo_grid_g1_1p017_seed_20260823.in"
OUT = BASE / "reports/expcreg_tevatron_stress_parallel"
DYTURBO = Path("/home/dustin/src/dyturbo-1.4.2/bin/dyturbo")
DYROOT = DYTURBO.parent.parent
VARIANTS = {"expcreg05": "0.5", "expcreg075": "0.75", "default1": "1.0", "expcreg15": "1.5", "expcreg2": "2.0"}
CALLS = 100000


def run_one(item: tuple[str, str]) -> dict:
    name, value = item
    source_text = SOURCE.read_text()
    text = re.sub(r"vegasncallsBORN\s*=\s*\d+", f"vegasncallsBORN   = {CALLS}", source_text)
    text = re.sub(r"vegasncallsCT\s*=\s*\d+", f"vegasncallsCT     = {CALLS}", text)
    text = re.sub(r"vegasncallsVJLO\s*=\s*\d+", f"vegasncallsVJLO   = {CALLS}", text)
    text = re.sub(r"vegasncallsVJREAL\s*=\s*\d+", f"vegasncallsVJREAL = {CALLS}", text)
    text = re.sub(r"vegasncallsVJVIRT\s*=\s*\d+", f"vegasncallsVJVIRT = {CALLS}", text)
    text = re.sub(r"qt_bins = \[ [^\]]+ \]", "qt_bins = [ 0 0.5 1 2 4 8 16 24 32 44 ]", text, count=1)
    text = re.sub(r"output_filename\s*=\s*\S+", f"output_filename = cdf_run2_expcreg_{name}", text)
    text += f"\n# Isolated exponentiation-regularization probe.\nexpcreg = {value}\n"
    card = OUT / f"cdf_run2_{name}.in"
    log = OUT / f"cdf_run2_{name}.log"
    table = DYROOT / f"cdf_run2_expcreg_{name}.txt"
    card.parent.mkdir(parents=True, exist_ok=True)
    card.write_text(text)
    table.unlink(missing_ok=True)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(x for x in ("/home/dustin/miniforge3/envs/pdf-fit/lib", str(DYROOT / "lib"), env.get("LD_LIBRARY_PATH", "")) if x)
    with log.open("w") as handle:
        try:
            proc = subprocess.run([str(DYTURBO), str(card)], cwd=DYROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, timeout=900, check=False)
            code, timed_out = proc.returncode, False
        except subprocess.TimeoutExpired:
            code, timed_out = None, True
    warnings = log.read_text(errors="replace").count("dequad abnormal termination")
    first = None
    if table.exists():
        for line in table.read_text(errors="replace").splitlines():
            if line.startswith("#") or not line.strip():
                continue
            try:
                first = list(map(float, line.split()[:2]))
                break
            except ValueError:
                pass
    return {"variant": name, "expcreg": float(value), "return_code": code, "timed_out": timed_out,
            "warning_count": warnings, "output_exists": table.exists(), "first_qT_value_unc": first,
            "card": str(card), "log": str(log), "table": str(table)}


def main() -> None:
    global CALLS
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, default=100000)
    ap.add_argument("--only", nargs="*", choices=tuple(VARIANTS), default=None)
    args = ap.parse_args()
    CALLS = int(args.calls)
    selected = VARIANTS if args.only is None else {key: VARIANTS[key] for key in args.only}
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(3, len(selected))) as pool:
        results = list(pool.map(run_one, selected.items()))
    status = {"status": "isolated_expcreg_tevatron_parallel", "calls_per_component": CALLS,
              "concurrent_dyturbo_jobs": 3, "results": results, "frozen_production_modified": False}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stress_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
