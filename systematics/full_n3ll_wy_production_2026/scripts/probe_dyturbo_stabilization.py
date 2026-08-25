#!/usr/bin/env python3
"""Isolated DYTurbo numerical-stability pilot for one fixed-target row.

This never edits the DYTurbo source or frozen production inputs.  It clones one
diagnostic card, varies only inverse-transform/large-b settings, and records
whether the resulting full W+Y calculation terminates with finite output.
"""

from __future__ import annotations

import concurrent.futures
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
DYTURBO = Path("/home/dustin/src/dyturbo-1.4.2/bin/dyturbo")
DYROOT = DYTURBO.parent.parent
SOURCE = ROOT / "reports/fixed_target_fixed_y_grid_floor_probe/cards/E288_200_30_full_n3ll_nnlo_g1_1p017_seed_20260860.in"
OUTDIR = ROOT / "reports/dyturbo_stabilization_pilot_e288_200_30"

VARIANTS = {
    "blim1": {"blim": "1.0"},
    "blim05": {"blim": "0.5"},
    "bint1e3": {"bintaccuracy": "1.0e-3"},
    "bint1e2": {"bintaccuracy": "1.0e-2"},
    "blim1_bint1e3": {"blim": "1.0", "bintaccuracy": "1.0e-3"},
    "sumlogs": {"sumlogs": "true"},
    "blim1_bstar_all": {
        "blim": "1.0",
        "bstar_pdf": "true",
        "bstar_sudakov": "true",
        "bstar_expc": "true",
    },
    "qtcut02": {"qtcut": "0.02", "xqtcut": "0.0"},
    "qtcut05": {"qtcut": "0.05", "xqtcut": "0.0"},
    "xqtcut01": {"xqtcut": "0.01", "qtcut": "0.0"},
    "bpres4": {"bprescription": "4", "blim": "2.0"},
    "bpres4_blim1": {"bprescription": "4", "blim": "1.0"},
    "qt_bins05": {"qt_bins": "[ 0.05 0.2 ]"},
    "qt_bins10": {"qt_bins": "[ 0.10 0.2 ]"},
    "qt_bins15": {"qt_bins": "[ 0.15 0.2 ]"},
    "modlog2": {"modlog": "2"},
    "modlog3": {"modlog": "3"},
    "modlog4": {"modlog": "4"},
    "fixedorder_only": {"fixedorder_only": "true"},
    "order2": {"order": "2"},
    "order1": {"order": "1"},
    "evolmode1": {"evolmode": "1"},
    "evolmode3": {"evolmode": "3"},
    "order_evol2": {"order_evol": "2"},
    "order_expc2": {"order_expc": "2"},
    "expcreg0": {"expcreg": "0"},
    "expcreg2": {"expcreg": "2"},
    "expc1": {"expc": "1"},
    "expc2": {"expc": "2"},
    "expc4": {"expc": "4"},
    "expcreg05": {"expcreg": "0.5"},
    "expcreg075": {"expcreg": "0.75"},
    "expcreg125": {"expcreg": "1.25"},
    "expcreg15": {"expcreg": "1.5"},
    "expcreg25": {"expcreg": "2.5"},
}


def clone_card(name: str, settings: dict[str, str]) -> tuple[Path, Path]:
    text = SOURCE.read_text()
    text = re.sub(r"vegasncallsBORN\s*=\s*\d+", "vegasncallsBORN   = 100000", text)
    text = re.sub(r"vegasncallsCT\s*=\s*\d+", "vegasncallsCT     = 100000", text)
    text = re.sub(r"vegasncallsVJLO\s*=\s*\d+", "vegasncallsVJLO   = 100000", text)
    text = re.sub(r"vegasncallsVJREAL\s*=\s*\d+", "vegasncallsVJREAL = 100000", text)
    text = re.sub(r"vegasncallsVJVIRT\s*=\s*\d+", "vegasncallsVJVIRT = 100000", text)
    text = re.sub(r"niterVJ\s*=\s*\d+", "niterVJ       = 5", text)
    text = re.sub(r"output_filename\s*=\s*\S+", f"output_filename = e288_200_30_stab_{name}", text)
    text += "\n# Isolated stabilization-pilot overrides.\n"
    for key, value in settings.items():
        text += f"{key} = {value}\n"
    card = OUTDIR / "cards" / f"e288_200_30_{name}.in"
    log = OUTDIR / "logs" / f"e288_200_30_{name}.log"
    card.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    card.write_text(text)
    return card, log


def run_one(item: tuple[str, dict[str, str]]) -> dict:
    name, settings = item
    card, log = clone_card(name, settings)
    output = DYROOT / f"e288_200_30_stab_{name}.txt"
    output.unlink(missing_ok=True)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        x for x in ["/home/dustin/miniforge3/envs/pdf-fit/lib", str(DYROOT / "lib"), env.get("LD_LIBRARY_PATH", "")] if x
    )
    start = time.time()
    timed_out = False
    with log.open("w") as handle:
        try:
            proc = subprocess.run([str(DYTURBO), str(card.resolve())], cwd=DYROOT, env=env,
                                  stdout=handle, stderr=subprocess.STDOUT, timeout=180, check=False)
            return_code = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            return_code = None
    elapsed = time.time() - start
    output_text = output.read_text(errors="replace") if output.exists() else ""
    finite = bool(output_text) and all(token not in output_text.lower() for token in ("nan", "inf"))
    log_text = log.read_text(errors="replace")
    return {
        "variant": name,
        "settings": settings,
        "return_code": return_code,
        "timed_out": timed_out,
        "elapsed_s": elapsed,
        "output_exists": output.exists(),
        "output_finite_text": finite,
        "warning_count": log_text.count("dequad abnormal termination"),
        "output": str(output),
        "card": str(card),
        "log": str(log),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", choices=tuple(VARIANTS), default=None)
    args = parser.parse_args()
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    selected = VARIANTS if args.only is None else {key: VARIANTS[key] for key in args.only}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run_one, selected.items()))
    status = {
        "status": "isolated_dyturbo_stabilization_pilot",
        "source_card": str(SOURCE),
        "calls_per_component": 100000,
        "timeout_s": 180,
        "results": results,
    }
    (OUTDIR / "pilot_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
