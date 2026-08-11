#!/usr/bin/env python3
"""Run genuine Z+jet NLO checks using the frozen Tevatron MCFM runner."""

from __future__ import annotations

import importlib.util
import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[4]
RUNNER = ROOT / "v23/tools/run_tevatron_mcfm_benchmark.py"
spec = importlib.util.spec_from_file_location("tevatron_mcfm_runner", RUNNER)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load {RUNNER}")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

_lo_card_text = runner.card_text

pilot = argparse.ArgumentParser(add_help=False)
pilot.add_argument("--nlo-real-calls", type=int)
pilot.add_argument("--nlo-virtual-calls", type=int)
pilot.add_argument("--iterbatch-warmup", type=int, default=1)
pilot.add_argument("--iterbatch-first", type=int, default=1)
pilot.add_argument("--iterbatch-later", type=int, default=1)
pilot.add_argument("--call-boost", type=float, default=1.0)
pilot.add_argument("--iter-call-mult", type=float, default=1.0)
pilot.add_argument("--warmup-precision-goal", type=float, default=10.0)
pilot.add_argument("--warmup-chisq-goal", type=float, default=1.0e9)
pilot.add_argument("--result-precision-goal", type=float, default=10.0)
pilot.add_argument("--internal-scalevar", action="store_true")
PILOT, remaining = pilot.parse_known_args()
sys.argv = [sys.argv[0], *remaining]


def nlo_card_text(*args: object, **kwargs: object) -> str:
    text = _lo_card_text(*args, **kwargs)
    calls = int(kwargs["calls"])
    real_calls = PILOT.nlo_real_calls or calls
    virtual_calls = PILOT.nlo_virtual_calls or calls
    text = text.replace("    part = lo\n", "    part = nlo\n", 1)
    text = text.replace("    initcallsnloreal=1000000\n", f"    initcallsnloreal={real_calls}\n")
    text = text.replace("    initcallsnlovirt=200000\n", f"    initcallsnlovirt={virtual_calls}\n")
    text = text.replace("    precisiongoal = 0.003\n", f"    precisiongoal = {PILOT.result_precision_goal}\n")
    text = text.replace("    warmupprecisiongoal = 0.25\n", f"    warmupprecisiongoal = {PILOT.warmup_precision_goal}\n")
    text = text.replace("    warmupchisqgoal = 3.0\n", f"    warmupchisqgoal = {PILOT.warmup_chisq_goal}\n")
    if PILOT.internal_scalevar:
        text = text.replace("    doscalevar = .false.\n", "    doscalevar = .true.\n", 1)
        text = text.replace("    maxscalevar = 6\n", "    maxscalevar = 6\n", 1)
    text += (
        f"    iterbatchwarmup = {PILOT.iterbatch_warmup}\n"
        f"    iterbatch1 = {PILOT.iterbatch_first}\n"
        f"    iterbatch2 = {PILOT.iterbatch_later}\n"
        f"    callboost = {PILOT.call_boost}\n"
        f"    itercallmult = {PILOT.iter_call_mult}\n"
    )
    return text


runner.card_text = nlo_card_text

if __name__ == "__main__":
    runner.main()
