#!/usr/bin/env python3
"""Run an isolated DYTurbo NNLO fixed-order capability probe.

This is not a production fit.  It checks whether the installed external
Tevatron observable engine can provide the NNLO fixed-order side needed for a
standard unprimed N3LL+NNLO W+Y construction.  All cards, logs, and tables are
written below this campaign directory.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SYSTEMATICS = Path(__file__).resolve().parents[2]
PROJECT = SYSTEMATICS.parent
RUNNER_PATH = PROJECT / "b-space-public/v23/tools/run_tevatron_dyturbo_benchmark.py"

spec = importlib.util.spec_from_file_location("candidate_dyturbo_runner", RUNNER_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {RUNNER_PATH}")
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)

_base_card_text = runner.card_text


def nnlo_card_text(*args: object, **kwargs: object) -> str:
    text = _base_card_text(*args, **kwargs)
    # DYTurbo's unprimed order convention counts the inclusive Born order:
    # order=3 with primed=false supplies the NNLO V+jet component needed for
    # an N3LL+NNLO observable.
    text = text.replace("order           = 1\n", "order           = 3\n", 1)
    text = text.replace("primed          = true\n", "primed          = false\n", 1)
    text = text.replace("doVJREAL = false", "doVJREAL = true", 1)
    text = text.replace("doVJVIRT = false", "doVJVIRT = true", 1)
    text = text.replace("VJquad = true", "VJquad = false", 1)
    text = text.replace("intDimVJ   = 3", "intDimVJ   = -1", 1)
    text = text.replace("makecuts = false", "makecuts = true", 1)
    calls = kwargs.pop("vj_calls", 10_000_000)
    text = text.replace("vegasncallsVJREAL = 100000", f"vegasncallsVJREAL = {int(calls)}", 1)
    text = text.replace("vegasncallsVJVIRT = 100000", f"vegasncallsVJVIRT = {int(calls)}", 1)
    # Keep this probe explicitly fixed-order: no external NP or resummed
    # contribution is being mistaken for the candidate's fitted W term.
    text = text.replace("fixedorder_only = true\n", "fixedorder_only = true\n", 1)
    text = text.replace("output_filename = {output_name}", "output_filename = {output_name}")
    return text


runner.card_text = nnlo_card_text

if __name__ == "__main__":
    # The imported runner accepts the same row/data/output options.  We add a
    # stable candidate default when the caller does not provide --out.
    if "--out" not in sys.argv:
        sys.argv.extend([
            "--out",
            str(SYSTEMATICS / "full_n3ll_wy_production_2026/reports/dyturbo_nnlo_probe"),
        ])
    runner.main()
