#!/usr/bin/env python3
"""Compare the historical E288 invariant-cross-section convention in isolation.

The archived DYTurbo E288 benchmarks used the published invariant-like ``A``
quantity directly, with no ``dsdxf`` or ``edsdp3`` switch.  This probe evaluates
the same one-point bin with the new unprimed N3LL+NNLO W+Y runtime and records
both the raw result and the current ``CS=A/PreFactor`` target.  It is an audit
only: no fit-ready data or production output is modified.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import pandas as pd

from run_dyturbo_full_n3ll_nnlo_probe import DYTURBO, DYROOT, full_card_text, load_runner


SYSTEMATICS = Path(__file__).resolve().parents[2]
PROJECT = SYSTEMATICS.parent
DATA_ROOT = PROJECT / "Data/v23a_fixed_target_plus_tevatron_absolute_fit_ready"
OUT = SYSTEMATICS / "full_n3ll_wy_production_2026/reports/fixed_target_legacy_invariant_probe"


def parse_first(path: Path) -> tuple[float, float]:
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2 and not parts[0].startswith("#"):
            try:
                return float(parts[0]), float(parts[1])
            except ValueError:
                continue
    raise RuntimeError(f"no numeric output in {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="E288_200")
    ap.add_argument("--row", default="E288_200:0")
    ap.add_argument("--g1", type=float, default=1.017)
    ap.add_argument("--calls", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=20260828)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    load_runner()
    frame = pd.read_csv(DATA_ROOT / f"{args.dataset}.csv")
    selected = frame[frame.row_id.eq(args.row)]
    if len(selected) != 1:
        raise SystemExit(f"row is not unique: {args.row}")
    row = selected.iloc[0].copy()
    eps_q = max(1.0e-3, 1.0e-3 * float(row.qT))
    row["qT_low"] = max(1.0e-6, float(row.qT) - eps_q)
    row["qT_high"] = float(row.qT) + eps_q
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"{args.dataset}_{args.row.split(':')[-1]}_legacy_invariant_wy"
    card = OUT / f"{name}.in"
    log = OUT / f"{name}.log"
    table = DYROOT / f"{name}.txt"

    text = full_card_text(row, output_name=name, pdf_set="NNPDF40_nnlo_as_01180",
                          pdf_member=0, cores=4, calls=int(args.calls), seed=int(args.seed))
    text = text.replace("ih2          = -1", "ih2          = 1", 1)
    text = text.replace("g1 = 0.0", f"g1 = {args.g1:.12g}", 1)
    # Explicitly retain the historical observable convention.
    text += "\ndsdqt2 = false\ndsdxf = false\nedsdp3 = false\n"
    # A zero-width qT interval is invalid; use a narrow symmetric bin while
    # retaining the published center, and retain the published y/m intervals.
    text = re.sub(r"qt_bins = \[ [^\]]+ \]", f"qt_bins = [ {float(row.qT_low):.12g} {float(row.qT_high):.12g} ]", text, count=1)
    text = re.sub(r"y_bins  = \[ [^\]]+ \]", f"y_bins  = [ {float(row.y_Low):.12g} {float(row.y_High):.12g} ]", text, count=1)
    text = re.sub(r"m_bins  = \[ [^\]]+ \]", f"m_bins  = [ {float(row.QM_Low):.12g} {float(row.QM_High):.12g} ]", text, count=1)
    card.write_text(text)

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(p for p in ("/home/dustin/miniforge3/envs/pdf-fit/lib", str(DYROOT / "lib"), env.get("LD_LIBRARY_PATH", "")) if p)
    if args.force:
        table.unlink(missing_ok=True)
    proc = None
    if not table.exists():
        with log.open("w") as handle:
            proc = subprocess.run([str(DYTURBO), str(card)], cwd=DYROOT, env=env,
                                  stdout=handle, stderr=subprocess.STDOUT,
                                  check=False, timeout=3600)
    value = uncertainty = None
    if table.exists() and table.stat().st_size > 0:
        value, uncertainty = parse_first(table)
    result = {
        "status": "fixed_target_legacy_invariant_probe_complete_not_production" if value is not None and (proc is None or proc.returncode == 0) else "fixed_target_legacy_invariant_probe_failed",
        "dataset": args.dataset, "row_id": args.row,
        "formula": "W_N3LL + (FO_NNLO - ASY_NNLO)",
        "engine": str(DYTURBO), "order": {"resummation": "N3LL_unprimed", "fixed_order": "NNLO_VJ", "primed": False},
        "observable": "historical invariant-like A convention; dsdxf=false; edsdp3=false",
        "target_assumption": "proton beam on proton target; no nuclear correction",
        "g1_GeV2": float(args.g1), "calls_per_vegas_component": int(args.calls), "random_seed": int(args.seed),
        "raw_dyturbo_value": value, "raw_dyturbo_uncertainty": uncertainty,
        "published_A": float(row.A), "published_dA": float(row.dA),
        "fit_ready_CS_A_over_PreFactor": float(row.CS), "fit_ready_error": float(row.error),
        "raw_to_A_ratio": None if value is None else value / float(row.A),
        "raw_to_fit_ready_CS_ratio": None if value is None else value / float(row.CS),
        "card": str(card), "log": str(log), "table": str(table),
        "return_code": None if proc is None else proc.returncode,
        "interpretation": "legacy observable/unit diagnostic only; no fixed-target production authorization",
        "frozen_baseline_unchanged": True, "production_outputs_modified": False,
    }
    (OUT / "probe_status.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result["status"].endswith("failed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
