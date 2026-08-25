#!/usr/bin/env python3
"""Probe whether the external unprimed N3LL+NNLO W+Y engine can cover fixed target rows.

This is a capability test only.  It does not alter the accepted 329-row fit
or claim a fixed-target production result.  The probe uses a proton beam on a
proton target (``ih1=ih2=1``), the published rapidity and mass bin, and the
same `W=RES`, `ASY=-CT`, `FO=VJ` decomposition used for the Tevatron oracle.
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
DATA_ROOT = PROJECT / "Data/v23a_tevatron_plus_lhcb7_fiducial_candidate"
OUT = SYSTEMATICS / "full_n3ll_wy_production_2026/reports/fixed_target_n3ll_nnlo_capability_probe"


def qT_half_width(dataset: str) -> float:
    """Return the published fixed-target qT-bin half-width.

    E288 and E605 quote 0.2-GeV qT bins centered at 0.1, 0.3, ...;
    E772 quotes 0.25-GeV bins centered at 0.125, 0.375, ....  The old
    capability probe used a generic 0.1-GeV bin, which is retained only in
    its historical artifacts and must not be used for normalization.
    """
    return 0.125 if dataset == "E772" else 0.1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="E288_200")
    ap.add_argument("--row", default="E288_200:0")
    ap.add_argument("--g1", type=float, default=1.017)
    ap.add_argument("--calls", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--target-z", type=float, default=None,
                    help="isolated nuclear-PDF proton count; requires --target-a")
    ap.add_argument("--target-a", type=float, default=None,
                    help="isolated nuclear-PDF mass count; requires --target-z")
    ap.add_argument("--tag", default=None,
                    help="optional isolated output tag to avoid concurrent probe collisions")
    ap.add_argument("--out", default=str(OUT),
                    help="isolated output directory for this probe")
    ap.add_argument("--force", action="store_true", help="rerun and replace this isolated probe table")
    args = ap.parse_args()
    if (args.target_z is None) != (args.target_a is None):
        raise SystemExit("--target-z and --target-a must be supplied together")

    data = pd.read_csv(DATA_ROOT / f"{args.dataset}.csv")
    row = data[data.row_id.eq(args.row)]
    if len(row) != 1:
        raise SystemExit(f"requested row is not unique: {args.row}")
    row = row.iloc[0].copy()
    half_width = qT_half_width(args.dataset)
    row["qT_low"] = max(0.0, float(row.qT) - half_width)
    row["qT_high"] = float(row.qT) + half_width
    load_runner()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    target_tag = "pp" if args.target_z is None else f"nuc_z{args.target_z:g}_a{args.target_a:g}".replace(".", "p")
    if args.tag:
        target_tag = str(args.tag)
    name = f"{args.dataset}_{args.row.split(':')[-1]}_full_n3ll_nnlo_probe_{target_tag}"
    card = out / f"{name}.in"
    log = out / f"{name}.log"
    table = DYROOT / f"{name}.txt"
    text = full_card_text(row, output_name=name, pdf_set="NNPDF40_nnlo_as_01180", pdf_member=0, cores=4, calls=int(args.calls), seed=int(args.seed))
    # Fixed-target fit-ready tables are inclusive in the lepton decay.  Use
    # the same no-lepton-cut convention as the historical E288/E605/E772
    # DYTurbo cards; the Tevatron/LHCb fiducial cards remain separate.
    text = text.replace("makecuts = true", "makecuts = false", 1)
    text = text.replace("ih2          = -1", "ih2          = 1", 1)
    if args.target_z is not None:
        text = text.replace(
            "nproc        = 3",
            "nproc        = 3\nnuclearpdf   = true\nZ1 = 1\nA1 = 1\nZ2 = %.12g\nA2 = %.12g" % (args.target_z, args.target_a),
            1,
        )
    text = text.replace("g1 = 0.0", f"g1 = {args.g1:.12g}", 1)
    qlow = max(0.0, float(row.qT) - half_width)
    qhigh = float(row.qT) + half_width
    text = re.sub(r"qt_bins = \[ [^\]]+ \]", f"qt_bins = [ {qlow:.12g} {qhigh:.12g} ]", text, count=1)
    text = re.sub(r"y_bins  = \[ [^\]]+ \]", f"y_bins  = [ {float(row.y_Low):.12g} {float(row.y_High):.12g} ]", text, count=1)
    text = re.sub(r"m_bins  = \[ [^\]]+ \]", f"m_bins  = [ {float(row.QM_Low):.12g} {float(row.QM_High):.12g} ]", text, count=1)
    card.write_text(text)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(p for p in ("/home/dustin/miniforge3/envs/pdf-fit/lib", str(DYROOT / "lib"), env.get("LD_LIBRARY_PATH", "")) if p)
    if args.force and table.exists():
        table.unlink()
    if not table.exists():
        with log.open("w") as handle:
            proc = subprocess.run([str(DYTURBO), str(card)], cwd=DYROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, timeout=3600)
    else:
        proc = None
    finite = table.exists() and table.stat().st_size > 0
    value = uncertainty = None
    if finite:
        # Keep the numerical observable in the status artifact rather than
        # requiring a later parser to infer it from a shared DYTurbo table.
        value, uncertainty = load_runner().parse_first_value(table)
    q_width = float(qhigh - qlow)
    status = {
        "status": "fixed_target_external_n3ll_nnlo_probe_complete" if finite else "fixed_target_external_n3ll_nnlo_probe_failed",
        "dataset": args.dataset,
        "row_id": args.row,
        "target_assumption": (
            "proton beam on proton target; nuclear corrections not included"
            if args.target_z is None else
            "proton beam on isospin-mixed nuclear target; no nuclear modification beyond proton/neutron counting"
        ),
        "target_Z": None if args.target_z is None else float(args.target_z),
        "target_A": None if args.target_a is None else float(args.target_a),
        "formula": "W_N3LL + (FO_NNLO - ASY_NNLO)",
        "engine": str(DYTURBO),
        "order": {"resummation": "N3LL_unprimed", "fixed_order": "NNLO_VJ", "primed": False},
        "g1_GeV2": float(args.g1),
        "calls_per_vegas_component": int(args.calls),
        "random_seed": int(args.seed),
        "card": str(card),
        "log": str(log),
        "table": str(table),
        "return_code": None if proc is None else proc.returncode,
        "table_nonempty": bool(finite),
        "raw_full_wy_fb_per_bin": None if value is None else float(value),
        "raw_full_wy_unc_fb_per_bin": None if uncertainty is None else float(uncertainty),
        "qT_bin_width_GeV": q_width,
        "full_wy_fb_per_GeV": None if value is None else float(value / q_width),
        "full_wy_unc_fb_per_GeV": None if uncertainty is None else float(uncertainty / q_width),
        "per_nucleon_full_wy_fb_per_GeV": (
            None if value is None or args.target_a is None else float(value / q_width / args.target_a)
        ),
        "per_nucleon_full_wy_unc_fb_per_GeV": (
            None if uncertainty is None or args.target_a is None else float(uncertainty / q_width / args.target_a)
        ),
        "production_authorized": False,
        "frozen_baseline_unchanged": True,
        "production_outputs_modified": False,
    }
    # Tagged runs must never overwrite one another's status.  Preserve the
    # historical generic status only for an untagged probe.
    status_path = out / f"{name}_status.json"
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    if args.tag is None:
        (out / "probe_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))
    if not finite:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
