#!/usr/bin/env python3
"""Isolated full unprimed N3LL+NNLO W+Y grid for the six LHCb_7 rows.

The LHCb table is a fiducial candidate.  This diagnostic evaluates the same
fiducial lepton cuts used to define its DYTurbo acceptance factor, at the
observable level, without changing the accepted production data or cache.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from run_dyturbo_full_n3ll_nnlo_probe import DYTURBO, DYROOT, full_card_text, load_runner


SYSTEMATICS = Path(__file__).resolve().parents[2]
PROJECT = SYSTEMATICS.parent
DATA_ROOT = PROJECT / "Data/v23a_tevatron_plus_lhcb7_fiducial_candidate"
BASE = SYSTEMATICS / "full_n3ll_wy_production_2026"
DEFAULT_OUT = BASE / "reports/lhcb7_full_n3ll_nnlo_fiducial_grid"


def parse_grid(path: Path, first_edge: float, last_edge: float) -> pd.DataFrame:
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 4 or parts[0].startswith("#"):
            continue
        try:
            lo, hi, value, unc = map(float, parts[:4])
        except ValueError:
            continue
        rows.append({"qT_low": lo, "qT_high": hi,
                     "raw_full_wy_fb_per_bin": value,
                     "raw_full_wy_unc_fb_per_bin": unc})
    rows = [r for r in rows if not (abs(r["qT_low"] - first_edge) < 1e-12 and
                                   abs(r["qT_high"] - last_edge) < 1e-12)]
    if not rows:
        raise RuntimeError(f"no qT grid in {path}")
    return pd.DataFrame(rows)


def make_lhcb_card(row: pd.Series, *, name: str, calls: int, seed: int,
                   cores: int, g1: float, expcreg: float | None) -> str:
    text = full_card_text(row, output_name=name,
                          pdf_set="NNPDF40_nnlo_as_01180", pdf_member=0,
                          cores=cores, calls=calls, seed=seed)
    text = text.replace("ih2          = -1", "ih2          = 1", 1)
    text = text.replace("g1 = 0.0", f"g1 = {g1:.12g}", 1)
    text = text.replace("y_bins  = [ -5.0 5.0 ]", "y_bins  = [ 2.0 4.25 ]", 1)
    text = text.replace("qt_bins = [ 0 2.2 ]", "qt_bins = [ 0 2.2 3.4 4.6 5.8 7.2 8.7 ]", 1)
    text = text.replace("lcptcut = 0", "lcptcut = 20", 1)
    text = text.replace("lcymin = 0", "lcymin = 2.0", 1)
    text = text.replace("lcymax = 1000", "lcymax = 4.5", 1)
    text = text.replace("lfptcut = 0", "lfptcut = 20", 1)
    text = text.replace("lfymin = 0", "lfymin = 2.0", 1)
    text = text.replace("lfymax = 1000", "lfymax = 4.5", 1)
    # The common card template does not always emit the charged-lepton cut
    # keys; append explicit values so the fiducial convention is unambiguous.
    text += "\n# Explicit LHCb fiducial lepton cuts.\nlcptcut = 20\nlcymin = 2.0\nlcymax = 4.5\nlfptcut = 20\nlfymin = 2.0\nlfymax = 4.5\n"
    if expcreg is not None:
        text += f"\n# Isolated regularization audit; not production.\nexpcreg = {float(expcreg):.12g}\n"
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=20260860)
    ap.add_argument("--cores", type=int, default=8)
    ap.add_argument("--g1", type=float, default=1.017)
    ap.add_argument("--expcreg", type=float, default=None)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--reuse-table", action="store_true",
                    help="reuse an existing DYTurbo table from the same card")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    data = pd.read_csv(DATA_ROOT / "LHCb_7.csv")
    # Scope authority is the six low-qT rows; rows 6--13 are explicitly not
    # part of the 353-row candidate.
    data = data[data.row_id.astype(str).isin([f"LHCb_7:{i}" for i in range(6)])].copy()
    data = data.sort_values("qT_low").reset_index(drop=True)
    if len(data) != 6:
        raise RuntimeError("LHCb_7 authority rows are not the expected six")
    edges = [float(data.qT_low.iloc[0])] + data.qT_high.astype(float).tolist()
    out = Path(args.out).resolve()
    cards, logs = out / "cards", out / "logs"
    cards.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    runner = load_runner()
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(p for p in ("/home/dustin/miniforge3/envs/pdf-fit/lib",
                                                    str(DYROOT / "lib"), env.get("LD_LIBRARY_PATH", "")) if p)
    reg_tag = "default" if args.expcreg is None else f"expcreg_{str(float(args.expcreg)).replace('.', 'p')}"
    name = f"LHCb_7_full_n3ll_nnlo_g1_{str(float(args.g1)).replace('.', 'p')}_{reg_tag}_seed_{args.seed}"
    card = cards / f"{name}.in"
    log = logs / f"{name}.log"
    table = DYROOT / f"{name}.txt"
    text = make_lhcb_card(data.iloc[0], name=name, calls=args.calls,
                          seed=args.seed, cores=args.cores, g1=args.g1,
                          expcreg=args.expcreg)
    text = re.sub(r"qt_bins = \[ [^\]]+ \]", "qt_bins = [ " + " ".join(f"{x:.12g}" for x in edges) + " ]", text, count=1)
    card.write_text(text)
    if table.exists() and not args.reuse_table:
        table.unlink()
    if not table.exists():
        with log.open("w") as handle:
            subprocess.run([str(DYTURBO), str(card)], cwd=DYROOT, env=env,
                           stdout=handle, stderr=subprocess.STDOUT,
                           check=True, timeout=args.timeout)
    grid = parse_grid(table, edges[0], edges[-1])
    if len(grid) != len(data):
        raise RuntimeError(f"parsed {len(grid)} qT bins, expected {len(data)}")
    merged = data[["dataset", "row_id", "qT_low", "qT_high", "CS", "error",
                   "theory_fiducial_factor"]].reset_index(drop=True).join(
                       grid.drop(columns=["qT_low", "qT_high"]).reset_index(drop=True))
    width = merged.qT_high - merged.qT_low
    merged["full_wy_pb_per_GeV"] = merged.raw_full_wy_fb_per_bin / width / 1000.0
    merged["full_wy_unc_pb_per_GeV"] = merged.raw_full_wy_unc_fb_per_bin / width / 1000.0
    merged["full_wy_to_data_ratio"] = merged.full_wy_pb_per_GeV / merged.CS
    merged.to_csv(out / "lhcb7_full_wy_grid.csv", index=False)
    rel_unc = merged.full_wy_unc_pb_per_GeV / merged.full_wy_pb_per_GeV.abs()
    status = {
        "status": "isolated_lhcb7_full_unprimed_n3ll_nnlo_fiducial_grid_complete_not_production",
        "formula": "W_N3LL + (FO_NNLO - ASY_NNLO)", "order": 3, "primed": False,
        "g1_GeV2": float(args.g1), "expcreg": args.expcreg,
        "calls_per_vegas_component": int(args.calls), "row_count": int(len(merged)),
        "fiducial_cuts": "both leptons pT>20 GeV, 2<eta<4.5; pp beams; y=2--4.25; Q=60--120 GeV",
        "checks": {
            "all_finite": bool(np.isfinite(merged.full_wy_pb_per_GeV).all()),
            "all_positive": bool((merged.full_wy_pb_per_GeV > 0).all()),
            "mean_relative_mc_uncertainty": float(rel_unc.mean()),
            "max_relative_mc_uncertainty": float(rel_unc.max()),
            "median_full_wy_to_data_ratio": float(merged.full_wy_to_data_ratio.median()),
            "min_full_wy_to_data_ratio": float(merged.full_wy_to_data_ratio.min()),
            "max_full_wy_to_data_ratio": float(merged.full_wy_to_data_ratio.max()),
        },
        "acceptance_factor_source": "row theory_fiducial_factor from documented DYTurbo NLO+NLL acceptance candidate; direct fiducial W+Y is reported separately",
        "card": str(card), "log": str(log), "table": str(table),
        "frozen_baseline_unchanged": True, "production_outputs_modified": False,
        "promotion_authorized": False,
    }
    (out / "lhcb7_grid_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
