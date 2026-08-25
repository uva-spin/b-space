#!/usr/bin/env python3
"""Evaluate the six LHCb authority-row finite tails as full minus RES.

The direct fiducial full W+Y table has large cancellation-driven integration
errors.  This companion diagnostic evaluates the same card twice, with the
RES-only card providing the b-space W reference, and forms the row-level
finite tail by subtraction.  It is not a production claim.
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

from run_lhcb_full_n3ll_nnlo_grid import (
    BASE, DATA_ROOT, DYTURBO, DYROOT, load_runner, make_lhcb_card, parse_grid,
)


def set_res_only(text: str) -> str:
    for key in ("doCT", "doVJ", "doVJREAL", "doVJVIRT"):
        text = re.sub(rf"(^\s*{key}\s*=\s*)true\b", rf"\1false", text, flags=re.MULTILINE)
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--cores", type=int, default=8)
    ap.add_argument("--g1", type=float, default=1.017)
    ap.add_argument("--expcreg", type=float, default=2.0)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--reuse-full-grid", default=str(BASE / "reports/lhcb7_full_n3ll_nnlo_fiducial_expcreg2p0_1m/lhcb7_full_wy_grid.csv"),
                    help="Existing six-row full W+Y table to reuse; set to empty to rerun it.")
    ap.add_argument("--out", default=str(BASE / "reports/lhcb7_y_fullminusres_expcreg2p0_1m"))
    args = ap.parse_args()
    data = pd.read_csv(DATA_ROOT / "LHCb_7.csv")
    data = data[data.row_id.astype(str).isin([f"LHCb_7:{i}" for i in range(6)])].sort_values("qT_low").reset_index(drop=True)
    if len(data) != 6:
        raise RuntimeError("LHCb authority rows are not the expected six")
    edges = [float(data.qT_low.iloc[0])] + data.qT_high.astype(float).tolist()
    out = Path(args.out).resolve(); cards, logs = out / "cards", out / "logs"
    cards.mkdir(parents=True, exist_ok=True); logs.mkdir(parents=True, exist_ok=True)
    runner = load_runner()
    env = os.environ.copy(); env["LD_LIBRARY_PATH"] = ":".join(p for p in ("/home/dustin/miniforge3/envs/pdf-fit/lib", str(DYROOT / "lib"), env.get("LD_LIBRARY_PATH", "")) if p)
    vals = {}
    for label, res_only in (("full", False), ("res", True)):
        if label == "full" and args.reuse_full_grid:
            reuse = Path(args.reuse_full_grid)
            if reuse.exists() and not args.force:
                old = pd.read_csv(reuse).sort_values("qT_low")
                if len(old) == 6 and {"raw_full_wy_fb_per_bin", "raw_full_wy_unc_fb_per_bin"}.issubset(old.columns):
                    vals[label] = old[["qT_low", "qT_high", "raw_full_wy_fb_per_bin", "raw_full_wy_unc_fb_per_bin"]].reset_index(drop=True)
                    continue
        name = f"LHCb_7_fullminusres_n3ll_g1_{str(float(args.g1)).replace('.', 'p')}_expcreg_{str(float(args.expcreg)).replace('.', 'p')}_seed_{args.seed}_{label}"
        card = cards / f"{name}.in"; log = logs / f"{name}.log"; table = DYROOT / f"{name}.txt"
        text = make_lhcb_card(data.iloc[0], name=name, calls=args.calls, seed=args.seed, cores=args.cores, g1=args.g1, expcreg=args.expcreg)
        text = re.sub(r"qt_bins = \[ [^\]]+ \]", "qt_bins = [ " + " ".join(f"{x:.12g}" for x in edges) + " ]", text, count=1)
        if res_only: text = set_res_only(text)
        card.write_text(text)
        if args.force: table.unlink(missing_ok=True)
        if not table.exists():
            with log.open("w") as handle:
                subprocess.run([str(DYTURBO), str(card)], cwd=DYROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True, timeout=args.timeout)
        grid = parse_grid(table, edges[0], edges[-1])
        if len(grid) != 6: raise RuntimeError(f"{label}: expected six bins, got {len(grid)}")
        vals[label] = grid
    full, res = vals["full"], vals["res"]
    y = full.raw_full_wy_fb_per_bin.to_numpy(float) - res.raw_full_wy_fb_per_bin.to_numpy(float)
    y_unc = np.hypot(full.raw_full_wy_unc_fb_per_bin.to_numpy(float), res.raw_full_wy_unc_fb_per_bin.to_numpy(float))
    out_frame = data[["dataset", "row_id", "qT_low", "qT_high", "CS", "error", "theory_fiducial_factor"]].copy()
    out_frame["full_raw_fb_per_bin"] = full.raw_full_wy_fb_per_bin.to_numpy(float)
    out_frame["full_unc_fb_per_bin"] = full.raw_full_wy_unc_fb_per_bin.to_numpy(float)
    out_frame["RES_raw_fb_per_bin"] = res.raw_full_wy_fb_per_bin.to_numpy(float)
    out_frame["RES_unc_fb_per_bin"] = res.raw_full_wy_unc_fb_per_bin.to_numpy(float)
    out_frame["Y_raw_fb_per_bin"] = y; out_frame["Y_unc_fb_per_bin"] = y_unc
    width = out_frame.qT_high - out_frame.qT_low
    for label, arr in (("full", full.raw_full_wy_fb_per_bin.to_numpy(float)), ("RES", res.raw_full_wy_fb_per_bin.to_numpy(float)), ("Y", y), ("Y_unc", y_unc)):
        out_frame[label + ("_pb_per_GeV" if label != "Y_unc" else "_pb_per_GeV")] = arr / width.to_numpy(float) / 1000.0
    out_frame["full_unc_pb_per_GeV"] = full.raw_full_wy_unc_fb_per_bin.to_numpy(float) / width.to_numpy(float) / 1000.0
    out_frame["RES_unc_pb_per_GeV"] = res.raw_full_wy_unc_fb_per_bin.to_numpy(float) / width.to_numpy(float) / 1000.0
    out_frame["full_to_data"] = out_frame.full_pb_per_GeV / out_frame.CS
    out_frame["Y_to_data"] = out_frame.Y_pb_per_GeV / out_frame.CS
    out_frame.to_csv(out / "lhcb7_y_full_minus_res.csv", index=False)
    rel_unc_full = out_frame.full_unc_pb_per_GeV / out_frame.full_pb_per_GeV.abs()
    rel_unc_y = out_frame.Y_unc_pb_per_GeV / np.maximum(out_frame.Y_pb_per_GeV.abs(), 1e-300)
    status = {
        "status": "isolated_lhcb7_y_full_minus_res_complete_not_production",
        "identity": "Y=(RES+CT+VJ)-RES; same fiducial card and cuts",
        "calls_per_vegas_component": int(args.calls), "row_count": 6, "g1_GeV2": float(args.g1), "expcreg": float(args.expcreg),
        "checks": {"all_finite": bool(np.isfinite(out_frame[["full_pb_per_GeV", "RES_pb_per_GeV", "Y_pb_per_GeV"]].to_numpy(float)).all()),
                    "all_full_positive": bool((out_frame.full_pb_per_GeV > 0).all()),
                    "full_mean_relative_mc_uncertainty": float(rel_unc_full.mean()),
                    "full_max_relative_mc_uncertainty": float(rel_unc_full.max()),
                    "Y_mean_relative_subtraction_uncertainty": float(rel_unc_y.replace([np.inf, -np.inf], np.nan).dropna().mean()),
                    "Y_max_relative_subtraction_uncertainty": float(rel_unc_y.replace([np.inf, -np.inf], np.nan).dropna().max()),
                    "positive_y_count": int((out_frame.Y_pb_per_GeV > 0).sum()),
                    "negative_y_count": int((out_frame.Y_pb_per_GeV < 0).sum())},
        "artifact_csv": str(out / "lhcb7_y_full_minus_res.csv"), "frozen_baseline_unchanged": True,
        "production_outputs_modified": False, "promotion_authorized": False,
    }
    (out / "lhcb7_y_full_minus_res_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__": main()
