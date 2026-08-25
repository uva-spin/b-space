#!/usr/bin/env python3
"""Build a fixed-target Y diagnostic from matched full minus RES cards.

For fixed-target cards DYTurbo's standalone CT switch is not an additive
observable: it depends on the Born/resummation setup.  The stable row-level
identity is therefore evaluated directly as

    Y = (RES + CT + VJ) - RES,

which is algebraically the same ``FO_NNLO - ASY_NNLO`` construction when all
terms use the same card conventions.  The result is still candidate-side;
the target and fixed-y conversion assumptions remain explicit in the status.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from run_fixed_target_n3ll_nnlo_grid import (
    DATASETS,
    DYTURBO,
    DYROOT,
    TARGETS,
    full_card_text,
    load_runner,
    qT_half_width,
    resolve_rows,
    QT_DIAGNOSTIC_FLOOR_GEV,
)


BASE = Path(__file__).resolve().parents[1]
DEFAULT_OUT = BASE / "reports/fixed_target_y_full_minus_res_expcreg2p0_100k"


def card_for(row: dict, *, name: str, calls: int, seed: int, cores: int, expcreg: float, res_only: bool) -> str:
    s = pd.Series(row)
    dataset = str(s.dataset)
    half = qT_half_width(dataset)
    qlow = max(QT_DIAGNOSTIC_FLOOR_GEV, float(s.qT) - half)
    qhigh = float(s.qT) + half
    ylow, yhigh = sorted((float(s.y_Low), float(s.y_High)))
    card_row = s.copy()
    card_row["qT_low"] = qlow
    card_row["qT_high"] = qhigh
    text = full_card_text(card_row, output_name=name, pdf_set="NNPDF40_nnlo_as_01180", pdf_member=0,
                          cores=cores, calls=calls, seed=seed)
    z, a, _ = TARGETS[dataset]
    text = text.replace("makecuts = true", "makecuts = false", 1)
    text = text.replace("ih2          = -1", "ih2          = 1", 1)
    text = text.replace("nproc        = 3", f"nproc        = 3\nnuclearpdf   = true\nZ1 = 1\nA1 = 1\nZ2 = {z:.12g}\nA2 = {a:.12g}", 1)
    text = text.replace("g1 = 0.0", "g1 = 1.017", 1)
    text = re.sub(r"qt_bins = \[ [^\]]+ \]", f"qt_bins = [ {qlow:.12g} {qhigh:.12g} ]", text, count=1)
    text = re.sub(r"y_bins  = \[ [^\]]+ \]", f"y_bins  = [ {ylow:.12g} {yhigh:.12g} ]", text, count=1)
    text = re.sub(r"m_bins  = \[ [^\]]+ \]", f"m_bins  = [ {float(s.QM_Low):.12g} {float(s.QM_High):.12g} ]", text, count=1)
    text += "\n# Explicit fixed-target Eq. (3.3) observable conversion.\ndsdqt2 = true\ndsdxf = false\nedsdp3 = false\n"
    text += f"expcreg = {float(expcreg):.12g}\n"
    if res_only:
        for key in ("doCT", "doVJ", "doVJREAL", "doVJVIRT"):
            text = re.sub(rf"(^\s*{key}\s*=\s*)true\b", rf"\1false", text, flags=re.MULTILINE)
    return text


def parse_value(path: Path) -> tuple[float, float]:
    runner = load_runner()
    return tuple(float(x) for x in runner.parse_first_value(path))


def run_one(payload: tuple[dict, str, int, int, int, float, bool]) -> dict:
    row, out_str, calls, seed, cores, expcreg, force = payload
    out = Path(out_str)
    cards, logs = out / "cards", out / "logs"
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(p for p in ("/home/dustin/miniforge3/envs/pdf-fit/lib", str(DYROOT / "lib"), env.get("LD_LIBRARY_PATH", "")) if p)
    rid = str(row["row_id"])
    stem_base = f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', rid)}_fullminusres_n3ll_g1_1p017_expcreg_{str(expcreg).replace('.', 'p')}_seed_{seed}"
    values = {}
    for label, res_only in (("full", False), ("res", True)):
        stem = f"{stem_base}_{label}"
        card, log, table = cards / f"{stem}.in", logs / f"{stem}.log", DYROOT / f"{stem}.txt"
        if force:
            table.unlink(missing_ok=True)
        if not table.exists():
            card.write_text(card_for(row, name=stem, calls=calls, seed=seed, cores=cores, expcreg=expcreg, res_only=res_only))
            with log.open("w") as handle:
                subprocess.run([str(DYTURBO), str(card)], cwd=DYROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True, timeout=1800)
        values[label] = parse_value(table)
    dataset = str(row["dataset"])
    z, a, target_status = TARGETS[dataset]
    half = qT_half_width(dataset)
    qlow, qhigh = max(QT_DIAGNOSTIC_FLOOR_GEV, float(row["qT"]) - half), float(row["qT"]) + half
    dq2 = qhigh * qhigh - qlow * qlow
    ylow, yhigh = sorted((float(row["y_Low"]), float(row["y_High"])))
    dy = yhigh - ylow
    conv = 1.0 / (1000.0 * a * np.pi * dq2 * dy * float(row["PreFactor"]))
    full, full_unc = values["full"]
    res, res_unc = values["res"]
    y = full - res
    y_unc = float(np.hypot(full_unc, res_unc))
    return {
        "dataset": dataset, "row_id": rid, "qT_low": qlow, "qT_high": qhigh,
        "y_low": ylow, "y_high": yhigh, "QM_Low": float(row["QM_Low"]), "QM_High": float(row["QM_High"]),
        "target_Z": z, "target_A": a, "target_status": target_status,
        "full_raw_fb_per_bin": full, "full_unc_fb_per_bin": full_unc,
        "RES_raw_fb_per_bin": res, "RES_unc_fb_per_bin": res_unc,
        "Y_raw_fb_per_bin": y, "Y_unc_fb_per_bin": y_unc,
        "full_CS": full * conv, "RES_CS": res * conv, "Y_CS": y * conv, "Y_CS_unc": y_unc * conv,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    ap.add_argument("--rows", nargs="*", default=None)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--calls", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=20260880)
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--expcreg", type=float, default=2.0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    load_runner()
    out = Path(args.out).resolve()
    (out / "cards").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    wanted = set(args.rows or [])
    selected: list[dict] = []
    for dataset in args.datasets:
        frame = resolve_rows(dataset)
        if wanted:
            frame = frame[frame.row_id.astype(str).isin(wanted)]
        selected.extend(frame.to_dict("records"))
    if args.max_rows is not None:
        selected = selected[: int(args.max_rows)]
    if not selected:
        raise RuntimeError("no rows selected")
    payloads = [(r, str(out), int(args.calls), int(args.seed) + i, int(args.cores), float(args.expcreg), bool(args.force)) for i, r in enumerate(selected)]
    records: list[dict] = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [pool.submit(run_one, p) for p in payloads]
        for fut in as_completed(futures):
            records.append(fut.result())
            records.sort(key=lambda r: str(r["row_id"]))
            pd.DataFrame(records).to_csv(out / "fixed_target_y_full_minus_res.csv", index=False)
            print(json.dumps(records[-1]), flush=True)
    frame = pd.DataFrame(records)
    status = {
        "status": "isolated_fixed_target_y_full_minus_res_complete_not_production",
        "identity": "Y=(RES+CT+VJ)-RES; conventional FO_NNLO-ASY_NNLO equivalent when terms share the card",
        "row_count": int(len(frame)), "calls_per_vegas_component": int(args.calls), "workers": int(args.workers),
        "expcreg": float(args.expcreg), "datasets": list(args.datasets),
        "checks": {"all_finite": bool(np.isfinite(frame[["full_CS", "RES_CS", "Y_CS"]].to_numpy(float)).all()),
                    "all_full_positive": bool((frame.full_CS > 0).all()),
                    "positive_y_count": int((frame.Y_CS > 0).sum()),
                    "negative_y_count": int((frame.Y_CS < 0).sum()),
                    "median_full_CS": float(frame.full_CS.median()),
                    "median_Y_CS": float(frame.Y_CS.median())},
        "artifact_csv": str(out / "fixed_target_y_full_minus_res.csv"),
        "frozen_baseline_unchanged": True, "production_outputs_modified": False, "promotion_authorized": False,
    }
    (out / "fixed_target_y_full_minus_res_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
