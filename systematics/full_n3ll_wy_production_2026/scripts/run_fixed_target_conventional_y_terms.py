#!/usr/bin/env python3
"""Evaluate RES/CT/VJ separately for fixed-target rows.

This is the missing perturbative ingredient for a coupled F_NP fit: the
trainer needs the observable-level conventional term ``Y=VJ+CT`` while its
long b-space W kernel remains a separate input.  The run is deliberately
candidate-local and writes only below ``full_n3ll_wy_production_2026``.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
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
DEFAULT_OUT = BASE / "reports/fixed_target_conventional_y_terms_expcreg2p0_100k"


def term_card(row: pd.Series, *, name: str, calls: int, seed: int, cores: int, expcreg: float, g1: float) -> str:
    dataset = str(row.dataset)
    half = qT_half_width(dataset)
    qlow = max(QT_DIAGNOSTIC_FLOOR_GEV, float(row.qT) - half)
    qhigh = float(row.qT) + half
    ylow, yhigh = sorted((float(row.y_Low), float(row.y_High)))
    card_row = row.copy()
    card_row["qT_low"] = qlow
    card_row["qT_high"] = qhigh
    text = full_card_text(
        card_row,
        output_name=name,
        pdf_set="NNPDF40_nnlo_as_01180",
        pdf_member=0,
        cores=cores,
        calls=calls,
        seed=seed,
    )
    z, a, _ = TARGETS[dataset]
    text = text.replace("makecuts = true", "makecuts = false", 1)
    text = text.replace("ih2          = -1", "ih2          = 1", 1)
    text = text.replace(
        "nproc        = 3",
        f"nproc        = 3\nnuclearpdf   = true\nZ1 = 1\nA1 = 1\nZ2 = {z:.12g}\nA2 = {a:.12g}",
        1,
    )
    text = text.replace("g1 = 0.0", f"g1 = {float(g1):.12g}", 1)
    text = re.sub(r"qt_bins = \[ [^\]]+ \]", f"qt_bins = [ {qlow:.12g} {qhigh:.12g} ]", text, count=1)
    text = re.sub(r"y_bins  = \[ [^\]]+ \]", f"y_bins  = [ {ylow:.12g} {yhigh:.12g} ]", text, count=1)
    text = re.sub(r"m_bins  = \[ [^\]]+ \]", f"m_bins  = [ {float(row.QM_Low):.12g} {float(row.QM_High):.12g} ]", text, count=1)
    text += "\n# Explicit fixed-target Eq. (3.3) observable conversion.\n"
    text += "dsdqt2 = true\ndsdxf = false\nedsdp3 = false\n"
    text += f"expcreg = {float(expcreg):.12g}\n"
    return text


def set_term(text: str, term: str) -> str:
    term = str(term).upper()
    if term not in {"RES", "CT", "VJ"}:
        raise ValueError(term)
    for key in ("doBORN", "doCT", "doVJ", "doVJREAL", "doVJVIRT"):
        text = re.sub(rf"(^\s*{key}\s*=\s*)true\b", rf"\1false", text, flags=re.MULTILINE)
    if term == "RES":
        text = re.sub(r"(^\s*doBORN\s*=\s*)false\b", r"\1true", text, flags=re.MULTILINE)
    elif term == "CT":
        text = re.sub(r"(^\s*doCT\s*=\s*)false\b", r"\1true", text, flags=re.MULTILINE)
    else:
        text = re.sub(r"(^\s*doVJ\s*=\s*)false\b", r"\1true", text, flags=re.MULTILINE)
        text = re.sub(r"(^\s*doVJREAL\s*=\s*)false\b", r"\1true", text, flags=re.MULTILINE)
        text = re.sub(r"(^\s*doVJVIRT\s*=\s*)false\b", r"\1true", text, flags=re.MULTILINE)
    return text


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    ap.add_argument("--rows", nargs="*", default=None)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--calls", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=20260870)
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--expcreg", type=float, default=2.0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    # ``full_card_text`` resolves the shared benchmark module by its stable
    # import name; load it once before constructing any cards.
    load_runner()
    out = Path(args.out).resolve()
    cards, logs = out / "cards", out / "logs"
    cards.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(p for p in ("/home/dustin/miniforge3/envs/pdf-fit/lib", str(DYROOT / "lib"), env.get("LD_LIBRARY_PATH", "")) if p)
    selected: list[dict] = []
    wanted = set(args.rows or [])
    for dataset in args.datasets:
        frame = resolve_rows(dataset)
        if wanted:
            frame = frame[frame.row_id.astype(str).isin(wanted)]
        selected.extend(frame.to_dict("records"))
    if args.max_rows is not None:
        selected = selected[: int(args.max_rows)]
    if not selected:
        raise RuntimeError("no rows selected")

    records: list[dict] = []
    for index, raw in enumerate(selected):
        row = pd.Series(raw)
        rid = str(row.row_id)
        values: dict[str, dict[str, float]] = {}
        for term in ("RES", "CT", "VJ"):
            stem = f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', rid)}_n3ll_nnlo_{term.lower()}_expcreg_{str(args.expcreg).replace('.', 'p')}_seed_{args.seed + index}"
            card = cards / f"{stem}.in"
            log = logs / f"{stem}.log"
            table = DYROOT / f"{stem}.txt"
            if args.force:
                table.unlink(missing_ok=True)
            if not table.exists():
                card.write_text(set_term(term_card(row, name=stem, calls=args.calls, seed=args.seed + index, cores=args.cores, expcreg=args.expcreg, g1=1.017), term))
                with log.open("w") as handle:
                    subprocess.run([str(DYTURBO), str(card)], cwd=DYROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True, timeout=int(args.timeout))
            value, uncertainty = load_runner().parse_first_value(table)
            values[term] = {"raw_fb_per_bin": float(value), "raw_unc_fb_per_bin": float(uncertainty)}
        z, a, target_status = TARGETS[str(row.dataset)]
        half = qT_half_width(str(row.dataset))
        qlow, qhigh = max(QT_DIAGNOSTIC_FLOOR_GEV, float(row.qT) - half), float(row.qT) + half
        dq2 = qhigh * qhigh - qlow * qlow
        ylow, yhigh = sorted((float(row.y_Low), float(row.y_High)))
        dy = yhigh - ylow
        def convert(v: float) -> float:
            return v / (1000.0 * a * np.pi * dq2 * dy) / float(row.PreFactor)
        y_raw = values["CT"]["raw_fb_per_bin"] + values["VJ"]["raw_fb_per_bin"]
        y_unc = np.hypot(values["CT"]["raw_unc_fb_per_bin"], values["VJ"]["raw_unc_fb_per_bin"])
        records.append({
            "dataset": str(row.dataset), "row_id": rid,
            "qT_low": qlow, "qT_high": qhigh, "y_low": ylow, "y_high": yhigh,
            "QM_Low": float(row.QM_Low), "QM_High": float(row.QM_High),
            "target_Z": z, "target_A": a, "target_status": target_status,
            "RES_raw_fb_per_bin": values["RES"]["raw_fb_per_bin"],
            "RES_unc_fb_per_bin": values["RES"]["raw_unc_fb_per_bin"],
            "CT_raw_fb_per_bin": values["CT"]["raw_fb_per_bin"],
            "CT_unc_fb_per_bin": values["CT"]["raw_unc_fb_per_bin"],
            "VJ_raw_fb_per_bin": values["VJ"]["raw_fb_per_bin"],
            "VJ_unc_fb_per_bin": values["VJ"]["raw_unc_fb_per_bin"],
            "Y_raw_fb_per_bin": y_raw, "Y_unc_fb_per_bin": y_unc,
            "Y_CS": convert(y_raw), "Y_CS_unc": convert(y_unc),
            "RES_CS": convert(values["RES"]["raw_fb_per_bin"]),
            "cards": str(cards), "logs": str(logs),
        })
        pd.DataFrame(records).to_csv(out / "fixed_target_conventional_y_terms.csv", index=False)
        print(json.dumps(records[-1]), flush=True)
    frame = pd.DataFrame(records)
    status = {
        "status": "isolated_fixed_target_conventional_y_term_grid_complete_not_production",
        "formula": "W=RES; ASY=-CT; FO=VJ; Y=FO-ASY=VJ+CT",
        "row_count": int(len(frame)), "calls_per_vegas_component": int(args.calls),
        "expcreg": float(args.expcreg), "datasets": list(args.datasets),
        "checks": {
            "all_finite": bool(np.isfinite(frame[["RES_CS", "Y_CS"]].to_numpy(float)).all()),
            "all_y_finite": bool(np.isfinite(frame.Y_CS.to_numpy(float)).all()),
            "positive_y_count": int((frame.Y_CS > 0).sum()),
            "negative_y_count": int((frame.Y_CS < 0).sum()),
        },
        "artifact_csv": str(out / "fixed_target_conventional_y_terms.csv"),
        "frozen_baseline_unchanged": True, "production_outputs_modified": False,
        "promotion_authorized": False,
    }
    (out / "fixed_target_conventional_y_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
