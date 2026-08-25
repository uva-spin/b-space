#!/usr/bin/env python3
"""Run an isolated fixed-target unprimed N3LL+NNLO W+Y capability grid.

The fit-ready fixed-target tables store ``CS=A/PreFactor`` rather than the
Tevatron ``pb/GeV`` convention.  This diagnostic evaluates the published
fixed-y bins directly (``dsdxf=false``, ``dsdqt2=true``), converts the
integrated DYTurbo result to the invariant ``A`` using Eq. (3.3), and then
applies the row's ``PreFactor``.  The native DYTurbo ``dsdxf`` route is not
used because its xF restriction failed an identity test in the full W+Y path.

This is a diagnostic grid.  It does not modify the accepted fit, the frozen
lambda=1 package, or any production cache.  The E772 isoscalar target remains
marked as an assumption in the status file even when the numerical grid is
finite.
"""

from __future__ import annotations

import argparse
import importlib.util
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
BASE = SYSTEMATICS / "full_n3ll_wy_production_2026"
AUDIT_SCRIPT = BASE / "scripts/audit_353_candidate_scope.py"
CORE_OUTPUT = SYSTEMATICS / "collins_factorization_validity/outputs/rowidfix_stageFT_all_qmax0p20_lam0p50_central_s303/predictions.csv"
DATA_ROOT = PROJECT / "Data/v23a_fixed_target_plus_tevatron_absolute_fit_ready"
DEFAULT_OUT = BASE / "reports/fixed_target_n3ll_nnlo_grid"
DATASETS = ("E288_200", "E288_300", "E288_400", "E605", "E772")
TARGETS = {
    "E288_200": (4.0, 9.0121831, "Be candidate assumption"),
    "E288_300": (4.0, 9.0121831, "Be candidate assumption"),
    "E288_400": (4.0, 9.0121831, "Be candidate assumption"),
    "E605": (29.0, 63.546, "Cu candidate assumption"),
    "E772": (0.5, 1.0, "isoscalar diagnostic fallback"),
}

# DYTurbo's dsdqt2 resummed integrand is formally integrable at qT=0, but the
# external quadrature can overflow when a broad bin starts exactly at zero.
# Keep this diagnostic endpoint floor explicit and reproducible.
QT_DIAGNOSTIC_FLOOR_GEV = 0.02


def qT_half_width(dataset: str) -> float:
    return 0.125 if dataset == "E772" else 0.1


def resolve_rows(dataset: str) -> pd.DataFrame:
    """Resolve the 329-row authority selection against fit-ready source rows."""
    audit_spec = importlib.util.spec_from_file_location("scope_audit", AUDIT_SCRIPT)
    if audit_spec is None or audit_spec.loader is None:
        raise RuntimeError(AUDIT_SCRIPT)
    audit = importlib.util.module_from_spec(audit_spec)
    audit_spec.loader.exec_module(audit)
    core = pd.read_csv(CORE_OUTPUT)
    wanted = core[core.dataset.astype(str).eq(dataset)].copy()
    if wanted.empty:
        raise RuntimeError(f"no selected rows for {dataset}")
    return audit.resolve_core_rows(DATA_ROOT, dataset, wanted)


def make_card(row: pd.Series, *, name: str, g1: float, calls: int, seed: int, cores: int,
              expcreg: float | None = None) -> str:
    """Build a no-lepton-cut nuclear-target card from the common W+Y card."""
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
    text = re.sub(r"qt_bins = \[ [^\]]+ \]", f"qt_bins = [ {qlow:.12g} {qhigh:.12g} ]", text, count=1)
    text = re.sub(r"y_bins  = \[ [^\]]+ \]", f"y_bins  = [ {ylow:.12g} {yhigh:.12g} ]", text, count=1)
    text = re.sub(r"m_bins  = \[ [^\]]+ \]", f"m_bins  = [ {float(row.QM_Low):.12g} {float(row.QM_High):.12g} ]", text, count=1)
    text = text.replace("g1 = 0.0", f"g1 = {g1:.12g}", 1)
    # Native dsdxf failed an identity test against fixed-y integration in the
    # full W+Y path.  Use the published fixed-y bins and Eq. (3.3) instead.
    text += "\n# Explicit fixed-target Eq. (3.3) observable conversion.\n"
    text += "dsdqt2 = true\ndsdxf = false\nedsdp3 = false\n"
    if expcreg is not None:
        text += "# Isolated coefficient-exponentiation regularization audit; not production.\n"
        text += f"expcreg = {float(expcreg):.12g}\n"
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    ap.add_argument("--g1", type=float, default=1.017)
    ap.add_argument("--calls", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=20260860)
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--rows", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--expcreg", type=float, default=None,
                    help="isolated DYTurbo order-3 coefficient regularization override")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    out = Path(args.out).resolve()
    cards, logs = out / "cards", out / "logs"
    cards.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    common = load_runner()
    # The common module is already imported above; use its public parser.
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(p for p in ("/home/dustin/miniforge3/envs/pdf-fit/lib", str(DYROOT / "lib"), env.get("LD_LIBRARY_PATH", "")) if p)
    selected: list[pd.Series] = []
    for dataset in args.datasets:
        frame = resolve_rows(dataset)
        if args.rows:
            frame = frame[frame.row_id.astype(str).isin(set(args.rows))]
        selected.extend(frame.to_dict("records"))
    if args.max_rows is not None:
        selected = selected[: int(args.max_rows)]
    if not selected:
        raise RuntimeError("no fixed-target rows selected")
    records = []
    for index, raw in enumerate(selected):
        row = pd.Series(raw)
        tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(row.row_id))
        reg_tag = "default" if args.expcreg is None else f"expcreg_{str(float(args.expcreg)).replace('.', 'p')}"
        name = f"{tag}_full_n3ll_nnlo_g1_{str(args.g1).replace('.', 'p')}_{reg_tag}_seed_{args.seed}"
        card = cards / f"{name}.in"
        log = logs / f"{name}.log"
        table = DYROOT / f"{name}.txt"
        card.write_text(make_card(row, name=name, g1=args.g1, calls=args.calls, seed=args.seed + index, cores=args.cores,
                                  expcreg=args.expcreg))
        if args.force:
            table.unlink(missing_ok=True)
        if not table.exists():
            with log.open("w") as handle:
                subprocess.run([str(DYTURBO), str(card)], cwd=DYROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True, timeout=args.timeout)
        value, uncertainty = common.parse_first_value(table)
        z, a, target_status = TARGETS[str(row.dataset)]
        half = qT_half_width(str(row.dataset))
        qlow, qhigh = max(QT_DIAGNOSTIC_FLOOR_GEV, float(row.qT) - half), float(row.qT) + half
        dq2 = qhigh * qhigh - qlow * qlow
        ylow, yhigh = sorted((float(row.y_Low), float(row.y_High)))
        dy = yhigh - ylow
        prefactor = float(row.PreFactor)
        # DYTurbo returns fb after integrating dQ^2 dqT^2 dy for this card;
        # Eq. (3.3) gives A = I/[1000*A*pi*Delta(qT^2)*Delta(y)].
        pred_a = float(value) / (1000.0 * a * np.pi * dq2 * dy)
        pred_cs = pred_a / prefactor
        records.append({
            "dataset": str(row.dataset), "row_id": str(row.row_id),
            "qT_low": qlow, "qT_high": qhigh, "qT_bin_width_GeV": qhigh - qlow,
            "QM_Low": float(row.QM_Low), "QM_High": float(row.QM_High),
            "qT2_bin_width_GeV2": dq2, "y_low": ylow, "y_high": yhigh,
            "y_bin_width": dy, "PreFactor": prefactor,
            "data_A": float(row.A), "data_CS_A_over_PreFactor": float(row.CS),
            "data_error": float(row.error),
            "target_Z": z, "target_A": a, "target_status": target_status,
            "raw_full_wy_fb_per_bin_nucleus": float(value), "raw_unc_fb_per_bin_nucleus": float(uncertainty),
            "per_nucleon_full_wy_fb_per_bin": float(value / a),
            "per_nucleon_unc_fb_per_bin": float(uncertainty / a),
            "predicted_A": pred_a, "predicted_CS": pred_cs,
            "predicted_A_to_data": float(pred_a / float(row.A)),
            "predicted_CS_to_data": float(pred_cs / float(row.CS)),
            "card": str(card), "log": str(log), "table": str(table),
        })
        pd.DataFrame(records).to_csv(out / "fixed_target_full_wy_grid.csv", index=False)
        print(json.dumps(records[-1]), flush=True)
    result = pd.DataFrame(records)
    values = result.predicted_CS.to_numpy(float)
    status = {
        "status": "isolated_fixed_target_unprimed_n3ll_nnlo_wy_grid_complete_not_production",
        "engine": str(DYTURBO), "order": 3, "primed": False,
        "convention": "W=RES, ASY=-CT, FO=VJ, Y=FO-ASY=VJ+CT",
        "datasets": list(args.datasets), "row_count": int(len(result)),
        "calls_per_vegas_component": int(args.calls), "g1_GeV2": float(args.g1),
        "expcreg": args.expcreg,
        "checks": {
            "all_finite": bool(np.isfinite(values).all()),
            "all_positive": bool((values > 0).all()),
            "median_predicted_CS_to_data": float(result.predicted_CS_to_data.median()),
            "min_predicted_CS_to_data": float(result.predicted_CS_to_data.min()),
            "max_predicted_CS_to_data": float(result.predicted_CS_to_data.max()),
        },
        "observable_convention": "fixed-y Eq. (3.3): A=I/[1000*A*pi*Delta(qT^2)*Delta(y)], then CS=A/PreFactor",
        "qT_diagnostic_floor_GeV": QT_DIAGNOSTIC_FLOOR_GEV,
        "scope_limit": "E772 isoscalar fallback remains diagnostic; fixed-y rectangular approximation must be assessed before promotion",
        "artifact_csv": str(out / "fixed_target_full_wy_grid.csv"),
        "frozen_baseline_unchanged": True, "production_outputs_modified": False,
        "promotion_authorized": False,
    }
    (out / "fixed_target_grid_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
