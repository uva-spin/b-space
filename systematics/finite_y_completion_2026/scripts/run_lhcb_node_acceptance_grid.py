#!/usr/bin/env python3
"""Build an isolated DYTurbo node-level LHCb fiducial-acceptance grid.

The ratio is evaluated at the explicit qT/y quadrature nodes used by the
finite-Y kernel campaign.  DYTurbo supplies the fiducial NLO V+jet numerator
and an otherwise identical no-lepton-cut denominator.  Outputs live only
under ``finite_y_completion_2026``; this script never edits production data.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "Data/v23a_tevatron_plus_lhcb7_fiducial_candidate/LHCb_7.csv"
DY_SCRIPT = ROOT / "systematics/finite_y_tail_benchmark/scripts/run_lhcb7_dyturbo_benchmark.py"
DY_ROOT = Path("/home/dustin/src/dyturbo-1.4.2")
DY_EXE = DY_ROOT / "bin/dyturbo"
CONDA_LIB = Path("/home/dustin/miniforge3/envs/pdf-fit/lib")
DEFAULT_ROWS = ("LHCb_7:6", "LHCb_7:8", "LHCb_7:9", "LHCb_7:11")


def load_runner():
    spec = importlib.util.spec_from_file_location("lhcb_dyturbo_runner", DY_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(DY_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def node_bins(row: pd.Series, q_order: int, y_order: int) -> list[dict]:
    # Match the finite-Y campaign's Gauss-Legendre node locations.  The small
    # widths are only a local acceptance probe; numerator/denominator widths
    # cancel in the ratio.
    qt_nodes, qt_weights = np.polynomial.legendre.leggauss(q_order)
    y_nodes, y_weights = np.polynomial.legendre.leggauss(y_order)
    qlo, qhi = float(row.qT_low), float(row.qT_high)
    ylo, yhi = float(row.y_Low), float(row.y_High)
    q_nodes = 0.5 * (qhi + qlo) + 0.5 * (qhi - qlo) * qt_nodes
    y_nodes = 0.5 * (yhi + ylo) + 0.5 * (yhi - ylo) * y_nodes
    out = []
    for iq, q in enumerate(q_nodes):
        q_width = max(0.02, min(0.20, 0.01 * float(q)))
        for iy, y in enumerate(y_nodes):
            y_width = 0.02
            out.append({
                "node": f"q{iq}y{iy}", "qT": float(q), "y": float(y),
                "q_weight": float(qt_weights[iq]),
                "y_weight": float(y_weights[iy]),
                "qT_low": float(max(0.001, q - q_width)),
                "qT_high": float(q + q_width),
                "y_low": float(y - y_width), "y_high": float(y + y_width),
            })
    return out


def card_with_mode(runner, row: pd.Series, node: dict, output_name: str,
                   *, fiducial: bool, cores: int, seed: int,
                   qcd_order: int, vj_calls: int) -> str:
    probe = row.copy()
    probe["qT_low"] = node["qT_low"]
    probe["qT_high"] = node["qT_high"]
    probe["y_Low"] = node["y_low"]
    probe["y_High"] = node["y_high"]
    text = runner.card_text(
        probe, output_name=output_name, pdf_set="NNPDF40_nnlo_as_01180",
        pdf_member=0, cores=cores, seed=seed,
    )
    # The benchmark card is an NLO V+jet template with the real/virtual pieces
    # disabled.  For this isolated acceptance check, permit a genuine NNLO
    # numerator and denominator without touching the benchmark runner or any
    # production card.
    if qcd_order == 2:
        text = text.replace("order           = 1", "order           = 2")
        text = text.replace("doVJREAL = false", "doVJREAL = true")
        text = text.replace("doVJVIRT = false", "doVJVIRT = true")
        text = text.replace("vegasncallsVJREAL = 100000", f"vegasncallsVJREAL = {vj_calls}")
        text = text.replace("vegasncallsVJVIRT = 100000", f"vegasncallsVJVIRT = {vj_calls}")
        text = text.replace("VJquad = true", "VJquad = false")
        text = text.replace("intDimVJ   = 3", "intDimVJ   = -1")
    if not fiducial:
        text = text.replace("makecuts = true", "makecuts = false")
        text = text.replace("lcptcut = 20", "lcptcut = 0")
        text = text.replace("lcymin = 2.0", "lcymin = -1000")
        text = text.replace("lcymax = 4.5", "lcymax = 1000")
        text = text.replace("lfptcut = 20", "lfptcut = 0")
        text = text.replace("lfymin = 2.0", "lfymin = -1000")
        text = text.replace("lfymax = 4.5", "lfymax = 1000")
    return text


def run_one(runner, row: pd.Series, node: dict, out: Path, *, fiducial: bool,
            cores: int, seed: int, timeout: int, qcd_order: int,
            vj_calls: int) -> tuple[float, float, str]:
    mode = "fid" if fiducial else "inc"
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(row.row_id))
    stem = f"{tag}_{node['node']}_{mode}_dyturbo"
    card = out / "cards" / f"{stem}.in"
    log = out / "logs" / f"{stem}.log"
    table_dir = out / "tables"
    cached_table = table_dir / f"{stem}.txt"
    if cached_table.exists():
        value, unc = runner.parse_first_value(cached_table)
        return float(value), float(unc), str(cached_table)
    card.write_text(card_with_mode(runner, row, node, stem, fiducial=fiducial,
                                   cores=cores, seed=seed, qcd_order=qcd_order,
                                   vj_calls=vj_calls))
    src = DY_ROOT / f"{stem}.txt"
    dat = DY_ROOT / f"{stem}.dat"
    for path in (src, dat):
        if path.exists():
            path.unlink()
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        [str(CONDA_LIB), str(DY_ROOT / "lib"), env.get("LD_LIBRARY_PATH", "")]
    )
    with log.open("w") as handle:
        subprocess.run([str(DY_EXE), str(card.resolve())], cwd=DY_ROOT,
                       env=env, stdout=handle, stderr=subprocess.STDOUT,
                       check=True, timeout=timeout)
    value, unc = runner.parse_first_value(src)
    dst = table_dir / src.name
    src.replace(dst)
    if dat.exists():
        dat.replace(table_dir / dat.name)
    return float(value), float(unc), str(dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", default=list(DEFAULT_ROWS))
    ap.add_argument("--order", type=int, default=2,
                    help="Set both qT and rapidity node orders")
    ap.add_argument("--q-order", type=int, help="Override qT node order")
    ap.add_argument("--y-order", type=int, help="Override rapidity node order")
    ap.add_argument("--out", default=str(ROOT / "systematics/finite_y_completion_2026/reports/lhcb_node_acceptance_grid"))
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--seed", type=int, default=20260817)
    ap.add_argument("--qcd-order", type=int, choices=(1, 2), default=1,
                    help="DYTurbo fixed-order: 1=NLO template, 2=NNLO real+virtual")
    ap.add_argument("--vj-calls", type=int, default=100000,
                    help="NNLO real/virtual Vegas calls per node; ignored at NLO")
    args = ap.parse_args()
    q_order = args.q_order if args.q_order is not None else args.order
    y_order = args.y_order if args.y_order is not None else args.order
    if q_order < 2 or y_order < 2:
        raise SystemExit("node orders must be at least 2")
    runner = load_runner()
    data = pd.read_csv(DATA)
    selected = data[data.row_id.isin(args.rows)].copy()
    missing = sorted(set(args.rows) - set(selected.row_id))
    if missing:
        raise SystemExit(f"missing rows: {missing}")
    out = Path(args.out)
    for name in ("cards", "logs", "tables"):
        (out / name).mkdir(parents=True, exist_ok=True)
    records = []
    for _, row in selected.iterrows():
        for node in node_bins(row, q_order, y_order):
            fid, fid_unc, fid_path = run_one(
                runner, row, node, out, fiducial=True, cores=args.cores,
                seed=args.seed, timeout=args.timeout, qcd_order=args.qcd_order,
                vj_calls=args.vj_calls,
            )
            inc, inc_unc, inc_path = run_one(
                runner, row, node, out, fiducial=False, cores=args.cores,
                seed=args.seed + 1, timeout=args.timeout, qcd_order=args.qcd_order,
                vj_calls=args.vj_calls,
            )
            records.append({
                "row_id": str(row.row_id), "node": node["node"],
                "qT": node["qT"], "y": node["y"],
                "q_weight": node["q_weight"], "y_weight": node["y_weight"],
                "qT_low": node["qT_low"], "qT_high": node["qT_high"],
                "y_low": node["y_low"], "y_high": node["y_high"],
                "fiducial_pb": fid, "fiducial_unc_pb": fid_unc,
                "inclusive_pb": inc, "inclusive_unc_pb": inc_unc,
                "acceptance": fid / inc if inc > 0.0 else (0.0 if fid == 0.0 else np.nan),
                "acceptance_unc": (fid / inc) * np.sqrt((fid_unc / fid) ** 2 + (inc_unc / inc) ** 2)
                if fid > 0.0 and inc > 0.0 else 0.0,
                "fiducial_table": fid_path, "inclusive_table": inc_path,
            })
            print(json.dumps(records[-1]), flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(out / "lhcb_node_acceptance.csv", index=False)
    all_nonnegative = bool(
        (frame.fiducial_pb.to_numpy(float) >= 0.0).all()
        and (frame.inclusive_pb.to_numpy(float) >= 0.0).all()
    )
    summary = {
        "status": ("isolated_lhcb_node_acceptance_grid_complete"
                   if all_nonnegative else
                   "isolated_lhcb_node_acceptance_grid_failed_numeric_gate"),
        "rows": list(args.rows), "qT_node_order": q_order,
        "rapidity_node_order": y_order,
        "qcd_order": args.qcd_order,
        "vj_calls": args.vj_calls if args.qcd_order == 2 else None,
        "node_count": int(len(frame)),
        "all_nonzero_nodes_finite_positive": bool(
            np.isfinite(frame.loc[frame.inclusive_pb > 0.0, ["acceptance", "acceptance_unc"]].to_numpy()).all()
            and (frame.loc[frame.inclusive_pb > 0.0, "fiducial_pb"].to_numpy() >= 0.0).all()
            and (frame.loc[frame.inclusive_pb > 0.0, "inclusive_pb"].to_numpy() > 0.0).all()
        ),
        "all_node_integrals_nonnegative": all_nonnegative,
        "zero_cross_section_node_count": int((frame.inclusive_pb == 0.0).sum()),
        "acceptance_min": float(frame.acceptance.min()),
        "acceptance_max": float(frame.acceptance.max()),
        "bin_weighted_acceptance": float(
            np.sum(frame.fiducial_pb * frame.q_weight * frame.y_weight)
            / np.sum(frame.inclusive_pb * frame.q_weight * frame.y_weight)
        ),
        "production_outputs_modified": False,
        "interpretation": "Node ratios are a candidate fiducial weight grid. They must be compared with the existing bin-level acceptance factors and checked for quadrature/bin-width stability before entering a finite-Y kernel campaign.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
