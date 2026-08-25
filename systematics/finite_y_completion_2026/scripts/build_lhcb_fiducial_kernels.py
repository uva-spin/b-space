#!/usr/bin/env python3
"""Build isolated LHCb fiducial W kernels using the DYTurbo node grid."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
GRID = ROOT / "systematics/finite_y_completion_2026/reports/lhcb_node_acceptance_grid_production_candidate/lhcb_node_acceptance.csv"
DATA = ROOT / "Data/v23a_tevatron_plus_lhcb7_fiducial_candidate/LHCb_7.csv"
BACKEND = ROOT / "v23/backends/bt_internal_css_backend_v22_tevatron.py"
PILOT = ROOT / "systematics/high_qt_direct_production_benchmark/experimental_matched_y/scripts/run_asymptotic_pilot.py"
OUT_DEFAULT = ROOT / "systematics/finite_y_completion_2026/reports/lhcb_fiducial_w_kernels_nb640"
_STATE: dict[str, object] = {}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def initialize_worker(n_b: int) -> None:
    import torch
    torch.set_num_threads(1)
    pilot = load_module("lhcb_kernel_pilot", PILOT)
    backend = pilot.load_backend()
    cfg = pilot.production_cfg(backend, n_b=n_b)
    pdf = backend.LHAPDFProvider("NNPDF40_nnlo_as_01180", 0, use_toy_pdf=False)
    b_grid = np.asarray(backend.make_b_grid(cfg), dtype=float)
    _STATE.update(backend=backend, cfg=cfg, pdf=pdf, b_grid=b_grid)


def build_one(payload: dict) -> dict:
    row = pd.Series(payload["row"])
    grid = pd.DataFrame(payload["grid"])
    out = Path(payload["out"])
    row_id = str(row.row_id)
    safe = row_id.lower().replace(":", "_")
    target = out / "rows" / f"{safe}.npz"
    if target.exists():
        with np.load(target) as saved:
            if str(saved["row_id"]) == row_id:
                return {"row_id": row_id, "status": "resumed", "path": str(target)}
    backend = _STATE["backend"]
    cfg = _STATE["cfg"]
    pdf = _STATE["pdf"]
    b_grid = _STATE["b_grid"]
    q_values = sorted(grid.qT.unique())
    y_values = sorted(grid.y.unique())
    q_index = {value: i for i, value in enumerate(q_values)}
    y_index = {value: i for i, value in enumerate(y_values)}
    kernels = np.empty((len(q_values), len(y_values), len(b_grid)), dtype=float)
    acceptance = np.empty((len(q_values), len(y_values)), dtype=float)
    q_weights = np.empty(len(q_values), dtype=float)
    y_weights = np.empty(len(y_values), dtype=float)
    x1_nodes = np.empty((len(q_values), len(y_values)), dtype=float)
    x2_nodes = np.empty((len(q_values), len(y_values)), dtype=float)
    started = time.monotonic()
    tau = float(row.QM) / float(row.SqrtS)
    for item in grid.to_dict(orient="records"):
        iq, iy = q_index[item["qT"]], y_index[item["y"]]
        q_weights[iq] = float(item["q_weight"])
        y_weights[iy] = float(item["y_weight"])
        y = float(item["y"])
        current = row.copy()
        current["qT"] = float(item["qT"])
        current["y"] = y
        current["x1"] = tau * np.exp(y)
        current["x2"] = tau * np.exp(-y)
        # Make the pp backend evaluate this node rather than reintegrating the
        # original rapidity bin.  The finite bin average is reconstructed by
        # the saved qT/y quadrature weights.
        current["qT_low"] = np.nan
        current["qT_high"] = np.nan
        current["qT_bin_width"] = np.nan
        current["y_Low"] = y
        current["y_High"] = y
        values = np.asarray(backend.wpert_cs_for_row(current, b_grid, pdf, cfg), dtype=float)
        if values.shape != b_grid.shape or not np.isfinite(values).all():
            raise FloatingPointError(f"invalid LHCb W kernel at {row_id} q={item['qT']} y={y}")
        kernels[iq, iy] = values * float(item["acceptance"])
        acceptance[iq, iy] = float(item["acceptance"])
        x1_nodes[iq, iy] = current["x1"]
        x2_nodes[iq, iy] = current["x2"]
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target, row_id=np.asarray(row_id), dataset=np.asarray(str(row.dataset)),
        qT_nodes=np.asarray(q_values), y_nodes=np.asarray(y_values),
        qT_weights=q_weights, y_weights=y_weights, b_grid=b_grid,
        kernels=kernels, acceptance=acceptance,
        x1_nodes=x1_nodes, x2_nodes=x2_nodes,
    )
    return {"row_id": row_id, "status": "built", "path": str(target),
            "elapsed_seconds": time.monotonic() - started,
            "min_acceptance": float(np.min(acceptance)),
            "max_acceptance": float(np.max(acceptance))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", default=["LHCb_7:10", "LHCb_7:11", "LHCb_7:12", "LHCb_7:13"])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--n-b", type=int, default=640)
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    args = ap.parse_args()
    grid = pd.read_csv(GRID)
    data = pd.read_csv(DATA)
    selected = data[data.row_id.isin(args.rows)].copy()
    if len(selected) != len(args.rows):
        raise SystemExit("requested LHCb rows are missing")
    payloads = []
    for _, row in selected.iterrows():
        sub = grid[grid.row_id.eq(row.row_id)].copy()
        if sub.empty:
            raise SystemExit(f"missing acceptance grid row {row.row_id}")
        payloads.append({"row": row.to_dict(), "grid": sub.to_dict(orient="records"), "out": args.out})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records = []
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=initialize_worker,
                             initargs=(args.n_b,)) as pool:
        futures = {pool.submit(build_one, payload): payload["row"]["row_id"] for payload in payloads}
        for future in as_completed(futures):
            result = future.result()
            records.append(result)
            print(json.dumps(result), flush=True)
    records = sorted(records, key=lambda item: item["row_id"])
    pd.DataFrame(records).to_csv(out / "kernel_manifest.csv", index=False)
    status = {
        "status": "isolated_lhcb_fiducial_w_kernel_campaign_complete",
        "rows": list(args.rows), "n_b": args.n_b,
        "all_rows_complete": bool(all(item["status"] in {"built", "resumed"} for item in records)),
        "fiducial_acceptance_source": str(GRID),
        "backend_source": str(BACKEND),
        "production_outputs_modified": False,
        "next_step": "Recompute lambda=1 endpoint W and unitary matched predictions using these fiducial kernels and DYTurbo NLO bin endpoints.",
    }
    (out / "campaign_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
