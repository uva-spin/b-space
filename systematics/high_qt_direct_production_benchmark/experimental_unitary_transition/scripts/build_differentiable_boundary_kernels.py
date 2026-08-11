#!/usr/bin/env python3
"""Build/resume differentiable b-space kernels for the 24 added Tevatron rows."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from systematics.high_qt_direct_production_benchmark.experimental_matched_y.backend.exact_bin_asymptotic import (
    _gauss_interval,
    node_row,
    rapidity_interval,
)
from systematics.high_qt_direct_production_benchmark.experimental_matched_y.scripts.run_asymptotic_pilot import (
    load_backend,
    production_cfg,
)


BASE = ROOT / "systematics/high_qt_direct_production_benchmark/experimental_unitary_transition"
ROW_LIST = BASE / "outputs/unitary_smootherstep_v1_nodes_nb640_nqt2_ny2_simpson/tevatron_rows.csv"
DATA = ROOT / "Data/v23a_tevatron_plus_lhcb7_fiducial_candidate"
OUT = BASE / "outputs/unitary_smootherstep_v1_differentiable_kernels_nb640_nqt2_ny2"
_STATE: dict[str, object] = {}


def initialize_worker(n_b: int) -> None:
    backend = load_backend()
    cfg = production_cfg(backend, n_b=n_b)
    pdf = backend.LHAPDFProvider("NNPDF40_nnlo_as_01180", 0, use_toy_pdf=False)
    b_grid = np.asarray(backend.make_b_grid(cfg), dtype=float)
    _STATE.update(backend=backend, cfg=cfg, pdf=pdf, b_grid=b_grid)


def safe_tag(row_id: str) -> str:
    return row_id.lower().replace(":", "_")


def build_one(payload: dict) -> dict:
    row = pd.Series(payload)
    row_id = str(row.row_id)
    target = OUT / "rows" / f"{safe_tag(row_id)}.npz"
    if target.exists():
        with np.load(target) as cached:
            if str(cached["row_id"]) == row_id and cached["kernels"].shape == (2, 2, 640):
                return {"row_id": row_id, "status": "resumed", "path": str(target), "elapsed_seconds": 0.0}

    started = time.monotonic()
    backend = _STATE["backend"]
    cfg = _STATE["cfg"]
    pdf = _STATE["pdf"]
    b_grid = _STATE["b_grid"]
    qt_nodes, qt_weights = _gauss_interval(float(row.qT_low), float(row.qT_high), 2)
    y_low, y_high = rapidity_interval(row)
    y_nodes, y_weights = _gauss_interval(y_low, y_high, 2)
    kernels = np.empty((2, 2, len(b_grid)), dtype=np.float64)
    x1_nodes = np.empty((2, 2), dtype=np.float64)
    x2_nodes = np.empty((2, 2), dtype=np.float64)
    for iq, qt in enumerate(qt_nodes):
        for iy, y in enumerate(y_nodes):
            current = node_row(row, qT=float(qt), y=float(y))
            values = np.asarray(backend.wpert_cs_for_row(current, b_grid, pdf, cfg), dtype=float)
            rapidity_factor = float(backend._tevatron_rapidity_factor(current))
            if not math.isfinite(rapidity_factor) or rapidity_factor <= 0.0:
                raise FloatingPointError(f"invalid rapidity factor for {row_id}")
            values = values / rapidity_factor
            if values.shape != b_grid.shape or not np.isfinite(values).all():
                raise FloatingPointError(f"invalid kernel for {row_id}")
            kernels[iq, iy] = values
            x1_nodes[iq, iy] = float(current.x1)
            x2_nodes[iq, iy] = float(current.x2)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        row_id=np.asarray(row_id), dataset=np.asarray(str(row.dataset)),
        qT_low=float(row.qT_low), qT_high=float(row.qT_high), QM=float(row.QM),
        b_grid=b_grid, qt_nodes=qt_nodes, qt_weights=qt_weights,
        y_nodes=y_nodes, y_weights=y_weights, x1_nodes=x1_nodes, x2_nodes=x2_nodes,
        kernels=kernels,
    )
    return {
        "row_id": row_id, "status": "built", "path": str(target),
        "elapsed_seconds": time.monotonic() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    requested = pd.read_csv(ROW_LIST)[["dataset", "row_id"]]
    payloads = []
    for dataset, group in requested.groupby("dataset", sort=False):
        data = pd.read_csv(DATA / f"{dataset}.csv")
        selected = data.loc[data.row_id.isin(group.row_id)]
        if len(selected) != len(group):
            raise RuntimeError(f"row coverage mismatch for {dataset}")
        payloads.extend(selected.to_dict(orient="records"))
    records = []
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=initialize_worker, initargs=(640,)
    ) as pool:
        futures = {pool.submit(build_one, payload): payload["row_id"] for payload in payloads}
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(json.dumps(record), flush=True)
    records = sorted(records, key=lambda item: item["row_id"])
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(OUT / "kernel_manifest.csv", index=False)
    status = {
        "status": "experimental_unitary_transition_not_production",
        "tag": OUT.name,
        "requested_row_count": int(len(requested)),
        "kernel_row_count": int(len(records)),
        "all_rows_complete": bool(len(records) == len(requested)),
        "n_b": 640, "n_qT": 2, "n_y": 2,
        "integration_rule": "simpson",
        "rapidity_correction": "inclusive backend approximation removed at every node",
        "differentiable_with_respect_to_fnp": True,
        "validation_pass": False,
        "full_fnp_refit_authorized": False,
        "next_gate": "reconstruct the frozen-FNP 24-row W predictions from the saved kernels",
    }
    (OUT / "campaign_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
