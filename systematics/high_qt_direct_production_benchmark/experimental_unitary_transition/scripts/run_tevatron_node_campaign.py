#!/usr/bin/env python3
"""Run rapidity-corrected exact-node W and unitary profiles for Tier-1 Tevatron rows."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from systematics.high_qt_direct_production_benchmark.experimental_matched_y.backend.exact_bin_asymptotic import (
    integrate_exact_bin,
    make_resummed_w_point_evaluators,
)
from systematics.high_qt_direct_production_benchmark.experimental_matched_y.scripts.run_asymptotic_pilot import (
    load_backend,
    production_cfg,
)
from systematics.high_qt_direct_production_benchmark.experimental_matched_y.scripts.run_resummed_w_cancellation_pilot import (
    load_np_factor,
    make_pair_factor,
)
from systematics.high_qt_direct_production_benchmark.experimental_unitary_transition.backend.unitary_transition import (
    bin_averaged_profile,
    unitary_transition,
)

BASE = ROOT / "systematics/high_qt_direct_production_benchmark"
EXTERNAL = BASE / "summaries/tier1_boundary/central/external_pairs.csv"
HERE = BASE / "experimental_unitary_transition"
METRICS = ROOT / "systematics/collins_factorization_validity/outputs/rowidfix_stageFT_all_qmax0p20_lam0p50_central_s303/metrics.json"
PROFILES = (
    ("early_0p18_0p28", 0.18, 0.28),
    ("central_0p20_0p30", 0.20, 0.30),
    ("late_0p22_0p32", 0.22, 0.32),
)

_STATE = {}


def initialize_worker(n_b: int, n_qT: int, n_y: int, integration_rule: str) -> None:
    torch.set_num_threads(1)
    config = json.loads(METRICS.read_text())["config"]
    device = torch.device("cpu")
    dtype = torch.float32
    backend = load_backend()
    cfg = production_cfg(backend, n_b=n_b)
    pdf = backend.LHAPDFProvider("NNPDF40_nnlo_as_01180", 0, use_toy_pdf=False)
    np_factor = load_np_factor(config, device=device, dtype=dtype)
    _, fitted = make_resummed_w_point_evaluators(
        backend=backend, pdf=pdf, cfg=cfg,
        np_pair_factor=make_pair_factor(np_factor, device=device, dtype=dtype),
        remove_inclusive_rapidity_approximation=True,
        integration_rule=integration_rule,
    )
    _STATE.update(fitted=fitted, n_qT=n_qT, n_y=n_y, n_b=n_b)


def evaluate_row(payload: dict) -> dict:
    row = pd.Series(payload)
    started = time.monotonic()
    result = integrate_exact_bin(
        row, point_evaluator=_STATE["fitted"],
        n_qT=_STATE["n_qT"], n_y=_STATE["n_y"],
    )
    return {
        "dataset": str(row.dataset), "row_id": str(row.row_id),
        "qT": float(row.qT), "qT_low": float(row.qT_low), "qT_high": float(row.qT_high),
        "QM": float(row.QM), "qT_over_Q": float(row.qT / row.QM),
        "w_fitted_pb_per_GeV": result.value_pb_per_GeV,
        "n_b": _STATE["n_b"], "n_qT": _STATE["n_qT"], "n_y": _STATE["n_y"],
        "elapsed_seconds": time.monotonic() - started,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-b", type=int, default=160)
    ap.add_argument("--n-qt", type=int, default=2)
    ap.add_argument("--n-y", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--integration-rule", choices=("trapezoid", "simpson"), default="trapezoid")
    ap.add_argument("--row", action="append", default=[])
    ap.add_argument("--out")
    args = ap.parse_args()
    external = pd.read_csv(EXTERNAL)
    external = external.loc[external.dataset.ne("LHCb_7")].copy()
    payloads = []
    if args.row:
        for dataset in sorted({row_id.split(":", 1)[0] for row_id in args.row}):
            data = pd.read_csv(ROOT / "Data/v23a_tevatron_plus_lhcb7_fiducial_candidate" / f"{dataset}.csv")
            requested = [row_id for row_id in args.row if row_id.startswith(f"{dataset}:")]
            selected = data.loc[data.row_id.isin(requested)].copy()
            if len(selected) != len(requested):
                raise RuntimeError(f"row coverage mismatch for {dataset}")
            payloads.extend(selected.to_dict(orient="records"))
    else:
        for dataset, group in external.groupby("dataset", sort=False):
            data = pd.read_csv(ROOT / "Data/v23a_tevatron_plus_lhcb7_fiducial_candidate" / f"{dataset}.csv")
            selected = data.loc[data.row_id.isin(group.row_id)].copy()
            if len(selected) != len(group):
                raise RuntimeError(f"row coverage mismatch for {dataset}")
            payloads.extend(selected.to_dict(orient="records"))

    rows = []
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=initialize_worker,
        initargs=(args.n_b, args.n_qt, args.n_y, args.integration_rule),
    ) as pool:
        futures = {pool.submit(evaluate_row, payload): payload["row_id"] for payload in payloads}
        for future in as_completed(futures):
            record = future.result()
            rows.append(record)
            print(json.dumps({"completed": record["row_id"], "w": record["w_fitted_pb_per_GeV"]}), flush=True)

    frame = pd.DataFrame(rows).sort_values(["dataset", "qT"]).reset_index(drop=True)
    frame = frame.merge(
        external[["row_id", "dyturbo_pb_per_GeV", "mcfm_pb_per_GeV"]], on="row_id", how="left", validate="one_to_one"
    )
    frame["external_fo_average_pb_per_GeV"] = 0.5 * (frame.dyturbo_pb_per_GeV + frame.mcfm_pb_per_GeV)
    for name, start, end in PROFILES:
        pcol = f"profile_{name}"
        ycol = f"unitary_{name}_pb_per_GeV"
        frame[pcol] = [bin_averaged_profile(a, b, q, r_start=start, r_end=end, n=32)
                       for a, b, q in zip(frame.qT_low, frame.qT_high, frame.QM)]
        if frame.external_fo_average_pb_per_GeV.notna().all():
            frame[ycol] = unitary_transition(
                frame.w_fitted_pb_per_GeV, frame.external_fo_average_pb_per_GeV, frame[pcol]
            )
        else:
            frame[ycol] = float("nan")
    suffix = "" if args.integration_rule == "trapezoid" else f"_{args.integration_rule}"
    if args.row:
        suffix += "_selected"
    out = (Path(args.out).resolve() if args.out else
           HERE / f"outputs/unitary_smootherstep_v1_nodes_nb{args.n_b}_nqt{args.n_qt}_ny{args.n_y}{suffix}")
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "tevatron_rows.csv", index=False)
    status = {
        "status": "experimental_unitary_transition_not_production",
        "tag": out.name,
        "row_count": len(frame),
        "integration_rule": args.integration_rule,
        "datasets": frame.groupby("dataset").size().to_dict(),
        "all_finite": bool(frame.select_dtypes("number").notna().all().all()),
        "rapidity_correction": "inclusive backend approximation removed before explicit y integration",
        "lhcb_included": False,
        "lhcb_blocker": "high-qT node-level fiducial acceptance unavailable",
        "b_grid_convergence_pass": False,
        "direct_production_approval_pass": False,
        "next_gate": "compare against doubled n_b campaign and audit continuity/FO convergence",
    }
    (out / "campaign_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
