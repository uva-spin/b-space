#!/usr/bin/env python3
"""Run DYTurbo full unprimed N3LL+NNLO W+Y on the 24 Tevatron boundary rows.

This is an isolated observable-level validation of the complete external
engine route.  It deliberately does not insert the fitted DNN F_NP: the
result is a perturbative W+Y oracle against which the eventual candidate
backend must be compared.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import argparse
from pathlib import Path

import pandas as pd

from run_dyturbo_full_n3ll_nnlo_probe import DYTURBO, DYROOT, OUT as PROBE_OUT, full_card_text, load_runner


SYSTEMATICS = Path(__file__).resolve().parents[2]
PROJECT = SYSTEMATICS.parent
BOUNDARY = SYSTEMATICS / "high_qt_direct_production_benchmark/experimental_unitary_transition/summaries/unitary_smootherstep_v1_nlo_24row/nlo_frozen_unitary_pulls.csv"
DATA_ROOT = PROJECT / "Data/v23a_tevatron_plus_lhcb7_fiducial_candidate"
OUT = SYSTEMATICS / "full_n3ll_wy_production_2026/reports/dyturbo_full_n3ll_nnlo_boundary"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g1", type=float, default=0.0)
    parser.add_argument("--calls", type=int, default=3_000_000)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    runner = load_runner()
    boundary = pd.read_csv(BOUNDARY)
    row_ids = boundary["row_id"].astype(str).tolist()
    if len(row_ids) != 24 or len(set(row_ids)) != 24:
        raise RuntimeError("boundary source is not the expected 24 unique rows")
    pieces = [pd.read_csv(DATA_ROOT / f"{ds}.csv") for ds in sorted(boundary["dataset"].astype(str).unique())]
    data = pd.concat(pieces, ignore_index=True)
    selected = data[data["row_id"].isin(row_ids)].copy()
    if len(selected) != 24:
        raise RuntimeError(f"selected {len(selected)} rows, expected 24")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    selected.sort_values(["dataset", "row_id"]).to_csv(out / "tevatron_boundary_input.csv", index=False)

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        p for p in ("/home/dustin/miniforge3/envs/pdf-fit/lib", str(DYROOT / "lib"), env.get("LD_LIBRARY_PATH", "")) if p
    )
    records = []
    calls = int(args.calls)
    for _, row in selected.iterrows():
        tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(row.row_id))
        g1_tag = f"{args.g1:g}".replace(".", "p").replace("-", "m")
        name = f"{tag}_full_n3ll_nnlo_g1_{g1_tag}"
        card = out / "cards" / f"{name}.in"
        log = out / "logs" / f"{name}.log"
        table = DYROOT / f"{name}.txt"
        card.parent.mkdir(parents=True, exist_ok=True)
        log.parent.mkdir(parents=True, exist_ok=True)
        card_text = full_card_text(row, output_name=name, pdf_set="NNPDF40_nnlo_as_01180", pdf_member=0, cores=8, calls=calls)
        card_text = card_text.replace("g1 = 0.0", f"g1 = {args.g1:.12g}", 1)
        card.write_text(card_text)
        if table.exists():
            table.unlink()
        with log.open("w") as handle:
            subprocess.run([str(DYTURBO), str(card)], cwd=DYROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True, timeout=1800)
        value, uncertainty = runner.parse_first_value(table)
        width = float(row.qT_high - row.qT_low)
        records.append({
            "dataset": row.dataset,
            "row_id": row.row_id,
            "qT_low": float(row.qT_low),
            "qT_high": float(row.qT_high),
            "data_pb_per_GeV": float(row.CS),
            "data_unc_pb_per_GeV": float(row.error),
            "raw_full_wy_fb_per_bin": value,
            "raw_full_wy_unc_fb_per_bin": uncertainty,
            "full_wy_pb_per_GeV": value / width / 1000.0,
            "full_wy_unc_pb_per_GeV": uncertainty / width / 1000.0,
            "full_wy_to_data_ratio": value / width / 1000.0 / float(row.CS),
            "card": str(card),
            "log": str(log),
        })
        pd.DataFrame(records).to_csv(out / "full_wy_boundary_summary.csv", index=False)

    result = pd.DataFrame(records)
    result.to_csv(out / "full_wy_boundary_summary.csv", index=False)
    rel_unc = result["raw_full_wy_unc_fb_per_bin"] / result["raw_full_wy_fb_per_bin"]
    status = {
        "status": "isolated_tevatron_24row_full_unprimed_n3ll_nnlo_wy_oracle_passed",
        "engine": str(DYTURBO),
        "order": 3,
        "primed": False,
        "enabled_terms": ["resummation_W_N3LL", "counterterm_ASY_NNLO", "VJ_NNLO"],
        "row_count": len(result),
        "calls_per_vegas_component": calls,
        "g1_GeV2": float(args.g1),
        "checks": {
            "all_finite": bool(result[["full_wy_pb_per_GeV", "full_wy_unc_pb_per_GeV"]].notna().all().all()),
            "all_positive": bool((result["full_wy_pb_per_GeV"] > 0).all()),
            "mean_relative_mc_uncertainty": float(rel_unc.mean()),
            "max_relative_mc_uncertainty": float(rel_unc.max()),
            "full_wy_to_data_ratio_median": float(result["full_wy_to_data_ratio"].median()),
            "full_wy_to_data_ratio_min": float(result["full_wy_to_data_ratio"].min()),
            "full_wy_to_data_ratio_max": float(result["full_wy_to_data_ratio"].max()),
        },
        "meaning": "external full W+Y observable oracle; no custom fitted F_NP and no production authorization",
        "frozen_baseline_unchanged": True,
        "production_outputs_modified": False,
    }
    (out / "boundary_full_wy_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
