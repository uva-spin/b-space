#!/usr/bin/env python3
"""Run isolated unprimed N3LL+NNLO W+Y scale variations on Tevatron bins.

This is a perturbative-accuracy diagnostic for the external DYTurbo candidate.
It reuses the exact conventional decomposition and Gaussian nonperturbative
candidate used by the central grid.  It writes only under the new campaign
directory and never touches the frozen production package.
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
DATASETS = ("CDF_RUN_1", "CDF_RUN_2", "D0_RUN_1")
DEFAULT_OUT = SYSTEMATICS / "full_n3ll_wy_production_2026/reports/tevatron_scale_variations_g1_1p024191"
VARIATIONS = ((0.5, 0.5), (0.5, 1.0), (1.0, 0.5), (1.0, 1.0), (1.0, 2.0), (2.0, 1.0), (2.0, 2.0))


def parse_grid(path: Path, *, first_edge: float, last_edge: float) -> pd.DataFrame:
    rows = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 4 or parts[0].startswith("#"):
            continue
        try:
            lo, hi, value, unc = map(float, parts[:4])
        except ValueError:
            continue
        if abs(lo - first_edge) < 1e-12 and abs(hi - last_edge) < 1e-12:
            continue
        rows.append({"qT_low": lo, "qT_high": hi, "raw_fb_per_bin": value, "raw_unc_fb_per_bin": unc})
    if not rows:
        raise RuntimeError(f"no table rows parsed from {path}")
    return pd.DataFrame(rows)


def tag(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--g1", type=float, default=1.0241911542738864)
    ap.add_argument("--calls", type=int, default=3_000_000)
    ap.add_argument("--seed", type=int, default=97531)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    runner = load_runner()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(p for p in ("/home/dustin/miniforge3/envs/pdf-fit/lib", str(DYROOT / "lib"), env.get("LD_LIBRARY_PATH", "")) if p)
    records = []
    unresolved_rows = []
    for mur, muf in VARIATIONS:
        scale_tag = f"mur{tag(mur)}_muf{tag(muf)}"
        for dataset in DATASETS:
            data = pd.read_csv(DATA_ROOT / f"{dataset}.csv").sort_values("qT_low").reset_index(drop=True)
            edges = [float(data.iloc[0].qT_low)] + [float(x) for x in data.qT_high]
            if len(data) > 1 and not np.allclose(data.qT_low.iloc[1:].to_numpy(float), data.qT_high.iloc[:-1].to_numpy(float), atol=1e-9, rtol=0):
                raise RuntimeError(f"non-contiguous bins for {dataset}")
            first = data.iloc[0]
            name = f"{dataset}_n3ll_nnlo_{scale_tag}_g1_{tag(args.g1)}"
            card = out / "cards" / f"{name}.in"
            log = out / "logs" / f"{name}.log"
            table = DYROOT / f"{name}.txt"
            card.parent.mkdir(parents=True, exist_ok=True)
            log.parent.mkdir(parents=True, exist_ok=True)
            text = full_card_text(first, output_name=name, pdf_set="NNPDF40_nnlo_as_01180", pdf_member=0, cores=8, calls=args.calls, seed=args.seed)
            text = text.replace("g1 = 0.0", f"g1 = {args.g1:.12g}", 1)
            text = text.replace("kmuren = 1", f"kmuren = {mur:.12g}", 1)
            text = text.replace("kmufac = 1", f"kmufac = {muf:.12g}", 1)
            text = re.sub(r"qt_bins = \[ [^\]]+ \]", "qt_bins = [ " + " ".join(f"{x:.12g}" for x in edges) + " ]", text, count=1)
            card.write_text(text)
            if table.exists():
                table.unlink()
            with log.open("w") as handle:
                subprocess.run([str(DYTURBO), str(card)], cwd=DYROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True, timeout=3600)
            grid = parse_grid(table, first_edge=edges[0], last_edge=edges[-1])
            if len(grid) != len(data):
                raise RuntimeError(f"{dataset} {scale_tag}: parsed {len(grid)}, expected {len(data)}")
            raw = grid[["raw_fb_per_bin", "raw_unc_fb_per_bin"]].to_numpy(float)
            if not np.isfinite(raw).all():
                table.unlink(missing_ok=True)
                raise RuntimeError(f"{dataset} {scale_tag}: DYTurbo table contains nonfinite values")
            # Cancellation-dominated low-qT bins can have a negative central
            # Vegas estimate while the quoted integration error is larger
            # than the estimate.  Preserve those rows for the dedicated
            # high-statistics refinement rather than throwing away the whole
            # seven-point scan.  A negative value remains a hard gate failure
            # until refinement replaces it with a positive estimate.
            bad = grid[grid.raw_fb_per_bin.to_numpy(float) <= 0]
            for _, bad_row in bad.iterrows():
                unresolved_rows.append({
                    "dataset": dataset, "row_id": str(data.iloc[int(bad_row.name)].row_id),
                    "scale_muR": mur, "scale_muF": muf,
                    "raw_fb_per_bin": float(bad_row.raw_fb_per_bin),
                    "raw_unc_fb_per_bin": float(bad_row.raw_unc_fb_per_bin),
                })
            merged = data[["dataset", "row_id", "qT_low", "qT_high", "CS", "error"]].join(grid, rsuffix="_grid")
            if not np.allclose(merged.qT_low, merged.qT_low_grid, atol=1e-9, rtol=0) or not np.allclose(merged.qT_high, merged.qT_high_grid, atol=1e-9, rtol=0):
                raise RuntimeError(f"{dataset} {scale_tag}: bin mismatch")
            width = merged.qT_high - merged.qT_low
            merged["scale_muR"] = mur
            merged["scale_muF"] = muf
            merged["g1_GeV2"] = args.g1
            merged["wy_pb_per_GeV"] = merged.raw_fb_per_bin / width / 1000.0
            merged["wy_unc_pb_per_GeV"] = merged.raw_unc_fb_per_bin / width / 1000.0
            merged["ratio_to_data"] = merged.wy_pb_per_GeV / merged.CS
            records.append(merged.drop(columns=["qT_low_grid", "qT_high_grid"]))
    result = pd.concat(records, ignore_index=True)
    result.to_csv(out / "tevatron_scale_variations.csv", index=False)
    central = result[(result.scale_muR == 1.0) & (result.scale_muF == 1.0)].set_index("row_id")
    wide = result.pivot(index="row_id", columns=["scale_muR", "scale_muF"], values="wy_pb_per_GeV").loc[central.index]
    vals = wide.to_numpy(float)
    central_values = central.wy_pb_per_GeV.to_numpy(float)
    envelope_low = vals.min(axis=1)
    envelope_high = vals.max(axis=1)
    summary = {
        "status": ("isolated_tevatron_unprimed_n3ll_nnlo_wy_scale_variation_complete_not_production"
                   if bool((vals > 0).all()) else
                   "isolated_tevatron_unprimed_n3ll_nnlo_wy_scale_variation_needs_refinement_not_production"),
        "engine": str(DYTURBO), "order": 3, "primed": False,
        "convention": "W=RES, ASY=-CT, FO=VJ, Y=FO-ASY=VJ+CT",
        "datasets": list(DATASETS), "row_count": int(len(central)),
        "g1_GeV2": float(args.g1), "calls_per_component": int(args.calls),
        "variations": [{"muR": r, "muF": f} for r, f in VARIATIONS],
        "all_finite": bool(np.isfinite(vals).all()), "all_positive": bool((vals > 0).all()),
        "unresolved_rows": unresolved_rows,
        "central_stat_only_chi2_per_row": float(np.mean(((central_values - central.CS.to_numpy(float)) / central.error.to_numpy(float)) ** 2)),
        "scale_envelope_relative_halfwidth_median": float(np.median((envelope_high - envelope_low) / (2 * central_values))),
        "scale_envelope_relative_halfwidth_max": float(np.max((envelope_high - envelope_low) / (2 * central_values))),
        "scale_envelope_relative_max_excursion_median": float(np.median(np.maximum(envelope_high - central_values, central_values - envelope_low) / central_values)),
        "scale_envelope_relative_max_excursion_max": float(np.max(np.maximum(envelope_high - central_values, central_values - envelope_low) / central_values)),
        "artifact_csv": str(out / "tevatron_scale_variations.csv"),
        "frozen_baseline_unchanged": True, "production_outputs_modified": False,
    }
    (out / "scale_variation_status.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
