#!/usr/bin/env python3
"""Run isolated true-NLO DYTurbo V+jet checks for selected LHCb bins.

The earlier LHCb cards were labelled NLO but set ``doVJREAL`` and
``doVJVIRT`` false, which evaluates only the V+jet LO piece.  This runner
turns on the real and virtual NLO pieces, keeps fixed-order mode, and writes
fresh artifacts under the finite-Y completion campaign only.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import re
import subprocess

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "Data/v23a_tevatron_plus_lhcb7_fiducial_candidate/LHCb_7.csv"
RUNNER = ROOT / "systematics/finite_y_tail_benchmark/scripts/run_lhcb7_dyturbo_benchmark.py"
DEFAULT_OUT = ROOT / "systematics/finite_y_completion_2026/reports/lhcb7_external_true_nlo"
DEFAULT_DYTURBO = Path("/home/dustin/src/dyturbo-1.4.2/bin/dyturbo")
DEFAULT_ROOT = Path("/home/dustin/src/dyturbo-1.4.2")
DEFAULT_LIB = Path("/home/dustin/miniforge3/envs/pdf-fit/lib")


def load_runner():
    spec = importlib.util.spec_from_file_location("existing_lhcb_dyturbo_runner", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_first_value(path: Path) -> tuple[float, float]:
    for line in path.read_text().splitlines():
        text = line.strip()
        if text and not text.startswith("#"):
            parts = text.split()
            if len(parts) >= 2:
                return float(parts[0]), float(parts[1])
    raise RuntimeError(f"no DYTurbo result in {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", default=["LHCb_7:10"])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--dyturbo", default=str(DEFAULT_DYTURBO))
    ap.add_argument("--dyturbo-root", default=str(DEFAULT_ROOT))
    ap.add_argument("--cores", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--mu-r-factor", type=float, default=1.0)
    ap.add_argument("--mu-f-factor", type=float, default=1.0)
    ap.add_argument("--qcd-order", type=int, choices=(1, 2), default=1,
                    help="DYTurbo QCD order: 1=NLO, 2=NNLO")
    ap.add_argument("--pdf-set", default="NNPDF40_nnlo_as_01180")
    ap.add_argument("--pdf-member", type=int, default=0)
    ap.add_argument("--vj-calls", type=int, default=100000,
                    help="V+jet real/virtual Vegas calls per integration")
    ap.add_argument("--no-lepton-cuts", action="store_true",
                    help="diagnostic boson-level comparison; disable the fiducial lepton cuts")
    ap.add_argument("--no-gamma", action="store_true",
                    help="diagnostic: disable the gamma* contribution")
    ap.add_argument("--negative-y", action="store_true",
                    help="diagnostic: mirror the selected positive-rapidity bin to negative rapidity")
    ap.add_argument("--inclusive-boson-y", action="store_true",
                    help="integrate over the positive-arm boson rapidity while retaining the lepton cuts")
    ap.add_argument("--boson-y-low", type=float, default=None)
    ap.add_argument("--boson-y-high", type=float, default=None)
    ap.add_argument("--ewscheme", type=int, default=1,
                    help="DYTurbo electroweak scheme for convention scan")
    args = ap.parse_args()
    runner = load_runner()
    data = pd.read_csv(DATA)
    selected = data[data.row_id.isin(args.rows)].copy()
    if len(selected) != len(args.rows):
        raise SystemExit("requested rows are missing")
    out = Path(args.out)
    cards, logs, tables = out / "cards", out / "logs", out / "tables"
    for path in (cards, logs, tables):
        path.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        [str(DEFAULT_LIB), str(Path(args.dyturbo_root) / "lib"), env.get("LD_LIBRARY_PATH", "")]
    )
    records = []
    scale_tag = f"o{args.qcd_order}_mur{args.mu_r_factor:g}_muf{args.mu_f_factor:g}".replace(".", "p")
    for _, row in selected.iterrows():
        tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(row.row_id))
        output_name = f"{tag}_dyturbo_vj_true_nlo_fid_{scale_tag}"
        card_text = runner.card_text(
            row, output_name=output_name, pdf_set=args.pdf_set, pdf_member=args.pdf_member,
            cores=args.cores, mu_r_factor=args.mu_r_factor, mu_f_factor=args.mu_f_factor,
        )
        card_text = card_text.replace("doVJREAL = false", "doVJREAL = true")
        card_text = card_text.replace("doVJVIRT = false", "doVJVIRT = true")
        card_text = card_text.replace("order           = 1", f"order           = {args.qcd_order}")
        card_text = card_text.replace("vegasncallsVJREAL = 100000", f"vegasncallsVJREAL = {args.vj_calls}")
        card_text = card_text.replace("vegasncallsVJVIRT = 100000", f"vegasncallsVJVIRT = {args.vj_calls}")
        card_text = card_text.replace("VJquad = true", "VJquad = false")
        card_text = card_text.replace("intDimVJ   = 3", "intDimVJ   = -1")
        card_text = card_text.replace("ewscheme = 1", f"ewscheme = {args.ewscheme}")
        if args.inclusive_boson_y or args.boson_y_low is not None or args.boson_y_high is not None:
            y_low = 0.0 if args.inclusive_boson_y and args.boson_y_low is None else (
                float(args.boson_y_low) if args.boson_y_low is not None else float(row.y_Low)
            )
            y_high = 6.0 if args.inclusive_boson_y and args.boson_y_high is None else (
                float(args.boson_y_high) if args.boson_y_high is not None else float(row.y_High)
            )
            card_text = card_text.replace(
                f"y_bins  = [ {float(row.y_Low):.12g} {float(row.y_High):.12g} ]",
                f"y_bins  = [ {y_low:.12g} {y_high:.12g} ]",
            )
        if args.negative_y:
            ylo = -float(row.y_High)
            yhi = -float(row.y_Low)
            card_text = card_text.replace(
                f"y_bins  = [ {float(row.y_Low):.12g} {float(row.y_High):.12g} ]",
                f"y_bins  = [ {ylo:.12g} {yhi:.12g} ]",
            )
            # Mirror the fiducial lepton acceptance for this code-closure test.
            card_text = card_text.replace("lcymin = 2.0", "lcymin = -4.5")
            card_text = card_text.replace("lcymax = 4.5", "lcymax = -2.0")
            card_text = card_text.replace("lfymin = 2.0", "lfymin = -4.5")
            card_text = card_text.replace("lfymax = 4.5", "lfymax = -2.0")
        if args.no_gamma:
            card_text = card_text.replace("useGamma = true", "useGamma = false")
        if args.no_lepton_cuts:
            card_text = card_text.replace("makecuts = true", "makecuts = false")
            card_text = card_text.replace("lcptcut = 20", "lcptcut = 0")
            card_text = card_text.replace("lcymin = 2.0", "lcymin = 0.0")
            card_text = card_text.replace("lcymax = 4.5", "lcymax = 1000.0")
            card_text = card_text.replace("lfptcut = 20", "lfptcut = 0")
            card_text = card_text.replace("lfymin = 2.0", "lfymin = 0.0")
            card_text = card_text.replace("lfymax = 4.5", "lfymax = 1000.0")
        card = cards / f"{output_name}.in"
        card.write_text(card_text)
        log = logs / f"{output_name}.log"
        txt_src = Path(args.dyturbo_root) / f"{output_name}.txt"
        dat_src = Path(args.dyturbo_root) / f"{output_name}.dat"
        for path in (txt_src, dat_src):
            if path.exists():
                path.unlink()
        with log.open("w") as handle:
            subprocess.run([str(Path(args.dyturbo).resolve()), str(card.resolve())],
                           cwd=args.dyturbo_root, env=env, stdout=handle,
                           stderr=subprocess.STDOUT, check=True, timeout=args.timeout)
        txt_dst = tables / txt_src.name
        if txt_src.exists():
            txt_src.replace(txt_dst)
        if dat_src.exists():
            dat_src.unlink()
        value, unc = parse_first_value(txt_dst)
        width = float(row.qT_high - row.qT_low)
        records.append({
            "dataset": row.dataset, "row_id": row.row_id, "qT": row.qT,
            "qT_low": row.qT_low, "qT_high": row.qT_high, "qT_over_Q": row.qT_over_Q,
            "data_pb_per_GeV": row.CS, "data_bin_pb": row.CS * width,
            "dyturbo_raw_fb_bin": value, "dyturbo_raw_unc_fb_bin": unc,
            "dyturbo_pb_bin": value / 1000.0, "dyturbo_pb_bin_unc": unc / 1000.0,
            "dyturbo_pb_per_GeV": value / (1000.0 * width),
            "dyturbo_pb_per_GeV_unc": unc / (1000.0 * width),
            "card": str(card), "log": str(log), "txt": str(txt_dst),
            "true_nlo_vj": True,
        })
    result = pd.DataFrame(records)
    result.to_csv(out / "dyturbo_true_nlo_summary.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
