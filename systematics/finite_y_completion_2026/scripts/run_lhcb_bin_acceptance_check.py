#!/usr/bin/env python3
"""Run isolated full-bin DYTurbo fiducial/inclusive checks for LHCb rows."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import subprocess

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "Data/v23a_tevatron_plus_lhcb7_fiducial_candidate/LHCb_7.csv"
DY_ROOT = Path("/home/dustin/src/dyturbo-1.4.2")
DY_EXE = DY_ROOT / "bin/dyturbo"
CONDA_LIB = Path("/home/dustin/miniforge3/envs/pdf-fit/lib")
RUNNER_PATH = ROOT / "systematics/finite_y_tail_benchmark/scripts/run_lhcb7_dyturbo_benchmark.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("lhcb_dyturbo_runner_bin", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cores", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--qcd-order", type=int, choices=(1, 2), default=1,
                    help="DYTurbo fixed-order: 1=NLO template, 2=NNLO real+virtual")
    ap.add_argument("--vj-calls", type=int, default=100000,
                    help="NNLO real/virtual Vegas calls; ignored at NLO")
    args = ap.parse_args()
    runner = load_runner()
    data = pd.read_csv(DATA)
    selected = data[data.row_id.isin(args.rows)].copy()
    if len(selected) != len(args.rows):
        raise SystemExit("requested LHCb rows are missing")
    out = Path(args.out)
    for name in ("cards", "logs", "tables"):
        (out / name).mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join([str(CONDA_LIB), str(DY_ROOT / "lib"), env.get("LD_LIBRARY_PATH", "")])
    records = []
    for _, row in selected.iterrows():
        tag = str(row.row_id).replace(":", "_")
        values = {}
        for mode, fiducial in (("fid", True), ("inc", False)):
            stem = f"{tag}_bin_{mode}_dyturbo"
            card = out / "cards" / f"{stem}.in"
            card.write_text(runner.card_text(
                row, output_name=stem, pdf_set="NNPDF40_nnlo_as_01180",
                pdf_member=0, cores=args.cores, seed=20260817,
            ))
            text = card.read_text()
            if args.qcd_order == 2:
                text = text.replace("order           = 1", "order           = 2")
                text = text.replace("doVJREAL = false", "doVJREAL = true")
                text = text.replace("doVJVIRT = false", "doVJVIRT = true")
                text = text.replace("vegasncallsVJREAL = 100000", f"vegasncallsVJREAL = {args.vj_calls}")
                text = text.replace("vegasncallsVJVIRT = 100000", f"vegasncallsVJVIRT = {args.vj_calls}")
                text = text.replace("VJquad = true", "VJquad = false")
                text = text.replace("intDimVJ   = 3", "intDimVJ   = -1")
            card.write_text(text)
            if not fiducial:
                text = card.read_text()
                text = text.replace("makecuts = true", "makecuts = false")
                text = text.replace("lcptcut = 20", "lcptcut = 0").replace("lcymin = 2.0", "lcymin = -1000").replace("lcymax = 4.5", "lcymax = 1000")
                text = text.replace("lfptcut = 20", "lfptcut = 0").replace("lfymin = 2.0", "lfymin = -1000").replace("lfymax = 4.5", "lfymax = 1000")
                card.write_text(text)
            log = out / "logs" / f"{stem}.log"
            src = DY_ROOT / f"{stem}.txt"
            dat = DY_ROOT / f"{stem}.dat"
            for path in (src, dat):
                if path.exists():
                    path.unlink()
            with log.open("w") as handle:
                subprocess.run([str(DY_EXE), str(card.resolve())], cwd=DY_ROOT,
                               env=env, stdout=handle, stderr=subprocess.STDOUT,
                               check=True, timeout=args.timeout)
            value, unc = runner.parse_first_value(src)
            destination = out / "tables" / src.name
            src.replace(destination)
            if dat.exists():
                dat.replace(out / "tables" / dat.name)
            values[mode] = (float(value), float(unc), str(destination))
        fid, fid_unc, fid_path = values["fid"]
        inc, inc_unc, inc_path = values["inc"]
        records.append({
            "row_id": str(row.row_id), "fiducial_pb": fid,
            "fiducial_unc_pb": fid_unc, "inclusive_pb": inc,
            "inclusive_unc_pb": inc_unc,
            "acceptance": fid / inc if inc > 0.0 else float("nan"),
            "fiducial_table": fid_path, "inclusive_table": inc_path,
        })
    result = pd.DataFrame(records)
    result.to_csv(out / "lhcb_bin_acceptance.csv", index=False)
    report = {
        "status": "isolated_lhcb_bin_acceptance_check_complete",
        "rows": list(args.rows),
        "qcd_order": args.qcd_order,
        "vj_calls": args.vj_calls if args.qcd_order == 2 else None,
        "all_fiducial_nonnegative": bool((result.fiducial_pb.to_numpy(float) >= 0.0).all()),
        "all_inclusive_positive": bool((result.inclusive_pb.to_numpy(float) > 0.0).all()),
        "production_outputs_modified": False,
    }
    (out / "summary.json").write_text(__import__("json").dumps(report, indent=2) + "\n")
    print(result.to_string(index=False))
    print(__import__("json").dumps(report, indent=2))


if __name__ == "__main__":
    main()
