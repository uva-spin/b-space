#!/usr/bin/env python3
"""Run isolated scale, seed, or PDF-member external-code variations."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "systematics" / "high_qt_direct_production_benchmark"
POLICY = json.loads((STUDY / "config" / "promotion_policy.json").read_text())
PYTHON = Path("/home/dustin/miniforge3/envs/pdf-fit/bin/python")


def runner(dataset: str, code: str) -> Path:
    if dataset == "LHCb_7":
        return ROOT / "systematics/finite_y_tail_benchmark/scripts" / f"run_lhcb7_{code}_benchmark.py"
    return ROOT / "v23/tools" / f"run_tevatron_{code}_benchmark.py"


def variants(kind: str, pdf_members: list[int] | None) -> list[dict]:
    if kind == "scale":
        return [
            {"tag": f"mur{mur:g}_muf{muf:g}".replace(".", "p"), "mu_r": mur, "mu_f": muf,
             "pdf_member": 0, "dy_seed": 123456, "mc_seed": 0}
            for mur, muf in POLICY["scale_variation"]["points"] if (mur, muf) != (1.0, 1.0)
        ]
    if kind == "seed":
        return [{"tag": "alternate_seed", "mu_r": 1.0, "mu_f": 1.0, "pdf_member": 0,
                 "dy_seed": POLICY["seed_reproducibility"]["dyturbo_seeds"][1],
                 "mc_seed": POLICY["seed_reproducibility"]["mcfm_seeds"][1]}]
    if not pdf_members:
        raise SystemExit("--pdf-members is required for kind=pdf")
    return [{"tag": f"member_{member:04d}", "mu_r": 1.0, "mu_f": 1.0, "pdf_member": member,
             "dy_seed": 123456, "mc_seed": 0} for member in pdf_members]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["scale", "seed", "pdf"])
    ap.add_argument("--tier", default="tier1_boundary")
    ap.add_argument("--codes", nargs="+", choices=["dyturbo", "mcfm"], default=["dyturbo", "mcfm"])
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--rows", nargs="+", default=None)
    ap.add_argument("--pdf-members", nargs="+", type=int, default=None)
    ap.add_argument("--variant-tags", nargs="+", default=None)
    ap.add_argument("--output-stage", default="variations", choices=["variations", "precision_variations"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--keep-going", action="store_true")
    ap.add_argument("--mcfm-calls", type=int, default=100000)
    ap.add_argument("--timeout", type=int, default=1200)
    args = ap.parse_args()

    source = STUDY / "summaries" / args.tier / "central" / "external_pairs.csv"
    rows = pd.read_csv(source).sort_values(["dataset", "qT_over_Q", "row_id"])
    if args.datasets:
        rows = rows.loc[rows["dataset"].isin(args.datasets)]
    if args.rows:
        rows = rows.loc[rows["row_id"].isin(args.rows)]
    if args.limit is not None:
        rows = rows.head(args.limit)
    if rows.empty:
        raise SystemExit("No rows selected")

    failures = 0
    events = STUDY / "logs" / f"{args.tier}_{args.kind}_variations.jsonl"
    selected_variants = variants(args.kind, args.pdf_members)
    if args.variant_tags:
        selected_variants = [variant for variant in selected_variants if variant["tag"] in set(args.variant_tags)]
        missing_tags = sorted(set(args.variant_tags) - {variant["tag"] for variant in selected_variants})
        if missing_tags:
            raise SystemExit(f"Unknown or unavailable variant tags: {missing_tags}")
    for variant in selected_variants:
        for _, row in rows.iterrows():
            dataset, row_id = str(row.dataset), str(row.row_id)
            row_tag = row_id.replace(":", "_").lower()
            data = ROOT / "Data/v23a_tevatron_plus_lhcb7_fiducial_candidate" / f"{dataset}.csv"
            for code in args.codes:
                out = STUDY / "outputs" / args.output_stage / args.tier / args.kind / variant["tag"] / dataset.lower() / row_tag / code
                summary = out / f"{code}_benchmark_summary.csv"
                if summary.exists() and not args.rerun:
                    print(f"SKIP {variant['tag']} {row_id} {code}", flush=True)
                    continue
                out.mkdir(parents=True, exist_ok=True)
                command = [str(PYTHON), str(runner(dataset, code)), "--data", str(data), "--rows", row_id,
                           "--out", str(out), "--pdf-member", str(variant["pdf_member"]),
                           "--mu-r-factor", str(variant["mu_r"]), "--mu-f-factor", str(variant["mu_f"]),
                           "--timeout", str(args.timeout)]
                if code == "dyturbo":
                    command += ["--cores", "4", "--seed", str(variant["dy_seed"])]
                else:
                    command += ["--calls", str(args.mcfm_calls), "--seed", str(variant["mc_seed"])]
                event = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "kind": args.kind,
                         "variant": variant, "dataset": dataset, "row_id": row_id, "code": code,
                         "command": command}
                started = time.monotonic()
                print(f"RUN {variant['tag']} {row_id} {code}", flush=True)
                with (out / "runner_stdout.log").open("w") as stream:
                    result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                event["elapsed_seconds"] = time.monotonic() - started
                event["status"] = "pass" if result.returncode == 0 else "fail"
                event["returncode"] = result.returncode
                with events.open("a") as stream:
                    stream.write(json.dumps(event) + "\n")
                if result.returncode:
                    failures += 1
                    if not args.keep_going:
                        raise SystemExit(f"Failed; see {out / 'runner_stdout.log'}")
    if failures:
        raise SystemExit(f"Completed with {failures} failures")


if __name__ == "__main__":
    main()
