#!/usr/bin/env python3
"""Create a data-replica x PDF-replica plan for v23a DY TMD uncertainty runs.

The output CSV is intentionally simple and is consumed by the launcher and
postprocessor:

  seed,pdf_set,pdf_member,run_dir,cache_dir,cache_tag

Examples:

  # 20-replica pilot, cycling NNPDF members 1..20
  python3 v23/tools/make_v23a_data_pdf_replica_plan.py \
    --n-replicas 20 \
    --seed-start 1001 \
    --pdf-members 1-20 \
    --out-root replica_pilot_v23a_dataPDF_lambda3_p2p5 \
    --run-prefix v23a_dataPDF_lambda3_p2p5 \
    --out replica_pilot_v23a_dataPDF_lambda3_p2p5/replica_plan.csv

  # 100-replica production-style plan, random members with replacement
  python3 v23/tools/make_v23a_data_pdf_replica_plan.py \
    --n-replicas 100 \
    --seed-start 1001 \
    --pdf-members 1-100 \
    --member-strategy random \
    --random-seed 303 \
    --out-root replica_v23a_dataPDF_lambda3_p2p5_100rep \
    --run-prefix v23a_dataPDF_lambda3_p2p5 \
    --out replica_v23a_dataPDF_lambda3_p2p5_100rep/replica_plan.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re

import pandas as pd


def parse_members(tokens: list[str]) -> list[int]:
    out: list[int] = []
    for tok in tokens:
        for piece in str(tok).replace(",", " ").split():
            m = re.fullmatch(r"(\d+)-(\d+)", piece)
            if m:
                a, b = int(m.group(1)), int(m.group(2))
                if b < a:
                    raise SystemExit(f"Bad pdf-member range: {piece}")
                out.extend(range(a, b + 1))
            else:
                out.append(int(piece))
    members = sorted(dict.fromkeys(out))
    if not members:
        raise SystemExit("No PDF members supplied.")
    if any(m < 0 for m in members):
        raise SystemExit(f"PDF members must be nonnegative: {members}")
    return members


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-replicas", type=int, required=True)
    ap.add_argument("--seed-start", type=int, default=1001)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--pdf-set", default="NNPDF40_nnlo_as_01180")
    ap.add_argument("--pdf-members", nargs="+", default=["1-100"],
                    help="PDF members, e.g. '1-20' or '1 2 3'. Avoid 0 for uncertainty replicas.")
    ap.add_argument("--member-strategy", choices=["cycle", "random"], default="cycle")
    ap.add_argument("--random-seed", type=int, default=303)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--run-prefix", default="v23a_dataPDF_lambda3")
    ap.add_argument("--cache-prefix", default="v23a_dataPDF_pdf")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.seeds:
        seeds = list(args.seeds)
        if len(seeds) != args.n_replicas:
            raise SystemExit("--seeds length must equal --n-replicas")
    else:
        seeds = list(range(int(args.seed_start), int(args.seed_start) + int(args.n_replicas)))

    members = parse_members(args.pdf_members)
    rng = random.Random(int(args.random_seed))
    if args.member_strategy == "cycle":
        chosen = [members[i % len(members)] for i in range(len(seeds))]
    else:
        chosen = [rng.choice(members) for _ in seeds]

    out_root = Path(args.out_root)
    rows = []
    for i, (seed, member) in enumerate(zip(seeds, chosen)):
        pdf_tag = f"pdf{member:04d}"
        run_name = f"{args.run_prefix}_{pdf_tag}_s{seed}"
        cache_tag = f"{args.cache_prefix}{member:04d}"
        rows.append({
            "replica_index": i,
            "seed": int(seed),
            "pdf_set": args.pdf_set,
            "pdf_member": int(member),
            "run_label": run_name,
            "run_dir": str(out_root / "outputs" / run_name),
            "cache_dir": str(out_root / "pdf_caches" / pdf_tag),
            "cache_tag": cache_tag,
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(out, index=False)

    manifest = {
        "n_replicas": int(args.n_replicas),
        "seeds": seeds,
        "pdf_set": args.pdf_set,
        "available_pdf_members": members,
        "member_strategy": args.member_strategy,
        "random_seed": int(args.random_seed),
        "n_unique_pdf_members_in_plan": int(table["pdf_member"].nunique()),
        "out_root": str(out_root),
        "plan_csv": str(out),
        "interpretation": (
            "Each row defines one experimental pseudo-data replica and one PDF member. "
            "The corresponding fit must use backend W/Y grids built with that same PDF member, "
            "and the TMD grid for that replica must be reconstructed with that same PDF member."
        ),
    }
    out.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")

    print("\n=== data x PDF replica plan ===")
    print(json.dumps(manifest, indent=2))
    print("\nPreview:")
    print(table.head(min(20, len(table))).to_string(index=False))
    print("\nwrote:", out)
    print("wrote:", out.with_suffix(".json"))


if __name__ == "__main__":
    main()
