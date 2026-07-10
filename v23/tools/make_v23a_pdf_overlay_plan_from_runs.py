#!/usr/bin/env python3
"""Create a PDF-overlay plan from already-trained experimental data replicas.

This is the fast route for a b-space TMD uncertainty that combines:
  * experimental/data-replica F_NP variation from existing runs;
  * PDF-member variation in the perturbative/OPE TMD reconstruction.

It does NOT retrain F_NP with each PDF member.  That approximation is often the
right first diagnostic because the expensive per-PDF W/Y cross-section caches
are avoided.

Example:

  PYTHONPATH=. python3 v23/tools/make_v23a_pdf_overlay_plan_from_runs.py \
    --run-glob 'replica_pilot_v23a_lambda3_normpriors15_p2p5_E772_E288400_cached_cuda/outputs/v23a_lambda3_normpriors15_p2p5_E772_E288400_cached_cuda_s*' \
    --pdf-members 1-50 \
    --out-root replica_v23a_expPDF_overlay_lambda3_normpriors15_p2p5_50rep \
    --out replica_v23a_expPDF_overlay_lambda3_normpriors15_p2p5_50rep/replica_plan.csv
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import re
import random

import pandas as pd


def parse_members(tokens: list[str]) -> list[int]:
    out: list[int] = []
    for tok in tokens:
        for piece in str(tok).replace(",", " ").split():
            m = re.fullmatch(r"(\d+)-(\d+)", piece)
            if m:
                a, b = int(m.group(1)), int(m.group(2))
                if b < a:
                    raise SystemExit(f"Bad member range: {piece}")
                out.extend(range(a, b + 1))
            else:
                out.append(int(piece))
    members = sorted(dict.fromkeys(out))
    if not members:
        raise SystemExit("No PDF members supplied.")
    return members


def infer_seed(path: Path) -> int:
    m = re.search(r"_s(\d+)$", path.name)
    if m:
        return int(m.group(1))
    m = re.search(r"s(\d+)", path.name)
    if m:
        return int(m.group(1))
    raise SystemExit(f"Could not infer seed from run directory name: {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-glob", required=True)
    ap.add_argument("--pdf-set", default="NNPDF40_nnlo_as_01180")
    ap.add_argument("--pdf-members", nargs="+", default=["1-100"])
    ap.add_argument("--member-strategy", choices=["cycle", "random"], default="cycle")
    ap.add_argument("--random-seed", type=int, default=303)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--run-prefix", default="v23a_expPDF_overlay")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    runs = [Path(p) for p in sorted(glob.glob(args.run_glob))]
    runs = [p for p in runs if p.is_dir()]
    if not runs:
        raise SystemExit(f"No run directories matched: {args.run_glob}")

    members = parse_members(args.pdf_members)
    rng = random.Random(int(args.random_seed))
    if args.member_strategy == "cycle":
        chosen = [members[i % len(members)] for i in range(len(runs))]
    else:
        chosen = [rng.choice(members) for _ in runs]

    out_root = Path(args.out_root)
    rows = []
    for i, (run, member) in enumerate(zip(runs, chosen)):
        seed = infer_seed(run)
        rows.append({
            "replica_index": i,
            "seed": seed,
            "pdf_set": args.pdf_set,
            "pdf_member": int(member),
            "run_label": run.name,
            "run_dir": str(run),
            # The following columns are not used by the overlay TMD constructor,
            # but preserving the schema keeps it compatible with the true dataPDF plan.
            "cache_dir": "",
            "cache_tag": f"overlay_pdf{member:04d}",
            "plan_role": "existing data-replica F_NP plus plan-assigned PDF member",
        })

    table = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)

    manifest = {
        "run_glob": args.run_glob,
        "n_replicas": int(len(table)),
        "pdf_set": args.pdf_set,
        "available_pdf_members": members,
        "member_strategy": args.member_strategy,
        "random_seed": int(args.random_seed),
        "n_unique_pdf_members_in_plan": int(table["pdf_member"].nunique()),
        "out_root": str(out_root),
        "plan_csv": str(out),
        "interpretation": (
            "Fast overlay plan: uses already-trained experimental data replicas for F_NP, "
            "then reconstructs the TMD with the plan-assigned PDF member. It propagates PDF "
            "uncertainty into the TMD grid without retraining cross sections for each PDF member."
        ),
        "limitation": (
            "This does not propagate PDF uncertainty back through the fitted F_NP. "
            "A true PDF-through-refit ensemble still requires W/Y caches and training per PDF member."
        ),
    }
    out.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")

    print("\n=== v23a PDF-overlay plan ===")
    print(json.dumps(manifest, indent=2))
    print("\nPreview:")
    print(table.head(min(25, len(table))).to_string(index=False))
    print("\nwrote:", out)
    print("wrote:", out.with_suffix(".json"))


if __name__ == "__main__":
    main()
