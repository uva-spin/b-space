#!/usr/bin/env python3
"""Create a v23a data directory with explicit CSV normalization priors.

Why
---
The v23a CSV-norm refit can produce huge normalization pulls if sysNorm in the
CSV is zero/missing.  This script writes a new data directory with explicit
per-dataset sysNorm values, leaving CS/error/kinematics unchanged.

Use values as fractions or percents:
  E772=0.10    means 10%
  E772=10%     means 10%
  E772=10      also means 10%  (numbers >1 are interpreted as percent)

The output is intended for runs with:
  --norm-source csv --ptp-source csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import re

import numpy as np
import pandas as pd


def parse_prior(text: str) -> tuple[str, float, str]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"Expected DATASET=VALUE, got {text!r}")
    dataset, raw = text.split("=", 1)
    dataset = dataset.strip()
    raw = raw.strip()
    if not dataset:
        raise argparse.ArgumentTypeError(f"Empty dataset in {text!r}")
    is_percent = raw.endswith("%")
    raw_num = raw[:-1] if is_percent else raw
    try:
        value = float(raw_num)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Bad norm value in {text!r}") from exc
    if is_percent or value > 1.0:
        frac = value / 100.0
    else:
        frac = value
    if not np.isfinite(frac) or frac <= 0.0:
        raise argparse.ArgumentTypeError(f"Norm prior must be positive, got {text!r}")
    return dataset, frac, f"{100.0 * frac:.8g}%"


def parse_existing_sysnorm(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    out = []
    for item in s:
        if item.lower() in {"", "nan", "none"}:
            out.append(np.nan)
            continue
        pct = item.endswith("%")
        item2 = item[:-1] if pct else item
        try:
            val = float(item2)
        except Exception:
            out.append(np.nan)
            continue
        out.append(val / 100.0 if pct or val > 1.0 else val)
    return pd.Series(out, index=series.index, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-data-dir",
        default="Data/v23a_fixed_target_lowQ_row99_variants/corrected_E288_300_99",
    )
    parser.add_argument(
        "--out-data-dir",
        required=True,
    )
    parser.add_argument(
        "--sysnorm",
        nargs="+",
        required=True,
        help="Per-dataset normalization priors, e.g. E288_200=8% E772=10%",
    )
    parser.add_argument(
        "--ptp-mode",
        choices=["keep", "zero"],
        default="keep",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source = Path(args.source_data_dir)
    out = Path(args.out_data_dir)

    if not source.exists():
        raise SystemExit(f"Missing source data dir: {source}")

    if out.exists():
        if not args.force:
            raise SystemExit(f"Refusing to overwrite existing {out}; use --force")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    priors = dict((dataset, (frac, pct)) for dataset, frac, pct in map(parse_prior, args.sysnorm))

    files = sorted(source.glob("*.csv"))
    if not files:
        raise SystemExit(f"No CSV files found in {source}")

    rows = []
    datasets_seen = []
    for path in files:
        dataset = path.stem
        datasets_seen.append(dataset)
        if dataset not in priors:
            raise SystemExit(f"No --sysnorm value supplied for dataset {dataset}")

        frac, pct_text = priors[dataset]
        df = pd.read_csv(path)

        old = parse_existing_sysnorm(df["sysNorm"]) if "sysNorm" in df.columns else pd.Series(np.nan, index=df.index)
        if "sysNorm" not in df.columns:
            df["sysNorm"] = pct_text
        else:
            df["sysNorm"] = pct_text

        if args.ptp_mode == "zero":
            df["sysP2P"] = "0.0%"
        elif "sysP2P" not in df.columns:
            df["sysP2P"] = "0.0%"

        df.to_csv(out / path.name, index=False)

        finite_old = old[np.isfinite(old)]
        rows.append({
            "dataset": dataset,
            "n_rows": int(len(df)),
            "old_sysNorm_nfinite": int(len(finite_old)),
            "old_sysNorm_min": float(finite_old.min()) if len(finite_old) else None,
            "old_sysNorm_median": float(finite_old.median()) if len(finite_old) else None,
            "old_sysNorm_max": float(finite_old.max()) if len(finite_old) else None,
            "new_sysNorm_fraction": frac,
            "new_sysNorm_percent_text": pct_text,
            "ptp_mode": args.ptp_mode,
        })

    extra = sorted(set(priors) - set(datasets_seen))
    if extra:
        raise SystemExit(f"Supplied priors for datasets not found as files: {extra}")

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "norm_prior_summary.csv", index=False)

    manifest = {
        "source_data_dir": str(source),
        "out_data_dir": str(out),
        "datasets": datasets_seen,
        "priors_fraction": {k: v[0] for k, v in priors.items()},
        "priors_percent_text": {k: v[1] for k, v in priors.items()},
        "ptp_mode": args.ptp_mode,
        "note": "CS/error/kinematics are unchanged. Only sysNorm and optionally sysP2P are rewritten.",
    }
    (out / "norm_prior_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print("\n=== v23a explicit CSV norm-prior data dir created ===")
    print(json.dumps(manifest, indent=2))
    print("\n=== Summary ===")
    print(summary.to_string(index=False))
    print("\nwrote:", out)


if __name__ == "__main__":
    main()
