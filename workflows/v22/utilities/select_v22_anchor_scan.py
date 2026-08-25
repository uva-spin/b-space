#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as handle:
        return json.load(handle)


def parse_lambda(dirname: str) -> float:
    tag = dirname.replace("lambda_", "")
    tag = tag.replace("p", ".").replace("m", "-")
    return float(tag)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-root", default="replica_scan_v22_stage1_logf_anchor")
    parser.add_argument("--out", default="replica_scan_v22_stage1_logf_anchor/anchor_scan_selection")
    parser.add_argument("--max-chi2", type=float, default=2.0)
    parser.add_argument("--max-norm-pull", type=float, default=4.0)
    parser.add_argument("--min-useful-band", type=float, default=0.02)
    parser.add_argument("--max-band", type=float, default=0.25)
    args = parser.parse_args()

    scan_root = Path(args.scan_root)
    if not scan_root.exists():
        raise SystemExit(f"Missing scan root: {scan_root}")

    rows = []
    for scan in sorted(scan_root.glob("lambda_*")):
        if not scan.is_dir():
            continue

        basic = load_json(scan / "audit_basic" / "v22_replica_pilot_basic_summary.json")
        band = load_json(scan / "tmd_bspace_bands_exactx" / "audit" / "bspace_band_audit_summary.json")

        if not basic and not band:
            continue

        lam = parse_lambda(scan.name)
        max_hw = band.get("max_relative_68_halfwidth_active", np.nan)
        row = {
            "lambda_logf_anchor": lam,
            "scan_dir": str(scan),
            "n_replicas": basic.get("n_replicas"),
            "replica_chi2_median": basic.get("replica_chi2_median"),
            "replica_chi2_max": basic.get("replica_chi2_max"),
            "max_abs_norm_pull": basic.get("max_abs_norm_pull"),
            "basic_pass": basic.get("V22_THREE_REPLICA_BASIC_PASS"),
            "band_technical_pass": band.get("BSPACE_TMD_BAND_TECHNICAL_PASS"),
            "band_uncertainty_useful_pass": band.get("BSPACE_TMD_BAND_UNCERTAINTY_USEFUL_PASS"),
            "max_relative_68_halfwidth_active": max_hw,
            "max_central_vs_replica_median_rel_p90_active": band.get("max_central_vs_replica_median_rel_p90_active"),
            "interpolated_x_values": band.get("interpolated_x_values"),
        }

        row["fit_norm_pass"] = bool(
            row["basic_pass"]
            and row["replica_chi2_max"] is not None
            and float(row["replica_chi2_max"]) < float(args.max_chi2)
            and (
                row["max_abs_norm_pull"] is None
                or float(row["max_abs_norm_pull"]) < float(args.max_norm_pull)
            )
        )

        row["band_width_in_target_window"] = bool(
            row["band_technical_pass"]
            and row["max_relative_68_halfwidth_active"] is not None
            and np.isfinite(float(row["max_relative_68_halfwidth_active"]))
            and float(row["max_relative_68_halfwidth_active"]) >= float(args.min_useful_band)
            and float(row["max_relative_68_halfwidth_active"]) <= float(args.max_band)
        )

        row["candidate_pass"] = bool(row["fit_norm_pass"] and row["band_width_in_target_window"])
        rows.append(row)

    table = pd.DataFrame(rows).sort_values("lambda_logf_anchor", ascending=False)

    candidates = table[table["candidate_pass"]].copy()
    selected = None
    if not candidates.empty:
        # Choose the strongest anchor among those that are useful.  This is the
        # most conservative non-collapsed option.
        selected = candidates.sort_values("lambda_logf_anchor", ascending=False).iloc[0].to_dict()

    decision = {
        "n_anchor_points": int(len(table)),
        "max_chi2": float(args.max_chi2),
        "max_norm_pull": float(args.max_norm_pull),
        "min_useful_band": float(args.min_useful_band),
        "max_band": float(args.max_band),
        "selected": selected,
        "ANCHOR_SCAN_HAS_USABLE_CANDIDATE": selected is not None,
        "interpretation": (
            "Select the strongest anchor that gives non-collapsed bands while preserving fit and normalization quality. "
            "If no candidate passes, add intermediate anchors or weaken the trust-region protocol."
        ),
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "anchor_scan_summary.csv", index=False)
    (out / "anchor_scan_decision.json").write_text(json.dumps(decision, indent=2) + "\n")

    print("\n=== v22 anchor scan summary ===")
    print(table.to_string(index=False))
    print("\n=== Decision ===")
    print(json.dumps(decision, indent=2))
    print("\nwrote:", out)

    if selected is None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
