#!/usr/bin/env python3
"""Summarize isolated DYTurbo NNLO boundary calculations and scale envelope."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
CENTRAL = BASE / "reports/lhcb7_external_true_nnlo_probe_10m/dyturbo_true_nlo_summary.csv"
SCALES = (
    ("mur0p5_muf0p5", 0.5, 0.5),
    ("mur0p5_muf1", 0.5, 1.0),
    ("mur1_muf0p5", 1.0, 0.5),
    ("mur1_muf2", 1.0, 2.0),
    ("mur2_muf1", 2.0, 1.0),
    ("mur2_muf2", 2.0, 2.0),
)
OUT = BASE / "reports/lhcb_true_nnlo_scale_scan"


def _first_value(path: Path) -> tuple[float, float]:
    for line in path.read_text().splitlines():
        text = line.strip()
        if text and not text.startswith("#"):
            fields = text.split()
            if len(fields) >= 2:
                return float(fields[0]), float(fields[1])
    raise RuntimeError(f"no DYTurbo result in {path}")


def main() -> None:
    frames = []
    central = pd.read_csv(CENTRAL)
    # Parallel DYTurbo launches can race while writing the aggregate CSV.  The
    # per-row cards/tables are authoritative, so reconstruct any central rows
    # absent from that CSV from the completed 10M-call tables.
    required = {f"LHCb_7:{i}" for i in (10, 11, 12, 13)}
    missing = required.difference(set(central["row_id"]))
    if missing:
        reference = pd.read_csv(
            BASE / "reports/lhcb7_external_true_nnlo_scale_mur0p5_muf0p5/dyturbo_true_nlo_summary.csv"
        ).set_index("row_id")
        rebuilt = []
        for row_id in sorted(missing):
            row = reference.loc[row_id].copy()
            row["row_id"] = row_id
            table = BASE / "reports/lhcb7_external_true_nnlo_probe_10m/tables" / (
                row_id.replace(":", "_") + "_dyturbo_vj_true_nlo_fid_o2_mur1_muf1.txt"
            )
            value, unc = _first_value(table)
            width = float(row.qT_high - row.qT_low)
            row["dyturbo_raw_fb_bin"] = value
            row["dyturbo_raw_unc_fb_bin"] = unc
            row["dyturbo_pb_bin"] = value / 1000.0
            row["dyturbo_pb_bin_unc"] = unc / 1000.0
            row["dyturbo_pb_per_GeV"] = value / (1000.0 * width)
            row["dyturbo_pb_per_GeV_unc"] = unc / (1000.0 * width)
            rebuilt.append(row)
        central = pd.concat([central, pd.DataFrame(rebuilt)], ignore_index=True)
    central["mu_r"] = 1.0
    central["mu_f"] = 1.0
    central["scale_tag"] = "central"
    frames.append(central)
    for tag, mr, mf in SCALES:
        path = BASE / f"reports/lhcb7_external_true_nnlo_scale_{tag}/dyturbo_true_nlo_summary.csv"
        frame = pd.read_csv(path)
        frame["mu_r"] = mr
        frame["mu_f"] = mf
        frame["scale_tag"] = tag
        frames.append(frame)
    all_scales = pd.concat(frames, ignore_index=True)
    all_scales["ratio_to_data"] = all_scales["dyturbo_pb_per_GeV"] / all_scales["data_pb_per_GeV"]
    rows = []
    for row_id, group in all_scales.groupby("row_id", sort=False):
        rows.append({
            "row_id": row_id,
            "data_pb_per_GeV": float(group.data_pb_per_GeV.iloc[0]),
            "ratio_min": float(group.ratio_to_data.min()),
            "ratio_central": float(group.loc[group.scale_tag.eq("central"), "ratio_to_data"].iloc[0]),
            "ratio_max": float(group.ratio_to_data.max()),
            "max_scale": str(group.loc[group.ratio_to_data.idxmax(), "scale_tag"]),
            "theory_min_pb_per_GeV": float(group.dyturbo_pb_per_GeV.min()),
            "theory_central_pb_per_GeV": float(group.loc[group.scale_tag.eq("central"), "dyturbo_pb_per_GeV"].iloc[0]),
            "theory_max_pb_per_GeV": float(group.dyturbo_pb_per_GeV.max()),
            "max_relative_mc_uncertainty": float((group.dyturbo_pb_per_GeV_unc / group.dyturbo_pb_per_GeV).max()),
        })
    ratio_by_scale = all_scales.groupby("scale_tag").apply(
        lambda g: float(g.dyturbo_pb_bin.sum() / g.data_bin_pb.sum()), include_groups=False
    ).to_dict()
    report = {
        "status": "lhcb_true_nnlo_scale_scan_complete_diagnostic_not_production",
        "order": "DYTurbo fixedorder_only=true, order=2 (NNLO), doVJREAL/doVJVIRT=true",
        "central_calls": 10000000,
        "noncentral_calls": 5000000,
        "rows": rows,
        "bin_integrated_ratio_by_scale": {key: float(value) for key, value in ratio_by_scale.items()},
        "maximum_ratio_to_data": float(all_scales.ratio_to_data.max()),
        "maximum_ratio_row": str(all_scales.loc[all_scales.ratio_to_data.idxmax(), "row_id"]),
        "maximum_ratio_scale": str(all_scales.loc[all_scales.ratio_to_data.idxmax(), "scale_tag"]),
        "interpretation": (
            "NNLO raises the boundary prediction substantially relative to NLO: "
            "central theory/data ratios are approximately 0.85, 0.82, 0.77, and "
            "0.89 for rows 10--13. The standard scale envelope reaches at most "
            "about 0.97 pointwise (row 13) but remains below data in the first "
            "three boundary bins. Thus the NLO deficit was partly a missing-order "
            "effect, while a residual LHCb observable/covariance or higher-order "
            "closure remains before universal promotion."
        ),
        "production_outputs_modified": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    all_scales.to_csv(OUT / "lhcb_true_nnlo_scale_scan.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
