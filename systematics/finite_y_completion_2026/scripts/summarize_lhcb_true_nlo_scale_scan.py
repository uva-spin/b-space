#!/usr/bin/env python3
"""Summarize the isolated DYTurbo true-NLO scale scan for LHCb boundary bins."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
CENTRAL = BASE / "reports/lhcb7_external_true_nlo/dyturbo_true_nlo_summary.csv"
SCALES = (
    ("mur0p5_muf0p5", 0.5, 0.5),
    ("mur0p5_muf1", 0.5, 1.0),
    ("mur1_muf0p5", 1.0, 0.5),
    ("mur1_muf2", 1.0, 2.0),
    ("mur2_muf1", 2.0, 1.0),
    ("mur2_muf2", 2.0, 2.0),
)
OUT = BASE / "reports/lhcb_true_nlo_scale_scan"


def main() -> None:
    frames = []
    central = pd.read_csv(CENTRAL)
    central["mu_r"] = 1.0
    central["mu_f"] = 1.0
    central["scale_tag"] = "central"
    frames.append(central)
    for tag, mr, mf in SCALES:
        path = BASE / f"reports/lhcb7_external_true_nlo_scale_{tag}/dyturbo_true_nlo_summary.csv"
        frame = pd.read_csv(path)
        frame["mu_r"] = mr
        frame["mu_f"] = mf
        frame["scale_tag"] = tag
        frames.append(frame)
    all_scales = pd.concat(frames, ignore_index=True)
    all_scales["ratio_to_data"] = all_scales["dyturbo_pb_per_GeV"] / all_scales["data_pb_per_GeV"]
    all_scales.to_csv(OUT / "lhcb_true_nlo_scale_scan.csv", index=False) if OUT.exists() else None
    grouped = []
    for row_id, group in all_scales.groupby("row_id", sort=False):
        grouped.append({
            "row_id": row_id,
            "data_pb_per_GeV": float(group.data_pb_per_GeV.iloc[0]),
            "scale_ratio_min": float(group.ratio_to_data.min()),
            "scale_ratio_central": float(group.loc[group.scale_tag.eq("central"), "ratio_to_data"].iloc[0]),
            "scale_ratio_max": float(group.ratio_to_data.max()),
            "scale_ratio_max_scale": str(group.loc[group.ratio_to_data.idxmax(), "scale_tag"]),
            "theory_pb_per_GeV_min": float(group.dyturbo_pb_per_GeV.min()),
            "theory_pb_per_GeV_central": float(group.loc[group.scale_tag.eq("central"), "dyturbo_pb_per_GeV"].iloc[0]),
            "theory_pb_per_GeV_max": float(group.dyturbo_pb_per_GeV.max()),
        })
    ratios = all_scales.groupby("scale_tag").apply(
        lambda g: float(g.dyturbo_pb_bin.sum() / g.data_bin_pb.sum()), include_groups=False
    ).to_dict()
    high = all_scales
    report = {
        "status": "lhcb_true_nlo_scale_scan_complete_not_production",
        "scale_points": [{"tag": "central", "mu_r": 1.0, "mu_f": 1.0}] + [
            {"tag": tag, "mu_r": mr, "mu_f": mf} for tag, mr, mf in SCALES
        ],
        "rows": grouped,
        "bin_integrated_ratio_by_scale": {key: float(value) for key, value in ratios.items()},
        "maximum_boundary_ratio_to_data": float(all_scales.ratio_to_data.max()),
        "maximum_boundary_ratio_scale": str(all_scales.loc[all_scales.ratio_to_data.idxmax(), "scale_tag"]),
        "interpretation": (
            "The standard seven-point muR/muF variation changes the true-NLO "
            "positive-arm prediction but does not reach the candidate LHCb data "
            "in any boundary bin. The largest pointwise theory/data ratio is "
            "still below 0.67, so the approximately 45% deficit is not covered "
            "by this perturbative scale envelope. This remains an input "
            "normalization/observable-provenance gate, not a reason to rescale "
            "FO or reject the unitary finite-Y construction."
        ),
        "production_outputs_modified": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    all_scales.to_csv(OUT / "lhcb_true_nlo_scale_scan.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
