#!/usr/bin/env python3
"""Collect localized FNP profiles and report only bracketed Delta-chi2 intervals."""

from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
TARGET = BASE / "summaries" / "localized_bT_profiles_x0p1"


def crossing(x0: float, y0: float, x1: float, y1: float, level: float = 1.0) -> float:
    return float(x0 + (level - y0) * (x1 - x0) / (y1 - y0))


def interval(group: pd.DataFrame) -> dict:
    group = group.sort_values("achieved_ratio").reset_index(drop=True)
    minimum_index = int(group["unpenalized_chi2"].idxmin())
    minimum = group.iloc[minimum_index]
    dchi2 = group["delta_chi2_grid"].to_numpy(float)
    ratios = group["achieved_ratio"].to_numpy(float)

    lower = None
    for i in range(minimum_index - 1, -1, -1):
        if dchi2[i] >= 1.0 and dchi2[i + 1] < 1.0:
            lower = crossing(ratios[i + 1], dchi2[i + 1], ratios[i], dchi2[i])
            break
    upper = None
    for i in range(minimum_index + 1, len(group)):
        if dchi2[i] >= 1.0 and dchi2[i - 1] < 1.0:
            upper = crossing(ratios[i - 1], dchi2[i - 1], ratios[i], dchi2[i])
            break

    return {
        "candidate_id": minimum["candidate_id"],
        "x": float(minimum["x"]),
        "bT": float(minimum["bT"]),
        "grid_minimum_ratio": float(minimum["achieved_ratio"]),
        "grid_minimum_unpenalized_chi2": float(minimum["unpenalized_chi2"]),
        "minimum_at_lower_edge": minimum_index == 0,
        "minimum_at_upper_edge": minimum_index == len(group) - 1,
        "lower_1sigma_ratio": lower,
        "upper_1sigma_ratio": upper,
        "lower_1sigma_bracketed": lower is not None,
        "upper_1sigma_bracketed": upper is not None,
        "both_1sigma_sides_bracketed": lower is not None and upper is not None,
        "ratio_grid_min": float(ratios.min()),
        "ratio_grid_max": float(ratios.max()),
        "max_delta_chi2": float(dchi2.max()),
        "max_target_miss_relative": float(group["target_miss_relative"].max()),
    }


def main() -> None:
    rows = []
    for filename in glob.glob(str(BASE / "outputs/profile_*/fit_status.json")):
        status = json.loads(Path(filename).read_text())
        profile = status.get("point_profile", {})
        if profile.get("target_ratio") is None:
            continue
        reference = float(profile["reference_fnp"])
        achieved = float(profile["achieved_fnp"])
        rows.append({
            "tag": Path(filename).parent.name,
            "candidate_id": Path(filename).parent.name.split("_x")[0].removeprefix("profile_"),
            "x": float(profile["x"]),
            "bT": float(profile["bT"]),
            "target_ratio": float(profile["target_ratio"]),
            "achieved_ratio": achieved / reference,
            "target_miss_relative": abs(
                achieved / (reference * float(profile["target_ratio"])) - 1.0),
            "unpenalized_chi2": float(status["final"]["unpenalized_total_chi2"]),
            "profile_penalty_per_row": float(
                status["final"]["profile_penalty_per_row_objective"]),
            "closure_evaluations": int(status["lbfgs"]["closure_evaluations"]),
            "fnp_gradient_l2": float(
                status["final"]["fnp_gradient_l2_per_row_objective"]),
        })
    frame = pd.DataFrame(rows)
    frame["delta_chi2_grid"] = frame["unpenalized_chi2"] - frame.groupby(
        ["candidate_id", "x", "bT"])["unpenalized_chi2"].transform("min")
    summaries = pd.DataFrame([
        interval(group) for _, group in frame.groupby(["candidate_id", "x", "bT"])
    ])

    TARGET.mkdir(parents=True, exist_ok=True)
    frame.sort_values(["candidate_id", "bT", "achieved_ratio"]).to_csv(
        TARGET / "profiles_long.csv", index=False)
    summaries.sort_values(["candidate_id", "bT"]).to_csv(
        TARGET / "interval_status.csv", index=False)
    (TARGET / "summary.json").write_text(json.dumps({
        "status": "localized_profile_diagnostic_not_production",
        "profile_count": len(frame),
        "interval_count": len(summaries),
        "delta_chi2_definition": "unpenalized data chi2 plus normalization penalty, relative to the lowest scanned point at fixed candidate/x/bT",
        "interval_method": "linear interpolation of raw adjacent scan points; no extrapolation and no interval reported for an unbracketed side",
        "caveat": "Profiles are conditional on the Gaussian-windowed local deformation coordinate and finite optimizer horizon.",
        "intervals": summaries.to_dict(orient="records"),
        "production_sources_modified": False,
    }, indent=2) + "\n")

    candidates = sorted(frame["candidate_id"].unique())
    fig, axes = plt.subplots(len(candidates), 1, figsize=(7.2, 3.0 * len(candidates)),
                             sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, candidate in zip(axes, candidates):
        selected = frame[frame["candidate_id"].eq(candidate)]
        for bT, group in selected.groupby("bT"):
            group = group.sort_values("achieved_ratio")
            ax.plot(group["achieved_ratio"], group["delta_chi2_grid"],
                    marker="o", ms=3, label=fr"$b_T={bT:g}$")
        ax.axhline(1.0, color="black", ls="--", lw=1)
        ax.set_title(candidate)
        ax.set_ylabel(r"$\Delta\chi^2$")
        ax.set_ylim(bottom=-0.05)
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, ncol=3, fontsize=8)
    axes[-1].set_xlabel(r"constrained $F_{\rm NP}$ / three-start median")
    fig.savefig(TARGET / "localized_profiles.png", dpi=220)
    fig.savefig(TARGET / "localized_profiles.pdf")
    plt.close(fig)
    print(summaries.to_string(index=False))


if __name__ == "__main__":
    main()
