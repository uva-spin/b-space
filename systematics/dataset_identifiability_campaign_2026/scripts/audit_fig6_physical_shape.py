#!/usr/bin/env python3
"""Audit physical/numerical shape of an isolated Fig. 6 ensemble median."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCES = [
    BASE.parents[1]
    / f"plots/prd_q020_figures/kspace_fixedx_q10_{flavor}_current/"
    "v23a_regularized_kspace_bands.csv"
    for flavor in ("u", "d")
]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kspace-ensemble", type=Path, required=True)
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--reference-band", type=Path, action="append")
    parser.add_argument("--display-kmax", type=float, default=2.25)
    parser.add_argument("--active-fraction", type=float, default=0.05)
    parser.add_argument("--negative-tolerance", type=float, default=-0.01)
    parser.add_argument("--active-rise-tolerance", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    frame = pd.read_csv(args.kspace_ensemble)
    frame = frame[
        frame["flavor"].astype(str).isin(["u", "d"])
        & np.isclose(frame["x"], 0.1)
        & np.isclose(frame["Q"], 10.0)
        & (frame["kT"] <= args.display_kmax)
    ].copy()
    value_column = "value" if "value" in frame else "median"
    candidate = (
        frame.groupby(["flavor", "kT"], observed=False)[value_column]
        .median().rename("candidate").reset_index()
    )
    reference_paths = args.reference_band or DEFAULT_REFERENCES
    reference_all = pd.concat(
        [pd.read_csv(path) for path in reference_paths], ignore_index=True)
    reference = reference_all[
        reference_all["flavor"].astype(str).isin(["u", "d"])
        & np.isclose(reference_all["x"], 0.1)
        & np.isclose(reference_all["Q"], 10.0)
        & reference_all["quantity"].eq("ftilde")
        & (reference_all["kT"] <= args.display_kmax)
    ][["flavor", "kT", "median"]].rename(
        columns={"median": "reference"})
    joined = candidate.merge(
        reference, on=["flavor", "kT"], how="left", validate="one_to_one")

    rows = []
    for flavor, group in joined.groupby("flavor", sort=False):
        group = group.sort_values("kT")
        k = group["kT"].to_numpy(float)
        y = group["candidate"].to_numpy(float)
        ref = group["reference"].to_numpy(float)
        peak = float(np.max(y))
        active = y > args.active_fraction * peak
        if not np.any(active):
            raise RuntimeError(f"{flavor} has no positive active branch")
        active_indices = np.flatnonzero(active)
        # The physical branch begins at kT=0; do not let a later ringing lobe
        # redefine it as active.
        first_gap = np.flatnonzero(~active[active_indices[0]:])
        active_end = (
            active_indices[0] + first_gap[0] - 1
            if len(first_gap) else active_indices[-1])
        branch = np.arange(active_indices[0], active_end + 1)
        rises = np.diff(y[branch])
        first_nonpositive = np.flatnonzero(y <= 0.0)
        after_zero = (
            y[first_nonpositive[0] + 1:]
            if len(first_nonpositive) else np.empty(0))
        ref_peak = float(np.nanmax(ref))
        ref_active = ref > args.active_fraction * ref_peak
        scale = np.maximum(np.abs(ref), args.active_fraction * ref_peak)
        rows.append({
            "flavor": flavor,
            "peak": peak,
            "positive_branch_5pct_kT_max_GeV": float(k[active_end]),
            "minimum_displayed_over_peak": float(np.min(y) / peak),
            "maximum_active_step_rise_over_peak": float(
                max(np.max(rises), 0.0) / peak if len(rises) else 0.0),
            "maximum_secondary_positive_lobe_over_peak": float(
                max(np.max(after_zero), 0.0) / peak if len(after_zero) else 0.0),
            "peak_ratio_to_frozen_reference": float(peak / ref_peak),
            "max_relative_displacement_in_frozen_active_region": float(
                np.max(np.abs(y[ref_active] - ref[ref_active])
                       / scale[ref_active])),
        })
    metrics = pd.DataFrame(rows)
    positivity_pass = bool(
        (metrics["minimum_displayed_over_peak"]
         >= args.negative_tolerance).all())
    monotonic_active_pass = bool(
        (metrics["maximum_active_step_rise_over_peak"]
         <= args.active_rise_tolerance).all())
    summary = {
        "status": "isolated_fig6_physical_shape_audit_not_production",
        "source_kspace_ensemble": str(args.kspace_ensemble.resolve()),
        "reference_bands": [str(path.resolve()) for path in reference_paths],
        "display_kT_max_GeV": args.display_kmax,
        "active_definition": (
            f"contiguous branch from kT=0 above {100*args.active_fraction:g}% "
            "of the candidate positive peak"),
        "negative_tolerance_over_peak": args.negative_tolerance,
        "active_step_rise_tolerance_over_peak": args.active_rise_tolerance,
        "positivity_gate_pass": positivity_pass,
        "active_monotonicity_gate_pass": monotonic_active_pass,
        "physical_shape_gate_pass": bool(
            positivity_pass and monotonic_active_pass),
        "central_displacement_is_diagnostic_not_a_gate": True,
        "production_sources_modified": False,
        "flavors": rows,
    }
    target = BASE / "summaries" / args.target_name
    target.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(target / "flavor_metrics.csv", index=False)
    (target / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
