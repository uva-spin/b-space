#!/usr/bin/env python3
"""Build full b-space TMD curves from isolated fitted FNP states."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
PROJECT = BASE.parents[1]
DEFAULT_REFERENCE = (
    BASE.parent
    / "collins_factorization_validity/plots/"
    "rowidfix_stageFT_all_qmax0p20_lam0p50_central_exactx/"
    "v22_scheme_tmd_bspace_long.csv"
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag-glob")
    parser.add_argument("--run-tag", action="append", default=None)
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--x", type=float, default=0.1)
    parser.add_argument("--Q", type=float, action="append", default=None)
    parser.add_argument(
        "--flavor", action="append",
        choices=("u", "d", "s", "ubar", "dbar", "sbar"), default=None)
    parser.add_argument("--reference-bspace", type=Path, default=DEFAULT_REFERENCE)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    q_values = args.Q or [7.5, 10.0]
    flavors = args.flavor or ["u", "d", "s", "ubar", "dbar", "sbar"]
    if bool(args.tag_glob) == bool(args.run_tag):
        raise ValueError("choose exactly one of --tag-glob or repeated --run-tag")
    if args.run_tag:
        run_dirs = sorted(BASE / "outputs" / tag for tag in args.run_tag)
        missing = [path for path in run_dirs if not (path / "fit_status.json").exists()]
        if missing:
            raise RuntimeError(f"incomplete explicit runs: {missing}")
    else:
        run_dirs = sorted(
            path.parent for path in
            (Path(item) for item in glob.glob(
                str(BASE / "outputs" / args.tag_glob / "fit_status.json")))
        )
    if not run_dirs:
        raise RuntimeError(f"no completed runs match {args.tag_glob!r}")

    reference_all = pd.read_csv(args.reference_bspace)
    reference = reference_all[
        np.isclose(reference_all["x"], args.x)
        & reference_all["Q"].isin(q_values)
        & reference_all["flavor"].astype(str).isin(flavors)
    ].copy()
    expected = len(q_values) * len(flavors)
    if reference.groupby(["Q", "flavor"]).ngroups != expected:
        raise RuntimeError("reference b-space grid does not cover requested curves")

    curve_rows = []
    run_rows = []
    for run in run_dirs:
        status = json.loads((run / "fit_status.json").read_text())
        grid = pd.read_csv(run / "fnp_grid.csv")
        x_grid = grid[np.isclose(grid["x"], args.x)].sort_values("bT")
        if len(x_grid) < 3:
            raise RuntimeError(f"{run} has no FNP grid at x={args.x}")
        for (q_value, flavor), group in reference.groupby(
                ["Q", "flavor"], sort=False):
            group = group.sort_values("bT").copy()
            b = group["bT"].to_numpy(float)
            fnp = np.interp(
                b, x_grid["bT"].to_numpy(float),
                x_grid["F_NP"].to_numpy(float))
            group["F_NP"] = fnp
            group["ftilde"] = group["ftilde_no_np"].to_numpy(float) * fnp
            group["x_ftilde"] = group["x"].to_numpy(float) * group["ftilde"]
            group["b_ftilde"] = b * group["ftilde"]
            group["b_x_ftilde"] = (
                b * group["x"].to_numpy(float) * group["ftilde"])
            group["run_tag"] = run.name
            group["fit_seed"] = status["seed"]
            group["replica_seed"] = status.get("replica_seed")
            group["lambda_logcurv"] = status.get(
                "regularization", {}).get(
                    "logf_curvature", {}).get("lambda", 0.0)
            group["lambda_loglength"] = status.get(
                "regularization", {}).get(
                    "logf_arc_length", {}).get("lambda", 0.0)
            curve_rows.append(group)
        run_rows.append({
            "run_tag": run.name,
            "fit_seed": status["seed"],
            "replica_seed": status.get("replica_seed"),
            "source_production": status["source_production"],
            "lambda_logcurv": status.get(
                "regularization", {}).get(
                    "logf_curvature", {}).get("lambda", 0.0),
            "lambda_loglength": status.get(
                "regularization", {}).get(
                    "logf_arc_length", {}).get("lambda", 0.0),
            "np_width": status.get(
                "model_complexity", {}).get("np_width"),
            "np_cond_width": status.get(
                "model_complexity", {}).get("np_cond_width"),
            "np_blocks": status.get(
                "model_complexity", {}).get("np_blocks"),
            "production_state_modified": status.get(
                "production_state_modified", False),
        })

    curves = pd.concat(curve_rows, ignore_index=True)
    target = BASE / "summaries" / args.target_name
    target.mkdir(parents=True, exist_ok=True)
    curves.to_csv(target / "bspace_tmd_ensemble_long.csv", index=False)
    pd.DataFrame(run_rows).to_csv(target / "ensemble_runs.csv", index=False)
    metadata = {
        "status": "isolated_bspace_tmd_ensemble_not_production",
        "run_count": len(run_dirs),
        "curve_count": int(
            curves.groupby(["run_tag", "Q", "flavor"]).ngroups),
        "x": args.x,
        "Q_GeV": q_values,
        "flavors": flavors,
        "reference_bspace": str(args.reference_bspace.resolve()),
        "construction": (
            "frozen perturbative ftilde_no_np multiplied by each isolated "
            "fit's interpolated FNP at the requested x"),
        "production_sources_modified": False,
    }
    (target / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
