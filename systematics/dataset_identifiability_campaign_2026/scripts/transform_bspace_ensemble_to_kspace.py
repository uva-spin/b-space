#!/usr/bin/env python3
"""Apply the frozen regularized finite-b transform to an isolated ensemble."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
PROJECT = BASE.parents[1]
TRANSFORMER = PROJECT / "workflows/v23a/construction/construct_v23a_regularized_kspace_tmd_v2.py"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bspace-ensemble", type=Path, required=True)
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--x", type=float, default=0.1)
    parser.add_argument("--Q", type=float, default=10.0)
    parser.add_argument("--flavor", action="append", default=None)
    return parser.parse_args()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = arguments()
    flavors = args.flavor or ["u", "d"]
    transform = load_module("campaign_regularized_transform", TRANSFORMER)
    all_curves = pd.read_csv(args.bspace_ensemble)
    curves = all_curves[
        np.isclose(all_curves["x"], args.x)
        & np.isclose(all_curves["Q"], args.Q)
        & all_curves["flavor"].astype(str).isin(flavors)
    ].copy()
    if not len(curves):
        raise RuntimeError("requested b-space curves are absent")
    expected = curves["run_tag"].nunique() * len(flavors)
    if curves.groupby(["run_tag", "flavor"]).ngroups != expected:
        raise RuntimeError("not every ensemble member covers every flavor")

    curves["_replica_key"] = curves["run_tag"].astype(str)
    curves["seed"] = curves["fit_seed"].fillna(-1).astype(int)
    curves["pdf_member"] = 0
    settings = argparse.Namespace(
        quantities=["ftilde"],
        tail_mode="expb2",
        tail_fit_bmin=None,
        eps=1.0e-300,
        b_transform_max=24.0,
        n_b_transform=6001,
        k_max=4.0,
        n_k=401,
        end_taper_start_fraction=0.92,
    )
    kspace, transform_meta = transform.transform_curves(curves, settings)
    bands = transform.make_bands(kspace)
    target = BASE / "summaries" / args.target_name
    target.mkdir(parents=True, exist_ok=True)
    kspace.to_csv(target / "kspace_tmd_ensemble_long.csv", index=False)
    bands.to_csv(target / "kspace_tmd_quantiles.csv", index=False)
    summary = {
        "status": "isolated_regularized_kspace_transform_not_production",
        "source_bspace_ensemble": str(args.bspace_ensemble.resolve()),
        "ensemble_member_count": int(curves["run_tag"].nunique()),
        "x": args.x,
        "Q_GeV": args.Q,
        "flavors": flavors,
        "transform": transform_meta,
        "quantiles": [0.16, 0.50, 0.84],
        "quantile_interpretation": (
            "descriptive until the input ensemble's statistical construction "
            "and endpoint stability gates pass"),
        "production_sources_modified": False,
    }
    (target / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
