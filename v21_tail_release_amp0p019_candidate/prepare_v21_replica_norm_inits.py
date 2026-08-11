#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def reproduce_replica(
    df: pd.DataFrame,
    seed: int,
) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    rng = np.random.default_rng(int(seed))
    central = df["CS"].to_numpy(float)
    sigma = df["sigma_uncorr"].to_numpy(float)
    target = central.copy()
    draws: dict[str, dict[str, float]] = {}

    for dataset, subset in df.groupby("dataset", sort=False):
        indices = subset.index.to_numpy()
        norm_rel = float(subset["norm_rel"].iloc[0])
        z = float(rng.normal())
        factor = 1.0 + z * norm_rel
        target[indices] = factor * central[indices]
        draws[str(dataset)] = {
            "z": z,
            "norm_rel": norm_rel,
            "replica_norm_factor": factor,
        }

    target = target + rng.normal(0.0, 1.0, size=len(df)) * sigma
    return target, draws


def profiled_scales(
    df: pd.DataFrame,
    target: np.ndarray,
) -> tuple[pd.DataFrame, float, float]:
    raw = df["pred_match_CS_raw_before_dataset_norm"].to_numpy(float)
    sigma = df["sigma_uncorr"].to_numpy(float)

    rows = []
    data_sum = 0.0
    penalty_sum = 0.0

    for dataset, subset in df.groupby("dataset", sort=False):
        indices = subset.index.to_numpy()
        norm_rel = float(subset["norm_rel"].iloc[0])
        central_scale = float(subset["dataset_norm_factor"].iloc[0])

        r = raw[indices]
        t = target[indices]
        s = sigma[indices]
        weights = 1.0 / np.square(s)

        # Exact optimum at fixed central raw theory for
        #   sum_i ((scale*r_i - target_i)/sigma_i)^2
        #   + ((scale-1)/delta)^2.
        denominator = float(np.sum(weights * np.square(r)) + 1.0 / norm_rel**2)
        numerator = float(np.sum(weights * r * t) + 1.0 / norm_rel**2)
        scale = numerator / denominator
        pull = (scale - 1.0) / norm_rel

        data_sum += float(np.sum(np.square((scale * r - t) / s)))
        penalty_sum += float(pull**2)

        rows.append({
            "dataset": str(dataset),
            "norm_scale": scale,
            "norm_pull": pull,
            "norm_rel": norm_rel,
            "central_norm_scale": central_scale,
        })

    n = len(df)
    return (
        pd.DataFrame(rows),
        data_sum / n,
        penalty_sum / n,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--central-predictions", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    source = Path(args.central_predictions)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(source).reset_index(drop=True)
    required = {
        "dataset",
        "CS",
        "sigma_uncorr",
        "norm_rel",
        "dataset_norm_factor",
        "pred_match_CS_raw_before_dataset_norm",
    }
    missing = required.difference(df.columns)
    if missing:
        raise SystemExit(
            f"Missing columns in {source}: {sorted(missing)}"
        )

    summary_rows = []

    for seed in args.seeds:
        target, draws = reproduce_replica(df, seed)
        norms, data_chi2, norm_penalty = profiled_scales(df, target)

        seed_dir = out_root / f"s{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        norms.to_csv(seed_dir / "dataset_norms.csv", index=False)

        target_hash = hashlib.sha256(
            np.asarray(target, dtype="<f8").tobytes()
        ).hexdigest()

        diagnostics = {
            "seed": int(seed),
            "target_sha256_float64": target_hash,
            "fixed_central_model_data_chi2": data_chi2,
            "normalization_penalty_per_point": norm_penalty,
            "fixed_central_model_objective": data_chi2 + norm_penalty,
            "replica_normalization_draws": draws,
            "profiled_dataset_norms": norms.to_dict(orient="records"),
        }
        with (seed_dir / "diagnostics.json").open("w") as handle:
            json.dump(diagnostics, handle, indent=2)

        summary_rows.append({
            "seed": int(seed),
            "data_chi2_epoch0": data_chi2,
            "norm_penalty_epoch0": norm_penalty,
            "objective_epoch0": data_chi2 + norm_penalty,
            "max_abs_norm_pull": float(norms["norm_pull"].abs().max()),
        })

        print(f"\n=== seed {seed} ===")
        print(norms.to_string(index=False))
        print("epoch-zero data chi2:", data_chi2)
        print("epoch-zero norm penalty:", norm_penalty)
        print("epoch-zero objective:", data_chi2 + norm_penalty)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_root / "summary.csv", index=False)

    print("\n=== replica normalization initialization summary ===")
    print(summary.to_string(index=False))
    print("\nwrote", out_root)


if __name__ == "__main__":
    main()
