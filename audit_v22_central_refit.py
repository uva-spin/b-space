#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_metrics(run: Path) -> dict:
    p = run / "metrics.json"
    if not p.exists():
        return {}
    with p.open() as handle:
        return json.load(handle)


def weighted_total(metrics: dict) -> float:
    per = pd.DataFrame(metrics.get("per_dataset", []))
    if per.empty:
        return float("nan")
    return float((per["chi2_like"] * per["n"]).sum() / per["n"].sum())


def norm_pull_summary(metrics: dict) -> dict[str, float]:
    pulls = metrics.get("dataset_norm_pulls", {})
    if not pulls:
        return {"max_abs_norm_pull": float("nan")}
    values = [abs(float(v)) for v in pulls.values()]
    return {"max_abs_norm_pull": float(max(values))}


def prediction_frame(label: str, run: Path) -> pd.DataFrame:
    p = run / "predictions.csv"
    if not p.exists():
        raise SystemExit(f"Missing {p}")
    frame = pd.read_csv(p)
    frame["run_label"] = label
    return frame


def mean_pull2(frame: pd.DataFrame, pred: str, data: str, sigma: str) -> float:
    good = (
        np.isfinite(frame[pred])
        & np.isfinite(frame[data])
        & np.isfinite(frame[sigma])
        & (frame[sigma] > 0)
    )
    pull = (frame.loc[good, pred] - frame.loc[good, data]) / frame.loc[good, sigma]
    return float(np.mean(pull.to_numpy(float) ** 2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--central-run", required=True)
    parser.add_argument("--warmcheck-run", default="outputs/v22_full_backend_warmcheck_s303")
    parser.add_argument("--refit-run", default="outputs/v22_full_backend_central_refit_stage1_s303")
    parser.add_argument("--out", default="v22/outputs/v22_central_refit_stage1_audit")
    parser.add_argument("--pass-chi2", type=float, default=1.05)
    parser.add_argument("--pass-norm-pull", type=float, default=4.0)
    args = parser.parse_args()

    runs = {
        "v21_frozen_central": Path(args.central_run),
        "v22_warmcheck_old_np": Path(args.warmcheck_run),
        "v22_refit_stage1": Path(args.refit_run),
    }

    metrics = {label: load_metrics(run) for label, run in runs.items()}
    summary_rows = []
    for label, m in metrics.items():
        row = {
            "run_label": label,
            "chi2_total": weighted_total(m),
        }
        row.update(norm_pull_summary(m))
        train = m.get("train", {})
        for key in ["best_epoch", "epochs_run", "best_objective", "restored_best"]:
            if key in train:
                row[key] = train[key]
        summary_rows.append(row)
    run_summary = pd.DataFrame(summary_rows)

    central = prediction_frame("central", runs["v21_frozen_central"])
    warm = prediction_frame("warm", runs["v22_warmcheck_old_np"])
    refit = prediction_frame("refit", runs["v22_refit_stage1"])

    keys = ["dataset", "row_id"]
    merged = (
        central[keys + ["pred_match_CS", "CS", "sigma_used"]]
        .rename(columns={"pred_match_CS": "pred_central", "CS": "data", "sigma_used": "sigma"})
        .merge(
            warm[keys + ["pred_match_CS"]].rename(columns={"pred_match_CS": "pred_warm"}),
            on=keys,
            validate="one_to_one",
        )
        .merge(
            refit[keys + ["pred_match_CS"]].rename(columns={"pred_match_CS": "pred_refit"}),
            on=keys,
            validate="one_to_one",
        )
    )

    merged["warm_over_central"] = merged["pred_warm"] / merged["pred_central"].replace(0, np.nan)
    merged["refit_over_central"] = merged["pred_refit"] / merged["pred_central"].replace(0, np.nan)
    merged["warm_delta_pull"] = (merged["pred_warm"] - merged["pred_central"]) / merged["sigma"].replace(0, np.nan)
    merged["refit_delta_pull"] = (merged["pred_refit"] - merged["pred_central"]) / merged["sigma"].replace(0, np.nan)
    merged["refit_pull_to_data"] = (merged["pred_refit"] - merged["data"]) / merged["sigma"].replace(0, np.nan)

    by_dataset = (
        merged.groupby("dataset", observed=False)
        .agg(
            n=("row_id", "size"),
            warm_ratio_median=("warm_over_central", "median"),
            refit_ratio_median=("refit_over_central", "median"),
            warm_delta_abs_p90=("warm_delta_pull", lambda x: float(np.nanquantile(np.abs(x), 0.90))),
            refit_delta_abs_p90=("refit_delta_pull", lambda x: float(np.nanquantile(np.abs(x), 0.90))),
            refit_chi2=("refit_pull_to_data", lambda x: float(np.nanmean(np.asarray(x) ** 2))),
        )
        .reset_index()
    )

    refit_metrics = metrics["v22_refit_stage1"]
    refit_chi2 = weighted_total(refit_metrics)
    max_norm = norm_pull_summary(refit_metrics)["max_abs_norm_pull"]

    passed = bool(
        np.isfinite(refit_chi2)
        and refit_chi2 < float(args.pass_chi2)
        and (
            not np.isfinite(max_norm)
            or max_norm < float(args.pass_norm_pull)
        )
    )

    decision = {
        "refit_chi2_total": refit_chi2,
        "max_abs_norm_pull": max_norm,
        "pass_chi2_threshold": float(args.pass_chi2),
        "pass_norm_pull_threshold": float(args.pass_norm_pull),
        "V22_STAGE1_CENTRAL_REFIT_PASS": passed,
        "next_if_pass": (
            "Construct v22 scheme-defined TMD grids and run post-peak/shape audits; "
            "then launch a small profiled replica pilot."
        ),
        "next_if_fail": (
            "Inspect training history and outliers; continue from checkpoint or adjust "
            "learning rate before changing physics settings."
        ),
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run_summary.to_csv(out / "central_refit_run_summary.csv", index=False)
    merged.to_csv(out / "central_refit_prediction_comparison.csv", index=False)
    by_dataset.to_csv(out / "central_refit_by_dataset.csv", index=False)
    (out / "central_refit_decision.json").write_text(json.dumps(decision, indent=2) + "\n")

    print("\n=== Run summary ===")
    print(run_summary.to_string(index=False))
    print("\n=== By dataset ===")
    print(by_dataset.to_string(index=False))
    print("\n=== Decision ===")
    print(json.dumps(decision, indent=2))
    print("\nwrote:", out)

    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
