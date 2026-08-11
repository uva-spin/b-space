#!/usr/bin/env python3
"""Audit reference-distance regularization across completed isolated campaigns.

This is read-only with respect to production inputs.  It decomposes the stored
objective into the unpenalized cross-section fit and reference-distance term,
then measures the resulting direct-F_NP distance and start spread at x=0.1.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
OUTPUTS = BASE / "outputs"
REFERENCE = BASE / "summaries/exact_baseline_fnp_median/fnp_median.csv"
TARGET = BASE / "summaries/reference_distance_objective_decomposition"


def lambda1_new_tags() -> list[str]:
    summary = json.loads(
        (BASE / "summaries/lambda1_start_expansion48/summary.json").read_text()
    )
    return list(summary["new_start_terminal_tags"])


def lambda600_tags() -> list[str]:
    summary = json.loads(
        (BASE / "summaries/lambda600_like_for_like_decision/summary.json").read_text()
    )
    return list(summary["full24"]["endpoint_tags"])


def endpoint_groups() -> dict[str, list[str]]:
    return {
        "lambda1_old24": [
            f"exactbaseline_matched_reference_distance_b0p1_2p0_lam1e00_s{s}"
            for s in range(303, 327)
        ],
        "lambda1_new24": lambda1_new_tags(),
        "lambda300_full24": [
            f"lambda300_candidate_full24_s{s}_polish64_300000"
            for s in range(303, 327)
        ],
        "lambda600_full24": lambda600_tags(),
    }


def direct_reference_metrics(grid: pd.DataFrame, reference: pd.DataFrame) -> dict[str, float]:
    merged = grid.merge(reference, on=["x", "bT"], suffixes=("_model", "_ref"))
    out: dict[str, float] = {}
    for bmax in (2.0, 4.0):
        native = merged[(merged["bT"] >= 0.1) & (merged["bT"] <= bmax)]
        relative = (native["F_NP_model"] - native["F_NP_ref"]) / np.maximum(
            native["F_NP_ref"], 0.05
        )
        out[f"reference_relative_rms_bmax{bmax:g}"] = float(np.sqrt(np.mean(relative**2)))
        out[f"reference_relative_maxabs_bmax{bmax:g}"] = float(np.max(np.abs(relative)))
        x01 = native[np.isclose(native["x"], 0.1)]
        rel01 = (x01["F_NP_model"] - x01["F_NP_ref"]) / np.maximum(
            x01["F_NP_ref"], 0.05
        )
        out[f"reference_relative_rms_x0p1_bmax{bmax:g}"] = float(np.sqrt(np.mean(rel01**2)))
        out[f"reference_relative_maxabs_x0p1_bmax{bmax:g}"] = float(np.max(np.abs(rel01)))
    return out


def collect_endpoint_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group, tags in endpoint_groups().items():
        for tag in tags:
            status_path = OUTPUTS / tag / "fit_status.json"
            grid_path = OUTPUTS / tag / "fnp_grid.csv"
            if not status_path.exists() or not grid_path.exists():
                raise FileNotFoundError(f"incomplete endpoint: {tag}")
            status = json.loads(status_path.read_text())
            final = status["final"]
            regularization = status["regularization"]["fnp_reference_distance"]
            lam = float(regularization["lambda"])
            declared_reference_path = Path(str(regularization["target_csv"]))
            declared_reference = pd.read_csv(declared_reference_path)
            row: dict[str, object] = {
                "group": group,
                "tag": tag,
                "seed": int(status["seed"]),
                "lambda": lam,
                "reference_bmin": float(regularization["b_min"]),
                "reference_bmax": float(regularization["b_max"]),
                "declared_reference": str(declared_reference_path),
                "data_chi2": float(final["data_chi2"]),
                "norm_penalty": float(final["norm_penalty"]),
                "unpenalized_total_chi2": float(final["unpenalized_total_chi2"]),
                "total_chi2": float(final["total_chi2"]),
                "objective_per_row": float(final["objective_per_row"]),
                "weighted_likelihood_per_row": float(final["weighted_likelihood_per_row_objective"]),
                "reference_penalty_per_row": float(final["reference_distance_penalty_per_row_objective"]),
                "raw_reference_mse_per_row": float(
                    final["reference_distance_penalty_per_row_objective"] / lam
                ),
                "reference_term_fraction_of_weighted_likelihood": float(
                    final["reference_distance_penalty_per_row_objective"]
                    / max(final["weighted_likelihood_per_row_objective"], 1.0e-30)
                ),
                "max_prediction_shift_over_experimental_sigma": float(
                    final["max_prediction_shift_over_experimental_sigma"]
                ),
                "runner_stopped_on_plateau": bool(status.get("stopped_on_plateau", False)),
                "runner_convergence_gate_pass": bool(status.get("convergence_gate_pass", False)),
            }
            row.update(direct_reference_metrics(pd.read_csv(grid_path), declared_reference))
            rows.append(row)
    return pd.DataFrame(rows)


def spread_table(reference: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group, tags in endpoint_groups().items():
        grids = []
        for tag in tags:
            grid = pd.read_csv(OUTPUTS / tag / "fnp_grid.csv")
            grids.append(grid[grid["x"].eq(0.1)][["bT", "F_NP"]].set_index("bT")["F_NP"])
        values = pd.concat(grids, axis=1).to_numpy(float)
        b = pd.concat(grids, axis=1).index.to_numpy(float)
        for bmax in (2.0, 4.0):
            mask = (b >= 0.1) & (b <= bmax)
            median = np.median(values[mask], axis=1)
            q16, q84 = np.quantile(values[mask], [0.16, 0.84], axis=1)
            scale = np.maximum(median, 0.05)
            full = (values[mask].max(axis=1) - values[mask].min(axis=1)) / scale
            qwidth = (q84 - q16) / scale
            rows.append(
                {
                    "group": group,
                    "member_count": len(tags),
                    "bmax": bmax,
                    "max_q16_q84_relative_width": float(np.max(qwidth)),
                    "median_q16_q84_relative_width": float(np.median(qwidth)),
                    "max_full_start_range_relative": float(np.max(full)),
                    "median_full_start_range_relative": float(np.median(full)),
                }
            )
    return pd.DataFrame(rows)


def plot_endpoint_tradeoffs(endpoint: pd.DataFrame, spread: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), constrained_layout=True)
    colors = {"lambda1_old24": "#0072B2", "lambda1_new24": "#56B4E9", "lambda300_full24": "#E69F00", "lambda600_full24": "#D55E00"}
    labels = {"lambda1_old24": "lambda=1 old 24", "lambda1_new24": "lambda=1 new 24", "lambda300_full24": "lambda=300", "lambda600_full24": "lambda=600"}
    for group, frame in endpoint.groupby("group", sort=False):
        axes[0].scatter(frame["raw_reference_mse_per_row"], frame["unpenalized_total_chi2"], s=18, alpha=0.75, color=colors[group], label=labels[group])
        axes[1].scatter(frame["raw_reference_mse_per_row"], frame["reference_penalty_per_row"], s=18, alpha=0.75, color=colors[group])
    axes[0].set_xlabel("raw reference MSE per row")
    axes[0].set_ylabel("unpenalized total $\\chi^2$")
    axes[1].set_xlabel("raw reference MSE per row")
    axes[1].set_ylabel("weighted reference term per row")
    for group, frame in spread[spread["bmax"].eq(2.0)].groupby("group", sort=False):
        axes[2].scatter(frame["max_q16_q84_relative_width"], frame["max_full_start_range_relative"], s=45, color=colors[group], label=labels[group])
    axes[2].set_xlabel("max q16--q84 $F_{NP}$ width")
    axes[2].set_ylabel("max full start range")
    axes[2].set_xlim(left=0)
    axes[2].set_ylim(bottom=0)
    axes[0].legend(frameon=False, fontsize=7)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(TARGET / "reference_distance_tradeoffs.png", dpi=220)
    fig.savefig(TARGET / "reference_distance_tradeoffs.pdf")
    plt.close(fig)


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    reference = pd.read_csv(REFERENCE)
    endpoint = collect_endpoint_rows()
    spread = spread_table(reference)
    endpoint.to_csv(TARGET / "endpoint_objective_decomposition.csv", index=False)
    spread.to_csv(TARGET / "fnp_start_spread.csv", index=False)
    plot_endpoint_tradeoffs(endpoint, spread)
    group_summary = (
        endpoint.groupby("group")
        .agg(
            member_count=("tag", "size"),
            lambda_value=("lambda", "first"),
            data_chi2_median=("data_chi2", "median"),
            data_chi2_min=("data_chi2", "min"),
            data_chi2_max=("data_chi2", "max"),
            unpenalized_chi2_median=("unpenalized_total_chi2", "median"),
            total_chi2_median=("total_chi2", "median"),
            weighted_reference_term_median=("reference_penalty_per_row", "median"),
            raw_reference_mse_median=("raw_reference_mse_per_row", "median"),
            reference_term_fraction_median=("reference_term_fraction_of_weighted_likelihood", "median"),
            direct_reference_rms_median=("reference_relative_rms_bmax2", "median"),
            direct_reference_rms_bmax4_median=("reference_relative_rms_bmax4", "median"),
            prediction_shift_sigma_max=("max_prediction_shift_over_experimental_sigma", "max"),
        )
        .reset_index()
    )
    summary = {
        "status": "complete_isolated_objective_decomposition",
        "question": "whether reference-distance regularization is ineffective or mis-scaled",
        "reference": str(REFERENCE),
        "group_summary": group_summary.to_dict(orient="records"),
        "fnp_start_spread": spread.to_dict(orient="records"),
        "interpretation": {
            "lambda_effect": "Increasing lambda strongly reduces the raw direct-FNP distance to the declared reference, but the resulting weighted term grows and the cross-section fit/replica stability must be judged separately.",
            "lambda1_effective_control": "lambda=1 is weak in the sense that the 48-start expansion exposes substantially larger downstream TMD variation than the incumbent 24 starts.",
            "not_a_complete_uniqueness_test": "The audit compares completed endpoints; it does not turn a fixed reference prior into a statistical uniqueness guarantee.",
            "production_state_modified": False,
        },
    }
    (TARGET / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
