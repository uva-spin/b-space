#!/usr/bin/env python3
"""Summarize dataset/start ambiguity without assigning it a probability law."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
PROJECT = BASE.parents[1]
SOURCE_B = BASE / "summaries/decision_preview_all_polished/bspace_tmd_ensemble_long.csv"
SOURCE_K = BASE / "summaries/decision_preview_all_polished_kspace/kspace_tmd_ensemble_long.csv"
TARGET = BASE / "summaries/decision_preview_uncertainty_decomposition"
EXP_ROOT = PROJECT / "plots/prd_q020_figures"


def candidate(tag: pd.Series) -> pd.Series:
    return tag.str.replace(r"_s7\d\d_polish64$", "", regex=True)


def components(frame: pd.DataFrame, coordinate: str, value: str) -> pd.DataFrame:
    keys = ["flavor", coordinate]
    frame = frame.copy()
    frame["candidate"] = candidate(frame["run_tag"])
    medians = frame.groupby(keys + ["candidate"], as_index=False)[value].median()
    pooled = frame.groupby(keys)[value].agg(
        pooled_low="min", pooled_median="median", pooled_high="max").reset_index()
    between = medians.groupby(keys)[value].agg(
        dataset_low="min", dataset_median="median", dataset_high="max").reset_index()
    within = frame.merge(
        medians.rename(columns={value: "candidate_median"}),
        on=keys + ["candidate"], validate="many_to_one")
    within["excursion"] = (within[value] - within["candidate_median"]).abs()
    within = within.groupby(keys, as_index=False)["excursion"].max().rename(
        columns={"excursion": "max_within_dataset_start_excursion"})
    return pooled.merge(between, on=keys).merge(within, on=keys)


def relative_metrics(table: pd.DataFrame, coordinate: str, maximum: float) -> dict:
    result = {}
    for flavor, group in table.groupby("flavor"):
        group = group[group[coordinate] <= maximum].copy()
        center = group["pooled_median"].to_numpy(float)
        active = center > 0.05 * np.max(center)
        scale = np.maximum(np.abs(center), 1.0e-30)
        pooled = np.maximum(
            group["pooled_high"].to_numpy(float) - center,
            center - group["pooled_low"].to_numpy(float)) / scale
        datasets = np.maximum(
            group["dataset_high"].to_numpy(float) - group["dataset_median"].to_numpy(float),
            group["dataset_median"].to_numpy(float) - group["dataset_low"].to_numpy(float),
        ) / np.maximum(np.abs(group["dataset_median"].to_numpy(float)), 1.0e-30)
        starts = group["max_within_dataset_start_excursion"].to_numpy(float) / scale
        result[str(flavor)] = {
            "active_coordinate_max": float(group.loc[active, coordinate].max()),
            "max_pooled_directional_excursion": float(np.max(pooled[active])),
            "median_pooled_directional_excursion": float(np.median(pooled[active])),
            "max_between_dataset_median_excursion": float(np.max(datasets[active])),
            "max_within_dataset_start_excursion": float(np.max(starts[active])),
        }
    return result


def plot_fig6(table: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    colors = {"u": "#0072B2", "d": "#D55E00"}
    for flavor in ("u", "d"):
        group = table[(table.flavor == flavor) & (table.kT <= 2.25)]
        exp = pd.read_csv(
            EXP_ROOT / f"kspace_fixedx_q10_{flavor}_current/v23a_regularized_kspace_bands.csv")
        exp = exp[
            exp.quantity.eq("ftilde") & exp.flavor.astype(str).eq(flavor)
            & np.isclose(exp.x, 0.1) & np.isclose(exp.Q, 10.0)
        ][["kT", "q16", "median", "q84"]]
        group = group.merge(exp, on="kT", validate="one_to_one")
        center = group.pooled_median.to_numpy(float)
        exp_low = center + group.q16.to_numpy(float) - group["median"].to_numpy(float)
        exp_high = center + group.q84.to_numpy(float) - group["median"].to_numpy(float)
        ax.fill_between(group.kT, exp_low, exp_high, color=colors[flavor], alpha=0.16)
        ax.fill_between(
            group.kT, group.pooled_low, group.pooled_high,
            facecolor="none", edgecolor=colors[flavor], hatch="////", linewidth=0.0,
            label=f"{flavor}: dataset/start envelope")
        ax.plot(group.kT, center, color=colors[flavor], lw=2.0, label=f"{flavor} median")
    ax.text(0.98, 0.96, r"$x=0.1\qquad Q=10\ \mathrm{GeV}$",
            ha="right", va="top", transform=ax.transAxes)
    ax.set(xlim=(0, 2.25), ylim=(0, None), xlabel=r"$k_T\ (\mathrm{GeV})$",
           ylabel=r"$f_1^q(x,k_T;Q)$")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig(TARGET / "fig6_decision_preview.png", dpi=220)
    fig.savefig(TARGET / "fig6_decision_preview.pdf")
    plt.close(fig)


def plot_fig2(table: pd.DataFrame) -> None:
    flavors = ["u", "d", "s", "ubar", "dbar", "sbar"]
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.0), sharex=True, constrained_layout=True)
    for ax, flavor in zip(axes.flat, flavors):
        group = table[(table.flavor == flavor) & (table.bT <= 4.0)]
        ax.fill_between(group.bT, group.pooled_low, group.pooled_high,
                        color="#56B4E9", alpha=0.30)
        ax.plot(group.bT, group.pooled_median, color="black", lw=1.6)
        ax.set_title(flavor)
        ax.grid(alpha=0.18)
    fig.supxlabel(r"$b_T\ (\mathrm{GeV}^{-1})$")
    fig.supylabel(r"$\widetilde f_1^q(x,b_T;Q)$")
    fig.suptitle(r"Decision preview: $x=0.1,\ Q=7.5\ \mathrm{GeV}$; dataset/start envelope")
    fig.savefig(TARGET / "fig2_decision_preview.png", dpi=220)
    fig.savefig(TARGET / "fig2_decision_preview.pdf")
    plt.close(fig)


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    b = pd.read_csv(SOURCE_B)
    b = b[np.isclose(b.Q, 7.5)].copy()
    k = pd.read_csv(SOURCE_K).rename(columns={"_replica_key": "run_tag"})
    b_table = components(b, "bT", "ftilde")
    k_table = components(k, "kT", "value")
    b_table.to_csv(TARGET / "fig2_components.csv", index=False)
    k_table.to_csv(TARGET / "fig6_components.csv", index=False)
    plot_fig2(b_table)
    plot_fig6(k_table)
    summary = {
        "status": "decision_preview_not_a_confidence_interval",
        "input": "33 completed polish64 fits: 11 dataset selections x 3 starts",
        "center": "pointwise median of the 33 descriptive curves",
        "model_band": "pointwise min/max dataset-and-start envelope; no probability interpretation",
        "experimental_overlay_fig6": (
            "frozen conditional 68% replica excursions transferred to the descriptive median; "
            "shown separately and not combined"),
        "fig2_active_region_metrics": relative_metrics(b_table, "bT", 4.0),
        "fig6_active_region_metrics": relative_metrics(k_table, "kT", 2.25),
        "warning": (
            "This preview cannot define a new central estimator or full one-sigma band because "
            "dataset selections and optimization starts have no validated sampling weights."),
        "production_sources_modified": False,
    }
    (TARGET / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
