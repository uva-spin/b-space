#!/usr/bin/env python3
"""Summarize available DYTurbo/MCFM finite-tail benchmark coverage.

This is intentionally conservative: rows are marked externally benchmarked only
when both DYTurbo and MCFM outputs exist for the same processed row and agree
after explicit unit conversion.
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
SYS = ROOT / "systematics" / "finite_y_tail_benchmark"
DATA_DIR = ROOT / "Data" / "v23a_tevatron_plus_lhcb7_fiducial_candidate"

COLLIDER_DATASETS = ["CDF_RUN_1", "CDF_RUN_2", "D0_RUN_1", "LHCb_7"]
STRICT_R = 0.10
COLLINS_ACCEPTED_R = 0.20
EXTERNAL_AGREE_WARN = 0.05


def _load_processed_rows() -> pd.DataFrame:
    frames = []
    for dataset in COLLIDER_DATASETS:
        path = DATA_DIR / f"{dataset}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["source_table"] = str(path)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No collider data tables found in {DATA_DIR}")
    rows = pd.concat(frames, ignore_index=True)
    rows["region"] = np.select(
        [
            rows["qT_over_Q"] <= STRICT_R,
            rows["qT_over_Q"] <= COLLINS_ACCEPTED_R,
        ],
        [
            "strict_core",
            "collins_envelope",
        ],
        default="high_qT_candidate",
    )
    return rows


def _load_cdf_run2_external() -> pd.DataFrame:
    dy_path = ROOT / "outputs" / "tevatron_dyturbo_benchmark_cdf_run2" / "dyturbo_benchmark_summary.csv"
    mcfm_path = ROOT / "outputs" / "tevatron_mcfm_benchmark_cdf_run2" / "mcfm_benchmark_summary.csv"
    if not dy_path.exists() or not mcfm_path.exists():
        return pd.DataFrame()

    dy = pd.read_csv(dy_path)
    mcfm = pd.read_csv(mcfm_path)
    keep = [
        "dataset",
        "row_id",
        "qT",
        "qT_low",
        "qT_high",
        "qT_over_Q",
        "QM_Low",
        "QM_High",
        "data_pb_per_GeV",
        "data_bin_pb",
        "dyturbo_raw",
        "dyturbo_raw_unc",
        "card",
        "log",
        "txt",
    ]
    dy = dy[[c for c in keep if c in dy.columns]].copy()
    dy = dy.rename(columns={"card": "dyturbo_card", "log": "dyturbo_log", "txt": "dyturbo_txt"})

    # DYTurbo text values in these artifacts are fb/bin. Convert to pb/bin and pb/GeV.
    bin_width = dy["qT_high"] - dy["qT_low"]
    dy["dyturbo_pb_bin"] = dy["dyturbo_raw"] / 1000.0
    dy["dyturbo_pb_bin_unc"] = dy["dyturbo_raw_unc"] / 1000.0
    dy["dyturbo_pb_per_GeV"] = dy["dyturbo_pb_bin"] / bin_width
    dy["dyturbo_pb_per_GeV_unc"] = dy["dyturbo_pb_bin_unc"] / bin_width

    mkeep = [
        "row_id",
        "mcfm_pb_bin",
        "mcfm_pb_bin_unc",
        "mcfm_pb_per_GeV",
        "mcfm_pb_per_GeV_unc",
        "log",
    ]
    mcfm = mcfm[[c for c in mkeep if c in mcfm.columns]].copy()
    mcfm = mcfm.rename(columns={"log": "mcfm_log"})

    merged = dy.merge(mcfm, on="row_id", how="inner")
    avg = 0.5 * (merged["dyturbo_pb_per_GeV"].abs() + merged["mcfm_pb_per_GeV"].abs())
    merged["dyturbo_mcfm_rel_diff"] = (
        (merged["dyturbo_pb_per_GeV"] - merged["mcfm_pb_per_GeV"]).abs() / avg.replace(0.0, np.nan)
    )
    merged["external_code_agreement_pass"] = merged["dyturbo_mcfm_rel_diff"] <= EXTERNAL_AGREE_WARN
    merged["mcfm_scale"] = 1.0
    merged["scale_note"] = ""
    merged["benchmark_scope"] = "CDF Run II rapidity-inclusive Z pT rows; mass window and pT bins from processed table"
    return merged


def _parse_mcfm_log_result(path: Path) -> tuple[float, float, str] | None:
    if not path.exists():
        return None
    text = path.read_text(errors="replace")
    if "=== Result for PDF set" not in text:
        return None
    matches = re.findall(
        r"Value of integral is\s+([-+0-9.Ee]+)\s*(?:±|\+/-)?\s*([-+0-9.Ee]+)\s+([fp]b)",
        text,
    )
    if not matches:
        return None
    value, unc, unit = matches[-1]
    return float(value), float(unc), unit


def _canonicalize_external_pair(
    dy_path: Path,
    mcfm_path: Path,
    *,
    scope: str,
    mcfm_scale: float = 1.0,
    scale_note: str = "",
) -> pd.DataFrame:
    if not dy_path.exists() or not mcfm_path.exists():
        return pd.DataFrame()

    dy = pd.read_csv(dy_path)
    mcfm = pd.read_csv(mcfm_path)
    records = []
    for _, row in dy.iterrows():
        match = mcfm[mcfm["row_id"].eq(row["row_id"])]
        if match.empty:
            continue
        mrow = match.iloc[-1]
        bin_width = float(row["qT_high"]) - float(row["qT_low"])
        dy_bin_pb = float(row["dyturbo_raw"]) / 1000.0
        dy_unc_pb = float(row["dyturbo_raw_unc"]) / 1000.0
        mcfm_pb_bin = float(mrow["mcfm_pb_bin"]) * float(mcfm_scale)
        mcfm_pb_bin_unc = float(mrow["mcfm_pb_bin_unc"]) * float(mcfm_scale)
        mcfm_pb_per_GeV = float(mrow["mcfm_pb_per_GeV"]) * float(mcfm_scale)
        mcfm_pb_per_GeV_unc = float(mrow["mcfm_pb_per_GeV_unc"]) * float(mcfm_scale)
        records.append(
            {
                "dataset": row["dataset"],
                "row_id": row["row_id"],
                "qT": row["qT"],
                "qT_low": row["qT_low"],
                "qT_high": row["qT_high"],
                "qT_over_Q": row["qT_over_Q"],
                "QM_Low": row["QM_Low"],
                "QM_High": row["QM_High"],
                "data_pb_per_GeV": row["data_pb_per_GeV"],
                "data_bin_pb": row["data_bin_pb"],
                "dyturbo_pb_bin": dy_bin_pb,
                "dyturbo_pb_bin_unc": dy_unc_pb,
                "dyturbo_pb_per_GeV": dy_bin_pb / bin_width,
                "dyturbo_pb_per_GeV_unc": dy_unc_pb / bin_width,
                "mcfm_pb_bin": mcfm_pb_bin,
                "mcfm_pb_bin_unc": mcfm_pb_bin_unc,
                "mcfm_pb_per_GeV": mcfm_pb_per_GeV,
                "mcfm_pb_per_GeV_unc": mcfm_pb_per_GeV_unc,
                "mcfm_scale": float(mcfm_scale),
                "scale_note": scale_note,
                "dyturbo_card": row.get("card", np.nan),
                "dyturbo_log": row.get("log", np.nan),
                "dyturbo_txt": row.get("txt", np.nan),
                "mcfm_log": mrow.get("log", np.nan),
                "benchmark_scope": scope,
            }
        )
    if not records:
        return pd.DataFrame()
    merged = pd.DataFrame(records)
    avg = 0.5 * (merged["dyturbo_pb_per_GeV"].abs() + merged["mcfm_pb_per_GeV"].abs())
    merged["dyturbo_mcfm_rel_diff"] = (
        (merged["dyturbo_pb_per_GeV"] - merged["mcfm_pb_per_GeV"]).abs() / avg.replace(0.0, np.nan)
    )
    merged["external_code_agreement_pass"] = merged["dyturbo_mcfm_rel_diff"] <= EXTERNAL_AGREE_WARN
    return merged


def _load_cdf_run1_external() -> pd.DataFrame:
    return _canonicalize_external_pair(
        SYS / "outputs" / "cdf_run1_dyturbo" / "dyturbo_benchmark_summary.csv",
        SYS / "outputs" / "cdf_run1_mcfm" / "mcfm_benchmark_summary.csv",
        scope="CDF Run I rapidity-inclusive Z pT rows; mass window and pT bins from processed table",
    )


def _load_d0_run1_external() -> pd.DataFrame:
    return _canonicalize_external_pair(
        SYS / "outputs" / "d0_run1_dyturbo" / "dyturbo_benchmark_summary.csv",
        SYS / "outputs" / "d0_run1_mcfm" / "mcfm_benchmark_summary.csv",
        scope="D0 Run I rapidity-inclusive Z pT rows; mass window and pT bins from processed table",
    )


def _load_lhcb7_external() -> pd.DataFrame:
    return _canonicalize_external_pair(
        SYS / "outputs" / "lhcb7_dyturbo" / "dyturbo_benchmark_summary.csv",
        SYS / "outputs" / "lhcb7_mcfm" / "mcfm_benchmark_summary.csv",
        scope="LHCb 7 TeV pp forward dimuon fiducial rows; positive-arm DYTurbo compared to half of MCFM absolute-eta fiducial result",
        mcfm_scale=0.5,
        scale_note="MCFM etaleptmin/etaleptmax cuts are absolute-value cuts and include both pp forward arms; divide by 2 for the single positive LHCb arm.",
    )


def _load_cdf_run1_partial_external() -> pd.DataFrame:
    dy_path = SYS / "outputs" / "cdf_run1_dyturbo" / "dyturbo_benchmark_summary.csv"
    mcfm_logs = SYS / "outputs" / "cdf_run1_mcfm" / "logs"
    if not dy_path.exists() or not mcfm_logs.exists():
        return pd.DataFrame()

    dy = pd.read_csv(dy_path)
    records = []
    for _, row in dy.iterrows():
        row_number = str(row["row_id"]).split(":")[-1]
        log = mcfm_logs / f"tev{row_number}.log"
        parsed = _parse_mcfm_log_result(log)
        if parsed is None:
            continue
        value, unc, unit = parsed
        unit_to_pb = 1.0 if unit == "pb" else 1.0 / 1000.0
        bin_width = float(row["qT_high"]) - float(row["qT_low"])
        dy_bin_pb = float(row["dyturbo_raw"]) / 1000.0
        dy_unc_pb = float(row["dyturbo_raw_unc"]) / 1000.0
        mcfm_bin_pb = value * unit_to_pb
        mcfm_unc_pb = unc * unit_to_pb
        records.append(
            {
                "dataset": row["dataset"],
                "row_id": row["row_id"],
                "qT": row["qT"],
                "qT_low": row["qT_low"],
                "qT_high": row["qT_high"],
                "qT_over_Q": row["qT_over_Q"],
                "QM_Low": row["QM_Low"],
                "QM_High": row["QM_High"],
                "data_pb_per_GeV": row["data_pb_per_GeV"],
                "data_bin_pb": row["data_bin_pb"],
                "dyturbo_pb_bin": dy_bin_pb,
                "dyturbo_pb_bin_unc": dy_unc_pb,
                "dyturbo_pb_per_GeV": dy_bin_pb / bin_width,
                "dyturbo_pb_per_GeV_unc": dy_unc_pb / bin_width,
                "mcfm_pb_bin": mcfm_bin_pb,
                "mcfm_pb_bin_unc": mcfm_unc_pb,
                "mcfm_pb_per_GeV": mcfm_bin_pb / bin_width,
                "mcfm_pb_per_GeV_unc": mcfm_unc_pb / bin_width,
                "dyturbo_card": row.get("card", np.nan),
                "dyturbo_log": row.get("log", np.nan),
                "dyturbo_txt": row.get("txt", np.nan),
                "mcfm_log": str(log),
                "benchmark_scope": "CDF Run I rapidity-inclusive Z pT rows; mass window and pT bins from processed table",
            }
        )
    if not records:
        return pd.DataFrame()
    merged = pd.DataFrame(records)
    avg = 0.5 * (merged["dyturbo_pb_per_GeV"].abs() + merged["mcfm_pb_per_GeV"].abs())
    merged["dyturbo_mcfm_rel_diff"] = (
        (merged["dyturbo_pb_per_GeV"] - merged["mcfm_pb_per_GeV"]).abs() / avg.replace(0.0, np.nan)
    )
    merged["external_code_agreement_pass"] = merged["dyturbo_mcfm_rel_diff"] <= EXTERNAL_AGREE_WARN
    return merged


def main() -> None:
    summary_dir = SYS / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_processed_rows()
    cdf_run2_external = _load_cdf_run2_external()
    cdf_run1_external = _load_cdf_run1_external()
    if cdf_run1_external.empty:
        cdf_run1_external = _load_cdf_run1_partial_external()
    d0_run1_external = _load_d0_run1_external()
    lhcb7_external = _load_lhcb7_external()
    external = pd.concat([cdf_run2_external, cdf_run1_external, d0_run1_external, lhcb7_external], ignore_index=True)

    inventory = (
        rows.groupby(["dataset", "region"], dropna=False)
        .agg(
            n_rows=("row_id", "count"),
            min_qT_over_Q=("qT_over_Q", "min"),
            max_qT_over_Q=("qT_over_Q", "max"),
        )
        .reset_index()
        .sort_values(["dataset", "region"])
    )

    gate_cols = [
        "row_id",
        "dyturbo_pb_per_GeV",
        "dyturbo_pb_per_GeV_unc",
        "mcfm_pb_per_GeV",
        "mcfm_pb_per_GeV_unc",
        "dyturbo_mcfm_rel_diff",
        "external_code_agreement_pass",
        "benchmark_scope",
        "dyturbo_card",
        "dyturbo_log",
        "mcfm_log",
    ]
    gate = rows[
        [
            "dataset",
            "row_id",
            "qT",
            "qT_low",
            "qT_high",
            "qT_over_Q",
            "QM_Low",
            "QM_High",
            "y_Low",
            "y_High",
            "CS",
            "error",
            "region",
            "source_table",
        ]
    ].copy()
    if not external.empty:
        gate = gate.merge(external[[c for c in gate_cols if c in external.columns]], on="row_id", how="left")
    else:
        for col in gate_cols:
            if col != "row_id":
                gate[col] = np.nan

    gate["external_benchmark_available"] = gate["dyturbo_mcfm_rel_diff"].notna()
    gate["tail_benchmark_status"] = np.select(
        [
            gate["region"].eq("strict_core"),
            gate["external_benchmark_available"] & gate["external_code_agreement_pass"].fillna(False),
            gate["external_benchmark_available"] & ~gate["external_code_agreement_pass"].fillna(False),
        ],
        [
            "not_required_strict_core",
            "external_pass",
            "external_disagreement",
        ],
        default="pending_external_benchmark",
    )
    gate["production_high_qT_action"] = np.select(
        [
            gate["region"].eq("strict_core"),
            gate["tail_benchmark_status"].eq("external_pass"),
            gate["region"].eq("collins_envelope"),
        ],
        [
            "eligible_without_tail_gate",
            "eligible_as_tail_benchmarked_candidate",
            "eligible_only_with_factorization_validity_uncertainty_until_tail_benchmarked",
        ],
        default="exclude_from_production_until_tail_benchmarked",
    )

    summary = {
        "n_collider_rows_total": int(len(gate)),
        "n_strict_core": int(gate["region"].eq("strict_core").sum()),
        "n_collins_envelope": int(gate["region"].eq("collins_envelope").sum()),
        "n_high_qT_candidate": int(gate["region"].eq("high_qT_candidate").sum()),
        "n_external_benchmark_available": int(gate["external_benchmark_available"].sum()),
        "n_external_pass": int(gate["tail_benchmark_status"].eq("external_pass").sum()),
        "external_agreement_warn": EXTERNAL_AGREE_WARN,
    }
    pd.DataFrame([summary]).to_csv(summary_dir / "tail_benchmark_summary.csv", index=False)
    inventory.to_csv(summary_dir / "collider_row_inventory_by_region.csv", index=False)
    gate.to_csv(summary_dir / "tail_benchmark_row_gate.csv", index=False)

    if not cdf_run2_external.empty:
        cdf_run2_external.to_csv(summary_dir / "cdf_run2_dyturbo_mcfm_canonical.csv", index=False)
    if not cdf_run1_external.empty:
        cdf_run1_external.to_csv(summary_dir / "cdf_run1_dyturbo_mcfm_canonical.csv", index=False)
    if not d0_run1_external.empty:
        d0_run1_external.to_csv(summary_dir / "d0_run1_dyturbo_mcfm_canonical.csv", index=False)
    if not lhcb7_external.empty:
        lhcb7_external.to_csv(summary_dir / "lhcb7_dyturbo_mcfm_canonical.csv", index=False)
    if not external.empty:
        external.to_csv(summary_dir / "external_tail_benchmarks_canonical.csv", index=False)

    print(pd.DataFrame([summary]).to_string(index=False))
    print(f"wrote {summary_dir / 'tail_benchmark_summary.csv'}")
    print(f"wrote {summary_dir / 'collider_row_inventory_by_region.csv'}")
    print(f"wrote {summary_dir / 'tail_benchmark_row_gate.csv'}")


if __name__ == "__main__":
    main()
