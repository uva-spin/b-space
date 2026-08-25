#!/usr/bin/env python3
"""Reduce the frozen MCFM/DYTurbo artifacts into candidate tail targets.

This is a provenance reducer, not a physics benchmark yet.  It summarizes
which historical logs exist, the final integral in each MCFM log, candidate
point/cut labels from filenames, DYTurbo row numbers from filenames, and the
mapped fixed-target kinematics.

Run from ~/work/bT-TMD after freeze_v22_external_benchmark_artifacts.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd


POINT_LABELS = [
    "onepoint",
    "secondpoint",
    "thirdpoint",
    "fourthpoint",
]

CUT_LABELS = [
    "bridgecut",
    "bridge",
    "nearcut",
    "midcut",
    "widecheck",
    "widecut",
    "patched",
    "rowbin",
]

ROW_RE = re.compile(r"v15_E288_400_(\d+)_dyturbo", re.IGNORECASE)

SETTING_PATTERNS = {
    "runstring": re.compile(r"\brunstring\s*=\s*(.+)", re.IGNORECASE),
    "sqrts": re.compile(r"\bsqrts\s*=\s*([0-9Ee.+-]+)", re.IGNORECASE),
    "m34min": re.compile(r"\bm34min\s*=\s*([0-9Ee.+-]+)", re.IGNORECASE),
    "m34max": re.compile(r"\bm34max\s*=\s*([0-9Ee.+-]+)", re.IGNORECASE),
    "pt34min": re.compile(r"\bpt34min\s*=\s*([0-9Ee.+-]+)", re.IGNORECASE),
    "pt34max": re.compile(r"\bpt34max\s*=\s*([0-9Ee.+-]+)", re.IGNORECASE),
    "y34min": re.compile(r"\by34min\s*=\s*([0-9Ee.+-]+)", re.IGNORECASE),
    "y34max": re.compile(r"\by34max\s*=\s*([0-9Ee.+-]+)", re.IGNORECASE),
    "eta34min": re.compile(r"\beta34min\s*=\s*([0-9Ee.+-]+)", re.IGNORECASE),
    "eta34max": re.compile(r"\beta34max\s*=\s*([0-9Ee.+-]+)", re.IGNORECASE),
    "lhapdfset": re.compile(r"\blhapdfset\s*=\s*(.+)", re.IGNORECASE),
    "PDFname": re.compile(r"\*\s+PDFname\s+(.+?)\s+\*", re.IGNORECASE),
}


def infer_label(filename: str, labels: list[str]) -> str:
    lower = filename.lower()
    for label in labels:
        if label in lower:
            return label
    return ""


def read_text(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


def parse_extract_settings(extract_path: Path) -> dict[str, str]:
    text = read_text(extract_path)
    settings: dict[str, str] = {}

    for key, pattern in SETTING_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            # Keep the last occurrence because the tail/head extracts often
            # show the active card after earlier warmup context.
            settings[key] = str(matches[-1]).strip()

    return settings


def first_numeric(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return np.nan


def summarize_mcfm(ref: Path) -> pd.DataFrame:
    p = ref / "mcfm_integrals.csv"
    if not p.exists():
        raise SystemExit(f"Missing {p}")

    df = pd.read_csv(p)
    if df.empty:
        return pd.DataFrame()

    rows = []
    for filename, g in df.groupby("filename", sort=True):
        g = g.reset_index(drop=False)
        first = g.iloc[0]
        last = g.iloc[-1]

        extract = ref / "extracts" / f"mcfm__{filename}.extract.txt"
        settings = parse_extract_settings(extract)

        value = first_numeric(last["integral_value"])
        err = first_numeric(last["integral_uncertainty"])

        rows.append({
            "filename": filename,
            "point_label": infer_label(filename, POINT_LABELS),
            "cut_label": infer_label(filename, CUT_LABELS),
            "n_integral_records": int(len(g)),
            "first_integral": first_numeric(first["integral_value"]),
            "first_uncertainty": first_numeric(first["integral_uncertainty"]),
            "final_integral": value,
            "final_uncertainty": err,
            "final_rel_uncertainty": (
                abs(err / value) if value not in [0.0, np.nan] and np.isfinite(value) and value != 0 else np.nan
            ),
            "unit": str(last.get("unit", "")),
            "source_path": str(last.get("source_path", "")),
            "extract_path": str(extract if extract.exists() else ""),
            **settings,
        })

    out = pd.DataFrame(rows)
    order = [
        "filename", "point_label", "cut_label", "n_integral_records",
        "final_integral", "final_uncertainty", "final_rel_uncertainty",
        "unit", "m34min", "m34max", "pt34min", "pt34max",
        "y34min", "y34max", "sqrts", "lhapdfset", "PDFname",
        "source_path", "extract_path",
    ]
    order = [c for c in order if c in out.columns]
    return out[order + [c for c in out.columns if c not in order]]


def summarize_dyturbo(ref: Path) -> pd.DataFrame:
    p = ref / "artifact_manifest.csv"
    if not p.exists():
        raise SystemExit(f"Missing {p}")

    manifest = pd.read_csv(p)
    manifest = manifest[manifest["category"].astype(str) == "dyturbo"].copy()

    rows = []
    for _, row in manifest.iterrows():
        source = str(row["source_path"])
        filename = Path(source).name
        match = ROW_RE.search(filename)
        if not match:
            continue
        number = int(match.group(1))
        rows.append({
            "benchmark_number": number,
            "filename": filename,
            "source_path": source,
            "copied_relative_path": row.get("copied_relative_path", ""),
            "sha256": row.get("sha256", ""),
            "modified_local": row.get("modified_local", ""),
            "kind": (
                "quad" if "quad" in filename.lower()
                else "order1_gmu" if "order1_gmu" in filename.lower()
                else "order0" if "order0" in filename.lower()
                else "q2test" if "q2test" in filename.lower()
                else "fullswitch" if "fullswitch" in filename.lower()
                else "default"
            ),
        })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(["benchmark_number", "kind", "filename"])


def build_point_map(ref: Path, mcfm: pd.DataFrame) -> pd.DataFrame:
    rows_path = ref / "benchmark_rows.csv"
    if not rows_path.exists():
        raise SystemExit(f"Missing {rows_path}")

    rows = pd.read_csv(rows_path)
    if rows.empty:
        return pd.DataFrame()

    # Conservative inferred mapping from the naming established during v15:
    # onepoint was the documented E288_400:80 bridge row.  The other point
    # labels are left as candidates unless the log/card extracts identify
    # them explicitly.
    inferred = {
        "onepoint": 80,
        "secondpoint": np.nan,
        "thirdpoint": np.nan,
        "fourthpoint": np.nan,
    }

    candidates = []
    for _, m in mcfm.iterrows():
        point = str(m.get("point_label", ""))
        cut = str(m.get("cut_label", ""))

        if point == "":
            continue

        candidates.append({
            "point_label": point,
            "cut_label": cut,
            "mcfm_filename": m["filename"],
            "mcfm_final_integral": m["final_integral"],
            "mcfm_final_uncertainty": m["final_uncertainty"],
            "mcfm_final_rel_uncertainty": m["final_rel_uncertainty"],
            "unit": m.get("unit", ""),
            "inferred_benchmark_number": inferred.get(point, np.nan),
            "mapping_status": (
                "documented_from_prior_v15_context"
                if point == "onepoint" else
                "needs_log_or_card_provenance"
            ),
        })

    cand = pd.DataFrame(candidates)
    if cand.empty:
        return cand

    merged = cand.merge(
        rows,
        left_on="inferred_benchmark_number",
        right_on="benchmark_number",
        how="left",
        suffixes=("", "_data"),
    )

    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ref",
        default="v22/reference/external_tail_benchmark",
    )
    parser.add_argument(
        "--out",
        default="v22/outputs/external_tail_benchmark_reduced",
    )
    args = parser.parse_args()

    ref = Path(args.ref)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if not ref.exists():
        raise SystemExit(f"Missing reference directory: {ref}")

    summary_path = ref / "summary.json"
    summary = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())

    mcfm = summarize_mcfm(ref)
    dyturbo = summarize_dyturbo(ref)
    point_map = build_point_map(ref, mcfm)

    mcfm.to_csv(out / "mcfm_final_by_log.csv", index=False)
    dyturbo.to_csv(out / "dyturbo_logs_by_row.csv", index=False)
    point_map.to_csv(out / "mcfm_point_mapping_candidates.csv", index=False)

    # Keep a concise candidate list excluding logs that were explicitly zero
    # or broad inclusive sanity checks.
    candidate_cuts = {
        "bridgecut", "bridge", "nearcut", "midcut", "widecut", "widecheck"
    }
    candidates = mcfm[
        mcfm["cut_label"].isin(candidate_cuts)
        & np.isfinite(mcfm["final_integral"].astype(float))
        & (mcfm["final_integral"].astype(float) > 0.0)
    ].copy()
    candidates = candidates.sort_values(
        ["point_label", "cut_label", "final_integral"]
    )
    candidates.to_csv(out / "mcfm_candidate_targets.csv", index=False)

    report = {
        "source_summary": summary,
        "n_mcfm_logs": int(len(mcfm)),
        "n_mcfm_candidate_targets": int(len(candidates)),
        "n_dyturbo_logs_with_row_number": int(len(dyturbo)),
        "known_dyturbo_rows": (
            sorted(dyturbo["benchmark_number"].unique().tolist())
            if not dyturbo.empty else []
        ),
        "important_notes": [
            "The onepoint -> E288_400:80 mapping is the only one currently treated as documented from prior v15 context.",
            "secondpoint/thirdpoint/fourthpoint require card/log provenance before being used as hard regression targets.",
            "Do not use v15_onepoint_mcfm.log inclusive pb values for the tail regression.",
            "Prefer high-statistics final integral records, but retain all intermediate values for provenance.",
        ],
    }
    (out / "reduction_summary.json").write_text(json.dumps(report, indent=2) + "\n")

    print("\n=== Reduction summary ===")
    print(json.dumps(report, indent=2))

    print("\n=== MCFM final value by log ===")
    display_cols = [
        c for c in [
            "filename", "point_label", "cut_label", "n_integral_records",
            "final_integral", "final_uncertainty", "final_rel_uncertainty", "unit",
            "m34min", "m34max", "pt34min", "pt34max", "y34min", "y34max"
        ]
        if c in mcfm.columns
    ]
    print(mcfm[display_cols].to_string(index=False))

    print("\n=== DYTurbo logs by mapped row ===")
    if dyturbo.empty:
        print("None found")
    else:
        print(dyturbo[["benchmark_number", "kind", "filename"]].to_string(index=False))

    print("\n=== Candidate target mappings ===")
    if point_map.empty:
        print("None")
    else:
        show = [
            c for c in [
                "point_label", "cut_label", "mcfm_filename", "mcfm_final_integral",
                "mcfm_final_uncertainty", "unit", "inferred_benchmark_number",
                "row_id", "qT", "QM", "frac_error", "mapping_status"
            ]
            if c in point_map.columns
        ]
        print(point_map[show].to_string(index=False))

    print("\nwrote:", out)
    print(
        "\nNext: inspect mcfm_candidate_targets.csv and the per-log extracts "
        "for second/third/fourth point provenance."
    )


if __name__ == "__main__":
    main()
