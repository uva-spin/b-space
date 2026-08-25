#!/usr/bin/env python3
"""Audit the exact 353-row candidate scope without running or promoting it.

The accepted low-qT core is inherited only as a row-selection authority from
the frozen diagnostic fit.  The 24 Tevatron boundary rows are the explicit
high-qT extension.  This script records the source table, target convention,
and known readiness boundaries so a future full W+Y runner cannot silently
drop rows or call a partial grid a 353-row result.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
PROJECT = BASE.parents[1]
CORE_OUTPUT = (
    PROJECT
    / "systematics/collins_factorization_validity/outputs/rowidfix_stageFT_all_qmax0p20_lam0p50_central_s303/predictions.csv"
)
BOUNDARY = BASE / "reports/dyturbo_full_n3ll_nnlo_boundary_g1_1p017/tevatron_boundary_input.csv"
ABSOLUTE_ROOT = PROJECT / "Data/v23a_fixed_target_plus_tevatron_absolute_fit_ready"
LHC_ROOT = PROJECT / "Data/v23a_tevatron_plus_lhcb7_fiducial_candidate"
OUT = BASE / "reports/tevatron_353_scope_audit.json"


def load_rows(root: Path, dataset: str, row_ids: set[str]) -> pd.DataFrame:
    path = root / f"{dataset}.csv"
    if not path.exists():
        raise RuntimeError(f"missing source table for {dataset}: {path}")
    frame = pd.read_csv(path)
    return frame[frame["row_id"].astype(str).isin(row_ids)].copy()


def resolve_core_rows(root: Path, dataset: str, wanted: pd.DataFrame) -> pd.DataFrame:
    """Resolve legacy compressed row IDs against the current source table.

    The 329-row fit selection predates the fit-ready source tables and, for a
    few fixed-target tables, its local row labels are compressed after rejected
    source rows were removed.  Physical bin coordinates and the published CS
    value are the stable join keys; a silent positional join would be unsafe.
    """
    path = root / f"{dataset}.csv"
    if not path.exists():
        raise RuntimeError(f"missing source table for {dataset}: {path}")
    source = pd.read_csv(path)
    direct = source[source["row_id"].astype(str).isin(set(wanted.row_id.astype(str)))].copy()
    if len(direct) == len(wanted):
        return direct
    keys = ["qT", "QM", "y", "y_Low", "y_High", "QM_Low", "QM_High", "CS"]
    resolved = []
    for _, row in wanted.iterrows():
        mask = pd.Series(True, index=source.index)
        for key in keys:
            if key not in source or key not in row.index:
                continue
            value = row[key]
            if pd.isna(value):
                mask &= source[key].isna()
            else:
                mask &= source[key].notna() & (source[key].astype(float) - float(value)).abs().lt(2.0e-7)
        match = source[mask]
        if len(match) != 1:
            raise RuntimeError(f"could not uniquely resolve {row.row_id} in {path}: {len(match)} matches")
        resolved.append(match.iloc[0])
    return pd.DataFrame(resolved).reset_index(drop=True)


def main() -> None:
    core = pd.read_csv(CORE_OUTPUT)
    boundary = pd.read_csv(BOUNDARY)
    if len(core) != 329 or core.row_id.astype(str).nunique() != 329:
        raise RuntimeError("accepted low-qT core is not the expected unique 329 rows")
    if len(boundary) != 24 or boundary.row_id.astype(str).nunique() != 24:
        raise RuntimeError("Tevatron boundary is not the expected unique 24 rows")
    overlap = set(core.row_id.astype(str)) & set(boundary.row_id.astype(str))
    if overlap:
        raise RuntimeError(f"core/boundary overlap: {sorted(overlap)}")

    core_datasets = sorted(core.dataset.astype(str).unique())
    source_parts = []
    for dataset in core_datasets:
        ids = set(core.loc[core.dataset.astype(str).eq(dataset), "row_id"].astype(str))
        root = LHC_ROOT if dataset == "LHCb_7" else ABSOLUTE_ROOT
        part = resolve_core_rows(root, dataset, core[core.dataset.astype(str).eq(dataset)])
        if len(part) != len(ids):
            missing = sorted(ids - set(part.row_id.astype(str)))
            raise RuntimeError(f"missing source rows for {dataset}: {missing}")
        source_parts.append(part)
    # Boundary source is already the validated absolute Tevatron table.
    source_parts.append(boundary.copy())
    source = pd.concat(source_parts, ignore_index=True, sort=False)
    all_ids = pd.concat(
        [core[["dataset", "row_id"]], boundary[["dataset", "row_id"]]], ignore_index=True
    )
    if len(source) != 353 or source.row_id.astype(str).nunique() != 353:
        raise RuntimeError("resolved 353-row source table is not unique")

    rows = []
    target_map = {
        "E288_200": {"mode": "nuclearpdf", "Z": 4.0, "A": 9.0121831, "status": "candidate_assumption"},
        "E288_300": {"mode": "nuclearpdf", "Z": 4.0, "A": 9.0121831, "status": "candidate_assumption"},
        "E288_400": {"mode": "nuclearpdf", "Z": 4.0, "A": 9.0121831, "status": "candidate_assumption"},
        "E605": {"mode": "nuclearpdf", "Z": 29.0, "A": 63.546, "status": "candidate_assumption"},
        "E772": {"mode": "nuclearpdf", "Z": 0.5, "A": 1.0, "status": "unresolved_isoscalar_fallback"},
        "LHCb_7": {"mode": "pp_fiducial", "Z": None, "A": None, "status": "diagnostic_not_production"},
        "CDF_RUN_1": {"mode": "pbar_p", "Z": None, "A": None, "status": "validated_tevatron"},
        "CDF_RUN_2": {"mode": "pbar_p", "Z": None, "A": None, "status": "validated_tevatron"},
        "D0_RUN_1": {"mode": "pbar_p", "Z": None, "A": None, "status": "validated_tevatron"},
    }
    for dataset, group in source.groupby(source["dataset"].astype(str), sort=True):
        target = target_map.get(dataset)
        if target is None:
            raise RuntimeError(f"no target convention for {dataset}")
        rows.append(
            {
                "dataset": dataset,
                "row_count": int(len(group)),
                "row_ids": group.row_id.astype(str).tolist(),
                "target": target,
                "unit_values": sorted(group.get("unit", pd.Series(dtype=str)).dropna().astype(str).unique()),
                "qT_over_Q_min": float(group.qT_over_Q.min()) if "qT_over_Q" in group else None,
                "qT_over_Q_max": float(group.qT_over_Q.max()) if "qT_over_Q" in group else None,
            }
        )

    payload = {
        "status": "353_row_scope_resolved_not_yet_run",
        "formula_target": "W_N3LL + (FO_NNLO - ASY_NNLO)",
        "core": {
            "row_count": 329,
            "selection_source": str(CORE_OUTPUT),
            "selection_is_authority_only": True,
            "datasets": core_datasets,
        },
        "boundary": {
            "row_count": 24,
            "source": str(BOUNDARY),
            "datasets": sorted(boundary.dataset.astype(str).unique()),
        },
        "resolved_row_count": 353,
        "datasets": rows,
        "readiness": {
            "tevatron_122_plus_boundary_24": "external DYTurbo path validated, final high-stat grid in progress",
            "fixed_target": "requires target/unit/per-nucleon convention closure before production claim",
            "E772": "target composition remains unresolved; isoscalar fallback is diagnostic only",
            "LHCb_7_core_rows": "fiducial acceptance candidate remains diagnostic-only",
        },
        "production_authorized": False,
        "frozen_baseline_unchanged": True,
        "production_outputs_modified": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in ("status", "resolved_row_count", "readiness")}, indent=2))


if __name__ == "__main__":
    main()
