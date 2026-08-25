#!/usr/bin/env python3
"""Rebuild missing b-space W rows for the isolated 353-row scope.

The historical cache was generated with a narrower row selection.  This
script evaluates only scope rows absent from that cache using the same
candidate internal N3LL-pilot W configuration, and writes a new candidate
fragment; it never edits the original cache.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
SYSTEMATICS = BASE.parent
PROJECT = SYSTEMATICS.parent
SCOPE = BASE / "reports/tevatron_353_candidate_g1_1p017_expcreg2p0/candidate_353_full_wy.csv"
SOURCE_ROOT = PROJECT / "Data/v23a_fixed_target_plus_tevatron_absolute_fit_ready"
W_CACHE = PROJECT / "outputs/v23a_tevatron_plus_lhcb7_fidacc_merged_cache/backend_cache/wpert_v23a_tevatron_plus_lhcb7_fidacc_b160.csv"
BACKEND = PROJECT / "bt_internal_css_backend_v16.py"
DEFAULT_OUT = BASE / "reports/missing_scope_bspace_w_n3llp_nloQ96_b160"


def load_backend():
    spec = importlib.util.spec_from_file_location("missing_scope_css_backend", BACKEND)
    if spec is None or spec.loader is None:
        raise RuntimeError(BACKEND)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    backend = load_backend()
    scope = pd.read_csv(SCOPE)
    cache = pd.read_csv(W_CACHE)
    cached = set(cache.row_id.astype(str))
    missing = scope[~scope.row_id.astype(str).isin(cached)].copy()
    # LHCb's b-space cache is already present in W_CACHE; the missing rows are
    # fixed-target source rows whose historical cache was cut by selection.
    missing = missing[missing.dataset.astype(str).isin(["E288_200", "E288_300", "E288_400", "E605", "E772"])]
    if missing.empty:
        raise RuntimeError("no missing fixed-target scope rows")
    source = []
    for dataset in sorted(missing.dataset.astype(str).unique()):
        frame = pd.read_csv(SOURCE_ROOT / f"{dataset}.csv")
        source.append(frame[frame.row_id.astype(str).isin(set(missing[missing.dataset.eq(dataset)].row_id.astype(str)))])
    rows = pd.concat(source, ignore_index=True)
    cfg = backend.CSSConfig(
        b_min=1.0e-4, b_max=8.0, n_b=160, bstar_bmax=1.5, mu_min=1.3,
        cap_mub_at_Q=True, q0=2.0, resum_order="n3llp", nf=5,
        n_sudakov_quad=32, prefactor_scheme="oldA_to_CS", target_mode="nuclear_isospin",
        y_mode="zero", flavors=(1, 2, 3),
    )
    b = backend.make_b_grid(cfg)
    pdf = backend.LHAPDFProvider("NNPDF40_nnlo_as_01180", 0, use_toy_pdf=False)
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    records = []
    for _, row in rows.iterrows():
        rid = str(row.row_id)
        path = out / f"{rid.replace(':', '_')}.csv"
        if path.exists() and not args.force:
            part = pd.read_csv(path); records.extend(part.to_dict("records")); continue
        values = backend.wpert_cs_for_row(row, b, pdf, cfg)
        if not np.isfinite(values).all():
            raise RuntimeError(f"nonfinite W for {rid}")
        part = pd.DataFrame({"row_id": rid, "dataset": str(row.dataset), "bT": b, "Wpert_CS": values})
        part.to_csv(path, index=False)
        records.extend(part.to_dict("records"))
        print(rid, "done", flush=True)
    fragment = pd.DataFrame(records)
    fragment.to_csv(out / "missing_scope_bspace_w.csv", index=False)
    status = {
        "status": "isolated_missing_scope_bspace_w_rebuild_complete_not_production",
        "row_count": int(fragment.row_id.nunique()), "b_nodes": int(fragment.bT.nunique()),
        "row_ids": sorted(fragment.row_id.astype(str).unique()),
        "all_finite": bool(np.isfinite(fragment.Wpert_CS.to_numpy(float)).all()),
        "source_cache": str(W_CACHE), "backend": str(BACKEND),
        "configuration": {"resum_order": "n3llp", "n_b": 160, "b_max": 8.0, "target_mode": "nuclear_isospin"},
        "frozen_baseline_unchanged": True, "production_outputs_modified": False, "promotion_authorized": False,
    }
    (out / "missing_scope_bspace_w_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__": main()
