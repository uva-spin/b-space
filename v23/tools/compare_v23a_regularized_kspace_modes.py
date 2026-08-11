#!/usr/bin/env python3
"""Compare regularized kT-space TMD outputs across tail/regularization modes.

Inputs are directories produced by construct_v23a_regularized_kspace_tmd.py.
The comparison is pointwise on the median curves in v23a_regularized_kspace_bands.csv.

Example:
  PYTHONPATH=. python3 v23/tools/compare_v23a_regularized_kspace_modes.py \
    --dirs \
      replica_v23a_expPDF_overlay_lambda3_normpriors15_p2p5_50rep/kspace_regularized_expPDF_overlay_expb2 \
      replica_v23a_expPDF_overlay_lambda3_normpriors15_p2p5_50rep/kspace_regularized_expPDF_overlay_expb \
      replica_v23a_expPDF_overlay_lambda3_normpriors15_p2p5_50rep/kspace_regularized_expPDF_overlay_taper \
    --reference expb2 \
    --out replica_v23a_expPDF_overlay_lambda3_normpriors15_p2p5_50rep/kspace_regularized_comparison
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def mode_name(d: Path) -> str:
    summary = d / "v23a_regularized_kspace_summary.json"
    if summary.exists():
        try:
            data = json.loads(summary.read_text())
            if "tail_mode" in data:
                return str(data["tail_mode"])
        except Exception:
            pass
    name = d.name
    for tok in ["expb2", "expb", "taper", "zero", "hold"]:
        if tok in name:
            return tok
    return name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--reference", default=None, help="Reference mode name. Defaults to first directory.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--active-frac", type=float, default=0.05)
    ap.add_argument("--max-rel-diff-pass", type=float, default=0.10)
    args = ap.parse_args()

    dirs = [Path(d) for d in args.dirs]
    tables = {}
    summaries = {}
    for d in dirs:
        p = d / "v23a_regularized_kspace_bands.csv"
        if not p.exists():
            raise SystemExit(f"Missing {p}")
        m = mode_name(d)
        if m in tables:
            m = f"{m}_{len(tables)}"
        tables[m] = pd.read_csv(p)
        sp = d / "v23a_regularized_kspace_summary.json"
        summaries[m] = json.loads(sp.read_text()) if sp.exists() else {}

    ref_name = args.reference or next(iter(tables))
    if ref_name not in tables:
        raise SystemExit(f"Reference {ref_name!r} not in modes {list(tables)}")

    key_cols = ["quantity", "pid", "flavor", "x", "Q", "kT"]
    ref = tables[ref_name][key_cols + ["median"]].rename(columns={"median": f"median_{ref_name}"})

    rows = []
    merged_all = ref.copy()
    for name, tab in tables.items():
        if name == ref_name:
            continue
        m = ref.merge(tab[key_cols + ["median"]].rename(columns={"median": f"median_{name}"}), on=key_cols, how="inner")
        m["abs_diff"] = (m[f"median_{name}"] - m[f"median_{ref_name}"]).abs()

        # Active threshold per curve relative to reference peak.
        for curve_key, g in m.groupby(["quantity", "pid", "flavor", "x", "Q"], observed=False):
            peak = float(np.nanmax(np.abs(g[f"median_{ref_name}"])))
            active = np.abs(g[f"median_{ref_name}"]) > float(args.active_frac) * max(peak, 1e-300)
            rel = g.loc[active, "abs_diff"] / np.maximum(np.abs(g.loc[active, f"median_{ref_name}"]), 1e-300)
            rows.append({
                "mode": name,
                "reference": ref_name,
                "quantity": curve_key[0],
                "pid": int(curve_key[1]),
                "flavor": curve_key[2],
                "x": float(curve_key[3]),
                "Q": float(curve_key[4]),
                "active_points": int(active.sum()),
                "rel_diff_median_active": float(np.nanmedian(rel)) if len(rel) else np.nan,
                "rel_diff_p90_active": float(np.nanquantile(rel, 0.90)) if len(rel) else np.nan,
                "rel_diff_max_active": float(np.nanmax(rel)) if len(rel) else np.nan,
            })

        merged_all = merged_all.merge(
            tab[key_cols + ["median"]].rename(columns={"median": f"median_{name}"}),
            on=key_cols,
            how="outer",
        )

    comp = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    comp.to_csv(out / "regularization_mode_curve_comparison.csv", index=False)
    merged_all.to_csv(out / "regularization_mode_pointwise_medians.csv", index=False)

    decision = {
        "modes": list(tables),
        "reference": ref_name,
        "active_frac": float(args.active_frac),
        "max_rel_diff_pass": float(args.max_rel_diff_pass),
        "rel_diff_p90_global_max": float(comp["rel_diff_p90_active"].max()) if len(comp) else np.nan,
        "rel_diff_max_global_max": float(comp["rel_diff_max_active"].max()) if len(comp) else np.nan,
        "KT_REGULARIZATION_STABILITY_PASS": bool(
            len(comp) > 0 and np.isfinite(comp["rel_diff_p90_active"]).all()
            and float(comp["rel_diff_p90_active"].max()) <= float(args.max_rel_diff_pass)
        ),
        "interpretation": (
            "Compares median kT-space curves across tail/regularization modes. "
            "Use this to choose a conservative kT range. If the full kT<=kmax comparison fails, "
            "rerun the constructor with a smaller --k-max and compare again."
        ),
        "summaries": summaries,
    }
    (out / "regularization_mode_comparison_summary.json").write_text(json.dumps(decision, indent=2) + "\n")
    print("\n=== kT regularization-mode comparison ===")
    print(json.dumps(decision, indent=2))
    print("\nwrote:", out)


if __name__ == "__main__":
    main()
