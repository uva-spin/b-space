#!/usr/bin/env python3
"""Construct b-space TMD bands from joint experimental+PDF replicas.

Unlike v22/tools/construct_v22_bspace_tmd_bands_from_replicas.py, this tool
does NOT multiply every replica F_NP by one central perturbative factor.
Instead, for each replica run it reconstructs the b-space TMD grid using the
same PDF member assigned to that replica, then aggregates the resulting TMD
grids.

This creates the first genuine data-replica x PDF-replica b-space TMD ensemble
within the current v23a workflow.  It still does not include scale/profile,
nuclear-model, or model-form variations unless those are added to the plan.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


QUANTITIES = ["F_NP", "ftilde", "x_ftilde", "b_ftilde", "b_x_ftilde"]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def get_pdf_member(run: Path, plan_member: int | None) -> int:
    meta = load_json(run / "pdf_replica_meta.json")
    if "pdf_member" in meta:
        return int(meta["pdf_member"])
    metrics = load_json(run / "metrics.json")
    cfg = metrics.get("config", {}) if isinstance(metrics, dict) else {}
    if "pdf_member" in cfg:
        return int(cfg["pdf_member"])
    if plan_member is not None:
        return int(plan_member)
    raise SystemExit(f"Could not infer pdf_member for {run}")


def get_seed(run: Path, plan_seed: int | None) -> str:
    meta = load_json(run / "pdf_replica_meta.json")
    if "seed" in meta:
        return str(meta["seed"])
    if plan_seed is not None:
        return str(plan_seed)
    for tok in run.name.replace("-", "_").split("_"):
        if tok.startswith("s") and tok[1:].isdigit():
            return tok[1:]
    return run.name


def read_plan(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"run_dir", "pdf_member", "seed"}
    missing = required.difference(df.columns)
    if missing:
        raise SystemExit(f"Plan missing columns: {sorted(missing)}")
    return df.copy()


def run_grid_tool(args: argparse.Namespace, run: Path, seed: str, member: int, grid_dir: Path) -> None:
    grid_csv = grid_dir / "v22_scheme_tmd_bspace_long.csv"
    if grid_csv.exists() and not args.rebuild_grids:
        return

    grid_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        args.grid_tool,
        "--run", str(run),
        "--backend-script", args.backend_script,
        "--pdf-set", args.pdf_set,
        "--pdf-member", str(member),
        "--resum-order", args.resum_order,
        "--pids", *map(str, args.pids),
        "--x-values", *map(str, args.x_values),
        "--Q-values", *map(str, args.Q_values),
        "--b-min", str(args.b_min),
        "--b-max", str(args.b_max),
        "--n-b", str(args.n_b),
        "--out", str(grid_dir),
    ]
    if args.allow_x_interpolation:
        cmd.append("--allow-x-interpolation")

    log = grid_dir / "construct_grid.log"
    print(f"grid seed={seed} pdf={member}: {grid_dir}")
    with log.open("w") as fh:
        fh.write(" ".join(cmd) + "\n\n")
        fh.flush()
        subprocess.run(cmd, check=True, stdout=fh, stderr=subprocess.STDOUT)


def plot_quantity(summary: pd.DataFrame, out: Path, quantity: str, ylabel: str) -> None:
    pdf_path = out / f"{quantity}_dataPDF_bands.pdf"
    with PdfPages(pdf_path) as pdf:
        for Q in sorted(summary["Q"].unique()):
            for pid in sorted(summary["pid"].unique()):
                page = summary[(np.isclose(summary["Q"], Q)) & (summary["pid"].astype(int) == int(pid))]
                if page.empty:
                    continue
                fig, ax = plt.subplots(figsize=(8.5, 5.5))
                for x in sorted(page["x"].unique()):
                    g = page[np.isclose(page["x"], x)].sort_values("bT")
                    ax.plot(g["bT"], g[f"{quantity}_median"], label=f"x={x:g}")
                    ax.fill_between(
                        g["bT"].to_numpy(float),
                        g[f"{quantity}_q16"].to_numpy(float),
                        g[f"{quantity}_q84"].to_numpy(float),
                        alpha=0.22,
                    )
                flavor = str(page["flavor"].iloc[0])
                ax.set_title(f"{quantity}: {flavor}, Q={Q:g} GeV")
                ax.set_xlabel(r"$b_T\,[{\rm GeV}^{-1}]$")
                ax.set_ylabel(ylabel)
                ax.grid(True, alpha=0.3)
                ax.legend(ncol=2)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, help="CSV from make_v23a_data_pdf_replica_plan.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid-tool", default="v22/tools/construct_v22_scheme_tmd_grid.py")
    ap.add_argument("--backend-script", default="v22/backends/bt_internal_css_backend_v22_full.py")
    ap.add_argument("--pdf-set", default="NNPDF40_nnlo_as_01180")
    ap.add_argument("--resum-order", default="n3llp")
    ap.add_argument("--pids", nargs="+", type=int, default=[2, 1, -2, -1])
    ap.add_argument("--x-values", nargs="+", type=float, default=[0.10, 0.20, 0.30, 0.50])
    ap.add_argument("--Q-values", nargs="+", type=float, default=[5.0, 10.0])
    ap.add_argument("--b-min", type=float, default=0.0)
    ap.add_argument("--b-max", type=float, default=8.0)
    ap.add_argument("--n-b", type=int, default=321)
    ap.add_argument("--allow-x-interpolation", action="store_true")
    ap.add_argument("--rebuild-grids", action="store_true")
    ap.add_argument("--max-replicas", type=int, default=None)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    grid_root = out / "replica_grids"
    grid_root.mkdir(exist_ok=True)

    plan = read_plan(Path(args.plan))
    if args.max_replicas:
        plan = plan.head(int(args.max_replicas)).copy()

    rows = []
    replica_meta = []
    for _, r in plan.iterrows():
        run = Path(str(r["run_dir"]))
        if not run.exists():
            raise SystemExit(f"Missing run_dir: {run}")
        seed = get_seed(run, int(r["seed"]))
        member = get_pdf_member(run, int(r["pdf_member"]))
        grid_dir = grid_root / f"s{seed}_pdf{member:04d}"
        run_grid_tool(args, run, seed, member, grid_dir)

        grid_csv = grid_dir / "v22_scheme_tmd_bspace_long.csv"
        if not grid_csv.exists():
            raise SystemExit(f"Grid tool did not produce {grid_csv}")
        g = pd.read_csv(grid_csv)

        required = {"pid", "flavor", "x", "Q", "bT", "F_NP", "ftilde"}
        missing = required.difference(g.columns)
        if missing:
            raise SystemExit(f"{grid_csv} missing columns: {sorted(missing)}")

        if "x_ftilde" not in g.columns:
            g["x_ftilde"] = g["x"].astype(float) * g["ftilde"].astype(float)
        if "b_ftilde" not in g.columns:
            g["b_ftilde"] = g["bT"].astype(float) * g["ftilde"].astype(float)
        if "b_x_ftilde" not in g.columns:
            g["b_x_ftilde"] = g["bT"].astype(float) * g["x"].astype(float) * g["ftilde"].astype(float)

        g["seed"] = seed
        g["pdf_member"] = int(member)
        g["run_dir"] = str(run)
        rows.append(g)

        replica_meta.append({
            "seed": seed,
            "pdf_member": int(member),
            "run_dir": str(run),
            "grid_dir": str(grid_dir),
        })

    long = pd.concat(rows, ignore_index=True)
    # Keep only columns needed plus optional ftilde_no_np.
    keep = ["seed", "pdf_member", "run_dir", "pid", "flavor", "x", "Q", "bT"]
    for c in ["ftilde_no_np", *QUANTITIES]:
        if c in long.columns:
            keep.append(c)
    long = long[keep].copy()

    agg_dict = {}
    for q in QUANTITIES:
        if q not in long.columns:
            continue
        agg_dict[f"{q}_median"] = (q, "median")
        agg_dict[f"{q}_q16"] = (q, lambda v: float(np.nanquantile(v, 0.16)))
        agg_dict[f"{q}_q84"] = (q, lambda v: float(np.nanquantile(v, 0.84)))

    bands = (
        long.groupby(["pid", "flavor", "x", "Q", "bT"], observed=False)
        .agg(**agg_dict)
        .reset_index()
    )

    long_path = out / "v23a_dataPDF_tmd_replica_bspace_long.csv"
    band_path = out / "v23a_dataPDF_tmd_replica_bspace_bands.csv"
    long.to_csv(long_path, index=False)
    bands.to_csv(band_path, index=False)

    rel_rows = []
    for quantity in QUANTITIES:
        med_col = f"{quantity}_median"
        if med_col not in bands.columns:
            continue
        for (pid, flavor, x, Q), group in bands.groupby(["pid", "flavor", "x", "Q"], observed=False):
            med = group[med_col].to_numpy(float)
            hw = 0.5 * (group[f"{quantity}_q84"].to_numpy(float) - group[f"{quantity}_q16"].to_numpy(float))
            active = np.abs(med) > 0.05 * np.nanmax(np.abs(med))
            rel = hw[active] / np.maximum(np.abs(med[active]), 1.0e-300)
            rel_rows.append({
                "quantity": quantity,
                "pid": int(pid),
                "flavor": str(flavor),
                "x": float(x),
                "Q": float(Q),
                "active_points": int(np.sum(active)),
                "relative_68_halfwidth_median_active": float(np.nanmedian(rel)) if len(rel) else np.nan,
                "relative_68_halfwidth_p90_active": float(np.nanquantile(rel, 0.90)) if len(rel) else np.nan,
                "relative_68_halfwidth_max_active": float(np.nanmax(rel)) if len(rel) else np.nan,
            })
    rel = pd.DataFrame(rel_rows)
    rel_path = out / "v23a_dataPDF_relative_band_summary.csv"
    rel.to_csv(rel_path, index=False)

    labels = {
        "F_NP": r"$F_{\rm NP}(x,b_T)$",
        "ftilde": r"$\widetilde f_{1,q/p}(x,b_T;Q,Q^2)$",
        "x_ftilde": r"$x\,\widetilde f_{1,q/p}$",
        "b_ftilde": r"$b_T\,\widetilde f_{1,q/p}$",
        "b_x_ftilde": r"$b_T\,x\,\widetilde f_{1,q/p}$",
    }
    for q in QUANTITIES:
        if f"{q}_median" in bands.columns:
            plot_quantity(bands, out, q, labels[q])

    manifest = {
        "plan": str(args.plan),
        "out": str(out),
        "n_replicas": int(len(replica_meta)),
        "pdf_set": args.pdf_set,
        "pdf_members": sorted({m["pdf_member"] for m in replica_meta}),
        "replicas": replica_meta,
        "definition": (
            "Each replica TMD grid is reconstructed with that replica's fitted F_NP and "
            "the same PDF member assigned to its experimental pseudo-data fit.  Bands are "
            "q16/q50/q84 over the joint data x PDF replica ensemble."
        ),
        "not_included": [
            "renormalization/factorization/profile scale variations",
            "nuclear-model variations beyond the fixed target_mode",
            "model-form/anchor variations unless represented by additional plan rows",
            "kT-space production transform uncertainty",
        ],
        "outputs": {
            "long": str(long_path),
            "bands": str(band_path),
            "relative_summary": str(rel_path),
        },
    }
    (out / "v23a_dataPDF_tmd_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print("\n=== v23a data x PDF b-space TMD bands created ===")
    print(json.dumps({k: manifest[k] for k in ["n_replicas", "pdf_set", "pdf_members", "definition"]}, indent=2))
    print("\nRelative band summary preview:")
    print(rel.head(30).to_string(index=False))
    print("\nwrote:", out)


if __name__ == "__main__":
    main()
