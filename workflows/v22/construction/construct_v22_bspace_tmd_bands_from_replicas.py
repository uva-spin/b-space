#!/usr/bin/env python3
"""Build b-space v22 TMD bands from profiled replica F_NP grids.

This is fast because the perturbative factor

    [C tensor f] * exp[-S/2]

is read from the central v22 TMD grid (`ftilde_no_np`) and only the fitted
F_NP factor is replaced by each replica's F_NP grid.

No k-space transform is performed here.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


def infer_column(frame: pd.DataFrame, names: list[str], purpose: str) -> str:
    lower = {c.lower(): c for c in frame.columns}
    for name in names:
        if name in frame.columns:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    raise SystemExit(f"Could not infer {purpose} column. Available: {list(frame.columns)}")


class FNPGrid:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.x_col = infer_column(frame, ["x"], "x")
        self.b_col = infer_column(frame, ["bT", "b", "bt", "b_GeV_inv"], "bT")
        self.f_col = infer_column(frame, ["F_NP", "F_NP_mean", "fnp", "FNP", "f_np"], "F_NP")
        work = frame[[self.x_col, self.b_col, self.f_col]].dropna().copy()
        self.xs = np.array(sorted(work[self.x_col].unique()), dtype=float)
        self.by_x = {}
        for x, g in work.groupby(self.x_col, observed=False):
            gg = g.sort_values(self.b_col)
            self.by_x[float(x)] = (
                gg[self.b_col].to_numpy(float),
                gg[self.f_col].to_numpy(float),
            )

    def eval_known_x(self, x: float, b: float) -> float:
        idx = int(np.argmin(np.abs(self.xs - float(x))))
        x0 = float(self.xs[idx])
        if abs(x0 - float(x)) > 5.0e-8:
            raise KeyError(x)
        bg, fg = self.by_x[x0]
        return float(np.interp(float(b), bg, fg, left=fg[0], right=fg[-1]))

    def __call__(self, x: float, b: float) -> float:
        try:
            return self.eval_known_x(x, b)
        except KeyError:
            pass
        if float(x) < self.xs.min() or float(x) > self.xs.max():
            raise ValueError(f"x={x} outside F_NP grid range")
        hi = int(np.searchsorted(self.xs, float(x), side="right"))
        lo = hi - 1
        x_lo = float(self.xs[lo])
        x_hi = float(self.xs[hi])
        f_lo = self.eval_known_x(x_lo, b)
        f_hi = self.eval_known_x(x_hi, b)
        t = (float(x) - x_lo) / (x_hi - x_lo)
        return float((1 - t) * f_lo + t * f_hi)


def seed_from_run(run: Path) -> str:
    for token in run.name.replace("-", "_").split("_"):
        if token.startswith("s") and token[1:].isdigit():
            return token[1:]
    return run.name


def load_replica_fnp(run: Path) -> tuple[str, FNPGrid]:
    candidates = [run / "fnp_debug_grid.csv", run / "run" / "fnp_debug_grid.csv"]
    found = None
    for c in candidates:
        if c.exists():
            found = c
            break
    if found is None:
        matches = sorted(run.rglob("fnp_debug_grid.csv"))
        if matches:
            found = matches[0]
    if found is None:
        raise SystemExit(f"Could not find fnp_debug_grid.csv under {run}")
    return seed_from_run(run), FNPGrid(pd.read_csv(found))


def plot_quantity(summary: pd.DataFrame, central: pd.DataFrame, out: Path, quantity: str, ylabel: str) -> None:
    pdf_path = out / f"{quantity}_bands.pdf"
    with PdfPages(pdf_path) as pdf:
        for Q in sorted(summary["Q"].unique()):
            for pid in sorted(summary["pid"].unique()):
                page = summary[(np.isclose(summary["Q"], Q)) & (summary["pid"].astype(int) == int(pid))]
                if page.empty:
                    continue
                fig, ax = plt.subplots(figsize=(8.5, 5.5))
                for x in sorted(page["x"].unique()):
                    g = page[np.isclose(page["x"], x)].sort_values("bT")
                    c = central[
                        (central["pid"].astype(int) == int(pid))
                        & np.isclose(central["Q"], Q)
                        & np.isclose(central["x"], x)
                    ].sort_values("bT")
                    ax.plot(g["bT"], g[f"{quantity}_median"], label=f"x={x:g}")
                    ax.fill_between(
                        g["bT"].to_numpy(float),
                        g[f"{quantity}_q16"].to_numpy(float),
                        g[f"{quantity}_q84"].to_numpy(float),
                        alpha=0.22,
                    )
                    if not c.empty and quantity in c.columns:
                        ax.plot(c["bT"], c[quantity], linestyle="--", linewidth=1.2)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--central-grid", required=True)
    parser.add_argument("--replica-glob", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    central_path = Path(args.central_grid)
    if not central_path.exists():
        raise SystemExit(f"Missing central grid: {central_path}")

    central = pd.read_csv(central_path)
    required = {"pid", "flavor", "x", "Q", "bT", "ftilde_no_np", "ftilde", "F_NP"}
    missing = required.difference(central.columns)
    if missing:
        raise SystemExit(f"Central grid missing columns: {sorted(missing)}")

    runs = [Path(p) for p in sorted(glob.glob(args.replica_glob))]
    if not runs:
        raise SystemExit(f"No replica runs matched: {args.replica_glob}")

    replicas = [load_replica_fnp(run) for run in runs]

    rows = []
    for seed, grid in replicas:
        for _, row in central.iterrows():
            x = float(row["x"])
            bT = float(row["bT"])
            F = float(grid(x, bT))
            ftilde = float(row["ftilde_no_np"]) * F
            rows.append({
                "seed": seed,
                "pid": int(row["pid"]),
                "flavor": str(row["flavor"]),
                "x": x,
                "Q": float(row["Q"]),
                "bT": bT,
                "F_NP": F,
                "ftilde": ftilde,
                "x_ftilde": x * ftilde,
                "b_ftilde": bT * ftilde,
                "b_x_ftilde": bT * x * ftilde,
            })

    long = pd.DataFrame(rows)

    q = (
        long.groupby(["pid", "flavor", "x", "Q", "bT"], observed=False)
        .agg(
            F_NP_median=("F_NP", "median"),
            F_NP_q16=("F_NP", lambda v: float(np.nanquantile(v, 0.16))),
            F_NP_q84=("F_NP", lambda v: float(np.nanquantile(v, 0.84))),
            ftilde_median=("ftilde", "median"),
            ftilde_q16=("ftilde", lambda v: float(np.nanquantile(v, 0.16))),
            ftilde_q84=("ftilde", lambda v: float(np.nanquantile(v, 0.84))),
            x_ftilde_median=("x_ftilde", "median"),
            x_ftilde_q16=("x_ftilde", lambda v: float(np.nanquantile(v, 0.16))),
            x_ftilde_q84=("x_ftilde", lambda v: float(np.nanquantile(v, 0.84))),
            b_ftilde_median=("b_ftilde", "median"),
            b_ftilde_q16=("b_ftilde", lambda v: float(np.nanquantile(v, 0.16))),
            b_ftilde_q84=("b_ftilde", lambda v: float(np.nanquantile(v, 0.84))),
            b_x_ftilde_median=("b_x_ftilde", "median"),
            b_x_ftilde_q16=("b_x_ftilde", lambda v: float(np.nanquantile(v, 0.16))),
            b_x_ftilde_q84=("b_x_ftilde", lambda v: float(np.nanquantile(v, 0.84))),
        )
        .reset_index()
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    long.to_csv(out / "v22_tmd_replica_bspace_long.csv", index=False)
    q.to_csv(out / "v22_tmd_replica_bspace_bands.csv", index=False)

    plot_quantity(q, central, out, "F_NP", r"$F_{\rm NP}(x,b_T)$")
    plot_quantity(q, central, out, "ftilde", r"$\widetilde f_{1,q/p}(x,b_T;Q,Q^2)$")
    plot_quantity(q, central, out, "b_ftilde", r"$b_T\,\widetilde f_{1,q/p}$")
    plot_quantity(q, central, out, "b_x_ftilde", r"$b_T\,x\,\widetilde f_{1,q/p}$")

    rel_rows = []
    for (pid, flavor, x, Q), group in q.groupby(["pid", "flavor", "x", "Q"], observed=False):
        med = group["ftilde_median"].to_numpy(float)
        hw = 0.5 * (group["ftilde_q84"].to_numpy(float) - group["ftilde_q16"].to_numpy(float))
        active = np.abs(med) > 0.05 * np.nanmax(np.abs(med))
        rel = hw[active] / np.maximum(np.abs(med[active]), 1.0e-300)
        rel_rows.append({
            "pid": int(pid),
            "flavor": flavor,
            "x": float(x),
            "Q": float(Q),
            "active_points": int(np.sum(active)),
            "relative_68_halfwidth_median_active": float(np.nanmedian(rel)) if len(rel) else np.nan,
            "relative_68_halfwidth_p90_active": float(np.nanquantile(rel, 0.90)) if len(rel) else np.nan,
        })
    rel_summary = pd.DataFrame(rel_rows)
    rel_summary.to_csv(out / "v22_tmd_relative_band_summary.csv", index=False)

    manifest = {
        "central_grid": str(central_path),
        "replica_glob": args.replica_glob,
        "n_replicas": len(replicas),
        "seeds": [seed for seed, _ in replicas],
        "definition": "replica ftilde = central perturbative factor ftilde_no_np times replica F_NP",
        "kspace_status": "not constructed; b-space bands only",
    }
    (out / "v22_tmd_replica_band_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print("\n=== v22 b-space TMD replica bands created ===")
    print(json.dumps(manifest, indent=2))
    print("\nRelative band summary preview:")
    print(rel_summary.head(20).to_string(index=False))
    print("\nwrote:", out)


if __name__ == "__main__":
    main()
