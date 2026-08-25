#!/usr/bin/env python3
"""Render Fig. 2/Fig. 6-style diagnostics for the corrected W+Y controls.

Each case is an isolated start ensemble at x=0.1, Q=10 GeV.  The plots use
the common frozen perturbative TMD reference only as a visual normalization
and show the fitted candidate F_NP start spread; they are explicitly
isolated diagnostics, not production figures.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
REPORTS = BASE / "reports"
PLOTTER = BASE / "scripts/plot_scope_353_fnp_start_replica_fig2_fig6.py"
DEFAULT_CASES = {
    "all_y": {
        "label": "all W+Y (including LHCb Y)",
        "dirs": [REPORTS / f"scope_329_exactprotocol_aligned_wy_s{s}" for s in (303, 304, 305)],
    },
    "non_lhcb_y": {
        "label": "W+Y with LHCb Y set to zero",
        "dirs": [REPORTS / f"scope_329_y_no_lhcb_decomp_wy_s{s}" for s in (303, 304, 305)],
    },
    "lhcb_only_y": {
        "label": "W+Y with LHCb Y only",
        "dirs": [REPORTS / f"scope_329_y_lhcb_only_decomp_wy_s{s}" for s in (303, 304, 305)],
    },
}

PERTURBED_CASES = {
    "all_y": {
        "label": "all W+Y (1% perturbed starts; no tail prior)",
        "dirs": [REPORTS / f"scope_329_perturbed1pct_all_y_wy_s{s}" for s in range(303, 311)],
    },
    "non_lhcb_y": {
        "label": "W+Y with LHCb Y set to zero (1% perturbed starts)",
        "dirs": [REPORTS / f"scope_329_perturbed1pct_non_lhcb_y_wy_s{s}" for s in range(303, 311)],
    },
    "lhcb_only_y": {
        "label": "W+Y with LHCb Y only (1% perturbed starts)",
        "dirs": [REPORTS / f"scope_329_perturbed1pct_lhcb_only_y_wy_s{s}" for s in range(303, 311)],
    },
}

TAIL1_CASES = {
    "all_y": {
        "label": "all W+Y (1% perturbed starts; tail prior $\\lambda=1$)",
        "dirs": [REPORTS / f"scope_329_perturbed1pct_all_y_tail1_wy_s{s}" for s in range(303, 311)],
    },
    "non_lhcb_y": {
        "label": "W+Y with LHCb Y set to zero (1% perturbed starts; tail prior $\\lambda=1$)",
        "dirs": [REPORTS / f"scope_329_perturbed1pct_non_lhcb_y_tail1_wy_s{s}" for s in range(303, 311)],
    },
    "lhcb_only_y": {
        "label": "W+Y with LHCb Y only (1% perturbed starts; tail prior $\\lambda=1$)",
        "dirs": [REPORTS / f"scope_329_perturbed1pct_lhcb_only_y_tail1_wy_s{s}" for s in range(303, 311)],
    },
}


def make_source(name: str, dirs: list[Path], target: Path) -> tuple[Path, dict]:
    pieces = []
    starts = []
    for index, directory in enumerate(dirs):
        path = directory / "fnp_debug_grid.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        frame = frame[np.isclose(frame["x"].to_numpy(float), 0.1)].copy()
        if frame.empty:
            raise RuntimeError(f"{path}: no x=0.1 F_NP grid")
        frame = frame[["bT", "F_NP"]].sort_values("bT")
        frame["member"] = index
        pieces.append(frame)
        starts.append({"member": index, "source": str(directory)})
    source = target / "fnp_start_replica_crossed_long_x0p1.csv"
    pd.concat(pieces, ignore_index=True).to_csv(source, index=False)
    return source, {"case": name, "starts": starts, "source": str(source)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-root", type=Path,
                    default=REPORTS / "wy_control_fig2_fig6")
    ap.add_argument("--cases", nargs="+", choices=tuple(DEFAULT_CASES),
                    default=tuple(DEFAULT_CASES))
    ap.add_argument("--family", choices=("original", "perturbed1pct", "tail1"),
                    default="original",
                    help="Which isolated start ensemble to render.")
    ap.add_argument("--start-count", type=int, default=None,
                    help="Number of starts; inferred from the selected family when omitted.")
    args = ap.parse_args()
    args.target_root.mkdir(parents=True, exist_ok=True)
    families = {"original": DEFAULT_CASES, "perturbed1pct": PERTURBED_CASES,
                "tail1": TAIL1_CASES}
    selected_cases = families[args.family]
    manifest = {
        "status": "isolated_corrected_wy_control_figures_complete_not_production",
        "family": args.family, "cases": {}, "frozen_production_modified": False,
        "promotion_authorized": False,
    }
    for name in args.cases:
        if name not in selected_cases:
            raise ValueError(f"case {name!r} is not available in family {args.family!r}")
        spec = selected_cases[name]
        target = args.target_root / name
        target.mkdir(parents=True, exist_ok=True)
        source, record = make_source(name, spec["dirs"], target)
        cmd = [
            "/home/dustin/miniforge3/envs/pdf-fit/bin/python", str(PLOTTER),
            "--source", str(source), "--target", str(target),
            "--start-count", str(args.start_count or len(spec["dirs"])), "--replica-count", "1",
            "--title-label", spec["label"],
        ]
        subprocess.run(cmd, check=True)
        summary_path = target / "summary.json"
        summary = json.loads(summary_path.read_text())
        summary["case_label"] = spec["label"]
        summary["source_fits"] = record["starts"]
        summary["frozen_production_modified"] = False
        summary["promotion_authorized"] = False
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        manifest["cases"][name] = {**record, "target": str(target),
                                   "fig2": str(target / "fig2.png"),
                                   "fig6": str(target / "fig6.png")}
    (args.target_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
