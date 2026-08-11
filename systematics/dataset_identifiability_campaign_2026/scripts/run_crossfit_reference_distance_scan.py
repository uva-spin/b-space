#!/usr/bin/env python3
"""Two-fold cross-fitted scan of the empirical FNP-distance strength."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess


BASE = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/dustin/miniforge3/envs/pdf-fit/bin/python")
BUILD_REF = BASE / "scripts/build_exact_baseline_fnp_median.py"
RUN = BASE / "scripts/run_matched_baseline_reference_distance.py"
SEEDS = tuple(range(303, 327))
FOLDS = {
    "evenref": tuple(s for s in SEEDS if s % 2 == 0),
    "oddref": tuple(s for s in SEEDS if s % 2 == 1),
}
STRENGTHS = (0.5, 0.75, 1.0, 1.5, 2.0)
TARGET = BASE / "summaries/crossfit_reference_distance_scan"


def token(value: float) -> str:
    return f"{value:.2g}".replace(".", "p")


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "running",
        "design": "reciprocal two-fold cross-fit; no evaluated seed contributes to its reference",
        "strengths": list(STRENGTHS),
        "folds": {},
        "production_sources_modified": False,
    }
    for label, reference_seeds in FOLDS.items():
        evaluation_seeds = tuple(s for s in SEEDS if s not in reference_seeds)
        ref_name = f"crossfit_reference_{label}"
        ref_path = BASE / "summaries" / ref_name / "fnp_median.csv"
        command = [str(PYTHON), str(BUILD_REF), "--target-name", ref_name]
        for seed in reference_seeds:
            command.extend(["--seed", str(seed)])
        subprocess.run(command, check=True)
        manifest["folds"][label] = {
            "reference_seeds": list(reference_seeds),
            "evaluation_seeds": list(evaluation_seeds),
            "reference": str(ref_path),
        }
        (TARGET / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        for strength in STRENGTHS:
            for seed in evaluation_seeds:
                subprocess.run([
                    str(PYTHON), str(RUN), "--seed", str(seed),
                    "--strength", str(strength), "--reference", str(ref_path),
                    "--reference-label", label,
                ], check=True)
    manifest["status"] = "complete"
    (TARGET / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
